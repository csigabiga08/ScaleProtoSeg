"""CPU-only tests for catalog queries, summaries, and derived plots."""

import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

from segmentation.analysis.metrics import catalog_cli, summaries, visualizations
from segmentation.analysis.metrics.catalog import CatalogError, load_measurement


CONSISTENCY_PROTOCOL = {
    "version": 1,
    "metric": "consistency",
    "quantile": 0.8,
    "activation": {"quantile_interpolation": "linear"},
    "centroids": {"component_rule": "connected components"},
    "parts": {
        "valid_part_ids": [1, 2],
        "mapping_identity": {"status": "available", "sha256": "c" * 64},
        "class_dependent_part_definitions": {
            "fixture": {"part_id_to_name": {"1": "part one", "2": "part two"}},
        },
    },
    "classes": {"evaluation_class_id_to_label": {"1": "fixture", "2": "other"}},
    "prototype_selection": {"class_identity_rule": "recorded"},
    "consistency": {
        "threshold": 0.8,
        "strict_comparison": "mean part presence > threshold",
        "aggregation": "unweighted mean over class/prototype rows",
    },
    "observation_columns": ["part_1", "part_2"],
}

STABILITY_PROTOCOL = {
    "version": 1,
    "metric": "stability",
    "quantile": 0.8,
    "parts": {"valid_part_ids": [1, 2]},
    "stability": {
        "aggregation": "unweighted mean of exact all-part equality per paired sample/class/prototype row",
        "missing_value_comparison": "zero versus zero for parts without annotated centroids",
    },
    "observation_columns": ["part_1", "part_2", "noisy_part_1", "noisy_part_2"],
}


def legacy_record(measurement_id, *, metric="consistency", quantile=0.8, status="complete"):
    protocol = json.loads(json.dumps(CONSISTENCY_PROTOCOL if metric == "consistency" else STABILITY_PROTOCOL))
    protocol["quantile"] = quantile
    return {
        "schema_version": 1,
        "measurement_id": measurement_id,
        "origin": "legacy",
        "metric": metric,
        "status": status,
        "scores": {},
        "model": {
            "label": "fixture",
            "inputs": {
                "prototype_checkpoint": {
                    "path": "checkpoint.pth",
                    "bytes": 4,
                    "sha256": "a" * 64,
                    "status": "available",
                }
            },
        },
        "dataset": {"name": "fixture-set", "split": "val", "identity": "fixture-dataset-v1"},
        "protocol": protocol,
        "parameters": {},
        "artifacts": {},
        "provenance_quality": {
            "missing_fields": ["candidate membership", "sample order"],
            "warnings": ["Historical sample table was not retained."],
        },
        "extension": {"preserved": {"note": "unknown extension"}},
    }


def write_measurement(root, record, observations=None):
    bundle = root / "measurements" / record["measurement_id"]
    bundle.mkdir(parents=True)
    if observations is not None:
        observation_path = bundle / "observations.csv"
        with observation_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(observations[0]))
            writer.writeheader()
            writer.writerows(observations)
        record["artifacts"]["observations"] = {"path": "observations.csv"}
        record["observation_schema"] = {"columns": list(observations[0])}
    (bundle / "result.json").write_text(json.dumps(record), encoding="utf-8")
    return bundle


class MetricSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "catalog"
        (self.root / "measurements").mkdir(parents=True)

    def test_dotenv_root_is_used_without_overwriting_exported_root(self):
        checkout = Path(self.temporary.name) / "checkout"
        checkout.mkdir()
        configured = Path(self.temporary.name) / "configured"
        exported = Path(self.temporary.name) / "exported"
        configured.mkdir()
        exported.mkdir()
        (checkout / ".env").write_text(f"MEASUREMENT_CATALOG_ROOT={configured}\n")

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(catalog_cli.configured_catalog_root(checkout), configured.resolve())
            self.assertEqual(os.environ["MEASUREMENT_CATALOG_ROOT"], str(configured))
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(exported)}):
            self.assertEqual(catalog_cli.configured_catalog_root(checkout), exported.resolve())
            self.assertEqual(os.environ["MEASUREMENT_CATALOG_ROOT"], str(exported))

    def test_list_defaults_to_complete_and_all_status_keeps_unknown_legacy_visible(self):
        complete = legacy_record("complete")
        complete["provenance_quality"]["missing_fields"].append("model architecture")
        write_measurement(self.root, complete)
        write_measurement(self.root, legacy_record("running", status="running"))
        write_measurement(self.root, legacy_record("failed", status="failed"))
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            completed = catalog_cli.find_records()
            self.assertEqual([row["measurement_id"] for row in completed], ["complete"])
            all_records = catalog_cli.find_records(all_status=True)
            filtered = catalog_cli.find_records(
                all_status=True, metric="consistency", dataset="fixture-set", provenance_gap="model architecture"
            )
        with self.assertRaisesRegex(CatalogError, "requires --all-status"):
            catalog_cli.find_records(self.root, statuses=("failed",))
        failed = catalog_cli.find_records(self.root, all_status=True, statuses=("failed",))
        self.assertEqual(
            {row["measurement_id"] for row in all_records},
            {"complete", "running", "failed"},
        )
        self.assertIn("failed", {row["measurement_id"] for row in all_records})
        self.assertEqual([row["measurement_id"] for row in failed], ["failed"])
        self.assertEqual(all_records[0]["extension"]["preserved"]["note"], "unknown extension")
        self.assertEqual([row["measurement_id"] for row in filtered], ["complete"])

    def test_listing_reports_unsupported_major_schema_without_reinterpreting_it(self):
        record = legacy_record("future-schema")
        record["schema_version"] = 2
        write_measurement(self.root, record)
        diagnostic = io.StringIO()
        with (
            patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}),
            contextlib.redirect_stderr(diagnostic),
        ):
            self.assertEqual(catalog_cli.find_records(), [])
        self.assertIn("Unsupported measurement schema version: 2", diagnostic.getvalue())

    def test_metadata_listing_does_not_open_raw_tables_or_checkpoint_payloads(self):
        record = legacy_record("metadata-only")
        record["artifacts"]["observations"] = {"path": "missing-large-array.csv"}
        record["model"]["inputs"]["prototype_checkpoint"]["path"] = "missing-checkpoint.pth"
        write_measurement(self.root, record)
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            self.assertEqual(catalog_cli.find_records()[0]["measurement_id"], "metadata-only")

    def test_filter_and_export_preserve_unknown_extension_fields(self):
        record = legacy_record("export-me")
        write_measurement(self.root, record)
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            selected = catalog_cli.find_records(metric="consistency", quantile="0.8")
            output = self.root / "exports" / "records.csv"
            catalog_cli.export_records(selected, output)
        with output.open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["extension.preserved.note"], "unknown extension")
        self.assertEqual(json.loads(row["metadata_json"])["extension"], record["extension"])

    def test_cli_list_filter_export_validate_and_summarize_commands(self):
        record = legacy_record("cli")
        rows = [
            {"sample_id": "a", "class_id": "1", "prototype_id": "1", "repetition": "0", "part_1": "1", "part_2": "0"},
            {"sample_id": "b", "class_id": "1", "prototype_id": "1", "repetition": "0", "part_1": "1", "part_2": "0"},
        ]
        bundle = write_measurement(self.root, record, rows)
        outside = Path(self.temporary.name) / "summary-export"
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(catalog_cli.main(["list"]), 0)
            self.assertEqual(json.loads(output.getvalue())[0]["measurement_id"], "cli")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(catalog_cli.main(["filter", "--metric", "stability"]), 0)
            self.assertEqual(json.loads(output.getvalue()), [])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(catalog_cli.main(["export", "--output", str(outside / "records.csv")]), 0)
            self.assertTrue((outside / "records.csv").is_file())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(catalog_cli.main(["validate", "cli"]), 1)
            self.assertEqual(json.loads(output.getvalue())[0]["origin"], "legacy")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(catalog_cli.main(["summarize", "cli", "--output-dir", str(outside)]), 0)
        self.assertTrue((outside / "summary.json").is_file())
        summary = json.loads((outside / "summary.json").read_text())
        self.assertEqual(summary["score"]["value"], 1.0)

    def test_summarize_all_status_allows_finalized_failures_but_never_running(self):
        failed = legacy_record("failed-summary", status="failed")
        rows = [
            {"sample_id": "a", "class_id": "1", "prototype_id": "1", "repetition": "0", "part_1": "1", "part_2": "0"},
        ]
        write_measurement(self.root, failed, rows)
        write_measurement(self.root, legacy_record("running-summary", status="running"))
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as failure:
                        catalog_cli.main(["summarize", "failed-summary"])
            self.assertEqual(failure.exception.code, 2)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    catalog_cli.main(["summarize", "failed-summary", "--all-status"]),
                    0,
                )
            self.assertEqual(json.loads(output.getvalue())["score"]["value"], 1.0)
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                with self.assertRaises(SystemExit) as failure:
                    catalog_cli.main(["summarize", "running-summary", "--all-status"])
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("Running measurements cannot be summarized", error.getvalue())

    def test_legacy_partial_validation_surfaces_unknown_sample_membership(self):
        record = legacy_record("partial")
        bundle = write_measurement(self.root, record)
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            diagnostics = catalog_cli.validate_measurement(bundle)
        self.assertEqual(diagnostics["status"], "needs-attention")
        self.assertIn("candidate membership/order is unknown", diagnostics["needs"])

    def test_catalog_reader_accepts_truthful_legacy_metadata_with_partial_samples(self):
        record = legacy_record("partial-table")
        record["scores"] = {
            "consistency": {
                "value": 0.5,
                "unit": "fraction",
                "numerator": 1,
                "denominator": 2,
            }
        }
        rows = [
            {
                "sample_id": "known",
                "class_id": "1",
                "prototype_id": "1",
                "repetition": "0",
                "part_1": "1",
                "part_2": "NaN",
            },
        ]
        bundle = write_measurement(self.root, record, rows)
        observation_path = bundle / "observations.csv"
        observation_bytes = observation_path.read_bytes()
        samples_path = bundle / "samples.csv"
        with samples_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("sample_index", "sample_id", "outcome", "reason", "observation_count"),
            )
            writer.writeheader()
            writer.writerow({
                "sample_index": "3",
                "sample_id": "known",
                "outcome": "contributed",
                "reason": "",
                "observation_count": "1",
            })
        record["artifacts"]["observations"].update({
            "bytes": len(observation_bytes),
            "sha256": hashlib.sha256(observation_bytes).hexdigest(),
            "schema_version": 1,
            "columns": list(rows[0]),
        })
        sample_bytes = samples_path.read_bytes()
        record["artifacts"]["samples"] = {
            "path": "samples.csv",
            "bytes": len(sample_bytes),
            "sha256": hashlib.sha256(sample_bytes).hexdigest(),
            "schema_version": 1,
            "columns": ["sample_index", "sample_id", "outcome", "reason", "observation_count"],
        }
        record["observation_schema"] = {"version": 1, "columns": list(rows[0])}
        record["provenance_quality"]["missing_fields"] = [
            "candidate membership/order",
            "historical sample population",
        ]
        (bundle / "result.json").write_text(json.dumps(record), encoding="utf-8")

        loaded = load_measurement(bundle)
        self.assertEqual(loaded["origin"], "legacy")
        self.assertEqual(loaded["artifacts"]["samples"]["columns"], [
            "sample_index", "sample_id", "outcome", "reason", "observation_count",
        ])
        self.assertNotIn(
            "population",
            loaded,
            "partial legacy sample tables are not promoted to complete native populations",
        )
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            diagnostics = catalog_cli.validate_measurement(bundle)
        self.assertIn("candidate membership/order is unknown", diagnostics["needs"])

    def test_consistency_uses_strict_threshold_available_values_and_row_weighting(self):
        record = legacy_record("weighted")
        record["origin"] = "native"
        record["protocol"] = json.loads(json.dumps(CONSISTENCY_PROTOCOL))
        record["protocol"]["consistency"]["threshold"] = 0.8
        rows = [
            {"sample_id": "a", "class_id": "1", "prototype_id": "1", "repetition": "0", "part_1": "1", "part_2": ""},
            {"sample_id": "b", "class_id": "1", "prototype_id": "1", "repetition": "0", "part_1": "0", "part_2": "1"},
            {"sample_id": "c", "class_id": "2", "prototype_id": "2", "repetition": "0", "part_1": "1", "part_2": ""},
            {"sample_id": "d", "class_id": "2", "prototype_id": "2", "repetition": "0", "part_1": "1", "part_2": ""},
            {"sample_id": "e", "class_id": "2", "prototype_id": "2", "repetition": "0", "part_1": "1", "part_2": ""},
            {"sample_id": "f", "class_id": "2", "prototype_id": "2", "repetition": "0", "part_1": "1", "part_2": ""},
            {"sample_id": "g", "class_id": "2", "prototype_id": "2", "repetition": "0", "part_1": "0", "part_2": ""},
            {"sample_id": "h", "class_id": "2", "prototype_id": "3", "repetition": "0", "part_1": "", "part_2": ""},
            {"sample_id": "i", "class_id": "2", "prototype_id": "3", "repetition": "0", "part_1": "0", "part_2": ""},
            {"sample_id": "j", "class_id": "3", "prototype_id": "4", "repetition": "0", "part_1": "", "part_2": ""},
        ]
        result = summaries.summarize_observations(record, rows)
        self.assertEqual(
            result["score"],
            {"value": 1 / 4, "numerator": 1, "denominator": 4, "unit": "fraction"},
        )
        prototype = next(item for item in result["prototypes"] if item["prototype_id"] == 2)
        self.assertFalse(prototype["consistent"], "exactly 0.8 must fail strict comparison")
        part = next(item for item in result["parts"] if item["class_id"] == 2 and item["part_id"] == 1)
        self.assertEqual(part["mean_presence"], 0.8)
        missing = next(item for item in result["parts"] if item["class_id"] == 2 and item["part_id"] == 2)
        self.assertIsNone(missing["mean_presence"], "unavailable part values must not become zeros")
        unavailable = next(item for item in result["prototypes"] if item["prototype_id"] == 4)
        self.assertIsNone(unavailable["max_part_frequency"])
        self.assertFalse(unavailable["consistent"])
        classes = {item["class_id"]: item for item in result["classes"]}
        self.assertEqual(classes[1]["value"], 1.0)
        self.assertEqual(classes[2]["value"], 0.0)
        self.assertEqual(classes[3]["value"], 0.0)

    def test_stability_compares_paired_vectors_without_coercing_missing_to_zero(self):
        record = legacy_record("paired", metric="stability")
        rows = [
            {
                "sample_id": "same", "class_id": "1", "prototype_id": "4", "repetition": "0",
                "part_1": "0", "part_2": "1", "noisy_part_1": "0", "noisy_part_2": "1",
            },
            {
                "sample_id": "missing", "class_id": "1", "prototype_id": "4", "repetition": "0",
                "part_1": "", "part_2": "1", "noisy_part_1": "", "noisy_part_2": "1",
            },
            {
                "sample_id": "changed", "class_id": "2", "prototype_id": "8", "repetition": "0",
                "part_1": "0", "part_2": "1", "noisy_part_1": "1", "noisy_part_2": "1",
            },
        ]
        result = summaries.summarize_observations(record, rows)
        self.assertEqual(result["score"], {"value": 1 / 3, "numerator": 1, "denominator": 3, "unit": "fraction"})
        self.assertFalse(result["stability_rows"][1]["stable"])
        self.assertEqual(result["stability_rows"][1]["part_equalities"], [None, True])

    def test_unknown_aggregation_protocol_is_rejected(self):
        record = legacy_record("unknown")
        record["protocol"]["consistency"]["aggregation"] = "unknown future pooling"
        with self.assertRaisesRegex(summaries.SummaryError, "Unsupported consistency aggregation"):
            summaries.summarize_observations(record, [])

    def test_unknown_historical_protocol_version_is_rejected(self):
        record = legacy_record("future-protocol")
        record["protocol"]["version"] = "scaleprotoseg-v2"
        with self.assertRaisesRegex(summaries.SummaryError, "Unsupported historical protocol version"):
            summaries.summarize_observations(record, [])

    def test_historical_named_protocol_version_is_explicitly_supported(self):
        record = legacy_record("historical")
        record["protocol"]["version"] = "scaleprotoseg-v1"
        record["protocol"]["consistency"]["aggregation"] = (
            "historical ScaleProtoSeg: unweighted mean over class/prototype rows"
        )
        result = summaries.summarize_observations(record, [])
        self.assertEqual(result["aggregation"], record["protocol"]["consistency"]["aggregation"])

    def test_summary_exports_are_outside_canonical_measurement_bundles(self):
        record = legacy_record("export-guard")
        bundle = write_measurement(self.root, record)
        with self.assertRaisesRegex(CatalogError, "outside catalog measurements"):
            catalog_cli.ensure_external_output(bundle / "result.json", self.root)

    def test_plots_keep_attempts_separate_and_export_exact_tables(self):
        records = []
        for measurement_id, quantile, sample_ids in (
            ("q70", 0.7, ("s1", "s2")),
            ("q70-repeat", 0.7, ("s1", "s2")),
            ("q80", 0.8, ("s1", "s3")),
        ):
            record = legacy_record(measurement_id, quantile=quantile)
            record["model"]["inputs"]["prototype_checkpoint"]["sha256"] = "b" * 64
            rows = [
                {
                    "sample_id": sample_id, "class_id": "1", "prototype_id": "2", "repetition": "0",
                    "part_1": "1" if index == 0 else "0", "part_2": "0",
                }
                for index, sample_id in enumerate(sample_ids)
            ]
            bundle = write_measurement(self.root, record, rows)
            records.append((bundle, record))
        summaries_by_id = []
        for bundle, record in records:
            rows = catalog_cli.read_observations(bundle, record)
            summaries_by_id.append(summaries.summarize_observations(record, rows))
        destination = Path(self.temporary.name) / "plots"
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                rendered = visualizations.render_measurements(
                    [item[1] for item in records],
                    summaries_by_id,
                    destination,
                )
        self.assertEqual(rendered["support_comparable"], False)
        self.assertEqual(len(rendered["tables"]["support"]), 6)
        self.assertEqual(
            {row["part_name"] for row in rendered["tables"]["support"]},
            {"part one", "part two"},
        )
        for name in (
            "01_prototype_part_heatmaps.png",
            "01_prototype_part_heatmaps.pdf",
            "02_class_consistency.csv",
            "03_prototype_scores.csv",
            "04_usable_images.csv",
            "all_four_visualizations.pdf",
        ):
            self.assertTrue((destination / name).is_file(), name)
        with (destination / "04_usable_images.csv").open(newline="", encoding="utf-8") as stream:
            support = list(csv.DictReader(stream))
        self.assertEqual([row["usable_images"] for row in support], ["2"] * 6)
        metadata = json.loads((destination / "visualization_metadata.json").read_text())
        self.assertEqual(metadata["measurement_ids"], ["q70", "q70-repeat", "q80"])
        self.assertFalse(metadata["support_comparable"])
        unidentified = json.loads(json.dumps(records[0][1]))
        unidentified["measurement_id"] = "unidentified"
        unidentified["dataset"]["identity"] = "partial"
        with self.assertRaisesRegex(visualizations.VisualizationError, "unknown dataset identity"):
            visualizations.render_measurements(
                [records[0][1], unidentified],
                [summaries_by_id[0], summaries_by_id[0]],
                Path(self.temporary.name) / "incomparable",
                root=self.root,
            )
        cli_output = Path(self.temporary.name) / "cli-plots"
        with (
            patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}),
            contextlib.redirect_stdout(io.StringIO()),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertEqual(
                catalog_cli.main([
                    "plot", "q70", "q70-repeat", "q80", "--output-dir", str(cli_output),
                ]),
                0,
            )
        self.assertTrue((cli_output / "all_four_visualizations.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
