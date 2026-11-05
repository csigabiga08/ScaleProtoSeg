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
    "activation": {
        "prototype_distance_activation": "log",
        "prototype_activation_function": "log",
        "normalization_mean": [0.1, 0.2, 0.3],
        "normalization_std": [1.0, 1.0, 1.0],
        "resize_interpolation": "nearest",
        "quantile_support": "semantic-class pixels",
        "quantile_interpolation": "linear",
        "active_comparison": "strictly greater than quantile",
        "quantile_class_mask": "semantic class mask",
    },
    "centroids": {
        "component_rule": "connected components",
        "background_component_centroid_included": False,
        "exclude_background_centroid": True,
    },
    "parts": {
        "valid_part_ids": [1, 2],
        "mapping_identity": {"status": "available", "sha256": "c" * 64},
        "class_dependent_part_definitions": {
            "fixture": {"part_id_to_name": {"1": "part one", "2": "part two"}},
        },
    },
    "classes": {
        "id_convention": {"model_class_index": "class_id - 1"},
        "evaluation_class_id_to_label": {"1": "fixture", "2": "other"},
        "ignored_or_void_evaluation_class_id": 0,
        "background_evaluation_class_id": None,
        "excluded_evaluation_class_ids": [],
    },
    "prototype_selection": {
        "class_identity_rule": "recorded",
        "excluded_evaluation_class_ids": [],
        "grouping_active": False,
        "group_nonzero_rule": "nonzero",
        "group_intersection_rule": "none",
    },
    "model": {"epsilon": 1e-4, "prototype_activation_function": "log"},
    "input": {
        "source": "fixture image",
        "resolution": "native",
        "normalization": {
            "conversion": "RGB tensor in [0, 1]",
            "mean": [0.1, 0.2, 0.3],
            "std": [1.0, 1.0, 1.0],
        },
    },
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
        summary_output = outside / "summary"
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
                self.assertEqual(
                    catalog_cli.main(["summarize", "cli", "--output-dir", str(summary_output)]),
                    0,
                )
        self.assertTrue((summary_output / "summary.json").is_file())
        summary = json.loads((summary_output / "summary.json").read_text())
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

    def test_summary_export_rejects_existing_symlink_without_touching_target(self):
        destination = Path(self.temporary.name) / "summary-output"
        destination.mkdir()
        sentinel = Path(self.temporary.name) / "sentinel.json"
        sentinel.write_text('{"keep":"untouched"}\n', encoding="utf-8")
        (destination / "summary.json").symlink_to(sentinel)

        with self.assertRaises(CatalogError):
            catalog_cli.export_summary(
                legacy_record("symlink-summary"),
                {"score": {"value": 1.0}},
                destination,
                root=self.root,
            )
        self.assertEqual(sentinel.read_text(encoding="utf-8"), '{"keep":"untouched"}\n')

    def test_summary_export_refuses_reuse_to_prevent_stale_tables(self):
        destination = Path(self.temporary.name) / "summary-output"
        first = {
            "score": {"value": 1.0},
            "classes": [{"class_id": 1, "value": 1.0}],
            "parts": [{"class_id": 1, "part_id": 2, "mean_presence": 1.0}],
        }
        catalog_cli.export_summary(legacy_record("first-summary"), first, destination, root=self.root)
        parts_before = (destination / "parts.csv").read_bytes()

        with self.assertRaises(CatalogError):
            catalog_cli.export_summary(
                legacy_record("empty-summary"),
                {"score": {"value": None}, "classes": [], "parts": []},
                destination,
                root=self.root,
            )
        self.assertEqual((destination / "parts.csv").read_bytes(), parts_before)

    def test_plots_keep_attempts_separate_and_export_exact_tables(self):
        records = []
        for measurement_id, quantile, sample_ids in (
            ("q70", 0.7, ("s1", "s2")),
            ("q70-repeat", 0.7, ("s1", "s2")),
            ("q80", 0.8, ("s1", "s3")),
        ):
            record = legacy_record(measurement_id, quantile=quantile)
            record["model"]["inputs"]["prototype_checkpoint"]["sha256"] = "b" * 64
            record["dataset"]["identity"] = "same-manifest"
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
        sentinel = Path(self.temporary.name) / "plot-sentinel.csv"
        sentinel.write_text("keep this content\n", encoding="utf-8")
        heatmap_export = destination / "01_heatmap_values.csv"
        heatmap_export.unlink()
        heatmap_export.symlink_to(sentinel)
        with self.assertRaises(CatalogError):
            visualizations.render_measurements(
                [item[1] for item in records],
                summaries_by_id,
                destination,
                root=self.root,
            )
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep this content\n")
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

    def test_comparison_rejects_matching_but_unknown_protocol_fields(self):
        records = []
        summaries_by_id = []
        for measurement_id, quantile in (("unknown-q70", 0.7), ("unknown-q80", 0.8)):
            record = legacy_record(measurement_id, quantile=quantile)
            record["model"]["inputs"]["prototype_checkpoint"]["sha256"] = "b" * 64
            record["dataset"]["identity"] = "same-manifest"
            record["protocol"]["activation"]["quantile_interpolation"] = None
            record["provenance_quality"]["missing_fields"].append(
                "activation.quantile_interpolation"
            )
            rows = [{
                "sample_id": "sample",
                "class_id": "1",
                "prototype_id": "1",
                "repetition": "0",
                "part_1": "1",
                "part_2": "0",
            }]
            bundle = write_measurement(self.root, record, rows)
            records.append(record)
            summaries_by_id.append(
                summaries.summarize_observations(record, catalog_cli.read_observations(bundle, record))
            )
        with self.assertRaisesRegex(visualizations.VisualizationError, "quantile_interpolation"):
            visualizations.render_measurements(
                records,
                summaries_by_id,
                Path(self.temporary.name) / "unknown-protocol-plots",
                root=self.root,
            )

    def test_comparison_rejects_null_model_and_input_protocol_values(self):
        unknown_fields = (
            ("input.normalization.mean", None),
            ("model.epsilon", None),
        )
        for field_path, unknown_value in unknown_fields:
            with self.subTest(field=field_path):
                record = legacy_record("unknown-" + field_path.replace(".", "-"))
                record["model"]["inputs"]["prototype_checkpoint"]["sha256"] = "b" * 64
                record["dataset"]["identity"] = "same-manifest"
                section, field = field_path.split(".", 1)
                target = record["protocol"][section]
                for component in field.split(".")[:-1]:
                    target = target[component]
                target[field.split(".")[-1]] = unknown_value
                record["provenance_quality"]["missing_fields"].append(field_path)

                with self.assertRaisesRegex(
                    visualizations.VisualizationError,
                    field_path.replace(".", r"\."),
                ):
                    visualizations._compatibility_signature(record)

    def test_all_unavailable_part_combinations_are_retained_as_unverified_support(self):
        record = legacy_record("sparse-support")
        rows = [
            {
                "sample_id": "available",
                "class_id": "1",
                "prototype_id": "1",
                "repetition": "0",
                "part_1": "1",
                "part_2": "",
            },
            {
                "sample_id": "unavailable",
                "class_id": "1",
                "prototype_id": "2",
                "repetition": "0",
                "part_1": "",
                "part_2": "",
            },
        ]
        result = summaries.summarize_observations(record, rows)
        part_two = next(item for item in result["support"] if item["part_id"] == 2)
        self.assertEqual(part_two["usable_images"], 0)
        self.assertEqual(part_two["sample_ids"], [])
        self.assertFalse(part_two["prototype_support_matches"])
        rendered_tables, support_comparable = visualizations._tables([record], [result])
        self.assertFalse(support_comparable)
        self.assertIn(
            (1, 2),
            {(row["class_id"], row["part_id"]) for row in rendered_tables["support"]},
        )


if __name__ == "__main__":
    unittest.main()
