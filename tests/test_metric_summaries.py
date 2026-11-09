"""CPU-only tests for catalog queries, summaries, and derived plots."""

import contextlib
import copy
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
from segmentation.analysis.metrics.catalog import CatalogError, load_observations, write_derived_csv


CONSISTENCY_PROTOCOL = {
    "metric": "consistency",
    "quantile": 0.8,
    "parts": {
        "valid_part_ids": [1, 2],
        "class_dependent_part_definitions": {
            "fixture": {"part_id_to_name": {"1": "part one", "2": "part two"}},
        },
    },
    "classes": {"evaluation_class_id_to_label": {"1": "fixture", "2": "other"}},
    "consistency": {
        "threshold": 0.8,
        "strict_comparison": "mean part presence > threshold",
        "aggregation": "unweighted mean over class/prototype rows",
    },
    "observation_columns": ["part_1", "part_2"],
}

STABILITY_PROTOCOL = {
    "metric": "stability",
    "quantile": 0.8,
    "parts": {"valid_part_ids": [1, 2]},
    "stability": {
        "aggregation": "unweighted mean of exact all-part equality per paired sample/class/prototype row",
        "missing_value_comparison": "zero versus zero for parts without annotated centroids",
    },
    "observation_columns": ["part_1", "part_2", "noisy_part_1", "noisy_part_2"],
}


def obs(sample_id, class_id=1, prototype_id=1, part_1="1", part_2="0", **extra):
    return {
        "sample_id": sample_id, "class_id": str(class_id), "prototype_id": str(prototype_id),
        "repetition": "0", "part_1": part_1, "part_2": part_2, **extra,
    }


def make_record(measurement_id, *, metric="consistency", quantile=0.8, status="complete", checkpoint="b" * 64):
    protocol = copy.deepcopy(CONSISTENCY_PROTOCOL if metric == "consistency" else STABILITY_PROTOCOL)
    protocol["quantile"] = quantile
    return {
        "measurement_id": measurement_id,
        "metric": metric,
        "status": status,
        "scores": {},
        "model": {"label": "fixture", "inputs": {"prototype_checkpoint": {"sha256": checkpoint}}},
        "dataset": {"name": "fixture-set", "split": "val"},
        "protocol": protocol,
        "parameters": {"quantile": quantile},
        "artifacts": {},
    }


def write_measurement(root, record, observations=None):
    """Write a bundle; with observations the artifact entry, counts, and score are filled in."""
    bundle = root / "measurements" / record["measurement_id"]
    bundle.mkdir(parents=True)
    if observations is not None:
        path = bundle / "observations.csv"
        columns = ["sample_id", "class_id", "prototype_id", "repetition"] + record["protocol"]["observation_columns"]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(observations)
        record["artifacts"]["observations"] = {
            "path": "observations.csv",
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        samples = {row["sample_id"] for row in observations}
        record["population"] = {"candidates": len(samples), "contributing": len(samples), "observations": len(observations)}
        if record["status"] == "complete":
            record["scores"][record["metric"]] = {
                "value": 1.0, "unit": "fraction", "numerator": 1, "denominator": 1,
            }
    (bundle / "result.json").write_text(json.dumps(record), encoding="utf-8")
    return bundle


class MetricSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "catalog"
        (self.root / "measurements").mkdir(parents=True)
        patcher = patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cli(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = catalog_cli.main(list(args))
        return code, output.getvalue()

    def test_dotenv_root_is_used_without_overwriting_exported_root(self):
        checkout = Path(self.temporary.name) / "checkout"
        checkout.mkdir()
        configured = Path(self.temporary.name) / "configured"
        configured.mkdir()
        (checkout / ".env").write_text(f"MEASUREMENT_CATALOG_ROOT={configured}\n")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(catalog_cli.configured_catalog_root(checkout), configured.resolve())
        self.assertEqual(catalog_cli.configured_catalog_root(checkout), self.root.resolve())

    def test_list_defaults_to_complete_and_filters_by_parameters(self):
        write_measurement(self.root, make_record("complete"), [obs("a")])
        write_measurement(self.root, make_record("running", status="running"))
        write_measurement(self.root, make_record("failed", status="failed"))
        write_measurement(self.root, make_record("other-q", quantile=0.9), [obs("a")])
        self.assertEqual({r["measurement_id"] for r in catalog_cli.find_records()}, {"complete", "other-q"})
        everything = catalog_cli.find_records(all_status=True)
        self.assertEqual(len(everything), 4)
        with self.assertRaisesRegex(CatalogError, "requires --all-status"):
            catalog_cli.find_records(statuses=("failed",))
        failed = catalog_cli.find_records(all_status=True, statuses=("failed",))
        self.assertEqual([r["measurement_id"] for r in failed], ["failed"])
        selected = catalog_cli.find_records(metric="consistency", dataset="fixture-set", quantile="0.9")
        self.assertEqual([r["measurement_id"] for r in selected], ["other-q"])

    def test_invalid_metadata_is_skipped_with_a_warning(self):
        record = make_record("broken")
        del record["protocol"]
        write_measurement(self.root, record)
        diagnostic = io.StringIO()
        with contextlib.redirect_stderr(diagnostic):
            self.assertEqual(catalog_cli.find_records(), [])
        self.assertIn("skipping broken", diagnostic.getvalue())

    def test_listing_does_not_open_observation_tables(self):
        record = make_record("metadata-only")
        bundle = write_measurement(self.root, record, [obs("a")])
        (bundle / "observations.csv").unlink()
        self.assertEqual(catalog_cli.find_records()[0]["measurement_id"], "metadata-only")

    def test_cli_list_filter_export_validate_and_summarize_commands(self):
        write_measurement(self.root, make_record("cli"), [obs("a"), obs("b")])
        outside = Path(self.temporary.name) / "exports"
        code, text = self.run_cli("list")
        self.assertEqual((code, json.loads(text)[0]["measurement_id"]), (0, "cli"))
        self.assertEqual(json.loads(self.run_cli("filter", "--metric", "stability")[1]), [])
        self.assertEqual(self.run_cli("export", "--output", str(outside / "records.csv"))[0], 0)
        self.assertTrue((outside / "records.csv").is_file())
        code, text = self.run_cli("validate", "cli")
        self.assertEqual((code, json.loads(text)[0]["status"]), (0, "valid"))
        code, _ = self.run_cli("summarize", "cli", "--output-dir", str(outside / "summary"))
        self.assertEqual(code, 0)
        summary = json.loads((outside / "summary" / "summary.json").read_text())
        self.assertEqual(summary["score"]["value"], 1.0)

    def test_validate_reports_invalid_bundles(self):
        bundle = write_measurement(self.root, make_record("tampered"), [obs("a")])
        (bundle / "observations.csv").write_text("corrupt\n")
        code, text = self.run_cli("validate", "tampered")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(text)[0]["status"], "invalid")

    def test_summarize_parses_each_artifact_once(self):
        write_measurement(self.root, make_record("single-parse"), [obs("a")])
        parsed = []
        original_reader = csv.DictReader

        def count_reader(stream, *args, **kwargs):
            parsed.append(Path(stream.name).name)
            return original_reader(stream, *args, **kwargs)

        with patch("segmentation.analysis.metrics.catalog.csv.DictReader", side_effect=count_reader):
            self.assertEqual(self.run_cli("summarize", "single-parse")[0], 0)
        self.assertEqual(parsed, ["observations.csv"])

    def test_load_observations_distinguishes_empty_table_from_absent_evidence(self):
        bundle = write_measurement(self.root, make_record("empty", status="failed"), [obs("a")])
        (bundle / "observations.csv").write_text("sample_id,class_id,prototype_id,repetition,part_1,part_2\r\n")
        result = json.loads((bundle / "result.json").read_text())
        result["artifacts"]["observations"].update(
            bytes=(bundle / "observations.csv").stat().st_size,
            sha256=hashlib.sha256((bundle / "observations.csv").read_bytes()).hexdigest(),
        )
        result["population"] = {"candidates": 0, "contributing": 0, "observations": 0}
        (bundle / "result.json").write_text(json.dumps(result))
        self.assertEqual(load_observations(bundle)[1], [])

        absent = write_measurement(self.root, make_record("absent", status="failed"))
        with self.assertRaisesRegex(CatalogError, "finalized observation evidence"):
            load_observations(absent)

    def test_summarize_all_status_allows_finalized_failures_but_never_running(self):
        write_measurement(self.root, make_record("failed-summary", status="failed"), [obs("a")])
        write_measurement(self.root, make_record("running-summary", status="running"))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as failure:
                self.run_cli("summarize", "failed-summary")
        self.assertEqual(failure.exception.code, 2)
        code, text = self.run_cli("summarize", "failed-summary", "--all-status")
        self.assertEqual((code, json.loads(text)["score"]["value"]), (0, 1.0))
        error = io.StringIO()
        with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
            self.run_cli("summarize", "running-summary", "--all-status")
        self.assertIn("Running measurements cannot be summarized", error.getvalue())

    def test_consistency_uses_strict_threshold_available_values_and_row_weighting(self):
        rows = [
            obs("a", 1, 1, "1", ""), obs("b", 1, 1, "0", "1"),
            obs("c", 2, 2, "1", ""), obs("d", 2, 2, "1", ""), obs("e", 2, 2, "1", ""),
            obs("f", 2, 2, "1", ""), obs("g", 2, 2, "0", ""),
            obs("h", 2, 3, "", ""), obs("i", 2, 3, "0", ""),
            obs("j", 3, 4, "", ""),
        ]
        result = summaries.summarize_observations(make_record("weighted"), rows)
        self.assertEqual(result["score"], {"value": 1 / 4, "numerator": 1, "denominator": 4, "unit": "fraction"})
        prototype = next(item for item in result["prototypes"] if item["prototype_id"] == 2)
        self.assertFalse(prototype["consistent"], "exactly 0.8 must fail strict comparison")
        part = next(item for item in result["parts"] if item["class_id"] == 2 and item["part_id"] == 1)
        self.assertEqual(part["mean_presence"], 0.8)
        missing = next(item for item in result["parts"] if item["class_id"] == 2 and item["part_id"] == 2)
        self.assertIsNone(missing["mean_presence"], "unavailable part values must not become zeros")
        unavailable = next(item for item in result["prototypes"] if item["prototype_id"] == 4)
        self.assertIsNone(unavailable["max_part_frequency"])
        classes = {item["class_id"]: item for item in result["classes"]}
        self.assertEqual([classes[i]["value"] for i in (1, 2, 3)], [1.0, 0.0, 0.0])

    def test_stability_compares_paired_vectors_without_coercing_missing_to_zero(self):
        rows = [
            obs("same", 1, 4, "0", "1", noisy_part_1="0", noisy_part_2="1"),
            obs("missing", 1, 4, "", "1", noisy_part_1="", noisy_part_2="1"),
            obs("changed", 2, 8, "0", "1", noisy_part_1="1", noisy_part_2="1"),
        ]
        result = summaries.summarize_observations(make_record("paired", metric="stability"), rows)
        self.assertEqual(result["score"], {"value": 1 / 3, "numerator": 1, "denominator": 3, "unit": "fraction"})
        self.assertFalse(result["stability_rows"][1]["stable"])
        self.assertEqual(result["stability_rows"][1]["part_equalities"], [None, True])

    def test_unknown_aggregation_protocol_is_rejected(self):
        record = make_record("unknown")
        record["protocol"]["consistency"]["aggregation"] = "unknown future pooling"
        with self.assertRaisesRegex(summaries.SummaryError, "Unsupported consistency aggregation"):
            summaries.summarize_observations(record, [])

    def test_summary_exports_are_outside_canonical_measurement_bundles(self):
        bundle = write_measurement(self.root, make_record("export-guard"))
        with self.assertRaisesRegex(CatalogError, "outside catalog measurements"):
            catalog_cli.ensure_external_output(bundle / "result.json", self.root)

    def test_derived_csv_writer_preserves_headers_unicode_nulls_and_nested_values(self):
        destination = Path(self.temporary.name) / "derived.csv"
        write_derived_csv(destination, [{"label": "café", "empty": None, "metadata": {"雪": ["é", 2]}}],
                          ["label", "empty", "metadata"])
        with destination.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            row = next(reader)
        self.assertEqual(reader.fieldnames, ["label", "empty", "metadata"])
        self.assertEqual(row, {"label": "café", "empty": "", "metadata": '{"雪": ["é", 2]}'})

    def test_summary_export_omits_empty_tables_and_refuses_reuse_or_symlinks(self):
        destination = Path(self.temporary.name) / "summary-tables"
        summary = {"score": {"value": 1.0}, "classes": [{"label": "café", "value": 1.0}], "parts": []}
        catalog_cli.export_summary(make_record("s"), summary, destination, root=self.root)
        self.assertEqual({p.name for p in destination.iterdir()}, {"summary.json", "classes.csv"})
        before = (destination / "classes.csv").read_bytes()
        with self.assertRaises(CatalogError):
            catalog_cli.export_summary(make_record("s"), {"score": {"value": None}}, destination, root=self.root)
        self.assertEqual((destination / "classes.csv").read_bytes(), before)

        linked = Path(self.temporary.name) / "linked"
        linked.mkdir()
        sentinel = Path(self.temporary.name) / "sentinel.json"
        sentinel.write_text('{"keep":"untouched"}\n', encoding="utf-8")
        (linked / "summary.json").symlink_to(sentinel)
        with self.assertRaises(CatalogError):
            catalog_cli.export_summary(make_record("s"), {"score": {"value": 1.0}}, linked, root=self.root)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), '{"keep":"untouched"}\n')

    def test_visualization_exports_empty_tables_with_explicit_headers(self):
        destination = Path(self.temporary.name) / "empty-plots"
        summary = {
            "quantile": 0.8, "score": {"value": None}, "threshold": 0.8,
            "classes": [], "prototypes": [], "parts": [], "support": [],
        }
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            visualizations.render_measurements([make_record("empty-viz")], [summary], destination, root=self.root)
        with (destination / "02_class_consistency.csv").open(newline="", encoding="utf-8") as stream:
            self.assertEqual(next(csv.reader(stream)), [
                "measurement_id", "quantile", "class_id", "class_name", "numerator",
                "denominator", "consistency_fraction",
            ])

    def test_plots_keep_attempts_separate_and_export_exact_tables(self):
        records, summaries_by_id = [], []
        for measurement_id, quantile, sample_ids in (
            ("q70", 0.7, ("s1", "s2")), ("q70-repeat", 0.7, ("s1", "s2")), ("q80", 0.8, ("s1", "s3")),
        ):
            rows = [obs(s, prototype_id=2, part_1="1" if i == 0 else "0") for i, s in enumerate(sample_ids)]
            bundle = write_measurement(self.root, make_record(measurement_id, quantile=quantile), rows)
            record, loaded = load_observations(bundle)
            records.append(record)
            summaries_by_id.append(summaries.summarize_observations(record, loaded))
        destination = Path(self.temporary.name) / "plots"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rendered = visualizations.render_measurements(records, summaries_by_id, destination, root=self.root)
        self.assertFalse(rendered["support_comparable"])
        self.assertEqual(len(rendered["tables"]["support"]), 6)
        for name in ("01_prototype_part_heatmaps.png", "02_class_consistency.csv",
                     "03_prototype_scores.csv", "04_usable_images.csv", "all_four_visualizations.pdf"):
            self.assertTrue((destination / name).is_file(), name)
        with (destination / "04_usable_images.csv").open(newline="", encoding="utf-8") as stream:
            self.assertEqual([r["usable_images"] for r in csv.DictReader(stream)], ["2"] * 6)
        cli_output = Path(self.temporary.name) / "cli-plots"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code, _ = self.run_cli("plot", "q70", "q70-repeat", "q80", "--output-dir", str(cli_output))
        self.assertEqual(code, 0)

    def test_comparison_requires_checkpoint_identity_and_matching_setup(self):
        known = make_record("known")
        unknown = make_record("unknown-checkpoint", checkpoint=None)
        with self.assertRaisesRegex(visualizations.VisualizationError, "comparability is unverified"):
            visualizations._compatibility_signature(unknown)
        base = visualizations._compatibility_signature(known)
        self.assertEqual(base, visualizations._compatibility_signature(make_record("q", quantile=0.5)))
        flagged = make_record("flag")
        flagged["parameters"]["official_parts_only"] = True
        other_dataset = make_record("dataset")
        other_dataset["dataset"]["split"] = "test"
        for record in (flagged, other_dataset, make_record("ckpt", checkpoint="c" * 64)):
            self.assertNotEqual(base, visualizations._compatibility_signature(record))

    def test_all_unavailable_part_combinations_are_retained_as_unverified_support(self):
        record = make_record("sparse-support")
        rows = [obs("available", 1, 1, "1", ""), obs("unavailable", 1, 2, "", "")]
        result = summaries.summarize_observations(record, rows)
        part_two = next(item for item in result["support"] if item["part_id"] == 2)
        self.assertEqual(part_two["usable_images"], 0)
        self.assertFalse(part_two["prototype_support_matches"])
        tables, support_comparable = visualizations._tables([record], [result])
        self.assertFalse(support_comparable)
        self.assertIn((1, 2), {(row["class_id"], row["part_id"]) for row in tables["support"]})


if __name__ == "__main__":
    unittest.main()
