"""Contract tests for measurement bundles: writer, loader, and root resolution."""

import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from segmentation.analysis.metrics.catalog import (
    CatalogError,
    iter_measurements,
    load_measurement,
    load_observations,
    resolve_catalog_root,
)
from segmentation.analysis.metrics.output import MetricOutput


class MetricCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "catalog"
        self.root.mkdir()
        self.checkpoint = Path(self.temporary.name) / "checkpoint.pth"
        self.checkpoint.write_bytes(b"tiny checkpoint payload")

    def make_output(self, **parameters):
        return MetricOutput(
            "consistency",
            self.checkpoint,
            "pascal",
            model_name="fixture",
            training_phase="push",
            protocol={
                "metric": "consistency",
                "quantile": 0.8,
                "consistency": {"threshold": 0.8},
                "observation_columns": ["part_1", "part_2", "part_3"],
            },
            **parameters,
        )

    def finished_bundle(self):
        output = self.make_output()
        output.set_candidates(["used", "other"])
        output.start_sample("used")
        output.add_observation(
            sample_id="used", class_id=3, prototype_id=12, repetition=0, part_1=1, part_2=None, part_3=0,
        )
        output.complete_sample("used")
        output.mark_excluded("other")
        output.finish(value=1.0, numerator=1, denominator=1)
        return output.path

    def test_exported_root_overrides_checkout_dotenv(self):
        dotenv = Path(self.temporary.name) / ".env"
        dotenv.write_text(f"MEASUREMENT_CATALOG_ROOT={self.root}\n")
        exported = Path(self.temporary.name) / "exported"
        exported.mkdir()
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(exported)}):
            self.assertEqual(resolve_catalog_root(checkout=Path(self.temporary.name)), exported.resolve())

    def test_checkout_dotenv_is_used_without_overwriting_exported_values(self):
        dotenv = Path(self.temporary.name) / ".env"
        dotenv.write_text(f"MEASUREMENT_CATALOG_ROOT='{self.root}'\n")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_catalog_root(checkout=Path(self.temporary.name)), self.root.resolve())
            self.assertEqual(os.environ["MEASUREMENT_CATALOG_ROOT"], str(self.root))

    def test_missing_or_unwritable_root_fails_without_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(CatalogError, "MEASUREMENT_CATALOG_ROOT"):
                resolve_catalog_root(checkout=Path(self.temporary.name))
        missing = Path(self.temporary.name) / "missing"
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(missing)}):
            with self.assertRaisesRegex(CatalogError, "exist"):
                resolve_catalog_root(checkout=Path(self.temporary.name))
        with (
            patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}),
            patch("segmentation.analysis.metrics.catalog.os.access", return_value=False),
        ):
            with self.assertRaisesRegex(CatalogError, "not writable"):
                resolve_catalog_root(checkout=Path(self.temporary.name))

    def test_dotenv_uses_standard_interpolation_without_loading_other_settings(self):
        dotenv = Path(self.temporary.name) / ".env"
        dotenv.write_text(
            f'CATALOG_PARENT="{self.root.parent}"\n'
            'MEASUREMENT_CATALOG_ROOT="${CATALOG_PARENT}/catalog" # shared output\n'
            'UNRELATED_SETTING=do-not-load\n'
        )
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_catalog_root(checkout=dotenv.parent), self.root.resolve())
            self.assertNotIn("CATALOG_PARENT", os.environ)
            self.assertNotIn("UNRELATED_SETTING", os.environ)

    def test_helpers_import_without_torch_or_settings(self):
        code = (
            "import sys; "
            "from segmentation.analysis.metrics import catalog, output; "
            "assert 'torch' not in sys.modules; "
            "assert 'settings' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True, cwd=Path.cwd())

    def test_writer_publishes_hashed_observations_and_population_counts(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output(quantile=0.8, seed=41)
            self.assertEqual(output.result["status"], "running")
            output.set_candidates(["missing", "empty", "used"])
            output.mark_excluded("missing")
            output.start_sample("empty")
            output.complete_sample("empty")
            output.start_sample("used")
            output.add_observation(
                sample_id="used", class_id=3, prototype_id=12, repetition=0, part_1=1, part_2=None, part_3=0,
            )
            output.complete_sample("used")
            output.set_selected_prototype_ids([12, 19])
            output.finish(value=0.5, numerator=1, denominator=2)

            bundle = output.path
            self.assertEqual({path.name for path in bundle.iterdir()}, {"result.json", "observations.csv"})
            record = load_measurement(bundle)
            self.assertEqual(record["status"], "complete")
            self.assertEqual(
                record["scores"]["consistency"],
                {"value": 0.5, "unit": "fraction", "numerator": 1, "denominator": 2},
            )
            self.assertEqual(record["parameters"], {"quantile": 0.8, "seed": 41})
            self.assertEqual(record["population"], {
                "candidates": 3, "contributing": 1, "excluded": 1, "failed": 0,
                "not_attempted": 0, "observations": 1,
            })
            self.assertEqual(record["selected_prototype_ids"], [12, 19])
            artifact = record["artifacts"]["observations"]
            self.assertEqual(artifact["bytes"], (bundle / "observations.csv").stat().st_size)
            self.assertEqual(artifact["sha256"], hashlib.sha256((bundle / "observations.csv").read_bytes()).hexdigest())
            _result, rows = load_observations(bundle)
            self.assertEqual(rows, [{
                "sample_id": "used", "class_id": "3", "prototype_id": "12", "repetition": "0",
                "part_1": "1", "part_2": "", "part_3": "0",
            }])

    def test_nonfinite_score_is_null_with_reason(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output()
            output.set_candidates([])
            output.finish(value=float("nan"), reason="no eligible prototype observations")
            score = load_measurement(output.path)["scores"]["consistency"]
            self.assertIsNone(score["value"])
            self.assertEqual(score["reason"], "no eligible prototype observations")

    def test_metadata_requires_its_sections_and_artifact_for_complete_records(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            bundle = self.finished_bundle()
            original = json.loads((bundle / "result.json").read_text())
            for mutation in (
                lambda value: value.pop("protocol"),
                lambda value: value.pop("parameters"),
                lambda value: value["artifacts"].clear(),
                lambda value: value["scores"].clear(),
                lambda value: value["scores"]["consistency"].update(value=None),
                lambda value: value["artifacts"]["observations"].update(sha256="xyz"),
            ):
                result = json.loads(json.dumps(original))
                mutation(result)
                (bundle / "result.json").write_text(json.dumps(result))
                with self.assertRaisesRegex(CatalogError, "Invalid measurement metadata"):
                    load_measurement(bundle)

    def test_loader_rejects_tampered_artifacts_and_inconsistent_counts(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            bundle = self.finished_bundle()
            original = json.loads((bundle / "result.json").read_text())
            for key, value, message in (
                ("contributing", 2, "Contributing sample count"),
                ("observations", 5, "Observation count"),
            ):
                result = json.loads(json.dumps(original))
                result["population"][key] = value
                (bundle / "result.json").write_text(json.dumps(result))
                with self.assertRaisesRegex(CatalogError, message):
                    load_measurement(bundle)
            (bundle / "result.json").write_text(json.dumps(original))

            result = json.loads(json.dumps(original))
            result["protocol"]["observation_columns"] = ["part_1"]
            (bundle / "result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(CatalogError, "columns disagree"):
                load_observations(bundle)
            (bundle / "result.json").write_text(json.dumps(original))

            same_size = (bundle / "observations.csv").read_bytes().replace(b"1,,0", b"0,,0")
            (bundle / "observations.csv").write_bytes(same_size)
            with self.assertRaisesRegex(CatalogError, "hash"):
                load_measurement(bundle)
            self.assertEqual(load_measurement(bundle, verify_hashes=False)["status"], "complete")
            (bundle / "observations.csv").write_text("corrupt\n")
            with self.assertRaisesRegex(CatalogError, "byte size"):
                load_measurement(bundle)

    def test_artifact_paths_cannot_escape_their_bundle(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            bundle = self.finished_bundle()
            result = json.loads((bundle / "result.json").read_text())
            result["artifacts"]["observations"]["path"] = "../escape.csv"
            (bundle / "result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(CatalogError, "out-of-bundle"):
                load_measurement(bundle)

    def test_record_dataset_registers_candidates(self):
        dataset = type("Dataset", (), {"img_ids": ["one", "two"]})()
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output()
            output.record_dataset(dataset)
            self.assertEqual(output.result["dataset"]["candidate_count"], 2)

    def test_loaded_model_runtime_is_added_to_protocol_not_model_seed(self):
        model = type("Model", (), {"epsilon": 0.0007, "prototype_activation_function": "linear"})()
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output(seed=41)
            output.record_model_runtime(model, grouping_active=True, excluded_evaluation_class_ids=[1, 5])
            protocol = output.result["protocol"]
            self.assertEqual(protocol["model"]["epsilon"], 0.0007)
            self.assertEqual(protocol["model"]["prototype_activation_function"], "linear")
            self.assertTrue(protocol["prototype_selection"]["grouping_active"])
            self.assertNotIn("seed", output.result["model"])
            self.assertEqual(output.result["parameters"]["seed"], 41)

    def test_non_scalar_runtime_model_attributes_are_recorded_as_null(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output()
            output.record_model_runtime(MagicMock(), grouping_active=False, excluded_evaluation_class_ids=[])
        self.assertIsNone(output.result["protocol"]["model"]["epsilon"])
        self.assertIsNone(output.result["protocol"]["model"]["prototype_activation_function"])

    def test_discovery_yields_only_validated_uuid_records(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output()
            output.set_candidates([])
            output.finish(value=float("nan"), reason="no candidates")
            (self.root / "measurements" / "unrelated").mkdir()
            records = list(iter_measurements(self.root))
            self.assertEqual([record["measurement_id"] for record in records], [output.measurement_id])

    def test_failed_processing_is_published_with_partial_observations(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output()
            output.set_candidates(["first", "later"])
            output.start_sample("first")
            output.add_observation(
                sample_id="first", class_id=1, prototype_id=4, repetition=0, part_1=0, part_2=1, part_3=None,
            )
            output.fail(RuntimeError("fixture failed"))
            record, rows = load_observations(output.path)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["failure"]["type"], "RuntimeError")
            self.assertEqual(record["population"]["failed"], 1)
            self.assertEqual(record["population"]["not_attempted"], 1)
            self.assertEqual(len(rows), 1)

    def test_failed_measurement_records_error(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output()
            output.fail(RuntimeError("boom"))
            record = json.loads((output.path / "result.json").read_text())
            self.assertEqual(record["failure"], {"type": "RuntimeError", "message": "boom"})

    def test_writer_rejects_invalid_observations_and_unattempted_completion(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output()
            output.set_candidates(["sample", "pending"])
            output.start_sample("sample")
            row = dict(sample_id="sample", class_id=1, prototype_id=4, repetition=0, part_1=0, part_2=1, part_3=None)
            output.add_observation(**row)
            with self.assertRaisesRegex(CatalogError, "Duplicate"):
                output.add_observation(**row)
            with self.assertRaisesRegex(CatalogError, "zero, one, or unavailable"):
                output.add_observation(**{**row, "repetition": 1, "part_1": 0.5})
            with self.assertRaisesRegex(CatalogError, "outside the candidate population"):
                output.add_observation(**{**row, "sample_id": "unknown"})
            output.complete_sample("sample")
            with self.assertRaisesRegex(CatalogError, "unattempted"):
                output.finish(value=1.0, numerator=1, denominator=1)

    def test_terminal_records_cannot_be_modified(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output()
            output.set_candidates([])
            output.finish(value=0.0, numerator=0, denominator=1)
            with self.assertRaisesRegex(CatalogError, "terminal"):
                output.finish(value=1.0, numerator=1, denominator=1)

    def test_concurrent_allocations_are_disjoint(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            with ThreadPoolExecutor(max_workers=8) as pool:
                paths = list(pool.map(lambda _: self.make_output().path, range(16)))
            self.assertEqual(len(set(paths)), 16)
            self.assertTrue(all(path.parent == self.root / "measurements" for path in paths))

    def test_removed_output_directory_parameter_is_not_part_of_writer_api(self):
        self.assertNotIn("output_dir", inspect.signature(MetricOutput).parameters)
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            with self.assertRaisesRegex(TypeError, "output_dir was removed"):
                self.make_output(output_dir=self.root)


if __name__ == "__main__":
    unittest.main()
