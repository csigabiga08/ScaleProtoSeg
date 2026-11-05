"""Contract tests for native metric measurement bundles."""

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
    resolve_catalog_root,
)
from segmentation.analysis.metrics.output import MetricOutput
import segmentation.analysis.metrics.output as metric_output


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
                "version": 1,
                "metric": "consistency",
                "observation_columns": ["part_1", "part_2", "part_3"],
            },
            **parameters,
        )

    def test_helpers_import_without_torch_or_settings(self):
        code = (
            "import sys; "
            "from segmentation.analysis.metrics import catalog, output; "
            "assert 'torch' not in sys.modules; "
            "assert 'settings' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True, cwd=Path.cwd())

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

    def test_identical_input_paths_are_hashed_once_for_separate_checkpoint_roles(self):
        with (
            patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}),
            patch.object(metric_output, "sha256_file", wraps=metric_output.sha256_file) as hash_file,
        ):
            output = MetricOutput(
                "consistency",
                self.checkpoint,
                "pascal",
                model_name="fixture",
                training_phase="push",
                protocol={"version": 1, "observation_columns": ["part_1"]},
                group_checkpoint=self.checkpoint,
            )
            inputs = output.result["model"]["inputs"]
            self.assertEqual(inputs["prototype_checkpoint"]["sha256"], inputs["group_checkpoint"]["sha256"])
            hash_file.assert_called_once_with(self.checkpoint.resolve())

    def test_writer_publishes_three_hashed_files_and_reconciled_population(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output()
            self.assertEqual(output.result["status"], "running")
            output.set_candidates(["missing", "empty", "used"])
            output.mark_sample("missing", "excluded", "missing_part_annotation")
            output.start_sample("empty")
            output.complete_sample("empty")
            output.start_sample("used")
            output.add_observation(
                sample_id="used",
                class_id=3,
                prototype_id=12,
                repetition=0,
                part_1=1,
                part_2=None,
                part_3=0,
            )
            output.complete_sample("used")
            output.set_selected_prototype_ids([12, 19])
            output.finish(value=0.5, numerator=1, denominator=2)

            bundle = output.path
            self.assertEqual(
                {path.name for path in bundle.iterdir()},
                {"result.json", "observations.csv", "samples.csv"},
            )
            record = load_measurement(bundle)
            self.assertEqual(record["status"], "complete")
            self.assertEqual(record["scores"]["consistency"], {
                "value": 0.5,
                "unit": "fraction",
                "numerator": 1,
                "denominator": 2,
            })
            self.assertEqual(record["population"]["candidates"], 3)
            self.assertEqual(record["population"]["attempted"], 2)
            self.assertEqual(record["population"]["contributing"], 1)
            self.assertEqual(record["population"]["excluded"], 1)
            self.assertEqual(record["population"]["processed_no_observations"], 1)
            self.assertEqual(record["selected_prototype_ids"], [12, 19])
            self.assertEqual(record["contributing_prototype_ids"], [12])
            with (bundle / "observations.csv").open(newline="") as stream:
                observation = next(csv.DictReader(stream))
            self.assertEqual(observation["part_1"], "1")
            self.assertEqual(observation["part_2"], "")
            self.assertEqual(observation["part_3"], "0")
            with (bundle / "samples.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["outcome"] for row in rows], [
                "excluded", "processed_no_observations", "contributed",
            ])
            for artifact in record["artifacts"].values():
                artifact_path = bundle / artifact["path"]
                self.assertEqual(artifact["bytes"], artifact_path.stat().st_size)
                self.assertEqual(
                    artifact["sha256"],
                    hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                )

    def test_nonfinite_score_is_null_with_reason_and_complete_bundle(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output()
            output.set_candidates([])
            output.finish(value=float("nan"), reason="no eligible prototype observations")
            record = load_measurement(output.path)
            self.assertIsNone(record["scores"]["consistency"]["value"])
            self.assertEqual(
                record["scores"]["consistency"]["reason"],
                "no eligible prototype observations",
            )
            self.assertIsNone(record["scores"]["consistency"]["numerator"])

    def test_terminal_records_require_native_metadata_sections(self):
        required_sections = (
            ("model", "inputs"),
            ("protocol", None),
            ("execution", None),
            ("provenance_quality", None),
            ("dataset", "identity"),
            ("execution", "packages"),
            ("provenance_quality", "missing_fields"),
        )
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output()
            output.set_candidates([])
            output.finish(value=float("nan"), reason="no eligible prototype observations")
            result_path = output.path / "result.json"
            original = json.loads(result_path.read_text())
            for section, child in required_sections:
                with self.subTest(section=section, child=child):
                    result = json.loads(json.dumps(original))
                    if child is None:
                        result.pop(section)
                    elif section in result:
                        result[section].pop(child)
                    result_path.write_text(json.dumps(result))
                    with self.assertRaisesRegex(CatalogError, "missing"):
                        load_measurement(output.path)
            result_path.write_text(json.dumps(original))
            self.assertEqual(load_measurement(output.path)["status"], "complete")

    def test_explicit_unknown_provenance_is_valid_and_noise_seed_is_not_model_seed(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output(seed=41)
            output.result["protocol"]["model"] = {"epsilon": None}
            output.result["protocol"]["parts"] = {
                "mapping_identity": None,
                "annotation_provenance": "unknown",
            }
            output.result["provenance_quality"]["missing_fields"].append("model epsilon")
            output.set_candidates([])
            output.finish(value=0.0, numerator=0, denominator=1)
            record = load_measurement(output.path)
            self.assertNotIn("seed", record["model"])
            self.assertIsNone(record["model"]["training_seed"])
            self.assertEqual(record["parameters"]["seed"], 41)
            self.assertIsNone(record["protocol"]["model"]["epsilon"])
            self.assertEqual(record["protocol"]["parts"]["annotation_provenance"], "unknown")

    def test_dataset_records_available_pascal_part_mapping_and_unknown_annotation_provenance(self):
        mapping_root = Path(self.temporary.name) / "pascal-source"
        mapping_root.mkdir()
        mapping_path = mapping_root / "parts.yaml"
        mapping_path.write_text(
            "scene_class2part_classes:\n"
            "  person: [head, torso]\n"
            "countable_pids_groupings:\n"
            "  person:\n"
            "    head: [1, 2]\n"
        )
        annotation_root = Path(self.temporary.name) / "pascal-processed" / "annotations" / "val"
        annotation_root.mkdir(parents=True)
        dataset = type(
            "Dataset",
            (),
            {"img_ids": ["one"], "annotations_dir": str(annotation_root)},
        )()
        with (
            patch.dict(os.environ, {
                "MEASUREMENT_CATALOG_ROOT": str(self.root),
                "SOURCE_DATA_PATH_PASCAL": str(mapping_root),
            }),
        ):
            output = self.make_output()
            output.record_dataset(dataset)
            parts = output.result["protocol"]["parts"]
            self.assertEqual(parts["mapping_identity"]["path"], str(mapping_path.resolve()))
            self.assertEqual(parts["mapping_identity"]["status"], "available")
            self.assertEqual(
                parts["class_dependent_part_definitions"]["person"],
                {
                    "semantic_class_id": 1,
                    "part_ids": {"head": 1, "torso": 2},
                    "countable_pid_groups": {"head": [1, 2]},
                },
            )
            self.assertEqual(parts["annotation_provenance"], "unknown")

    def test_loaded_model_runtime_is_added_to_protocol_not_model_seed(self):
        model = type(
            "Model",
            (),
            {"epsilon": 0.0007, "prototype_activation_function": "linear"},
        )()
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output(seed=41)
            output.record_model_runtime(
                model,
                grouping_active=True,
                excluded_evaluation_class_ids=[1, 5],
            )
            protocol = output.result["protocol"]
            self.assertEqual(protocol["model"]["epsilon"], 0.0007)
            self.assertEqual(protocol["model"]["prototype_activation_function"], "linear")
            self.assertEqual(protocol["input"]["resolution"], "native cached image resolution; no metric resize before feature extraction")
            self.assertEqual(protocol["input"]["normalization"]["mean"], [0.485, 0.456, 0.406])
            self.assertTrue(protocol["prototype_selection"]["grouping_active"])
            self.assertIn("torch.nonzero(group_projection", protocol["prototype_selection"]["group_nonzero_rule"])
            self.assertNotIn("seed", output.result["model"])
            self.assertEqual(output.result["parameters"]["seed"], 41)

    def test_unknown_runtime_model_attributes_are_recorded_as_explicit_unknowns(self):
        model = MagicMock()
        try:
            with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
                output = self.make_output()
                output.record_model_runtime(
                    model,
                    grouping_active=False,
                    excluded_evaluation_class_ids=[],
                )
        except Exception:
            self.fail("non-scalar model attributes must be recorded as unknown, not serialized recursively")
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

    def test_failed_processing_is_published_and_later_samples_stay_unattempted(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output()
            output.set_candidates(["first", "later"])
            output.start_sample("first")
            output.fail(RuntimeError("fixture failed"))
            record = load_measurement(output.path)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["population"]["failed"], 1)
            with (output.path / "samples.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["outcome"] for row in rows], ["failed", "not_attempted"])

    def test_failure_metadata_redacts_secret_like_values(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output()
            output.set_candidates([])
            output.fail(RuntimeError("api_token=do-not-persist"))
            record = json.loads((output.path / "result.json").read_text())
            self.assertNotIn("do-not-persist", record["failure"]["message"])
            self.assertIn("[redacted]", record["failure"]["message"])

    def test_duplicate_observation_keys_and_corrupt_artifacts_are_rejected(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output()
            output.set_candidates(["sample"])
            output.start_sample("sample")
            row = dict(
                sample_id="sample", class_id=1, prototype_id=4, repetition=0,
                part_1=0, part_2=1, part_3=None,
            )
            output.add_observation(**row)
            with self.assertRaisesRegex(CatalogError, "Duplicate"):
                output.add_observation(**row)
            output.fail(ValueError("duplicate observation"))
            (output.path / "observations.csv").write_text("corrupt\n")
            with self.assertRaisesRegex(CatalogError, "byte size|hash"):
                load_measurement(output.path)

    def test_terminal_records_cannot_be_modified(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            output = self.make_output()
            output.set_candidates([])
            output.finish(value=0.0, numerator=0, denominator=1)
            with self.assertRaisesRegex(CatalogError, "terminal"):
                output.finish(value=1.0, numerator=1, denominator=1)

    def test_concurrent_allocations_are_disjoint(self):
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            def allocate(_):
                return self.make_output().path

            with ThreadPoolExecutor(max_workers=8) as pool:
                paths = list(pool.map(allocate, range(16)))
            self.assertEqual(len(set(paths)), 16)
            self.assertTrue(all(path.parent == self.root / "measurements" for path in paths))

    def test_removed_output_directory_parameter_is_not_part_of_writer_api(self):
        self.assertNotIn("output_dir", inspect.signature(MetricOutput).parameters)
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            with self.assertRaisesRegex(TypeError, "output_dir was removed"):
                self.make_output(output_dir=self.root)


if __name__ == "__main__":
    unittest.main()
