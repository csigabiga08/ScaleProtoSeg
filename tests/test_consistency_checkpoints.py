"""Checkpoint selection through the consistency CLI, without GPU inference."""

import os
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

import argh

from segmentation.analysis.metrics import consistency
from segmentation.analysis.metrics.catalog import CatalogError


class DatasetReached(Exception):
    pass


class ConsistencyCheckpointTests(unittest.TestCase):
    def test_loaded_model_protocol_records_actual_model_and_input_rules(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            checkpoint = root / "model" / "checkpoints" / "push_last.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fixture checkpoint bytes")
            catalog_root = root / "catalog"
            catalog_root.mkdir()
            model = MagicMock()
            model.cuda.return_value = model
            model.epsilon = 0.0007
            model.prototype_activation_function = "linear"
            with (
                patch.dict(os.environ, {
                    "RESULTS_DIR": str(root),
                    "MEASUREMENT_CATALOG_ROOT": str(catalog_root),
                }),
                patch.object(consistency, "load_model_checkpoint", return_value=model),
                patch.object(consistency, "PatchClassificationDataset", side_effect=DatasetReached),
            ):
                with self.assertRaises(DatasetReached):
                    consistency.run_consistency("model", "push", "pascal")
            bundle = next((catalog_root / "measurements").iterdir())
            result = json.loads((bundle / "result.json").read_text())
            self.assertEqual(result["protocol"]["model"]["epsilon"], 0.0007)
            self.assertEqual(result["protocol"]["model"]["prototype_activation_function"], "linear")
            self.assertIn("native cached image resolution", result["protocol"]["input"]["resolution"])
            self.assertIn("group_projection", result["protocol"]["prototype_selection"]["group_nonzero_rule"])
            self.assertEqual(result["protocol"]["prototype_selection"]["excluded_evaluation_class_ids"], [1, 5, 10, 12, 19])
            self.assertNotIn("seed", result["model"])
            self.assertIsNone(result["model"]["training_seed"])

    def test_catalog_root_is_validated_before_model_loading(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            checkpoint = root / "model" / "checkpoints" / "push_last.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fixture checkpoint bytes")
            with (
                patch.dict(os.environ, {
                    "RESULTS_DIR": str(root),
                    "MEASUREMENT_CATALOG_ROOT": str(root / "missing-catalog"),
                }),
                patch.object(consistency, "load_model_checkpoint") as load,
            ):
                with self.assertRaises(CatalogError):
                    consistency.run_consistency("model", "push", "pascal")
                load.assert_not_called()

    def test_output_directory_cli_option_is_removed(self):
        self.assertNotIn("output_dir", inspect.signature(consistency.run_consistency).parameters)
        with patch.object(consistency, "load_model_checkpoint") as load:
            with self.assertRaises(SystemExit):
                argh.dispatch_command(
                    consistency.run_consistency,
                    argv=["model", "push", "pascal", "--output-dir", "ignored"],
                )
            load.assert_not_called()

    def test_checkpoint_loading_failure_leaves_failed_initialization_record(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            checkpoint = root / "model" / "checkpoints" / "push_last.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fixture checkpoint bytes")
            catalog_root = root / "catalog"
            catalog_root.mkdir()
            with (
                patch.dict(os.environ, {
                    "RESULTS_DIR": str(root),
                    "MEASUREMENT_CATALOG_ROOT": str(catalog_root),
                }),
                patch.object(consistency, "load_model_checkpoint", side_effect=ValueError("bad checkpoint")) as load,
            ):
                with self.assertRaisesRegex(ValueError, "bad checkpoint"):
                    consistency.run_consistency("model", "push", "pascal")
            load.assert_called_once_with(str(checkpoint))
            bundles = list((catalog_root / "measurements").iterdir())
            self.assertEqual(len(bundles), 1)
            record = json.loads((bundles[0] / "result.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["failure"]["type"], "ValueError")
            self.assertEqual(record["artifacts"], {})

    def test_checkpoint_selection_through_cli(self):
        cases = [
            ([], False, ["model/checkpoints/push_last.pth"], None),
            (["--group-name", "group"], False,
             ["model/checkpoints/push_last.pth", "group/checkpoints/th-0.05-nopush-group_last.pth"], "group"),
            (["--checkpoint", "custom/best.pth"], False, ["custom/best.pth"], None),
            (["--checkpoint", "custom/best.pth"], True, ["custom/best.pth"], "model"),
            (["--group-name", "group", "--group-checkpoint", "custom/group.pth"], False,
             ["model/checkpoints/push_last.pth", "custom/group.pth"], "group"),
            (["--checkpoint", "custom/best.pth", "--group-checkpoint", "custom/group.pth"], True,
             ["custom/best.pth", "custom/group.pth"], "group"),
        ]
        for flags, grouped, paths, filter_source in cases:
            with self.subTest(flags=flags, grouped=grouped), tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
                root = Path(root)
                catalog_root = root / "catalog"
                catalog_root.mkdir()
                for relative in (
                    "model/checkpoints/push_last.pth",
                    "group/checkpoints/th-0.05-nopush-group_last.pth",
                    "custom/best.pth",
                    "custom/group.pth",
                ):
                    checkpoint = root / relative
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.write_bytes(b"fixture checkpoint bytes")
                flags = list(flags)
                for index, value in enumerate(flags[:-1]):
                    if value in ("--checkpoint", "--group-checkpoint"):
                        flags[index + 1] = str(root / flags[index + 1])
                paths = [str(root / path) if path.startswith("custom/") else path for path in paths]
                model = MagicMock()
                model.cuda.return_value = model
                if not grouped:
                    del model.group_projection
                group = MagicMock()
                with (
                    patch.dict(os.environ, {
                        "RESULTS_DIR": str(root),
                        "MEASUREMENT_CATALOG_ROOT": str(catalog_root),
                    }),
                    patch.object(consistency, "load_model_checkpoint", side_effect=[model, group]) as load,
                    patch.object(consistency, "proto_filter", return_value=[1]) as proto_filter,
                    patch.object(consistency, "PatchClassificationDataset", side_effect=DatasetReached),
                ):
                    with self.assertRaises(DatasetReached):
                        argh.dispatch_command(
                            consistency.run_consistency, argv=["model", "push", "cityscapes", *flags]
                        )
                    self.assertEqual(load.call_args_list, [call(os.path.join(root, p)) for p in paths])
                    if filter_source is None:
                        proto_filter.assert_not_called()
                    else:
                        proto_filter.assert_called_once_with(model if filter_source == "model" else group)


if __name__ == "__main__":
    unittest.main()
