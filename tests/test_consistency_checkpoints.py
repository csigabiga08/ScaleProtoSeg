"""Checkpoint selection through the consistency CLI, without GPU inference."""

import os
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

from pydantic import ValidationError

from segmentation.analysis.metrics import consistency
from segmentation.analysis.metrics.catalog import CatalogError
from segmentation.analysis.metrics.protocol import ConsistencyRunInput


def run_input(model_name="model", training_phase="push", data_type="pascal", **overrides):
    return ConsistencyRunInput(
        model_name=model_name,
        training_phase=training_phase,
        data_type=data_type,
        **overrides,
    )


class DatasetReached(Exception):
    pass


class ConsistencyCheckpointTests(unittest.TestCase):
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
                    consistency.run_consistency(run_input())
                load.assert_not_called()

    def test_runner_input_rejects_unknown_fields(self):
        self.assertNotIn("output_dir", inspect.signature(consistency.run_consistency).parameters)
        with self.assertRaises(ValidationError):
            run_input(output_dir="ignored")

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
                    consistency.run_consistency(run_input())
            load.assert_called_once_with(str(checkpoint))
            bundles = list((catalog_root / "measurements").iterdir())
            self.assertEqual(len(bundles), 1)
            record = json.loads((bundles[0] / "result.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["failure"]["type"], "ValueError")
            self.assertEqual(record["population"], {"candidates": None, "contributing": 0, "observations": 0})

    def test_checkpoint_selection_from_typed_input(self):
        cases = [
            ({}, False, ["model/checkpoints/push_last.pth"], None),
            ({"group_name": "group"}, False,
             ["model/checkpoints/push_last.pth", "group/checkpoints/th-0.05-nopush-group_last.pth"], "group"),
            ({"checkpoint": "custom/best.pth"}, False, ["custom/best.pth"], None),
            ({"checkpoint": "custom/best.pth"}, True, ["custom/best.pth"], "model"),
            ({"group_name": "group", "group_checkpoint": "custom/group.pth"}, False,
             ["model/checkpoints/push_last.pth", "custom/group.pth"], "group"),
            ({"checkpoint": "custom/best.pth", "group_checkpoint": "custom/group.pth"}, True,
             ["custom/best.pth", "custom/group.pth"], "group"),
        ]
        for overrides, grouped, paths, filter_source in cases:
            with self.subTest(overrides=overrides, grouped=grouped), tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
                root = Path(root)
                overrides = dict(overrides)
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
                for name in ("checkpoint", "group_checkpoint"):
                    if name in overrides:
                        overrides[name] = str(root / overrides[name])
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
                        consistency.run_consistency(
                            run_input(data_type="cityscapes", **overrides)
                        )
                    self.assertEqual(load.call_args_list, [call(os.path.join(root, p)) for p in paths])
                    if filter_source is None:
                        proto_filter.assert_not_called()
                    else:
                        proto_filter.assert_called_once_with(model if filter_source == "model" else group)


if __name__ == "__main__":
    unittest.main()
