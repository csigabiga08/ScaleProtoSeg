"""Check runtime part selection without requiring GPU inference."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import argh
import numpy as np
import torch
from PIL import Image

from segmentation.analysis.metrics import consistency
from segmentation.data.cityscapes_parts import filter_official_parts


class ConsistencyPartsTests(unittest.TestCase):
    def test_official_taxonomy(self):
        sids, pids = np.meshgrid(np.arange(35), np.arange(-1, 100), indexing="ij")
        expected = np.full_like(pids, -1)
        for sid, max_part in ((24, 4), (25, 4), (26, 5), (27, 5), (28, 5)):
            expected[sid, 2 : max_part + 2] = np.arange(1, max_part + 1)
        np.testing.assert_array_equal(filter_official_parts(sids, pids), expected)
        np.testing.assert_array_equal(pids[0], np.arange(-1, 100))

    def test_mismatched_shapes_rejected(self):
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            filter_official_parts(np.array([[24]]), np.array([[1, 5]]))

    def test_evaluator_switches_modes_without_changing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            semantic_dir = root / "annotations" / "val"
            part_dir = root / "annotations_PIDS" / "val"
            native_dir = root / "annotations_SIDS" / "val"
            for directory in (semantic_dir, part_dir, native_dir):
                directory.mkdir(parents=True)
            part_path = part_dir / "sample.npy"
            original_parts = np.array([[4, 5, 5, 5, 1]], dtype=np.int32)
            np.save(part_path, original_parts)
            original_bytes = part_path.read_bytes()
            np.save(semantic_dir / "sample.npy", [[18, 18, 19, 20, 25]])
            image_path = root / "sample.png"
            Image.new("RGB", (5, 1)).save(image_path)

            dataset = MagicMock()
            dataset.__len__.return_value = 1
            dataset.img_ids = ["sample"]
            dataset.annotations_dir = str(semantic_dir)
            dataset.get_img_path.return_value = str(image_path)
            dataset.convert_targets = np.vectorize(consistency.CITYSCAPES_19_EVAL_CATEGORIES.get)
            observation = [np.nan, 0, 0, 0, 1, np.nan, 0, "person", "sample"]
            with (
                patch.dict(os.environ, {"RESULTS_DIR": str(root)}),
                patch.object(consistency, "load_model_checkpoint"),
                patch.object(consistency, "PatchClassificationDataset", return_value=dataset),
                patch.object(consistency, "part_intersect", return_value=({}, [observation])) as intersect,
            ):
                # The default mode must work even without an SIDS cache.
                consistency.run_consistency("model", "push", "cityscapes")
                np.testing.assert_array_equal(intersect.call_args.args[2], original_parts)
                output = root / "model" / "metrics" / "push"
                score_path = output / "consistency_score_th_0.8_qt_0.8.txt"
                unfiltered_score = score_path.read_bytes()

                with self.assertRaises(FileNotFoundError):
                    consistency.run_consistency("model", "push", "cityscapes", official_parts_only=True)

                np.save(native_dir / "sample.npy", [[24, 24, 25, 26, 31]])
                consistency.run_consistency("model", "push", "cityscapes", official_parts_only=True)
                np.testing.assert_array_equal(intersect.call_args.args[2], [[4, -1, -1, 5, -1]])
                self.assertTrue((output / "official_parts" / score_path.name).is_file())
                self.assertEqual(score_path.read_bytes(), unfiltered_score)
                self.assertEqual(part_path.read_bytes(), original_bytes)

    def test_official_flag_rejects_pascal_before_loading_model(self):
        with patch.object(consistency, "load_model_checkpoint") as load_model:
            with self.assertRaisesRegex(ValueError, "only for cityscapes"):
                consistency.run_consistency("model", "push", "pascal", official_parts_only=True)
            load_model.assert_not_called()

    def test_pascal_semantic_match_cli_preserves_legacy_results_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            semantic_dir = root / "annotations" / "val"
            part_dir = root / "annotations_PIDS" / "val"
            native_dir = root / "annotations_SIDS" / "val"
            for directory in (semantic_dir, part_dir, native_dir):
                directory.mkdir(parents=True)
            part_path = part_dir / "sample.npy"
            original_parts = np.array([[3, 3, 3, 7, 3, -1, 2, 11]], dtype=np.int32)
            np.save(part_path, original_parts)
            np.save(semantic_dir / "sample.npy", [[15, 15, 2, 0, 255, 15, 20, 15]])
            original_bytes = part_path.read_bytes()
            image_path = root / "sample.png"
            Image.new("RGB", (8, 1)).save(image_path)

            dataset = MagicMock()
            dataset.__len__.return_value = 1
            dataset.img_ids = ["sample"]
            dataset.annotations_dir = str(semantic_dir)
            dataset.get_img_path.return_value = str(image_path)
            dataset.convert_targets = np.vectorize(consistency.PASCAL_ID_MAPPING.get)
            observation = [np.nan] * (consistency.MAX_PARTS_PASCAL + 1)
            observation[3] = 1
            observation += [0, "person", "sample"]
            with (
                patch.dict(os.environ, {"RESULTS_DIR": str(root)}),
                patch.object(consistency, "load_model_checkpoint"),
                patch.object(consistency, "PatchClassificationDataset", return_value=dataset),
                patch.object(consistency, "part_intersect", return_value=({}, [observation])) as intersect,
            ):
                # Legacy mode needs no panoptic semantic cache.
                consistency.run_consistency("model", "push", "pascal")
                np.testing.assert_array_equal(intersect.call_args.args[2], original_parts)
                output = root / "model" / "metrics" / "push"
                legacy_files = {path.name: path.read_bytes() for path in output.iterdir()}

                with self.assertRaises(FileNotFoundError):
                    consistency.run_consistency("model", "push", "pascal", require_semantic_match=True)

                # A bicycle saddle (PID 3) inside the VOC person mask must be removed;
                # a true person right-eye pixel (also PID 3) must survive the ID shift.
                np.save(native_dir / "sample.npy", [[2, 15, 2, 0, 15, 15, 20, 16]])
                argh.dispatch_command(
                    consistency.run_consistency,
                    argv=["model", "push", "pascal", "--require-semantic-match"],
                )
                np.testing.assert_array_equal(intersect.call_args.args[2], [[0, 3, 3, 0, 0, 0, 2, 0]])
                np.testing.assert_array_equal(intersect.call_args.args[1], [[16, 16, 3, 1, 0, 16, 21, 16]])
                for name, contents in legacy_files.items():
                    self.assertEqual((output / name).read_bytes(), contents)
                    self.assertTrue((output / "semantic_match" / name).is_file())
                self.assertEqual(part_path.read_bytes(), original_bytes)

                argh.dispatch_command(
                    consistency.run_consistency,
                    argv=["model", "push", "pascal", "--require-semantic-match", "--exclude-background-centroid"],
                )
                self.assertTrue(intersect.call_args.kwargs["exclude_background_centroid"])
                self.assertTrue(
                    (output / "semantic_match" / "foreground_centroids" / "consistency_score_th_0.8_qt_0.8.txt").is_file()
                )

                # Reject shapes which NumPy would otherwise silently broadcast.
                np.save(native_dir / "sample.npy", [[15]])
                with self.assertRaisesRegex(ValueError, "matching shapes"):
                    consistency.run_consistency("model", "push", "pascal", require_semantic_match=True)

    def test_semantic_match_flag_rejects_cityscapes_before_loading_model(self):
        with patch.object(consistency, "load_model_checkpoint") as load_model:
            with self.assertRaisesRegex(ValueError, "only for pascal"):
                consistency.run_consistency("model", "push", "cityscapes", require_semantic_match=True)
            load_model.assert_not_called()

    def test_background_centroid_can_create_a_false_part_hit(self):
        image = Image.new("RGB", (5, 5))
        semantic = np.full((5, 5), 2, dtype=np.int64)
        parts = np.zeros((5, 5), dtype=np.int64)
        parts[0, 0] = 1
        distances = torch.full((1, 1, 5, 5), 10.0)
        distances[0, 0, 2, 2] = 0.1  # Active at the background centroid, not the part.
        model = MagicMock()
        model.epsilon = 1e-4
        model.prototype_class_identity = torch.zeros((1, 21))
        model.prototype_class_identity[0, 1] = 1
        model.forward_from_conv_features.return_value = (None, distances)

        with (
            patch.object(consistency, "to_normalized_tensor", return_value=torch.zeros((3, 5, 5))),
            patch.object(torch.Tensor, "cuda", lambda self: self),
            patch.object(consistency, "MAX_PARTS", 24, create=True),
        ):
            _, legacy = consistency.part_intersect(image, semantic, parts, model, {1: "aeroplane"}, "sample", 0.8)
            _, foreground = consistency.part_intersect(
                image, semantic, parts, model, {1: "aeroplane"}, "sample", 0.8,
                exclude_background_centroid=True,
            )

        self.assertEqual(legacy[0][1], 1)
        self.assertEqual(foreground[0][1], 0)


if __name__ == "__main__":
    unittest.main()
