"""Check stability centroid selection and CLI routing without a GPU."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import argh
import numpy as np
import torch
from PIL import Image

from segmentation.analysis.metrics import stability


class StabilityPartsTests(unittest.TestCase):
    def test_background_centroid_can_create_a_false_part_hit(self):
        image = Image.new("RGB", (5, 5))
        semantic = np.full((5, 5), 2, dtype=np.int64)
        parts = np.zeros((5, 5), dtype=np.int64)
        parts[0, 0] = 1
        distances = torch.full((1, 1, 5, 5), 10.0)
        distances[0, 0, 2, 2] = 0.1  # Only the background centroid is active.
        model = MagicMock()
        model.epsilon = 1e-4
        model.prototype_class_identity = torch.zeros((1, 21))
        model.prototype_class_identity[0, 1] = 1
        model.forward_from_conv_features.return_value = (None, distances)

        with (
            patch.object(stability, "to_normalized_tensor", return_value=torch.zeros((3, 5, 5))),
            patch.object(torch.Tensor, "cuda", lambda self: self),
            patch.object(stability, "MAX_PARTS", 24, create=True),
        ):
            for noise in (None, 0.2):
                with self.subTest(noise=noise):
                    _, legacy = stability.part_intersect(
                        image, semantic, parts, model, {1: "aeroplane"}, "sample", 0.8, noise=noise,
                    )
                    _, foreground = stability.part_intersect(
                        image, semantic, parts, model, {1: "aeroplane"}, "sample", 0.8,
                        noise=noise, exclude_background_centroid=True,
                    )
                    self.assertEqual(legacy[0][1], 1)
                    self.assertEqual(foreground[0][1], 0)

    def test_cli_applies_flag_to_both_images_and_separates_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            semantic_dir = root / "annotations" / "val"
            part_dir = root / "annotations_PIDS" / "val"
            for directory in (semantic_dir, part_dir):
                directory.mkdir(parents=True)
            np.save(semantic_dir / "sample.npy", [[1]])
            np.save(part_dir / "sample.npy", [[1]])
            image_path = root / "sample.png"
            Image.new("RGB", (1, 1)).save(image_path)

            dataset = MagicMock()
            dataset.__len__.return_value = 1
            dataset.img_ids = ["sample"]
            dataset.annotations_dir = str(semantic_dir)
            dataset.get_img_path.return_value = str(image_path)
            observation = [0] * (stability.MAX_PARTS_PASCAL + 1) + [0, "aeroplane", "sample"]
            checkpoint = root / "model" / "checkpoints" / "nopush_best.pth"
            argv = [
                "model", "nopush", "pascal", "--quantile", "0.8", "--std-dev", "0.2",
                "--checkpoint-path", str(checkpoint),
            ]
            with (
                patch.dict(os.environ, {"RESULTS_DIR": str(root)}),
                patch.object(stability, "load_model_checkpoint") as load_model,
                patch.object(stability, "PatchClassificationDataset", return_value=dataset),
                patch.object(stability, "part_intersect", return_value=({}, [observation])) as intersect,
            ):
                argh.dispatch_command(stability.run_stability, argv=argv)
                output = root / "model" / "metrics" / "nopush_best"
                legacy_files = {path.name: path.read_bytes() for path in output.iterdir()}

                argh.dispatch_command(stability.run_stability, argv=argv + ["--exclude-background-centroid"])
                load_model.assert_called_with(str(checkpoint))
                self.assertEqual(intersect.call_count, 4)
                self.assertEqual(
                    [call.kwargs["exclude_background_centroid"] for call in intersect.call_args_list],
                    [False, False, True, True],
                )
                self.assertEqual(intersect.call_args.args[8], 0.2)
                for name, contents in legacy_files.items():
                    self.assertEqual((output / name).read_bytes(), contents)
                    self.assertTrue((output / "foreground_centroids" / name).is_file())
                self.assertEqual(
                    (output / "foreground_centroids" / "stability_score_qt_0.8.txt").read_text(), "1.0",
                )


if __name__ == "__main__":
    unittest.main()
