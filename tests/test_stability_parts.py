"""Check stability centroid selection and CLI routing without a GPU."""

import os
import json
import csv
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
import numpy as np
import torch
from PIL import Image
from pydantic import ValidationError

from segmentation.analysis.metrics import stability
from segmentation.analysis.metrics.protocol import StabilityRunInput


def latest_record(root):
    bundles = sorted((Path(root) / "measurements").iterdir(), key=lambda path: path.stat().st_mtime_ns)
    return bundles[-1], json.loads((bundles[-1] / "result.json").read_text())


class StabilityPartsTests(unittest.TestCase):
    def test_stability_serializer_preserves_existing_protocol_and_columns(self):
        run = StabilityRunInput(
            model_name="model",
            training_phase="push",
            data_type="pascal",
            quantile=0.7,
            std_dev=0.35,
            seed=41,
            official_parts_only=False,
            require_semantic_match=True,
            exclude_background_centroid=True,
        )

        protocol = run.to_protocol(
            semantic_id_to_evaluation_id={1: 2, 2: 3},
            evaluation_class_id_to_label={"2": "aeroplane", "3": "bicycle"},
            valid_part_ids=[1, 2, 3],
            observation_columns=[
                "part_0", "part_1", "part_2", "part_3",
                "noisy_part_0", "noisy_part_1", "noisy_part_2", "noisy_part_3",
            ],
        )

        self.assertEqual(protocol, {
            "metric": "stability",
            "quantile": 0.7,
            "parts": {
                "valid_part_ids": [1, 2, 3],
                "official_parts_only": False,
                "require_semantic_match": True,
                "exclude_background_centroid": True,
            },
            "classes": {
                "semantic_id_to_evaluation_id": {1: 2, 2: 3},
                "evaluation_class_id_to_label": {"2": "aeroplane", "3": "bicycle"},
                "background_evaluation_class_id": 1,
            },
            "stability": {
                "standard_deviation": 0.35,
                "seed": 41,
                "aggregation": (
                    "unweighted mean of exact all-part equality per paired "
                    "sample/class/prototype row"
                ),
            },
            "observation_columns": [
                "part_0", "part_1", "part_2", "part_3",
                "noisy_part_0", "noisy_part_1", "noisy_part_2", "noisy_part_3",
            ],
        })

    def test_stability_hydra_config_hands_validated_input_to_runner(self):
        config_dir = Path(__file__).resolve().parents[1] / "config" / "hydra" / "metrics"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            config = compose(
                config_name="stability",
                overrides=[
                    "model_name=test-model",
                    "training_phase=push",
                    "data_type=pascal",
                    "std_dev=0.35",
                    "seed=41",
                ],
            )
            with patch.object(stability, "run_stability") as runner:
                stability._run_config(config)

        run_input = runner.call_args.args[0]
        self.assertIsInstance(run_input, StabilityRunInput)
        self.assertEqual(run_input.model_name, "test-model")
        self.assertEqual(run_input.training_phase, "push")
        self.assertEqual(run_input.data_type, "pascal")
        self.assertEqual(run_input.std_dev, 0.35)
        self.assertEqual(run_input.seed, 41)

    def test_stability_hydra_config_defaults_seed_and_rejects_unknown_fields(self):
        config_dir = Path(__file__).resolve().parents[1] / "config" / "hydra" / "metrics"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            config = compose(
                config_name="stability",
                overrides=[
                    "model_name=test-model",
                    "training_phase=push",
                    "data_type=cityscapes",
                ],
            )
            with patch.object(stability, "run_stability") as runner:
                stability._run_config(config)
            self.assertEqual(runner.call_args.args[0].seed, 0)

            with self.assertRaises(ConfigCompositionException):
                compose(
                    config_name="stability",
                    overrides=[
                        "model_name=test-model",
                        "training_phase=push",
                        "data_type=cityscapes",
                        "unexpected_field=true",
                    ],
                )

            config_with_unknown_field = compose(
                config_name="stability",
                overrides=[
                    "model_name=test-model",
                    "training_phase=push",
                    "data_type=cityscapes",
                    "+unexpected_field=true",
                ],
            )
            with self.assertRaises(ValidationError):
                stability._run_config(config_with_unknown_field)

    def test_stability_observations_use_one_based_evaluator_class_ids(self):
        import pandas as pd

        from segmentation.analysis.metrics import stability

        output = MagicMock()
        max_parts = stability.MAX_PARTS_PASCAL
        values = {f"part_{part_id}": 0 for part_id in range(max_parts + 1)}
        values.update({f"noisy_part_{part_id}": 0 for part_id in range(max_parts + 1)})
        values.update(proto_id=7, **{"class": "aeroplane", "img_id": "sample"})
        paired = pd.DataFrame([values])
        stability._add_observations(output, paired, {1: "aeroplane"}, max_parts)
        observation = output.add_observation.call_args.kwargs
        self.assertEqual(observation["class_id"], 2)

    def test_runner_accepts_only_validated_stability_input(self):
        self.assertNotIn("output_dir", inspect.signature(stability.run_stability).parameters)
        self.assertEqual(
            list(inspect.signature(stability.run_stability).parameters),
            ["run_input"],
        )
        with self.assertRaises(ValidationError):
            StabilityRunInput(
                model_name="model",
                training_phase="push",
                data_type="pascal",
                output_dir="ignored",
            )

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

    def test_pairing_rejects_duplicate_and_unmatched_rows(self):
        row = [0] * (stability.MAX_PARTS_PASCAL + 1) + [2, "aeroplane", "sample"]
        with self.assertRaisesRegex(ValueError, "unique"):
            stability._paired_rows([row, row], [row], stability.MAX_PARTS_PASCAL)
        unmatched = list(row)
        unmatched[-3] = 3
        with self.assertRaisesRegex(ValueError, "different pairing keys"):
            stability._paired_rows([row], [unmatched], stability.MAX_PARTS_PASCAL)

    def test_runner_applies_flag_to_both_images_and_separates_results(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            checkpoint = root / "model" / "checkpoints" / "nopush_best.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fixture checkpoint bytes")
            catalog_root = root / "catalog"
            catalog_root.mkdir()
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
            run_input = StabilityRunInput(
                model_name="model",
                training_phase="nopush",
                data_type="pascal",
                checkpoint_path=str(checkpoint),
            )
            with (
                patch.dict(os.environ, {
                    "RESULTS_DIR": str(root),
                    "MEASUREMENT_CATALOG_ROOT": str(catalog_root),
                }),
                patch.object(stability, "load_model_checkpoint") as load_model,
                patch.object(stability, "PatchClassificationDataset", return_value=dataset),
                patch.object(stability, "part_intersect", return_value=({}, [observation])) as intersect,
            ):
                stability.run_stability(run_input)
                legacy_bundle, legacy_record = latest_record(catalog_root)
                self.assertEqual(legacy_record["status"], "complete")
                self.assertEqual(legacy_record["scores"]["stability"]["value"], 1.0)

                stability.run_stability(
                    run_input.model_copy(update={"exclude_background_centroid": True})
                )
                load_model.assert_called_with(str(checkpoint))
                self.assertEqual(intersect.call_count, 4)
                self.assertEqual(
                    [call.kwargs["exclude_background_centroid"] for call in intersect.call_args_list],
                    [False, False, True, True],
                )
                self.assertEqual(intersect.call_args.args[8], 0.2                )
                foreground_bundle, foreground_record = latest_record(catalog_root)
                self.assertNotEqual(foreground_bundle, legacy_bundle)
                self.assertTrue(foreground_record["protocol"]["parts"]["exclude_background_centroid"])
                self.assertEqual(foreground_record["scores"]["stability"]["value"], 1.0)
                with (foreground_bundle / "observations.csv").open(newline="") as stream:
                    columns = next(csv.reader(stream))
                self.assertIn("part_1", columns)
                self.assertIn("noisy_part_1", columns)
                self.assertFalse(any(column.startswith("eq_part_") for column in columns))
                self.assertNotIn("is_stable", columns)


if __name__ == "__main__":
    unittest.main()
