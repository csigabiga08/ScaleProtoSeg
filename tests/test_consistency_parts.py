"""Check runtime part selection without requiring GPU inference."""

import json
import os
import csv
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


def latest_record(root):
    bundles = sorted((Path(root) / "measurements").iterdir(), key=lambda path: path.stat().st_mtime_ns)
    return bundles[-1], json.loads((bundles[-1] / "result.json").read_text())


def prepare_run(root):
    checkpoint = root / "model" / "checkpoints" / "push_last.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture checkpoint bytes")
    catalog_root = root / "catalog"
    catalog_root.mkdir()
    return catalog_root


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

    def test_pascal_observations_use_one_based_evaluator_class_ids(self):
        output = MagicMock()
        max_parts = consistency.MAX_PARTS_PASCAL
        row = [np.nan] * (max_parts + 1) + [7, "aeroplane", "sample"]
        consistency._add_observations(output, [row], {1: "aeroplane"}, max_parts)
        observation = output.add_observation.call_args.kwargs
        self.assertEqual(observation["class_id"], 2)

        protocol = consistency._protocol(
            "consistency", "pascal", {1: "aeroplane"}, consistency.PASCAL_ID_MAPPING,
            max_parts, quantile=0.8, threshold=0.8, official_parts_only=False,
            require_semantic_match=False, exclude_background_centroid=False,
        )
        self.assertEqual(protocol["classes"]["id_convention"]["model_class_index"], "evaluation_class_id - 1")
        self.assertEqual(protocol["classes"]["evaluation_class_id_to_label"], {"2": "aeroplane"})
        self.assertEqual(protocol["classes"]["background_evaluation_class_id"], 1)
        self.assertIn(1, protocol["classes"]["excluded_evaluation_class_ids"])

    def test_evaluator_switches_modes_without_changing_cache(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            catalog_root = prepare_run(root)
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
                patch.dict(os.environ, {
                    "RESULTS_DIR": str(root),
                    "MEASUREMENT_CATALOG_ROOT": str(catalog_root),
                }),
                patch.object(consistency, "load_model_checkpoint"),
                patch.object(consistency, "PatchClassificationDataset", return_value=dataset),
                patch.object(consistency, "part_intersect", return_value=({}, [observation])) as intersect,
            ):
                # The default mode must work even without an SIDS cache.
                consistency.run_consistency("model", "push", "cityscapes")
                np.testing.assert_array_equal(intersect.call_args.args[2], original_parts)
                unfiltered_bundle, unfiltered = latest_record(catalog_root)
                self.assertEqual(unfiltered["status"], "complete")
                self.assertEqual(unfiltered["protocol"]["parts"]["annotation_provenance"], "unknown")
                self.assertEqual(
                    unfiltered["protocol"]["parts"]["class_dependent_part_definitions"]["person"],
                    {
                        "semantic_ids": [24],
                        "part_id_to_name": {"1": "torso", "2": "head", "3": "arm", "4": "leg"},
                    },
                )
                self.assertEqual(
                    unfiltered["protocol"]["parts"]["class_dependent_part_definitions"]["rider"],
                    {
                        "semantic_ids": [25],
                        "part_id_to_name": {"1": "torso", "2": "head", "3": "arm", "4": "leg"},
                    },
                )
                self.assertEqual(
                    unfiltered["protocol"]["parts"]["class_dependent_part_definitions"]["car"],
                    {
                        "semantic_ids": [26],
                        "part_id_to_name": {
                            "1": "window",
                            "2": "wheel",
                            "3": "light",
                            "4": "license plate",
                            "5": "chassis",
                        },
                    },
                )
                self.assertEqual(
                    unfiltered["protocol"]["parts"]["class_dependent_part_definitions"]["truck"]["semantic_ids"],
                    [27],
                )
                self.assertEqual(
                    unfiltered["protocol"]["parts"]["class_dependent_part_definitions"]["bus"]["semantic_ids"],
                    [28],
                )
                self.assertEqual(
                    unfiltered["protocol"]["parts"]["mapping_identity"]["identity_kind"],
                    "official Cityscapes filter implementation",
                )
                unfiltered_score = unfiltered["scores"]["consistency"]["value"]

                with self.assertRaises(FileNotFoundError):
                    consistency.run_consistency("model", "push", "cityscapes", official_parts_only=True)

                np.save(native_dir / "sample.npy", [[24, 24, 25, 26, 31]])
                consistency.run_consistency("model", "push", "cityscapes", official_parts_only=True)
                np.testing.assert_array_equal(intersect.call_args.args[2], [[4, -1, -1, 5, -1]])
                filtered_bundle, filtered = latest_record(catalog_root)
                self.assertEqual(filtered["status"], "complete")
                self.assertTrue(filtered["protocol"]["parts"]["official_parts_only"])
                self.assertEqual(filtered["scores"]["consistency"]["value"], unfiltered_score)
                self.assertNotEqual(filtered_bundle, unfiltered_bundle)
                self.assertEqual(part_path.read_bytes(), original_bytes)

    def test_official_flag_rejects_pascal_before_loading_model(self):
        with patch.object(consistency, "load_model_checkpoint") as load_model:
            with self.assertRaisesRegex(ValueError, "only for cityscapes"):
                consistency.run_consistency("model", "push", "pascal", official_parts_only=True)
            load_model.assert_not_called()

    def test_pascal_semantic_match_cli_preserves_legacy_results_and_cache(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            catalog_root = prepare_run(root)
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
                patch.dict(os.environ, {
                    "RESULTS_DIR": str(root),
                    "MEASUREMENT_CATALOG_ROOT": str(catalog_root),
                }),
                patch.object(consistency, "load_model_checkpoint"),
                patch.object(consistency, "PatchClassificationDataset", return_value=dataset),
                patch.object(consistency, "part_intersect", return_value=({}, [observation])) as intersect,
            ):
                # Legacy mode needs no panoptic semantic cache.
                consistency.run_consistency("model", "push", "pascal")
                np.testing.assert_array_equal(intersect.call_args.args[2], original_parts)
                legacy_bundle, legacy_record = latest_record(catalog_root)
                legacy_score = legacy_record["scores"]["consistency"]["value"]

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
                semantic_bundle, semantic_record = latest_record(catalog_root)
                self.assertEqual(semantic_record["status"], "complete")
                self.assertTrue(semantic_record["protocol"]["parts"]["require_semantic_match"])
                self.assertEqual(semantic_record["scores"]["consistency"]["value"], legacy_score)
                self.assertNotEqual(semantic_bundle, legacy_bundle)
                self.assertEqual(part_path.read_bytes(), original_bytes)

                argh.dispatch_command(
                    consistency.run_consistency,
                    argv=["model", "push", "pascal", "--require-semantic-match", "--exclude-background-centroid"],
                )
                self.assertTrue(intersect.call_args.kwargs["exclude_background_centroid"])
                _, foreground_record = latest_record(catalog_root)
                self.assertTrue(foreground_record["protocol"]["centroids"]["exclude_background_centroid"])

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

    def test_population_tracks_missing_annotations_and_empty_observations(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            catalog_root = prepare_run(root)
            semantic_dir = root / "annotations" / "val"
            part_dir = root / "annotations_PIDS" / "val"
            semantic_dir.mkdir(parents=True)
            part_dir.mkdir(parents=True)
            np.save(semantic_dir / "empty.npy", [[1]])
            np.save(part_dir / "empty.npy", [[1]])
            image_path = root / "empty.png"
            Image.new("RGB", (1, 1)).save(image_path)
            dataset = MagicMock()
            dataset.__len__.return_value = 2
            dataset.img_ids = ["missing", "empty"]
            dataset.annotations_dir = str(semantic_dir)
            dataset.get_img_path.return_value = str(image_path)
            dataset.convert_targets.side_effect = lambda value: value
            with (
                patch.dict(os.environ, {
                    "RESULTS_DIR": str(root),
                    "MEASUREMENT_CATALOG_ROOT": str(catalog_root),
                }),
                patch.object(consistency, "load_model_checkpoint"),
                patch.object(consistency, "PatchClassificationDataset", return_value=dataset),
                patch.object(consistency, "part_intersect", return_value=({}, [])),
            ):
                consistency.run_consistency("model", "push", "pascal")

            bundle, record = latest_record(catalog_root)
            self.assertEqual(record["status"], "complete")
            self.assertIsNone(record["scores"]["consistency"]["value"])
            self.assertEqual(
                record["scores"]["consistency"]["reason"],
                "no eligible class/prototype observations",
            )
            self.assertEqual(record["population"]["excluded"], 1)
            self.assertEqual(record["population"]["processed_no_observations"], 1)
            with (bundle / "samples.csv").open(newline="") as stream:
                samples = list(csv.DictReader(stream))
            self.assertEqual(
                [(row["sample_id"], row["outcome"]) for row in samples],
                [("missing", "excluded"), ("empty", "processed_no_observations")],
            )

    def test_processing_error_publishes_failed_and_unattempted_candidates(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            catalog_root = prepare_run(root)
            semantic_dir = root / "annotations" / "val"
            part_dir = root / "annotations_PIDS" / "val"
            semantic_dir.mkdir(parents=True)
            part_dir.mkdir(parents=True)
            (part_dir / "bad.npy").write_bytes(b"not a numpy array")
            dataset = MagicMock()
            dataset.__len__.return_value = 2
            dataset.img_ids = ["bad", "later"]
            dataset.annotations_dir = str(semantic_dir)
            dataset.get_img_path.return_value = str(root / "unused.png")
            with (
                patch.dict(os.environ, {
                    "RESULTS_DIR": str(root),
                    "MEASUREMENT_CATALOG_ROOT": str(catalog_root),
                }),
                patch.object(consistency, "load_model_checkpoint"),
                patch.object(consistency, "PatchClassificationDataset", return_value=dataset),
            ):
                with self.assertRaises(ValueError):
                    consistency.run_consistency("model", "push", "pascal")

            bundle, record = latest_record(catalog_root)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["population"]["failed"], 1)
            self.assertEqual(record["population"]["not_attempted"], 1)
            with (bundle / "samples.csv").open(newline="") as stream:
                samples = list(csv.DictReader(stream))
            self.assertEqual([row["outcome"] for row in samples], ["failed", "not_attempted"])
            self.assertIn("ValueError", samples[0]["reason"])


if __name__ == "__main__":
    unittest.main()
