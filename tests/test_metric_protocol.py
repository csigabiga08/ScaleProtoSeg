import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from segmentation.analysis.metrics.protocol import (
    ConsistencyRunInput,
    StabilityRunInput,
)
from segmentation.data.pascal_parts import left_right_part_groups


class MetricRunInputTests(unittest.TestCase):
    def test_raw_checkpoint_arguments_are_rejected(self):
        for input_type in (ConsistencyRunInput, StabilityRunInput):
            with self.assertRaises(ValidationError):
                input_type(model="registered", data_type="pascal", checkpoint="/raw.pth")

    def test_consistency_defaults_and_protocol_settings(self):
        run = ConsistencyRunInput(
            model="model",
            data_type="pascal",
        )

        self.assertEqual(run.quantile, 0.8)
        self.assertEqual(run.threshold, 0.8)
        self.assertFalse(run.official_parts_only)
        self.assertFalse(run.require_semantic_match)
        self.assertFalse(run.exclude_background_centroid)
        self.assertFalse(run.merge_left_right_parts)
        protocol = run.to_protocol(
            semantic_id_to_evaluation_id={1: 2},
            evaluation_class_id_to_label={"2": "cat"},
            valid_part_ids=[9, 3],
            observation_columns=["part_0", "part_3"],
            excluded_evaluation_class_ids=[1, 5, 10, 12, 19],
        )
        self.assertEqual(
            protocol,
            {
                "metric": "consistency",
                "quantile": 0.8,
                "parts": {
                    "valid_part_ids": [9, 3],
                    "official_parts_only": False,
                    "require_semantic_match": False,
                    "exclude_background_centroid": False,
                    "merge_left_right_parts": False,
                },
                "classes": {
                    "semantic_id_to_evaluation_id": {1: 2},
                    "evaluation_class_id_to_label": {"2": "cat"},
                    "background_evaluation_class_id": 1,
                    "excluded_evaluation_class_ids": [1, 5, 10, 12, 19],
                },
                "consistency": {
                    "threshold": 0.8,
                    "aggregation": "unweighted mean over class/prototype rows",
                },
                "observation_columns": ["part_0", "part_3"],
            },
        )

    def test_consistency_requires_explicit_excluded_class_ids(self):
        run = ConsistencyRunInput(
            model="model",
            data_type="pascal",
        )

        with self.assertRaises(TypeError):
            run.to_protocol(
                semantic_id_to_evaluation_id={1: 2},
                evaluation_class_id_to_label={"2": "cat"},
                valid_part_ids=[9, 3],
                observation_columns=["part_0", "part_3"],
            )

    def test_stability_defaults_and_protocol_settings(self):
        run = StabilityRunInput(
            model="model",
            data_type="cityscapes",
        )

        self.assertEqual(run.quantile, 0.8)
        self.assertEqual(run.std_dev, 0.2)
        self.assertEqual(run.seed, 0)
        self.assertFalse(run.official_parts_only)
        self.assertFalse(run.require_semantic_match)
        self.assertFalse(run.exclude_background_centroid)
        self.assertFalse(run.merge_left_right_parts)
        protocol = run.to_protocol(
            semantic_id_to_evaluation_id={7: 4},
            evaluation_class_id_to_label={"4": "car"},
            valid_part_ids=[5, 2],
            observation_columns=["part_0", "noisy_part_0"],
        )
        self.assertEqual(protocol["metric"], "stability")
        self.assertEqual(protocol["quantile"], 0.8)
        self.assertEqual(protocol["parts"]["valid_part_ids"], [5, 2])
        self.assertFalse(protocol["parts"]["merge_left_right_parts"])
        self.assertEqual(protocol["classes"]["background_evaluation_class_id"], None)
        self.assertNotIn("excluded_evaluation_class_ids", protocol["classes"])
        self.assertEqual(
            protocol["stability"],
            {
                "standard_deviation": 0.2,
                "seed": 0,
                "aggregation": ("unweighted mean of exact all-part equality per paired " "sample/class/prototype row"),
            },
        )
        self.assertEqual(protocol["observation_columns"], ["part_0", "noisy_part_0"])

    def test_models_reject_unknown_keys(self):
        for model in (ConsistencyRunInput, StabilityRunInput):
            with self.subTest(model=model.__name__):
                with self.assertRaises(ValidationError):
                    model(
                        model="model",
                        data_type="cityscapes",
                        unexpected=True,
                    )

    def test_models_reject_dataset_incompatible_flags(self):
        common = {
            "model": "model",
        }
        for model in (ConsistencyRunInput, StabilityRunInput):
            with self.subTest(model=model.__name__, flag="official_parts_only"):
                with self.assertRaises(ValidationError):
                    model(
                        **common,
                        data_type="pascal",
                        official_parts_only=True,
                    )
            with self.subTest(model=model.__name__, flag="require_semantic_match"):
                with self.assertRaises(ValidationError):
                    model(
                        **common,
                        data_type="cityscapes",
                        require_semantic_match=True,
                    )
            with self.subTest(model=model.__name__, flag="merge_left_right_parts"):
                with self.assertRaises(ValidationError):
                    model(
                        **common,
                        data_type="cityscapes",
                        merge_left_right_parts=True,
                    )

    def test_merged_part_groups_are_recorded_in_protocol(self):
        run = ConsistencyRunInput(
            model="model",
            data_type="pascal",
            merge_left_right_parts=True,
        )
        protocol = run.to_protocol(
            semantic_id_to_evaluation_id={1: 2},
            evaluation_class_id_to_label={"2": "person"},
            valid_part_ids=[1, 2],
            observation_columns=["part_0", "part_1"],
            excluded_evaluation_class_ids=[],
            left_right_part_groups=left_right_part_groups(),
        )

        self.assertTrue(protocol["parts"]["merge_left_right_parts"])
        self.assertEqual(
            protocol["parts"]["left_right_part_groups"]["person"][3],
            {
                "canonical_part_id": 13,
                "source_part_ids": [13, 16],
                "source_labels": ["llarm", "rlarm"],
            },
        )
        with self.assertRaisesRegex(ValueError, "canonical group definitions"):
            run.to_protocol(
                semantic_id_to_evaluation_id={1: 2},
                evaluation_class_id_to_label={"2": "person"},
                valid_part_ids=[1, 2],
                observation_columns=["part_0", "part_1"],
                excluded_evaluation_class_ids=[],
            )

    def test_protocol_excludes_operational_model_and_checkpoint_values(self):
        run = ConsistencyRunInput(
            model="private-model-name",
            data_type="cityscapes",
            grouping_model="private-group",
        )

        protocol = run.to_protocol(
            semantic_id_to_evaluation_id={1: 2},
            evaluation_class_id_to_label={"2": "car"},
            valid_part_ids=[4, 1],
            observation_columns=["part_1"],
            excluded_evaluation_class_ids=[],
        )

        self.assertEqual(protocol["parts"]["valid_part_ids"], [4, 1])
        self.assertFalse(protocol["parts"]["merge_left_right_parts"])
        self.assertEqual(protocol["classes"]["excluded_evaluation_class_ids"], [])
        serialized = repr(protocol)
        for operational_value in (
            "private-model-name",
            "/models/private/checkpoint.pth",
            "private-group",
            "/models/private/group.pth",
        ):
            self.assertNotIn(operational_value, serialized)
        self.assertEqual(
            set(protocol),
            {"metric", "quantile", "parts", "classes", "consistency", "observation_columns"},
        )


if __name__ == "__main__":
    unittest.main()
