"""Sparse part-vector averages, filters, and prototype correlations without model loading."""

import unittest

import numpy as np

from segmentation.analysis.metrics.part_activations import (
    consistent_prototypes,
    mean_part_activations,
    prototype_part_correlation,
)


def record(dataset="pascal", labels=None, **parts):
    return {
        "dataset": {"name": dataset},
        "protocol": {
            "observation_columns": ["part_0", "part_1", "part_2", "part_3", "noisy_part_1"],
            "classes": {"evaluation_class_id_to_label": labels or {"6": "bottle", "2": "aeroplane"}},
            "consistency": {"threshold": 0.8},
            "parts": parts,
        },
    }


def row(prototype_id, values, class_id=6):
    return {
        "class_id": str(class_id),
        "prototype_id": str(prototype_id),
        "part_0": "1",
        "noisy_part_1": "1",
        **{f"part_{index}": value for index, value in enumerate(values, 1)},
    }


class PartActivationsTest(unittest.TestCase):
    def test_sparse_means_ignore_blanks_but_count_zero_and_use_strict_threshold(self):
        means = mean_part_activations(
            [row(3, [1, "", None]), row(3, [0, 0, None]), row(3, [1, 1, None])],
            ["part_1", "part_2", "part_3"],
        )
        self.assertAlmostEqual(means.iloc[0]["part_1"], 2 / 3)
        self.assertEqual(means.iloc[0]["part_2"], 0.5)
        self.assertTrue(np.isnan(means.iloc[0]["part_3"]))
        self.assertFalse(consistent_prototypes(means, 2 / 3).iloc[0])
        self.assertTrue(consistent_prototypes(means, 0.6).iloc[0])

    def test_pearson_uses_jointly_observed_parts_and_class_coordinates(self):
        rows = [
            row(1, ["1", "0", ""]),
            row(2, ["0", "1", "0.5"]),
            row(3, ["1", "", ""]),
            row(4, ["0.5", "0.5", "0.5"]),
            row(5, ["1", "0", ""], class_id=2),
        ]
        means, correlation, shared = prototype_part_correlation(record(), rows)
        self.assertEqual(len(means.columns), 6)
        self.assertAlmostEqual(correlation.loc[1, 1], 1)
        self.assertAlmostEqual(correlation.loc[1, 2], -1)
        self.assertEqual(shared.loc[1, 2], 2)
        self.assertEqual(shared.loc[1, 5], 0)
        self.assertTrue(np.isnan(correlation.loc[1, 5]))
        self.assertTrue(np.isnan(correlation.loc[1, 3]))
        self.assertTrue(correlation.loc[4].isna().all())
        self.assertNotIn((6, "part_0"), means.columns)
        self.assertNotIn((6, "noisy_part_1"), means.columns)

    def test_semantic_filter_drops_foreign_ids_not_zeros_and_consistency_uses_full_means(self):
        rows = [row(1, [0.2, 0.7, 1]), row(2, [0.8, 0.3, 0.4]), row(3, [0, 0, 0.8])]
        _, all_parts, _ = prototype_part_correlation(record(), rows)
        means, own_parts, shared = prototype_part_correlation(record(), rows, semantic_parts_only=True)
        self.assertFalse(np.isclose(all_parts.loc[1, 2], -1))
        self.assertAlmostEqual(own_parts.loc[1, 2], -1)
        self.assertTrue(means[(6, "part_3")].isna().all())
        self.assertEqual(shared.loc[1, 2], 2)
        means, _, _ = prototype_part_correlation(record(), rows, consistent_only=True, semantic_parts_only=True)
        self.assertEqual(list(means.index), [1])

    def test_cityscapes_taxonomy_uses_evaluation_labels_not_native_ids(self):
        metadata = record("cityscapes", {"12": "person", "14": "car", "1": "road"})
        metadata["protocol"]["observation_columns"] += ["part_4", "part_5"]
        rows = [
            {**row(1, [1, 0, 1], 12), "part_4": 0, "part_5": 1},
            {**row(2, [1, 0, 1], 14), "part_4": 0, "part_5": 1},
            {**row(3, [1, 0, 1], 1), "part_4": 0, "part_5": 1},
        ]
        means, _, shared = prototype_part_correlation(metadata, rows, semantic_parts_only=True)
        self.assertTrue(np.isnan(means.loc[1, (12, "part_5")]))
        self.assertEqual(shared.loc[1, 1], 4)
        self.assertEqual(shared.loc[2, 2], 5)
        self.assertEqual(shared.loc[3, 3], 0)

    def test_merged_parts_use_recorded_canonical_coordinates(self):
        metadata = record(
            labels={"2": "aeroplane"},
            merge_left_right_parts=True,
            left_right_part_groups={
                "aeroplane": [{"canonical_part_id": 3, "source_part_ids": [3, 4]}],
            },
        )
        metadata["protocol"]["observation_columns"] += ["part_4"]
        means, _, _ = prototype_part_correlation(
            metadata, [{**row(1, [1, 0, 1], 2), "part_4": 0}], semantic_parts_only=True
        )
        self.assertEqual(means.loc[1, (2, "part_3")], 1)
        self.assertTrue(np.isnan(means.loc[1, (2, "part_4")]))

    def test_empty_missing_and_filtered_out_observations(self):
        for rows, options in [
            ([], {}),
            ([row(1, ["", "", ""])], {"consistent_only": True}),
            ([row(1, [0, 0, 0])], {"consistent_only": True}),
        ]:
            means, correlation, shared = prototype_part_correlation(record(), rows, **options)
            self.assertTrue(means.empty)
            self.assertTrue(correlation.empty)
            self.assertTrue(shared.empty)
        _, correlation, shared = prototype_part_correlation(record(), [row(1, ["", "", ""])])
        self.assertTrue(np.isnan(correlation.loc[1, 1]))
        self.assertEqual(shared.loc[1, 1], 0)

    def test_unsupported_or_malformed_evidence_is_not_silently_reinterpreted(self):
        with self.assertRaisesRegex(ValueError, "exactly one semantic class"):
            prototype_part_correlation(record(), [row(1, [1, 0, 1]), row(1, [1, 0, 1], 2)])
        with self.assertRaisesRegex(ValueError, "finite or missing"):
            prototype_part_correlation(record(), [row(1, ["inf", 0, 1])])
        with self.assertRaisesRegex(ValueError, "No Pascal part taxonomy"):
            prototype_part_correlation(record(labels={"6": "unknown"}), [row(1, [1, 0, 1])], semantic_parts_only=True)
        metadata = record()
        del metadata["protocol"]["consistency"]
        with self.assertRaisesRegex(ValueError, "threshold is required"):
            prototype_part_correlation(metadata, [row(1, [1, 0, 1])], consistent_only=True)
        means, _, _ = prototype_part_correlation(metadata, [row(1, [1, 0, 1])], consistent_only=True, threshold=0.8)
        self.assertEqual(list(means.index), [1])


if __name__ == "__main__":
    unittest.main()
