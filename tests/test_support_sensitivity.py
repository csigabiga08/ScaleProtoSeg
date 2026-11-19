"""Cohort completeness, matched populations and seed-level aggregation."""

import copy
import unittest
from itertools import product
from statistics import stdev

from segmentation.analysis.support_sensitivity import (
    COHORT,
    FLAGS,
    QUANTILES,
    SETTINGS,
    aggregate,
    evidence_from_rows,
    freeze_cohort,
    index_rows,
    paired_statistics,
    statistics,
)


def cohort_records():
    records = []
    for model, quantile, flags in product(COHORT, QUANTILES, SETTINGS):
        parameters = dict(zip(FLAGS, flags))
        parameters.update(quantile=quantile, threshold=0.8, quantile_scope="class", official_parts_only=False)
        records.append(
            {
                "measurement_id": f"{model}-{quantile}-{SETTINGS[flags]}",
                "model": {
                    "name": model,
                    "architecture": COHORT[model],
                    "inputs": {"prototype_checkpoint": {"sha256": model}},
                },
                "dataset": {"name": "pascal", "split": "val"},
                "metric": "consistency",
                "status": "complete",
                "parameters": parameters,
                "protocol": {
                    "quantile": quantile,
                    "consistency": {"threshold": 0.8},
                    "centroids": {"exclude_background_centroid": flags[1]},
                    "parts": dict(zip(FLAGS, flags)),
                    "prototype_selection": {},
                    "classes": {"evaluation_class_id_to_label": {"0": "cat"}},
                },
                "legacy_metadata": {
                    "description": {
                        "parameters": dict(parameters),
                        "validation_images": 10,
                        "image_order_sha256": "same",
                    }
                },
                "execution": {"command": []},
            }
        )
    return records


class SupportSensitivityTest(unittest.TestCase):
    def test_cohort_rejects_missing_duplicate_and_incompatible_cells(self):
        records = cohort_records()
        self.assertEqual(len(freeze_cohort(records)), 72)
        for modified in (records[:-1], records + records[:1]):
            with self.assertRaisesRegex(ValueError, "Incomplete|Duplicate"):
                freeze_cohort(modified)
        for change in (
            lambda record: record["parameters"].update(quantile_scope="image"),
            lambda record: record["protocol"]["parts"].update(merge_left_right_parts=True),
            lambda record: record["legacy_metadata"]["description"]["parameters"].update(quantile_scope="image"),
            lambda record: record["model"]["inputs"]["prototype_checkpoint"].update(sha256="other"),
            lambda record: record["legacy_metadata"]["description"].update(image_order_sha256="other"),
        ):
            modified = copy.deepcopy(records)
            change(modified[0])
            with self.assertRaises(ValueError):
                freeze_cohort(modified)

    def test_matched_rows_recompute_support_without_dropping_unsupported_prototypes(self):
        rows = [
            {"sample_id": "a", "class_id": 0, "prototype_id": 1, "part_1": "1", "part_2": "1"},
            {"sample_id": "b", "class_id": 0, "prototype_id": 1, "part_1": "", "part_2": "0"},
            {"sample_id": "c", "class_id": 0, "prototype_id": 2, "part_1": "", "part_2": ""},
        ]
        native = evidence_from_rows(rows, [1, 2])
        self.assertEqual(statistics(native, 1)["value"], 0.5)
        self.assertEqual(statistics(native, 2)["value"], 0)
        matched = evidence_from_rows(rows[:1], [1, 2], population=native)
        self.assertEqual(statistics(matched, 1)["denominator"], 2)
        self.assertEqual(statistics(matched, 1)["evidence_coverage"], 0.5)
        self.assertEqual(matched[0, 1]["qualifying_support"], 1)
        with self.assertRaisesRegex(ValueError, "Repeated"):
            index_rows(rows + rows[:1])

    def test_native_population_changes_are_not_counted_as_consistency_transitions(self):
        yes = {"qualifying_support": 10, "maximum_support": 10}
        no = {"qualifying_support": 0, "maximum_support": 10}
        left = {(0, 1): yes, (0, 2): no, (0, 3): yes}
        right = {(0, 1): no, (0, 2): yes}
        result = paired_statistics(left, right, 10)
        self.assertEqual((result["lost_consistency"], result["gained_consistency"]), (1, 1))
        self.assertEqual((result["left_only_prototypes"], result["right_only_prototypes"]), (1, 0))
        self.assertAlmostEqual(result["delta"], 0.5 - 2 / 3)
        common = {key: left[key] for key in right}
        self.assertEqual(paired_statistics(common, right, 10)["delta"], 0)

    def test_quantiles_are_averaged_within_seed_before_sample_sd(self):
        rows = [
            {"architecture": "model", "model": seed, "quantile": q, "value": value + offset, "denominator": 10**i}
            for i, (seed, value) in enumerate((("a", 0.1), ("b", 0.3), ("c", 0.5)))
            for q, offset in zip(QUANTILES, (-0.1, 0, 0.1))
        ]
        seeds, summary = aggregate(rows, ("architecture",), ("value",))
        self.assertEqual(len(seeds), 3)
        self.assertAlmostEqual(summary[0]["value_mean"], 0.3)
        self.assertAlmostEqual(summary[0]["value_sd"], stdev((0.1, 0.3, 0.5)))
        self.assertEqual(summary[0]["seeds"], 3)
        with self.assertRaisesRegex(ValueError, "quantile"):
            aggregate(rows[:-1], ("architecture",), ("value",))
        with self.assertRaisesRegex(ValueError, "three"):
            aggregate(rows[:6], ("architecture",), ("value",))


if __name__ == "__main__":
    unittest.main()
