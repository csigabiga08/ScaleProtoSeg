"""Availability masks, profile/event counterexamples and paired study exclusions."""

import copy
import unittest

from test_support_sensitivity import cohort_records

from segmentation.analysis.event_overlap import (
    analyze_measurement,
    frozen_cohort,
    summarize,
)
from segmentation.analysis.metrics.part_events import compare_events, event_profiles


def record(parts=3):
    return {
        "protocol": {
            "observation_columns": ["part_0", *[f"part_{p}" for p in range(1, parts + 1)]],
            "parts": {"valid_part_ids": list(range(parts + 1))},
            "classes": {"evaluation_class_id_to_label": {"1": "cat", "2": "dog"}},
        }
    }


def row(image, pid, values, cid=1):
    return {
        "sample_id": image,
        "class_id": cid,
        "prototype_id": pid,
        "part_0": 1,
        **{f"part_{part}": value for part, value in enumerate(values, 1)},
    }


class PartEventsTest(unittest.TestCase):
    def test_identical_mean_profiles_can_have_disjoint_positive_events(self):
        rows = [
            row("a", 1, [1, 0, 0]),
            row("b", 1, [0, 0, 0]),
            row("a", 2, [0, 0, 0]),
            row("b", 2, [1, 0, 0]),
        ]
        profiles = event_profiles(record(), rows)
        result = compare_events(profiles[1, 1], profiles[1, 2])
        self.assertAlmostEqual(result["rho"], 1)
        self.assertAlmostEqual(result["matched_rho"], 1)
        self.assertEqual((result["intersection"], result["union"], result["jaccard"]), (0, 2, 0))
        self.assertEqual(result["profile_mad"], 0)
        self.assertEqual(result["availability_dice"], 1)

    def test_missing_is_not_a_miss_and_image_masks_are_intersected(self):
        rows = [
            row("a", 1, [1, 0, ""]),
            row("b", 1, [0, 1, 1]),
            row("a", 2, [1, "", 1]),
            row("b", 2, ["", 0, None]),
            row("c", 2, [1, 1, 0]),
        ]
        profiles = event_profiles(record(), rows)
        result = compare_events(profiles[1, 1], profiles[1, 2])
        self.assertEqual(result["shared_parts"], 3)
        self.assertEqual(result["included_parts"], 2)
        self.assertEqual(result["joint_event_cells"], 2)
        self.assertEqual((result["left_positive_events"], result["right_positive_events"]), (2, 1))
        self.assertEqual((result["intersection"], result["union"]), (1, 2))
        self.assertEqual(result["jaccard"], 0.5)
        self.assertIsNone(result["rho"])
        self.assertIsNone(result["matched_rho"])
        filtered = compare_events(profiles[1, 1], profiles[1, 2], minimum_support=2)
        self.assertFalse(filtered["eligible"])
        self.assertIsNone(filtered["jaccard"])

    def test_two_part_degeneracy_constants_and_empty_positive_union(self):
        profiles = event_profiles(record(2), [row("a", 1, [1, 0]), row("a", 2, [0, 1]), row("a", 3, [0, 0])])
        two = compare_events(profiles[1, 1], profiles[1, 2])
        self.assertEqual(two["rho"], -1)
        self.assertFalse(compare_events(profiles[1, 1], profiles[1, 2], minimum_parts=5)["eligible"])
        constant = compare_events(profiles[1, 1], profiles[1, 3])
        self.assertIsNone(constant["rho"])
        empty = compare_events(profiles[1, 3], profiles[1, 3])
        self.assertIsNone(empty["jaccard"])
        self.assertEqual(empty["union"], 0)

    def test_joint_support_filter_recomputes_profiles_and_never_uses_background(self):
        rows = [
            row(str(image), pid, [1, 0, int(pid == 1) if image == 0 else ""]) for image in range(10) for pid in (1, 2)
        ]
        profiles = event_profiles(record(), rows)
        unrestricted = compare_events(profiles[1, 1], profiles[1, 2])
        self.assertAlmostEqual(unrestricted["rho"], 0.5)
        result = compare_events(profiles[1, 1], profiles[1, 2], minimum_support=10, minimum_parts=2)
        self.assertEqual(result["included_parts"], 2)
        self.assertEqual(result["minimum_joint_support"], 10)
        self.assertEqual(result["maximum_joint_support"], 10)
        self.assertAlmostEqual(result["rho"], 1)
        self.assertEqual(result["jaccard"], 1)

    def test_individual_support_does_not_substitute_for_joint_image_support(self):
        rows = [row(f"{pid}-{image}", pid, [1, 0, 0]) for pid in (1, 2) for image in range(10)]
        profiles = event_profiles(record(), rows)
        result = compare_events(profiles[1, 1], profiles[1, 2], minimum_support=10)
        self.assertEqual(result["shared_parts"], 3)
        self.assertEqual(result["included_parts"], 0)
        self.assertFalse(result["eligible"])
        self.assertIsNone(result["rho"])
        self.assertIsNone(result["jaccard"])

    def test_singleton_qualifier_and_prevalence_specialist_are_distinct(self):
        rows = [row(str(image), 1, [1 if image == 0 else "", 0, 0]) for image in range(5)] + [
            row(str(image), 2, [int(image < 2), 0, 0]) for image in range(5)
        ]
        profiles = event_profiles(record(), rows)
        self.assertTrue(profiles[1, 1].consistent)
        self.assertFalse(profiles[1, 2].consistent)
        self.assertEqual(compare_events(profiles[1, 1], profiles[1, 2])["minimum_joint_support"], 1)

    def test_summary_exclusions_and_conditional_denominators(self):
        profiles = event_profiles(record(2), [row("a", 1, [1, 0]), row("a", 2, [1, 0]), row("a", 3, [0, 0])])
        same = compare_events(profiles[1, 1], profiles[1, 2])
        constant = compare_events(profiles[1, 1], profiles[1, 3])
        excluded = compare_events(profiles[1, 1], profiles[1, 2], minimum_parts=5)
        result = summarize([same, constant, excluded])
        self.assertEqual((result["pairs"], result["eligible_pairs"], result["joint_defined"]), (3, 2, 1))
        self.assertEqual(result["undefined_rho"], 1)
        self.assertEqual(result["high_rho_fraction"], 1)
        self.assertEqual(result["high_jaccard_given_high_rho"], 1)
        self.assertIsNone(summarize([])["low_jaccard_given_high_rho"])

    def test_rejects_duplicate_nonbinary_and_cross_class_evidence(self):
        for rows, message in (
            ([row("a", 1, [1, 0, 1])] * 2, "Repeated"),
            ([row("a", 1, [0.5, 0, 1])], "binary"),
            ([row("a", 1, [1, 0, 1]), row("b", 1, [1, 0, 1], 2)], "exactly one class"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                event_profiles(record(), rows)
        profiles = event_profiles(record(), [row("a", 1, [1, 0, 1]), row("a", 2, [1, 0, 1], 2)])
        with self.assertRaisesRegex(ValueError, "class-local"):
            compare_events(profiles[1, 1], profiles[2, 2])

    def test_measurement_enumerates_only_unordered_within_class_pairs(self):
        metadata = record()
        metadata["scores"] = {"consistency": {"numerator": 3, "denominator": 3}}
        pairs, classes, measurements = analyze_measurement(
            metadata, [row("a", 1, [1, 0, 1]), row("a", 2, [1, 0, 0]), row("a", 3, [1, 0, 1], 2)]
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len(classes), 8)
        self.assertEqual(len(measurements), 4)
        self.assertTrue(all((p["left_prototype"], p["right_prototype"]) == (1, 2) for p in pairs))
        metadata["scores"]["consistency"]["numerator"] = 0
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            analyze_measurement(metadata, [row("a", 1, [1, 0, 1])])

    def test_frozen_cohort_rejects_missing_duplicate_and_wrong_dataset(self):
        measurements = [
            {
                "model": row["model"]["name"],
                "architecture": row["model"]["architecture"],
                "quantile": row["parameters"]["quantile"],
                "setting": "R",
            }
            for row in cohort_records()
            if row["parameters"]["require_semantic_match"]
            and row["parameters"]["exclude_background_centroid"]
            and not row["parameters"]["merge_left_right_parts"]
        ]
        manifest = {"analysis": "pascal-support-sensitivity-v1", "measurements": measurements}
        self.assertEqual(len(frozen_cohort(manifest, "pascal")), 18)
        for values in (measurements[:-1], measurements + measurements[:1]):
            modified = copy.deepcopy(manifest)
            modified["measurements"] = values
            with self.assertRaises(ValueError):
                frozen_cohort(modified, "pascal")
        with self.assertRaises(ValueError):
            frozen_cohort(manifest, "cityscapes")


if __name__ == "__main__":
    unittest.main()
