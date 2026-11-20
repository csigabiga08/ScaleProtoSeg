"""Reference-table reconstruction and incomplete-cohort safeguards."""

import unittest

from test_support_sensitivity import cityscapes_record

from segmentation.analysis.reference_tables import (
    activation_statistics,
    article_cohorts,
    reconstruct,
    select_cells,
)
from segmentation.analysis.support_sensitivity import aggregate


class ReferenceTablesTest(unittest.TestCase):
    def test_stability_uses_exact_vectors_including_shared_zeros(self):
        record = {"metric": "stability", "protocol": {"observation_columns": ["part_0", "noisy_part_0"]}}
        rows = [
            dict(sample_id=str(index), class_id="1", prototype_id="2", repetition="0", part_0=a, noisy_part_0=b)
            for index, (a, b) in enumerate((("0", "0"), ("1", "0"), ("1", "1")))
        ]
        self.assertEqual(reconstruct(record, rows), (2, 3))
        with self.assertRaisesRegex(ValueError, "Repeated"):
            reconstruct(record, rows + rows[:1])
        rows[0]["part_0"] = ""
        with self.assertRaises(ValueError):
            reconstruct(record, rows)

    def test_missing_seed_suppresses_aggregate_instead_of_averaging_two(self):
        rows = [
            dict(dataset="pascal", architecture="ProtoSeg", model=model, quantile=q, value=value)
            for model, value in (("default", 0.2), ("matched42", None), ("43", 0.4))
            for q in (0.7, 0.8, 0.9)
        ]
        _, summary = aggregate(rows, ("dataset", "architecture"), ("value",))
        self.assertIsNone(summary[0]["value_mean"])
        self.assertIsNone(summary[0]["value_sd"])
        self.assertIn("pascalvoc_seed-42_run02", article_cohorts()["pascal"])
        self.assertNotIn("pascalvoc_seed-42", article_cohorts()["pascal"])

    def test_duplicate_identity_and_noise_guards(self):
        record = cityscapes_record("cityscapes_kld_imnet_4_16v2_seed42_run01", 0.7, False)
        model = record["model"]
        identities = {
            ("cityscapes", model["name"]): {
                "architecture": "ProtoSeg",
                "identity": (model["inputs"]["prototype_checkpoint"]["sha256"], None),
            }
        }
        self.assertEqual(len(select_cells([record], identities)), 1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_cells([record, record], identities)
        record["metric"] = "stability"
        record["parameters"].update(std_dev=0.2, seed=1)
        record["protocol"]["stability"] = {"standard_deviation": 0.2, "seed": 1}
        with self.assertRaisesRegex(ValueError, "noise"):
            select_cells([record], identities)
        model["inputs"]["prototype_checkpoint"]["sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            select_cells([record], identities)

    def test_activation_ratio_population_and_duplicate_guards(self):
        rows = [
            dict(
                proto_id=str(index),
                inside_pixels="10",
                outside_pixels="20",
                inside_mean="2",
                outside_mean=str(outside),
                imprecise_activation=str(outside / 2),
            )
            for index, outside in enumerate((1, 3))
        ]
        stats = activation_statistics(rows)
        self.assertEqual((stats["prototypes"], stats["mean"], stats["median"]), (2, 1, 1))
        with self.assertRaisesRegex(ValueError, "Repeated"):
            activation_statistics(rows + rows[:1])
        rows[0]["imprecise_activation"] = "0.75"
        with self.assertRaisesRegex(ValueError, "reconstruction"):
            activation_statistics(rows)


if __name__ == "__main__":
    unittest.main()
