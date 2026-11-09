"""CPU-only tests for the catalog webapp's tabular views."""

import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from segmentation.analysis.metrics import catalog_table as table


def record(measurement_id, metric="consistency", quantile=0.8, value=0.5, numerator=1, denominator=2, **parameters):
    return {
        "measurement_id": measurement_id,
        "metric": metric,
        "status": "complete",
        "model": {"label": "fixture", "inputs": {}},
        "dataset": {"name": "pascal"},
        "parameters": {"quantile": quantile, **parameters},
        "protocol": {"classes": {"evaluation_class_id_to_label": {"1": "a"}}},
        "scores": {metric: {"value": value, "numerator": numerator, "denominator": denominator}},
        "artifacts": {},
    }


class CatalogTableTest(unittest.TestCase):
    def test_frame_marks_missing_inputs_and_hides_lookup_tables(self):
        frame = table.records_frame([
            record("a", flag=True),
            record("b", value=None, numerator=None, denominator=None),
        ])
        self.assertEqual(frame.loc["a", "parameters.flag"], "true")
        self.assertEqual(frame.loc["b", "parameters.flag"], table.MISSING)
        self.assertEqual(frame.loc["a", "model.training_stage"], table.MISSING)
        self.assertTrue(math.isnan(frame.loc["b", "score"]))
        self.assertFalse(any(column.startswith("protocol.classes") for column in frame.columns))
        self.assertIn("parameters.flag", table.input_columns(frame))

    def test_aggregate_never_mixes_metrics_and_pools_only_complete_counts(self):
        frame = table.records_frame([
            record("a", quantile=0.7, value=0.5, numerator=1, denominator=2),
            record("b", quantile=0.8, value=0.25, numerator=1, denominator=4),
            record("c", metric="stability", value=1.0, numerator=3, denominator=3),
            record("d", metric="stability", quantile=0.9, value=0.0, numerator=None, denominator=5),
        ])
        summary = table.aggregate(frame, []).set_index("metric")
        self.assertEqual(summary.loc["consistency", "runs"], 2)
        self.assertAlmostEqual(summary.loc["consistency", "mean"], 0.375)
        self.assertAlmostEqual(summary.loc["consistency", "pooled"], 2 / 6)
        self.assertAlmostEqual(summary.loc["stability", "mean"], 0.5)
        self.assertTrue(math.isnan(summary.loc["stability", "pooled"]))

    def test_pivot_mean_over_column_is_undefined_when_a_value_is_missing(self):
        frame = table.records_frame([
            record("a", quantile=0.7, value=0.2, model_seed=1),
            record("b", quantile=0.8, value=0.4, model_seed=1),
            record("c", quantile=0.8, value=0.6, model_seed=1),
            record("d", quantile=0.7, value=0.9, model_seed=2),
        ])
        pivot = table.pivot_means(frame, ["parameters.model_seed"], "parameters.quantile")
        pivot = pivot.set_index("parameters.model_seed")
        self.assertAlmostEqual(pivot.loc["1", "mean over parameters.quantile"], (0.2 + 0.5) / 2)
        self.assertTrue(math.isnan(pivot.loc["2", "mean over parameters.quantile"]))

    def test_differing_fields_lists_only_inputs_that_change(self):
        frame = table.records_frame([record("a", quantile=0.7), record("b", quantile=0.9)])
        diff = table.differing_fields(frame, ["a", "b"], table.input_columns(frame))
        self.assertEqual(list(diff.index), ["score", "numerator", "denominator", "parameters.quantile"])

    def test_boolean_columns_accepts_constant_flags_and_missing_values(self):
        frame = table.records_frame([
            record("a", flag=True, other=False, mixed=True),
            record("b", flag=False, other=False, mixed="legacy"),
            record("c", other=False),
        ])
        self.assertEqual(
            table.boolean_columns(frame, table.input_columns(frame)),
            ["parameters.flag", "parameters.other"],
        )

    def test_compare_flag_pairs_other_inputs_and_preserves_individual_scores(self):
        frame = table.records_frame([
            record("a", flag=False, other=True, value=0.2),
            record("b", flag=True, other=True, value=0.6),
            record("c", flag=True, other=True, value=0.8),
            record("d", flag=True, other=True, value=None),
            record("e", flag=True, other=True, quantile=0.9, value=1.0),
            record("f", flag=True, other=False, value=0.9),
            record("g", metric="stability", flag=False, other=True, value=0.1),
            record("h", other=True, value=1.0),
            record("i", flag=False, other=True, value=0.3),
        ])
        comparison = table.compare_flag(frame, "parameters.flag", [])
        self.assertEqual(len(comparison), 1)
        paired = comparison.iloc[0]
        self.assertEqual(paired[("", "metric")], "consistency")
        self.assertEqual(paired[("", "parameters.quantile")], "0.8")
        self.assertEqual(paired[("", "parameters.other")], "true")
        self.assertEqual(paired[("score", "Off")], [0.2, 0.3])
        self.assertEqual(paired[("score", "On")], [0.6, 0.8, None])
        self.assertNotIn("parameters.flag", comparison.columns.get_level_values(1))

    def test_compare_flag_respects_extra_fields_and_empty_filters(self):
        first = record("a", flag=False)
        second = record("b", flag=True)
        first["execution"] = {"commit": "old"}
        second["execution"] = {"commit": "new"}
        frame = table.records_frame([first, second])
        self.assertEqual(len(table.compare_flag(frame, "parameters.flag", [])), 1)
        comparison = table.compare_flag(frame, "parameters.flag", ["execution.commit"])
        self.assertTrue(comparison.empty)
        for empty in (frame.iloc[:0], frame.iloc[:1], table.records_frame([record("c", flag=None)])):
            comparison = table.compare_flag(empty, "parameters.flag", [])
            self.assertTrue(comparison.empty)
            self.assertIn(("score", "Off"), comparison)
            self.assertIn(("score", "On"), comparison)

    def test_compare_flag_keeps_pairs_even_when_scores_are_missing(self):
        frame = table.records_frame([
            record("a", flag=False, value=None),
            record("b", flag=True, value=None),
        ])
        comparison = table.compare_flag(frame, "parameters.flag", [])
        self.assertEqual(len(comparison), 1)
        self.assertEqual(comparison.iloc[0][("score", "Off")], [None])
        self.assertEqual(comparison.iloc[0][("score", "On")], [None])

    def test_root_resolution_requires_only_read_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "measurements").mkdir()
            os.chmod(root, 0o555)
            try:
                with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(root)}):
                    self.assertEqual(table.read_catalog_root(checkout=root), root.resolve())
            finally:
                os.chmod(root, 0o755)


if __name__ == "__main__":
    unittest.main()
