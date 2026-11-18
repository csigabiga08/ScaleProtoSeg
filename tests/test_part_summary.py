"""Persistence and support sensitivity from binary image/part observations."""

import contextlib
import copy
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_catalog_cli import make_record, obs, write_measurement

from segmentation.catalog import CatalogError, sha256_file
from segmentation.catalog.cli import main
from segmentation.catalog.part_summary import (
    CSV_NAME,
    METADATA_NAME,
    backfill_part_summary,
    load_part_summary,
    support_curve,
    support_statistics,
)


class PartSummaryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.record = make_record("fixture")
        self.record["protocol"]["observation_columns"] = [f"part_{i}" for i in range(4)]
        self.record["protocol"]["parts"]["valid_part_ids"] = [1, 2, 3]
        self.record["selected_prototype_ids"] = [1, 2, 3, 4]
        rows = [
            obs("a", part_1="1", part_2="", part_0="1", part_3=""),
            obs("b", part_1="1", part_2="0", part_0="1", part_3=""),
            obs("c", part_1="1", part_2="0", part_0="1", part_3=""),
            obs("a", prototype_id=2, part_1="0", part_2="1", part_0="1", part_3=""),
            obs("b", prototype_id=2, part_1="0", part_2="", part_0="1", part_3=""),
            obs("c", prototype_id=2, part_1="0", part_2="", part_0="1", part_3=""),
            obs("a", prototype_id=3, part_1="", part_2="", part_0="1", part_3=""),
        ]
        self.bundle = write_measurement(self.root, self.record, rows)

    def derived(self, name):
        return self.bundle / "derived" / name

    def test_roundtrip_counts_and_arbitrary_support_limits_preserve_originals(self):
        originals = {name: (self.bundle / name).read_bytes() for name in ("result.json", "observations.csv")}
        self.assertIsNone(load_part_summary(self.bundle))
        self.assertEqual(backfill_part_summary(self.bundle), "created")
        summary = load_part_summary(self.bundle)
        counts = {(row.prototype_id, row.part_id): row for row in summary.counts}
        self.assertEqual((counts[1, 1].observations, counts[1, 1].hits), (3, 3))
        self.assertEqual((counts[1, 2].observations, counts[1, 2].hits), (2, 0))
        self.assertEqual((counts[2, 2].observations, counts[2, 2].hits), (1, 1))
        self.assertEqual((counts[3, 3].observations, counts[3, 3].hits), (0, 0))
        self.assertEqual(len(summary.counts), 12)
        self.assertEqual(summary.metadata.selected_prototype_ids, [1, 2, 3, 4])
        one = support_statistics(summary)
        two = support_statistics(summary, 2)
        self.assertEqual((one["numerator"], one["denominator"], one["value"]), (2, 3, 2 / 3))
        self.assertEqual((two["numerator"], two["rare_only"], two["insufficient_evidence"]), (1, 1, 1))
        self.assertEqual(two["evidence_coverage"], 2 / 3)
        self.assertEqual(support_statistics(summary, 4)["value"], 0)
        self.assertEqual(support_statistics(summary, 1, threshold=1)["numerator"], 0)
        self.assertEqual([row["numerator"] for row in support_curve(summary, limits=[1, 2, 3, 4])], [2, 1, 1, 0])
        for name, content in originals.items():
            self.assertEqual((self.bundle / name).read_bytes(), content)
        with patch("segmentation.catalog.part_summary.load_observations", side_effect=AssertionError("not reread")):
            timestamps = [self.derived(name).stat().st_mtime_ns for name in (CSV_NAME, METADATA_NAME)]
            self.assertEqual(backfill_part_summary(self.bundle), "current")
            self.assertEqual([self.derived(name).stat().st_mtime_ns for name in (CSV_NAME, METADATA_NAME)], timestamps)

    def test_strict_threshold_and_all_qualifying_parts_not_only_argmax(self):
        record = make_record("threshold")
        rows = [obs(str(i), part_1="1" if i == 0 else "", part_2="1" if i < 4 else "0") for i in range(5)]
        bundle = write_measurement(self.root, record, rows)
        backfill_part_summary(bundle)
        summary = load_part_summary(bundle)
        self.assertEqual(support_statistics(summary, 2)["numerator"], 0)
        # With a lower threshold, the better-supported association qualifies too.
        self.assertEqual(support_statistics(summary, 2, threshold=0.79)["numerator"], 1)

    def test_empty_population_is_undefined_not_zero(self):
        bundle = write_measurement(self.root, make_record("empty"), [])
        backfill_part_summary(bundle)
        summary = load_part_summary(bundle)
        self.assertEqual(summary.counts, [])
        statistics = support_statistics(summary)
        self.assertEqual(statistics["denominator"], 0)
        self.assertIsNone(statistics["value"])
        self.assertIsNone(statistics["evidence_coverage"])

    def test_stale_protocol_and_corrupt_counts_are_rebuilt(self):
        backfill_part_summary(self.bundle)
        record = json.loads((self.bundle / "result.json").read_text())
        record["protocol"]["consistency"]["threshold"] = 1.0
        (self.bundle / "result.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(CatalogError, "stale"):
            load_part_summary(self.bundle)
        self.assertEqual(backfill_part_summary(self.bundle), "created")
        self.assertEqual(support_statistics(load_part_summary(self.bundle))["numerator"], 0)
        self.derived(CSV_NAME).write_text("corrupt")
        with self.assertRaisesRegex(CatalogError, "hash or byte size"):
            load_part_summary(self.bundle)
        self.assertEqual(backfill_part_summary(self.bundle), "created")

    def test_csv_constraints_are_checked_even_when_hash_matches(self):
        backfill_part_summary(self.bundle)
        rows = list(csv.DictReader(io.StringIO(self.derived(CSV_NAME).read_text())))
        for change in (
            lambda items: items[0].update(hits=999),
            lambda items: items.append(copy.deepcopy(items[0])),
            lambda items: items.pop(),
        ):
            modified = copy.deepcopy(rows)
            change(modified)
            with self.derived(CSV_NAME).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(modified)
            metadata = json.loads(self.derived(METADATA_NAME).read_text())
            digest, size = sha256_file(self.derived(CSV_NAME))
            metadata["csv"].update(sha256=digest, bytes=size)
            self.derived(METADATA_NAME).write_text(json.dumps(metadata))
            with self.assertRaisesRegex(CatalogError, "Invalid part summary"):
                load_part_summary(self.bundle)

    def test_repetitions_do_not_inflate_image_support(self):
        bundle = write_measurement(self.root, make_record("repeated"), [obs("a"), obs("a", repetition="1")])
        with self.assertRaisesRegex(CatalogError, "independent image support"):
            backfill_part_summary(bundle)
        self.assertFalse((bundle / "derived").exists())

    def test_binary_evidence_and_source_hash_are_required(self):
        bundle = write_measurement(self.root, make_record("continuous"), [obs("a", part_1="0.5")])
        with self.assertRaisesRegex(CatalogError, "binary or missing"):
            backfill_part_summary(bundle)
        (self.bundle / "observations.csv").write_text("corrupt")
        with self.assertRaisesRegex(CatalogError, "byte size mismatch"):
            backfill_part_summary(self.bundle)

    def test_interrupted_publication_has_no_manifest_and_can_be_repaired(self):
        with patch("segmentation.catalog.part_summary.json.dump", side_effect=OSError("interrupted")):
            with self.assertRaisesRegex(CatalogError, "interrupted"):
                backfill_part_summary(self.bundle)
        self.assertFalse(self.derived(METADATA_NAME).exists())
        self.assertEqual(list(self.derived(CSV_NAME).parent.glob("*.tmp")), [])
        self.assertIsNone(load_part_summary(self.bundle))
        self.assertEqual(backfill_part_summary(self.bundle), "created")

    def test_derived_paths_cannot_escape_bundle(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.bundle / "derived").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(CatalogError, "inside"):
            backfill_part_summary(self.bundle)
        self.assertEqual(list(outside.iterdir()), [])

    def test_other_metrics_and_partial_runs_are_not_support_evidence(self):
        for name, values in (
            ("running", {"status": "running"}),
            ("failed", {"status": "failed"}),
            ("stability", {"metric": "stability"}),
        ):
            record = make_record(name)
            record.update(values)
            bundle = write_measurement(self.root, record, [obs("a")])
            with self.assertRaisesRegex(CatalogError, "complete consistency"):
                backfill_part_summary(bundle)

    def test_cli_backfill_filters_and_reports_per_bundle_errors(self):
        write_measurement(self.root, make_record("other-q", quantile=0.9), [obs("a")])
        broken = write_measurement(self.root, make_record("broken"), [obs("a")])
        (broken / "observations.csv").write_text("corrupt")
        output = io.StringIO()
        with patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)}):
            with contextlib.redirect_stdout(output):
                code = main(["backfill-part-summaries", "--quantile", "0.8"])
            self.assertEqual(code, 1)
            results = {row["measurement_id"]: row for row in json.loads(output.getvalue())}
            self.assertEqual(results["fixture"]["status"], "created")
            self.assertEqual(results["broken"]["status"], "error")
            self.assertNotIn("other-q", results)
            self.assertFalse((self.root / "measurements/other-q/derived").exists())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["backfill-part-summaries", "fixture"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue()), [{"measurement_id": "fixture", "status": "current"}])


if __name__ == "__main__":
    unittest.main()
