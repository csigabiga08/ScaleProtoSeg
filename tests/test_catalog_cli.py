"""CPU-only tests for catalog queries and validation."""

import contextlib
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from segmentation.catalog import cli as catalog_cli
from segmentation.catalog import CatalogError, load_observations


CONSISTENCY_PROTOCOL = {
    "metric": "consistency",
    "quantile": 0.8,
    "parts": {
        "valid_part_ids": [1, 2],
        "class_dependent_part_definitions": {
            "fixture": {"part_id_to_name": {"1": "part one", "2": "part two"}},
        },
    },
    "classes": {"evaluation_class_id_to_label": {"1": "fixture", "2": "other"}},
    "consistency": {
        "threshold": 0.8,
        "strict_comparison": "mean part presence > threshold",
        "aggregation": "unweighted mean over class/prototype rows",
    },
    "observation_columns": ["part_1", "part_2"],
}

def obs(sample_id, class_id=1, prototype_id=1, part_1="1", part_2="0", **extra):
    return {
        "sample_id": sample_id, "class_id": str(class_id), "prototype_id": str(prototype_id),
        "repetition": "0", "part_1": part_1, "part_2": part_2, **extra,
    }


def make_record(measurement_id, *, quantile=0.8, status="complete", checkpoint="b" * 64):
    protocol = copy.deepcopy(CONSISTENCY_PROTOCOL)
    protocol["quantile"] = quantile
    return {
        "measurement_id": measurement_id,
        "metric": "consistency",
        "status": status,
        "scores": {},
        "model": {"label": "fixture", "inputs": {"prototype_checkpoint": {"sha256": checkpoint}}},
        "dataset": {"name": "fixture-set", "split": "val"},
        "protocol": protocol,
        "parameters": {"quantile": quantile},
        "artifacts": {},
    }


def write_measurement(root, record, observations=None):
    """Write a bundle; with observations the artifact entry, counts, and score are filled in."""
    bundle = root / "measurements" / record["measurement_id"]
    bundle.mkdir(parents=True)
    if observations is not None:
        path = bundle / "observations.csv"
        columns = ["sample_id", "class_id", "prototype_id", "repetition"] + record["protocol"]["observation_columns"]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(observations)
        record["artifacts"]["observations"] = {
            "path": "observations.csv",
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        samples = {row["sample_id"] for row in observations}
        record["population"] = {"candidates": len(samples), "contributing": len(samples), "observations": len(observations)}
        if record["status"] == "complete":
            record["scores"][record["metric"]] = {
                "value": 1.0, "unit": "fraction", "numerator": 1, "denominator": 1,
            }
    (bundle / "result.json").write_text(json.dumps(record), encoding="utf-8")
    return bundle


class CatalogCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "catalog"
        (self.root / "measurements").mkdir(parents=True)
        patcher = patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cli(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = catalog_cli.main(list(args))
        return code, output.getvalue()

    def test_list_defaults_to_complete_and_filters_by_parameters(self):
        write_measurement(self.root, make_record("complete"), [obs("a")])
        write_measurement(self.root, make_record("running", status="running"))
        write_measurement(self.root, make_record("failed", status="failed"))
        write_measurement(self.root, make_record("other-q", quantile=0.9), [obs("a")])
        self.assertEqual({r["measurement_id"] for r in catalog_cli.find_records()}, {"complete", "other-q"})
        everything = catalog_cli.find_records(all_status=True)
        self.assertEqual(len(everything), 4)
        with self.assertRaisesRegex(CatalogError, "requires --all-status"):
            catalog_cli.find_records(statuses=("failed",))
        failed = catalog_cli.find_records(all_status=True, statuses=("failed",))
        self.assertEqual([r["measurement_id"] for r in failed], ["failed"])
        selected = catalog_cli.find_records(metric="consistency", dataset="fixture-set", quantile="0.9")
        self.assertEqual([r["measurement_id"] for r in selected], ["other-q"])

    def test_invalid_metadata_is_skipped_with_a_warning(self):
        record = make_record("broken")
        del record["protocol"]
        write_measurement(self.root, record)
        diagnostic = io.StringIO()
        with contextlib.redirect_stderr(diagnostic):
            self.assertEqual(catalog_cli.find_records(), [])
        self.assertIn("skipping broken", diagnostic.getvalue())

    def test_listing_does_not_open_observation_tables(self):
        record = make_record("metadata-only")
        bundle = write_measurement(self.root, record, [obs("a")])
        (bundle / "observations.csv").unlink()
        self.assertEqual(catalog_cli.find_records()[0]["measurement_id"], "metadata-only")

    def test_cli_list_and_validate_commands(self):
        write_measurement(self.root, make_record("cli"), [obs("a"), obs("b")])
        code, text = self.run_cli("list")
        self.assertEqual((code, json.loads(text)[0]["measurement_id"]), (0, "cli"))
        self.assertEqual(json.loads(self.run_cli("list", "--metric", "stability")[1]), [])
        code, text = self.run_cli("validate", "cli")
        self.assertEqual((code, json.loads(text)[0]["status"]), (0, "valid"))

    def test_validate_reports_invalid_bundles(self):
        bundle = write_measurement(self.root, make_record("tampered"), [obs("a")])
        (bundle / "observations.csv").write_text("corrupt\n")
        code, text = self.run_cli("validate", "tampered")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(text)[0]["status"], "invalid")

    def test_load_observations_distinguishes_empty_table_from_absent_evidence(self):
        bundle = write_measurement(self.root, make_record("empty", status="failed"), [obs("a")])
        (bundle / "observations.csv").write_text("sample_id,class_id,prototype_id,repetition,part_1,part_2\r\n")
        result = json.loads((bundle / "result.json").read_text())
        result["artifacts"]["observations"].update(
            bytes=(bundle / "observations.csv").stat().st_size,
            sha256=hashlib.sha256((bundle / "observations.csv").read_bytes()).hexdigest(),
        )
        result["population"] = {"candidates": 0, "contributing": 0, "observations": 0}
        (bundle / "result.json").write_text(json.dumps(result))
        self.assertEqual(load_observations(bundle)[1], [])

        absent = write_measurement(self.root, make_record("absent", status="failed"))
        with self.assertRaisesRegex(CatalogError, "finalized observation evidence"):
            load_observations(absent)


if __name__ == "__main__":
    unittest.main()
