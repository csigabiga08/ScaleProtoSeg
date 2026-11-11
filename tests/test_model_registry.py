"""Registry uniqueness, registered inputs and reviewed migration, without model imports."""

import copy
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import ValidationError

from segmentation.catalog import CatalogError
from segmentation.catalog.migrate import migrate, migration_plan
from segmentation.catalog.models import find_model, list_models, register_model, relocate_model
from test_catalog_cli import make_record, obs, write_measurement


class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "weights.pth"
        self.checkpoint.write_bytes(b"checkpoint contents")

    def register(self, **overrides):
        options = dict(
            root=self.root,
            name="main",
            architecture="ProtoSeg",
            config={"stage": "push"},
            dataset="pascal",
            checkpoint=self.checkpoint,
        )
        return register_model(**{**options, **overrides})

    def test_unique_names_and_content_and_uuid_or_name_resolution(self):
        model = self.register(seed=17, tags={"experiment": "baseline"})
        self.assertEqual(find_model("main", self.root), find_model(model["model_id"], self.root))
        copied = self.root / "copied.pth"
        copied.write_bytes(self.checkpoint.read_bytes())
        with self.assertRaisesRegex(CatalogError, "already registered"):
            self.register(name="copied", checkpoint=copied)
        copied.write_bytes(b"other weights")
        with self.assertRaisesRegex(CatalogError, "already registered"):
            self.register(checkpoint=copied)
        self.assertEqual(len(list_models(self.root)), 1)

    def test_concurrent_duplicate_registration_has_one_winner(self):
        def attempt(index):
            try:
                self.register(name=f"model-{index}")
                return True
            except CatalogError:
                return False

        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(sum(pool.map(attempt, range(4))), 1)
        self.assertEqual(len(list_models(self.root)), 1)

    def test_architecture_schemas_reject_unknown_fields_and_versions(self):
        for overrides in (
            {"config": {"stage": "push", "typo": 1}},
            {"schema_version": 2},
            {"architecture": "unknown"},
            {"architecture": "ScaleProtoSeg", "config": {"stage": "push"}},
        ):
            with self.subTest(overrides=overrides), self.assertRaises((ValueError, ValidationError)):
                self.register(**overrides)

    def test_relocation_verifies_contents_and_identity_does_not_change(self):
        model = self.register()
        moved = self.root / "moved.pth"
        self.checkpoint.rename(moved)
        relocated = relocate_model("main", moved, self.root)
        self.assertEqual(relocated["model_id"], model["model_id"])
        self.assertEqual(find_model("main", self.root, verify=True)["checkpoint"]["path"], str(moved))
        moved.write_bytes(b"changed")
        with self.assertRaisesRegex(CatalogError, "changed"):
            find_model("main", self.root, verify=True)
        with self.assertRaisesRegex(CatalogError, "contents"):
            relocate_model("main", moved, self.root)

    def test_reviewed_migration_preserves_observations_and_backs_up_original(self):
        model = self.register()
        record = make_record("historical")
        record["dataset"]["name"] = "pascal"
        record["model"]["inputs"] = {"prototype_checkpoint": {"sha256": None}}
        record["legacy_metadata"] = {
            "description": {
                "checkpoint": str(self.checkpoint),
                "dataset": "pascal",
                "parameters": {"quantile_scope": "image", "merge_left_right_parts": True},
            }
        }
        bundle = write_measurement(self.root, record, [obs("sample")])
        original = (bundle / "result.json").read_bytes()
        observations = (bundle / "observations.csv").read_bytes()
        manifest = migration_plan(self.root)
        entry = manifest["measurements"]["historical"]
        entry["model"] = model["model_id"]
        with self.assertRaisesRegex(CatalogError, "confirmation"):
            migrate(self.root, manifest)
        entry["confirm_historical_identity"] = True
        self.assertEqual(migrate(self.root, manifest)["status"], "dry_run")
        self.assertEqual((bundle / "result.json").read_bytes(), original)
        self.assertFalse((self.root / "backups").exists())
        result = migrate(self.root, manifest, apply=True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual((Path(result["backup"]) / "historical.json").read_bytes(), original)
        self.assertEqual((bundle / "observations.csv").read_bytes(), observations)
        updated = json.loads((bundle / "result.json").read_text())
        self.assertEqual(updated["model"]["registration_id"], model["model_id"])
        self.assertEqual(updated["parameters"]["quantile_scope"], "image")
        self.assertTrue(updated["parameters"]["merge_left_right_parts"])

    def test_migration_requires_historical_group_and_rejects_hash_mismatch(self):
        model = self.register()
        record = make_record("historical")
        record["dataset"]["name"] = "pascal"
        record["legacy_metadata"] = {"description": {"group_checkpoint": "/historical/group.pth"}}
        write_measurement(self.root, record, [obs("sample")])
        entry = {"model": model["model_id"], "confirm_historical_identity": True}
        with self.assertRaisesRegex(CatalogError, "grouping model"):
            migrate(self.root, {"measurements": {"historical": entry}})
        entry["grouping_model"] = model["model_id"]
        with self.assertRaisesRegex(CatalogError, "hash mismatch"):
            migrate(self.root, {"measurements": {"historical": entry}})

    def test_migration_rejects_tampered_observations_without_writing(self):
        model = self.register()
        record = make_record("historical")
        record["dataset"]["name"] = "pascal"
        record["model"]["inputs"] = {"prototype_checkpoint": model["checkpoint"]}
        bundle = write_measurement(self.root, record, [obs("sample")])
        original = (bundle / "result.json").read_bytes()
        (bundle / "observations.csv").write_text("corrupted")
        with self.assertRaises(CatalogError):
            migrate(self.root, {"measurements": {"historical": {"model": "main"}}}, apply=True)
        self.assertEqual((bundle / "result.json").read_bytes(), original)
        self.assertFalse((self.root / "backups").exists())
