"""Registry identity, architecture validation and verified checkpoint relocation."""

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import ValidationError

from segmentation.catalog import CatalogError
from segmentation.catalog.models import find_model, list_models, register_model, relocate_model


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
