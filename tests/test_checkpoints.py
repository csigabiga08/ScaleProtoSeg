"""Regression checks for complete-model and historical checkpoint loading."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

import torch

from segmentation.checkpoints import load_model_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_complete_model_loads_on_cpu_in_both_formats(self):
        model = torch.nn.Linear(3, 2)
        model.description = "complete model checkpoint"
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pth"
            for zip_format in (True, False):
                with self.subTest(zip_format=zip_format):
                    torch.save(model, path, _use_new_zipfile_serialization=zip_format)
                    restored = load_model_checkpoint(path)
                    self.assertIsInstance(restored, torch.nn.Linear)
                    self.assertEqual(restored.description, model.description)
                    self.assertEqual(restored.weight.device.type, "cpu")
                    self.assertTrue(torch.equal(restored.weight, model.weight))

    def test_historical_group_model_module_name(self):
        from segmentation.model.model_multiscale_group import PPNetMultiScale

        # Serialize a small model with the module name used by released files.
        model = PPNetMultiScale.__new__(PPNetMultiScale)
        torch.nn.Module.__init__(model)
        model.group_projection = torch.nn.ModuleList([torch.nn.Linear(2, 1)])
        historical_module = ModuleType("model_multiscale_group")
        historical_module.PPNetMultiScale = PPNetMultiScale

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "historical.pth"
            for zip_format in (True, False):
                with self.subTest(zip_format=zip_format):
                    with patch.dict(sys.modules, {"model_multiscale_group": historical_module}):
                        with patch.object(PPNetMultiScale, "__module__", "model_multiscale_group"):
                            torch.save(model, path, _use_new_zipfile_serialization=zip_format)
                    restored = load_model_checkpoint(path)
                    self.assertIsInstance(restored, PPNetMultiScale)
                    self.assertTrue(torch.equal(
                        restored.group_projection[0].weight,
                        model.group_projection[0].weight,
                    ))

    def test_protoseg_package_names(self):
        import io

        from segmentation import checkpoint_pickle
        from segmentation.model.model import PPNet
        from segmentation.utils import MSC
        from deeplab_pytorch.libs.models.deeplabv2 import DeepLabV2

        # GLOBAL references taken from the downloaded ProtoSeg checkpoint.
        for module, name, expected in (
            ("proto_segmentation.model", "PPNet", PPNet),
            ("proto_segmentation.segmentation.utils", "MSC", MSC),
            ("proto_segmentation.deeplab_pytorch.libs.models.deeplabv2", "DeepLabV2", DeepLabV2),
        ):
            with self.subTest(module=module):
                serialized = f"c{module}\n{name}\n.".encode("ascii")
                self.assertIs(checkpoint_pickle.load(io.BytesIO(serialized)), expected)


if __name__ == "__main__":
    unittest.main()
