import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from segmentation.analysis.metrics import helpers
from segmentation.analysis.metrics.helpers import proto_filter, selected_prototypes


class MetricHelperTests(unittest.TestCase):
    def test_clean_and_noisy_intersections_preserve_rng_and_disable_gradients(self):
        image = Image.new("RGB", (2, 2))
        semantic = np.zeros((2, 2), dtype=int)
        parts = np.zeros_like(semantic)
        model = MagicMock()
        model.forward_from_conv_features.return_value = (None, torch.zeros((1, 1, 2, 2)))
        tensor = torch.zeros((3, 2, 2), requires_grad=True)
        options = dict(max_parts=5, missing_parts_nan=False)
        with (
            patch.object(helpers, "to_normalized_tensor", return_value=tensor),
            patch.object(torch.Tensor, "cuda", lambda self: self),
            torch.random.fork_rng(devices=[]),
        ):
            torch.manual_seed(41)
            before_clean = torch.get_rng_state()
            helpers.part_intersect(image, semantic, parts, model, {}, "sample", 0.8, **options)
            self.assertTrue(torch.equal(before_clean, torch.get_rng_state()))
            helpers.part_intersect(image, semantic, parts, model, {}, "sample", 0.8, noise=0.2, **options)
            noisy = model.conv_features.call_args.args[0]
            self.assertFalse(noisy.requires_grad)
            self.assertFalse(torch.equal(before_clean, torch.get_rng_state()))
            torch.manual_seed(41)
            expected = 0.2 * torch.randn_like(tensor.unsqueeze(0))
            self.assertTrue(torch.equal(noisy, expected))

            torch.manual_seed(41)
            helpers.part_intersect(image, semantic, parts, model, {}, "sample", 0.8, noise=0.0, **options)
            self.assertTrue(torch.equal(model.conv_features.call_args.args[0], tensor.unsqueeze(0)))
            self.assertFalse(torch.equal(before_clean, torch.get_rng_state()))

    def test_proto_filter_maps_class_local_ids_to_global_ids(self):
        model = SimpleNamespace(
            group_projection=[
                SimpleNamespace(weight=torch.tensor([[1.0, 0.0, 1.0]])),
                SimpleNamespace(weight=torch.tensor([[1.0]])),
            ],
            prototype_class_identity=torch.tensor(
                [
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [1, 0],
                ]
            ),
        )

        self.assertEqual(proto_filter(model), [0, 3, 2])

    def test_selected_prototypes_applies_allowed_ids_and_filtered_classes(self):
        model = SimpleNamespace(
            prototype_class_identity=torch.tensor(
                [
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 1],
                    [0, 0, 1],
                ]
            )
        )

        self.assertEqual(
            selected_prototypes(model, allowed_ids=[1, 2], filtered_classes=[2]),
            [1, 2],
        )
        self.assertEqual(selected_prototypes(model, allowed_ids=None), [0, 1, 1, 2, 2, 3])


if __name__ == "__main__":
    unittest.main()
