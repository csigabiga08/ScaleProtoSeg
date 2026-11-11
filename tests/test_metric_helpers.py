import unittest
from types import SimpleNamespace

import torch

from segmentation.analysis.metrics.helpers import proto_filter, selected_prototypes


class MetricHelperTests(unittest.TestCase):
    def test_proto_filter_maps_class_local_ids_to_global_ids(self):
        model = SimpleNamespace(
            group_projection=[
                SimpleNamespace(weight=torch.tensor([[1.0, 0.0, 1.0]])),
                SimpleNamespace(weight=torch.tensor([[1.0]])),
            ],
            prototype_class_identity=torch.tensor([
                [1, 0],
                [1, 0],
                [0, 1],
                [1, 0],
            ]),
        )

        self.assertEqual(proto_filter(model), [0, 3, 2])

    def test_selected_prototypes_applies_allowed_ids_and_filtered_classes(self):
        model = SimpleNamespace(
            prototype_class_identity=torch.tensor([
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
            ])
        )

        self.assertEqual(
            selected_prototypes(model, allowed_ids=[1, 2], filtered_classes=[2]),
            [1, 2],
        )
        self.assertEqual(selected_prototypes(model, allowed_ids=None), [0, 1, 1, 2, 2, 3])


if __name__ == "__main__":
    unittest.main()
