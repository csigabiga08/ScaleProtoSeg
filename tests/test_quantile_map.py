"""Regression tests for class-restricted activation quantiles."""

import unittest

import torch

from segmentation.analysis.equivariance import quantile_map


class QuantileMapTests(unittest.TestCase):
    def test_pixels_outside_class_do_not_affect_threshold_or_become_active(self):
        mask = torch.tensor([[True, True, False], [False, False, False]])
        activations = torch.tensor([[[1.0, 3.0, 0.0], [0.0, 0.0, 0.0]]])
        expected = torch.tensor([[[False], [True], [False]], [[False], [False], [False]]])
        for outside in (0.0, -1000.0, 1000.0, float("nan")):
            with self.subTest(outside=outside):
                activations[:, ~mask] = outside
                self.assertTrue(torch.equal(quantile_map(activations, 0.8, mask), expected))

    def test_zero_activations_inside_class_participate_in_quantile(self):
        activations = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 2.0]]])
        mask = torch.ones((1, 5), dtype=torch.bool)
        expected = torch.tensor([[[False], [False], [False], [True], [True]]])
        self.assertTrue(torch.equal(quantile_map(activations, 0.5, mask), expected))

    def test_channels_have_independent_quantiles_and_output_is_hwc(self):
        activations = torch.tensor([[[1.0, 3.0, 2.0]], [[30.0, 10.0, 20.0]]])
        mask = torch.ones((1, 3), dtype=torch.bool)
        expected = torch.tensor([[[False, True], [True, False], [False, False]]])
        result = quantile_map(activations, 0.5, mask)
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.device, activations.device)
        self.assertEqual(result.dtype, torch.bool)

    def test_empty_class_and_constant_activations_have_no_active_pixels(self):
        activations = torch.ones((2, 3, 4))
        for selected in (False, True):
            with self.subTest(selected=selected):
                mask = torch.full((3, 4), selected)
                result = quantile_map(activations, 0.8, mask)
                self.assertEqual(result.shape, (3, 4, 2))
                self.assertFalse(result.any().item())

    def test_endpoints_and_negative_activations(self):
        activations = torch.tensor([[[-3.0, -2.0, -1.0, 100.0]]])
        mask = torch.tensor([[True, True, True, False]])
        expected = torch.tensor([[[False], [True], [True], [False]]])
        self.assertTrue(torch.equal(quantile_map(activations, 0.0, mask), expected))
        self.assertFalse(quantile_map(activations, 1.0, mask).any().item())

    def test_low_precision_activations(self):
        activations = torch.tensor([[[1.0, 3.0]]], dtype=torch.float16)
        expected = torch.tensor([[[False], [True]]])
        self.assertTrue(torch.equal(quantile_map(activations, 0.5, torch.ones((1, 2))), expected))

    def test_invalid_quantiles_are_rejected_even_for_empty_class(self):
        for quantile in (-0.1, 1.1, float("nan")):
            with self.subTest(quantile=quantile), self.assertRaises(ValueError):
                quantile_map(torch.ones((1, 2, 3)), quantile, torch.zeros((2, 3), dtype=torch.bool))

    def test_invalid_shapes_and_dtype_are_rejected(self):
        with self.assertRaises(ValueError):
            quantile_map(torch.ones((2, 3)), 0.8, torch.ones((2, 3), dtype=torch.bool))
        with self.assertRaises(ValueError):
            quantile_map(torch.ones((1, 2, 3)), 0.8, torch.ones((3, 2), dtype=torch.bool))
        with self.assertRaises(TypeError):
            quantile_map(torch.ones((1, 2, 3), dtype=torch.int64), 0.8, torch.ones((2, 3)))

    def test_nonfinite_activations_inside_class_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quantile_map(torch.tensor([[[value]]]), 0.8, torch.ones((1, 1), dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
