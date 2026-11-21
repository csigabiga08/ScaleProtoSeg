import unittest

import numpy as np

from segmentation.analysis.metrics.imprecise_activations import (
    _activation_sums,
    _class_pixel_counts,
    _outside_inside_ratios,
)


class ImpreciseActivationTests(unittest.TestCase):
    def test_pixel_counts_and_activation_sums_exclude_void(self):
        labels = np.array([[1, 1, 2, 2], [0, 1, 2, 2]])
        counts = _class_pixel_counts(labels, (1, 2), num_classes=2)
        np.testing.assert_array_equal(counts, [[3, 0], [0, 4]])

        activations = np.array([[[2, 4]], [[1, 3]]])
        sums = _activation_sums(activations, counts, np.array([0, 1]))
        np.testing.assert_array_equal(sums, [[6, 16], [12, 3]])

    def test_pooled_ratio_is_outside_mean_over_inside_mean(self):
        sums = np.array([[6.0, 16.0], [12.0, 3.0]])
        counts = np.array([[3, 4], [4, 3]])
        np.testing.assert_allclose(_outside_inside_ratios(sums, counts), [2.0, 1 / 3])

    def test_zero_inside_mean_has_no_defined_ratio(self):
        ratios = _outside_inside_ratios(np.array([[0.0, 4.0]]), np.array([[2, 2]]))
        self.assertTrue(np.isnan(ratios[0]))


if __name__ == "__main__":
    unittest.main()
