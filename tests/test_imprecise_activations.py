import unittest

import numpy as np

from segmentation.analysis.metrics.imprecise_activations import (
    _activation_sums,
    _class_pixel_counts,
    _image_activation_scores,
    _mean_per_prototype,
)


class ImpreciseActivationTests(unittest.TestCase):
    def test_pixel_counts_and_activation_sums_exclude_void(self):
        labels = np.array([[1, 1, 2, 2], [0, 1, 2, 2]])
        counts = _class_pixel_counts(labels, (1, 2), num_classes=2)
        np.testing.assert_array_equal(counts, [[3, 0], [0, 4]])

        activations = np.array([[[2, 4]], [[1, 3]]])
        sums = _activation_sums(activations, counts, np.array([0, 1]))
        np.testing.assert_array_equal(sums, [[6, 16], [12, 3]])

    def test_per_image_score_is_normalized_and_empty_regions_are_zero(self):
        sums = np.array([[2.0, 2.0], [0.0, 8.0], [6.0, 0.0], [0.0, 0.0]])
        counts = np.array([[2, 2], [0, 4], [3, 0], [2, 2]])
        scores = _image_activation_scores(sums, counts)
        np.testing.assert_allclose(scores[:3], [0.5, 1.0, 0.0])
        self.assertTrue(np.isnan(scores[3]))

    def test_prototype_score_is_mean_of_image_scores(self):
        scores = _mean_per_prototype(np.array([[0.5], [0.1], [np.nan]]))
        np.testing.assert_allclose(scores, [0.3])

    def test_prototype_without_defined_image_scores_is_undefined(self):
        scores = _mean_per_prototype(np.array([[np.nan], [np.nan]]))
        self.assertTrue(np.isnan(scores[0]))


if __name__ == "__main__":
    unittest.main()
