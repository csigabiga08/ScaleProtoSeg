import unittest

import numpy as np

from segmentation.data.pascal_parts import (
    PASCAL_LEFT_RIGHT_PAIRS,
    merge_left_right_presence,
)


class PascalPartMergeTests(unittest.TestCase):
    def test_defines_54_class_specific_left_right_pairs(self):
        self.assertEqual(sum(len(pairs) for pairs in PASCAL_LEFT_RIGHT_PAIRS.values()), 54)

    def test_merges_observed_sides_and_preserves_unpaired_values(self):
        presence = [np.nan] * 25
        presence[12] = 1
        presence[13] = 1
        presence[16] = 0
        presence[14] = np.nan
        presence[17] = 0

        merged = merge_left_right_presence(presence, "person")

        self.assertEqual(merged[12], 1)
        self.assertEqual(merged[13], 1)
        self.assertTrue(np.isnan(merged[16]))
        self.assertEqual(merged[14], 0)
        self.assertTrue(np.isnan(merged[17]))
        self.assertTrue(np.isnan(merged[15]))
        self.assertTrue(np.isnan(merged[18]))
        self.assertEqual(presence[16], 0)


if __name__ == "__main__":
    unittest.main()
