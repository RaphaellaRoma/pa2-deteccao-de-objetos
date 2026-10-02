import unittest
import numpy as np
from parte5 import corrupt_detections


class StressTests(unittest.TestCase):
    def setUp(self):
        self.det = np.array([[1, 10, 10, 12, 30, .9], [3, 20, 20, 8, 25, .8]])

    def test_original_is_exact_and_not_mutated(self):
        result = corrupt_detections(self.det, 5, 64, 64,
                                   dict(drop_probability=0, box_noise_fraction=0,
                                        false_positives_per_frame=0), 1)
        np.testing.assert_array_equal(result, self.det)
        result[0, 1] = 99
        self.assertEqual(self.det[0, 1], 10)

    def test_drop_all_and_empty_frames(self):
        result = corrupt_detections(self.det, 5, 64, 64,
                                   dict(drop_probability=1, box_noise_fraction=0,
                                        false_positives_per_frame=0), 1)
        self.assertEqual(result.shape, (0, 6))

    def test_reproducibility_bounds_and_fp_on_empty_input(self):
        params = dict(drop_probability=.3, box_noise_fraction=.8,
                      false_positives_per_frame=10)
        a = corrupt_detections(self.det, 5, 64, 64, params, 7)
        b = corrupt_detections(self.det, 5, 64, 64, params, 7)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(a[:, 1:3] >= 0))
        self.assertTrue(np.all(a[:, 1:3] + a[:, 3:5] <= 64 + 1e-10))
        empty = corrupt_detections(np.empty((0, 6)), 5, 64, 64, params, 7)
        self.assertGreater(len(empty), 0)
        self.assertEqual(set(empty[:, 0]), {1, 2, 3, 4, 5})

    def test_invalid_parameters_rejected(self):
        for noise in [-1, float('nan')]:
            with self.assertRaises(ValueError):
                corrupt_detections(self.det, 5, 64, 64,
                                   dict(drop_probability=.1, box_noise_fraction=noise,
                                        false_positives_per_frame=1), 0)
