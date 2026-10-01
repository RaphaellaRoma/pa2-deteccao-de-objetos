import unittest
import numpy as np
from synthetic import generate, corrupt_detections


class SyntheticTests(unittest.TestCase):
    def test_occlusion(self):
        scene = generate(occlusion=10)
        obj = scene.gt[scene.gt[:, 1] == 1]
        hidden = np.flatnonzero(obj[:, 6] == 0)
        self.assertEqual(len(hidden), 10)
        self.assertTrue(np.all(np.diff(hidden) == 1))
        self.assertGreater(obj[hidden[0]-1, 6], 0)
        self.assertGreater(obj[hidden[-1]+1, 6], 0)

    def test_detector_drop(self):
        self.assertEqual(corrupt_detections(generate().gt, drop=1).shape, (0, 6))

    def test_repeatability(self):
        a, b = generate(seed=7), generate(seed=7)
        np.testing.assert_array_equal(a.frames, b.frames)
        np.testing.assert_array_equal(a.gt, b.gt)
