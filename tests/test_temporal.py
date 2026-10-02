import unittest

import numpy as np

from temporal import MotionGRU, TemporalTracker, clip_boxes


class TemporalTrackerTests(unittest.TestCase):
    def test_reappearance_within_and_after_memory_limit(self):
        # Cabeça zerada: a previsão inicial mantém a mesma caixa. Assim o teste
        # isola o contrato de duração da memória e associação, sem treino.
        for age, expected in ((2, 1), (1, 2)):
            tracker = TemporalTracker(MotionGRU(), 100, 100, max_age=age)
            first = tracker.update(1, [[10, 10, 10, 10]])
            self.assertEqual(first[0][1], 1)
            self.assertEqual(tracker.update(2, []), [])
            self.assertEqual(tracker.update(3, []), [])
            returned = tracker.update(4, [[10, 10, 10, 10]])
            self.assertEqual(returned[0][1], expected)

    def test_forecast_is_clipped_inside_frame(self):
        boxes = clip_boxes(np.array([[-4, 95, 20, 20], [99, 99, -3, -2]]), 100, 100)
        self.assertTrue((boxes[:, :2] >= 0).all())
        self.assertTrue((boxes[:, 2:] >= 1).all())
        self.assertTrue((boxes[:, :2] + boxes[:, 2:] <= 100).all())


if __name__ == '__main__':
    unittest.main()
