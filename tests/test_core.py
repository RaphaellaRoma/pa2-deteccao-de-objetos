import unittest
import numpy as np
from metrics import evaluate
from geometry import nms, match_iou
from baseline import IoUTracker


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.array([[f, i, 30*i, 0, 10, 10] for f in range(1, 5) for i in (1, 2)], float)

    def test_perfect(self):
        m = evaluate(self.gt, self.gt)
        self.assertEqual((m['idf1'], m['id_switches'], m['fragmentations']), (1, 0, 0))

    def test_swapped(self):
        p = self.gt.copy()
        p[p[:, 0] >= 3, 1] = 3-p[p[:, 0] >= 3, 1]
        m = evaluate(self.gt, p)
        self.assertEqual((m['idf1'], m['id_switches']), (0.5, 2))

    def test_split(self):
        p = self.gt.copy()
        p[(p[:, 0] >= 3) & (p[:, 1] == 1), 1] = 3
        m = evaluate(self.gt, p)
        self.assertEqual((m['idf1'], m['id_switches'], m['fragmentations']), (0.75, 1, 0))

    def test_gap(self):
        p = self.gt[~((self.gt[:, 0] == 2) & (self.gt[:, 1] == 1))]
        m = evaluate(self.gt, p)
        self.assertEqual(m['fragmentations'], 1)
        self.assertEqual(m['id_switches'], 0)
        self.assertAlmostEqual(m['idf1'], 14/15)

    def test_empty(self):
        self.assertEqual(evaluate([], [])['idf1'], 1)
        self.assertEqual(evaluate(self.gt, [])['idf1'], 0)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([self.gt[0], self.gt[0]], [])

    def test_global_id_permutation(self):
        p = self.gt.copy()
        p[:, 1] += 99
        self.assertEqual(evaluate(self.gt, p)['idf1'], 1)

    def test_nms(self):
        self.assertEqual(nms([[0,0,10,10], [0,0,10,10], [30,0,10,10]], [.9,.8,.7]), [0,2])

    def test_unmatched(self):
        self.assertEqual(match_iou([[0,0,10,10]], [[50,0,10,10]]), [])

    def test_lifetime(self):
        t = IoUTracker(max_age=1)
        first = t.update(1, [[0,0,10,10]])[0][1]
        t.update(2, [])
        self.assertEqual(t.update(3, [[0,0,10,10]])[0][1], first)
        t.update(4, [])
        t.update(5, [])
        self.assertNotEqual(t.update(6, [[0,0,10,10]])[0][1], first)


if __name__ == '__main__':
    unittest.main()
