"""NSGA-III (no. 15): key rule = reference points + niching keep one solution per reference direction."""
import unittest

import numpy as np

from engine.algorithms import nsga3
from engine.generator import generate
from engine.problem import TimetableProblem


class TestNSGA3(unittest.TestCase):
    def test_das_dennis(self):
        W = nsga3.das_dennis(3, 4)
        self.assertEqual(len(W), 15)  # C(4 + 2, 2) = 15
        np.testing.assert_allclose(W.sum(axis=1), 1)
        self.assertEqual(len(nsga3.reference_points(100, 4)), 84)  # C(6 + 3, 3) = 84

    def test_association(self):
        refs = nsga3.das_dennis(2, 2)  # (0,1), (.5,.5), (1,0)
        niche, d = nsga3.associate(np.array([[0.0, 1.0], [0.1, 0.9], [1.0, 0.0]]), refs)
        # (0.1, 0.9) is 0.1 from the (0,1) line but 0.566 from the diagonal
        self.assertEqual(niche.tolist(), [0, 0, 2])
        self.assertAlmostEqual(d[1], 0.1)

    def test_niching_keeps_both_ends(self):
        # three points crowd the (0,1) direction, one sits on (1,0); keeping 3 must keep both ends
        F = np.array([[0.0, 1.0], [0.1, 0.9], [0.12, 0.88], [1.0, 0.0]])
        refs = nsga3.das_dennis(2, 2)
        for seed in range(5):
            kept = set(nsga3.survival(F, np.zeros(4), 3, refs, np.random.default_rng(seed)).tolist())
            self.assertTrue({0, 3} <= kept)

    def test_run_four_objectives(self):
        p = TimetableProblem(generate(1))
        r = nsga3.run(p, {"pop_size": 20, "evaluations": 600, "seed": 1})
        self.assertEqual(r.F.shape[1], 4)
        self.assertTrue(np.all(r.V == 0))


if __name__ == "__main__":
    unittest.main()
