"""SPEA2 (no. 16): key rule = fitness R(i) + D(i) from strengths, and archive truncation."""
import unittest

import numpy as np

from engine.algorithms import spea2
from engine.generator import generate
from engine.problem import TimetableProblem


class TestSPEA2(unittest.TestCase):
    def test_raw_fitness_by_hand(self):
        # Dominance: (2,2) dominates (3,3),(4,4); (1,4),(4,1),(3,3) each dominate (4,4).
        # Strengths S = [1, 2, 1, 1, 0]. R(3,3) = S(2,2) = 2; R(4,4) = 1+2+1+1 = 5; others 0.
        F = np.array([[1, 4], [2, 2], [4, 1], [3, 3], [4, 4]], float)
        fit = spea2.fitness(F, np.zeros(5), k=2)
        np.testing.assert_array_equal(np.floor(fit), [0, 0, 0, 2, 5])
        self.assertTrue(np.all(fit[:3] < 1))  # non-dominated <=> fitness < 1 (density is < 1/2)

    def test_truncation_removes_most_crowded(self):
        # the two middle points are almost on top of each other: one of them must go
        F = np.array([[0, 1], [0.5, 0.5], [0.51, 0.49], [1, 0]])
        kept = sorted(spea2.truncate(F, np.arange(4), 3).tolist())
        self.assertIn(0, kept)
        self.assertIn(3, kept)
        self.assertEqual(len(set(kept) & {1, 2}), 1)

    def test_run(self):
        p = TimetableProblem(generate(1), objectives=["student_gaps", "teacher_gaps"])
        r = spea2.run(p, {"pop_size": 20, "archive_size": 20, "evaluations": 600, "seed": 2})
        self.assertEqual(r.evaluations, 600)
        self.assertTrue(np.all(r.V == 0))


if __name__ == "__main__":
    unittest.main()
