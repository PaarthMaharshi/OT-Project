"""NSGA-II (no. 14): key rule = elitist survival by fronts then crowding, with a static penalty."""
import unittest

import numpy as np

from engine.algorithms import nsga2
from engine.generator import generate
from engine.problem import TimetableProblem

PEN = {**nsga2.DEFAULTS}  # penalty handling, penalty = 100


class TestNSGA2(unittest.TestCase):
    def test_survival_whole_fronts_then_crowding(self):
        # front 1 = {0,1,2}, point 3 = (3,3) is dominated. n=3 keeps exactly front 1.
        F = np.array([[1, 4], [2, 2], [4, 1], [3, 3]], float)
        V = np.zeros(4)
        self.assertEqual(sorted(nsga2.survival(F, V, 3, PEN).tolist()), [0, 1, 2])
        # n=2 must split front 1: the two boundary points have infinite crowding distance and win
        self.assertEqual(sorted(nsga2.survival(F, V, 2, PEN).tolist()), [0, 2])

    def test_static_penalty(self):
        # (0,0) with V=1 becomes (100,100) after the penalty, so the feasible (5,5) dominates it
        F = np.array([[0, 0], [5, 5]], float)
        V = np.array([1.0, 0.0])
        self.assertEqual(nsga2.survival(F, V, 1, PEN).tolist(), [1])

    def test_tournament_prefers_lower_rank(self):
        rng = np.random.default_rng(0)
        rank, crowd = np.array([0, 1]), np.array([0.0, np.inf])
        wins = [nsga2.tournament(rank, crowd, rng) for _ in range(200)]
        # index 1 can only win when it is drawn twice (prob 1/4)
        self.assertGreater(wins.count(0), 120)

    def test_run_budget_and_determinism(self):
        p = TimetableProblem(generate(1), objectives=["student_gaps", "teacher_gaps"])
        a = nsga2.run(p, {"pop_size": 20, "evaluations": 600, "seed": 3})
        b = nsga2.run(p, {"pop_size": 20, "evaluations": 600, "seed": 3})
        self.assertEqual(a.evaluations, 600)
        np.testing.assert_array_equal(a.F, b.F)
        self.assertEqual(a.stopped, "budget")


if __name__ == "__main__":
    unittest.main()
