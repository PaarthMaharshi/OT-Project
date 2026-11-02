"""Constrained NSGA-II (no. 21): key rule = Deb's feasibility rules in survival, plus repair."""
import unittest

import numpy as np

from engine.algorithms import cnsga2, nsga2
from engine.generator import generate
from engine.problem import TimetableProblem

DEB = {**cnsga2.DEFAULTS}


class TestConstrainedNSGA2(unittest.TestCase):
    def test_feasible_beats_better_infeasible(self):
        # point 0 has the best objectives but V=2: rule 1 drops it in favour of the feasible points
        F = np.array([[0, 0], [5, 5], [6, 6]], float)
        V = np.array([2.0, 0.0, 0.0])
        self.assertEqual(sorted(nsga2.survival(F, V, 2, DEB).tolist()), [1, 2])

    def test_smaller_violation_wins_between_infeasible(self):
        # rule 2: both infeasible, objectives irrelevant, V=1 beats V=3
        F = np.array([[9, 9], [0, 0]], float)
        V = np.array([1.0, 3.0])
        self.assertEqual(nsga2.survival(F, V, 1, DEB).tolist(), [0])

    def test_run_returns_only_feasible_timetables(self):
        p = TimetableProblem(generate(1), objectives=["student_gaps", "teacher_gaps"])
        r = cnsga2.run(p, {"pop_size": 20, "evaluations": 800, "seed": 1})
        self.assertTrue(len(r.F) >= 1)
        self.assertTrue(np.all(r.V == 0))
        for x in r.X:  # re-check every returned timetable independently
            self.assertEqual(p.evaluate_full(x)[1], 0)


if __name__ == "__main__":
    unittest.main()
