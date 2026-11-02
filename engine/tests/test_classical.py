"""Classical methods: Weighted Sum (1), epsilon-Constraint (2), Goal Programming (6), Lexicographic (7)."""
import unittest

import numpy as np

from engine.algorithms import eps_constraint, goal_programming, lexicographic, weighted_sum
from engine.algorithms.scalar_solver import simplex_weights
from engine.generator import generate
from engine.problem import TimetableProblem

P2 = TimetableProblem(generate(1), objectives=["student_gaps", "teacher_gaps"])
SMALL = {"evaluations": 1500, "seed": 1}


class TestWeightedSum(unittest.TestCase):
    def test_weights(self):
        W = simplex_weights(2, 11)  # 0, 0.1, ..., 1
        self.assertEqual(len(W), 11)
        np.testing.assert_allclose(W.sum(axis=1), 1)
        self.assertGreaterEqual(len(simplex_weights(4, 10)), 10)

    def test_weighted_value(self):
        # w=(.25,.75), F=(4,2), nadir=(8,4): .25*.5 + .75*.5 = 0.5
        self.assertAlmostEqual(weighted_sum.weighted_value([4, 2], np.array([0.25, 0.75]), np.array([8.0, 4.0])), 0.5)

    def test_run(self):
        r = weighted_sum.run(P2, SMALL)
        self.assertEqual(r.stopped, "completed")
        self.assertTrue(np.all(r.V == 0))


class TestEpsilonConstraint(unittest.TestCase):
    def test_key(self):
        # primary f1; f2 = 5 exceeds eps 4 by 1, normalised by nadir 10 -> penalty 0.1; value 2/10
        self.assertEqual(eps_constraint.eps_key(np.array([2.0, 5.0]), 0, 0, np.array([0.0, 4.0]), np.array([10.0, 10.0])), (0.1, 0.2))

    def test_run_respects_epsilon(self):
        r = eps_constraint.run(P2, SMALL)
        levels = r.extra["eps_levels"]
        self.assertGreaterEqual(len(levels), 10)
        for e in levels:
            if e["eps_satisfied"]:
                self.assertLessEqual(e["objectives"][1], e["eps"][1] + 1e-9)


class TestGoalProgramming(unittest.TestCase):
    def test_deviation(self):
        # F=(3,5), goals=(2,6): d+ = (1, 0); w=(.5,.5), span=(1,1) -> 0.5 (being under a goal is free)
        self.assertAlmostEqual(goal_programming.deviation([3, 5], np.array([2.0, 6.0]), np.array([0.5, 0.5]), np.ones(2)), 0.5)

    def test_run(self):
        r = goal_programming.run(P2, SMALL)
        self.assertEqual(len(r.extra["goals"]), 2)
        self.assertTrue(np.all(r.V == 0))


class TestLexicographic(unittest.TestCase):
    def test_stage_key(self):
        order = ["teacher_clashes", "student_gaps", "teacher_gaps"]
        b = {"H1_teacher": 2, "H2_group": 1, "H3_room": 0, "H4_capacity": 0, "H5_sessions": 0}
        F_all = np.array([3.0, 7.0, 100.0, 4.0])
        # stage 2 (teacher_gaps): earlier optima (0, 1); teacher clashes 2 > 0 by 2, student gaps 3 > 1 by 2
        key = lexicographic.stage_key(order, 2, [0.0, 1.0], 0.0)(None, F_all, 3.0, b)
        self.assertEqual(key, (4.0, 7.0))

    def test_later_stages_never_worsen_earlier_criteria(self):
        r = lexicographic.run(P2, SMALL)
        stages = r.extra["stages"]
        final = stages[-1]["all_criteria"]
        for s in stages:
            self.assertLessEqual(final[s["criterion"]], s["optimum"] + 1e-9)
        self.assertEqual(len(r.F), 1)  # one timetable, not a front


if __name__ == "__main__":
    unittest.main()
