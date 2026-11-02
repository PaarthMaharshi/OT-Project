"""MOMA (no. 30): key rule = hill climbing never makes a child worse, and it spends budget."""
import unittest

import numpy as np

from engine.algorithms import cnsga2, moma
from engine.algorithms.common import Context, merged_params
from engine.generator import generate
from engine.local_search import hill_climb, scalar
from engine.problem import TimetableProblem


class TestMOMA(unittest.TestCase):
    def test_hill_climb_is_monotone(self):
        p = TimetableProblem(generate(1), objectives=["student_gaps", "teacher_gaps"])
        ctx = Context(p, merged_params(moma, {"evaluations": 10_000, "seed": 4}))
        rng = ctx.rng
        w = np.array([0.5, 0.5])
        for _ in range(10):
            x = p.repair(p.random_solution(rng), rng)
            F, V = ctx.evaluate(x[None])
            x2, f2, v2 = hill_climb(p, x, F[0], V[0], 15, ctx, w=w)
            self.assertLessEqual(v2, V[0])
            if v2 == V[0]:
                self.assertLessEqual(scalar(f2, w, ctx.hv_nadir), scalar(F[0], w, ctx.hv_nadir))

    def test_local_search_uses_budget(self):
        p = TimetableProblem(generate(1), objectives=["student_gaps", "teacher_gaps"])
        a = moma.run(p, {"pop_size": 20, "evaluations": 1000, "seed": 1})
        b = cnsga2.run(p, {"pop_size": 20, "evaluations": 1000, "seed": 1})
        self.assertLess(len(a.history), len(b.history))  # fewer generations: LS evaluations count
        self.assertTrue(np.all(a.V == 0))


if __name__ == "__main__":
    unittest.main()
