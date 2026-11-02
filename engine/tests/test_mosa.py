"""MOSA (no. 26): key rule = Boltzmann acceptance and geometric cooling; archive stays non-dominated."""
import math
import unittest

import numpy as np

from engine import moo_utils as mu
from engine.algorithms import mosa
from engine.generator import generate
from engine.problem import TimetableProblem


class TestMOSA(unittest.TestCase):
    def test_boltzmann_acceptance(self):
        self.assertEqual(mosa.accept_probability(-1.0, 1.0), 1.0)  # improvements always accepted
        self.assertAlmostEqual(mosa.accept_probability(1.0, 1.0), math.exp(-1))  # e^(-dE/T)
        self.assertAlmostEqual(mosa.accept_probability(1.0, 2.0), math.exp(-0.5))
        self.assertLess(mosa.accept_probability(1.0, 1e-3), 1e-100)  # cold: worse moves rejected

    def test_energy_change(self):
        # w=(.5,.5), nadir=(10,4): .5*(2/10) + .5*(-2/4) + (violation change 1) = 0.1 - 0.25 + 1 = 0.85
        dE = mosa.energy_change(np.array([5.0, 3.0]), 0, np.array([7.0, 1.0]), 1, np.array([0.5, 0.5]), np.array([10.0, 4.0]))
        self.assertAlmostEqual(dE, 0.85)

    def test_cooling_and_archive(self):
        p = TimetableProblem(generate(1), objectives=["student_gaps", "teacher_gaps"])
        r = mosa.run(p, {"iterations_per_temperature": 50, "cooling_rate": 0.9, "evaluations": 521, "seed": 1})
        # 1 start + 20 calibration moves + 10 temperature levels x 50 proposals = 521 evaluations
        self.assertEqual(r.evaluations, 521)
        self.assertTrue(np.all(r.V == 0))
        self.assertTrue(mu.nondominated_mask(r.F).all())  # the archive is mutually non-dominated

    def test_temperature_falls_geometrically(self):
        from engine.algorithms.common import Context, merged_params

        p = TimetableProblem(generate(1), objectives=["student_gaps", "teacher_gaps"])
        params = merged_params(mosa, {"iterations_per_temperature": 5, "cooling_rate": 0.8, "seed": 1})
        ctx = Context(p, params)
        state = mosa.initialize(p, params, ctx)
        t0 = state["T"]
        state = mosa.step(p, state, ctx)
        state = mosa.step(p, state, ctx)
        self.assertAlmostEqual(state["T"], t0 * 0.8**2)


if __name__ == "__main__":
    unittest.main()
