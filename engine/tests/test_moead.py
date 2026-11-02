"""MOEA/D (no. 18): key rule = Tchebycheff scalarisation and neighbourhood replacement."""
import unittest

import numpy as np

from engine.algorithms import moead
from engine.generator import generate
from engine.problem import TimetableProblem


class TestMOEAD(unittest.TestCase):
    def test_tchebycheff_by_hand(self):
        # f=(2,1), w=(.5,.5), z*=(0,0), z_nad=(4,2): max(.5*2/4, .5*1/2) = 0.25
        g = moead.tchebycheff(np.array([2.0, 1.0]), np.array([0.5, 0.5]), np.zeros(2), np.array([4.0, 2.0]))
        self.assertAlmostEqual(float(g), 0.25)

    def test_weights_and_neighbourhoods(self):
        W = moead.weight_vectors(5, 2)  # (0,1), (.25,.75), (.5,.5), (.75,.25), (1,0) with a 1e-6 floor
        np.testing.assert_allclose(W[:, 0], [1e-6, 0.25, 0.5, 0.75, 1.0], atol=1e-6)
        B = moead.neighbourhoods(W, 3)
        self.assertEqual(B[0].tolist(), [0, 1, 2])  # itself and its two nearest
        self.assertEqual(sorted(B[2].tolist()), [1, 2, 3])
        self.assertEqual(len(moead.weight_vectors(100, 4)), 84)  # 4-D lattice with p = 6

    def test_run(self):
        p = TimetableProblem(generate(1), objectives=["student_gaps", "teacher_gaps"])
        r = moead.run(p, {"pop_size": 20, "neighbourhood": 5, "evaluations": 600, "seed": 2})
        self.assertEqual(r.evaluations, 600)
        self.assertTrue(np.all(r.V == 0))


if __name__ == "__main__":
    unittest.main()
