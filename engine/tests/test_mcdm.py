"""TOPSIS (8) and VIKOR (9): hand-computed examples (all criteria are costs)."""
import unittest

import numpy as np

from engine import mcdm


class TestTOPSIS(unittest.TestCase):
    def test_dominance_chain(self):
        # (1,1) < (2,2) < (3,3). Column norm sqrt(14); with equal weights alternative i (1..3) has
        # S+ = sqrt2 * .5 (i-1)/sqrt14 and S- = sqrt2 * .5 (3-i)/sqrt14, so C = (3-i)/2 = 1, .5, 0
        r = mcdm.topsis([[1, 1], [2, 2], [3, 3]], [0.5, 0.5])
        np.testing.assert_allclose(r["closeness"], [1.0, 0.5, 0.0])
        self.assertEqual(r["ranking"].tolist(), [0, 1, 2])

    def test_weights_decide_a_trade_off(self):
        # (0,10) vs (10,0): closeness reduces to C0 = w0, C1 = w1 -> the heavier criterion wins
        F = [[0, 10], [10, 0]]
        self.assertEqual(int(mcdm.topsis(F, [0.6, 0.4])["ranking"][0]), 0)
        self.assertEqual(int(mcdm.topsis(F, [0.4, 0.6])["ranking"][0]), 1)


class TestVIKOR(unittest.TestCase):
    def test_hand_example(self):
        # d = [[0,0],[.5,.5],[1,1]] x w=.5 -> S = [0,.5,1], R = [0,.25,.5]; v=.5 -> Q = [0,.5,1]
        r = mcdm.vikor([[1, 1], [2, 2], [3, 3]], [0.5, 0.5], v=0.5)
        np.testing.assert_allclose(r["S"], [0, 0.5, 1])
        np.testing.assert_allclose(r["R"], [0, 0.25, 0.5])
        np.testing.assert_allclose(r["Q"], [0, 0.5, 1])
        # DQ = 1/(3-1) = .5 and Q(a2)-Q(a1) = .5 -> acceptable advantage; a1 best in S and R -> stable
        self.assertTrue(r["acceptable_advantage"] and r["acceptable_stability"])
        self.assertEqual(r["compromise_set"], [0])

    def test_v_changes_the_balance(self):
        # B is best on S (sum), A is best on R (max regret): v=1 trusts S, v=0 trusts R
        F = [[0, 6, 6], [4, 4, 4], [10, 0, 0]]
        w = [1 / 3, 1 / 3, 1 / 3]
        self.assertEqual(int(mcdm.vikor(F, w, v=0.0)["ranking"][0]), 1)  # smallest max regret
        S = mcdm.vikor(F, w, v=1.0)["S"]
        self.assertEqual(int(mcdm.vikor(F, w, v=1.0)["ranking"][0]), int(np.argmin(S)))


class TestSensitivity(unittest.TestCase):
    def test_flip_detected(self):
        # w = (.52,.48) picks alternative 0; lowering w0 by .1 -> (.467,.533) picks alternative 1
        res = mcdm.weight_sensitivity([[0, 10], [10, 0]], [0.52, 0.48], "topsis", 0.1)
        lower_w0 = [r for r in res if r["criterion"] == 0 and r["shift"] < 0][0]
        self.assertTrue(lower_w0["top_changed"])
        self.assertEqual(lower_w0["top"], 1)


if __name__ == "__main__":
    unittest.main()
