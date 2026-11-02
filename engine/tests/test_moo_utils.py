"""Unit tests for moo_utils. Every expected value can be checked by hand (worked out in the comments)."""
import unittest

import numpy as np

from engine import moo_utils as mu


class TestDominance(unittest.TestCase):
    def test_pareto_dominance(self):
        self.assertTrue(mu.dominates([1, 2], [2, 2]))  # better in f1, equal in f2
        self.assertFalse(mu.dominates([1, 2], [2, 1]))  # trade-off: neither dominates
        self.assertFalse(mu.dominates([2, 1], [1, 2]))
        self.assertFalse(mu.dominates([1, 1], [1, 1]))  # equal points do not dominate

    def test_deb_rules(self):
        # rule 1: feasible beats infeasible even with worse objectives
        self.assertTrue(mu.constrained_dominates([9, 9], [0, 0], 0, 3))
        self.assertFalse(mu.constrained_dominates([0, 0], [9, 9], 3, 0))
        # rule 2: both infeasible -> smaller violation wins
        self.assertTrue(mu.constrained_dominates([9, 9], [0, 0], 1, 2))
        # rule 3: both feasible -> Pareto dominance
        self.assertTrue(mu.constrained_dominates([1, 1], [2, 2], 0, 0))
        self.assertFalse(mu.constrained_dominates([1, 2], [2, 1], 0, 0))


class TestSorting(unittest.TestCase):
    def test_fast_non_dominated_sort(self):
        # (1,4),(2,2),(4,1) are mutually non-dominated; (3,3) is dominated by (2,2);
        # (4,4) is dominated by (3,3). So fronts: {0,1,2}, {3}, {4}.
        F = [[1, 4], [2, 2], [4, 1], [3, 3], [4, 4]]
        fronts = mu.fast_non_dominated_sort(F)
        self.assertEqual([sorted(f.tolist()) for f in fronts], [[0, 1, 2], [3], [4]])

    def test_constrained_sort_puts_feasible_first(self):
        # point 0 has the best objectives but violates constraints -> it drops behind feasible ones
        F = [[0, 0], [1, 2], [2, 1]]
        V = [5, 0, 0]
        fronts = mu.fast_non_dominated_sort(F, V)
        self.assertEqual(sorted(fronts[0].tolist()), [1, 2])
        self.assertEqual(fronts[1].tolist(), [0])

    def test_crowding_distance(self):
        # front sorted by f1: (1,4),(2,2),(4,1). Ends are infinite.
        # middle point: f1 gap (4-1)/(4-1)=1, f2 gap (4-1)/(4-1)=1 -> 2.0
        d = mu.crowding_distance([[1, 4], [2, 2], [4, 1]])
        self.assertTrue(np.isinf(d[0]) and np.isinf(d[2]))
        self.assertAlmostEqual(d[1], 2.0)


class TestIndicators(unittest.TestCase):
    def test_hv_2d(self):
        # staircase with ref (4,4): (1,3)->1x1=1, (2,2)->1x2=2, (3,1)->1x3=3, total 6
        self.assertAlmostEqual(mu.hv_2d([[1, 3], [2, 2], [3, 1]], [4, 4]), 6.0)
        # a dominated point adds nothing, a point outside the reference box adds nothing
        self.assertAlmostEqual(mu.hv_2d([[1, 3], [2, 2], [3, 1], [3, 3], [5, 0]], [4, 4]), 6.0)

    def test_hv_monte_carlo(self):
        # two boxes in the unit cube: 0.5x1x1 and 1x0.5x1, overlap 0.5x0.5x1 -> 0.5+0.5-0.25 = 0.75
        hv = mu.hv_monte_carlo([[0.5, 0, 0], [0, 0.5, 0]], [1, 1, 1], n_samples=200_000, seed=1)
        self.assertAlmostEqual(hv, 0.75, delta=0.01)
        # one point at the origin dominates the whole unit cube
        self.assertAlmostEqual(mu.hv_monte_carlo([[0, 0, 0]], [1, 1, 1], n_samples=10_000), 1.0)

    def test_igd(self):
        # A = {(0,0)}, Z = {(0,1),(1,0)}: both distances are 1 -> IGD = 1
        self.assertAlmostEqual(mu.igd([[0, 0]], [[0, 1], [1, 0]]), 1.0)
        self.assertAlmostEqual(mu.igd([[0, 1], [1, 0]], [[0, 1], [1, 0]]), 0.0)

    def test_spacing(self):
        # evenly spaced: every nearest-neighbour L1 distance is 2 -> spacing 0
        self.assertAlmostEqual(mu.spacing([[0, 2], [1, 1], [2, 0]]), 0.0)
        # (0,3),(1,2),(3,0): d = 2, 2, 4; mean 8/3; sum sq dev = 24/9; /(n-1)=12/9; sqrt = 1.1547
        self.assertAlmostEqual(mu.spacing([[0, 3], [1, 2], [3, 0]]), np.sqrt(4 / 3))

    def test_normalize(self):
        N = mu.normalize([[0, 10], [5, 20]], [0, 10], [5, 20])
        np.testing.assert_allclose(N, [[0, 0], [1, 1]])

    def test_merge_fronts(self):
        # (2,2) from run B dominates (3,3) from run A; duplicates (1,4) collapse
        ref = mu.merge_fronts([[[1, 4], [3, 3]], [[1, 4], [2, 2], [4, 1]]])
        self.assertEqual(sorted(map(tuple, ref.tolist())), [(1, 4), (2, 2), (4, 1)])


if __name__ == "__main__":
    unittest.main()
