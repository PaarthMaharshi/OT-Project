"""Tests for the problem model, repair, operators and generator, on a tiny instance checkable by hand."""
import copy
import unittest

import numpy as np

from engine.generator import generate
from engine.operators import mutate, uniform_crossover
from engine.problem import TimetableProblem, validate_instance

# 1 day x 4 slots. Rooms R1 (30 seats) and R2 (60). Teacher T1 is unavailable in slot 3.
# Sessions: s0, s1 = L1 (T1, G1); s2 = L2 (T2, G1); s3 = L3 (T1, G2).
TINY = {
    "days": 1,
    "slots_per_day": 4,
    "rooms": [{"id": "R1", "capacity": 30}, {"id": "R2", "capacity": 60}],
    "teachers": [{"id": "T1", "unavailable_slots": [3]}, {"id": "T2", "unavailable_slots": []}],
    "groups": [{"id": "G1", "size": 50}, {"id": "G2", "size": 20}],
    "lectures": [
        {"id": "L1", "course": "A", "teacher_id": "T1", "group_id": "G1", "sessions_per_week": 2},
        {"id": "L2", "course": "B", "teacher_id": "T2", "group_id": "G1", "sessions_per_week": 1},
        {"id": "L3", "course": "C", "teacher_id": "T1", "group_id": "G2", "sessions_per_week": 1},
    ],
}


class TestEvaluation(unittest.TestCase):
    def test_hand_computed_violations_and_objectives(self):
        p = TimetableProblem(TINY)
        # slots [0, 2, 0, 3], rooms [R2, R2, R1, R1]
        x = np.array([0, 2, 0, 3, 1, 1, 0, 0])
        F, V, b = p.evaluate_full(x)
        # H1: s3 (teacher T1) sits in T1's unavailable slot 3 -> 1
        # H2: G1 has s0 and s2 both in slot 0 -> 1
        # H3: no room is used twice in a slot -> 0
        # H4: s2 puts G1 (50) into R1 (30) -> 1
        self.assertEqual(b, {"H1_teacher": 1, "H2_group": 1, "H3_room": 0, "H4_capacity": 1, "H5_sessions": 0})
        self.assertEqual(V, 3)
        # f1: G1 busy in 0 and 2 -> span 3 - busy 2 = 1 gap; G2 only slot 3 -> 0
        # f2: T1 busy 0, 2, 3 -> span 4 - busy 3 = 1 gap; T2 one slot -> 0; no day with > 4 sessions
        # f3: s0, s1: 60-50=10 each; s2: room too small -> 0; s3: 30-20=10  -> 30
        # f4: sessions in slot 0 or slot 3: s0, s2, s3 -> 3
        np.testing.assert_array_equal(F, [1, 1, 30, 3])

    def test_objective_subset(self):
        p = TimetableProblem(TINY, objectives=["student_gaps", "teacher_gaps"])
        F, V = p.evaluate(np.array([0, 2, 0, 3, 1, 1, 0, 0]))
        np.testing.assert_array_equal(F, [1, 1])

    def test_overload_day_counts_in_f2(self):
        inst = copy.deepcopy(TINY)
        inst["slots_per_day"] = 6
        inst["groups"].append({"id": "G3", "size": 10})
        inst["lectures"].append({"id": "L4", "course": "D", "teacher_id": "T2", "group_id": "G2", "sessions_per_week": 2})
        inst["lectures"].append({"id": "L5", "course": "E", "teacher_id": "T2", "group_id": "G3", "sessions_per_week": 2})
        p = TimetableProblem(inst)
        # T2 teaches s2 (L2) + 2 x L4 + 2 x L5 = 5 sessions, all on the one day, in slots 0..4 (no gaps)
        slots = np.array([5, 5, 0, 5, 1, 2, 3, 4])  # L1 x2 and L3 parked in slot 5 (clashes do not matter here)
        rooms = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        F, _, _ = p.evaluate_full(np.concatenate([slots, rooms]))
        # T2: 5 sessions > 4 -> one overload day; T1: s0, s1, s3 all in slot 5 -> no gap
        self.assertEqual(F[1], 1)


class TestValidation(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(validate_instance(TINY), [])
        self.assertEqual(validate_instance(generate(1)), [])

    def test_unknown_ids(self):
        bad = copy.deepcopy(TINY)
        bad["lectures"][0]["teacher_id"] = "T9"
        bad["lectures"][1]["group_id"] = "G9"
        errs = validate_instance(bad)
        self.assertTrue(any("unknown teacher_id 'T9'" in e for e in errs))
        self.assertTrue(any("unknown group_id 'G9'" in e for e in errs))

    def test_group_larger_than_every_room(self):
        bad = copy.deepcopy(TINY)
        bad["groups"][0]["size"] = 500
        self.assertTrue(any("largest room holds only 60" in e for e in validate_instance(bad)))

    def test_more_sessions_than_slots(self):
        bad = copy.deepcopy(TINY)
        bad["lectures"][0]["sessions_per_week"] = 9  # G1 now needs 10 sessions in a 4-slot week
        errs = validate_instance(bad)
        self.assertTrue(any("week has only 4 slots" in e for e in errs))
        self.assertTrue(any("Teacher 'T1'" in e for e in errs))

    def test_bad_shapes(self):
        self.assertTrue(validate_instance([]))
        self.assertTrue(any("'days'" in e for e in validate_instance({**TINY, "days": 0})))
        self.assertTrue(any("outside" in e for e in validate_instance({**TINY, "teachers": [{"id": "T1", "unavailable_slots": [99]}, {"id": "T2"}]})))


class TestRepairAndOperators(unittest.TestCase):
    def test_repair_on_tight_tiny_instance(self):
        # TINY is deliberately tight (G1's three sessions only fit in R2, T1 is blocked in slot 3).
        # Greedy repair is a heuristic: it does not always succeed here (measured 70-81 % with 3 passes),
        # but it must never make things worse and must succeed most of the time.
        p = TimetableProblem(TINY)
        rng = np.random.default_rng(0)
        ok = 0
        for _ in range(200):
            x0 = p.random_solution(rng)
            x = p.repair(x0, rng)
            self.assertLessEqual(p.evaluate(x)[1], p.evaluate(x0)[1])
            ok += p.evaluate(x)[1] == 0
        self.assertGreaterEqual(ok, 140)  # measured 70-81 % across seeds
        # the hand-solved feasible timetable stays untouched (repair leaves feasible solutions alone)
        good = np.array([0, 1, 2, 2, 1, 1, 1, 0])
        self.assertEqual(p.evaluate(good)[1], 0)
        np.testing.assert_array_equal(p.repair(good, rng), good)

    def test_repair_on_lab_instance(self):
        p = TimetableProblem(generate(1))
        rng = np.random.default_rng(1)
        ok = sum(p.evaluate(p.repair(p.random_solution(rng), rng))[1] == 0 for _ in range(100))
        self.assertGreaterEqual(ok, 95)  # repair almost always reaches feasibility from random

    def test_operators_keep_genes_in_range(self):
        p = TimetableProblem(generate(2))
        rng = np.random.default_rng(3)
        a, b = p.random_solution(rng), p.random_solution(rng)
        c1, c2 = uniform_crossover(p, a, b, rng)
        for c in (c1, c2, mutate(p, c1, rng, rate=0.2)):
            s, r = p.split(c)
            self.assertTrue(((s >= 0) & (s < p.T)).all() and ((r >= 0) & (r < p.R)).all())
        # crossover keeps each session's (slot, room) pair together
        for s in range(p.S):
            self.assertIn((c1[s], c1[p.S + s]), {(a[s], a[p.S + s]), (b[s], b[p.S + s])})

    def test_generator_is_seeded(self):
        self.assertEqual(generate(4), generate(4))
        g = generate(1)
        self.assertEqual((len(g["groups"]), len(g["teachers"]), len(g["rooms"]), len(g["lectures"])), (5, 10, 8, 25))


if __name__ == "__main__":
    unittest.main()
