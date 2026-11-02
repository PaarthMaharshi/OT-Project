"""Core checks. Run with `python -m pytest tests` or, without pytest, `python -m tests.test_core`."""
from __future__ import annotations

import numpy as np

from src.algorithms.nsga2 import NSGA2Config, run_nsga2
from src.baseline.greedy_dsatur import greedy
from src.evaluator import _gaps, check_domains, evaluate, violation_breakdown
from src.instance_generator import generate
from src.multiobjective.hypervolume import hypervolume
from src.multiobjective.non_dominated_sort import dominates, fast_non_dominated_sort
from src.problem import Instance, Problem


def test_gap_measure_matches_formula():
    # Periods 0..7, lunch at 4. Busy at 0, 2, 6 -> span 0..6 = 7, busy 3, lunch inside -> gap 3.
    b = np.zeros((1, 1, 8), dtype=bool)
    b[0, 0, [0, 2, 6]] = True
    assert _gaps(b, 4)[0, 0] == 3
    b[:] = False
    assert _gaps(b, 4)[0, 0] == 0  # empty day


def test_greedy_is_feasible_and_in_domain():
    for sc in ["demo", "small", "medium"]:
        p = Problem(generate(sc, 1))
        s = greedy(p)
        assert s.V == 0, (sc, violation_breakdown(p, s))
        assert not check_domains(p, s)


def test_violation_counts_a_clash():
    p = Problem(generate("demo"))
    s = greedy(p)
    bad = s.copy()
    # Put two events of the same group in the same slot -> at least one student clash.
    e1, e2 = 0, 1
    bad.slot[e2] = bad.slot[e1]
    evaluate(p, bad)
    assert violation_breakdown(p, bad)["student_clash"] >= 1 and bad.V > 0


def test_room_wastage_formula():
    p = Problem(generate("demo"))
    s = greedy(p)
    cap = p.room_cap[s.room]
    expected = (p.ev_len * (cap - p.ev_head) / cap).sum() / p.ev_len.sum()
    assert abs(s.F[1] - expected) < 1e-12


def test_nsga2_front_is_feasible_and_non_dominated():
    p = Problem(generate("small", 2))
    r = run_nsga2(p, NSGA2Config(pop_size=30, generations=8, seed=3))
    assert r.archive
    F = np.array([s.F for s in r.archive])
    assert all(s.V == 0 and not check_domains(p, s) for s in r.archive)
    assert len(fast_non_dominated_sort(F)[0]) == len(F)
    g = greedy(p)
    assert any(dominates(f, g.F) or np.allclose(f, g.F) for f in F)


def test_hypervolume_simple_cases():
    assert abs(hypervolume(np.array([[0.0, 0.0, 0.0]]), np.ones(3)) - 1.0) < 1e-12
    pts = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    # union of two boxes 0.5x1x1 and 1x0.5x1 = 0.5 + 0.5 - 0.25
    assert abs(hypervolume(pts, np.ones(3)) - 0.75) < 1e-12


def test_instance_round_trip():
    inst = generate("medium", 4)
    again = Instance.from_dict(inst.to_dict())
    assert again.to_dict() == inst.to_dict()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
