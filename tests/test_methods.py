"""Checks for the comparison methods. Run with `python -m pytest tests` or `python -m tests.test_methods`."""
from __future__ import annotations

import numpy as np

from experiments.run_experiments import a12
from src.algorithms import cpsat_reference
from src.algorithms.nsga3 import das_dennis, reference_directions
from src.evaluator import check_domains
from src.instance_generator import generate
from src.methods import ABLATION, run_method
from src.problem import Problem

PROB = Problem(generate("small", 1))


def _valid(points, prob=PROB):
    return all(s.V == 0 and not check_domains(prob, s) for s in points)


def test_das_dennis_counts_and_sums():
    W = das_dennis(3, 4)
    assert len(W) == 15  # C(4+2, 2)
    assert np.allclose(W.sum(axis=1), 1)
    assert len(reference_directions(60)) <= 60


def test_every_method_returns_valid_timetables_within_budget():
    budget = 300
    for m in ["greedy", "random_search", "weighted_ga", "simulated_annealing", "nsga2", "nsga3", *ABLATION]:
        r = run_method(m, PROB, budget, seed=2, pop_size=20)
        assert _valid(r.points), m
        if m not in ("greedy",):
            # one generation / one weight's worth of slack over the budget at most
            assert r.evaluations <= budget + 2 * 20 + 10, (m, r.evaluations)
    # methods with repair always find clash-free timetables
    for m in ["random_search", "weighted_ga", "simulated_annealing", "nsga2", "nsga3"]:
        assert run_method(m, PROB, budget, seed=3, pop_size=20).points, m


def test_same_seed_same_result():
    a = run_method("nsga2", PROB, 300, seed=5, pop_size=20)
    b = run_method("nsga2", PROB, 300, seed=5, pop_size=20)
    assert sorted(tuple(s.F) for s in a.points) == sorted(tuple(s.F) for s in b.points)


def test_a12():
    assert a12([2, 3], [1, 1]) == 1.0
    assert a12([1], [1]) == 0.5
    assert a12([0], [1]) == 0.0


def test_cpsat_reference_is_feasible():
    if not cpsat_reference.available():
        print("  (skipped: OR-Tools not installed)")
        return
    demo = Problem(generate("demo"))
    r = run_method("cpsat", demo, 0, seed=0, cpsat_seconds=10)
    assert r.points and _valid(r.points, demo)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
