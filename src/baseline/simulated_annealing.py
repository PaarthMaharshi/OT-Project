"""Weighted-sum simulated annealing (proposal Section 10), one run per weight vector.

Starts from the Greedy/DSatur timetable. A neighbour applies one or two move / swap / room / teacher
changes (the same moves as the GA mutation); clashes it creates are repaired, so the walk stays feasible.
Metropolis acceptance with geometric cooling; T0 is set so a typical uphill step is accepted half the time.
"""
from __future__ import annotations

import math

import numpy as np

from ..algorithms.nsga2 import mutate
from ..algorithms.scalarise import scalar, scales_from, weight_vectors
from ..evaluator import evaluate
from ..hybrid.repair import repair
from ..problem import Problem, Solution
from .greedy_dsatur import greedy


def _neighbour(prob: Problem, s: Solution, rng) -> tuple[Solution, int]:
    n = s.copy()
    mutate(prob, n, 1.5, rng)
    evaluate(prob, n)
    evals = 1
    if n.V > 0:
        n = repair(prob, n, rng)
        evals += 1
    return n, evals


def sa_single(prob: Problem, start: Solution, w, scales, budget: int, rng, final_ratio: float = 1e-3) -> tuple[Solution, int]:
    cur = start.copy()
    cur_v = scalar(cur.F, w, scales)
    best, best_v = cur, cur_v
    used = 0

    # Calibrate T0 from a few random uphill steps.
    ups = []
    for _ in range(10):
        n, e = _neighbour(prob, cur, rng)
        used += e
        if n.V == 0:
            d = scalar(n.F, w, scales) - cur_v
            if d > 0:
                ups.append(d)
    t0 = (np.mean(ups) / math.log(2)) if ups else 0.05
    steps = max(1, (budget - used) // 1)
    alpha = final_ratio ** (1 / steps)
    T = t0
    while used < budget:
        n, e = _neighbour(prob, cur, rng)
        used += e
        T *= alpha**e
        if n.V > 0:
            continue
        v = scalar(n.F, w, scales)
        if v <= cur_v or rng.random() < math.exp(-(v - cur_v) / max(T, 1e-12)):
            cur, cur_v = n, v
            if v < best_v:
                best, best_v = n, v
    return best, used


def run_simulated_annealing(prob: Problem, max_evaluations: int, seed: int = 0, n_weights: int | None = None) -> tuple[list[Solution], int]:
    rng = np.random.default_rng(seed)
    start = greedy(prob)
    scales = scales_from(start.F)
    W = weight_vectors(n_weights)
    per = max(20, max_evaluations // len(W))
    out, used = [], 0
    for w in W:
        s, n = sa_single(prob, start, w, scales, per, rng)
        out.append(s)
        used += n
    return out, used
