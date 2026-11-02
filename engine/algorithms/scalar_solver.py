"""Single-objective simulated annealing used by the classical scalarisation methods
(Weighted Sum, epsilon-Constraint, Goal Programming, Lexicographic).

The timetable sub-problems are discrete, so scipy.optimize (continuous solvers) does not apply; this
hand-coded annealer solves each scalarised sub-problem instead.

Every sub-problem is written as a two-level key (penalty, value):
    penalty  hard-constraint violation V plus any extra constraint of the method
             (e.g. f2 > epsilon in the epsilon-constraint method). Must reach 0.
    value    the scalar the method minimises (weighted sum, deviation from goals, ...).

Acceptance rule ("feasibility first, then annealing"):
    * lower penalty            -> accept
    * higher penalty           -> reject
    * equal penalty            -> Boltzmann: accept with min(1, exp(-(value_new - value_old) / T))
Temperature: T0 = mean uphill value change of a few random neighbours / ln 2, geometric cooling to
T0 / 1000 over the sub-problem's budget.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..local_search import neighbour


def anneal(problem, key_fn: Callable, x0, budget: int, ctx, repair: bool = True, calibration: int = 10):
    """Minimise key_fn(F_selected, F_all, V, breakdown) -> (penalty, value) from x0.

    Returns (best_x, best_key, best_F_selected, best_F_all, best_V, evaluations_used).
    """
    rng = ctx.rng
    used = 0

    def ev(x):
        nonlocal used
        used += 1
        F, F_all, V, b = ctx.evaluate_full(x)
        return F, F_all, V, key_fn(F, F_all, V, b)

    x = np.array(x0, copy=True)
    F, F_all, V, k = ev(x)
    best = (x, k, F, F_all, V)

    # temperature calibration
    ups = []
    for _ in range(min(calibration, max(0, budget - 1))):
        y = neighbour(problem, x, rng)
        y = problem.repair(y, rng) if repair else y
        Fy, Fay, Vy, ky = ev(y)
        if ky[0] == k[0] and ky[1] > k[1]:
            ups.append(ky[1] - k[1])
        if ky < best[1]:
            best = (y, ky, Fy, Fay, Vy)
    T0 = (np.mean(ups) / math.log(2)) if ups else 0.05
    steps = max(1, budget - used)
    alpha = (1e-3) ** (1.0 / steps)
    T = T0
    while used < budget and ctx.stop_reason() is None:
        y = neighbour(problem, x, rng)
        y = problem.repair(y, rng) if repair else y
        Fy, Fay, Vy, ky = ev(y)
        if ky[0] < k[0]:
            accept = True
        elif ky[0] > k[0]:
            accept = False
        else:
            d = ky[1] - k[1]
            accept = d <= 0 or rng.random() < math.exp(-d / max(T, 1e-12))
        if accept:
            x, k, F, F_all, V = y, ky, Fy, Fay, Vy
            if k < best[1]:
                best = (x, k, F, F_all, V)
        T *= alpha
    bx, bk, bF, bFa, bV = best
    return bx, bk, bF, bFa, bV, used


def warm_start(problem, ctx, n: int = 20):
    """A few random repaired timetables: sets the objective normalisation (ctx.hv_nadir) and returns the
    best one (lowest V, then lowest sum of normalised objectives) as the first starting point."""
    rng = ctx.rng
    X = np.array([problem.repair(problem.random_solution(rng), rng) for _ in range(n)])
    F, V = ctx.evaluate(X)
    score = V * 1e6 + (F / ctx.hv_nadir).sum(axis=1)
    i = int(np.argmin(score))
    return X[i], F, V


def simplex_weights(m: int, at_least: int) -> np.ndarray:
    """At least `at_least` weight vectors on the simplex (evenly spaced lattice)."""
    from .nsga3 import das_dennis

    if m == 2:
        a = np.linspace(0, 1, max(at_least, 2))
        return np.column_stack([a, 1 - a])
    h = 1
    while len(das_dennis(m, h)) < at_least:
        h += 1
    return das_dennis(m, h)
