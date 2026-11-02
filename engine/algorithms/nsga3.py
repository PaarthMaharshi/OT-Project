"""NSGA-III (catalogue no. 15), for many objectives. Deb & Jain, 2014 (constrained version: Jain & Deb, 2014).

Lecture steps
    1. Das-Dennis reference points on the unit simplex (p divisions; the largest set <= pop_size).
    2. Random population P (size N), evaluate.
    3. Mating: constrained binary tournament (feasible > infeasible, smaller violation, else random),
       uniform crossover + mutation (+ repair) -> N children Q.
    4. R = P + Q. Constrained non-dominated sort. Take whole fronts while they fit (set S_t);
       the front that does not fit is the "last front" F_l.
    5. Normalise S_t: subtract the ideal point; find one extreme point per axis (achievement scalarising
       function); the hyperplane through them gives the intercepts; divide by the intercepts.
    6. Associate every member of S_t with its nearest reference line (perpendicular distance).
    7. Niching: count members per reference point among the already chosen fronts; repeatedly take the
       reference point with the smallest count and add one member of F_l associated with it
       (the closest one if the count is 0, a random one otherwise); drop reference points with no
       candidates. Stop when N members are chosen.
    8. Repeat 3-7 until the budget is used.

Default parameters
    pop_size 100, crossover_rate 0.9, mutation_rate 1/S, repair True.
    Reference points: 4 objectives -> p = 6 -> 84 points; 2 objectives -> 100 points.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .. import moo_utils as mu
from ..operators import mutate, uniform_crossover
from .common import feasible_front, run_loop

NAME = "nsga3"
DEFAULTS = {"pop_size": 100, "crossover_rate": 0.9, "mutation_rate": None, "repair": True}


def das_dennis(m: int, p: int) -> np.ndarray:
    """All vectors with m components from {0, 1/p, ..., 1} that sum to 1 ("stars and bars")."""
    out = []
    for bars in combinations(range(p + m - 1), m - 1):
        prev, parts = -1, []
        for b in bars:
            parts.append(b - prev - 1)
            prev = b
        parts.append(p + m - 2 - prev)
        out.append(parts)
    return np.array(out, dtype=float) / p


def reference_points(n: int, m: int) -> np.ndarray:
    p = 1
    while len(das_dennis(m, p + 1)) <= n:
        p += 1
    return das_dennis(m, p)


def normalise(F: np.ndarray) -> np.ndarray:
    """Steps 5: translate by the ideal point, divide by hyperplane intercepts (fallback: max values)."""
    ideal = F.min(axis=0)
    Fp = F - ideal
    m = F.shape[1]
    weights = np.eye(m) + 1e-6
    asf = np.max(Fp[:, None, :] / weights[None, :, :], axis=2)  # ASF of every point for every axis
    extremes = Fp[np.argmin(asf, axis=0)]
    try:
        b = np.linalg.solve(extremes, np.ones(m))
        intercepts = 1.0 / b
        if np.any(~np.isfinite(intercepts)) or np.any(intercepts <= 1e-9):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercepts = Fp.max(axis=0)
    intercepts = np.where(intercepts > 1e-9, intercepts, 1.0)
    return Fp / intercepts


def associate(Fn: np.ndarray, refs: np.ndarray):
    """Step 6: nearest reference line and the perpendicular distance to it."""
    unit = refs / np.linalg.norm(refs, axis=1, keepdims=True)
    proj = Fn @ unit.T
    d = np.sqrt(np.maximum((Fn**2).sum(axis=1, keepdims=True) - proj**2, 0.0))
    niche = d.argmin(axis=1)
    return niche, d[np.arange(len(Fn)), niche]


def survival(F, V, n, refs, rng) -> np.ndarray:
    chosen: list[int] = []
    last = None
    for front in mu.fast_non_dominated_sort(F, V):
        if len(chosen) + len(front) <= n:
            chosen.extend(front.tolist())
        else:
            last = front
            break
    if last is None or len(chosen) == n:
        return np.array(chosen)
    if np.any(V[last] > 0):  # infeasible tier: equal violations, pick at random
        return np.array(chosen + rng.permutation(last)[: n - len(chosen)].tolist())
    pool = np.array(chosen + last.tolist())
    niche, dist = associate(normalise(F[pool]), refs)
    k = len(chosen)
    count = np.bincount(niche[:k], minlength=len(refs))
    candidates = list(range(k, len(pool)))
    active = np.ones(len(refs), dtype=bool)
    picked = []
    while len(picked) < n - k:
        live = np.flatnonzero(active)
        j = int(rng.choice(live[count[live] == count[live].min()]))
        members = [i for i in candidates if niche[i] == j]
        if not members:
            active[j] = False
            continue
        i = min(members, key=lambda i: dist[i]) if count[j] == 0 else int(rng.choice(members))
        picked.append(i)
        candidates.remove(i)
        count[j] += 1
    return np.array(chosen + [int(pool[i]) for i in picked])


def tournament(V, rng) -> int:
    a, b = rng.integers(len(V), size=2)
    if (V[a] == 0) != (V[b] == 0):
        return int(a if V[a] == 0 else b)
    if V[a] > 0 and V[a] != V[b]:
        return int(a if V[a] < V[b] else b)
    return int(a if rng.random() < 0.5 else b)


def initialize(problem, params, ctx):
    rng = ctx.rng
    refs = reference_points(params["pop_size"], problem.M)
    X = np.array([problem.random_solution(rng) for _ in range(params["pop_size"])])
    if params["repair"]:
        X = np.array([problem.repair(x, rng) for x in X])
    F, V = evaluate(problem, X, ctx)
    return {"X": X, "F": F, "V": V, "refs": refs, "params": params}


def evaluate(problem, X, ctx):
    return ctx.evaluate(X)


def step(problem, state, ctx):
    p, rng = state["params"], ctx.rng
    X, F, V = state["X"], state["F"], state["V"]
    n = min(len(X), ctx.remaining()) or len(X)
    kids = []
    while len(kids) < n:
        a, b = X[tournament(V, rng)], X[tournament(V, rng)]
        c1, c2 = uniform_crossover(problem, a, b, rng) if rng.random() < p["crossover_rate"] else (a.copy(), b.copy())
        for c in (c1, c2):
            c = mutate(problem, c, rng, p["mutation_rate"])
            kids.append(problem.repair(c, rng) if p["repair"] else c)
    Q = np.array(kids[:n])
    FQ, VQ = evaluate(problem, Q, ctx)
    RX, RF, RV = np.vstack([X, Q]), np.vstack([F, FQ]), np.concatenate([V, VQ])
    keep = survival(RF, RV, len(X), state["refs"], rng)
    return {**state, "X": RX[keep], "F": RF[keep], "V": RV[keep]}


def output(state):
    return feasible_front(state["X"], state["F"], state["V"])


def run(problem, params=None, progress=None, should_stop=None):
    import sys

    return run_loop(sys.modules[__name__], problem, params, progress, should_stop)
