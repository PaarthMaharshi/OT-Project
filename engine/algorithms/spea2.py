"""SPEA2 (catalogue no. 16). Zitzler, Laumanns & Thiele, 2001.

Lecture steps
    1. Random population P (size N), empty archive A (size N_bar).
    2. Fitness of every member of U = P + A:
         strength  S(i) = number of solutions i dominates
         raw       R(i) = sum of S(j) over all j that dominate i     (0 for non-dominated)
         density   D(i) = 1 / (sigma_k(i) + 2), sigma_k = distance to the k-th nearest neighbour
                          in normalised objective space, k = sqrt(N + N_bar)
         fitness   F(i) = R(i) + D(i)                                  (< 1 means non-dominated)
    3. Environmental selection: the new archive takes every member with F < 1.
         too few  -> fill with the best remaining by fitness
         too many -> truncation: repeatedly delete the member whose sorted distance list to the others
                     is lexicographically smallest (the most crowded one)
    4. Mating: binary tournament on fitness in the archive; crossover + mutation (+ repair) give the new P.
    5. Repeat 2-4 until the budget is used. The archive is the output.

Constraints: "dominates" in step 2 is Deb's constrained dominance, so feasible solutions always get a
better raw fitness than infeasible ones.

Default parameters
    pop_size 100, archive_size 100, k = sqrt(pop_size + archive_size), crossover_rate 0.9,
    mutation_rate 1/S, repair True.
"""
from __future__ import annotations

import numpy as np

from .. import moo_utils as mu
from ..operators import mutate, uniform_crossover
from .common import feasible_front, run_loop

NAME = "spea2"
DEFAULTS = {"pop_size": 100, "archive_size": 100, "k": None, "crossover_rate": 0.9, "mutation_rate": None, "repair": True}


def fitness(F, V, k) -> np.ndarray:
    """SPEA2 fitness R + D for every row of (F, V)."""
    n = len(F)
    Dm = mu.dominance_matrix(F, V)  # Dm[i, j]: i dominates j
    S = Dm.sum(axis=1)  # strength
    R = Dm.T.astype(float) @ S  # R[i] = sum of S[j] for j dominating i
    Fn = mu.normalize(F, F.min(axis=0), F.max(axis=0))
    dist = np.sqrt(((Fn[:, None, :] - Fn[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    kk = min(k, n - 1) if n > 1 else 0
    sigma = np.sort(dist, axis=1)[:, kk - 1] if n > 1 else np.zeros(n)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    return R + 1.0 / (sigma + 2.0)


def truncate(F, idx: np.ndarray, size: int) -> np.ndarray:
    """Remove the most crowded members one by one until `size` remain."""
    idx = list(idx)
    Fn = mu.normalize(F[idx], F[idx].min(axis=0), F[idx].max(axis=0))
    dist = np.sqrt(((Fn[:, None, :] - Fn[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    alive = list(range(len(idx)))
    while len(alive) > size:
        sub = np.sort(dist[np.ix_(alive, alive)], axis=1)  # each row: distances to the others, ascending
        # lexicographically smallest row = most crowded member
        worst = min(range(len(alive)), key=lambda r: tuple(sub[r]))
        alive.pop(worst)
    return np.array([idx[a] for a in alive])


def environmental_selection(F, V, size, k) -> tuple[np.ndarray, np.ndarray]:
    fit = fitness(F, V, k)
    nd = np.flatnonzero(fit < 1)
    if len(nd) == size:
        chosen = nd
    elif len(nd) < size:
        chosen = np.argsort(fit, kind="stable")[:size]
    else:
        chosen = truncate(F, nd, size)
    return chosen, fit[chosen]


def initialize(problem, params, ctx):
    rng = ctx.rng
    params = {**params, "k": params["k"] or int(round(np.sqrt(params["pop_size"] + params["archive_size"])))}
    X = np.array([problem.random_solution(rng) for _ in range(params["pop_size"])])
    if params["repair"]:
        X = np.array([problem.repair(x, rng) for x in X])
    F, V = evaluate(problem, X, ctx)
    keep, fit = environmental_selection(F, V, params["archive_size"], params["k"])
    return {"X": X[keep], "F": F[keep], "V": V[keep], "fit": fit, "params": params}


def evaluate(problem, X, ctx):
    return ctx.evaluate(X)


def step(problem, state, ctx):
    p, rng = state["params"], ctx.rng
    A, FA, VA, fit = state["X"], state["F"], state["V"], state["fit"]

    def pick():  # binary tournament on fitness (lower is better)
        a, b = rng.integers(len(A), size=2)
        return A[a] if fit[a] <= fit[b] else A[b]

    n = min(p["pop_size"], ctx.remaining()) or p["pop_size"]
    kids = []
    while len(kids) < n:
        c1, c2 = uniform_crossover(problem, pick(), pick(), rng) if rng.random() < p["crossover_rate"] else (pick().copy(), pick().copy())
        for c in (c1, c2):
            c = mutate(problem, c, rng, p["mutation_rate"])
            kids.append(problem.repair(c, rng) if p["repair"] else c)
    Q = np.array(kids[:n])
    FQ, VQ = evaluate(problem, Q, ctx)
    UX, UF, UV = np.vstack([Q, A]), np.vstack([FQ, FA]), np.concatenate([VQ, VA])
    keep, fit = environmental_selection(UF, UV, p["archive_size"], p["k"])
    return {**state, "X": UX[keep], "F": UF[keep], "V": UV[keep], "fit": fit}


def output(state):
    return feasible_front(state["X"], state["F"], state["V"])


def run(problem, params=None, progress=None, should_stop=None):
    import sys

    return run_loop(sys.modules[__name__], problem, params, progress, should_stop)
