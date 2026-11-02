"""MOEA/D with Tchebycheff subproblems (catalogue no. 18). Zhang & Li, 2007.

Lecture steps
    1. Generate N evenly spread weight vectors lambda_1..lambda_N (simplex lattice).
    2. Neighbourhood B(i) = the T weight vectors closest to lambda_i (Euclidean).
    3. Random population x_1..x_N (one per subproblem), evaluate, ideal point z* = component-wise min.
    4. For each subproblem i:
         a. pick two parents from B(i) (with prob. delta, otherwise from the whole population),
         b. crossover + mutation (+ repair) -> child y, evaluate,
         c. update z* with f(y),
         d. for neighbours j in B(i) (random order): if y is better than x_j for subproblem j,
            replace x_j (at most nr replacements),
         e. add y to the external population EP if it is feasible and non-dominated.
    5. Repeat 4 until the budget is used. EP is the output.

Tchebycheff:  g(x | lambda, z*) = max_j lambda_j * |f_j(x) - z*_j| / (z_nad_j - z*_j)
    Objectives are normalised by the current population's range because room waste is measured in
    hundreds of seats while gaps are single digits.
Constraints: y is "better" than x_j if it has a smaller violation; with equal violation, smaller g wins.

Default parameters
    pop_size 100 (number of subproblems; with M > 2 the lattice may give slightly fewer),
    neighbourhood T 20, delta 0.9, nr 2, crossover_rate 1.0, mutation_rate 1/S, repair True.
"""
from __future__ import annotations

import numpy as np

from .. import moo_utils as mu
from ..operators import mutate, uniform_crossover
from .common import feasible_front, run_loop
from .nsga3 import das_dennis

NAME = "moead"
DEFAULTS = {"pop_size": 100, "neighbourhood": 20, "delta": 0.9, "nr": 2, "crossover_rate": 1.0, "mutation_rate": None, "repair": True}


def weight_vectors(n: int, m: int) -> np.ndarray:
    """Simplex-lattice weights: for m = 2 exactly n evenly spaced vectors, otherwise the largest
    lattice with at most n vectors. Zeros are replaced by 1e-6 so every objective counts a little."""
    if m == 2:
        a = np.linspace(0, 1, n)
        W = np.column_stack([a, 1 - a])
    else:
        h = 1
        while len(das_dennis(m, h + 1)) <= n:
            h += 1
        W = das_dennis(m, h)
    return np.maximum(W, 1e-6)


def neighbourhoods(W: np.ndarray, T: int) -> np.ndarray:
    """B[i] = indices of the T weight vectors closest to W[i] (W[i] itself first)."""
    dist = np.sqrt(((W[:, None, :] - W[None, :, :]) ** 2).sum(axis=2))
    return np.argsort(dist, axis=1, kind="stable")[:, : min(T, len(W))]


def tchebycheff(F, w, z, znad) -> np.ndarray:
    span = np.where(znad - z > 1e-12, znad - z, 1.0)
    return np.max(w * np.abs(F - z) / span, axis=-1)


def initialize(problem, params, ctx):
    rng = ctx.rng
    W = weight_vectors(params["pop_size"], problem.M)
    N = len(W)
    B = neighbourhoods(W, params["neighbourhood"])  # includes i itself (distance 0)
    X = np.array([problem.random_solution(rng) for _ in range(N)])
    if params["repair"]:
        X = np.array([problem.repair(x, rng) for x in X])
    F, V = evaluate(problem, X, ctx)
    EPX, EPF, EPV = feasible_front(X, F, V) if (V == 0).any() else (X[:0], F[:0], V[:0])
    return {"W": W, "B": B, "X": X, "F": F, "V": V, "z": F.min(axis=0), "EPX": EPX, "EPF": EPF, "EPV": EPV, "params": params}


def evaluate(problem, X, ctx):
    return ctx.evaluate(X)


def _update_ep(state, y, fy):
    """Keep EP = feasible non-dominated solutions seen so far."""
    EPF = state["EPF"]
    if len(EPF) and (any(mu.dominates(f, fy) for f in EPF) or any(np.array_equal(f, fy) for f in EPF)):
        return
    keep = np.array([not mu.dominates(fy, f) for f in EPF], dtype=bool) if len(EPF) else np.zeros(0, bool)
    state["EPX"] = np.vstack([state["EPX"][keep], y[None]])
    state["EPF"] = np.vstack([EPF[keep], fy[None]])
    state["EPV"] = np.concatenate([state["EPV"][keep], [0.0]])


def step(problem, state, ctx):
    p, rng = state["params"], ctx.rng
    W, B, X, F, V = state["W"], state["B"], state["X"], state["F"], state["V"]
    N = len(W)
    for i in rng.permutation(N):
        if ctx.remaining() <= 0:
            break
        pool = B[i] if rng.random() < p["delta"] else np.arange(N)
        a, b = rng.choice(pool, size=2, replace=False)
        y = uniform_crossover(problem, X[a], X[b], rng)[0] if rng.random() < p["crossover_rate"] else X[a].copy()
        y = mutate(problem, y, rng, p["mutation_rate"])
        if p["repair"]:
            y = problem.repair(y, rng)
        Fy, Vy = evaluate(problem, y[None], ctx)
        fy, vy = Fy[0], Vy[0]
        state["z"] = np.minimum(state["z"], fy)
        znad = F.max(axis=0)
        replaced = 0
        for j in rng.permutation(pool):
            better = vy < V[j] or (vy == V[j] and tchebycheff(fy, W[j], state["z"], znad) <= tchebycheff(F[j], W[j], state["z"], znad))
            if better:
                X[j], F[j], V[j] = y, fy, vy
                replaced += 1
                if replaced >= p["nr"]:
                    break
        if vy == 0:
            _update_ep(state, y, fy)
    return state


def output(state):
    if len(state["EPF"]):
        return feasible_front(state["EPX"], state["EPF"], state["EPV"])
    return feasible_front(state["X"], state["F"], state["V"])


def run(problem, params=None, progress=None, should_stop=None):
    import sys

    return run_loop(sys.modules[__name__], problem, params, progress, should_stop)
