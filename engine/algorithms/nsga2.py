"""NSGA-II (catalogue no. 14), the baseline. Deb, Pratap, Agarwal & Meyarivan, 2002.

Lecture steps
    1. Create a random population P of size N and evaluate it.
    2. Sort P into non-dominated fronts and compute the crowding distance inside each front.
    3. Binary tournament: the lower front wins; on a tie the larger crowding distance wins.
    4. Uniform crossover (prob. pc) and mutation (swap / slot-reassign / room-reassign) give N children Q.
    5. Evaluate Q. Merge R = P + Q (2N solutions).
    6. Survival: copy whole fronts of R into the next P until the next front does not fit; fill the
       remaining places from that front by descending crowding distance.
    7. Repeat 2-6 until the evaluation budget is used.

Constraint handling in this BASELINE: static penalty. Each objective becomes f_j + penalty * V, and
plain Pareto dominance is applied to the penalised values. No repair. (Constrained NSGA-II, no. 21,
replaces this by Deb's feasibility rules and adds repair.)

Default parameters
    pop_size 100, crossover_rate 0.9, mutation_rate 1/S (one mutation per child on average),
    penalty 100 per violation, repair False, constraint_handling "penalty".
"""
from __future__ import annotations

import numpy as np

from .. import moo_utils as mu
from ..operators import mutate, uniform_crossover
from .common import feasible_front, run_loop

NAME = "nsga2"
DEFAULTS = {
    "pop_size": 100,
    "crossover_rate": 0.9,
    "mutation_rate": None,  # None -> 1/S
    "penalty": 100.0,
    "repair": False,
    "constraint_handling": "penalty",  # "penalty" (this baseline) or "deb" (used by Constrained NSGA-II)
}


# ---------- helpers shared with cnsga2 / moma ----------
def sorting_inputs(F, V, params):
    """What the sort compares: penalised objectives (penalty) or raw objectives + V (Deb's rules)."""
    if params["constraint_handling"] == "penalty":
        return F + params["penalty"] * V[:, None], None
    return F, V


def rank_and_crowding(F, V, params):
    Fs, Vs = sorting_inputs(F, V, params)
    rank = np.empty(len(F), dtype=int)
    crowd = np.empty(len(F))
    for r, front in enumerate(mu.fast_non_dominated_sort(Fs, Vs)):
        rank[front] = r
        crowd[front] = mu.crowding_distance(Fs[front])
    return rank, crowd


def tournament(rank, crowd, rng) -> int:
    """Binary tournament: lower rank wins, then larger crowding distance."""
    a, b = rng.integers(len(rank), size=2)
    if rank[a] != rank[b]:
        return int(a if rank[a] < rank[b] else b)
    return int(a if crowd[a] >= crowd[b] else b)


def make_offspring(problem, X, rank, crowd, params, rng, n) -> np.ndarray:
    children = []
    while len(children) < n:
        pa, pb = X[tournament(rank, crowd, rng)], X[tournament(rank, crowd, rng)]
        if rng.random() < params["crossover_rate"]:
            c1, c2 = uniform_crossover(problem, pa, pb, rng)
        else:
            c1, c2 = pa.copy(), pb.copy()
        for c in (c1, c2):
            c = mutate(problem, c, rng, params["mutation_rate"])
            if params["repair"]:
                c = problem.repair(c, rng)
            children.append(c)
    return np.array(children[:n])


def survival(F, V, n, params) -> np.ndarray:
    """Indices of the n survivors: whole fronts first, then the least crowded of the split front."""
    Fs, Vs = sorting_inputs(F, V, params)
    chosen: list[int] = []
    for front in mu.fast_non_dominated_sort(Fs, Vs):
        if len(chosen) + len(front) <= n:
            chosen.extend(front.tolist())
        else:
            cd = mu.crowding_distance(Fs[front])
            chosen.extend(front[np.argsort(-cd, kind="stable")][: n - len(chosen)].tolist())
            break
    return np.array(chosen)


def random_population(problem, n, params, rng) -> np.ndarray:
    X = np.array([problem.random_solution(rng) for _ in range(n)])
    if params["repair"]:
        X = np.array([problem.repair(x, rng) for x in X])
    return X


# ---------- algorithm interface ----------
def initialize(problem, params, ctx):
    X = random_population(problem, params["pop_size"], params, ctx.rng)
    F, V = evaluate(problem, X, ctx)
    return {"X": X, "F": F, "V": V, "params": params}


def evaluate(problem, X, ctx):
    return ctx.evaluate(X)


def step(problem, state, ctx):
    params = state["params"]
    X, F, V = state["X"], state["F"], state["V"]
    n = min(len(X), ctx.remaining()) or len(X)  # last generation may be smaller so the budget is exact
    rank, crowd = rank_and_crowding(F, V, params)
    Q = make_offspring(problem, X, rank, crowd, params, ctx.rng, n)
    FQ, VQ = evaluate(problem, Q, ctx)
    RX, RF, RV = np.vstack([X, Q]), np.vstack([F, FQ]), np.concatenate([V, VQ])
    keep = survival(RF, RV, len(X), params)
    return {**state, "X": RX[keep], "F": RF[keep], "V": RV[keep]}


def output(state):
    return feasible_front(state["X"], state["F"], state["V"])


def run(problem, params=None, progress=None, should_stop=None):
    import sys

    return run_loop(sys.modules[__name__], problem, params, progress, should_stop)
