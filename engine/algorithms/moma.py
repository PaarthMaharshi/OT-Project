"""Multi-objective memetic algorithm, MOMA (catalogue no. 30) = Constrained NSGA-II + hill climbing.

Lecture steps
    1-4. As Constrained NSGA-II: random repaired population, constrained sort + crowding, tournament,
         uniform crossover + mutation + repair -> children Q, evaluate Q.
    5.   Local search: a fraction ls_rate of the children is improved by first-improvement hill climbing
         for ls_steps neighbour evaluations. Neighbourhood: swap the slots of two lectures, or move one
         lecture to a free slot. A neighbour is accepted if it has less violation, or equal violation and
         a lower weighted sum of normalised objectives (random weights per child, so different children
         are pulled towards different parts of the front).
    6.   Constrained survival of P + Q as in NSGA-II. Repeat until the budget is used.
    Local-search evaluations come out of the same budget, so MOMA runs fewer generations than NSGA-II.

Default parameters
    Constrained NSGA-II defaults + ls_rate 0.2, ls_steps 5.
"""
from __future__ import annotations

import numpy as np

from ..local_search import hill_climb
from . import nsga2
from .common import run_loop

NAME = "moma"
DEFAULTS = {**nsga2.DEFAULTS, "constraint_handling": "deb", "repair": True, "ls_rate": 0.2, "ls_steps": 5}


def initialize(problem, params, ctx):
    return nsga2.initialize(problem, params, ctx)


def evaluate(problem, X, ctx):
    return ctx.evaluate(X)


def step(problem, state, ctx):
    params, rng = state["params"], ctx.rng
    X, F, V = state["X"], state["F"], state["V"]
    n = min(len(X), ctx.remaining()) or len(X)
    rank, crowd = nsga2.rank_and_crowding(F, V, params)
    Q = nsga2.make_offspring(problem, X, rank, crowd, params, rng, n)
    FQ, VQ = evaluate(problem, Q, ctx)
    # memetic part: improve a random subset of the children
    for i in np.flatnonzero(rng.random(n) < params["ls_rate"]):
        Q[i], FQ[i], VQ[i] = hill_climb(problem, Q[i], FQ[i], VQ[i], params["ls_steps"], ctx, repair=params["repair"])
    RX, RF, RV = np.vstack([X, Q]), np.vstack([F, FQ]), np.concatenate([V, VQ])
    keep = nsga2.survival(RF, RV, len(X), params)
    return {**state, "X": RX[keep], "F": RF[keep], "V": RV[keep]}


def output(state):
    return nsga2.output(state)


def run(problem, params=None, progress=None, should_stop=None):
    import sys

    return run_loop(sys.modules[__name__], problem, params, progress, should_stop)
