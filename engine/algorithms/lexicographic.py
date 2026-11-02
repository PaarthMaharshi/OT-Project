"""Lexicographic method (catalogue no. 7): optimise criteria one after another in a fixed priority order.

Lecture steps
    1. Order the criteria by priority, e.g.
         teacher_clashes  >  other_violations  >  student_gaps  >  teacher_gaps  >  room_waste  >  time_penalty
       "teacher_clashes" is H1 alone; "other_violations" is H2 + H3 + H4 + H5.
    2. Stage 1: minimise criterion 1 -> optimum c1*.
    3. Stage k: minimise criterion k subject to c_i(x) <= c_i* + tolerance for every earlier criterion i
       (each earlier criterion becomes a constraint; violating it is added to the penalty level).
    4. The answer of the last stage is the lexicographic optimum: a SINGLE timetable, not a front.
    Each stage is solved by the annealer, warm-started from the previous stage's answer.

Default parameters
    order ["teacher_clashes", "other_violations", "student_gaps", "teacher_gaps", "room_waste",
    "time_penalty"], tolerance 0, repair True.
"""
from __future__ import annotations

import numpy as np

from ..problem import OBJECTIVES
from .common import run_loop
from .scalar_solver import anneal, warm_start

NAME = "lexicographic"
DEFAULTS = {
    "order": ["teacher_clashes", "other_violations", "student_gaps", "teacher_gaps", "room_waste", "time_penalty"],
    "tolerance": 0.0,
    "repair": True,
}
CRITERIA = ["teacher_clashes", "other_violations", "violation", *OBJECTIVES]


def criterion(name: str, F_all, V, breakdown) -> float:
    if name == "teacher_clashes":
        return float(breakdown["H1_teacher"])
    if name == "other_violations":
        return float(V - breakdown["H1_teacher"])
    if name == "violation":
        return float(V)
    return float(F_all[OBJECTIVES.index(name)])


def stage_key(order, k, optima, tol):
    """Penalty = how far earlier criteria exceed their optimum (+tolerance); value = criterion k."""

    def key(F, F_all, V, b):
        excess = sum(max(0.0, criterion(order[i], F_all, V, b) - optima[i] - tol) for i in range(k))
        return (excess, criterion(order[k], F_all, V, b))

    return key


def initialize(problem, params, ctx):
    unknown = [c for c in params["order"] if c not in CRITERIA]
    if unknown:
        raise ValueError(f"Unknown lexicographic criteria {unknown}; choose from {CRITERIA}")
    x0, F, V = warm_start(problem, ctx)
    per = max(10, (ctx.max_evals - ctx.evals) // len(params["order"]))
    return {"x": x0, "k": 0, "optima": [], "per": per, "best": None, "V": V, "params": params, "extra": {"stages": []}}


def evaluate(problem, X, ctx):
    return ctx.evaluate(X)


def step(problem, state, ctx):
    p = state["params"]
    order = p["order"]
    k = state["k"]
    if k >= len(order):
        state["done"] = True
        return state
    last = k == len(order) - 1
    budget = max(10, ctx.remaining()) if last else state["per"]
    x, key, F, F_all, V, _ = anneal(problem, stage_key(order, k, state["optima"], p["tolerance"]), state["x"], budget, ctx, p["repair"])
    b = problem.evaluate_full(x)[2]  # recomputing the breakdown of the stage winner (not counted)
    state["optima"].append(criterion(order[k], F_all, V, b))
    state["extra"]["stages"].append({"criterion": order[k], "optimum": state["optima"][-1],
                                     "all_criteria": {c: criterion(c, F_all, V, b) for c in order}})
    state.update(x=x, best=(x, F, V), V=np.array([V]))
    state["k"] += 1
    state["done"] = state["k"] >= len(order)
    return state


def output(state):
    if state["best"] is None:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0)
    x, F, V = state["best"]
    return x[None], F[None], np.array([V])


def run(problem, params=None, progress=None, should_stop=None):
    import sys

    return run_loop(sys.modules[__name__], problem, params, progress, should_stop)
