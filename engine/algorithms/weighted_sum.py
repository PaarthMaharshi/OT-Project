"""Weighted Sum method (catalogue no. 1), used as a baseline.

Lecture steps
    1. Normalise each objective: f~_j = f_j / nadir_j (nadir estimated from 20 random repaired timetables;
       the ideal is 0 because no objective can be negative).
    2. Choose K weight vectors w (at least 10; 11 for two objectives: w1 = 0, 0.1, ..., 1).
    3. For each w solve   min  sum_j w_j f~_j(x)   subject to all hard constraints
       with the single-objective annealer (warm-started from the previous sub-problem's answer).
    4. The non-dominated answers form the front.
    Known weakness: a weighted sum can only reach points on the CONVEX hull of the front; points in
    non-convex regions are optimal for no weight vector, however many weights are tried.

Default parameters
    n_weights 11, repair True; each sub-problem gets an equal share of the evaluation budget.
"""
from __future__ import annotations

import numpy as np

from .common import feasible_front, run_loop
from .scalar_solver import anneal, simplex_weights, warm_start

NAME = "weighted_sum"
DEFAULTS = {"n_weights": 11, "repair": True}


def weighted_value(F, w, nadir) -> float:
    return float(np.dot(w, np.asarray(F) / nadir))


def initialize(problem, params, ctx):
    x0, F, V = warm_start(problem, ctx)
    W = simplex_weights(problem.M, params["n_weights"])
    per = max(10, (ctx.max_evals - ctx.evals) // len(W))
    return {"W": W, "k": 0, "x": x0, "per": per, "X": [], "F": [], "Vs": [], "V": V, "params": params, "extra": {"weights": W.tolist()}}


def evaluate(problem, X, ctx):
    return ctx.evaluate(X)


def step(problem, state, ctx):
    if state["k"] >= len(state["W"]):  # all sub-problems solved: spend nothing more
        state["done"] = True
        return state
    w = state["W"][state["k"]]
    nadir = ctx.hv_nadir
    budget = state["per"] if state["k"] < len(state["W"]) - 1 else max(10, ctx.remaining())
    x, key, F, F_all, V, _ = anneal(problem, lambda F, Fa, V, b: (V, weighted_value(F, w, nadir)), state["x"], budget, ctx, state["params"]["repair"])
    state["X"].append(x)
    state["F"].append(F)
    state["Vs"].append(V)
    state["x"] = x
    state["k"] += 1
    state["V"] = np.array([V])
    state["done"] = state["k"] >= len(state["W"])
    return state


def output(state):
    if not state["X"]:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0)
    return feasible_front(np.array(state["X"]), np.array(state["F"]), np.array(state["Vs"]))


def run(problem, params=None, progress=None, should_stop=None):
    import sys

    return run_loop(sys.modules[__name__], problem, params, progress, should_stop)
