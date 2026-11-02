"""Goal Programming (catalogue no. 6), weighted (Archimedean) form, used as a baseline.

Lecture steps
    1. Payoff table (as in epsilon-constraint): minimise each objective alone -> ideal and nadir estimates.
    2. Goals g_j (aspiration levels). Default: g_j = ideal_j + 0.25 (nadir_j - ideal_j); the caller may
       pass "goals": {objective_name: value}.
    3. Over-achievement is free; only the under-achievement (deviation above the goal) is penalised:
         d_j+(x) = max(0, f_j(x) - g_j)
         min  sum_j w_j d_j+(x) / (nadir_j - ideal_j)   (+ 0.001 * sum_j f_j / nadir_j as a tie-breaker)
       subject to the hard constraints.
    4. Sweep at least 10 weight vectors w; the non-dominated answers form the front.
    The tie-breaker stops the method from returning weakly dominated timetables once every goal is met.

Default parameters
    goals None (25 % of the way from ideal to nadir), n_weights 11, repair True.
"""
from __future__ import annotations

import numpy as np

from .common import feasible_front, run_loop
from .scalar_solver import anneal, simplex_weights, warm_start

NAME = "goal_programming"
DEFAULTS = {"goals": None, "goal_fraction": 0.25, "n_weights": 11, "repair": True}


def deviation(F, goals, w, span) -> float:
    """Weighted, normalised sum of the deviations above the goals."""
    d_plus = np.maximum(0.0, np.asarray(F) - goals)
    return float(np.dot(w, d_plus / span))


def initialize(problem, params, ctx):
    x0, F, V = warm_start(problem, ctx)
    W = simplex_weights(problem.M, params["n_weights"])
    per = max(10, (ctx.max_evals - ctx.evals) // (problem.M + len(W)))
    return {"x": x0, "W": W, "per": per, "phase": "payoff", "k": 0, "payoff": [], "goals": None, "span": None,
            "X": [], "F": [], "Vs": [], "V": V, "params": params, "extra": {}}


def evaluate(problem, X, ctx):
    return ctx.evaluate(X)


def step(problem, state, ctx):
    p = state["params"]
    nadir = ctx.hv_nadir
    if state["phase"] == "payoff":
        j = state["k"]
        x, key, F, Fa, V, _ = anneal(problem, lambda F, Fa, V, b: (V, F[j] / nadir[j]), state["x"], state["per"], ctx, p["repair"])
        state["payoff"].append(F)
        state["X"].append(x), state["F"].append(F), state["Vs"].append(V)
        state["k"] += 1
        if state["k"] == problem.M:
            P = np.array(state["payoff"])
            ideal, worst = P.min(axis=0), P.max(axis=0)
            if p["goals"]:
                goals = np.array([float(p["goals"].get(o, ideal[i])) for i, o in enumerate(problem.objectives)])
            else:
                goals = ideal + p["goal_fraction"] * (worst - ideal)
            state["goals"], state["span"] = goals, np.where(worst - ideal > 1e-9, worst - ideal, 1.0)
            state["extra"] = {"goals": goals.tolist(), "payoff": P.tolist(), "weights": state["W"].tolist()}
            state["phase"], state["k"] = "sweep", 0
    elif state["k"] < len(state["W"]):
        w = state["W"][state["k"]]
        g, span = state["goals"], state["span"]
        last = state["k"] == len(state["W"]) - 1
        budget = max(10, ctx.remaining()) if last else state["per"]
        key_fn = lambda F, Fa, V, b: (V, deviation(F, g, w, span) + 1e-3 * float(np.sum(F / nadir)))  # noqa: E731
        x, key, F, Fa, V, _ = anneal(problem, key_fn, state["x"], budget, ctx, p["repair"])
        state["X"].append(x), state["F"].append(F), state["Vs"].append(V)
        state["x"] = x
        state["k"] += 1
    else:
        state["done"] = True
    state["V"] = np.array([state["Vs"][-1]]) if state["Vs"] else state["V"]
    if state["phase"] == "sweep" and state["k"] >= len(state["W"]):
        state["done"] = True
    return state


def output(state):
    if not state["X"]:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0)
    return feasible_front(np.array(state["X"]), np.array(state["F"]), np.array(state["Vs"]))


def run(problem, params=None, progress=None, should_stop=None):
    import sys

    return run_loop(sys.modules[__name__], problem, params, progress, should_stop)
