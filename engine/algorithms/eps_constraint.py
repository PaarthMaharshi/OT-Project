"""epsilon-Constraint method (catalogue no. 2), used as a baseline.

Lecture steps
    1. Payoff table: minimise every objective on its own (M sub-problems). The best value of each
       objective gives the ideal point; the worst value seen in the table gives the nadir estimate.
    2. Keep one primary objective (default f1, student gaps). Turn the others into constraints
         min f_1(x)   subject to   f_j(x) <= eps_j  (j != 1)  and all hard constraints.
    3. Sweep eps over at least 10 levels between ideal and nadir. With two objectives this is the
       classical one-dimensional sweep of eps_2. With more objectives all secondary objectives move
       together, eps_j(t) = ideal_j + t (nadir_j - ideal_j), t = 0 .. 1 (a diagonal sweep; a full grid
       would need 10^(M-1) sub-problems).
    4. Each sub-problem is solved with the annealer; "f_j > eps_j" is added to the penalty level so it
       is treated like a hard constraint (feasibility first). The non-dominated answers form the front.
    Unlike the weighted sum, this reaches non-convex parts of the front.

Default parameters
    primary "student_gaps" (falls back to the first optimised objective), n_eps 10, repair True.
"""
from __future__ import annotations

import numpy as np

from .common import feasible_front, run_loop
from .scalar_solver import anneal, warm_start

NAME = "eps_constraint"
DEFAULTS = {"primary": "student_gaps", "n_eps": 10, "repair": True}


def eps_key(F, V, primary: int, eps: np.ndarray, nadir: np.ndarray):
    """(penalty, value): penalty = V + normalised amount by which secondary objectives exceed eps."""
    excess = 0.0
    for j in range(len(F)):
        if j != primary:
            excess += max(0.0, F[j] - eps[j]) / nadir[j]
    return (V + excess, F[primary] / nadir[primary])


def initialize(problem, params, ctx):
    x0, F, V = warm_start(problem, ctx)
    primary = problem.objectives.index(params["primary"]) if params["primary"] in problem.objectives else 0
    n_sub = problem.M + params["n_eps"]
    per = max(10, (ctx.max_evals - ctx.evals) // n_sub)
    return {"x": x0, "primary": primary, "per": per, "phase": "payoff", "k": 0, "payoff": [], "eps": None,
            "X": [], "F": [], "Vs": [], "V": V, "params": params, "extra": {"eps_levels": [], "payoff": []}}


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
        if state["k"] == problem.M:  # payoff table done: build the eps levels
            P = np.array(state["payoff"])
            ideal, worst = P.min(axis=0), P.max(axis=0)
            t = np.linspace(0, 1, p["n_eps"])
            state["eps"] = ideal[None, :] + t[:, None] * (worst - ideal)[None, :]
            state["extra"]["payoff"] = P.tolist()
            state["phase"], state["k"] = "sweep", 0
    elif state["k"] < len(state["eps"]):
        eps = state["eps"][state["k"]]
        last = state["k"] == len(state["eps"]) - 1
        budget = max(10, ctx.remaining()) if last else state["per"]
        i = state["primary"]
        x, key, F, Fa, V, _ = anneal(problem, lambda F, Fa, V, b: eps_key(F, V, i, eps, nadir), state["x"], budget, ctx, p["repair"])
        satisfied = bool(key[0] - V <= 1e-12)
        state["extra"]["eps_levels"].append({"eps": eps.tolist(), "objectives": F.tolist(), "eps_satisfied": satisfied})
        state["X"].append(x), state["F"].append(F), state["Vs"].append(V)
        state["x"] = x
        state["k"] += 1
    else:
        state["done"] = True
    state["V"] = np.array([state["Vs"][-1]]) if state["Vs"] else state["V"]
    if state["phase"] == "sweep" and state["k"] >= len(state["eps"]):
        state["done"] = True
    return state


def output(state):
    if not state["X"]:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0)
    return feasible_front(np.array(state["X"]), np.array(state["F"]), np.array(state["Vs"]))


def run(problem, params=None, progress=None, should_stop=None):
    import sys

    return run_loop(sys.modules[__name__], problem, params, progress, should_stop)
