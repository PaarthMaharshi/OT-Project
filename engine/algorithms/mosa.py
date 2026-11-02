"""Multi-objective simulated annealing, MOSA (catalogue no. 26). Archive-based, in the style of
Ulungu et al. (1999) and Suppapitnarm et al. (2000).

Lecture steps
    1. Start from one random (repaired) timetable x. Archive A = {x} if feasible.
    2. Calibrate the starting temperature: T0 = mean uphill energy change of a few random neighbours / ln 2
       (so a typical worse move is accepted with probability 1/2 at the start).
    3. Repeat L times at the current temperature T:
         a. y = neighbour of x (swap two sessions' slots, or move one session to a free slot), repaired.
         b. If y constrained-dominates x, accept it.
            Otherwise compute the energy change
              dE = sum_j w_j (f_j(y) - f_j(x)) / nadir_j  +  (V(y) - V(x))
            with a random weight vector w (redrawn at every temperature level), and accept y with the
            Boltzmann probability min(1, exp(-dE / T)).
         c. Insert y into the archive if it is feasible and no archive member dominates it.
    4. Cool: T = alpha * T. Occasionally restart x from a random archive member (diversification).
    5. Repeat 3-4 until the budget is used. The archive is the output.

Default parameters
    iterations_per_temperature L 100, cooling_rate alpha 0.95, archive_size 100, restart_prob 0.05,
    calibration_moves 20, repair True.
"""
from __future__ import annotations

import math

import numpy as np

from .. import moo_utils as mu
from ..local_search import neighbour
from .common import feasible_front, run_loop

NAME = "mosa"
DEFAULTS = {
    "iterations_per_temperature": 100,
    "cooling_rate": 0.95,
    "archive_size": 100,
    "restart_prob": 0.05,
    "calibration_moves": 20,
    "repair": True,
}


def energy_change(fx, vx, fy, vy, w, nadir) -> float:
    return float(np.dot(w, (fy - fx) / nadir) + (vy - vx))


def accept_probability(dE: float, T: float) -> float:
    """Boltzmann acceptance: improvements always, worsening moves with exp(-dE/T)."""
    if dE <= 0:
        return 1.0
    return math.exp(-dE / max(T, 1e-12))


def _add_to_archive(state, y, fy, size):
    AF = state["AF"]
    if len(AF) and (any(mu.dominates(a, fy) for a in AF) or any(np.array_equal(a, fy) for a in AF)):
        return
    keep = np.array([not mu.dominates(fy, a) for a in AF], dtype=bool) if len(AF) else np.zeros(0, bool)
    AX, AF = np.vstack([state["AX"][keep], y[None]]), np.vstack([AF[keep], fy[None]])
    if len(AF) > size:  # drop the most crowded member
        cd = mu.crowding_distance(AF)
        drop = int(np.argmin(cd))
        AX, AF = np.delete(AX, drop, axis=0), np.delete(AF, drop, axis=0)
    state["AX"], state["AF"] = AX, AF


def _propose(problem, x, p, rng):
    y = neighbour(problem, x, rng)
    return problem.repair(y, rng) if p["repair"] else y


def initialize(problem, params, ctx):
    rng = ctx.rng
    x = problem.random_solution(rng)
    if params["repair"]:
        x = problem.repair(x, rng)
    F, V = evaluate(problem, x[None], ctx)
    state = {"x": x, "fx": F[0], "vx": V[0], "AX": x[None][:0], "AF": F[:0], "params": params, "V": V}
    if V[0] == 0:
        _add_to_archive(state, x, F[0], params["archive_size"])
    # temperature calibration with a few random neighbours (they count as evaluations)
    w = np.full(problem.M, 1.0 / problem.M)
    ups = []
    for _ in range(params["calibration_moves"]):
        y = _propose(problem, x, params, rng)
        Fy, Vy = evaluate(problem, y[None], ctx)
        dE = energy_change(F[0], V[0], Fy[0], Vy[0], w, ctx.hv_nadir)
        if dE > 0:
            ups.append(dE)
        if Vy[0] == 0:
            _add_to_archive(state, y, Fy[0], params["archive_size"])
    state["T"] = (np.mean(ups) / math.log(2)) if ups else 0.1
    state["T0"] = state["T"]
    return state


def evaluate(problem, X, ctx):
    return ctx.evaluate(X)


def step(problem, state, ctx):
    """One temperature level: L proposals, then cooling."""
    p, rng = state["params"], ctx.rng
    w = rng.dirichlet(np.ones(problem.M))
    x, fx, vx, T = state["x"], state["fx"], state["vx"], state["T"]
    for _ in range(p["iterations_per_temperature"]):
        if ctx.remaining() <= 0 or ctx.stop_reason() is not None:
            break
        y = _propose(problem, x, p, rng)
        Fy, Vy = evaluate(problem, y[None], ctx)
        fy, vy = Fy[0], Vy[0]
        if mu.constrained_dominates(fy, fx, vy, vx):
            x, fx, vx = y, fy, vy
        elif rng.random() < accept_probability(energy_change(fx, vx, fy, vy, w, ctx.hv_nadir), T):
            x, fx, vx = y, fy, vy
        if vy == 0:
            _add_to_archive(state, y, fy, p["archive_size"])
    state["T"] = T * p["cooling_rate"]
    if len(state["AF"]) and rng.random() < p["restart_prob"]:
        k = int(rng.integers(len(state["AF"])))
        x, fx, vx = state["AX"][k].copy(), state["AF"][k].copy(), 0.0
    state.update(x=x, fx=fx, vx=vx, V=np.array([vx]))
    return state


def output(state):
    if len(state["AF"]):
        return feasible_front(state["AX"], state["AF"], np.zeros(len(state["AF"])))
    return state["x"][None], state["fx"][None], np.array([state["vx"]])


def run(problem, params=None, progress=None, should_stop=None):
    import sys

    return run_loop(sys.modules[__name__], problem, params, progress, should_stop)
