"""Neighbourhood moves and hill climbing (used by MOSA and MOMA).

Moves
    swap  : exchange the time slots of two sessions
    move  : put one session into a slot where its teacher and group are free (and the teacher is
            available), in the smallest free room that fits; if there is none, a random slot
"""
from __future__ import annotations

import numpy as np

from . import moo_utils as mu
from .problem import TimetableProblem


def neighbour_swap(p: TimetableProblem, x, rng) -> np.ndarray:
    y = np.array(x, copy=True)
    a, b = rng.choice(p.S, size=2, replace=False)
    y[a], y[b] = y[b], y[a]
    return y


def neighbour_move(p: TimetableProblem, x, rng) -> np.ndarray:
    y = np.array(x, copy=True)
    slots, rooms = y[: p.S], y[p.S :]
    s = int(rng.integers(p.S))
    tc, gc, rc = p._counts(slots, rooms)
    t, g = p.sess_teacher[s], p.sess_group[s]
    tc[t, slots[s]] -= 1  # take s out before looking for a free place
    gc[g, slots[s]] -= 1
    rc[rooms[s], slots[s]] -= 1
    fits = p.fitting_rooms[s]
    slot_ok = (tc[t] == 0) & ~p.unavail[t] & (gc[g] == 0)
    room_free = rc[fits] == 0
    cand = np.flatnonzero(slot_ok & room_free.any(axis=0))
    cand = cand[cand != slots[s]]
    if len(cand):
        t_new = int(rng.choice(cand))
        free = fits[room_free[:, t_new]]
        slots[s], rooms[s] = t_new, int(free[np.argmin(p.cap[free])])
    else:
        slots[s] = int(rng.integers(p.T))
    return y


def neighbour(p: TimetableProblem, x, rng) -> np.ndarray:
    """Swap or move with equal probability."""
    return neighbour_swap(p, x, rng) if rng.random() < 0.5 else neighbour_move(p, x, rng)


def scalar(F, w, nadir) -> float:
    """Weighted sum of objectives normalised by `nadir` (ideal taken as 0)."""
    return float(np.dot(w, mu.normalize(F, np.zeros(len(F)), nadir)))


def hill_climb(p: TimetableProblem, x, f, v, steps, ctx, w=None, repair=True):
    """First-improvement hill climbing for `steps` neighbour evaluations.

    A neighbour y replaces x if it has a smaller violation, or the same violation and a smaller
    weighted sum w . f_normalised (w random per call, so different children are pushed towards
    different parts of the front). Each neighbour costs one evaluation from the shared budget.
    """
    rng = ctx.rng
    w = rng.dirichlet(np.ones(p.M)) if w is None else w
    nadir = ctx.hv_nadir
    for _ in range(steps):
        if ctx.remaining() <= 0:
            break
        y = neighbour(p, x, rng)
        if repair:
            y = p.repair(y, rng)
        Fy, Vy = ctx.evaluate(y[None, :])
        fy, vy = Fy[0], Vy[0]
        if vy < v or (vy == v and scalar(fy, w, nadir) < scalar(f, w, nadir)):
            x, f, v = y, fy, vy
    return x, f, v
