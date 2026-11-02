"""Objectives f1-f3 and constraint violation V(s) from proposal Section 7, vectorised with numpy."""
from __future__ import annotations

import numpy as np

from .problem import Problem, Solution

LAMBDA_FACULTY_GAPS = 0.5


def occupancy(prob: Problem, slot: np.ndarray, room: np.ndarray, fac: np.ndarray):
    """Per-slot usage counts: rooms [R,T], faculty [F,T], groups [G,T]."""
    T = prob.T
    cells = slot[prob.rep_e] + prob.rep_off
    room_occ = np.bincount(room[prob.rep_e] * T + cells, minlength=prob.R * T).reshape(prob.R, T)
    f_e = fac[prob.ev_off]
    fac_occ = np.bincount(f_e[prob.rep_e] * T + cells, minlength=prob.F * T).reshape(prob.F, T)
    gcells = slot[prob.grp_e] + prob.grp_off
    grp_occ = np.bincount(prob.grp_g * T + gcells, minlength=prob.G * T).reshape(prob.G, T)
    return room_occ, fac_occ, grp_occ


def _gaps(busy: np.ndarray, lunch: int) -> np.ndarray:
    """Idle periods between first and last lecture of each day (lunch excluded). busy: bool [N, D, P]."""
    P = busy.shape[2]
    any_ = busy.any(axis=2)
    first = busy.argmax(axis=2)
    last = P - 1 - busy[:, :, ::-1].argmax(axis=2)
    span = last - first + 1
    lunch_inside = (first < lunch) & (last > lunch)
    return np.where(any_, span - busy.sum(axis=2) - lunch_inside, 0)


def violation_breakdown(prob: Problem, s: Solution, occ=None) -> dict[str, float]:
    room_occ, fac_occ, grp_occ = occ if occ is not None else occupancy(prob, s.slot, s.room, s.fac)
    f_e = s.fac[prob.ev_off]
    load = np.bincount(f_e, weights=prob.ev_len, minlength=prob.F)
    daily = fac_occ.reshape(prob.F, prob.D, prob.P).sum(axis=2)
    # C8: at most one theory session of an offering per day
    th = prob.ev_theory
    key = prob.ev_off[th] * prob.D + s.slot[th] // prob.P
    per_day = np.bincount(key, minlength=prob.K * prob.D)
    return {
        "room_clash": float(np.maximum(room_occ - 1, 0).sum()),  # C2
        "faculty_clash": float(np.maximum(fac_occ - 1, 0).sum()),  # C3
        "student_clash": float(np.maximum(grp_occ - 1, 0).sum()),  # C4
        "unavailable": float((fac_occ * prob.fac_block).sum() + (room_occ * prob.room_block).sum()),  # C6
        "weekly_overload": float(np.maximum(load - prob.fac_max, 0).sum()),  # C7
        "daily_overload": float(np.maximum(daily - prob.fac_daily[:, None], 0).sum()),  # C7
        "same_day_theory": float(np.maximum(per_day - 1, 0).sum()),  # C8
    }


def evaluate(prob: Problem, s: Solution) -> Solution:
    """Fills s.F (f1, f2, f3) and s.V in place and returns s."""
    occ = occupancy(prob, s.slot, s.room, s.fac)
    _, fac_occ, grp_occ = occ
    s.V = sum(violation_breakdown(prob, s, occ).values())

    D, P = prob.D, prob.P
    f1 = _gaps(grp_occ.reshape(prob.G, D, P) > 0, prob.lunch).sum() + LAMBDA_FACULTY_GAPS * _gaps(
        fac_occ.reshape(prob.F, D, P) > 0, prob.lunch
    ).sum()

    cap = prob.room_cap[s.room]
    f2 = float((prob.ev_len * (cap - prob.ev_head) / cap).sum() / prob.ev_len.sum())

    load = np.bincount(s.fac[prob.ev_off], weights=prob.ev_len, minlength=prob.F)
    f3 = float(np.std(load / prob.fac_max))

    s.F = np.array([float(f1), f2, f3])
    return s


def check_domains(prob: Problem, s: Solution) -> list[str]:
    """Checks C1/C5/C6 domain rules that the solvers enforce by construction (used in tests)."""
    errs = []
    for e in range(prob.E):
        if s.slot[e] not in prob.valid_starts[e]:
            errs.append(f"event {e}: invalid start {s.slot[e]}")
        if s.room[e] not in prob.valid_rooms[e]:
            errs.append(f"event {e}: invalid room {s.room[e]}")
    for k in range(prob.K):
        if s.fac[k] not in prob.off_elig[k]:
            errs.append(f"offering {k}: ineligible faculty {s.fac[k]}")
    return errs


def practical_metrics(prob: Problem, s: Solution) -> dict[str, float]:
    """Section 13 practical metrics for reporting."""
    room_occ, fac_occ, grp_occ = occupancy(prob, s.slot, s.room, s.fac)
    D, P = prob.D, prob.P
    gb = grp_occ.reshape(prob.G, D, P) > 0
    g = _gaps(gb, prob.lunch)
    days_used = gb.any(axis=2).sum()

    def longest_run(b):  # longest consecutive busy stretch, lunch breaks a run
        best = 0
        for row in b.reshape(-1, P):
            run = 0
            for p, x in enumerate(row):
                run = run + 1 if x and p != prob.lunch else 0
                best = max(best, run)
        return best

    usable = prob.T - prob.D  # minus lunch
    return {
        "avg_gap_per_group_day": float(g.sum() / max(days_used, 1)),
        "max_gap": float(g.max(initial=0)),
        "room_utilisation": float((room_occ > 0).sum() / (prob.R * usable)),
        "max_daily_load": float(fac_occ.reshape(prob.F, D, P).sum(axis=2).max(initial=0)),
        "longest_consecutive_lectures": float(longest_run(gb)),
    }
