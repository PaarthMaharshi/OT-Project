"""Greedy / DSatur baseline (proposal Section 10).

Events are ordered by difficulty (fewest options first, then longest, then busiest groups) and each is
put in the first feasible slot (period-major, so mornings fill first) with the best-fitting room.
The same routine, randomised, seeds NSGA-II; with `preferred` it acts as the repair operator.
"""
from __future__ import annotations

import numpy as np

from ..evaluator import evaluate
from ..problem import Problem, Solution
from ..schedule_state import ScheduleState


def _assign_faculty(prob: Problem, st: ScheduleState, rng, randomized: bool, preferred: Solution | None) -> np.ndarray:
    fac = np.zeros(prob.K, dtype=np.int64)
    noisy = randomized or preferred is not None
    # Biggest, least flexible offerings first (bin-packing style), with a little noise when randomised.
    order = sorted(
        range(prob.K),
        key=lambda k: (-prob.off_need[k] * (1 + (0.5 * rng.random() if noisy else 0)), len(prob.off_elig[k])),
    )
    for k in order:
        need = prob.off_need[k]
        elig = prob.off_elig[k]
        if preferred is not None:
            f = preferred.fac[k]
            if st.fac_load[f] + need <= prob.fac_max[f]:
                fac[k] = f
                st.add_load(k, f)
                continue
        rel = (st.fac_load[elig] + need) / prob.fac_max[elig]
        fits = rel <= 1.0 + 1e-9
        cand = elig[fits] if fits.any() else elig[[np.argmin(rel)]]
        if randomized and len(cand) > 1:
            w = 1.0 / (0.2 + (st.fac_load[cand] + need) / prob.fac_max[cand])
            f = int(rng.choice(cand, p=w / w.sum()))
        else:
            r = (st.fac_load[cand] + need) / prob.fac_max[cand]
            f = int(cand[np.argmin(r)])
        fac[k] = f
        st.add_load(k, f)
    _rebalance(prob, st, fac, rng)
    return fac


def _rebalance(prob: Problem, st: ScheduleState, fac: np.ndarray, rng) -> None:
    """Move offerings off overloaded teachers (C7 weekly) to eligible teachers with spare hours."""
    for _ in range(prob.K * 2):
        over = np.flatnonzero(st.fac_load > prob.fac_max)
        if not len(over):
            return
        f = int(rng.choice(over))
        moved = False
        for k in rng.permutation(np.flatnonzero(fac == f)):
            need = prob.off_need[k]
            alt = [g for g in prob.off_elig[k] if g != f and st.fac_load[g] + need <= prob.fac_max[g]]
            if alt:
                g = min(alt, key=lambda g: st.fac_load[g] / prob.fac_max[g])
                st.add_load(k, f, -1)
                st.add_load(k, g)
                fac[k] = g
                moved = True
                break
        if not moved:
            # Two-step: free room on a teacher who could take one of f's offerings.
            for k in rng.permutation(np.flatnonzero(fac == f)):
                for g in prob.off_elig[k]:
                    if g == f:
                        continue
                    for k2 in np.flatnonzero(fac == g):
                        for h in prob.off_elig[k2]:
                            if h not in (f, g) and st.fac_load[h] + prob.off_need[k2] <= prob.fac_max[h]:
                                st.add_load(k2, g, -1), st.add_load(k2, h)
                                fac[k2] = h
                                if st.fac_load[g] + prob.off_need[k] <= prob.fac_max[g]:
                                    st.add_load(k, f, -1), st.add_load(k, g)
                                    fac[k] = g
                                    moved = True
                                break
                        if moved:
                            break
                    if moved:
                        break
                if moved:
                    break
        if not moved:
            return


def _static_order(prob: Problem, rng, noise: float) -> list[int]:
    group_load = np.zeros(prob.G)
    for e in range(prob.E):
        for g in prob.ev_groups[e]:
            group_load[g] += prob.ev_len[e]
    keys = []
    for e in range(prob.E):
        dom = len(prob.valid_starts[e]) * len(prob.valid_rooms[e])
        busy = max(group_load[g] for g in prob.ev_groups[e])
        keys.append((dom * (1 + noise * rng.random()), -prob.ev_len[e], -busy))
    return sorted(range(prob.E), key=lambda e: keys[e])


def _choose(prob: Problem, ts: np.ndarray, rs: np.ndarray, e: int, rng, randomized: bool, pref: tuple[int, int] | None):
    P = prob.P
    if pref is not None:
        pt, pr = pref
        dist = np.abs(ts // P - pt // P) * P + np.abs(ts % P - pt % P) + (rs != pr) * 0.5
        dist = dist + rng.random(len(ts)) * 0.25
        i = int(np.argmin(dist))
        return int(ts[i]), int(rs[i])
    period_major = (ts % P) * prob.D + ts // P
    fit = prob.room_cap[rs] - prob.ev_head[e]
    if randomized:
        score = period_major * rng.uniform(0.3, 1.0) + rng.random(len(ts)) * prob.D * 3 + fit / 40 * rng.random()
    else:
        score = period_major * 1000 + fit
    i = int(np.argmin(score))
    return int(ts[i]), int(rs[i])


def greedy(
    prob: Problem,
    rng: np.random.Generator | None = None,
    randomized: bool = False,
    preferred: Solution | None = None,
    dsatur: bool = True,
) -> Solution:
    rng = rng or np.random.default_rng(0)
    st = ScheduleState(prob)
    fac = _assign_faculty(prob, st, rng, randomized, preferred)
    slot = np.zeros(prob.E, dtype=np.int64)
    room = np.zeros(prob.E, dtype=np.int64)
    placed = np.zeros(prob.E, dtype=bool)

    # Repair mode: keep every event whose current position is still feasible.
    if preferred is not None:
        for e in rng.permutation(prob.E):
            f = fac[prob.ev_off[e]]
            ts, rs = st.options(e, f)
            hit = np.flatnonzero((ts == preferred.slot[e]) & (rs == preferred.room[e]))
            if len(hit):
                slot[e], room[e] = preferred.slot[e], preferred.room[e]
                st.place(e, slot[e], room[e], f)
                placed[e] = True

    todo = [e for e in _static_order(prob, rng, 0.6 if randomized else 0.0) if not placed[e]]
    # True DSatur (re-rank by remaining options) for the deterministic baseline; static order otherwise.
    dynamic = dsatur and not randomized and preferred is None
    refresh = max(1, prob.E // 40)
    since = refresh
    while todo:
        if dynamic and since >= refresh:
            counts = {e: len(st.options(e, fac[prob.ev_off[e]])[0]) for e in todo}
            todo.sort(key=lambda e: (counts[e], -prob.ev_len[e]))
            since = 0
        e = todo.pop(0)
        since += 1
        f = fac[prob.ev_off[e]]
        ts, rs = st.options(e, f)
        if len(ts):
            pref = (int(preferred.slot[e]), int(preferred.room[e])) if preferred is not None else None
            t, r = _choose(prob, ts, rs, e, rng, randomized, pref)
        else:
            t, r = st.least_conflict(e, f, rng)
        slot[e], room[e] = t, r
        st.place(e, t, r, f)
    return evaluate(prob, Solution(slot, room, fac))
