"""Repair operator (proposal Section 12).

1. Fix teacher overload (C7 weekly) by moving offerings to eligible teachers with spare hours.
2. Find events involved in any violation; place every other event as it is.
3. Re-insert the troubled events one by one: keep the current position if it is now feasible,
   otherwise move to the nearest feasible slot (same day first) with the best-fitting room.
Only troubled events are touched, so repair is cheap when a child has just a few clashes.
"""
from __future__ import annotations

import numpy as np

from ..baseline.greedy_dsatur import _choose, _rebalance
from ..evaluator import evaluate, occupancy
from ..problem import Problem, Solution
from ..schedule_state import ScheduleState


def troubled_events(prob: Problem, s: Solution) -> np.ndarray:
    room_occ, fac_occ, grp_occ = occupancy(prob, s.slot, s.room, s.fac)
    f_e = s.fac[prob.ev_off]
    bad = np.zeros(prob.E, dtype=bool)

    cell = s.slot[prob.rep_e] + prob.rep_off
    r_e = s.room[prob.rep_e]
    f_c = f_e[prob.rep_e]
    bad_cell = (room_occ[r_e, cell] > 1) | (fac_occ[f_c, cell] > 1) | prob.fac_block[f_c, cell] | prob.room_block[r_e, cell]
    bad[prob.rep_e[bad_cell]] = True

    gcell = s.slot[prob.grp_e] + prob.grp_off
    bad[prob.grp_e[grp_occ[prob.grp_g, gcell] > 1]] = True

    daily = fac_occ.reshape(prob.F, prob.D, prob.P).sum(axis=2)
    bad |= daily[f_e, s.slot // prob.P] > prob.fac_daily[f_e]

    th = np.flatnonzero(prob.ev_theory)
    key = prob.ev_off[th] * prob.D + s.slot[th] // prob.P
    counts = np.bincount(key, minlength=prob.K * prob.D)
    bad[th[counts[key] > 1]] = True
    return bad


def repair(prob: Problem, s: Solution, rng: np.random.Generator) -> Solution:
    s = s.copy()

    load = np.bincount(s.fac, weights=prob.off_need, minlength=prob.F).astype(np.int64)
    if (load > prob.fac_max).any():
        tmp = ScheduleState(prob)
        tmp.fac_load[:] = load
        _rebalance(prob, tmp, s.fac, rng)

    bad = troubled_events(prob, s)
    st = ScheduleState.from_solution(prob, s.slot, s.room, s.fac, ~bad)

    # Hardest first (fewest valid starts), random tie-break.
    todo = np.flatnonzero(bad)
    todo = todo[np.lexsort((rng.random(len(todo)), prob.n_starts[todo]))]
    for e in todo:
        f = int(s.fac[prob.ev_off[e]])
        t0, r0 = int(s.slot[e]), int(s.room[e])
        if st.fits(e, t0, r0, f):
            t, r = t0, r0
        else:
            ts, rs = st.options(e, f)
            if len(ts):
                t, r = _choose(prob, ts, rs, e, rng, False, (t0, r0))
            else:
                t, r = st.least_conflict(e, f, rng)
        s.slot[e], s.room[e] = t, r
        st.place(e, t, r, f)
    s.F = None
    return evaluate(prob, s)
