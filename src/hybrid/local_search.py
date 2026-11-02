"""Short local search (proposal Section 12), applied only while feasibility is kept:
  * best-fit room swap: move each event to the smallest free room that fits (lowers f2)
  * gap compaction: move an event next to a neighbour on the same day (lowers f1)
"""
from __future__ import annotations

import numpy as np

from ..evaluator import evaluate
from ..problem import Problem, Solution
from ..schedule_state import ScheduleState


def local_search(prob: Problem, s: Solution, rng: np.random.Generator, tries: int = 12) -> Solution:
    if s.V > 0:
        return s
    s = s.copy()
    st = ScheduleState.from_solution(prob, s.slot, s.room, s.fac, np.ones(prob.E, dtype=bool))

    # Best-fit rooms: same slot, smaller room that still fits and is free.
    for e in rng.permutation(prob.E):
        t, r = int(s.slot[e]), int(s.room[e])
        L = prob.ev_len[e]
        rooms = prob.valid_rooms[e]
        smaller = rooms[prob.room_cap[rooms] < prob.room_cap[r]]
        if not len(smaller):
            continue
        free = ~((st.room_occ[smaller, t : t + L] > 0) | prob.room_block[smaller, t : t + L]).any(axis=1)
        if free.any():
            cand = smaller[free]
            nr = int(cand[np.argmin(prob.room_cap[cand])])
            st.room_occ[r, t : t + L] -= 1
            st.room_occ[nr, t : t + L] += 1
            s.room[e] = nr
    evaluate(prob, s)

    # Gap compaction: try moving random events within their day; accept if f1 drops and f2 does not rise.
    P = prob.P
    for e in rng.choice(prob.E, size=min(tries, prob.E), replace=False):
        t, r, f = int(s.slot[e]), int(s.room[e]), int(s.fac[prob.ev_off[e]])
        st.remove(e, t, r, f)
        ts, rs = st.options(e, f)
        keep = (ts // P == t // P) & (ts != t)
        best = None
        if keep.any():
            cand_t = np.unique(ts[keep])
            for nt in rng.permutation(cand_t)[:4]:
                rooms = rs[ts == nt]
                nr = int(rooms[np.argmin(prob.room_cap[rooms])])
                trial = s.copy()
                trial.slot[e], trial.room[e] = nt, nr
                evaluate(prob, trial)
                if trial.V == 0 and trial.F[0] < s.F[0] - 1e-9 and trial.F[1] <= s.F[1] + 1e-9:
                    if best is None or trial.F[0] < best.F[0]:
                        best = trial
        if best is not None:
            s = best
        st.place(e, int(s.slot[e]), int(s.room[e]), f)
    return s
