"""Incremental occupancy state used by constructive heuristics (greedy, repair).

Answers "where can event e go right now?" checking C2-C4, C6, C7 (daily) and C8.
C5 and the lunch / day-boundary part of C6 are already built into the domains.
"""
from __future__ import annotations

import numpy as np

from .problem import Problem


class ScheduleState:
    def __init__(self, prob: Problem):
        self.p = prob
        self.room_occ = np.zeros((prob.R, prob.T), dtype=np.int32)
        self.fac_occ = np.zeros((prob.F, prob.T), dtype=np.int32)
        self.grp_occ = np.zeros((prob.G, prob.T), dtype=np.int32)
        self.fac_load = np.zeros(prob.F, dtype=np.int64)
        self.fac_day = np.zeros((prob.F, prob.D), dtype=np.int64)
        self.off_day = np.zeros((prob.K, prob.D), dtype=np.int32)
        # cells[e]: (num_valid_starts, length) slot indices covered by each candidate start
        self.cells = prob.cells

    @classmethod
    def from_solution(cls, prob: Problem, slot: np.ndarray, room: np.ndarray, fac: np.ndarray, mask: np.ndarray) -> "ScheduleState":
        """State with only the events where mask is True placed (vectorised)."""
        from .evaluator import occupancy

        st = cls(prob)
        idx = np.flatnonzero(mask)
        big = np.int64(-(10**9))
        s2 = np.where(mask, slot, big)  # unplaced events point far outside the grid
        cells = s2[prob.rep_e] + prob.rep_off
        keep = cells >= 0
        T = prob.T
        st.room_occ[:] = np.bincount(room[prob.rep_e][keep] * T + cells[keep], minlength=prob.R * T).reshape(prob.R, T)
        f_e = fac[prob.ev_off]
        st.fac_occ[:] = np.bincount(f_e[prob.rep_e][keep] * T + cells[keep], minlength=prob.F * T).reshape(prob.F, T)
        g = s2[prob.grp_e] + prob.grp_off
        gk = g >= 0
        st.grp_occ[:] = np.bincount(prob.grp_g[gk] * T + g[gk], minlength=prob.G * T).reshape(prob.G, T)
        st.fac_day[:] = st.fac_occ.reshape(prob.F, prob.D, prob.P).sum(axis=2)
        th = idx[prob.ev_theory[idx]]
        np.add.at(st.off_day, (prob.ev_off[th], slot[th] // prob.P), 1)
        st.fac_load[:] = np.bincount(fac, weights=prob.off_need, minlength=prob.F).astype(np.int64)
        return st

    def fits(self, e: int, t: int, r: int, f: int) -> bool:
        """Fast check of one position (same rules as options)."""
        p = self.p
        L = p.ev_len[e]
        cs = slice(t, t + L)
        if self.room_occ[r, cs].any() or p.room_block[r, cs].any():
            return False
        if self.fac_occ[f, cs].any() or p.fac_block[f, cs].any():
            return False
        for g in p.ev_groups[e]:
            if self.grp_occ[g, cs].any():
                return False
        d = t // p.P
        if self.fac_day[f, d] + L > p.fac_daily[f]:
            return False
        if p.ev_theory[e] and self.off_day[p.ev_off[e], d] > 0:
            return False
        return True

    def place(self, e: int, t: int, r: int, f: int, sign: int = 1) -> None:
        p = self.p
        L = p.ev_len[e]
        cs = slice(t, t + L)
        self.room_occ[r, cs] += sign
        self.fac_occ[f, cs] += sign
        for g in p.ev_groups[e]:
            self.grp_occ[g, cs] += sign
        d = t // p.P
        self.fac_day[f, d] += sign * L
        if p.ev_theory[e]:
            self.off_day[p.ev_off[e], d] += sign

    def remove(self, e: int, t: int, r: int, f: int) -> None:
        self.place(e, t, r, f, -1)

    def add_load(self, k: int, f: int, sign: int = 1) -> None:
        self.fac_load[f] += sign * self.p.off_need[k]

    def options(self, e: int, f: int) -> tuple[np.ndarray, np.ndarray]:
        """All feasible (start, room) pairs for event e taught by faculty f."""
        p = self.p
        starts = p.valid_starts[e]
        cells = self.cells[e]
        ok = np.ones(len(starts), dtype=bool)
        for g in p.ev_groups[e]:
            ok &= ~(self.grp_occ[g][cells] > 0).any(axis=1)
        ok &= ~((self.fac_occ[f][cells] > 0) | p.fac_block[f][cells]).any(axis=1)
        days = starts // p.P
        ok &= self.fac_day[f, days] + p.ev_len[e] <= p.fac_daily[f]
        if p.ev_theory[e]:
            ok &= self.off_day[p.ev_off[e], days] == 0
        if not ok.any():
            return starts[:0], starts[:0]
        rooms = p.valid_rooms[e]
        busy = (self.room_occ[rooms][:, cells] > 0) | p.room_block[rooms][:, cells]  # (Rv, S, L)
        free = ~busy.any(axis=2) & ok[None, :]
        ri, si = np.nonzero(free)
        return starts[si], rooms[ri]

    def least_conflict(self, e: int, f: int, rng: np.random.Generator) -> tuple[int, int]:
        """Fallback when nothing is feasible: the (start, room) with the fewest clashes."""
        p = self.p
        cells = self.cells[e]
        cost = np.zeros(len(p.valid_starts[e]))
        for g in p.ev_groups[e]:
            cost += self.grp_occ[g][cells].sum(axis=1)
        cost += self.fac_occ[f][cells].sum(axis=1) + p.fac_block[f][cells].sum(axis=1)
        rooms = p.valid_rooms[e]
        rcost = self.room_occ[rooms][:, cells].sum(axis=2)  # (Rv, S)
        total = rcost + cost[None, :] + rng.random(rcost.shape) * 0.1
        ri, si = np.unravel_index(np.argmin(total), total.shape)
        return int(p.valid_starts[e][si]), int(rooms[ri])
