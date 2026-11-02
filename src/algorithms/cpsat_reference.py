"""Exact CP-SAT reference (proposal Sections 8 and 10), using Google OR-Tools.

Model (proposal Section 7):
  x[e,t,r]  event e starts at slot t in room r   (only valid starts / fitting rooms are created: C5, C6)
  y[k,f]    teacher f teaches offering k
  b[e,f,c]  = occupancy of cell c by e  AND  y[k(e),f]   (teacher-busy, linearised exactly)
Hard constraints C1-C8 are all modelled. The objective is the same weighted sum used by the weighted GA
and SA, made linear:
  f1  gaps: idle[p] >= before[p] + after[p] - busy[p] - 1, with before/after the running OR of busy
  f2  room wastage: linear in x
  f3' load spread: max(load/H) - min(load/H) instead of the standard deviation (not linear)
Each solution is re-scored by the project's own evaluator (true f1, f2, f3 and V), so the reference is
checked independently. Intended for small and medium instances; large ones may hit the time limit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..baseline.greedy_dsatur import greedy
from ..evaluator import LAMBDA_FACULTY_GAPS, evaluate
from ..problem import Problem, Solution
from .scalarise import scales_from, weight_vectors

try:
    from ortools.sat.python import cp_model
except ImportError:  # optional dependency
    cp_model = None

COEF = 1_000_000  # objective scaling to integers
LOAD_RES = 1000  # relative load resolution for f3'


@dataclass
class CPSATRun:
    weight: list[float]
    status: str
    objective: float | None
    bound: float | None
    seconds: float
    solution: Solution | None


def available() -> bool:
    return cp_model is not None


def _gap_vars(m, busy: list, lunch: int) -> list:
    """Idle indicators for one row (one day) of 0/1 busy expressions."""
    P = len(busy)
    before = [m.NewBoolVar("") for _ in range(P)]
    after = [m.NewBoolVar("") for _ in range(P)]
    m.Add(before[0] == 0)
    m.Add(after[P - 1] == 0)
    for p in range(1, P):
        m.Add(before[p] >= busy[p - 1])
        m.Add(before[p] >= before[p - 1])
    for p in range(P - 2, -1, -1):
        m.Add(after[p] >= busy[p + 1])
        m.Add(after[p] >= after[p + 1])
    idle = []
    for p in range(P):
        if p == lunch:
            continue
        v = m.NewBoolVar("")
        m.Add(v >= before[p] + after[p] - busy[p] - 1)
        idle.append(v)
    return idle


def solve_weighted(prob: Problem, w: np.ndarray, scales: np.ndarray, time_limit: float, hint: Solution | None, workers: int = 8, seed: int = 0) -> CPSATRun:
    if cp_model is None:
        raise RuntimeError("OR-Tools is not installed: pip install ortools")
    t0 = time.time()
    m = cp_model.CpModel()
    D, P, T = prob.D, prob.P, prob.T

    # --- decision variables ---
    x: dict[tuple[int, int, int], object] = {}
    start: dict[tuple[int, int], list] = {}  # (e, t) -> x vars over rooms
    for e in range(prob.E):
        L = prob.ev_len[e]
        for t in prob.valid_starts[e]:
            t = int(t)
            for r in prob.valid_rooms[e]:
                r = int(r)
                if prob.room_block[r, t : t + L].any():
                    continue
                v = m.NewBoolVar(f"x{e}_{t}_{r}")
                x[e, t, r] = v
                start.setdefault((e, t), []).append(v)
        m.AddExactlyOne([v for (ee, _, _), v in x.items() if ee == e])  # C1

    y = {(k, int(f)): m.NewBoolVar(f"y{k}_{f}") for k in range(prob.K) for f in prob.off_elig[k]}
    for k in range(prob.K):
        m.AddExactlyOne([y[k, int(f)] for f in prob.off_elig[k]])  # C1

    # occupancy of cell c by event e (0/1 linear expression)
    occ: dict[tuple[int, int], list] = {}
    for (e, t), vs in start.items():
        for o in range(prob.ev_len[e]):
            occ.setdefault((e, t + o), []).extend(vs)

    # C2 rooms
    room_cell: dict[tuple[int, int], list] = {}
    for (e, t, r), v in x.items():
        for o in range(prob.ev_len[e]):
            room_cell.setdefault((r, t + o), []).append(v)
    for vs in room_cell.values():
        if len(vs) > 1:
            m.AddAtMostOne(vs)

    # C4 students
    grp_cell: dict[tuple[int, int], list] = {}
    for (e, c), vs in occ.items():
        for g in prob.ev_groups[e]:
            grp_cell.setdefault((g, c), []).extend(vs)
    for vs in grp_cell.values():
        m.Add(sum(vs) <= 1)

    # C8 at most one theory session of an offering per day
    for k in range(prob.K):
        evs = prob.off_events[k]
        if not prob.ev_theory[evs[0]]:
            continue
        for d in range(D):
            vs = [v for e in evs for t in range(d * P, (d + 1) * P) for v in start.get((int(e), t), [])]
            if len(vs) > 1:
                m.Add(sum(vs) <= 1)

    # Teacher-busy b[e,f,c] = occ(e,c) AND y[k(e),f]; C3, C6, C7
    fac_cell: dict[tuple[int, int], list] = {}
    for (e, c), vs in occ.items():
        k = int(prob.ev_off[e])
        o = sum(vs)
        for f in prob.off_elig[k]:
            f = int(f)
            yv = y[k, f]
            if prob.fac_block[f, c]:
                m.Add(o + yv <= 1)  # C6
                continue
            b = m.NewBoolVar("")
            m.Add(b <= o)
            m.Add(b <= yv)
            m.Add(b >= o + yv - 1)
            fac_cell.setdefault((f, c), []).append(b)
    for vs in fac_cell.values():
        if len(vs) > 1:
            m.AddAtMostOne(vs)  # C3
    fbusy = {fc: sum(vs) for fc, vs in fac_cell.items()}
    for f in range(prob.F):
        m.Add(sum(int(prob.off_need[k]) * y[k, f] for k in range(prob.K) if (k, f) in y) <= int(prob.fac_max[f]))  # C7 week
        for d in range(D):
            cells = [fbusy[f, c] for c in range(d * P, (d + 1) * P) if (f, c) in fbusy]
            if cells:
                m.Add(sum(cells) <= int(prob.fac_daily[f]))  # C7 day

    # --- objective ---
    terms = []
    c1 = w[0] / scales[0] * COEF
    zero = 0
    for g in range(prob.G):
        for d in range(D):
            busy = [sum(grp_cell.get((g, d * P + p), [])) if (g, d * P + p) in grp_cell else zero for p in range(P)]
            for v in _gap_vars(m, busy, prob.lunch):
                terms.append(int(round(c1)) * v)
    for f in range(prob.F):
        for d in range(D):
            busy = [fbusy.get((f, d * P + p), zero) for p in range(P)]
            for v in _gap_vars(m, busy, prob.lunch):
                terms.append(int(round(c1 * LAMBDA_FACULTY_GAPS)) * v)

    total_len = float(prob.ev_len.sum())
    c2 = w[1] / scales[1] * COEF
    for (e, t, r), v in x.items():
        waste = prob.ev_len[e] * (prob.room_cap[r] - prob.ev_head[e]) / prob.room_cap[r] / total_len
        coef = int(round(c2 * waste))
        if coef:
            terms.append(coef * v)

    hi = m.NewIntVar(0, LOAD_RES, "rel_max")
    lo = m.NewIntVar(0, LOAD_RES, "rel_min")
    for f in range(prob.F):
        load = sum(int(prob.off_need[k]) * y[k, f] for k in range(prob.K) if (k, f) in y)
        H = int(prob.fac_max[f])
        m.Add(load * LOAD_RES <= hi * H)
        m.Add(load * LOAD_RES >= lo * H)
    c3 = int(round(w[2] / scales[2] * COEF / LOAD_RES))
    terms.append(c3 * (hi - lo))
    m.Minimize(sum(terms))

    if hint is not None:
        # Complete hint (every x and y gets a value); CP-SAT uses complete hints far better than partial ones.
        for (e, t, r), v in x.items():
            m.AddHint(v, int(hint.slot[e] == t and hint.room[e] == r))
        for (k, f), v in y.items():
            m.AddHint(v, int(hint.fac[k] == f))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    status = solver.Solve(m)
    name = solver.StatusName(status)
    sol = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        slot = np.zeros(prob.E, dtype=np.int64)
        room = np.zeros(prob.E, dtype=np.int64)
        for (e, t, r), v in x.items():
            if solver.Value(v):
                slot[e], room[e] = t, r
        fac = np.zeros(prob.K, dtype=np.int64)
        for (k, f), v in y.items():
            if solver.Value(v):
                fac[k] = f
        sol = evaluate(prob, Solution(slot, room, fac))
    return CPSATRun(
        weight=[float(v) for v in w],
        status=name,
        objective=solver.ObjectiveValue() / COEF if sol is not None else None,
        bound=solver.BestObjectiveBound() / COEF if sol is not None else None,
        seconds=time.time() - t0,
        solution=sol,
    )


def run_cpsat(prob: Problem, time_limit_per_weight: float = 20.0, n_weights: int | None = None, seed: int = 0) -> list[CPSATRun]:
    """One CP-SAT solve per weight vector, warm-started from the Greedy timetable."""
    g = greedy(prob)
    scales = scales_from(g.F)
    return [solve_weighted(prob, w, scales, time_limit_per_weight, g if g.V == 0 else None, seed=seed) for w in weight_vectors(n_weights)]
