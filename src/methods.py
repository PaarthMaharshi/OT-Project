"""Every comparison method behind one interface, so experiments give each the same evaluation budget.

    result = run_method("nsga2", prob, budget=3000, seed=1)

Methods (proposal Sections 10-12):
    greedy              Greedy / DSatur (deterministic, one timetable)
    random_search       random timetables made feasible by repair
    weighted_ga         single-objective GA, one run per weight vector
    simulated_annealing weighted-sum SA, one run per weight vector
    nsga2               memetic NSGA-II (seeding + repair + local search)  <- main method
    nsga3               NSGA-III with the same operators and hybrids
    cpsat               exact CP-SAT reference, time-limited (needs OR-Tools)
  ablation variants of NSGA-II:
    nsga2_plain, nsga2_seed, nsga2_repair, nsga2_ls, nsga2_seed_repair
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .algorithms import cpsat_reference
from .algorithms.nsga2 import NSGA2Config, run_nsga2
from .algorithms.nsga3 import run_nsga3
from .algorithms.weighted_ga import run_weighted_ga
from .baseline.greedy_dsatur import greedy
from .baseline.random_search import random_search
from .baseline.simulated_annealing import run_simulated_annealing
from .multiobjective.non_dominated_sort import non_dominated_mask
from .problem import Problem, Solution

MAIN_METHOD = "nsga2"

ABLATION = {
    "nsga2_plain": dict(use_seeding=False, use_repair=False, use_local_search=False),
    "nsga2_seed": dict(use_seeding=True, use_repair=False, use_local_search=False),
    "nsga2_repair": dict(use_seeding=False, use_repair=True, use_local_search=False),
    "nsga2_ls": dict(use_seeding=False, use_repair=False, use_local_search=True),
    "nsga2_seed_repair": dict(use_seeding=True, use_repair=True, use_local_search=False),
}

COMPARISON = ["greedy", "random_search", "weighted_ga", "simulated_annealing", "nsga3", "nsga2"]
ALL_METHODS = COMPARISON + list(ABLATION) + ["cpsat"]

LABELS = {
    "greedy": "Greedy / DSatur",
    "random_search": "Random search",
    "weighted_ga": "Weighted-sum GA",
    "simulated_annealing": "Simulated annealing",
    "nsga2": "Memetic NSGA-II",
    "nsga3": "NSGA-III",
    "cpsat": "CP-SAT reference",
    "nsga2_plain": "NSGA-II (plain)",
    "nsga2_seed": "NSGA-II + seeding",
    "nsga2_repair": "NSGA-II + repair",
    "nsga2_ls": "NSGA-II + local search",
    "nsga2_seed_repair": "NSGA-II + seeding + repair",
}

DETERMINISTIC = {"greedy"}  # same answer for every seed: run once


@dataclass
class MethodResult:
    method: str
    points: list[Solution]  # non-dominated clash-free timetables returned
    evaluations: int
    runtime: float
    final_feasible_fraction: float  # share of the final population / returned set with V = 0
    convergence: list[dict] = field(default_factory=list)  # per generation (EA methods only)
    info: dict = field(default_factory=dict)


def _nondominated(sols: list[Solution]) -> list[Solution]:
    feas = [s for s in sols if s.V == 0]
    if not feas:
        return []
    F = np.array([s.F for s in feas])
    # drop exact duplicates, keep first
    _, idx = np.unique(np.round(F, 9), axis=0, return_index=True)
    feas = [feas[i] for i in sorted(idx)]
    mask = non_dominated_mask(np.array([s.F for s in feas]))
    return [s for s, m in zip(feas, mask) if m]


def run_method(name: str, prob: Problem, budget: int, seed: int, pop_size: int = 60, cpsat_seconds: float = 20.0) -> MethodResult:
    t0 = time.time()
    cfg = NSGA2Config(pop_size=pop_size, generations=10**9, max_evaluations=budget, seed=seed)

    if name == "greedy":
        s = greedy(prob)
        return MethodResult(name, _nondominated([s]), 1, time.time() - t0, float(s.V == 0))

    if name == "random_search":
        pts = random_search(prob, budget, seed)
        return MethodResult(name, pts, budget, time.time() - t0, float(bool(pts)))

    if name == "weighted_ga":
        best, used = run_weighted_ga(prob, NSGA2Config(**{**cfg.__dict__, "pop_size": max(20, pop_size // 2)}))
        return MethodResult(name, _nondominated(best), used, time.time() - t0, sum(s.V == 0 for s in best) / len(best))

    if name == "simulated_annealing":
        best, used = run_simulated_annealing(prob, budget, seed)
        return MethodResult(name, _nondominated(best), used, time.time() - t0, sum(s.V == 0 for s in best) / len(best))

    if name in ("nsga2", "nsga3") or name in ABLATION:
        if name in ABLATION:
            cfg = NSGA2Config(**{**cfg.__dict__, **ABLATION[name]})
        res = (run_nsga3 if name == "nsga3" else run_nsga2)(prob, cfg)
        conv = [{"generation": h["generation"], "evaluations": h["evaluations"], "front": h["front"]} for h in res.history]
        return MethodResult(
            name,
            _nondominated(res.archive),
            res.evaluations,
            time.time() - t0,
            sum(s.V == 0 for s in res.population) / len(res.population),
            conv,
            {"generations": res.generations_run},
        )

    if name == "cpsat":
        if not cpsat_reference.available():
            raise RuntimeError("cpsat needs OR-Tools: pip install ortools")
        runs = cpsat_reference.run_cpsat(prob, cpsat_seconds, seed=seed)
        sols = [r.solution for r in runs if r.solution is not None]
        info = {"solves": [{"weight": r.weight, "status": r.status, "objective": r.objective, "bound": r.bound, "seconds": r.seconds} for r in runs]}
        return MethodResult(name, _nondominated(sols), 0, time.time() - t0, len(sols) / len(runs), info=info)

    raise ValueError(f"Unknown method '{name}'. Choose from: {', '.join(ALL_METHODS)}")
