"""Memetic NSGA-II (proposal Sections 11-12).

* Constraint domination: any clash-free timetable beats any timetable with clashes.
* Crossover swaps whole offerings, so all sessions of a subject move together.
* Mutation: move an event, swap two events' slots, change room, change teacher.
* Optional hybrids (for the ablation): seeding, repair, local search.

The generational loop (`run_moea`) is shared with NSGA-III, which only replaces survivor selection.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..evaluator import evaluate
from ..hybrid.local_search import local_search
from ..hybrid.repair import repair
from ..hybrid.seeding import random_solution, seeded_population
from ..multiobjective.non_dominated_sort import constrained_fronts, crowding_distance, dominates
from ..problem import Problem, Solution


@dataclass
class NSGA2Config:
    pop_size: int = 60
    generations: int = 80
    crossover_rate: float = 0.9
    mutation_events: float = 3.0  # expected number of mutated events per child
    use_seeding: bool = True
    seed_fraction: float = 0.5
    use_repair: bool = True
    use_local_search: bool = True
    local_search_rate: float = 0.15
    max_seconds: float | None = None
    max_evaluations: int | None = None  # equal-budget comparisons stop on this
    seed: int = 0


@dataclass
class NSGA2Result:
    archive: list[Solution]  # every non-dominated clash-free timetable found
    population: list[Solution]
    history: list[dict] = field(default_factory=list)
    evaluations: int = 0
    runtime: float = 0.0
    generations_run: int = 0
    feasible_seen: int = 0  # children that came out clash-free
    children_seen: int = 0


# ---------------- variation (shared by all evolutionary methods) ----------------


def crossover(prob: Problem, a: Solution, b: Solution, rng) -> tuple[Solution, Solution]:
    # Whole offerings move together; offerings of the same student group are inherited as a block,
    # which keeps each group's week intact and avoids most student clashes.
    take_group = rng.random(prob.G) < 0.5
    take = take_group[prob.off_group]
    em = take[prob.ev_off]
    c1 = Solution(np.where(em, a.slot, b.slot), np.where(em, a.room, b.room), np.where(take, a.fac, b.fac))
    c2 = Solution(np.where(em, b.slot, a.slot), np.where(em, b.room, a.room), np.where(take, b.fac, a.fac))
    return c1, c2


def mutate(prob: Problem, s: Solution, mutation_events: float, rng) -> None:
    n = rng.binomial(prob.E, min(1.0, mutation_events / prob.E))
    for e in rng.choice(prob.E, size=max(1, n), replace=False):
        op = rng.random()
        if op < 0.4:  # move to another valid slot
            s.slot[e] = rng.choice(prob.valid_starts[e])
        elif op < 0.65:  # swap slots with an event of the same length
            o = rng.choice(prob.same_len[e])
            s.slot[e], s.slot[o] = s.slot[o], s.slot[e]
        elif op < 0.85:  # change room
            s.room[e] = rng.choice(prob.valid_rooms[e])
        else:  # change teacher of this offering
            k = prob.ev_off[e]
            s.fac[k] = rng.choice(prob.off_elig[k])


def key(s: Solution) -> bytes:
    return s.slot.tobytes() + s.room.tobytes() + s.fac.tobytes()


def update_archive(archive: dict, sols: list[Solution]) -> None:
    for s in sols:
        if s.V > 0:
            continue
        k = tuple(np.round(s.F, 9))
        if k in archive or any(dominates(a.F, s.F) for a in archive.values()):
            continue
        for kk in [kk for kk, a in archive.items() if dominates(s.F, a.F)]:
            del archive[kk]
        archive[k] = s


class Finisher:
    """Evaluate a child, then repair / local-search it; counts evaluations."""

    def __init__(self, prob: Problem, cfg: NSGA2Config, rng):
        self.prob, self.cfg, self.rng = prob, cfg, rng
        self.evals = 0

    def __call__(self, s: Solution) -> Solution:
        p, cfg, rng = self.prob, self.cfg, self.rng
        evaluate(p, s)
        self.evals += 1
        if cfg.use_repair and s.V > 0:
            s = repair(p, s, rng)
            self.evals += 1
        if cfg.use_local_search and s.V == 0 and rng.random() < cfg.local_search_rate:
            s = local_search(p, s, rng)
            self.evals += 1
        return s


def initial_population(prob: Problem, cfg: NSGA2Config, rng, fin: Finisher) -> list[Solution]:
    if cfg.use_seeding:
        pop = seeded_population(prob, cfg.pop_size, rng, cfg.seed_fraction)
    else:
        pop = [random_solution(prob, rng) for _ in range(cfg.pop_size)]
    fin.evals += len(pop)
    if cfg.use_repair:
        pop = [fin(s) if s.V > 0 else s for s in pop]
    return pop


# ---------------- selection ----------------


def rank_and_crowd(pop: list[Solution]) -> tuple[np.ndarray, np.ndarray]:
    F = np.array([s.F for s in pop])
    V = np.array([s.V for s in pop])
    rank = np.zeros(len(pop), dtype=int)
    crowd = np.zeros(len(pop))
    for i, fr in enumerate(constrained_fronts(F, V)):
        rank[fr] = i
        crowd[fr] = crowding_distance(F[fr])
    return rank, crowd


def tournament(rank, crowd, rng) -> int:
    a, b = rng.integers(len(rank), size=2)
    if rank[a] != rank[b]:
        return int(a if rank[a] < rank[b] else b)
    return int(a if crowd[a] >= crowd[b] else b)


def nsga2_survival(merged: list[Solution], n: int, rng) -> list[int]:
    F = np.array([s.F for s in merged])
    V = np.array([s.V for s in merged])
    nxt: list[int] = []
    for fr in constrained_fronts(F, V):
        if len(nxt) + len(fr) <= n:
            nxt.extend(fr.tolist())
        else:
            cd = crowding_distance(F[fr])
            nxt.extend(fr[np.argsort(-cd)][: n - len(nxt)].tolist())
            break
    return nxt


Survival = Callable[[list[Solution], int, np.random.Generator], list[int]]


def run_moea(prob: Problem, cfg: NSGA2Config, survival: Survival, progress: Callable[[dict], None] | None = None) -> NSGA2Result:
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()
    fin = Finisher(prob, cfg, rng)
    pop = initial_population(prob, cfg, rng, fin)
    archive: dict = {}
    update_archive(archive, pop)
    history: list[dict] = []
    feas_children = n_children = 0
    gen = 0

    def budget_left() -> bool:
        if cfg.max_evaluations is not None and fin.evals >= cfg.max_evaluations:
            return False
        if cfg.max_seconds and time.time() - t0 > cfg.max_seconds:
            return False
        return True

    for gen in range(1, cfg.generations + 1):
        if not budget_left():
            gen -= 1
            break
        rank, crowd = rank_and_crowd(pop)
        children: list[Solution] = []
        while len(children) < cfg.pop_size and budget_left():
            a, b = pop[tournament(rank, crowd, rng)], pop[tournament(rank, crowd, rng)]
            c1, c2 = crossover(prob, a, b, rng) if rng.random() < cfg.crossover_rate else (a.copy(), b.copy())
            for c in (c1, c2):
                mutate(prob, c, cfg.mutation_events, rng)
                children.append(fin(c))
        children = children[: cfg.pop_size]
        n_children += len(children)
        feas_children += sum(c.V == 0 for c in children)

        seen, merged = set(), []
        for s in pop + children:  # drop exact duplicates first
            k = key(s)
            if k not in seen:
                seen.add(k)
                merged.append(s)
        pop = [merged[i] for i in survival(merged, cfg.pop_size, rng)]
        while len(pop) < cfg.pop_size:  # tiny instances can run out of distinct solutions
            pop.append(pop[rng.integers(len(pop))].copy())

        update_archive(archive, children)
        front = np.array([a.F for a in archive.values()]) if archive else np.zeros((0, 3))
        entry = {
            "generation": gen,
            "evaluations": fin.evals,
            "feasible_fraction": sum(s.V == 0 for s in pop) / len(pop),
            "archive_size": len(archive),
            "best": front.min(axis=0).tolist() if len(front) else None,
            "front": front.tolist(),
            "seconds": time.time() - t0,
        }
        history.append(entry)
        if progress:
            progress(entry)

    return NSGA2Result(
        archive=sorted(archive.values(), key=lambda s: tuple(s.F)),
        population=pop,
        history=history,
        evaluations=fin.evals,
        runtime=time.time() - t0,
        generations_run=gen,
        feasible_seen=feas_children,
        children_seen=n_children,
    )


def run_nsga2(prob: Problem, cfg: NSGA2Config, progress: Callable[[dict], None] | None = None) -> NSGA2Result:
    return run_moea(prob, cfg, nsga2_survival, progress)
