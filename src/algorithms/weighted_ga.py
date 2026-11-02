"""Weighted-sum GA baseline (proposal Section 10).

A single-objective GA over a weighted mix of f1, f2, f3, run once per weight vector. It uses exactly the
same encoding, operators, seeding and repair as NSGA-II, so any difference comes from the scalarisation.
Selection: binary tournament (feasible first, then lower violation, then lower weighted sum);
survival: (mu + lambda) truncation.
"""
from __future__ import annotations

import time

import numpy as np

from ..baseline.greedy_dsatur import greedy
from ..problem import Problem, Solution
from .nsga2 import Finisher, NSGA2Config, crossover, initial_population, mutate
from .scalarise import scalar, scales_from, weight_vectors


def _fitness(pop: list[Solution], w, scales) -> np.ndarray:
    # Lexicographic: violation first, weighted sum second.
    return np.array([s.V * 1e6 + (scalar(s.F, w, scales) if s.V == 0 else 0.0) for s in pop])


def weighted_ga_single(prob: Problem, w: np.ndarray, scales: np.ndarray, cfg: NSGA2Config, rng) -> tuple[Solution, int]:
    fin = Finisher(prob, cfg, rng)
    pop = initial_population(prob, cfg, rng, fin)
    t0 = time.time()
    for _ in range(cfg.generations):
        if cfg.max_evaluations is not None and fin.evals >= cfg.max_evaluations:
            break
        if cfg.max_seconds and time.time() - t0 > cfg.max_seconds:
            break
        fit = _fitness(pop, w, scales)

        def pick():
            a, b = rng.integers(len(pop), size=2)
            return pop[a] if fit[a] <= fit[b] else pop[b]

        children = []
        while len(children) < cfg.pop_size and (cfg.max_evaluations is None or fin.evals < cfg.max_evaluations):
            c1, c2 = crossover(prob, pick(), pick(), rng) if rng.random() < cfg.crossover_rate else (pick().copy(), pick().copy())
            for c in (c1, c2):
                mutate(prob, c, cfg.mutation_events, rng)
                children.append(fin(c))
        merged = pop + children
        order = np.argsort(_fitness(merged, w, scales), kind="stable")
        pop = [merged[i] for i in order[: cfg.pop_size]]
    best = min(pop, key=lambda s: (s.V, scalar(s.F, w, scales)))
    return best, fin.evals


def run_weighted_ga(prob: Problem, cfg: NSGA2Config, n_weights: int | None = None) -> tuple[list[Solution], int]:
    """Returns the best timetable for each weight vector, and the evaluations used."""
    rng = np.random.default_rng(cfg.seed)
    scales = scales_from(greedy(prob).F)
    W = weight_vectors(n_weights)
    per = None if cfg.max_evaluations is None else max(cfg.pop_size * 2, cfg.max_evaluations // len(W))
    sub = NSGA2Config(**{**cfg.__dict__, "max_evaluations": per})
    best, used = [], 0
    for w in W:
        s, n = weighted_ga_single(prob, w, scales, sub, rng)
        best.append(s)
        used += n
    return best, used
