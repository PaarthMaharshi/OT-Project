"""Seeding (proposal Section 12): start part of the population from randomised Greedy/DSatur runs."""
from __future__ import annotations

import numpy as np

from ..baseline.greedy_dsatur import greedy
from ..evaluator import evaluate
from ..problem import Problem, Solution


def random_solution(prob: Problem, rng: np.random.Generator) -> Solution:
    """Uniformly random genes inside the domains (usually infeasible)."""
    slot = np.array([rng.choice(s) for s in prob.valid_starts], dtype=np.int64)
    room = np.array([rng.choice(r) for r in prob.valid_rooms], dtype=np.int64)
    fac = np.array([rng.choice(f) for f in prob.off_elig], dtype=np.int64)
    return evaluate(prob, Solution(slot, room, fac))


def seeded_population(prob: Problem, n: int, rng: np.random.Generator, seed_fraction: float = 0.5) -> list[Solution]:
    n_seed = max(1, int(round(n * seed_fraction)))
    # The deterministic Greedy/DSatur timetable plus randomised variants of it.
    pop = [greedy(prob, rng)] + [greedy(prob, rng, randomized=True) for _ in range(n_seed - 1)]
    pop += [random_solution(prob, rng) for _ in range(n - n_seed)]
    return pop
