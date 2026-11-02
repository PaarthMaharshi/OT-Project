"""Random feasible search baseline: random timetables made feasible by repair, same evaluation budget."""
from __future__ import annotations

import numpy as np

from ..hybrid.repair import repair
from ..hybrid.seeding import random_solution
from ..multiobjective.non_dominated_sort import non_dominated_mask
from ..problem import Problem, Solution


def random_search(prob: Problem, evaluations: int, seed: int = 0) -> list[Solution]:
    rng = np.random.default_rng(seed)
    feasible: list[Solution] = []
    used = 0
    while used < evaluations:
        s = random_solution(prob, rng)
        used += 1
        if s.V > 0:
            s = repair(prob, s, rng)
            used += 1
        if s.V == 0:
            feasible.append(s)
    if not feasible:
        return []
    mask = non_dominated_mask(np.array([s.F for s in feasible]))
    return [s for s, m in zip(feasible, mask) if m]
