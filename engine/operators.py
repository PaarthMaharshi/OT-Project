"""Integer-aware variation operators for the (slot, room) encoding.

A session's slot gene and room gene always travel together, so crossover never splits a session.
The encoding is not a permutation (two sessions may share a slot), so order crossover (OX) does not
apply; uniform crossover is used instead.
"""
from __future__ import annotations

import numpy as np

from .problem import TimetableProblem


def uniform_crossover(p: TimetableProblem, a, b, rng) -> tuple[np.ndarray, np.ndarray]:
    """Each session's (slot, room) pair comes from parent a or parent b with probability 1/2."""
    take_a = rng.random(p.S) < 0.5
    mask = np.concatenate([take_a, take_a])  # same choice for the slot gene and the room gene
    return np.where(mask, a, b), np.where(mask, b, a)


def swap_mutation(p: TimetableProblem, x, rng) -> None:
    """Swap the time slots of two sessions (rooms stay). In place."""
    s1, s2 = rng.choice(p.S, size=2, replace=False)
    x[s1], x[s2] = x[s2], x[s1]


def slot_reassign_mutation(p: TimetableProblem, x, rng) -> None:
    """Move one session to a uniformly random slot. In place."""
    x[rng.integers(p.S)] = rng.integers(p.T)


def room_reassign_mutation(p: TimetableProblem, x, rng) -> None:
    """Give one session a uniformly random room. In place."""
    x[p.S + rng.integers(p.S)] = rng.integers(p.R)


def mutate(p: TimetableProblem, x, rng, rate: float | None = None) -> np.ndarray:
    """Apply on average `rate * S` (default 1 per child) mutations, each one of:
    swap (40 %), slot reassign (40 %), room reassign (20 %). Returns a new vector."""
    y = np.array(x, copy=True)
    rate = (1.0 / p.S) if rate is None else rate
    n = max(1, rng.binomial(p.S, min(1.0, rate)))
    for _ in range(n):
        u = rng.random()
        if u < 0.4:
            swap_mutation(p, y, rng)
        elif u < 0.8:
            slot_reassign_mutation(p, y, rng)
        else:
            room_reassign_mutation(p, y, rng)
    return y
