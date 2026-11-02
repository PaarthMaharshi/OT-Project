"""Fast non-dominated sorting, constraint domination and crowding distance (Deb et al., 2002)."""
from __future__ import annotations

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    """F: (N, M) objectives to minimise. Returns fronts as index arrays, best first."""
    N = len(F)
    if N == 0:
        return []
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
    dom = le & lt  # dom[i, j]: i dominates j
    n_dominated_by = dom.sum(axis=0)
    fronts, current = [], np.flatnonzero(n_dominated_by == 0)
    while len(current):
        fronts.append(current)
        n_dominated_by = n_dominated_by - dom[current].sum(axis=0)
        n_dominated_by[current] = -1
        current = np.flatnonzero(n_dominated_by == 0)
    return fronts


def constrained_fronts(F: np.ndarray, V: np.ndarray) -> list[np.ndarray]:
    """Constraint domination: feasible solutions (V == 0) always rank above infeasible ones,
    infeasible ones are ranked by violation size."""
    feas = np.flatnonzero(V <= 0)
    infeas = np.flatnonzero(V > 0)
    fronts = [feas[f] for f in fast_non_dominated_sort(F[feas])]
    if len(infeas):
        vals = V[infeas]
        for v in np.unique(vals):
            fronts.append(infeas[vals == v])
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    if n <= 2:
        return np.full(n, np.inf)
    d = np.zeros(n)
    for j in range(m):
        order = np.argsort(F[:, j])
        span = F[order[-1], j] - F[order[0], j]
        d[order[0]] = d[order[-1]] = np.inf
        if span > 0:
            d[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
    return d


def non_dominated_mask(F: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(F), dtype=bool)
    if len(F):
        mask[fast_non_dominated_sort(F)[0]] = True
    return mask
