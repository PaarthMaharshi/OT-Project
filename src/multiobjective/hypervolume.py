"""Exact hypervolume for 2 and 3 objectives (minimisation) plus normalisation helpers."""
from __future__ import annotations

import numpy as np

from .non_dominated_sort import non_dominated_mask


def _hv2d(P: np.ndarray, ref: np.ndarray) -> float:
    P = P[np.all(P < ref, axis=1)]
    if not len(P):
        return 0.0
    P = P[np.argsort(P[:, 0])]
    hv, best_y = 0.0, ref[1]
    for x, y in P:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


def hypervolume(F: np.ndarray, ref: np.ndarray) -> float:
    """Volume dominated by F and bounded by ref (slicing along the first objective)."""
    F = np.asarray(F, dtype=float)
    ref = np.asarray(ref, dtype=float)
    F = F[np.all(F < ref, axis=1)]
    if not len(F):
        return 0.0
    F = F[non_dominated_mask(F)]
    if F.shape[1] == 2:
        return _hv2d(F, ref)
    F = F[np.argsort(F[:, 0])]
    hv = 0.0
    for i in range(len(F)):
        upper = F[i + 1, 0] if i + 1 < len(F) else ref[0]
        if upper > F[i, 0]:
            hv += (upper - F[i, 0]) * _hv2d(F[: i + 1, 1:], ref[1:])
    return hv


def normalise(F: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    span = np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
    return (np.asarray(F) - ideal) / span


def normalised_hv(F: np.ndarray, ideal: np.ndarray, nadir: np.ndarray, ref_point: float = 1.1) -> float:
    """Hypervolume after mapping [ideal, nadir] to [0, 1]; reference point fixed at 1.1 per axis."""
    if not len(F):
        return 0.0
    return hypervolume(normalise(F, ideal, nadir), np.full(np.asarray(F).shape[1], ref_point))


def spacing(F: np.ndarray) -> float:
    """Schott's spacing metric (lower = more evenly spread)."""
    F = np.asarray(F, dtype=float)
    if len(F) < 2:
        return 0.0
    d = np.abs(F[:, None, :] - F[None, :, :]).sum(axis=2)
    np.fill_diagonal(d, np.inf)
    dmin = d.min(axis=1)
    return float(np.sqrt(((dmin - dmin.mean()) ** 2).sum() / (len(F) - 1)))
