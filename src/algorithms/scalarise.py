"""Weighted-sum scalarisation shared by the single-objective baselines (weighted GA, SA, CP-SAT).

Objectives have very different units (periods, fractions, std), so each is divided by its value in the
deterministic Greedy timetable before weighting. A weighted sum needs the weights fixed in advance and
can only reach the convex part of the front, which is exactly what these baselines are meant to show.
"""
from __future__ import annotations

import numpy as np

# Corners, edge midpoints and the centre of the weight simplex; a small floor keeps every objective in
# play so the optimum is not weakly dominated.
WEIGHTS = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
        [1 / 3, 1 / 3, 1 / 3],
    ]
)
WEIGHT_FLOOR = 0.02


def weight_vectors(n: int | None = None) -> np.ndarray:
    W = WEIGHTS if n is None else WEIGHTS[:n]
    W = W + WEIGHT_FLOOR
    return W / W.sum(axis=1, keepdims=True)


def scales_from(reference_F: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(reference_F, dtype=float), np.array([1.0, 0.01, 0.01]))


def scalar(F: np.ndarray, w: np.ndarray, scales: np.ndarray) -> float:
    return float(np.dot(w, np.asarray(F) / scales))
