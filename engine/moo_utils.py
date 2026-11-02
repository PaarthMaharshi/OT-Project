"""Shared multi-objective toolkit, hand-coded with NumPy only.

Everything here assumes MINIMISATION of every objective.

Contents
    dominates                 Pareto dominance a < b
    constrained_dominates     Deb's three feasibility rules
    fast_non_dominated_sort   Deb et al. (2002), optionally with constrained dominance
    crowding_distance         NSGA-II density estimate
    nondominated_mask         which rows of F are non-dominated
    normalize                 map [ideal, nadir] -> [0, 1]
    hv_2d                     exact 2-D hypervolume (sweep line)
    hv_monte_carlo            Monte-Carlo hypervolume for 3+ objectives
    hypervolume               picks the right one of the two
    igd                       inverted generational distance
    spacing                   Schott's spacing metric
    merge_fronts              reference front = non-dominated union of many fronts
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Dominance
# ---------------------------------------------------------------------------
def dominates(a, b) -> bool:
    """True if a Pareto-dominates b: a is no worse in every objective and strictly better in one."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(a <= b) and np.any(a < b))


def constrained_dominates(a, b, va: float, vb: float) -> bool:
    """Deb's three feasibility rules (constraint violation va, vb >= 0; 0 means feasible).

    Rule 1: a feasible solution beats an infeasible one.
    Rule 2: between two infeasible solutions, the smaller violation wins.
    Rule 3: between two feasible solutions, ordinary Pareto dominance decides.
    """
    a_feasible, b_feasible = va <= 0, vb <= 0
    if a_feasible and not b_feasible:  # rule 1
        return True
    if not a_feasible and b_feasible:  # rule 1 (other way round)
        return False
    if not a_feasible and not b_feasible:  # rule 2
        return va < vb
    return dominates(a, b)  # rule 3


def dominance_matrix(F: np.ndarray, V: np.ndarray | None = None) -> np.ndarray:
    """D[i, j] = True if solution i (constrained-)dominates solution j.

    Vectorised version of the two functions above, used by the sorting routines for speed.
    """
    F = np.asarray(F, dtype=float)
    no_worse = (F[:, None, :] <= F[None, :, :]).all(axis=2)  # i <= j in every objective
    better = (F[:, None, :] < F[None, :, :]).any(axis=2)  # i < j in at least one
    pareto = no_worse & better
    if V is None:
        return pareto
    V = np.asarray(V, dtype=float)
    feas = V <= 0
    D = np.zeros_like(pareto)
    D |= feas[:, None] & ~feas[None, :]  # rule 1
    both_inf = ~feas[:, None] & ~feas[None, :]
    D |= both_inf & (V[:, None] < V[None, :])  # rule 2
    D |= feas[:, None] & feas[None, :] & pareto  # rule 3
    return D


# ---------------------------------------------------------------------------
# Sorting and density
# ---------------------------------------------------------------------------
def fast_non_dominated_sort(F, V=None) -> list[np.ndarray]:
    """Deb's fast non-dominated sort.

    For every solution p we know S_p (the set it dominates) and n_p (how many dominate it).
    Front 1 = all p with n_p == 0. Removing front k lowers n_q of the solutions it dominates;
    those that reach 0 form front k+1. Pass V to sort with Deb's constrained dominance.
    Returns a list of index arrays, best front first.
    """
    F = np.asarray(F, dtype=float)
    n = len(F)
    if n == 0:
        return []
    D = dominance_matrix(F, V)  # D[p, q]: p dominates q
    n_dominated_by = D.sum(axis=0).astype(int)  # n_p for every p
    fronts = []
    current = np.flatnonzero(n_dominated_by == 0)
    assigned = np.zeros(n, dtype=bool)
    while len(current):
        fronts.append(current)
        assigned[current] = True
        # every q dominated by a member of the current front loses one dominator
        n_dominated_by = n_dominated_by - D[current].sum(axis=0)
        current = np.flatnonzero((n_dominated_by == 0) & ~assigned)
    return fronts


def crowding_distance(F) -> np.ndarray:
    """NSGA-II crowding distance of the points in ONE front.

    For each objective: sort the front, give the two boundary points infinite distance and every
    other point the normalised gap between its two neighbours. Sum over objectives.
    """
    F = np.asarray(F, dtype=float)
    n, m = F.shape
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    for j in range(m):
        order = np.argsort(F[:, j], kind="stable")
        f_min, f_max = F[order[0], j], F[order[-1], j]
        dist[order[0]] = dist[order[-1]] = np.inf
        if f_max - f_min > 0:
            dist[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / (f_max - f_min)
    return dist


def nondominated_mask(F, V=None) -> np.ndarray:
    """Boolean mask of the first (constrained) front."""
    F = np.asarray(F, dtype=float)
    mask = np.zeros(len(F), dtype=bool)
    if len(F):
        mask[fast_non_dominated_sort(F, V)[0]] = True
    return mask


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize(F, ideal, nadir) -> np.ndarray:
    """Linear map so that ideal -> 0 and nadir -> 1 (objectives with zero range are left as F - ideal)."""
    F = np.asarray(F, dtype=float)
    ideal = np.asarray(ideal, dtype=float)
    span = np.asarray(nadir, dtype=float) - ideal
    span = np.where(span > 1e-12, span, 1.0)
    return (F - ideal) / span


# ---------------------------------------------------------------------------
# Quality indicators
# ---------------------------------------------------------------------------
def hv_2d(F, ref) -> float:
    """Exact hypervolume for two objectives.

    Keep the points that dominate the reference point, sort them by f1 ascending and add up the
    rectangles of the resulting staircase: width = distance to the next point's f1 (or to ref),
    height = ref_f2 - lowest f2 seen so far.
    """
    F = np.asarray(F, dtype=float)
    ref = np.asarray(ref, dtype=float)
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[nondominated_mask(F)]
    F = F[np.argsort(F[:, 0], kind="stable")]
    volume = 0.0
    for i in range(len(F)):
        next_x = F[i + 1, 0] if i + 1 < len(F) else ref[0]
        volume += (next_x - F[i, 0]) * (ref[1] - F[i, 1])
    return float(volume)


def hv_monte_carlo(F, ref, n_samples: int = 100_000, seed: int = 12345, lower=None) -> float:
    """Monte-Carlo hypervolume for 3 or more objectives.

    Draw points uniformly in the box [lower, ref]; the fraction that is dominated by at least one
    point of F, times the box volume, estimates the hypervolume. `lower` defaults to the per-objective
    minimum of F (nothing below it can be dominated). A fixed seed makes the estimate repeatable.
    Standard error ~ box_volume * sqrt(p(1-p)/n_samples).
    """
    F = np.asarray(F, dtype=float)
    ref = np.asarray(ref, dtype=float)
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    lower = F.min(axis=0) if lower is None else np.minimum(np.asarray(lower, dtype=float), F.min(axis=0))
    box = float(np.prod(ref - lower))
    rng = np.random.default_rng(seed)
    hits = 0
    done = 0
    chunk = 20_000
    while done < n_samples:
        k = min(chunk, n_samples - done)
        S = lower + rng.random((k, len(ref))) * (ref - lower)
        # a sample is dominated if some point of F is <= it in every objective
        dominated = np.zeros(k, dtype=bool)
        for p in F:
            dominated |= np.all(p <= S, axis=1)
        hits += int(dominated.sum())
        done += k
    return box * hits / n_samples


def hypervolume(F, ref, n_samples: int = 100_000, seed: int = 12345) -> float:
    """Exact for two objectives, Monte-Carlo otherwise."""
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return 0.0
    if F.shape[1] == 2:
        return hv_2d(F, ref)
    return hv_monte_carlo(F, ref, n_samples, seed, lower=np.zeros(F.shape[1]) if np.all(F >= 0) else None)


def igd(A, Z) -> float:
    """Inverted generational distance: mean distance from every reference point z in Z to its
    nearest point in the approximation A (Euclidean, on whatever scale the caller passes).
    Lower is better; 0 means A covers Z exactly."""
    A = np.asarray(A, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if len(A) == 0:
        return float("inf")
    d = np.sqrt(((Z[:, None, :] - A[None, :, :]) ** 2).sum(axis=2))  # |Z| x |A|
    return float(d.min(axis=1).mean())


def spacing(F) -> float:
    """Schott's spacing: standard deviation of each point's L1 distance to its nearest neighbour.
    0 means perfectly even spacing. Defined as 0 for fewer than two points."""
    F = np.asarray(F, dtype=float)
    n = len(F)
    if n < 2:
        return 0.0
    d = np.abs(F[:, None, :] - F[None, :, :]).sum(axis=2)
    np.fill_diagonal(d, np.inf)
    d_min = d.min(axis=1)
    return float(np.sqrt(((d_min.mean() - d_min) ** 2).sum() / (n - 1)))


def merge_fronts(fronts: list) -> np.ndarray:
    """Reference front: union of several fronts, duplicates removed, dominated points removed.
    Used because the true Pareto front of a timetabling instance is unknown."""
    parts = [np.asarray(f, dtype=float) for f in fronts if len(f)]
    if not parts:
        return np.zeros((0, 0))
    U = np.unique(np.vstack(parts), axis=0)
    return U[nondominated_mask(U)]
