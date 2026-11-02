"""NSGA-III (Deb & Jain, 2014) for comparison.

Same encoding, operators, seeding and repair as NSGA-II (see nsga2.run_moea); only survivor selection
differs: instead of crowding distance, the last front is thinned by niching around Das-Dennis reference
directions after adaptive normalisation. Constraint handling follows the constrained NSGA-III of
Jain & Deb (2014): infeasible solutions are only kept, by smallest violation, if too few are feasible.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np

from ..multiobjective.non_dominated_sort import constrained_fronts
from ..problem import Problem, Solution
from .nsga2 import NSGA2Config, NSGA2Result, run_moea


def das_dennis(n_obj: int, p: int) -> np.ndarray:
    """All weight vectors with components in {0, 1/p, ..., 1} summing to 1."""
    out = []
    for bars in combinations(range(p + n_obj - 1), n_obj - 1):
        prev, parts = -1, []
        for b in bars:
            parts.append(b - prev - 1)
            prev = b
        parts.append(p + n_obj - 2 - prev)
        out.append(parts)
    return np.array(out, dtype=float) / p


def reference_directions(pop_size: int, n_obj: int = 3) -> np.ndarray:
    p = 1
    while len(das_dennis(n_obj, p + 1)) <= pop_size:
        p += 1
    return das_dennis(n_obj, p)


def _normalise(F: np.ndarray) -> np.ndarray:
    ideal = F.min(axis=0)
    Fp = F - ideal
    m = F.shape[1]
    # Extreme point per axis via achievement scalarising function.
    W = np.eye(m) + 1e-6
    asf = np.max(Fp[:, None, :] / W[None, :, :], axis=2)  # (N, m)
    ext = Fp[np.argmin(asf, axis=0)]
    try:
        b = np.linalg.solve(ext, np.ones(m))
        intercepts = 1.0 / b
        if np.any(~np.isfinite(intercepts)) or np.any(intercepts <= 1e-9):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercepts = Fp.max(axis=0)
    intercepts = np.where(intercepts <= 1e-9, 1.0, intercepts)
    return Fp / intercepts


def _associate(Fn: np.ndarray, refs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = refs / np.linalg.norm(refs, axis=1, keepdims=True)
    proj = Fn @ unit.T  # (N, R)
    dist = np.sqrt(np.maximum((Fn**2).sum(axis=1, keepdims=True) - proj**2, 0))
    niche = dist.argmin(axis=1)
    return niche, dist[np.arange(len(Fn)), niche]


def make_nsga3_survival(refs: np.ndarray) -> Callable[[list[Solution], int, np.random.Generator], list[int]]:
    def survival(merged: list[Solution], n: int, rng) -> list[int]:
        F = np.array([s.F for s in merged])
        V = np.array([s.V for s in merged])
        chosen: list[int] = []
        last = None
        for fr in constrained_fronts(F, V):
            if len(chosen) + len(fr) <= n:
                chosen.extend(fr.tolist())
                if len(chosen) == n:
                    return chosen
            else:
                last = fr
                break
        if last is None:
            return chosen
        if np.any(V[last] > 0):  # infeasible tier: keep lowest violation, random tie-break
            take = rng.permutation(last)[: n - len(chosen)]
            return chosen + take.tolist()

        pool = np.array(chosen + last.tolist())
        Fn = _normalise(F[pool])
        niche, dist = _associate(Fn, refs)
        n_chosen = len(chosen)
        counts = np.bincount(niche[:n_chosen], minlength=len(refs))
        cand = list(range(n_chosen, len(pool)))  # positions in pool belonging to the last front
        need = n - n_chosen
        active = np.ones(len(refs), dtype=bool)
        picked: list[int] = []
        while len(picked) < need:
            avail = np.flatnonzero(active)
            jmin = avail[counts[avail] == counts[avail].min()]
            j = int(rng.choice(jmin))
            members = [i for i in cand if niche[i] == j]
            if not members:
                active[j] = False
                continue
            i = min(members, key=lambda i: dist[i]) if counts[j] == 0 else int(rng.choice(members))
            picked.append(i)
            cand.remove(i)
            counts[j] += 1
        return chosen + [int(pool[i]) for i in picked]

    return survival


def run_nsga3(prob: Problem, cfg: NSGA2Config, progress=None) -> NSGA2Result:
    return run_moea(prob, cfg, make_nsga3_survival(reference_directions(cfg.pop_size)), progress)
