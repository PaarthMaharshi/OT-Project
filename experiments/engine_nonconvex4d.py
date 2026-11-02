"""Which points of the 4-objective reference front can a weighted sum reach at all?

    python -m experiments.engine_nonconvex4d        (run after experiments.engine_lab)

A point is SUPPORTED if some weight vector w >= 0 makes it the minimiser of w . f_normalised over the
reference front. 200,000 random weight vectors (uniform on the simplex, plus its corners) are tried.
Points that are never the minimiser lie in non-convex parts of the front: no weighted sum finds them,
however many weights are swept. Writes results/lab/nonconvex_4d.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "results" / "lab"


def supported_points(F: np.ndarray, n_weights: int = 200_000, seed: int = 0) -> np.ndarray:
    lo, hi = F.min(axis=0), F.max(axis=0)
    Fn = (F - lo) / np.where(hi > lo, hi - lo, 1.0)  # same normalisation as the lab tables
    rng = np.random.default_rng(seed)
    W = np.vstack([np.eye(F.shape[1]), rng.dirichlet(np.ones(F.shape[1]), n_weights)])
    mask = np.zeros(len(F), dtype=bool)
    for chunk in np.array_split(W, 100):
        S = chunk @ Fn.T  # weighted sums, one row per weight vector
        mask |= np.isclose(S, S.min(axis=1, keepdims=True)).any(axis=0)
    return mask


def main():
    ref = json.loads((LAB / "reference_fronts.json").read_text())
    F = np.array(ref["student_gaps+teacher_gaps+room_waste+time_penalty"], dtype=float)
    sup = supported_points(F)
    out = {
        "reference_points": int(len(F)),
        "supported": int(sup.sum()),
        "unsupported": int((~sup).sum()),
        "unsupported_points": F[~sup].tolist(),
        "method": "200,000 random simplex weights + corners; a point is supported if it minimises the weighted sum for at least one weight",
    }
    (LAB / "nonconvex_4d.json").write_text(json.dumps(out, indent=2))
    print(f"4-objective reference front: {out['reference_points']} points, {out['supported']} supported, {out['unsupported']} unsupported")


if __name__ == "__main__":
    main()
