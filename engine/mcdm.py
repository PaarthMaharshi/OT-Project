"""Decision making on a front: TOPSIS (catalogue no. 8) and VIKOR (no. 9). All criteria are COSTS
(smaller is better), because every timetable objective is minimised.

TOPSIS steps
    1. Vector-normalise each column: r_ij = f_ij / sqrt(sum_i f_ij^2).
    2. Weight: v_ij = w_j r_ij.
    3. Ideal A+ = column minimum, anti-ideal A- = column maximum (cost criteria).
    4. Distances S+_i = ||v_i - A+||, S-_i = ||v_i - A-||.
    5. Closeness C_i = S-_i / (S+_i + S-_i); rank by C descending (1 = best).

VIKOR steps
    1. Best f*_j = column minimum, worst f-_j = column maximum.
    2. S_i = sum_j w_j (f_ij - f*_j)/(f-_j - f*_j)    (group utility)
       R_i = max_j w_j (f_ij - f*_j)/(f-_j - f*_j)    (individual regret)
    3. Q_i = v (S_i - S*)/(S- - S*) + (1 - v)(R_i - R*)/(R- - R*); rank by Q ascending.
    4. Compromise check: C1 acceptable advantage Q(a2) - Q(a1) >= 1/(m-1);
       C2 acceptable stability: a1 is also best by S or by R.
       If C1 fails, the compromise set is every alternative with Q - Q(a1) < 1/(m-1);
       if only C2 fails, it is {a1, a2}.
"""
from __future__ import annotations

import numpy as np


def _weights(w, n) -> np.ndarray:
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    if w.sum() <= 0:
        w = np.ones(n)
    return w / w.sum()


def topsis(F, w=None) -> dict:
    F = np.asarray(F, dtype=float)
    w = _weights(w, F.shape[1])
    norm = np.sqrt((F**2).sum(axis=0))
    R = np.divide(F, norm, out=np.zeros_like(F), where=norm > 0)
    Vw = R * w
    best, worst = Vw.min(axis=0), Vw.max(axis=0)
    s_plus = np.sqrt(((Vw - best) ** 2).sum(axis=1))
    s_minus = np.sqrt(((Vw - worst) ** 2).sum(axis=1))
    denom = s_plus + s_minus
    C = np.divide(s_minus, denom, out=np.ones_like(s_minus), where=denom > 0)  # identical rows -> 1
    ranking = np.argsort(-C, kind="stable")
    return {"closeness": C, "ranking": ranking}


def vikor(F, w=None, v: float = 0.5) -> dict:
    F = np.asarray(F, dtype=float)
    m, n = F.shape
    w = _weights(w, n)
    f_best, f_worst = F.min(axis=0), F.max(axis=0)
    span = np.where(f_worst - f_best > 0, f_worst - f_best, 1.0)
    d = w * (F - f_best) / span
    S, R = d.sum(axis=1), d.max(axis=1)

    def scaled(a):
        r = a.max() - a.min()
        return (a - a.min()) / r if r > 0 else np.zeros_like(a)

    Q = v * scaled(S) + (1 - v) * scaled(R)
    ranking = np.argsort(Q, kind="stable")
    DQ = 1.0 / (m - 1) if m > 1 else 0.0
    a1 = ranking[0]
    c1 = m < 2 or Q[ranking[1]] - Q[a1] >= DQ - 1e-12
    c2 = S[a1] == S.min() or R[a1] == R.min()
    if c1 and c2:
        compromise = [int(a1)]
    elif not c1:
        compromise = [int(i) for i in ranking if Q[i] - Q[a1] < DQ - 1e-12]
    else:
        compromise = [int(a1), int(ranking[1])]
    return {"S": S, "R": R, "Q": Q, "ranking": ranking, "acceptable_advantage": bool(c1), "acceptable_stability": bool(c2), "compromise_set": compromise}


def weight_sensitivity(F, w, method: str = "topsis", delta: float = 0.1, v: float = 0.5) -> list[dict]:
    """Shift each weight by +delta and by -delta (then renormalise) and report whether the top-ranked
    alternative changes."""
    F = np.asarray(F, dtype=float)
    w = _weights(w, F.shape[1])

    def rank(wt):
        return (topsis(F, wt) if method == "topsis" else vikor(F, wt, v))["ranking"]

    base = rank(w)
    out = []
    for j in range(len(w)):
        for sgn in (+1, -1):
            w2 = w.copy()
            w2[j] = max(0.0, w2[j] + sgn * delta)
            w2 = w2 / w2.sum()
            r = rank(w2)
            out.append({"criterion": j, "shift": sgn * delta, "weights": w2.round(4).tolist(),
                        "top": int(r[0]), "top_changed": bool(r[0] != base[0]), "top3": r[:3].tolist()})
    return out
