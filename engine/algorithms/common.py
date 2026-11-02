"""Run loop shared by every algorithm.

Each algorithm module provides
    DEFAULTS                      its default parameters
    initialize(problem, params, ctx) -> state
    step(problem, state, ctx) -> state        one generation / iteration batch
    output(state) -> (X, F, V)                the solutions the algorithm reports
and `run()` just calls `run_loop(module, ...)` below.

The Context object owns everything that must be identical for all algorithms:
    * the random generator (seeded -> deterministic runs)
    * evaluation counting and the evaluation budget
    * the time limit, cancellation and the progress callback
    * the hypervolume history
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .. import moo_utils as mu
from ..problem import TimetableProblem

COMMON_DEFAULTS = {
    "evaluations": 10_000,  # evaluation budget
    "seed": 1,
    "time_limit_s": None,  # wall-clock limit; the best front so far is returned when reached
    "history_every": None,  # compute HV every k generations (default 1 for 2 objectives, 10 otherwise)
    "hv_samples": 5_000,  # Monte-Carlo samples for the HV history when M >= 3
}


@dataclass
class RunResult:
    algorithm: str
    X: np.ndarray  # reported solutions (decision vectors)
    F: np.ndarray  # their objectives
    V: np.ndarray  # their constraint violation
    history: list = field(default_factory=list)
    evaluations: int = 0
    feasible_rate: float = 0.0  # share of ALL evaluated solutions that were feasible
    runtime_s: float = 0.0
    stopped: str = "budget"  # budget | time_limit | cancelled | completed
    params: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


class Context:
    def __init__(self, problem: TimetableProblem, params: dict, progress: Callable | None = None, should_stop: Callable | None = None):
        self.problem = problem
        self.params = params
        self.rng = np.random.default_rng(params["seed"])
        self.max_evals = int(params["evaluations"])
        self.time_limit = params.get("time_limit_s")
        self.progress = progress
        self.should_stop = should_stop
        self.cancelled = False
        self.evals = 0
        self.feasible_evals = 0
        self.t0 = time.perf_counter()
        self.history: list[dict] = []
        every = params.get("history_every")
        self.hv_every = every if every else (1 if problem.M == 2 else 10)
        self.hv_nadir = None  # fixed normalisation for the HV history, set from the first batch
        self._last_hv = 0.0

    # ---- evaluation with counting ----
    def evaluate(self, X) -> tuple[np.ndarray, np.ndarray]:
        F, V = self.problem.evaluate_population(X)
        self.evals += len(F)
        self.feasible_evals += int((V == 0).sum())
        if self.hv_nadir is None:
            # The first (random) batch fixes the HV normalisation for the whole run:
            # ideal = 0 (no objective can be negative), nadir = worst value in that batch.
            self.hv_nadir = np.maximum(F.max(axis=0), 1.0)
        return F, V

    def evaluate_full(self, x) -> tuple[np.ndarray, np.ndarray, float, dict]:
        """One solution: selected objectives, all four objectives, V and the H1-H5 breakdown.
        Counts as one evaluation (used by the lexicographic method, which ranks by H1 alone)."""
        F_all, V, breakdown = self.problem.evaluate_full(x)
        self.evals += 1
        self.feasible_evals += int(V == 0)
        F = F_all[self.problem.obj_idx]
        if self.hv_nadir is None:
            self.hv_nadir = np.maximum(F, 1.0)
        return F, F_all, V, breakdown

    # ---- stopping ----
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def stop_reason(self) -> str | None:
        if self.cancelled or (self.should_stop is not None and self.should_stop()):
            self.cancelled = True
            return "cancelled"
        if self.time_limit is not None and self.elapsed() >= self.time_limit:
            return "time_limit"
        if self.evals >= self.max_evals:
            return "budget"
        return None

    def remaining(self) -> int:
        return max(0, self.max_evals - self.evals)

    # ---- history / progress ----
    def hv_of(self, F, V) -> float:
        """Hypervolume of the feasible points, normalised by [0, hv_nadir], reference point 1.1."""
        F = np.asarray(F)
        feas = F[np.asarray(V) == 0] if len(F) else F
        if len(feas) == 0 or self.hv_nadir is None:
            return 0.0
        N = mu.normalize(feas, np.zeros(self.problem.M), self.hv_nadir)
        return mu.hypervolume(N, np.full(self.problem.M, 1.1), n_samples=self.params.get("hv_samples", 5000), seed=0)

    def record(self, gen: int, F_out, V_out, V_pop=None) -> None:
        if gen % self.hv_every == 0 or gen == 0:
            self._last_hv = self.hv_of(F_out, V_out)
        Vp = np.asarray(V_pop if V_pop is not None else V_out)
        entry = {
            "gen": gen,
            "evaluations": self.evals,
            "hv": round(float(self._last_hv), 6),
            "best_violation": float(Vp.min()) if len(Vp) else None,
            "n_front": int((np.asarray(V_out) == 0).sum()),
            "seconds": round(self.elapsed(), 3),
        }
        if self.params.get("record_fronts") and (gen % self.hv_every == 0 or gen == 0):
            # experiments only: keep the feasible front so convergence can be re-scored on a common scale
            entry["front"] = np.asarray(F_out)[np.asarray(V_out) == 0].tolist()
        self.history.append(entry)
        if self.progress is not None:
            # A progress callback may return False to ask for early cancellation.
            if self.progress(dict(entry)) is False:
                self.cancelled = True


def feasible_front(X, F, V):
    """The non-dominated feasible solutions; if none is feasible, the solutions with least violation
    (so the caller can still see how close the search got)."""
    X, F, V = np.asarray(X), np.asarray(F, dtype=float), np.asarray(V, dtype=float)
    if len(F) == 0:
        return X, F, V
    feas = V == 0
    if feas.any():
        Xf, Ff, Vf = X[feas], F[feas], V[feas]
        keep = mu.nondominated_mask(Ff)
        Xf, Ff, Vf = Xf[keep], Ff[keep], Vf[keep]
        _, first = np.unique(Ff, axis=0, return_index=True)  # drop duplicate objective vectors
        first = np.sort(first)
        return Xf[first], Ff[first], Vf[first]
    best = V == V.min()
    return X[best][:1], F[best][:1], V[best][:1]


def merged_params(module, params: dict | None) -> dict:
    p = {**COMMON_DEFAULTS, **getattr(module, "DEFAULTS", {})}
    p.update(params or {})
    return p


def run_loop(module, problem: TimetableProblem, params: dict | None = None, progress=None, should_stop=None) -> RunResult:
    """Generic driver: initialise, then step until budget / time limit / cancellation."""
    p = merged_params(module, params)
    ctx = Context(problem, p, progress, should_stop)
    state = module.initialize(problem, p, ctx)
    X, F, V = module.output(state)
    ctx.record(0, F, V, state.get("V"))
    gen = 0
    reason = ctx.stop_reason()
    while reason is None:
        gen += 1
        state = module.step(problem, state, ctx)
        X, F, V = module.output(state)
        ctx.record(gen, F, V, state.get("V"))
        # classical methods finish after their last sub-problem even if a few evaluations are left
        reason = "completed" if state.get("done") else ctx.stop_reason()
    # final HV on the reported set, whatever the history cadence was
    if ctx.history and ctx.history[-1]["gen"] % ctx.hv_every != 0:
        ctx.history[-1]["hv"] = round(ctx.hv_of(F, V), 6)
    return RunResult(
        algorithm=module.NAME,
        X=X,
        F=F,
        V=V,
        history=ctx.history,
        evaluations=ctx.evals,
        feasible_rate=ctx.feasible_evals / max(ctx.evals, 1),
        runtime_s=ctx.elapsed(),
        stopped=reason,
        params=p,
        extra={**state.get("extra", {}), "hv_nadir": None if ctx.hv_nadir is None else ctx.hv_nadir.tolist()},
    )
