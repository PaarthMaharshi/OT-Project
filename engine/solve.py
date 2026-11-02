"""The single entry point used by the frontend adapter:

    solve(instance: dict, progress=None, should_stop=None) -> dict      (always JSON-serialisable)

Input: the engine schema (README) or the website's own instance format (see frontend_bridge).
Output on success:
    {"status": "ok" | "no_feasible_solution",
     "algorithm", "objective_names", "params",
     "front": [{"objectives": [...], "objective_values": {name: value}, "violation": 0,
                "timetable": [{"lecture_id", "session", "day", "slot", "room_id", ...}]}],
     "recommended": index into front (TOPSIS by default, or VIKOR if params.decision == "vikor"),
     "ranking": {"topsis": [indices best first], "vikor": [indices best first]},
     "metrics": {"hv", "igd", "spacing", "feasible_rate", "runtime_s", "evaluations", "stopped"},
     "history": [{"gen", "hv", "best_violation", "evaluations", "n_front", "seconds"}],
     "warnings": [...]}
On bad input: {"error": "...", "details": [...]} - never an exception.

progress(entry) is called after every generation; returning False cancels the run.
should_stop() is polled between generations; returning True cancels the run.
Either way, and when time_limit_s is reached, the best front found so far is returned.
"""
from __future__ import annotations

import math
import traceback

import numpy as np

from . import mcdm
from . import moo_utils as mu
from .algorithms import DEFAULT_ALGORITHM, REGISTRY
from .frontend_bridge import decorate, from_frontend, is_frontend
from .problem import OBJECTIVES, TimetableProblem, validate_instance

PARAM_RULES = {
    "pop_size": (int, 4, 10_000),
    "evaluations": (int, 10, 10_000_000),
    "seed": (int, 0, 2**31 - 1),
    "time_limit_s": ((int, float), 0, 86_400),
}


def _check_params(params) -> list[str]:
    errs = []
    if not isinstance(params, dict):
        return ["'params' must be an object."]
    for k, (typ, lo, hi) in PARAM_RULES.items():
        if k not in params or params[k] is None:
            continue
        v = params[k]
        if isinstance(v, bool) or not isinstance(v, typ) or not (lo <= v <= hi):
            errs.append(f"params.{k} must be a number in [{lo}, {hi}] (got {v!r}).")
    obj = params.get("objectives")
    if obj is not None and (not isinstance(obj, list) or not obj or any(o not in OBJECTIVES for o in obj)):
        errs.append(f"params.objectives must be a non-empty list drawn from {OBJECTIVES}.")
    if params.get("decision") not in (None, "topsis", "vikor"):
        errs.append("params.decision must be 'topsis' or 'vikor'.")
    return errs


def _json_safe(o):
    """Turn numpy types into plain Python and non-finite floats into None."""
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def solve(instance, progress=None, should_stop=None) -> dict:
    try:
        return _json_safe(_solve(instance, progress, should_stop))
    except Exception as e:  # never crash the caller
        return {"error": f"Internal error: {type(e).__name__}: {e}", "details": traceback.format_exc().splitlines()[-3:]}


def _solve(instance, progress, should_stop) -> dict:
    warnings: list[str] = []
    mapping = None
    if not isinstance(instance, dict):
        return {"error": "The instance must be a JSON object.", "details": []}
    if is_frontend(instance):
        instance, mapping, warnings = from_frontend(instance)

    errs = validate_instance(instance)
    params = dict(instance.get("params") or {})
    errs += _check_params(params)
    algo = instance.get("algorithm", DEFAULT_ALGORITHM)
    if algo not in REGISTRY:
        errs.append(f"Unknown algorithm {algo!r}; choose one of {sorted(REGISTRY)}.")
    if errs:
        return {"error": "Invalid instance", "details": errs}

    weights_in = instance.get("weights") or {}
    objectives = params.pop("objectives", None)
    if objectives is None:
        # default: optimise the objectives the user put a positive weight on (all four if none given)
        objectives = [o for o in OBJECTIVES if weights_in.get(o, 0) > 0] or list(OBJECTIVES)
    if algo == "nsga3" and len(objectives) < 3:
        warnings.append("NSGA-III is designed for 3+ objectives; with 2 it still runs.")
    w = np.array([float(weights_in.get(o, 1.0)) for o in objectives])
    w = w / w.sum() if w.sum() > 0 else np.full(len(objectives), 1.0 / len(objectives))
    decision = params.pop("decision", "topsis")
    vikor_v = float(params.pop("vikor_v", 0.5))
    reference_front = params.pop("reference_front", None)

    problem = TimetableProblem(instance, objectives)
    module = REGISTRY[algo]
    res = module.run(problem, params, progress, should_stop)

    feasible = bool(len(res.V) and np.all(res.V == 0))
    front = []
    for x, f, v in zip(res.X, res.F, res.V):
        tt = problem.to_timetable(x)
        if mapping is not None:
            tt = decorate(tt, mapping)
        front.append({
            "objectives": [float(a) for a in f],
            "objective_values": {name: float(a) for name, a in zip(objectives, f)},
            "violation": float(v),
            "timetable": tt,
        })

    ranking = {"topsis": [], "vikor": []}
    recommended = None
    if front:
        F = np.asarray(res.F, dtype=float)
        ranking["topsis"] = mcdm.topsis(F, w)["ranking"].tolist()
        ranking["vikor"] = mcdm.vikor(F, w, vikor_v)["ranking"].tolist()
        recommended = ranking[decision][0]

    nadir = np.asarray(res.extra.get("hv_nadir") or np.maximum(np.asarray(res.F).max(axis=0), 1.0), dtype=float)
    F_norm = mu.normalize(res.F, np.zeros(problem.M), nadir) if feasible else np.zeros((0, problem.M))
    igd = None
    if reference_front is not None and feasible:
        Z = mu.normalize(np.asarray(reference_front, dtype=float), np.zeros(problem.M), nadir)
        igd = mu.igd(F_norm, Z)
    if not feasible:
        warnings.append("No timetable without hard-constraint violations was found; the least-violating one is returned.")

    return {
        "status": "ok" if feasible else "no_feasible_solution",
        "algorithm": algo,
        "objective_names": objectives,
        "weights": dict(zip(objectives, w.tolist())),
        "params": {k: v for k, v in res.params.items() if k != "reference_front"},
        "front": front,
        "recommended": recommended,
        "ranking": ranking,
        "metrics": {
            "hv": res.history[-1]["hv"] if res.history else 0.0,
            "hv_normalisation": {"ideal": [0.0] * problem.M, "nadir": nadir.tolist(), "reference_point": [1.1] * problem.M},
            "igd": igd,
            "spacing": mu.spacing(F_norm) if feasible else None,
            "feasible_rate": res.feasible_rate,
            "runtime_s": res.runtime_s,
            "evaluations": res.evaluations,
            "stopped": res.stopped,
        },
        "history": res.history,
        "extra": {k: v for k, v in res.extra.items() if k != "hv_nadir"},
        "warnings": warnings,
    }
