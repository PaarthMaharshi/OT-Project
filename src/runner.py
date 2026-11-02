"""One optimisation run end to end: build instance -> greedy baseline -> memetic NSGA-II -> JSON result.
Shared by the CLI (main.py) and the web API (api/server.py)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from .algorithms.nsga2 import NSGA2Config, run_nsga2
from .baseline.greedy_dsatur import greedy
from .evaluator import practical_metrics
from .instance_generator import generate
from .multiobjective.hypervolume import normalised_hv, spacing
from .problem import Problem


@dataclass
class RunParams:
    scenario: str = "demo"
    instance_seed: int = 1
    seed: int = 1
    pop_size: int = 60
    generations: int = 40
    use_seeding: bool = True
    use_repair: bool = True
    use_local_search: bool = True
    max_seconds: float | None = 120
    faculty_overrides: dict | None = None  # {faculty_id: {"unavailable": [...], "max_daily": n}} from the website


def run(params: RunParams, progress: Callable[[dict], None] | None = None) -> dict:
    inst = generate(params.scenario, params.instance_seed)
    for f in inst.faculty:  # teachers' own availability, saved on the website, overrides the generated data
        o = (params.faculty_overrides or {}).get(f.id)
        if o:
            f.unavailable = sorted(set(f.unavailable) | set(o.get("unavailable") or []))
            if o.get("max_daily"):
                f.max_daily = int(o["max_daily"])
    prob = Problem(inst)
    base = greedy(prob)
    cfg = NSGA2Config(
        pop_size=params.pop_size,
        generations=params.generations,
        use_seeding=params.use_seeding,
        use_repair=params.use_repair,
        use_local_search=params.use_local_search,
        max_seconds=params.max_seconds,
        seed=params.seed,
    )
    res = run_nsga2(prob, cfg, progress)

    front = res.archive
    # Feasible, dominated members of the final population (shown as grey dots), without duplicates.
    seen = {tuple(np.round(s.F, 9)) for s in front}
    dominated = []
    for s in res.population:
        k = tuple(np.round(s.F, 9))
        if s.V == 0 and k not in seen:
            seen.add(k)
            dominated.append(s)

    FA = np.array([s.F for s in front]) if front else np.zeros((0, 3))
    all_hist = [np.array(h["front"]) for h in res.history if h["front"]]
    union = np.vstack([FA, base.F[None]] + all_hist) if len(FA) else base.F[None]
    ideal, nadir = union.min(axis=0), union.max(axis=0)

    solutions = [s.to_dict(prob, f"S{i + 1:03d}", True) for i, s in enumerate(front)]
    solutions += [s.to_dict(prob, f"S{len(front) + i + 1:03d}", False) for i, s in enumerate(dominated)]
    for d, s in zip(solutions, front + dominated):
        d["metrics"] = practical_metrics(prob, s)

    baseline = base.to_dict(prob, "GREEDY", False)
    baseline["metrics"] = practical_metrics(prob, base)

    return {
        "params": asdict(params),
        "instance": inst.to_dict(),
        "solutions": solutions,
        "baseline": baseline,
        "stats": {
            "runtime": res.runtime,
            "evaluations": res.evaluations,
            "generations": res.generations_run,
            "frontSize": len(front),
            "hypervolume": normalised_hv(FA, ideal, nadir),
            "baselineHypervolume": normalised_hv(base.F[None], ideal, nadir),
            "spacing": spacing(FA),
            "baselineFeasible": bool(base.V == 0),
            "convergence": [
                {
                    "generation": h["generation"],
                    "hypervolume": normalised_hv(np.array(h["front"]), ideal, nadir) if h["front"] else 0.0,
                    "archiveSize": h["archive_size"],
                    "feasibleFraction": h["feasible_fraction"],
                }
                for h in res.history
            ],
        },
    }
