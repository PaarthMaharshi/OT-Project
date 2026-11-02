"""Command-line entry point.

    python main.py --scenario medium --generations 40 --seed 1
    python main.py --scenario small --no-repair --no-local-search     # ablation variants

Writes the full result (instance, Pareto front, baseline, convergence) to results/<name>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.runner import RunParams, run


def main():
    ap = argparse.ArgumentParser(description="Multi-objective college timetable optimisation (memetic NSGA-II)")
    ap.add_argument("--scenario", default="medium", choices=["demo", "small", "medium", "large"])
    ap.add_argument("--instance-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--no-seeding", action="store_true")
    ap.add_argument("--no-repair", action="store_true")
    ap.add_argument("--no-local-search", action="store_true")
    ap.add_argument("--out", default=None, help="output JSON path")
    a = ap.parse_args()

    params = RunParams(
        scenario=a.scenario,
        instance_seed=a.instance_seed,
        seed=a.seed,
        pop_size=a.pop,
        generations=a.generations,
        use_seeding=not a.no_seeding,
        use_repair=not a.no_repair,
        use_local_search=not a.no_local_search,
        max_seconds=None,
    )

    def progress(h):
        if h["generation"] % 5 == 0 or h["generation"] == a.generations:
            print(f"  gen {h['generation']:4d}  front={h['archive_size']:3d}  feasible={h['feasible_fraction']:.0%}  {h['seconds']:.1f}s")

    print(f"Running {a.scenario} (instance seed {a.instance_seed}, run seed {a.seed})")
    res = run(params, progress)
    st = res["stats"]
    b = res["baseline"]["objectives"]
    print(f"\nPareto front: {st['frontSize']} clash-free timetables in {st['runtime']:.1f}s ({st['evaluations']} evaluations)")
    print(f"Hypervolume  NSGA-II {st['hypervolume']:.3f}   vs Greedy {st['baselineHypervolume']:.3f}")
    print(f"Greedy point  gaps={b['f1']:.1f}  waste={b['f2']:.1%}  imbalance={b['f3']:.3f}")
    out = Path(a.out or f"results/{a.scenario}_i{a.instance_seed}_s{a.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
