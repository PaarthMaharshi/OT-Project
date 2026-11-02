"""Repeated seeded experiments (proposal Section 13).

Examples (run from the project root):
    python -m experiments.run_experiments --scenarios small medium --seeds 30
    python -m experiments.run_experiments --methods ablation --scenarios medium --seeds 30
    python -m experiments.run_experiments --scenarios small --seeds 3 --budget 800 --name quick   # smoke test

Every stochastic method gets the same evaluation budget and the same seeds 1..N. Greedy is
deterministic and runs once; CP-SAT runs once per instance with a time limit per weight vector.

Hypervolume is normalised per scenario with the ideal and nadir points of everything any method found
(plus the Greedy point), reference point 1.1 on each axis, so numbers are comparable across methods.
Statistics against the main method (memetic NSGA-II): Wilcoxon rank-sum p-value and Vargha-Delaney A12.

Outputs in results/experiments/<name>/:
    runs.csv        one row per (scenario, method, seed)
    summary.csv     mean / sd / median per (scenario, method) with p-value and A12
    summary.md      the same as readable tables, plus CP-SAT solve details
    fronts.json     objective vectors of every returned front
    figures/        HV box plots, Pareto projections, 3-D fronts, convergence curves
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.instance_generator import generate  # noqa: E402
from src.methods import ABLATION, ALL_METHODS, COMPARISON, DETERMINISTIC, LABELS, MAIN_METHOD, run_method  # noqa: E402
from src.multiobjective.hypervolume import normalised_hv, spacing  # noqa: E402
from src.multiobjective.non_dominated_sort import dominates  # noqa: E402
from src.problem import Problem  # noqa: E402

DEFAULT_BUDGET = {"demo": 3000, "small": 3000, "medium": 3000, "large": 1500}
# CP-SAT seconds per weight vector; medium needs ~20-30 s just to find a first solution.
DEFAULT_CPSAT_SECONDS = {"demo": 20, "small": 20, "medium": 60, "large": 120}

_problem_cache: dict = {}


def _problem(scenario: str, instance_seed: int) -> Problem:
    k = (scenario, instance_seed)
    if k not in _problem_cache:
        _problem_cache[k] = Problem(generate(scenario, instance_seed))
    return _problem_cache[k]


def _task(scenario: str, instance_seed: int, method: str, seed: int, budget: int, pop: int, cpsat_seconds: float, keep_convergence: bool) -> dict:
    prob = _problem(scenario, instance_seed)
    r = run_method(method, prob, budget, seed, pop, cpsat_seconds)
    return {
        "scenario": scenario,
        "method": method,
        "seed": seed,
        "evaluations": r.evaluations,
        "runtime": r.runtime,
        "final_feasible_fraction": r.final_feasible_fraction,
        "points": [s.F.tolist() for s in r.points],
        "convergence": [{"evaluations": c["evaluations"], "front": c["front"]} for c in r.convergence] if keep_convergence else [],
        "info": r.info,
    }


# ---------------- statistics ----------------


def a12(x: list[float], y: list[float]) -> float:
    """Vargha-Delaney A12: probability that a value from x is larger than one from y (ties count half)."""
    if not x or not y:
        return float("nan")
    gt = sum(1.0 for a in x for b in y if a > b)
    eq = sum(1.0 for a in x for b in y if a == b)
    return (gt + 0.5 * eq) / (len(x) * len(y))


def ranksum_p(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if np.allclose(x, x[0]) and np.allclose(y, y[0]) and np.isclose(x[0], y[0]):
        return 1.0
    from scipy.stats import ranksums

    return float(ranksums(x, y).pvalue)


def a12_label(a: float) -> str:
    d = abs(a - 0.5)
    return "negligible" if d < 0.06 else "small" if d < 0.14 else "medium" if d < 0.21 else "large"


# ---------------- figures ----------------


def make_figures(out: Path, scenario: str, rows: list[dict], fronts: dict, bounds: tuple, convergence: dict, greedy_F):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping figures")
        return
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = [m for m in dict.fromkeys(r["method"] for r in rows)]
    cmap = plt.cm.tab10 if len(methods) <= 10 else plt.cm.tab20
    colors = {m: cmap(i % cmap.N) for i, m in enumerate(methods)}

    # HV box plot
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(methods)), 4))
    data = [[r["hv"] for r in rows if r["method"] == m] for m in methods]
    ax.boxplot(data, tick_labels=[LABELS.get(m, m) for m in methods], showmeans=True)
    ax.set_ylabel("normalised hypervolume (higher is better)")
    ax.set_title(f"{scenario}: hypervolume over seeds")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{scenario}_hv_boxplot.png", dpi=150)
    plt.close(fig)

    # Representative front per method: the run with median hypervolume.
    rep = {}
    for m in methods:
        rs = sorted([r for r in rows if r["method"] == m], key=lambda r: r["hv"])
        if rs:
            rep[m] = np.array(fronts[(scenario, m, rs[len(rs) // 2]["seed"])])
    names = ["f1 idle gaps", "f2 room wastage", "f3 load imbalance"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        for m, F in rep.items():
            if len(F):
                ax.scatter(F[:, i], F[:, j], s=18 if m != MAIN_METHOD else 26, label=LABELS.get(m, m), color=colors[m], alpha=0.8)
        ax.scatter([greedy_F[i]], [greedy_F[j]], marker="D", s=60, facecolor="white", edgecolor="black", label="Greedy point", zorder=5)
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{scenario}: Pareto fronts (median-hypervolume run of each method)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{scenario}_front_projections.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    for m, F in rep.items():
        if len(F):
            ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=14, label=LABELS.get(m, m), color=colors[m])
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])
    ax.legend(fontsize=7)
    ax.set_title(f"{scenario}: 3-D Pareto fronts")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{scenario}_front_3d.png", dpi=150)
    plt.close(fig)

    # Convergence (first seed of each evolutionary method)
    conv = {m: c for (s, m), c in convergence.items() if s == scenario and c}
    if conv:
        ideal, nadir = bounds
        fig, ax = plt.subplots(figsize=(7, 4))
        for m, c in conv.items():
            xs = [e["evaluations"] for e in c]
            ys = [normalised_hv(np.array(e["front"]), ideal, nadir) if e["front"] else 0.0 for e in c]
            ax.plot(xs, ys, label=LABELS.get(m, m), color=colors.get(m))
        ax.axhline(normalised_hv(np.array([greedy_F]), ideal, nadir), color="grey", ls="--", label="Greedy point")
        ax.set_xlabel("evaluations")
        ax.set_ylabel("hypervolume of archive")
        ax.set_title(f"{scenario}: convergence (seed 1)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{scenario}_convergence.png", dpi=150)
        plt.close(fig)


# ---------------- main ----------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=["small", "medium"], choices=["demo", "small", "medium", "large"])
    ap.add_argument("--methods", nargs="+", default=["comparison"], help="method names, or the groups: comparison, ablation, all")
    ap.add_argument("--seeds", type=int, default=30, help="number of independent seeds (1..N)")
    ap.add_argument("--budget", type=int, default=None, help="evaluations per run (default depends on scenario)")
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--instance-seed", type=int, default=1)
    ap.add_argument("--cpsat", action="store_true", help="also run the CP-SAT reference (needs OR-Tools)")
    ap.add_argument("--cpsat-seconds", type=float, default=None, help="time limit per weight vector (default: 20 small, 60 medium, 120 large)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--name", default=None, help="output folder name under results/experiments/")
    a = ap.parse_args()

    methods: list[str] = []
    for m in a.methods:
        methods += COMPARISON if m == "comparison" else list(ABLATION) + ["nsga2"] if m == "ablation" else ALL_METHODS[:-1] if m == "all" else [m]
    if a.cpsat:
        methods.append("cpsat")
    methods = list(dict.fromkeys(methods))
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        ap.error(f"unknown method(s) {unknown}; choose from {ALL_METHODS}")

    out = ROOT / "results" / "experiments" / (a.name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a) | {"methods": methods}, indent=2))

    tasks = []
    for sc in a.scenarios:
        budget = a.budget or DEFAULT_BUDGET[sc]
        for m in methods:
            seeds = [1] if (m in DETERMINISTIC or m == "cpsat") else range(1, a.seeds + 1)
            for s in seeds:
                tasks.append((sc, a.instance_seed, m, s, budget, a.pop, a.cpsat_seconds or DEFAULT_CPSAT_SECONDS[sc], s == 1))
    # CP-SAT is multi-threaded and time-limited, so it runs alone afterwards with the whole machine.
    exact = [t for t in tasks if t[2] == "cpsat"]
    tasks = sorted([t for t in tasks if t[2] != "cpsat"], key=lambda t: t[0] != "large")
    print(f"{len(tasks) + len(exact)} runs: scenarios={a.scenarios} methods={methods} seeds={a.seeds} jobs={a.jobs} -> {out}")

    results = []
    t0 = time.time()
    for t in exact:
        try:
            results.append(_task(*t))
            r = results[-1]
            print(f"  [cpsat] {r['scenario']:6s} {len(r['points'])} pts  {r['runtime']:.1f}s")
        except Exception as e:
            print(f"  FAILED {t[0]}/cpsat: {type(e).__name__}: {e}")
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(_task, *t): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            t = futs[f]
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  FAILED {t[0]}/{t[2]}/seed {t[3]}: {type(e).__name__}: {e}")
                continue
            r = results[-1]
            print(f"  [{i}/{len(tasks)}] {r['scenario']:6s} {r['method']:20s} seed {r['seed']:2d}  {len(r['points']):3d} pts  {r['runtime']:6.1f}s  (elapsed {time.time() - t0:.0f}s)")

    fronts = {(r["scenario"], r["method"], r["seed"]): r["points"] for r in results}
    convergence = {(r["scenario"], r["method"]): r["convergence"] for r in results if r["convergence"]}
    (out / "fronts.json").write_text(json.dumps([{k: r[k] for k in ("scenario", "method", "seed", "points")} for r in results]))

    figure_jobs: list = []
    run_rows, summary_rows, md = [], [], [f"# Experiment results\n\nGenerated {datetime.now():%Y-%m-%d %H:%M}. Synthetic instances only.\n"]
    md.append(f"Seeds: {a.seeds}. Population: {a.pop}. Main method: **{LABELS[MAIN_METHOD]}**. HV is normalised per scenario (reference point 1.1).\n")
    for sc in a.scenarios:
        greedy_F = _greedy_point(sc, a.instance_seed)
        pts = [np.array(r["points"]) for r in results if r["scenario"] == sc and r["points"]]
        allF = np.vstack(pts + [greedy_F[None]])
        ideal, nadir = allF.min(axis=0), allF.max(axis=0)
        rows = []
        for r in results:
            if r["scenario"] != sc:
                continue
            F = np.array(r["points"]) if r["points"] else np.zeros((0, 3))
            row = {
                "scenario": sc,
                "method": r["method"],
                "seed": r["seed"],
                "evaluations": r["evaluations"],
                "runtime": round(r["runtime"], 3),
                "front_size": len(F),
                "hv": normalised_hv(F, ideal, nadir) if len(F) else 0.0,
                "spacing": spacing(F) if len(F) else float("nan"),
                "best_f1": F[:, 0].min() if len(F) else float("nan"),
                "best_f2": F[:, 1].min() if len(F) else float("nan"),
                "best_f3": F[:, 2].min() if len(F) else float("nan"),
                "feasible_run": int(len(F) > 0),
                "final_feasible_fraction": r["final_feasible_fraction"],
                "covers_greedy": int(any(dominates(f, greedy_F) or np.allclose(f, greedy_F) for f in F)),
            }
            rows.append(row)
        run_rows += rows

        main_hv = [r["hv"] for r in rows if r["method"] == MAIN_METHOD]
        md.append(f"\n## {sc}\n")
        md.append(f"Greedy point: gaps {greedy_F[0]:.1f}, waste {greedy_F[1]:.1%}, imbalance {greedy_F[2]:.3f}. Budget: {a.budget or DEFAULT_BUDGET[sc]} evaluations per run.\n")
        md.append("| Method | Runs | HV mean ± sd | HV median | p vs NSGA-II | A12 (NSGA-II better) | Front size | Spacing | Best gaps | Best waste | Best imbalance | Feasible runs | Matches/beats greedy | Time (s) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for m in methods:
            rs = [r for r in rows if r["method"] == m]
            if not rs:
                continue
            hv = [r["hv"] for r in rs]
            p = ranksum_p(main_hv, hv) if m != MAIN_METHOD else float("nan")
            A = a12(main_hv, hv) if m != MAIN_METHOD else float("nan")

            def mean(k):
                v = [r[k] for r in rs if not np.isnan(r[k])]
                return float(np.mean(v)) if v else float("nan")

            srow = {
                "scenario": sc,
                "method": m,
                "runs": len(rs),
                "hv_mean": float(np.mean(hv)),
                "hv_sd": float(np.std(hv, ddof=1)) if len(hv) > 1 else 0.0,
                "hv_median": float(np.median(hv)),
                "p_value_vs_main": p,
                "a12_main_vs_method": A,
                "front_size_mean": mean("front_size"),
                "spacing_mean": mean("spacing"),
                "best_f1_mean": mean("best_f1"),
                "best_f2_mean": mean("best_f2"),
                "best_f3_mean": mean("best_f3"),
                "feasible_run_rate": mean("feasible_run"),
                "final_feasible_fraction_mean": mean("final_feasible_fraction"),
                "covers_greedy_rate": mean("covers_greedy"),
                "runtime_mean": mean("runtime"),
            }
            summary_rows.append(srow)
            ptxt = "—" if np.isnan(p) else (f"{p:.2e}" if p < 0.001 else f"{p:.3f}")
            atxt = "—" if np.isnan(A) else f"{A:.2f} ({a12_label(A)})"
            def fmt(v, spec):
                return "—" if v is None or np.isnan(v) else format(v, spec)

            md.append(
                f"| {LABELS.get(m, m)} | {len(rs)} | {srow['hv_mean']:.3f} ± {srow['hv_sd']:.3f} | {srow['hv_median']:.3f} | {ptxt} | {atxt} | "
                f"{srow['front_size_mean']:.1f} | {fmt(srow['spacing_mean'], '.3f')} | {fmt(srow['best_f1_mean'], '.1f')} | "
                f"{fmt(srow['best_f2_mean'], '.1%')} | {fmt(srow['best_f3_mean'], '.3f')} | "
                f"{srow['feasible_run_rate']:.0%} | {srow['covers_greedy_rate']:.0%} | {srow['runtime_mean']:.1f} |"
            )
        cps = [r for r in results if r["scenario"] == sc and r["method"] == "cpsat"]
        if cps:
            md.append("\n**CP-SAT solves** (weighted sum; f3 modelled as max-min load spread):\n")
            md.append("| Weights (gaps, waste, imbalance) | Status | Objective | Bound | Gap | Time (s) |")
            md.append("|---|---|---|---|---|---|")
            for s in cps[0]["info"].get("solves", []):
                obj, bd = s["objective"], s["bound"]
                gap = "—" if obj is None or bd is None or obj == 0 else f"{abs(obj - bd) / abs(obj):.1%}"
                md.append(
                    f"| {', '.join(f'{x:.2f}' for x in s['weight'])} | {s['status']} | {'—' if obj is None else f'{obj:.4f}'} | "
                    f"{'—' if bd is None else f'{bd:.4f}'} | {gap} | {s['seconds']:.1f} |"
                )
        md.append(f"\nFigures: `figures/{sc}_hv_boxplot.png`, `figures/{sc}_front_projections.png`, `figures/{sc}_front_3d.png`, `figures/{sc}_convergence.png`\n")
        figure_jobs.append((sc, rows, (ideal, nadir), greedy_F))

    md.append(
        "\nA12 is the probability that a NSGA-II run has higher hypervolume than a run of the other method "
        "(0.5 = no difference; effect size thresholds 0.56 / 0.64 / 0.71). p-values: two-sided Wilcoxon rank-sum.\n"
    )
    _write_csv(out / "runs.csv", run_rows)
    _write_csv(out / "summary.csv", summary_rows)
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    for sc, rows, bounds, greedy_F in figure_jobs:  # tables are written first, so a plotting error loses nothing
        try:
            make_figures(out, sc, rows, fronts, bounds, convergence, greedy_F)
        except Exception as e:
            print(f"  figures for {sc} failed: {type(e).__name__}: {e}")
    print(f"\nDone in {time.time() - t0:.0f}s. Results in {out}")
    print(f"  {out / 'summary.md'}")


def _greedy_point(scenario: str, instance_seed: int) -> np.ndarray:
    from src.baseline.greedy_dsatur import greedy

    return greedy(_problem(scenario, instance_seed)).F


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
