"""All lab experiments for the engine (section 4 of the lab brief). Report only - not used by the website.

    python -m experiments.engine_lab                 # everything, 10 seeds (several minutes)
    python -m experiments.engine_lab --seeds 2 --evaluations 2000 --out results/lab_quick   # smoke test

Studies (seeds 1..N, every run gets the same evaluation budget):
    main      2 objectives (f1 student gaps, f2 teacher gaps), as the lab brief specifies: NSGA-II,
              Constrained NSGA-II, SPEA2, MOEA/D, MOSA, MOMA, Weighted Sum, eps-Constraint, Goal Programming,
              Lexicographic. On this instance f1 and f2 do NOT conflict (the best-known point (0, 2)
              dominates everything), so this study measures convergence towards one point.
    main2     the same ten methods on 2 objectives that DO conflict (f3 room waste, f4 early/late penalty).
              A pair check (MOSA + C-NSGA-II, seeds 1-3, all six pairs) found a 6-point front for this pair
              and at most 3 points for every other pair.
    sens      on main2's objectives: Constrained NSGA-II pop_size {50,100,200}; MOEA/D neighbourhood
              {5,20,50}; MOSA cooling_rate {0.90,0.95,0.99} (the middle value is the main2 run)
    ablation  on main2's objectives: repair vs penalty, {Deb rules, penalty} x {repair, no repair};
              MOMA vs Constrained NSGA-II; Weighted Sum vs the non-convex parts of the main2 front
    many      4 objectives: Constrained NSGA-II vs NSGA-III vs MOEA/D
    mcdm      TOPSIS and VIKOR (v = .25/.5/.75) on the 4-objective reference front, weight sensitivity

The true Pareto front is unknown. For every objective space, the REFERENCE FRONT is the non-dominated
union of every feasible solution returned by every run of every algorithm in that space. Objectives are
normalised by the reference front's ideal and nadir points. If the reference front has no spread on an
axis (ideal == nadir, e.g. a one-point front), that axis's nadir is taken from the worst value in the
union of all feasible points returned in that space instead. HV uses reference point 1.1 per axis
(exact in 2-D, Monte-Carlo with 200,000 samples in 4-D); IGD and Spacing are computed on normalised values.

Outputs (results/lab/): runs.csv, summary.csv, summary.md, convergence.csv, reference_fronts.json,
fronts.json, mcdm_*.csv, instance.json / instance.txt, figures/*.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import mcdm  # noqa: E402
from engine import moo_utils as mu  # noqa: E402
from engine.algorithms import REGISTRY  # noqa: E402
from engine.generator import describe, generate  # noqa: E402
from engine.problem import OBJECTIVES, TimetableProblem  # noqa: E402

PAIR = ["student_gaps", "teacher_gaps"]  # f1 vs f2, as specified (does not conflict on this instance)
PAIR2 = ["room_waste", "time_penalty"]  # f3 vs f4, the pair that does conflict
NONCONVEX_PAIR = PAIR2
MAIN_ALGOS = ["nsga2", "constrained_nsga2", "spea2", "moead", "mosa", "moma", "weighted_sum", "eps_constraint", "goal_programming", "lexicographic"]
ALL4 = list(OBJECTIVES)

LABELS = {
    "nsga2": "NSGA-II (penalty)",
    "constrained_nsga2": "Constrained NSGA-II",
    "spea2": "SPEA2",
    "moead": "MOEA/D",
    "mosa": "MOSA",
    "moma": "MOMA",
    "nsga3": "NSGA-III",
    "weighted_sum": "Weighted Sum",
    "eps_constraint": "eps-Constraint",
    "goal_programming": "Goal Programming",
    "lexicographic": "Lexicographic",
}


def specs(seeds: int, budget: int) -> list[dict]:
    """Every run of every study."""
    out = []

    def add(study, label, algo, objectives, **params):
        for s in range(1, seeds + 1):
            out.append({"study": study, "label": label, "algorithm": algo, "objectives": objectives, "seed": s,
                        "params": {"evaluations": budget, "seed": s, "record_fronts": True, **params}})

    for algo in MAIN_ALGOS:
        add("main", LABELS[algo], algo, PAIR)
        add("main2", LABELS[algo], algo, PAIR2)
    for pop in (50, 200):
        add("sens", f"C-NSGA-II pop={pop}", "constrained_nsga2", PAIR2, pop_size=pop)
    for t in (5, 50):
        add("sens", f"MOEA/D T={t}", "moead", PAIR2, neighbourhood=t)
    for a in (0.90, 0.99):
        add("sens", f"MOSA alpha={a}", "mosa", PAIR2, cooling_rate=a)
    add("ablation", "Deb rules, no repair", "constrained_nsga2", PAIR2, repair=False)
    add("ablation", "Penalty + repair", "nsga2", PAIR2, repair=True)
    for algo in ["constrained_nsga2", "nsga3", "moead"]:
        add("many", LABELS[algo], algo, ALL4)
    return out


_PROBLEMS: dict = {}


def run_one(spec: dict) -> dict:
    key = tuple(spec["objectives"])
    if key not in _PROBLEMS:
        _PROBLEMS[key] = TimetableProblem(generate(1), spec["objectives"])
    p = _PROBLEMS[key]
    r = REGISTRY[spec["algorithm"]].run(p, spec["params"])
    feas = r.V == 0
    return {
        **{k: spec[k] for k in ("study", "label", "algorithm", "seed")},
        "objectives": spec["objectives"],
        "front": r.F[feas].tolist(),
        "evaluations": r.evaluations,
        "runtime_s": r.runtime_s,
        "feasible_rate": r.feasible_rate,
        "stopped": r.stopped,
        "history": [{"evaluations": h["evaluations"], "front": h.get("front", []), "n_front": h["n_front"]} for h in r.history],
        "extra": {k: v for k, v in r.extra.items() if k in ("weights", "goals", "stages")},
    }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
class Scorer:
    """Normalisation + indicators for one objective space, built from its reference front.

    ideal = per-objective minimum of the reference front.
    nadir = per-objective maximum of the reference front, EXCEPT on axes where the reference front has no
            spread (ideal == nadir, e.g. a one-point front): there the nadir is the worst value in the union
            of all feasible points returned in this space, so HV/IGD still measure how close each run got.
    """

    def __init__(self, ref_front: np.ndarray, union: np.ndarray):
        self.ref = ref_front
        self.ideal = ref_front.min(axis=0)
        nadir = ref_front.max(axis=0)
        self.fallback_axes = np.flatnonzero(nadir - self.ideal <= 1e-12).tolist()
        worst = union.max(axis=0) if len(union) else nadir
        nadir = np.where(nadir - self.ideal > 1e-12, nadir, worst)
        self.nadir = np.where(nadir - self.ideal > 1e-12, nadir, self.ideal + 1.0)  # still flat: unit range
        self.M = ref_front.shape[1]

    def norm(self, F):
        return mu.normalize(np.asarray(F, dtype=float).reshape(-1, self.M), self.ideal, self.nadir)

    def hv(self, F) -> float:
        if len(F) == 0:
            return 0.0
        return mu.hypervolume(self.norm(F), np.full(self.M, 1.1), n_samples=200_000, seed=7) if self.M > 2 else mu.hv_2d(self.norm(F), np.full(2, 1.1))

    def igd(self, F) -> float:
        return mu.igd(self.norm(F), self.norm(self.ref)) if len(F) else float("nan")

    def spacing(self, F) -> float:
        return mu.spacing(self.norm(F)) if len(F) > 1 else float("nan")


def mean_sd(values) -> tuple[float, float]:
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def fmt(m, s, d=3):
    return "n/a" if np.isnan(m) else f"{m:.{d}f} ± {s:.{d}f}"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--evaluations", type=int, default=10_000)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=str(ROOT / "results" / "lab"))
    a = ap.parse_args()
    out = Path(a.out)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    inst = generate(1)
    (out / "instance.json").write_text(json.dumps(inst, indent=2))
    (out / "instance.txt").write_text(describe(inst))
    print(describe(inst), "\n")

    S = specs(a.seeds, a.evaluations)
    print(f"{len(S)} runs, {a.jobs} parallel workers, budget {a.evaluations} evaluations, seeds 1..{a.seeds}")
    t0 = time.time()
    results = []
    with Pool(a.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, S), 1):
            results.append(r)
            if i % 10 == 0 or i == len(S):
                print(f"  {i}/{len(S)} runs done ({time.time() - t0:.0f}s)")
    results.sort(key=lambda r: (r["study"], r["label"], r["seed"]))

    # main-study configurations double as the default point of the sensitivity / ablation studies
    by_space: dict[tuple, list] = {}
    for r in results:
        by_space.setdefault(tuple(r["objectives"]), []).append(r)
    scorers, ref_fronts = {}, {}
    for space, rs in by_space.items():
        ref = mu.merge_fronts([r["front"] for r in rs])
        union = np.array([f for r in rs for f in r["front"]], dtype=float).reshape(-1, len(space))
        ref_fronts[space] = ref
        scorers[space] = Scorer(ref, union)
        sc_ = scorers[space]
        print(f"reference front {space}: {len(ref)} points (merged from {len(rs)} runs); "
              f"ideal {sc_.ideal.tolist()}, nadir {sc_.nadir.tolist()}, fallback axes {sc_.fallback_axes}")

    rows = []
    for r in results:
        sc = scorers[tuple(r["objectives"])]
        F = np.array(r["front"]).reshape(-1, len(r["objectives"]))
        row = {
            "study": r["study"], "label": r["label"], "algorithm": r["algorithm"], "seed": r["seed"],
            "objectives": "+".join(r["objectives"]), "evaluations": r["evaluations"], "runtime_s": round(r["runtime_s"], 3),
            "found_feasible": int(len(F) > 0), "feasible_rate": round(r["feasible_rate"], 4), "front_size": len(F),
            "hv": sc.hv(F), "igd": sc.igd(F), "spacing": sc.spacing(F),
            "reference_points_hit": int(sum(any(np.allclose(f, z) for f in F) for z in sc.ref)),
        }
        if F.size:
            for j, o in enumerate(r["objectives"]):
                row[f"best_{o}"] = float(F[:, j].min())
        rows.append(row)

    # extra rows: link the default configurations into sens/ablation tables
    def rows_of(study, label):
        return [x for x in rows if x["study"] == study and x["label"] == label]

    def summary(rs) -> dict:
        hv, igd, sp = mean_sd([x["hv"] for x in rs]), mean_sd([x["igd"] for x in rs]), mean_sd([x["spacing"] for x in rs])
        return {
            "runs": len(rs), "hv_mean": hv[0], "hv_sd": hv[1], "igd_mean": igd[0], "igd_sd": igd[1], "spacing_mean": sp[0], "spacing_sd": sp[1],
            "found_feasible": float(np.mean([x["found_feasible"] for x in rs])), "feasible_rate": float(np.mean([x["feasible_rate"] for x in rs])),
            "front_size": float(np.mean([x["front_size"] for x in rs])), "runtime_s": float(np.mean([x["runtime_s"] for x in rs])),
            "evaluations": float(np.mean([x["evaluations"] for x in rs])),
        }

    tables = {
        "main": [(LABELS[a_], rows_of("main", LABELS[a_])) for a_ in MAIN_ALGOS],
        "main2": [(LABELS[a_], rows_of("main2", LABELS[a_])) for a_ in MAIN_ALGOS],
        "sens_nsga2": [("pop_size=50", rows_of("sens", "C-NSGA-II pop=50")), ("pop_size=100", rows_of("main2", LABELS["constrained_nsga2"])), ("pop_size=200", rows_of("sens", "C-NSGA-II pop=200"))],
        "sens_moead": [("T=5", rows_of("sens", "MOEA/D T=5")), ("T=20", rows_of("main2", LABELS["moead"])), ("T=50", rows_of("sens", "MOEA/D T=50"))],
        "sens_mosa": [("alpha=0.90", rows_of("sens", "MOSA alpha=0.9")), ("alpha=0.95", rows_of("main2", LABELS["mosa"])), ("alpha=0.99", rows_of("sens", "MOSA alpha=0.99"))],
        "ablation_repair": [("Deb rules + repair (C-NSGA-II)", rows_of("main2", LABELS["constrained_nsga2"])), ("Deb rules, no repair", rows_of("ablation", "Deb rules, no repair")),
                            ("Penalty + repair", rows_of("ablation", "Penalty + repair")), ("Penalty, no repair (NSGA-II)", rows_of("main2", LABELS["nsga2"]))],
        "ablation_memetic": [("Constrained NSGA-II", rows_of("main2", LABELS["constrained_nsga2"])), ("MOMA (= C-NSGA-II + hill climbing)", rows_of("main2", LABELS["moma"]))],
        "many": [(LABELS[a_], rows_of("many", LABELS[a_])) for a_ in ["constrained_nsga2", "nsga3", "moead"]],
    }
    summary_rows = []
    for name, entries in tables.items():
        for label, rs in entries:
            if rs:
                summary_rows.append({"table": name, "method": label, **summary(rs)})

    _write_csv(out / "runs.csv", rows)
    _write_csv(out / "summary.csv", summary_rows)
    (out / "reference_fronts.json").write_text(json.dumps({"+".join(k): v.tolist() for k, v in ref_fronts.items()}, indent=1))
    (out / "fronts.json").write_text(json.dumps([{k: r[k] for k in ("study", "label", "seed", "objectives", "front")} for r in results]))

    # convergence: HV of the reported set vs evaluations, re-scored on the common normalisation
    conv_rows = []
    for r in results:
        sc = scorers[tuple(r["objectives"])]
        for h in r["history"]:
            if "front" in h and r["study"] in ("main", "main2", "many"):
                F = np.array(h["front"]).reshape(-1, len(r["objectives"]))
                conv_rows.append({"study": r["study"], "label": r["label"], "seed": r["seed"], "evaluations": h["evaluations"],
                                  "hv": sc.hv(F) if (r["study"] != "many" or r["seed"] <= 3) else None, "n_front": h["n_front"]})
    conv_rows = [c for c in conv_rows if c["hv"] is not None or c["study"] == "many"]
    _write_csv(out / "convergence.csv", conv_rows)

    # non-convexity: which reference-front points can a weighted sum reach at all?
    nc_space = tuple(NONCONVEX_PAIR)
    sc = scorers[nc_space]
    ref_n = sc.norm(ref_fronts[nc_space])
    supported = supported_mask(ref_n)
    ws_points = {tuple(f) for r in results if r["study"] == "main2" and r["algorithm"] == "weighted_sum" for f in r["front"]}
    eps_points = {tuple(f) for r in results if r["study"] == "main2" and r["algorithm"] == "eps_constraint" for f in r["front"]}
    ref_list = [tuple(z) for z in ref_fronts[nc_space].tolist()]
    nonconvex_info = {
        "reference_points": len(ref_list),
        "supported": int(supported.sum()),
        "unsupported": int((~supported).sum()),
        "ws_found_supported": sum(1 for z, s in zip(ref_list, supported) if s and z in ws_points),
        "ws_found_unsupported": sum(1 for z, s in zip(ref_list, supported) if (not s) and z in ws_points),
        "eps_found_unsupported": sum(1 for z, s in zip(ref_list, supported) if (not s) and z in eps_points),
        "ws_distinct_points": len(ws_points),
        "eps_distinct_points": len(eps_points),
    }
    (out / "nonconvex.json").write_text(json.dumps(nonconvex_info, indent=2))

    # MCDM on the 4-objective reference front
    mcdm_info = run_mcdm(out, ref_fronts[tuple(ALL4)], inst["weights"])

    # many-objective diagnostics: share of the population that is non-dominated at the end
    many_nd = {}
    for lbl in ("Constrained NSGA-II", "NSGA-III"):
        for space, pop in ((tuple(ALL4), 100), (tuple(PAIR2), 100)):
            study = "many" if space == tuple(ALL4) else "main2"
            rs = [r for r in results if r["study"] == study and r["label"] == lbl]
            if rs:
                many_nd[f"{lbl} M={len(space)}"] = float(np.mean([r["history"][-1]["n_front"] / pop for r in rs]))
    (out / "many_nondominated_share.json").write_text(json.dumps(many_nd, indent=2))

    make_figures(out, fig, results, rows, scorers, ref_fronts, conv_rows, supported, nonconvex_info)
    write_summary_md(out, a, summary_rows, nonconvex_info, mcdm_info, many_nd, ref_fronts, scorers)
    print(f"\nDone in {time.time() - t0:.0f}s -> {out}")


def supported_mask(Fn: np.ndarray, steps: int = 2001) -> np.ndarray:
    """A point is 'supported' if some weight vector (w, 1-w) makes it the weighted-sum minimiser
    among the reference front. Unsupported points lie in non-convex dents: no weighted sum finds them."""
    mask = np.zeros(len(Fn), dtype=bool)
    for w in np.linspace(0, 1, steps):
        s = Fn @ np.array([w, 1 - w])
        mask |= np.isclose(s, s.min())
    return mask


def run_mcdm(out: Path, F: np.ndarray, weights: dict) -> dict:
    w = np.array([weights[o] for o in ALL4], dtype=float)
    w = w / w.sum()
    t = mcdm.topsis(F, w)
    res = {"weights": w.tolist(), "n_alternatives": len(F), "topsis_top3": t["ranking"][:3].tolist(),
           "topsis_closeness_top3": t["closeness"][t["ranking"][:3]].tolist(), "vikor": {}}
    rows = []
    for i, idx in enumerate(t["ranking"][:3]):
        rows.append({"method": "TOPSIS", "rank": i + 1, "alternative": int(idx), "score": float(t["closeness"][idx]), **dict(zip(ALL4, F[idx].tolist()))})
    for v in (0.25, 0.5, 0.75):
        vk = mcdm.vikor(F, w, v)
        res["vikor"][str(v)] = {"top3": vk["ranking"][:3].tolist(), "compromise_set": vk["compromise_set"],
                                "acceptable_advantage": vk["acceptable_advantage"], "acceptable_stability": vk["acceptable_stability"]}
        for i, idx in enumerate(vk["ranking"][:3]):
            rows.append({"method": f"VIKOR v={v}", "rank": i + 1, "alternative": int(idx), "score": float(vk["Q"][idx]), **dict(zip(ALL4, F[idx].tolist()))})
    _write_csv(out / "mcdm_top3.csv", rows)
    sens = []
    for method in ("topsis", "vikor"):
        for s in mcdm.weight_sensitivity(F, w, method, 0.1, 0.5):
            sens.append({"method": method, "criterion": ALL4[s["criterion"]], "shift": s["shift"], "new_top": s["top"],
                         "top_changed": s["top_changed"], "new_top3": s["top3"], "weights": s["weights"]})
    _write_csv(out / "mcdm_sensitivity.csv", sens)
    res["sensitivity"] = sens
    return res


def make_figures(out, fig, results, rows, scorers, ref_fronts, conv_rows, supported, nc_info):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    markers = "osD^v<>pXh*"

    # 1. fronts: union of each method's fronts over all seeds, for both 2-objective studies
    labels = [LABELS[a] for a in MAIN_ALGOS]
    for study, space, fname, xl, yl, offset in (
        ("main", PAIR, "fronts_f1_f2.png", "f1 student gaps", "f2 teacher gaps + overload days", 0.06),
        ("main2", PAIR2, "fronts_f3_f4.png", "f3 room waste (unused seats)", "f4 early/late sessions", 3.0),
    ):
        figr, ax = plt.subplots(figsize=(8, 6))
        ref = ref_fronts[tuple(space)]
        o = np.argsort(ref[:, 0])
        ax.plot(ref[o, 0], ref[o, 1], "k*--", lw=1, ms=12, label="reference front (merged)", zorder=1)
        for i, lbl in enumerate(labels):
            pts = [f for r in results if r["study"] == study and r["label"] == lbl for f in r["front"]]
            if pts:
                P = mu.merge_fronts([pts])
                off = (i - 4.5) * offset  # small horizontal offset so overlapping integer points stay visible
                ax.scatter(P[:, 0] + off, P[:, 1], marker=markers[i % len(markers)], color=colors[i % 10], s=40, label=lbl, alpha=0.8, zorder=2)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{study}: non-dominated union of each method over all seeds (offset for visibility)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        figr.tight_layout()
        figr.savefig(fig / fname, dpi=150)
        plt.close(figr)

    # 2. HV / IGD box plots for both 2-objective studies
    for study in ("main", "main2"):
        pair = "f1 vs f2" if study == "main" else "f3 vs f4"
        for metric in ("hv", "igd"):
            figr, ax = plt.subplots(figsize=(10, 4.5))
            data = [[r[metric] for r in rows if r["study"] == study and r["label"] == lbl and np.isfinite(r[metric])] for lbl in labels]
            ax.boxplot([d if d else [np.nan] for d in data], tick_labels=labels, showmeans=True)
            ax.set_ylabel(metric.upper() + (" (higher is better)" if metric == "hv" else " (lower is better)"))
            ax.set_title(f"{study} ({pair}): {metric.upper()} over seeds, equal evaluation budget")
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
            ax.grid(alpha=0.3, axis="y")
            figr.tight_layout()
            figr.savefig(fig / f"{study}_{metric}_box.png", dpi=150)
            plt.close(figr)

    # 3. convergence (mean HV over seeds vs evaluations)
    for study, title in (("main", "f1 vs f2"), ("main2", "f3 vs f4"), ("many", "4 objectives (seeds 1-3)")):
        figr, ax = plt.subplots(figsize=(8, 4.5))
        for i, lbl in enumerate(sorted({c["label"] for c in conv_rows if c["study"] == study})):
            cs = [c for c in conv_rows if c["study"] == study and c["label"] == lbl and c["hv"] is not None]
            if not cs:
                continue
            grid = np.linspace(0, max(c["evaluations"] for c in cs), 60)
            curves = []
            for s in sorted({c["seed"] for c in cs}):
                pts = sorted((c["evaluations"], c["hv"]) for c in cs if c["seed"] == s)
                xs, ys = zip(*pts)
                curves.append(np.interp(grid, xs, ys, left=0.0))
            ax.plot(grid, np.mean(curves, axis=0), label=lbl, color=colors[i % 10])
        ax.set_xlabel("evaluations")
        ax.set_ylabel("hypervolume (common normalisation)")
        ax.set_title(f"Convergence, {title}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        figr.tight_layout()
        figr.savefig(fig / f"convergence_{study}.png", dpi=150)
        plt.close(figr)

    # 4. sensitivity
    sens_sets = [
        ("C-NSGA-II pop_size", [("50", "sens", "C-NSGA-II pop=50"), ("100", "main2", LABELS["constrained_nsga2"]), ("200", "sens", "C-NSGA-II pop=200")]),
        ("MOEA/D neighbourhood T", [("5", "sens", "MOEA/D T=5"), ("20", "main2", LABELS["moead"]), ("50", "sens", "MOEA/D T=50")]),
        ("MOSA cooling rate", [("0.90", "sens", "MOSA alpha=0.9"), ("0.95", "main2", LABELS["mosa"]), ("0.99", "sens", "MOSA alpha=0.99")]),
    ]
    figr, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (title, items) in zip(axes, sens_sets):
        data = [[r["hv"] for r in rows if r["study"] == st and r["label"] == lb] for _, st, lb in items]
        ax.boxplot(data, tick_labels=[v for v, _, _ in items], showmeans=True)
        ax.set_title(title)
        ax.set_ylabel("HV")
        ax.grid(alpha=0.3, axis="y")
    figr.suptitle("Parameter sensitivity, f3 vs f4 (10 seeds each)")
    figr.tight_layout()
    figr.savefig(fig / "sensitivity.png", dpi=150)
    plt.close(figr)

    # 5. repair vs penalty
    items = [("Deb + repair", "main2", LABELS["constrained_nsga2"]), ("Deb, no repair", "ablation", "Deb rules, no repair"),
             ("Penalty + repair", "ablation", "Penalty + repair"), ("Penalty only", "main2", LABELS["nsga2"])]
    figr, axes = plt.subplots(1, 2, figsize=(11, 4))
    feas = [np.mean([r["feasible_rate"] for r in rows if r["study"] == st and r["label"] == lb]) for _, st, lb in items]
    hvs = [[r["hv"] for r in rows if r["study"] == st and r["label"] == lb] for _, st, lb in items]
    axes[0].bar([n for n, _, _ in items], feas, color=colors[:4])
    axes[0].set_ylabel("share of evaluated timetables that were feasible")
    axes[0].set_ylim(0, 1.05)
    axes[1].boxplot(hvs, tick_labels=[n for n, _, _ in items], showmeans=True)
    axes[1].set_ylabel("HV (0 = no feasible timetable found)")
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        ax.grid(alpha=0.3, axis="y")
    figr.suptitle("Ablation: repair vs penalty (f3 vs f4)")
    figr.tight_layout()
    figr.savefig(fig / "ablation_repair_vs_penalty.png", dpi=150)
    plt.close(figr)

    # 6. non-convexity
    nc = tuple(NONCONVEX_PAIR)
    ref = ref_fronts[nc]
    order = np.argsort(ref[:, 0])
    figr, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(ref[order, 0], ref[order, 1], "k--", lw=1, label="reference front")
    ax.scatter(ref[supported, 0], ref[supported, 1], s=110, facecolor="none", edgecolor="green", label="supported (on convex hull)")
    ax.scatter(ref[~supported, 0], ref[~supported, 1], s=110, facecolor="none", edgecolor="red", label="unsupported (non-convex dent)")
    for i, (algo, m) in enumerate([("weighted_sum", "x"), ("eps_constraint", "+")]):
        pts = mu.merge_fronts([[f for r in results if r["study"] == "main2" and r["algorithm"] == algo for f in r["front"]]])
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], marker=m, s=60, color=colors[i], label=LABELS[algo] + " (all seeds)")
    ax.set_xlabel("f3 room waste (unused seats)")
    ax.set_ylabel("f4 early/late sessions")
    ax.set_title(f"Weighted sum vs non-convex front: {nc_info['unsupported']} of {nc_info['reference_points']} points unsupported")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    figr.tight_layout()
    figr.savefig(fig / "nonconvex_weighted_sum.png", dpi=150)
    plt.close(figr)

    # 7. many objectives: pairwise plot of the median-HV run of each method
    names = ["f1 gaps", "f2 teacher", "f3 waste", "f4 early/late"]
    figr, axes = plt.subplots(4, 4, figsize=(12, 12))
    for k, lbl in enumerate([LABELS["constrained_nsga2"], LABELS["nsga3"], LABELS["moead"]]):
        rs = sorted([r for r in rows if r["study"] == "many" and r["label"] == lbl], key=lambda r: r["hv"])
        if not rs:
            continue
        seed = rs[len(rs) // 2]["seed"]
        F = np.array([r["front"] for r in results if r["study"] == "many" and r["label"] == lbl and r["seed"] == seed][0])
        for i in range(4):
            for j in range(4):
                ax = axes[i, j]
                if i == j:
                    ax.hist(F[:, i], bins=12, alpha=0.5, color=colors[k])
                else:
                    ax.scatter(F[:, j], F[:, i], s=8, color=colors[k], alpha=0.6, label=lbl if (i, j) == (0, 1) else None)
                if i == 3:
                    ax.set_xlabel(names[j])
                if j == 0:
                    ax.set_ylabel(names[i])
    axes[0, 1].legend(fontsize=7)
    figr.suptitle("4 objectives: pairwise projections of the median-HV run of each method")
    figr.tight_layout()
    figr.savefig(fig / "many_pairwise.png", dpi=130)
    plt.close(figr)

    figr, ax = plt.subplots(figsize=(7, 4))
    lbls = [LABELS["constrained_nsga2"], LABELS["nsga3"], LABELS["moead"]]
    ax.boxplot([[r["hv"] for r in rows if r["study"] == "many" and r["label"] == l_] for l_ in lbls], tick_labels=lbls, showmeans=True)
    ax.set_ylabel("HV (Monte-Carlo, 200k samples)")
    ax.set_title("4 objectives: hypervolume over 10 seeds")
    ax.grid(alpha=0.3, axis="y")
    figr.tight_layout()
    figr.savefig(fig / "many_hv_box.png", dpi=150)
    plt.close(figr)


def write_summary_md(out, a, summary_rows, nc_info, mcdm_info, many_nd, ref_fronts, scorers):
    titles = {
        "main": "Main comparison A: f1 student gaps vs f2 teacher gaps + overload days (as specified; NOT conflicting here)",
        "main2": "Main comparison B: f3 room waste vs f4 early/late penalty (a conflicting pair)",
        "sens_nsga2": "Sensitivity (f3 vs f4): Constrained NSGA-II population size",
        "sens_moead": "Sensitivity (f3 vs f4): MOEA/D neighbourhood size",
        "sens_mosa": "Sensitivity (f3 vs f4): MOSA cooling rate",
        "ablation_repair": "Ablation (f3 vs f4): repair vs penalty",
        "ablation_memetic": "Ablation (f3 vs f4): local search (MOMA vs Constrained NSGA-II)",
        "many": "Many objectives (M = 4)",
    }
    md = [f"# Lab results\n\nSynthetic instance (generator seed 1, see instance.txt). Seeds 1..{a.seeds}. "
          f"Budget {a.evaluations} evaluations per run. HV: reference point 1.1 after normalising by the merged "
          f"reference front's ideal/nadir; exact in 2-D, Monte-Carlo (200k samples) in 4-D. IGD and Spacing on "
          f"normalised objectives. 'Feasible' = share of evaluated timetables with zero hard-constraint violations; "
          f"'Found' = share of runs that returned at least one feasible timetable.\n"]
    for space, ref in ref_fronts.items():
        sc = scorers[space]
        note = ""
        if sc.fallback_axes:
            names = [space[i] for i in sc.fallback_axes]
            note = f" No spread on {names}: their nadir is the worst feasible value returned by any run."
        md.append(f"- Reference front for {' + '.join(space)}: {len(ref)} points; ideal {sc.ideal.tolist()}, "
                  f"nadir used {[round(x, 3) for x in sc.nadir.tolist()]}.{note}")
    for t, title in titles.items():
        md.append(f"\n## {title}\n")
        md.append("| Method | Runs | HV | IGD | Spacing | Front size | Found | Feasible | Evaluations | Runtime (s) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in [s for s in summary_rows if s["table"] == t]:
            md.append(f"| {r['method']} | {r['runs']} | {fmt(r['hv_mean'], r['hv_sd'])} | {fmt(r['igd_mean'], r['igd_sd'])} | "
                      f"{fmt(r['spacing_mean'], r['spacing_sd'])} | {r['front_size']:.1f} | {r['found_feasible']:.0%} | {r['feasible_rate']:.0%} | "
                      f"{r['evaluations']:.0f} | {r['runtime_s']:.1f} |")
    md.append("\n## Non-convexity check\n")
    md.append(f"Reference front (f3, f4): {nc_info['reference_points']} points, {nc_info['supported']} supported (reachable by some "
              f"weighted sum), {nc_info['unsupported']} unsupported. Weighted Sum found {nc_info['ws_found_supported']} supported and "
              f"{nc_info['ws_found_unsupported']} unsupported reference points ({nc_info['ws_distinct_points']} distinct points over all seeds); "
              f"eps-Constraint found {nc_info['eps_found_unsupported']} unsupported points ({nc_info['eps_distinct_points']} distinct).\n")
    md.append("## Many-objective diagnostic: share of the final population that is non-dominated\n")
    for k, v in many_nd.items():
        md.append(f"- {k}: {v:.0%}")
    md.append("\n## TOPSIS / VIKOR on the 4-objective reference front\n")
    md.append(f"Weights (student gaps, teacher gaps, room waste, time penalty) = {[round(x, 3) for x in mcdm_info['weights']]}; "
              f"{mcdm_info['n_alternatives']} alternatives. Top 3 per method in mcdm_top3.csv.\n")
    md.append(f"- TOPSIS top 3: {mcdm_info['topsis_top3']}")
    for v, info in mcdm_info["vikor"].items():
        md.append(f"- VIKOR v={v} top 3: {info['top3']}, compromise set {info['compromise_set']} "
                  f"(acceptable advantage {info['acceptable_advantage']}, stability {info['acceptable_stability']})")
    flips = [s for s in mcdm_info["sensitivity"] if s["top_changed"]]
    md.append(f"- Weight sensitivity (+/-0.1 on one weight, renormalised): {len(flips)} of {len(mcdm_info['sensitivity'])} shifts change the top choice:")
    for s in flips:
        md.append(f"  - {s['method']}: {s['criterion']} {s['shift']:+.1f} -> new top {s['new_top']}")
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
