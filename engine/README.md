# Timetable optimisation engine (M.Tech Optimization Techniques lab)

Hand-coded multi-objective engine for university timetabling. **No UI**: the Clockwork website shows the results.

- **Libraries:** Python 3.10+, NumPy and the standard library only. Matplotlib is used only by the offline experiment scripts.
- **Hand-coded:** every algorithm and every shared tool. No pymoo, DEAP, Platypus, jMetalPy, inspyred or SciPy.
- **No SciPy at all:** the scalarised sub-problems of the classical methods are discrete, so `scipy.optimize` does not apply. A hand-coded simulated annealer solves them instead (`algorithms/scalar_solver.py`).
- **Entry point:** `engine.solve(instance: dict) -> dict`. The transport lives in a separate thin layer, `../adapter.py`.
- **Formulation:** one-page problem formulation in [FORMULATION.md](FORMULATION.md).

## Run commands

Run everything from the project root (`D:\OT project`).

```bash
python -m unittest discover -s engine/tests -t .                 # all unit + end-to-end tests
python adapter.py solve samples/lab_instance.json -o out.json --progress   # CLI: JSON file in, JSON out
python adapter.py solve - < samples/frontend_instance.json       # CLI: stdin -> stdout
python adapter.py serve --port 8001                              # HTTP server for the website
python -m engine.generator --seed 1 --out results/lab/instance.json   # print + save the lab instance
python -m experiments.engine_lab                                 # every lab experiment (~11 min, 11 workers)
python -m experiments.engine_nonconvex4d                         # weighted-sum reachability on the 4-D front
```

### Transport choice: standard-library `http.server` plus a CLI

- **HTTP:** the website already talks JSON over HTTP through Vite's `/api` proxy, so HTTP is what it can call.
- **Standard library only:** the lab rules allow only NumPy and the standard library, so no Flask or FastAPI is used.
- **CLI:** lab scripts and CI can pipe JSON straight through.

| Endpoint | Purpose |
|---|---|
| `GET /engine/health` | `{"ok": true}` |
| `GET /engine/algorithms` | names, catalogue numbers, default parameters |
| `POST /engine/solve` | `solve(body)`, synchronous |
| `POST /engine/runs` | start a background run, returns `{"run_id"}` |
| `GET /engine/runs/<id>` | `{"status", "progress", "result"?}`; progress = latest generation, HV, best violation |
| `POST /engine/runs/<id>/cancel` | early cancellation; the best front found so far is returned |

`solve(instance, progress=None, should_stop=None)` has two hooks:
- `progress(entry)` is called after every generation; returning `False` cancels the run.
- `should_stop()` is polled between generations.

A run with the same input and seed always gives the same result. One exception: a run cut short by `time_limit_s` depends on how fast the machine is.

## Input schema

### Engine format

Slot indices in `unavailable_slots` can be flat (`day * slots_per_day + slot`) or `[day, slot]` pairs.

```json
{
  "days": 5, "slots_per_day": 6,
  "rooms":    [{"id": "R1", "capacity": 40}],
  "teachers": [{"id": "T1", "unavailable_slots": [7, [2, 3]]}],
  "groups":   [{"id": "G1", "size": 46}],
  "lectures": [{"id": "L1", "course": "Optimization", "teacher_id": "T1", "group_id": "G1", "sessions_per_week": 3}],
  "weights":  {"student_gaps": 0.4, "teacher_gaps": 0.3, "room_waste": 0.2, "time_penalty": 0.1},
  "algorithm": "constrained_nsga2",
  "params":   {"pop_size": 100, "evaluations": 10000, "seed": 1, "time_limit_s": 60}
}
```

Optional `params`:
- `objectives`: which of the four to optimise. Default: those with a positive weight, or all four.
- `decision`: `"topsis"` (default) or `"vikor"`, used to pick `recommended`.
- `vikor_v`: default 0.5.
- `reference_front`: objective vectors; if given, IGD is computed against it.
- Any algorithm parameter (see `GET /engine/algorithms`).

### Website format

`solve()` also accepts the website's own instance format, unchanged:

```json
{"instance": {"grid": {"days": [...], "periods": [...], "lunchPeriod": 4}, "groups": [...], "faculty": [...], "rooms": [...], "offerings": [...]},
 "algorithm": "...", "weights": {...}, "params": {...}}
```

`frontend_bridge.py` maps it onto the engine format. The lab model is simpler than the website's, so three things are reduced, and each is reported in `warnings`:
- **Fixed teacher:** each offering's teacher is fixed to its **first** eligible teacher.
- **Single group:** only the **first** group of a shared offering is scheduled.
- **Labs:** a 2-period lab becomes two 1-slot sessions, and they are **not forced to be consecutive**.

Lunch is removed from the slot grid. Each returned timetable entry gets the website's `period` index (with lunch put back), plus `day_name` and `period_label`.

### Validation

Bad input returns `{"error": "Invalid instance", "details": [...]}` and never raises. It catches:
- unknown or duplicate ids;
- a group larger than every room;
- more sessions than a group's weekly slots, or than a teacher's available slots;
- more sessions in total than room-slots;
- unavailable slots outside the grid;
- bad weights or parameters;
- an unknown algorithm.

## Output schema

```json
{"status": "ok" | "no_feasible_solution", "algorithm": "...", "objective_names": [...], "weights": {...},
 "front": [{"objectives": [...], "objective_values": {...}, "violation": 0,
            "timetable": [{"lecture_id", "session", "day", "slot", "room_id"}]}],
 "recommended": 3,
 "ranking": {"topsis": [3, 0, ...], "vikor": [...]},
 "metrics": {"hv", "hv_normalisation", "igd", "spacing", "feasible_rate", "runtime_s", "evaluations", "stopped"},
 "history": [{"gen", "evaluations", "hv", "best_violation", "n_front", "seconds"}],
 "warnings": [...]}
```

Notes on the fields:
- **`hv`:** normalised by ideal = 0 and the nadir of the first evaluated batch, with reference point 1.1 on each axis.
- **`igd`:** `null` unless a `reference_front` is passed, because a single run has no true front to compare against.
- **`feasible_rate`:** the share of ALL evaluated timetables that had zero hard-constraint violations.
- **`stopped`:** one of `budget`, `time_limit`, `cancelled` or `completed`.

## Implementation log

Experiment set-up:
- **Instance:** generator seed 1 (5 groups, 10 teachers, 8 rooms, 25 lectures, 89 sessions, 5 days × 6 slots).
- **Runs:** 10 seeds (1–10) per method, 10,000 evaluations each.
- **Hypervolume:** normalised by the merged reference front, reference point 1.1.
- **2-objective numbers:** taken from **study B (f3 room waste vs f4 early/late)**, the conflicting pair; see the findings below.
- **Runtime:** wall-clock per run with 11 runs in parallel; a run on its own is about 30–40% faster.

Source: `results/lab/summary.md` and `summary.csv`.

| No. | Algorithm (`algorithm` name) | Budget | HV (mean ± sd) | IGD (mean ± sd) | Runtime (s) | Status |
|---|---|---|---|---|---|---|
| 14 | NSGA-II, penalty (`nsga2`) | 10,000 | 0.000 ± 0.000 | n/a (never feasible) | 5.9 | Works as specified; the penalty baseline found **no** feasible timetable in 10/10 runs |
| 21 | Constrained NSGA-II (`constrained_nsga2`, default) | 10,000 | 0.110 ± 0.089 | 0.569 ± 0.177 | 23.4 | Works |
| 16 | SPEA2 (`spea2`) | 10,000 | 0.141 ± 0.116 | 0.496 ± 0.180 | 26.5 | Works |
| 18 | MOEA/D, Tchebycheff (`moead`) | 10,000 | 0.200 ± 0.141 | 0.361 ± 0.142 | 36.8 | Works |
| 26 | MOSA (`mosa`) | 10,000 | **0.237 ± 0.117** | **0.354 ± 0.189** | 12.2 | Works |
| 30 | MOMA (`moma`) | 10,000 | 0.009 ± 0.028 | 1.230 ± 0.377 | 20.7 | Works, but underperforms (see ablation) |
| 15 | NSGA-III (`nsga3`), 4 objectives | 10,000 | 0.054 ± 0.046 (M=4) | 0.816 ± 0.126 (M=4) | 33.1 | Works; weakest of the three at M = 4 |
| 1 | Weighted Sum (`weighted_sum`) | 10,000 | 0.053 ± 0.079 | 0.655 ± 0.167 | 12.3 | Works |
| 2 | ε-Constraint (`eps_constraint`) | 10,000 | 0.045 ± 0.095 | 0.736 ± 0.164 | 11.9 | Works; every ε level satisfied in the tests |
| 6 | Goal Programming (`goal_programming`) | 10,000 | 0.038 ± 0.047 | 0.719 ± 0.099 | 12.2 | Works |
| 7 | Lexicographic (`lexicographic`) | 10,000 | 0.000 ± 0.000 | 1.358 ± 0.359 | 15.5 | Works (returns one timetable by design) |
| 8 | TOPSIS (`mcdm.topsis`) | n/a | n/a | n/a | <0.01 | Works |
| 9 | VIKOR (`mcdm.vikor`) | n/a | n/a | n/a | <0.01 | Works |
| | Greedy repair (`problem.repair`) | n/a | n/a | n/a | ~1 ms | **Partly works:** 100% success from random timetables on the lab instance, only 70–81% on the deliberately tight 4-slot test instance |
| | Website-format bridge (`frontend_bridge`) | n/a | n/a | n/a | n/a | **Partly works:** fixed teacher, first group only, labs not kept consecutive |
| | Order crossover | n/a | n/a | n/a | n/a | **Not done:** the (slot, room) encoding is not a permutation, so uniform crossover is used |

Tests: `python -m unittest discover -s engine/tests -t .` has 72 tests, all passing. They are the hand-checkable toolkit tests, one key-rule test per algorithm, the end-to-end tests and the HTTP adapter tests.

## Findings from my own runs

Each finding is backed by a table in `results/lab/summary.md` or a figure in `results/lab/figures/`.

**1. f1 and f2 do not conflict on this instance.**
- The merged reference front for (student gaps, teacher gaps) is one point, (0, 2), found by MOSA. It dominates every other timetable any method returned.
- So study A (`fronts_f1_f2.png`, `main_hv_box.png`) measures convergence towards that point, not trade-off coverage. Its HV is normalised against the worst feasible value returned, because the front has no spread.
- On study A, MOSA leads again: HV 1.190 ± 0.017, IGD 0.026.
- The generator's defaults (3–4 sessions per lecture, 3–6 blocked slots per teacher) were already tightened from 2–3 and 1–3 for this reason: with the looser settings even f1 = f2 = 0 was reachable.
- A check of all six objective pairs (MOSA + Constrained NSGA-II, seeds 1–3) found a real front only for **f3 room waste vs f4 early/late**: 6 points, from (1464, 10) to (1614, 5). That pair is study B.

**2. Repair is essential; the constraint-handling rule matters much less** (`ablation_repair_vs_penalty.png`).

| Constraint handling | Feasible runs | HV (f3 vs f4) |
|---|---|---|
| Deb rules + repair | 10/10 | 0.110 ± 0.089 |
| Penalty + repair | 10/10 | 0.088 ± 0.071 |
| Deb rules, no repair | **0/10** | 0 |
| Penalty, no repair | **0/10** | 0 |

With 89 sessions in 30 slots, uniform crossover and mutation almost never produce a clash-free timetable on their own.

**3. Local search hurts at this budget.**
- MOMA scored HV 0.009 vs 0.110 for Constrained NSGA-II, and IGD 1.230 vs 0.569.
- Hill climbing spends 20% of children × 5 evaluations from the same 10,000 budget, so MOMA runs only about half as many generations, and the small local moves do not repay that.

**4. MOSA is the best 2-objective method here, and the fastest.**
- On study B: HV 0.237, IGD 0.354, 12 s per run.
- Its single-solution walk with repair works well on a front of only 6 points.

**5. Sensitivity** (`sensitivity.png`). All three parameters show that, at a fixed 10,000-evaluation budget, the setting that allows more iterations or more focus wins:

| Parameter | Values tried | HV | Best setting |
|---|---|---|---|
| Constrained NSGA-II population size | 50 / 100 / 200 | 0.396 / 0.110 / 0.000 | 50: smaller populations get more generations |
| MOEA/D neighbourhood T | 5 / 20 / 50 | 0.035 / 0.200 / 0.250 | 50: a wider neighbourhood spreads good children further |
| MOSA cooling rate α | 0.90 / 0.95 / 0.99 | 0.414 / 0.237 / 0.000 | 0.90: at 0.99 the walk is still hot after 100 temperature levels (0.99¹⁰⁰ ≈ 0.37) |

**6. Weighted Sum and non-convex fronts.**
- **2-D (study B):** the front is convex (all 6 points supported, `nonconvex_weighted_sum.png`), so the non-convexity weakness doesn't show there. Weighted Sum is simply weaker: HV 0.053, 17 distinct points over 10 seeds, none on the reference front.
- **4-D:** **16 of the 43 reference-front timetables (37%) are unsupported**. No weight vector makes them the weighted-sum optimum, so no number of weights can find them (`results/lab/nonconvex_4d.json`, 200,000 random weight vectors).

**7. Many objectives (M = 4)** (`many_hv_box.png`, `many_pairwise.png`):

| Method | HV | IGD |
|---|---|---|
| MOEA/D | **0.633 ± 0.269** | **0.285** |
| Constrained NSGA-II | 0.180 ± 0.134 | 0.604 |
| NSGA-III | 0.054 ± 0.046 | 0.816 |

What degrades:
- **Selection pressure:** about half of the final population is non-dominated at M = 4 (Constrained NSGA-II 48%, NSGA-III 50%) against 4% at M = 2. With half the population tied on rank, selection falls back on crowding or niching. Pareto ranking then barely separates good and bad timetables.
- **NSGA-III:** its 84 reference directions spread the population over the whole simplex. With so few evaluations, that spends effort on regions that never get good.
- **MOEA/D:** its scalarised subproblems keep selection pressure regardless of M.

**8. Decision making** (`mcdm_top3.csv`, `mcdm_sensitivity.csv`).
- **Set-up:** the 43-point 4-objective reference front, with weights (0.4, 0.3, 0.2, 0.1).
- **Top choices:** TOPSIS ranks alternatives 1, 0, 2. VIKOR picks alternative 8 for every v = 0.25 / 0.5 / 0.75, with compromise set {8, 9} at v = 0.25 and {8} otherwise.
- **Robustness:** shifting one weight by ±0.1 flips TOPSIS's top choice in 5 of 8 shifts and VIKOR's in 0 of 8. VIKOR's compromise choice is more robust here.

**9. Convergence** (`convergence_main2.png`).
- MOSA, MOEA/D, SPEA2 and Constrained NSGA-II are still improving at 10,000 evaluations, so a larger budget would change the absolute numbers.
- The scalarising methods plateau early, because each sub-problem only gets 1/11 to 1/13 of the budget.

## Comparison: advantages and disadvantages

Figures are on study B (f3 vs f4) unless marked.

| Algorithm | Advantages (from my runs) | Disadvantages (from my runs) |
|---|---|---|
| NSGA-II, penalty (14) | Simple; fastest per run (5.9 s) | 0/10 runs feasible: a static penalty without repair cannot reach feasibility here |
| Constrained NSGA-II (21) | 10/10 feasible; Deb's rules need no penalty weight; robust default | Mid-table HV 0.110; needs a small population (pop 50 gave 0.396) |
| SPEA2 (16) | Better than Constrained NSGA-II (HV 0.141, IGD 0.496); largest fronts among GAs (5.9 points) | Slower (26.5 s): fitness needs all pairwise distances |
| MOEA/D (18) | Best at M = 4 by far (HV 0.633); second on 2 objectives (0.200) | Slowest (36.8 s, 50 s with T = 50); sensitive to T (T = 5 gave 0.035) |
| MOSA (26) | Best on 2 objectives (HV 0.237, IGD 0.354) and on f1 vs f2 (1.190); fastest of the working methods (12.2 s) | Very sensitive to cooling: α = 0.99 gave HV 0; one walk gives no diversity guarantee |
| MOMA (30) | Local moves reach compact days quickly within a run | HV 0.009 vs 0.110 for Constrained NSGA-II: local search eats the budget |
| NSGA-III (15) | Even spread by design; good in theory for M ≥ 3 | Worst at M = 4 (HV 0.054) at this budget; 84 reference directions are too many for 10,000 evaluations |
| Weighted Sum (1) | Easy to explain; one clear answer per weight vector | HV 0.053; cannot reach 16 of 43 points of the 4-D front (non-convex parts) |
| ε-Constraint (2) | Reaches non-convex points in principle; ε constraints always satisfied in tests | HV 0.045; most diverse output (39 distinct points) but far from the front: 1/13 of the budget per sub-problem |
| Goal Programming (6) | Natural when the office has targets ("at most 10 early/late sessions") | HV 0.038; worst spacing (1.430); quality depends on the goals chosen |
| Lexicographic (7) | Matches a strict priority order exactly; each stage provably keeps the earlier optima (test) | One timetable, no trade-off (HV 0 on study B); later criteria get leftovers |
| TOPSIS (8) | Quick, transparent ranking of a front | Top choice flips in 5/8 one-weight shifts |
| VIKOR (9) | Same top choice for v = .25/.5/.75 and in 8/8 weight shifts; reports a compromise set | Needs v and the acceptance conditions explained to the decision-maker |

## Which algorithm for which situation

- **Heavy hard constraints** (many clashes to avoid, tight rooms or teachers): use a method **with repair**. Every method without repair produced 0 feasible runs. Constrained NSGA-II with repair is the safest default (10/10 feasible, no penalty weight to tune). Avoid plain penalty NSGA-II.
- **Clear priority order** (e.g. "no teacher clashes, then student gaps, then everything else"): use the **Lexicographic** method. It returns the single timetable that respects the order exactly. Avoid it when the office wants to see trade-offs.
- **Many objectives (3+):** use **MOEA/D**, which kept selection pressure and had the best HV at M = 4 (0.633). Avoid plain Pareto ranking (NSGA-II / SPEA2) and, at small budgets, NSGA-III.
- **Tight budget or quick answer:** use **MOSA** with a fast cooling rate (α = 0.90). It was the fastest working method and the best on two objectives. Avoid MOMA and large populations, which spend the budget on local search or on breadth.
- **The office has targets rather than weights:** **Goal Programming**. **To pick one timetable** from the front: **VIKOR** for a decision that is stable under small weight changes, TOPSIS for a quick ranking. Avoid relying on a **Weighted Sum** sweep alone when the front may be non-convex.

## Files

```
engine/
  solve.py              solve() entry point, validation of params, MCDM, metrics
  problem.py            validation, encoding, H1-H5, f1-f4, repair
  operators.py          uniform crossover, swap / slot-reassign / room-reassign mutation
  local_search.py       swap / move neighbourhood, hill climbing
  moo_utils.py          dominance, Deb's rules, FNDS, crowding, HV (2-D exact, MC), IGD, Spacing, normalisation
  mcdm.py               TOPSIS, VIKOR, weight sensitivity
  generator.py          seeded synthetic instance (tests and lab only)
  frontend_bridge.py    website format -> engine format
  algorithms/           one module per algorithm (initialize, evaluate, step, output, run) + common run loop
  tests/                unittest suites
adapter.py              CLI + http.server transport
experiments/engine_lab.py, experiments/engine_nonconvex4d.py   lab experiments
results/lab/            CSVs, JSON, summary.md and figures behind every number above
samples/                lab-format and website-format example inputs
```
