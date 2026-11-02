# Lab results

Synthetic instance (generator seed 1, see instance.txt). Seeds 1..10. Budget 10000 evaluations per run. HV: reference point 1.1 after normalising by the merged reference front's ideal/nadir; exact in 2-D, Monte-Carlo (200k samples) in 4-D. IGD and Spacing on normalised objectives. 'Feasible' = share of evaluated timetables with zero hard-constraint violations; 'Found' = share of runs that returned at least one feasible timetable.

- Reference front for room_waste + time_penalty: 6 points; ideal [1464.0, 5.0], nadir used [1614.0, 10.0].
- Reference front for student_gaps + teacher_gaps: 1 points; ideal [0.0, 2.0], nadir used [32.0, 36.0]. No spread on ['student_gaps', 'teacher_gaps']: their nadir is the worst feasible value returned by any run.
- Reference front for student_gaps + teacher_gaps + room_waste + time_penalty: 43 points; ideal [0.0, 2.0, 1464.0, 7.0], nadir used [12.0, 13.0, 1994.0, 23.0].

## Main comparison A: f1 student gaps vs f2 teacher gaps + overload days (as specified; NOT conflicting here)

| Method | Runs | HV | IGD | Spacing | Front size | Found | Feasible | Evaluations | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| NSGA-II (penalty) | 10 | 0.000 ± 0.000 | n/a | n/a | 0.0 | 0% | 0% | 10000 | 6.1 |
| Constrained NSGA-II | 10 | 1.160 ± 0.023 | 0.062 ± 0.022 | 0.004 ± 0.012 | 2.0 | 100% | 100% | 10000 | 21.8 |
| SPEA2 | 10 | 1.154 ± 0.034 | 0.085 ± 0.048 | 0.025 ± 0.032 | 2.8 | 100% | 100% | 10000 | 24.5 |
| MOEA/D | 10 | 1.169 ± 0.020 | 0.046 ± 0.015 | 0.005 ± 0.014 | 1.8 | 100% | 100% | 10000 | 35.3 |
| MOSA | 10 | 1.190 ± 0.017 | 0.026 ± 0.017 | 0.000 ± 0.000 | 1.3 | 100% | 100% | 10000 | 13.9 |
| MOMA | 10 | 1.006 ± 0.047 | 0.218 ± 0.063 | 0.050 ± 0.045 | 4.3 | 100% | 100% | 10000 | 20.4 |
| Weighted Sum | 10 | 1.172 ± 0.021 | 0.065 ± 0.019 | 0.024 ± 0.075 | 2.1 | 100% | 100% | 10000 | 12.3 |
| eps-Constraint | 10 | 1.144 ± 0.015 | 0.090 ± 0.026 | 0.088 ± 0.110 | 2.9 | 100% | 100% | 10000 | 11.2 |
| Goal Programming | 10 | 1.088 ± 0.044 | 0.127 ± 0.045 | 0.226 ± 0.115 | 2.8 | 100% | 100% | 10000 | 12.2 |
| Lexicographic | 10 | 1.113 ± 0.030 | 0.086 ± 0.026 | n/a | 1.0 | 100% | 100% | 10000 | 11.9 |

## Main comparison B: f3 room waste vs f4 early/late penalty (a conflicting pair)

| Method | Runs | HV | IGD | Spacing | Front size | Found | Feasible | Evaluations | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| NSGA-II (penalty) | 10 | 0.000 ± 0.000 | n/a | n/a | 0.0 | 0% | 0% | 10000 | 5.9 |
| Constrained NSGA-II | 10 | 0.110 ± 0.089 | 0.569 ± 0.177 | 0.071 ± 0.082 | 4.1 | 100% | 100% | 10000 | 23.4 |
| SPEA2 | 10 | 0.141 ± 0.116 | 0.496 ± 0.180 | 0.169 ± 0.158 | 5.9 | 100% | 100% | 10000 | 26.5 |
| MOEA/D | 10 | 0.200 ± 0.141 | 0.361 ± 0.142 | 0.056 ± 0.075 | 6.9 | 100% | 100% | 10000 | 36.8 |
| MOSA | 10 | 0.237 ± 0.117 | 0.354 ± 0.189 | 0.057 ± 0.064 | 6.6 | 100% | 100% | 10000 | 12.2 |
| MOMA | 10 | 0.009 ± 0.028 | 1.230 ± 0.377 | 0.256 ± 0.144 | 3.9 | 100% | 100% | 10000 | 20.7 |
| Weighted Sum | 10 | 0.053 ± 0.079 | 0.655 ± 0.167 | 0.285 ± 0.443 | 2.6 | 100% | 100% | 10000 | 12.3 |
| eps-Constraint | 10 | 0.045 ± 0.095 | 0.736 ± 0.164 | 0.552 ± 0.281 | 5.1 | 100% | 100% | 10000 | 11.9 |
| Goal Programming | 10 | 0.038 ± 0.047 | 0.719 ± 0.099 | 1.430 ± 0.573 | 3.2 | 100% | 100% | 10000 | 12.2 |
| Lexicographic | 10 | 0.000 ± 0.000 | 1.358 ± 0.359 | n/a | 1.0 | 100% | 100% | 10000 | 15.5 |

## Sensitivity (f3 vs f4): Constrained NSGA-II population size

| Method | Runs | HV | IGD | Spacing | Front size | Found | Feasible | Evaluations | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| pop_size=50 | 10 | 0.396 ± 0.113 | 0.215 ± 0.103 | 0.017 ± 0.036 | 5.5 | 100% | 100% | 10000 | 23.8 |
| pop_size=100 | 10 | 0.110 ± 0.089 | 0.569 ± 0.177 | 0.071 ± 0.082 | 4.1 | 100% | 100% | 10000 | 23.4 |
| pop_size=200 | 10 | 0.000 ± 0.000 | 1.341 ± 0.353 | 0.134 ± 0.078 | 4.2 | 100% | 100% | 10000 | 39.3 |

## Sensitivity (f3 vs f4): MOEA/D neighbourhood size

| Method | Runs | HV | IGD | Spacing | Front size | Found | Feasible | Evaluations | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| T=5 | 10 | 0.035 ± 0.034 | 0.671 ± 0.099 | 0.188 ± 0.153 | 7.0 | 100% | 100% | 10000 | 37.7 |
| T=20 | 10 | 0.200 ± 0.141 | 0.361 ± 0.142 | 0.056 ± 0.075 | 6.9 | 100% | 100% | 10000 | 36.8 |
| T=50 | 10 | 0.250 ± 0.092 | 0.315 ± 0.086 | 0.177 ± 0.182 | 6.0 | 100% | 100% | 10000 | 50.0 |

## Sensitivity (f3 vs f4): MOSA cooling rate

| Method | Runs | HV | IGD | Spacing | Front size | Found | Feasible | Evaluations | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| alpha=0.90 | 10 | 0.414 ± 0.160 | 0.188 ± 0.128 | 0.035 ± 0.060 | 6.1 | 100% | 100% | 10000 | 14.4 |
| alpha=0.95 | 10 | 0.237 ± 0.117 | 0.354 ± 0.189 | 0.057 ± 0.064 | 6.6 | 100% | 100% | 10000 | 12.2 |
| alpha=0.99 | 10 | 0.000 ± 0.000 | 1.655 ± 0.518 | 0.153 ± 0.088 | 6.5 | 100% | 100% | 10000 | 12.4 |

## Ablation (f3 vs f4): repair vs penalty

| Method | Runs | HV | IGD | Spacing | Front size | Found | Feasible | Evaluations | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| Deb rules + repair (C-NSGA-II) | 10 | 0.110 ± 0.089 | 0.569 ± 0.177 | 0.071 ± 0.082 | 4.1 | 100% | 100% | 10000 | 23.4 |
| Deb rules, no repair | 10 | 0.000 ± 0.000 | n/a | n/a | 0.0 | 0% | 0% | 10000 | 5.3 |
| Penalty + repair | 10 | 0.088 ± 0.071 | 0.576 ± 0.172 | 0.079 ± 0.077 | 4.6 | 100% | 100% | 10000 | 23.4 |
| Penalty, no repair (NSGA-II) | 10 | 0.000 ± 0.000 | n/a | n/a | 0.0 | 0% | 0% | 10000 | 5.9 |

## Ablation (f3 vs f4): local search (MOMA vs Constrained NSGA-II)

| Method | Runs | HV | IGD | Spacing | Front size | Found | Feasible | Evaluations | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| Constrained NSGA-II | 10 | 0.110 ± 0.089 | 0.569 ± 0.177 | 0.071 ± 0.082 | 4.1 | 100% | 100% | 10000 | 23.4 |
| MOMA (= C-NSGA-II + hill climbing) | 10 | 0.009 ± 0.028 | 1.230 ± 0.377 | 0.256 ± 0.144 | 3.9 | 100% | 100% | 10000 | 20.7 |

## Many objectives (M = 4)

| Method | Runs | HV | IGD | Spacing | Front size | Found | Feasible | Evaluations | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| Constrained NSGA-II | 10 | 0.180 ± 0.134 | 0.604 ± 0.158 | 0.120 ± 0.022 | 48.2 | 100% | 100% | 10000 | 29.8 |
| NSGA-III | 10 | 0.054 ± 0.046 | 0.816 ± 0.126 | 0.151 ± 0.035 | 49.6 | 100% | 100% | 10000 | 33.1 |
| MOEA/D | 10 | 0.633 ± 0.269 | 0.285 ± 0.107 | 0.124 ± 0.024 | 54.7 | 100% | 100% | 10000 | 38.9 |

## Non-convexity check

Reference front (f3, f4): 6 points, 6 supported (reachable by some weighted sum), 0 unsupported. Weighted Sum found 0 supported and 0 unsupported reference points (17 distinct points over all seeds); eps-Constraint found 0 unsupported points (39 distinct).

## Many-objective diagnostic: share of the final population that is non-dominated

- Constrained NSGA-II M=4: 48%
- Constrained NSGA-II M=2: 4%
- NSGA-III M=4: 50%

## TOPSIS / VIKOR on the 4-objective reference front

Weights (student gaps, teacher gaps, room waste, time penalty) = [0.4, 0.3, 0.2, 0.1]; 43 alternatives. Top 3 per method in mcdm_top3.csv.

- TOPSIS top 3: [1, 0, 2]
- VIKOR v=0.25 top 3: [8, 9, 10], compromise set [8, 9] (acceptable advantage False, stability True)
- VIKOR v=0.5 top 3: [8, 10, 9], compromise set [8] (acceptable advantage True, stability True)
- VIKOR v=0.75 top 3: [8, 10, 0], compromise set [8] (acceptable advantage True, stability True)
- Weight sensitivity (+/-0.1 on one weight, renormalised): 5 of 16 shifts change the top choice:
  - topsis: student_gaps -0.1 -> new top 8
  - topsis: teacher_gaps +0.1 -> new top 8
  - topsis: room_waste -0.1 -> new top 2
  - topsis: time_penalty +0.1 -> new top 2
  - topsis: time_penalty -0.1 -> new top 0