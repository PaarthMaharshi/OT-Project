# Problem formulation (one page)

**Data.** Days D, slots per day P, T = D·P time slots. Rooms r with capacity cap_r. Teachers k with a set of
unavailable slots U_k. Student groups g with size n_g. Lectures l with a fixed teacher k(l), a fixed
group g(l) and a required number of weekly sessions m_l. The sessions s = 1..S (S = Σ m_l) inherit their
teacher k(s) and group g(s) from their lecture.

**Decision variables (encoding).** For every session s, a pair (t_s, r_s): its time slot t_s ∈ {0..T−1}
(t = day·P + slot) and its room r_s ∈ {0..R−1}. A solution is the integer vector
x = (t_1..t_S, r_1..r_S) ∈ ℤ^{2S}. Teacher and group are data, not variables.

**Standard form.**

  min F(x) = (f_1(x), f_2(x) [, f_3(x), f_4(x)])  subject to  V(x) = 0,  x ∈ ℤ^{2S}, 0 ≤ t_s < T, 0 ≤ r_s < R

with total violation V(x) = H1 + H2 + H3 + H4 + H5 (counts):

| | Hard constraint | Count added to V |
|---|---|---|
| H1 | teacher in one place per slot, and available | Σ_{k,t} max(0, #sessions of k at t − 1) + #{s : t_s ∈ U_{k(s)}} |
| H2 | group in one lecture per slot | Σ_{g,t} max(0, #sessions of g at t − 1) |
| H3 | room not double-booked | Σ_{r,t} max(0, #sessions in r at t − 1) |
| H4 | room capacity ≥ group size | #{s : cap_{r_s} < n_{g(s)}} |
| H5 | exactly m_l sessions per lecture | Σ_l abs(placed_l − m_l) + out-of-range genes (0 by construction) |

**Objectives (all minimised).** For a row of busy flags b_1..b_P of one group (teacher) on one day,
gap = (last − first + 1) − Σ b_p, and 0 for an empty day.

- f1 student gaps = Σ_{g,day} gap
- f2 teacher gaps + overload days = Σ_{k,day} gap + #{(k, day) : sessions > 4}
- f3 room waste = Σ_s max(0, cap_{r_s} − n_{g(s)}) (unused seats)
- f4 early/late penalty = #{s : slot(t_s) ∈ {first, last slot of the day}}

Bi-objective version: (f1, f2). Many-objective version: (f1, f2, f3, f4).

**Constraint handling.** Main solver: Deb's feasibility rules (feasible beats infeasible; between
infeasible the smaller V wins; between feasible Pareto dominance) plus a greedy repair operator that
moves clashing sessions to the nearest free slot and the smallest fitting free room. Baseline: static
penalty f_j + 100·V with plain Pareto dominance and no repair. Scalarising methods use "feasibility
first": a lower penalty level always wins, then the scalar is compared.

**Objective normalisation.** f̃_j = (f_j − z_j^ideal) / (z_j^nadir − z_j^ideal).
Inside a run (progress history, MOSA energy, MOEA/D Tchebycheff, weighted sums) the ideal is 0 (no
objective can be negative) and the nadir is the worst value in the first evaluated batch. In the
experiments the ideal and nadir come from the merged reference front. If that front has no spread on an
axis (ideal = nadir, which happens when one timetable dominates everything, as for f1 vs f2 on the lab
instance), that axis's nadir is instead the worst value of the union of all feasible points returned in
that objective space, so HV and IGD still measure how close each run came.

**Reference front.** The true Pareto front is unknown. The reference front for each objective space is
the non-dominated union of every feasible solution returned by every run of every algorithm in that space.

**Hypervolume reference point.** r = (1.1, …, 1.1) in normalised space. HV is exact in 2-D (sweep line)
and Monte-Carlo in 3+ D (200,000 samples in the experiments, fixed seed).

**Termination.** Stop when the evaluation budget is used (default 10,000), when time_limit_s is reached,
or when the caller cancels; the best front found so far is returned in every case. Classical methods
also stop once their last sub-problem is solved.
