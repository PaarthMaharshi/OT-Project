"""Problem model: input validation, encoding, constraint violation, objectives and repair.

Encoding
    Every lecture needs `sessions_per_week` sessions. Sessions are numbered s = 0 .. S-1.
    A solution x is one integer vector of length 2S:
        x[s]     = time slot of session s, t = day * slots_per_day + slot   (0 .. T-1)
        x[S + s] = room index of session s                                 (0 .. R-1)
    Teacher and group of a session come from its lecture, so they are not decision variables.

Hard constraints (V(x) = total number of violations, 0 means feasible)
    H1 teacher clash: two sessions of one teacher in the same slot (+1 per extra session),
       plus +1 for every session placed in one of the teacher's unavailable slots
    H2 group clash:   two sessions of one student group in the same slot
    H3 room clash:    two sessions in one room in the same slot
    H4 capacity:      +1 for every session whose room is smaller than its group
    H5 sessions:      every lecture has exactly its required number of sessions. The encoding gives
                      each lecture exactly that many genes, so H5 holds by construction; it is still
                      counted (|placed - required| plus out-of-range genes) as a safety check.

Objectives (all minimised)
    f1 student_gaps  idle slots between a group's first and last session of a day, summed
    f2 teacher_gaps  the same for teachers, plus 1 for every teacher-day with more than 4 sessions
    f3 room_waste    unused seats, sum over sessions of max(capacity - group size, 0)
    f4 time_penalty  number of sessions in the first or last slot of a day
"""
from __future__ import annotations

import numpy as np

OBJECTIVES = ["student_gaps", "teacher_gaps", "room_waste", "time_penalty"]
OVERLOAD_LIMIT = 4  # more than this many sessions in a day is an "overload day" (f2)
HARD_CONSTRAINTS = ["H1_teacher", "H2_group", "H3_room", "H4_capacity", "H5_sessions"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _slot_index(item, days: int, spd: int):
    """Unavailable slots may be given as a flat index or as [day, slot]."""
    if isinstance(item, (list, tuple)) and len(item) == 2:
        d, s = item
        if not (isinstance(d, int) and isinstance(s, int)) or not (0 <= d < days and 0 <= s < spd):
            return None
        return d * spd + s
    if isinstance(item, int) and not isinstance(item, bool) and 0 <= item < days * spd:
        return item
    return None


def validate_instance(inst) -> list[str]:
    """Return a list of human-readable problems; an empty list means the instance is usable."""
    errs: list[str] = []
    if not isinstance(inst, dict):
        return ["The instance must be a JSON object."]

    def pos_int(key):
        v = inst.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            errs.append(f"'{key}' must be a positive integer (got {v!r}).")
            return None
        return v

    days, spd = pos_int("days"), pos_int("slots_per_day")

    for key in ("rooms", "teachers", "groups", "lectures"):
        if not isinstance(inst.get(key), list) or not inst.get(key):
            errs.append(f"'{key}' must be a non-empty list.")
    if errs:
        return errs

    def ids_of(key, required_fields):
        seen = {}
        for i, item in enumerate(inst[key]):
            if not isinstance(item, dict):
                errs.append(f"{key}[{i}] must be an object.")
                continue
            for f in required_fields:
                if f not in item:
                    errs.append(f"{key}[{i}] is missing '{f}'.")
            if "id" in item:
                if item["id"] in seen:
                    errs.append(f"Duplicate id {item['id']!r} in '{key}'.")
                seen[item["id"]] = item
        return seen

    rooms = ids_of("rooms", ["id", "capacity"])
    teachers = ids_of("teachers", ["id"])
    groups = ids_of("groups", ["id", "size"])
    lectures = ids_of("lectures", ["id", "teacher_id", "group_id", "sessions_per_week"])
    if errs:
        return errs

    for r in rooms.values():
        if not isinstance(r["capacity"], (int, float)) or isinstance(r["capacity"], bool) or r["capacity"] <= 0:
            errs.append(f"Room {r['id']!r}: capacity must be a positive number.")
    for g in groups.values():
        if not isinstance(g["size"], (int, float)) or isinstance(g["size"], bool) or g["size"] <= 0:
            errs.append(f"Group {g['id']!r}: size must be a positive number.")
    if errs:
        return errs

    T = days * spd
    unavailable = {}
    for t in teachers.values():
        slots = t.get("unavailable_slots", [])
        if not isinstance(slots, list):
            errs.append(f"Teacher {t['id']!r}: 'unavailable_slots' must be a list.")
            continue
        idx = set()
        for item in slots:
            k = _slot_index(item, days, spd)
            if k is None:
                errs.append(f"Teacher {t['id']!r}: unavailable slot {item!r} is outside the {days}x{spd} grid.")
            else:
                idx.add(k)
        unavailable[t["id"]] = idx

    max_cap = max(r["capacity"] for r in rooms.values())
    group_load = {g: 0 for g in groups}
    teacher_load = {t: 0 for t in teachers}
    for lec in lectures.values():
        n = lec["sessions_per_week"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            errs.append(f"Lecture {lec['id']!r}: sessions_per_week must be an integer >= 1 (got {n!r}).")
            continue
        if lec["teacher_id"] not in teachers:
            errs.append(f"Lecture {lec['id']!r}: unknown teacher_id {lec['teacher_id']!r}.")
        else:
            teacher_load[lec["teacher_id"]] += n
        if lec["group_id"] not in groups:
            errs.append(f"Lecture {lec['id']!r}: unknown group_id {lec['group_id']!r}.")
        else:
            group_load[lec["group_id"]] += n
    for g in groups.values():
        if g["size"] > max_cap:
            errs.append(f"Group {g['id']!r} has {g['size']} students but the largest room holds only {max_cap}.")
        if group_load[g["id"]] > T:
            errs.append(f"Group {g['id']!r} needs {group_load[g['id']]} sessions but the week has only {T} slots.")
    for tid, load in teacher_load.items():
        free = T - len(unavailable.get(tid, ()))
        if load > free:
            errs.append(f"Teacher {tid!r} must teach {load} sessions but is available in only {free} slots.")
    total = sum(group_load.values())
    if total > T * len(rooms):
        errs.append(f"{total} sessions do not fit into {len(rooms)} rooms x {T} slots = {T * len(rooms)} room-slots.")

    w = inst.get("weights", {})
    if w is not None and not isinstance(w, dict):
        errs.append("'weights' must be an object.")
    elif w:
        for k, v in w.items():
            if k not in OBJECTIVES:
                errs.append(f"Unknown weight {k!r}; expected some of {OBJECTIVES}.")
            elif not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                errs.append(f"Weight {k!r} must be a number >= 0.")
    return errs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _gaps(busy: np.ndarray) -> np.ndarray:
    """Idle slots between first and last busy slot of each row. busy: bool array [..., P].
    gap = (last - first + 1) - number_of_busy_slots, and 0 for an empty day."""
    P = busy.shape[-1]
    any_busy = busy.any(axis=-1)
    first = busy.argmax(axis=-1)
    last = P - 1 - busy[..., ::-1].argmax(axis=-1)
    return np.where(any_busy, (last - first + 1) - busy.sum(axis=-1), 0)


class TimetableProblem:
    """Indexed form of a validated instance. `objectives` selects which of f1..f4 are optimised."""

    def __init__(self, inst: dict, objectives: list[str] | None = None):
        errs = validate_instance(inst)
        if errs:
            raise ValueError("Invalid instance: " + "; ".join(errs))
        self.inst = inst
        self.objectives = list(objectives or OBJECTIVES)
        unknown = [o for o in self.objectives if o not in OBJECTIVES]
        if unknown:
            raise ValueError(f"Unknown objective(s) {unknown}; choose from {OBJECTIVES}")
        self.obj_idx = np.array([OBJECTIVES.index(o) for o in self.objectives])
        self.M = len(self.objectives)

        self.D, self.P = inst["days"], inst["slots_per_day"]
        self.T = self.D * self.P
        self.room_ids = [r["id"] for r in inst["rooms"]]
        self.cap = np.array([r["capacity"] for r in inst["rooms"]], dtype=float)
        self.R = len(self.room_ids)
        self.teacher_ids = [t["id"] for t in inst["teachers"]]
        self.NT = len(self.teacher_ids)
        self.unavail = np.zeros((self.NT, self.T), dtype=bool)
        for i, t in enumerate(inst["teachers"]):
            for item in t.get("unavailable_slots", []):
                self.unavail[i, _slot_index(item, self.D, self.P)] = True
        self.group_ids = [g["id"] for g in inst["groups"]]
        self.size = np.array([g["size"] for g in inst["groups"]], dtype=float)
        self.NG = len(self.group_ids)
        self.lecture_ids = [l["id"] for l in inst["lectures"]]
        self.L = len(self.lecture_ids)

        t_index = {t: i for i, t in enumerate(self.teacher_ids)}
        g_index = {g: i for i, g in enumerate(self.group_ids)}
        lec, tea, grp, num = [], [], [], []
        self.required = np.zeros(self.L, dtype=int)
        for li, l in enumerate(inst["lectures"]):
            self.required[li] = l["sessions_per_week"]
            for k in range(l["sessions_per_week"]):
                lec.append(li)
                tea.append(t_index[l["teacher_id"]])
                grp.append(g_index[l["group_id"]])
                num.append(k)
        self.sess_lecture = np.array(lec)  # lecture of every session
        self.sess_teacher = np.array(tea)
        self.sess_group = np.array(grp)
        self.sess_number = np.array(num)  # 0,1,2.. within its lecture
        self.sess_size = self.size[self.sess_group]
        self.S = len(lec)
        self.n_var = 2 * self.S
        # rooms each session fits in (used by repair and by the capacity-aware room mutation)
        self.fitting_rooms = [np.flatnonzero(self.cap >= self.sess_size[s]) for s in range(self.S)]

    # ---- encoding helpers ----
    def split(self, x):
        return x[: self.S], x[self.S :]

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        """Uniform random slot and uniform random room for every session (usually infeasible)."""
        return np.concatenate([rng.integers(0, self.T, self.S), rng.integers(0, self.R, self.S)])

    # ---- evaluation ----
    def _counts(self, slots, rooms):
        T = self.T
        tc = np.bincount(self.sess_teacher * T + slots, minlength=self.NT * T).reshape(self.NT, T)
        gc = np.bincount(self.sess_group * T + slots, minlength=self.NG * T).reshape(self.NG, T)
        rc = np.bincount(rooms * T + slots, minlength=self.R * T).reshape(self.R, T)
        return tc, gc, rc

    def evaluate_full(self, x) -> tuple[np.ndarray, float, dict]:
        """All four objectives, total violation V and the per-constraint breakdown."""
        x = np.asarray(x)
        slots, rooms = self.split(x)
        out_of_range = int(((slots < 0) | (slots >= self.T)).sum() + ((rooms < 0) | (rooms >= self.R)).sum())
        slots = np.clip(slots, 0, self.T - 1)
        rooms = np.clip(rooms, 0, self.R - 1)
        tc, gc, rc = self._counts(slots, rooms)

        h1 = np.maximum(tc - 1, 0).sum() + self.unavail[self.sess_teacher, slots].sum()
        h2 = np.maximum(gc - 1, 0).sum()
        h3 = np.maximum(rc - 1, 0).sum()
        h4 = (self.cap[rooms] < self.sess_size).sum()
        placed = np.bincount(self.sess_lecture, minlength=self.L)
        h5 = np.abs(placed - self.required).sum() + out_of_range
        breakdown = {"H1_teacher": int(h1), "H2_group": int(h2), "H3_room": int(h3), "H4_capacity": int(h4), "H5_sessions": int(h5)}

        g_busy = gc.reshape(self.NG, self.D, self.P) > 0
        t_load = tc.reshape(self.NT, self.D, self.P)
        f1 = _gaps(g_busy).sum()
        f2 = _gaps(t_load > 0).sum() + (t_load.sum(axis=2) > OVERLOAD_LIMIT).sum()
        f3 = np.maximum(self.cap[rooms] - self.sess_size, 0).sum()
        period = slots % self.P
        f4 = ((period == 0) | (period == self.P - 1)).sum()
        F = np.array([f1, f2, f3, f4], dtype=float)
        return F, float(sum(breakdown.values())), breakdown

    def evaluate(self, x) -> tuple[np.ndarray, float]:
        """Selected objectives and total violation of one solution."""
        F, V, _ = self.evaluate_full(x)
        return F[self.obj_idx], V

    def evaluate_population(self, X) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(X)
        F = np.empty((len(X), self.M))
        V = np.empty(len(X))
        for i, x in enumerate(X):
            F[i], V[i] = self.evaluate(x)
        return F, V

    # ---- output ----
    def to_timetable(self, x) -> list[dict]:
        slots, rooms = self.split(np.asarray(x))
        return [
            {
                "lecture_id": self.lecture_ids[self.sess_lecture[s]],
                "session": int(self.sess_number[s]),
                "day": int(slots[s] // self.P),
                "slot": int(slots[s] % self.P),
                "room_id": self.room_ids[rooms[s]],
            }
            for s in range(self.S)
        ]

    # ---- repair ----
    def repair(self, x, rng: np.random.Generator, passes: int = 3) -> np.ndarray:
        """Run up to `passes` greedy repair passes, stopping early once nothing is violated.
        A later pass can succeed where an earlier one got stuck, because the other sessions have moved."""
        x = self._repair_pass(x, rng)
        for _ in range(passes - 1):
            if self.evaluate_full(x)[1] == 0:
                break
            x = self._repair_pass(x, rng)
        return x

    def _repair_pass(self, x, rng: np.random.Generator) -> np.ndarray:
        """One greedy repair pass over H1-H4.

        1. Count, for every teacher / group / room and slot, how many sessions sit there.
        2. A session is in trouble if any of its cells is over-booked, its teacher is unavailable
           or its room is too small.
        3. Visit troubled sessions in random order. Take the session out of the counts; if its spot is
           still bad, move it to a slot where its teacher is free and available and its group is free,
           choosing the slot closest to the old one (same day first) and the smallest free room that
           fits. If no such place exists the session is left where it is (the violation stays visible).
        """
        x = np.array(x, copy=True)
        slots, rooms = x[: self.S], x[self.S :]  # views: edits change x
        tc, gc, rc = self._counts(slots, rooms)
        te, gr = self.sess_teacher, self.sess_group
        bad = (
            (tc[te, slots] > 1)
            | self.unavail[te, slots]
            | (gc[gr, slots] > 1)
            | (rc[rooms, slots] > 1)
            | (self.cap[rooms] < self.sess_size)
        )
        for s in rng.permutation(np.flatnonzero(bad)):
            t0, r0, tt, g = slots[s], rooms[s], te[s], gr[s]
            tc[tt, t0] -= 1
            gc[g, t0] -= 1
            rc[r0, t0] -= 1
            ok_here = tc[tt, t0] == 0 and not self.unavail[tt, t0] and gc[g, t0] == 0 and rc[r0, t0] == 0 and self.cap[r0] >= self.sess_size[s]
            if not ok_here:
                fits = self.fitting_rooms[s]
                slot_ok = (tc[tt] == 0) & ~self.unavail[tt] & (gc[g] == 0)
                room_free = rc[fits] == 0  # (rooms that fit, T)
                cand = np.flatnonzero(slot_ok & room_free.any(axis=0))
                if len(cand):
                    # distance: other days cost a whole day, then the period difference; random tie-break
                    dist = np.abs(cand // self.P - t0 // self.P) * self.P + np.abs(cand % self.P - t0 % self.P)
                    dist = dist + rng.random(len(cand)) * 0.5
                    t_new = int(cand[np.argmin(dist)])
                    free_rooms = fits[room_free[:, t_new]]
                    r_new = int(free_rooms[np.argmin(self.cap[free_rooms])])
                    slots[s], rooms[s] = t_new, r_new
                elif self.cap[r0] < self.sess_size[s]:
                    # no free slot at all: at least fix the capacity if a fitting room is free here
                    free_here = fits[rc[fits, t0] == 0]
                    if len(free_here):
                        rooms[s] = int(free_here[np.argmin(self.cap[free_here])])
            tc[tt, slots[s]] += 1
            gc[g, slots[s]] += 1
            rc[rooms[s], slots[s]] += 1
        return x
