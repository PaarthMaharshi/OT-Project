"""Problem definition: input entities (proposal Section 3) and an indexed form used by the solvers.

A timetable (chromosome) is three integer arrays:
    slot[e]  start time slot of event e (t = day * P + period)
    room[e]  room index of event e
    fac[k]   faculty index teaching offering k (same teacher for every session)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Group:
    id: str
    name: str
    strength: int


@dataclass
class Faculty:
    id: str
    name: str
    max_weekly: int
    max_daily: int
    unavailable: list[int] = field(default_factory=list)  # blocked slot indices


@dataclass
class Room:
    id: str
    name: str
    capacity: int
    type: str  # "classroom" | "lab"
    unavailable: list[int] = field(default_factory=list)


@dataclass
class Offering:
    id: str
    subject: str
    code: str
    group_ids: list[str]
    type: str  # "theory" | "lab" | "tutorial"
    sessions_per_week: int
    eligible_faculty: list[str]


@dataclass
class Event:
    id: str
    offering_id: str
    length: int
    headcount: int


@dataclass
class TimeGrid:
    days: list[str]
    periods: list[str]
    lunch_period: int


@dataclass
class Instance:
    name: str
    grid: TimeGrid
    groups: list[Group]
    faculty: list[Faculty]
    rooms: list[Room]
    offerings: list[Offering]
    events: list[Event]

    # ---- serialisation (camelCase JSON matches frontend/src/types.ts) ----
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "grid": {"days": self.grid.days, "periods": self.grid.periods, "lunchPeriod": self.grid.lunch_period},
            "groups": [asdict(g) for g in self.groups],
            "faculty": [
                {"id": f.id, "name": f.name, "maxWeekly": f.max_weekly, "maxDaily": f.max_daily, "unavailable": f.unavailable}
                for f in self.faculty
            ],
            "rooms": [asdict(r) for r in self.rooms],
            "offerings": [
                {
                    "id": o.id,
                    "subject": o.subject,
                    "code": o.code,
                    "groupIds": o.group_ids,
                    "type": o.type,
                    "sessionsPerWeek": o.sessions_per_week,
                    "eligibleFaculty": o.eligible_faculty,
                }
                for o in self.offerings
            ],
            "events": [{"id": e.id, "offeringId": e.offering_id, "length": e.length, "headcount": e.headcount} for e in self.events],
        }

    @staticmethod
    def from_dict(d: dict) -> "Instance":
        g = d["grid"]
        return Instance(
            name=d["name"],
            grid=TimeGrid(g["days"], g["periods"], g["lunchPeriod"]),
            groups=[Group(**x) for x in d["groups"]],
            faculty=[Faculty(x["id"], x["name"], x["maxWeekly"], x["maxDaily"], x.get("unavailable", [])) for x in d["faculty"]],
            rooms=[Room(x["id"], x["name"], x["capacity"], x["type"], x.get("unavailable", [])) for x in d["rooms"]],
            offerings=[
                Offering(x["id"], x["subject"], x["code"], x["groupIds"], x["type"], x["sessionsPerWeek"], x["eligibleFaculty"])
                for x in d["offerings"]
            ],
            events=[Event(x["id"], x["offeringId"], x["length"], x["headcount"]) for x in d["events"]],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: str | Path) -> "Instance":
        return Instance.from_dict(json.loads(Path(path).read_text()))


class Problem:
    """Integer-indexed view of an Instance with everything the solvers need precomputed."""

    def __init__(self, inst: Instance):
        self.inst = inst
        self.D = len(inst.grid.days)
        self.P = len(inst.grid.periods)
        self.T = self.D * self.P
        self.lunch = inst.grid.lunch_period
        self.E, self.K = len(inst.events), len(inst.offerings)
        self.G, self.F, self.R = len(inst.groups), len(inst.faculty), len(inst.rooms)

        gi = {g.id: i for i, g in enumerate(inst.groups)}
        fi = {f.id: i for i, f in enumerate(inst.faculty)}
        ki = {o.id: i for i, o in enumerate(inst.offerings)}

        self.ev_len = np.array([e.length for e in inst.events], dtype=np.int64)
        self.ev_head = np.array([e.headcount for e in inst.events], dtype=np.float64)
        self.ev_off = np.array([ki[e.offering_id] for e in inst.events], dtype=np.int64)
        self.ev_theory = np.array([inst.offerings[k].type != "lab" for k in self.ev_off])
        self.ev_groups = [[gi[g] for g in inst.offerings[k].group_ids] for k in self.ev_off]

        self.off_events = [np.flatnonzero(self.ev_off == k) for k in range(self.K)]
        self.off_elig = [np.array([fi[f] for f in o.eligible_faculty], dtype=np.int64) for o in inst.offerings]
        self.off_need = np.array([self.ev_len[ev].sum() for ev in self.off_events], dtype=np.int64)

        self.fac_max = np.array([f.max_weekly for f in inst.faculty], dtype=np.float64)
        self.fac_daily = np.array([f.max_daily for f in inst.faculty], dtype=np.int64)
        self.fac_block = np.zeros((self.F, self.T), dtype=bool)
        for i, f in enumerate(inst.faculty):
            self.fac_block[i, f.unavailable] = True
        self.room_cap = np.array([r.capacity for r in inst.rooms], dtype=np.float64)
        self.room_block = np.zeros((self.R, self.T), dtype=bool)
        for i, r in enumerate(inst.rooms):
            self.room_block[i, r.unavailable] = True

        # C5/C6 are enforced through the domains: only valid starts and fitting rooms are ever sampled.
        self.valid_starts: list[np.ndarray] = []
        self.valid_rooms: list[np.ndarray] = []
        for e, ev in enumerate(inst.events):
            L = ev.length
            starts = [
                d * self.P + p
                for d in range(self.D)
                for p in range(self.P - L + 1)
                if not (p <= self.lunch < p + L)
            ]
            self.valid_starts.append(np.array(starts, dtype=np.int64))
            want = "lab" if inst.offerings[self.ev_off[e]].type == "lab" else "classroom"
            rooms = [r for r, rm in enumerate(inst.rooms) if (rm.type == want or (want == "classroom" and rm.type == "hall")) and rm.capacity >= ev.headcount]
            if not rooms:
                raise ValueError(f"No room fits event {ev.id} ({want}, {ev.headcount} students)")
            self.valid_rooms.append(np.array(rooms, dtype=np.int64))

        self.n_starts = np.array([len(s) for s in self.valid_starts], dtype=np.int64)
        self.cells = [s[:, None] + np.arange(L) for s, L in zip(self.valid_starts, self.ev_len)]
        self.same_len = [np.flatnonzero(self.ev_len == L) for L in self.ev_len]  # swap partners
        self.off_group = np.array([gi[o.group_ids[0]] for o in inst.offerings], dtype=np.int64)

        # Flattened (event, cell-offset) and (event, group, cell-offset) arrays for vectorised occupancy.
        self.rep_e = np.repeat(np.arange(self.E), self.ev_len)
        self.rep_off = np.concatenate([np.arange(L) for L in self.ev_len]) if self.E else np.zeros(0, dtype=np.int64)
        ge, gg, go = [], [], []
        for e in range(self.E):
            for g in self.ev_groups[e]:
                for o in range(self.ev_len[e]):
                    ge.append(e), gg.append(g), go.append(o)
        self.grp_e, self.grp_g, self.grp_off = np.array(ge), np.array(gg), np.array(go)

    def day(self, t):
        return t // self.P

    def period(self, t):
        return t % self.P


@dataclass
class Solution:
    slot: np.ndarray
    room: np.ndarray
    fac: np.ndarray
    F: np.ndarray | None = None  # objectives (f1, f2, f3)
    V: float = 0.0  # constraint violation

    def copy(self) -> "Solution":
        return Solution(self.slot.copy(), self.room.copy(), self.fac.copy(), None if self.F is None else self.F.copy(), self.V)

    def to_dict(self, prob: Problem, sid: str, on_front: bool) -> dict:
        inst = prob.inst
        return {
            "id": sid,
            "objectives": {"f1": float(self.F[0]), "f2": float(self.F[1]), "f3": float(self.F[2])},
            "violation": float(self.V),
            "assignments": [
                {"eventId": ev.id, "day": int(self.slot[e] // prob.P), "period": int(self.slot[e] % prob.P), "roomId": inst.rooms[self.room[e]].id}
                for e, ev in enumerate(inst.events)
            ],
            "facultyOf": {o.id: inst.faculty[self.fac[k]].id for k, o in enumerate(inst.offerings)},
            "onFront": on_front,
        }
