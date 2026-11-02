"""Synthetic instance generator (proposal Section 9). All data here is SYNTHETIC.

Scenarios follow Section 13: small (~40 events), medium (~150), large (~400).
`demo` reproduces the frontend's built-in mock instance.
"""
from __future__ import annotations

import numpy as np

from .problem import Event, Faculty, Group, Instance, Offering, Room, TimeGrid

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
PERIODS = ["9:00 - 10:00", "10:00 - 11:00", "11:00 - 12:00", "12:00 - 1:00", "1:00 - 2:00", "2:00 - 3:00", "3:00 - 4:00", "4:00 - 5:00"]
LUNCH = 4

SCENARIOS = {
    "small": {"groups": 2, "rooms": 5, "faculty": 6, "label": "Small (2 groups)"},
    "medium": {"groups": 8, "rooms": 12, "faculty": 20, "label": "Medium (one department)"},
    "large": {"groups": 20, "rooms": 25, "faculty": 50, "label": "Large (institute)"},
}

# (code, subject, type, sessions/week)
SUBJECT_POOL = [
    ("DSA", "Data Structures", "theory", 3),
    ("DBMS", "Database Systems", "theory", 3),
    ("OS", "Operating Systems", "theory", 3),
    ("MATHS", "Engineering Maths III", "theory", 3),
    ("COA", "Computer Organisation", "theory", 2),
    ("CN", "Computer Networks", "theory", 3),
    ("TOC", "Theory of Computation", "theory", 2),
    ("SE", "Software Engineering", "theory", 2),
    ("AI", "Artificial Intelligence", "theory", 3),
    ("EVS", "Environmental Studies", "theory", 1),
    ("DSA LAB", "Data Structures Lab", "lab", 1),
    ("DBMS LAB", "Database Lab", "lab", 1),
    ("OS LAB", "Operating Systems Lab", "lab", 1),
    ("CN LAB", "Networks Lab", "lab", 1),
]

SURNAMES = [
    "Mehta", "Iyer", "Kulkarni", "Fernandes", "Rao", "Shaikh", "Joshi", "Patil", "Nair", "Desai", "Banerjee", "Kapoor",
    "Menon", "Gupta", "Pillai", "Reddy", "Sharma", "Bhat", "Chopra", "Dutta", "Ghosh", "Hegde", "Jain", "Kamat",
    "Lobo", "Mishra", "Naik", "Pandey", "Qureshi", "Rathod", "Saxena", "Thakur", "Upadhyay", "Verma", "Wagh",
    "Yadav", "Zaveri", "Agarwal", "Bose", "Chauhan", "D'Souza", "Gokhale", "Kale", "More", "Pawar", "Sawant",
    "Shetty", "Tiwari", "Vaidya", "Apte",
]


def demo_instance() -> Instance:
    """Same data as frontend/src/mock/instance.ts."""
    subjects = [
        ("DSA", "Data Structures", "theory", 3, ["f1", "f2"]),
        ("DBMS", "Database Systems", "theory", 3, ["f2", "f3"]),
        ("OS", "Operating Systems", "theory", 3, ["f3", "f4"]),
        ("MATHS", "Engineering Maths III", "theory", 3, ["f5", "f6"]),
        ("COA", "Computer Organisation", "theory", 2, ["f4", "f1"]),
        ("EVS", "Environmental Studies", "theory", 1, ["f6", "f5"]),
        ("DSA LAB", "Data Structures Lab", "lab", 1, ["f1", "f2"]),
        ("DBMS LAB", "Database Lab", "lab", 1, ["f3", "f2"]),
    ]
    groups = [Group("gA", "SY-Comp Div A", 58), Group("gB", "SY-Comp Div B", 52)]
    offerings, events = [], []
    for g in groups:
        for code, name, typ, n, fac in subjects:
            oid = f"{g.id}-{code.replace(' ', '')}"
            offerings.append(Offering(oid, name, code, [g.id], typ, n, fac))
            for i in range(n):
                events.append(Event(f"{oid}-{i + 1}", oid, 2 if typ == "lab" else 1, g.strength))
    return Instance(
        name="SY Computer Engineering - Semester III",
        grid=TimeGrid(DAYS, PERIODS, LUNCH),
        groups=groups,
        faculty=[
            Faculty("f1", "Prof. Mehta", 12, 4),
            Faculty("f2", "Dr. Iyer", 12, 4),
            Faculty("f3", "Prof. Kulkarni", 10, 4),
            Faculty("f4", "Dr. Fernandes", 10, 3),
            Faculty("f5", "Prof. Rao", 8, 3),
            Faculty("f6", "Dr. Shaikh", 8, 3),
        ],
        rooms=[
            Room("r1", "CR-101", 60, "classroom"),
            Room("r2", "CR-102", 72, "classroom"),
            Room("r3", "LH-201", 120, "classroom"),
            Room("r4", "Lab A-301", 64, "lab"),
            Room("r5", "Lab B-302", 70, "lab"),
        ],
        offerings=offerings,
        events=events,
    )


def generate(scenario: str = "medium", seed: int = 0, faculty_blocked_slots: int = 2) -> Instance:
    """Parameterised synthetic instance on a typical engineering-college structure."""
    if scenario == "demo":
        return demo_instance()
    cfg = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    T = len(DAYS) * len(PERIODS)
    nG, nR, nF = cfg["groups"], cfg["rooms"], cfg["faculty"]

    years = ["SY", "TY", "BE", "FY"]
    groups = [
        Group(f"g{i + 1}", f"{years[(i // 2) % 4]}-Comp Div {chr(65 + i % 2)}" + (f" ({i // 8 + 1})" if nG > 8 else ""), int(rng.integers(40, 76)))
        for i in range(nG)
    ]

    faculty_ids = [f"f{i + 1}" for i in range(nF)]
    theory = [s for s in SUBJECT_POOL if s[2] == "theory"]
    labs = [s for s in SUBJECT_POOL if s[2] == "lab"]

    offerings, events = [], []
    for g in groups:
        picked = [theory[i] for i in rng.choice(len(theory), size=7, replace=False)] + [
            labs[i] for i in rng.choice(len(labs), size=2, replace=False)
        ]
        for code, name, typ, n in picked:
            oid = f"{g.id}-{code.replace(' ', '')}"
            offerings.append(Offering(oid, name, code, [g.id], typ, n, []))
            for i in range(n):
                events.append(Event(f"{oid}-{i + 1}", oid, 2 if typ == "lab" else 1, g.strength))

    total_load = sum(e.length for e in events)
    base = int(np.ceil(total_load / nF * 1.5))

    # Each subject gets a pool of qualified faculty sized to its demand; every offering of that
    # subject may be taught by 2-3 of them.
    demand: dict[str, int] = {}
    for o in offerings:
        demand[o.code] = demand.get(o.code, 0) + o.sessions_per_week * (2 if o.type == "lab" else 1)
    pools = {
        code: list(rng.choice(faculty_ids, size=min(nF, max(3, int(np.ceil(1.6 * d / base)))), replace=False))
        for code, d in demand.items()
    }
    for o in offerings:
        pool = pools[o.code]
        o.eligible_faculty = [str(f) for f in rng.choice(pool, size=min(len(pool), int(rng.integers(2, 4))), replace=False)]
    names = list(rng.permutation(SURNAMES))
    faculty = []
    for i, fid in enumerate(faculty_ids):
        title = "Dr." if rng.random() < 0.45 else "Prof."
        blocked = sorted(int(t) for t in rng.choice(T, size=faculty_blocked_slots, replace=False)) if faculty_blocked_slots else []
        faculty.append(Faculty(fid, f"{title} {names[i % len(names)]}", int(base + rng.integers(0, 4)), 4, blocked))

    # Guarantee a load-feasible teacher assignment exists (C7): assign hardest offerings first to the
    # eligible teacher with most slack; if none has room, qualify the teacher with the most slack overall.
    slack = {f.id: f.max_weekly for f in faculty}
    for o in sorted(offerings, key=lambda o: -o.sessions_per_week * (2 if o.type == "lab" else 1)):
        need = o.sessions_per_week * (2 if o.type == "lab" else 1)
        best = max(o.eligible_faculty, key=lambda f: slack[f])
        if slack[best] < need:
            best = max(slack, key=lambda f: slack[f])
            o.eligible_faculty.append(best)
        slack[best] -= need

    n_labs = sum(1 for o in offerings if o.type == "lab")
    n_lab_rooms = max(2, int(np.ceil(n_labs * 2 / 22)))
    n_class = nR - n_lab_rooms
    caps = [60, 72, 90, 120]
    rooms = []
    for i in range(n_class):
        cap = caps[i % len(caps)] if i >= 2 else 80  # at least two rooms that fit any group
        rooms.append(Room(f"r{i + 1}", f"CR-{101 + i}" if cap < 120 else f"LH-{201 + i}", cap, "hall" if cap >= 120 else "classroom"))
    for j in range(n_lab_rooms):
        rooms.append(Room(f"r{n_class + j + 1}", f"Lab {chr(65 + j)}-30{j + 1}", int(rng.choice([64, 70, 80])), "lab"))
    # Every group must fit at least one lab.
    biggest = max(g.strength for g in groups)
    rooms[-1].capacity = max(rooms[-1].capacity, biggest)

    return Instance(
        name=f"Synthetic {scenario} instance (seed {seed})",
        grid=TimeGrid(DAYS, PERIODS, LUNCH),
        groups=groups,
        faculty=faculty,
        rooms=rooms,
        offerings=offerings,
        events=events,
    )
