"""Seeded synthetic instance for tests and lab experiments ONLY (the engine itself takes JSON input).

Default size (lab specification): 5 groups, 10 teachers, 8 rooms, 25 lectures, 5 days x 6 slots.
The generator guarantees a feasible timetable can exist: every group fits in a room, and no group or
teacher needs more sessions than it has available slots.

    python -m engine.generator --seed 1 --out results/lab/instance_seed1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COURSES = [
    "Optimization Techniques", "Machine Learning", "Data Structures", "Operating Systems", "Computer Networks",
    "Linear Algebra", "Probability", "Compilers", "Databases", "Computer Vision", "Cryptography", "Robotics",
    "Embedded Systems", "Graph Theory", "Numerical Methods", "Signal Processing", "Cloud Computing",
    "Information Retrieval", "Soft Computing", "Game Theory", "Parallel Computing", "Formal Methods",
    "Deep Learning", "Bioinformatics", "Research Methodology",
]


def generate(
    seed: int = 1,
    n_groups=5,
    n_teachers=10,
    n_rooms=8,
    n_lectures=25,
    days=5,
    slots_per_day=6,
    sessions=(3, 4),
    unavailable=(3, 6),
) -> dict:
    """`sessions` and `unavailable` are inclusive ranges per lecture / per teacher. The defaults
    ("lab difficulty") were chosen because with fewer sessions (2-3) and fewer blocked slots (1-3)
    student gaps and teacher gaps stop conflicting: the search reaches the ideal point (0, 0) and the
    bi-objective front collapses to one timetable (measured, see README)."""
    rng = np.random.default_rng(seed)
    T = days * slots_per_day
    groups = [{"id": f"G{i + 1}", "size": int(rng.integers(25, 71))} for i in range(n_groups)]
    capacities = sorted(int(c) for c in rng.choice([30, 40, 50, 60, 75, 90, 120], size=n_rooms))
    capacities[-1] = max(capacities[-1], max(g["size"] for g in groups))  # every group fits somewhere
    rooms = [{"id": f"R{i + 1}", "capacity": c} for i, c in enumerate(capacities)]
    teachers = []
    for i in range(n_teachers):
        blocked = sorted(int(t) for t in rng.choice(T, size=int(rng.integers(unavailable[0], unavailable[1] + 1)), replace=False))
        teachers.append({"id": f"T{i + 1}", "unavailable_slots": blocked})

    lectures = []
    for i in range(n_lectures):
        lectures.append(
            {
                "id": f"L{i + 1}",
                "course": COURSES[i % len(COURSES)],
                "teacher_id": teachers[i % n_teachers]["id"] if i < n_teachers else teachers[int(rng.integers(n_teachers))]["id"],
                "group_id": groups[i % n_groups]["id"],
                "sessions_per_week": int(rng.integers(sessions[0], sessions[1] + 1)),
            }
        )
    return {
        "name": f"Synthetic lab instance (seed {seed})",
        "days": days,
        "slots_per_day": slots_per_day,
        "rooms": rooms,
        "teachers": teachers,
        "groups": groups,
        "lectures": lectures,
        "weights": {"student_gaps": 0.4, "teacher_gaps": 0.3, "room_waste": 0.2, "time_penalty": 0.1},
        "algorithm": "constrained_nsga2",
        "params": {"pop_size": 100, "evaluations": 10000, "seed": 1, "time_limit_s": 60},
    }


def describe(inst: dict) -> str:
    """Human-readable dump of every number in the instance (the lab asks to print all data)."""
    lines = [inst.get("name", "instance"), f"{inst['days']} days x {inst['slots_per_day']} slots"]
    lines.append("Rooms:    " + ", ".join(f"{r['id']}({r['capacity']})" for r in inst["rooms"]))
    lines.append("Groups:   " + ", ".join(f"{g['id']}({g['size']})" for g in inst["groups"]))
    lines.append("Teachers: " + ", ".join(f"{t['id']} unavailable {t['unavailable_slots']}" for t in inst["teachers"]))
    lines.append("Lectures:")
    for l in inst["lectures"]:
        lines.append(f"  {l['id']:4s} {l['course']:24s} teacher {l['teacher_id']:4s} group {l['group_id']}  x{l['sessions_per_week']}")
    total = sum(l["sessions_per_week"] for l in inst["lectures"])
    lines.append(f"Total sessions: {total}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    inst = generate(a.seed)
    print(describe(inst))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(inst, indent=2))
        print(f"saved {a.out}")
