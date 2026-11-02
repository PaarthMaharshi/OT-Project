"""Translate the website's instance format (frontend/src/types.ts) into the engine format.

Website format (camelCase):
    {"name", "grid": {"days": [names], "periods": [labels], "lunchPeriod": i},
     "groups": [{"id", "name", "strength"}], "faculty": [{"id", "name", "maxWeekly", "maxDaily", "unavailable"?}],
     "rooms": [{"id", "name", "capacity", "type"}],
     "offerings": [{"id", "subject", "code", "groupIds", "type", "sessionsPerWeek", "eligibleFaculty"}],
     "events": [...]}
It may be sent on its own or wrapped as {"instance": {...}, "algorithm", "params", "weights"}.

Mapping (the lab model is simpler than the website's, so some features are reduced):
    days            = number of day names
    slots_per_day   = number of periods minus the lunch period (lunch is never scheduled)
    groups.size     = strength
    teachers        = faculty; "unavailable" (website slot index day*periods+period) is mapped onto the
                      engine grid, lunch entries dropped
    lectures        = offerings: teacher_id = FIRST eligible teacher (the lab model fixes the teacher),
                      group_id = FIRST group id, sessions = sessionsPerWeek, times 2 for labs (a 2-period
                      lab becomes two 1-slot sessions; they are NOT forced to be consecutive)
Each timetable entry the engine returns is given the website's "period" index (lunch re-inserted) and
the day/period labels, so the website can place it on its own grid.
"""
from __future__ import annotations


def is_frontend(d) -> bool:
    if not isinstance(d, dict):
        return False
    inner = d.get("instance", d)
    return isinstance(inner, dict) and "grid" in inner and "offerings" in inner


def from_frontend(d: dict) -> tuple[dict, dict, list[str]]:
    """Returns (engine_instance, mapping, warnings)."""
    fi = d.get("instance", d)
    warnings: list[str] = []
    grid = fi["grid"]
    day_names = list(grid["days"])
    periods = list(grid["periods"])
    lunch = grid.get("lunchPeriod")
    teach_periods = [i for i in range(len(periods)) if i != lunch]
    spd = len(teach_periods)
    to_engine_slot = {}
    for day in range(len(day_names)):
        for k, per in enumerate(teach_periods):
            to_engine_slot[day * len(periods) + per] = day * spd + k

    teachers = []
    for f in fi.get("faculty", []):
        un = [to_engine_slot[s] for s in f.get("unavailable", []) if s in to_engine_slot]
        teachers.append({"id": f["id"], "name": f.get("name", f["id"]), "unavailable_slots": un})
    lectures = []
    fixed, shared = [], []
    for o in fi.get("offerings", []):
        elig = o.get("eligibleFaculty") or [None]
        gids = o.get("groupIds") or [None]
        if len(elig) > 1:
            fixed.append(o["id"])
        if len(gids) > 1:
            shared.append(o["id"])
        n = int(o.get("sessionsPerWeek", 1)) * (2 if o.get("type") == "lab" else 1)
        lectures.append({"id": o["id"], "course": o.get("subject", o["id"]), "teacher_id": elig[0], "group_id": gids[0], "sessions_per_week": n})
    if fixed:
        warnings.append(f"{len(fixed)} offerings list several eligible teachers; the first one is used (lab model: fixed teacher).")
    if shared:
        warnings.append(f"{len(shared)} offerings are shared by several groups; only the first group is scheduled: {shared}.")
    if any(o.get("type") == "lab" for o in fi.get("offerings", [])):
        warnings.append("Labs are scheduled as separate 1-period sessions; consecutive periods are not enforced.")

    inst = {
        "name": fi.get("name", "website instance"),
        "days": len(day_names),
        "slots_per_day": spd,
        "rooms": [{"id": r["id"], "capacity": r["capacity"]} for r in fi.get("rooms", [])],
        "teachers": teachers,
        "groups": [{"id": g["id"], "size": g.get("strength", g.get("size"))} for g in fi.get("groups", [])],
        "lectures": lectures,
    }
    for key in ("weights", "algorithm", "params"):
        if key in d:
            inst[key] = d[key]
    mapping = {"teach_periods": teach_periods, "day_names": day_names, "period_labels": periods}
    return inst, mapping, warnings


def decorate(timetable: list[dict], mapping: dict) -> list[dict]:
    """Add the website's period index and labels to every entry."""
    tp, days, labels = mapping["teach_periods"], mapping["day_names"], mapping["period_labels"]
    for e in timetable:
        e["period"] = tp[e["slot"]]
        e["day_name"] = days[e["day"]]
        e["period_label"] = labels[e["period"]]
    return timetable
