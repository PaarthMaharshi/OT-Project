"""Timetable optimisation engine (M.Tech Optimization Techniques lab).

    from engine import solve
    result = solve(instance_dict)

Pure engine: NumPy and the standard library only (Matplotlib is used by the experiment scripts only).
Transport (CLI / HTTP) lives in ../adapter.py.
"""
from .solve import solve

__all__ = ["solve"]
