"""Registry: algorithm name (as used in the input JSON) -> module.

Every module exposes NAME, DEFAULTS, initialize, evaluate, step, output and run.
"""
from . import (
    cnsga2,
    eps_constraint,
    goal_programming,
    lexicographic,
    moead,
    moma,
    mosa,
    nsga2,
    nsga3,
    spea2,
    weighted_sum,
)

REGISTRY = {
    "nsga2": nsga2,  # 14
    "constrained_nsga2": cnsga2,  # 21 (default, main solver)
    "spea2": spea2,  # 16
    "moead": moead,  # 18
    "mosa": mosa,  # 26
    "moma": moma,  # 30
    "nsga3": nsga3,  # 15
    "weighted_sum": weighted_sum,  # 1
    "eps_constraint": eps_constraint,  # 2
    "goal_programming": goal_programming,  # 6
    "lexicographic": lexicographic,  # 7
}
DEFAULT_ALGORITHM = "constrained_nsga2"
