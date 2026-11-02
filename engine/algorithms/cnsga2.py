"""Constrained NSGA-II (catalogue no. 21): the main solver for the hard constraints.

Same loop as NSGA-II (no. 14); the differences are only in how constraints are handled:
    * Deb's three feasibility rules replace Pareto dominance everywhere (sorting, tournament, survival):
        1. feasible beats infeasible,
        2. two infeasible: smaller total violation V wins,
        3. two feasible: Pareto dominance.
    * Every child (and every initial solution) goes through the greedy repair operator.

Lecture steps
    1. Random population, repaired, evaluated.
    2. Constrained non-dominated sort + crowding distance.
    3. Binary tournament on (constrained rank, crowding).
    4. Uniform crossover + mutation, then repair.
    5. Merge parents and children, constrained survival (fronts, then crowding).
    6. Repeat until the budget is used.

Default parameters
    pop_size 100, crossover_rate 0.9, mutation_rate 1/S, repair True, constraint_handling "deb".
"""
from __future__ import annotations

from . import nsga2
from .common import run_loop

NAME = "constrained_nsga2"
DEFAULTS = {**nsga2.DEFAULTS, "constraint_handling": "deb", "repair": True}


def initialize(problem, params, ctx):
    return nsga2.initialize(problem, params, ctx)


def evaluate(problem, X, ctx):
    return ctx.evaluate(X)


def step(problem, state, ctx):
    # nsga2.step reads constraint_handling="deb" and repair=True from the params in `state`
    return nsga2.step(problem, state, ctx)


def output(state):
    return nsga2.output(state)


def run(problem, params=None, progress=None, should_stop=None):
    import sys

    return run_loop(sys.modules[__name__], problem, params, progress, should_stop)
