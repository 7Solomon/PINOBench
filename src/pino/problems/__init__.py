"""Benchmark problems.

Every module in this package is imported here so that its ``@register_problem`` decorators
run. A new benchmark therefore needs no central list of problems.
"""

import importlib
import pkgutil

from pino.problems.base import Problem
from pino.problems.registry import available_problems, build_problem, problem_class, register_problem

for _module in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module.name}")

__all__ = ["Problem", "available_problems", "build_problem", "problem_class", "register_problem"]
