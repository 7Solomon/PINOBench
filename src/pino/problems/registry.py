"""Decorator-based mapping from benchmark names to problem classes."""

from __future__ import annotations

from collections.abc import Callable

from pino.problems.base import Problem

_REGISTRY: dict[str, type[Problem]] = {}


def register_problem(name: str) -> Callable[[type[Problem]], type[Problem]]:
    """Register a problem class under ``name``, the value of ``problem.name`` in the config."""

    def decorator(cls: type[Problem]) -> type[Problem]:
        if name in _REGISTRY:
            raise ValueError(f"Problem '{name}' is already registered.")
        if not isinstance(getattr(cls, "WRAPPER_VERSION", None), str):
            raise TypeError(f"Problem '{name}' must define the string constant WRAPPER_VERSION.")
        inputs = getattr(cls, "SUPPORTED_INPUTS", None)
        if not (isinstance(inputs, tuple) and inputs and all(isinstance(item, str) for item in inputs)):
            raise TypeError(f"Problem '{name}' must define SUPPORTED_INPUTS, a non-empty tuple of names.")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_problems() -> list[str]:
    return sorted(_REGISTRY)


def problem_class(name: str) -> type[Problem]:
    """Return the registered class of ``name`` without instantiating it."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown problem '{name}'. Available: {available_problems()}")
    return _REGISTRY[name]


def build_problem(cfg) -> Problem:
    """Instantiate the problem selected by ``cfg.name`` and configure its variants."""
    problem = problem_class(cfg.name)(cfg)
    problem.configure_variants(cfg)
    return problem
