"""Training losses: ``physics``, ``data`` and their weighted sum ``hybrid``."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from pino.problems.base import Problem

MODES = ("physics", "data", "hybrid")


def uses_physics(mode: str) -> bool:
    return mode in ("physics", "hybrid")


def uses_data(mode: str) -> bool:
    return mode in ("data", "hybrid")


def compute_loss(
    mode: str,
    weights,
    problem: Problem,
    prediction: Tensor,
    inputs: Tensor,
    grid: Tensor,
    target: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Return the total loss and its unweighted components for logging."""
    if mode not in MODES:
        raise ValueError(f"Unknown training mode '{mode}'. Available: {MODES}")

    total = torch.zeros((), device=prediction.device)
    components: dict[str, float] = {}
    if uses_physics(mode):
        physics = problem.residual(prediction, inputs, grid).pow(2).mean()
        total = total + float(weights.physics) * physics
        components["physics"] = physics.item()
    if uses_data(mode):
        scale = problem.data_scale(inputs, grid)
        difference = prediction - target if scale is None else (prediction - target) / scale
        data = difference.pow(2).mean()
        total = total + float(weights.data) * data
        components["data"] = data.item()
    components["total"] = total.item()
    return total, components


def overridden_keys(overrides: Iterable[str]) -> set[str]:
    """Return the config keys set on the command line, e.g. ``{"training.weights.data"}``.

    Accepts Hydra override strings such as ``training.weights.data=300`` or ``++seed=1``.
    """
    return {override.split("=", 1)[0].lstrip("+~") for override in overrides}


def resolve_loss_weights(training_cfg, problem_cfg, overridden: Iterable[str] = ()) -> dict[str, float]:
    """Return the loss weights of a run.

    Order of precedence, highest first:

    1. ``training.weights.<term>`` set on the command line,
    2. the optional ``loss_weights`` of the problem config,
    3. ``weights`` of the training config.

    ``loss_weights`` only acts in the ``hybrid`` mode. ``physics`` and ``data`` have a single
    term whose weight is fixed by the mode, so a problem cannot rescale them.
    """
    overridden = set(overridden)
    weights = {name: float(value) for name, value in training_cfg.weights.items()}
    problem_weights = problem_cfg.get("loss_weights")
    if training_cfg.mode != "hybrid" or not problem_weights:
        return weights
    for name, value in problem_weights.items():
        if name not in weights:
            raise KeyError(f"Unknown loss term '{name}' in loss_weights. Available: {sorted(weights)}")
        if "training.weights" not in overridden and f"training.weights.{name}" not in overridden:
            weights[name] = float(value)
    return weights
