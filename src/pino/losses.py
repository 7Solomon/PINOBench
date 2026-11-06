"""Training losses: ``physics``, ``data`` and their weighted sum ``hybrid``."""

from __future__ import annotations

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
