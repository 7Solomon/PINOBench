"""Training losses: ``physics``, ``data`` and their weighted sum ``hybrid``.

With ``output_param = soft`` every mode adds the boundary penalty ``L_bc`` with weight
``weights.bc``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from pino.problems.base import Problem

MODES = ("physics", "data", "hybrid")
#: Loss terms whose weights are fixed by the mode; a problem's ``loss_weights`` may change them
#: only in ``hybrid``. Every other term (``bc``) can be set per problem in every mode.
MODE_TERMS = ("physics", "data")


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
    if problem.soft_constraint:
        bc = boundary_loss(problem, prediction, inputs, grid)
        total = total + float(weights["bc"]) * bc
        components["bc"] = bc.item()
    components["total"] = total.item()
    return total, components


def boundary_loss(problem: Problem, prediction: Tensor, inputs: Tensor, grid: Tensor) -> Tensor:
    """``L_bc = mean over the boundary points of ((u - g) / s)^2``.

    The points and ``g`` come from :meth:`Problem.boundary_values` (where the hard constraint
    would act), ``s`` from :meth:`Problem.boundary_scale`, the normalization of the data loss.
    """
    g, mask = problem.boundary_values(inputs, grid)
    difference = (prediction - g) / problem.boundary_scale(inputs, grid)
    return difference.pow(2)[mask.expand_as(difference)].mean()


def overridden_keys(overrides: Iterable[str]) -> set[str]:
    """Return the config keys set on the command line, e.g. ``{"training.weights.data"}``.

    Accepts Hydra override strings such as ``training.weights.data=300`` or ``++seed=1``.
    """
    return {override.split("=", 1)[0].lstrip("+~") for override in overrides}


#: Training settings a problem YAML may change in its optional ``training`` block.
PROBLEM_TRAINING_KEYS = ("epochs", "batch_size", "lr", "n_train", "n_val", "n_test", "eval_every")


def resolve_training_settings(training_cfg, problem_cfg, overridden: Iterable[str] = ()) -> dict:
    """Return the training settings of a run after the problem's optional ``training`` block.

    Order of precedence, highest first: ``training.<key>`` on the command line, the problem's
    ``training.<key>``, the training config. Used by expensive problems (2D+t) to fix their own
    epochs and batch size, so that a plain ``train.py problem=<name>`` stays within budget.
    """
    overridden = set(overridden)
    settings = {}
    for key, value in (problem_cfg.get("training") or {}).items():
        if key not in PROBLEM_TRAINING_KEYS:
            raise KeyError(f"Problem training block may set {PROBLEM_TRAINING_KEYS}, not '{key}'.")
        # ``training=physics`` on the command line selects the config group, it overrides nothing.
        if f"training.{key}" not in overridden:
            settings[key] = value
    return {key: settings.get(key, training_cfg[key]) for key in PROBLEM_TRAINING_KEYS}


def resolve_loss_weights(training_cfg, problem_cfg, overridden: Iterable[str] = ()) -> dict[str, float]:
    """Return the loss weights of a run.

    Order of precedence, highest first:

    1. ``training.weights.<term>`` set on the command line,
    2. the optional ``loss_weights`` of the problem config,
    3. ``weights`` of the training config.

    ``loss_weights`` sets ``physics`` and ``data`` only in the ``hybrid`` mode: in ``physics``
    and ``data`` these have a single term whose weight is fixed by the mode. ``bc`` (the
    boundary penalty of ``output_param = soft``) is taken in every mode.
    """
    overridden = set(overridden)
    weights = {name: float(value) for name, value in training_cfg.weights.items()}
    problem_weights = problem_cfg.get("loss_weights") or {}
    for name, value in problem_weights.items():
        if name not in weights:
            raise KeyError(f"Unknown loss term '{name}' in loss_weights. Available: {sorted(weights)}")
        if name in MODE_TERMS and training_cfg.mode != "hybrid":
            continue
        if "training.weights" not in overridden and f"training.weights.{name}" not in overridden:
            weights[name] = float(value)
    return weights
