"""Prediction with the hard constraint, error metrics and the resolution study."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from pino.data.dataset import OperatorDataset, build_dataset
from pino.numerics.constraints import apply_hard_constraint
from pino.problems.base import Problem


def predict(model: nn.Module, problem: Problem, inputs: Tensor, grid: Tensor) -> Tensor:
    """Evaluate the model on ``inputs`` and apply the problem's hard constraint."""
    g, phi = problem.boundary(inputs, grid)
    return apply_hard_constraint(model(problem.to_fno_input(inputs, grid)), g, phi)


def relative_l2(prediction: Tensor, target: Tensor) -> Tensor:
    """Return ``||u - u_hat|| / ||u||`` per example and channel, shape ``(B, C)``."""
    batch, channels = target.shape[:2]
    difference = (prediction - target).reshape(batch, channels, -1).norm(dim=-1)
    return difference / target.reshape(batch, channels, -1).norm(dim=-1)


def summarize_errors(errors: Tensor) -> dict[str, float]:
    return {
        "mean": errors.mean().item(),
        "median": errors.median().item(),
        "max": errors.max().item(),
    }


def summarize_per_variable(problem: Problem, errors: Tensor) -> dict[str, dict[str, float]]:
    """Summaries keyed by solution variable, for errors of shape ``(B, C)``."""
    return {name: summarize_errors(errors[:, c]) for c, name in enumerate(problem.variable_names["solution"])}


@torch.no_grad()
def predict_dataset(
    model: nn.Module, problem: Problem, dataset: OperatorDataset, device: torch.device, batch_size: int = 32
) -> tuple[Tensor, Tensor]:
    """Return all predictions and the errors ``(B, C)``, both on the CPU."""
    was_training = model.training
    model.eval()
    grid = dataset.grid.to(device)
    predictions, errors = [], []
    for start in range(0, len(dataset), batch_size):
        inputs = dataset.inputs[start : start + batch_size].to(device)
        target = dataset.targets[start : start + batch_size].to(device)
        prediction = predict(model, problem, inputs, grid)
        predictions.append(prediction.cpu())
        errors.append(relative_l2(prediction, target).cpu())
    model.train(was_training)
    return torch.cat(predictions), torch.cat(errors)


@dataclass
class EvaluationResult:
    """Test-set evaluation at the test resolution and the resolution study."""

    dataset: OperatorDataset
    predictions: Tensor
    #: Relative L2 per example and output channel, ``(B, C)``.
    errors: Tensor
    main_channel: int
    #: Resolution -> variable -> summary.
    resolution_errors: dict[int, dict[str, dict[str, float]]]
    main_variable: str

    @property
    def main_errors(self) -> Tensor:
        return self.errors[:, self.main_channel]

    def main_resolution_errors(self) -> dict[int, dict[str, float]]:
        return {resolution: errors[self.main_variable] for resolution, errors in self.resolution_errors.items()}


def test_seed(seed: int) -> int:
    """Seed of the test inputs, different from the training and validation seeds."""
    return seed + 1000


def validation_seed(seed: int) -> int:
    """Seed of the validation inputs, different from the training and test seeds."""
    return seed + 2000


@dataclass
class ValidationResult:
    """Errors against the analytical solution ``(B, C)`` and the mean physics loss."""

    errors: Tensor
    physics: float


@torch.no_grad()
def validate(
    model: nn.Module, problem: Problem, dataset: OperatorDataset, device: torch.device, batch_size: int = 32
) -> ValidationResult:
    """Evaluate the relative L2 per variable and the physics loss ``mean(residual^2)``.

    The physics loss uses only the inputs, never the analytical solution, so it can select a
    checkpoint in the ``physics`` mode.
    """
    was_training = model.training
    model.eval()
    grid = dataset.grid.to(device)
    errors, squared, count = [], 0.0, 0
    for start in range(0, len(dataset), batch_size):
        inputs = dataset.inputs[start : start + batch_size].to(device)
        target = dataset.targets[start : start + batch_size].to(device)
        prediction = predict(model, problem, inputs, grid)
        residual = problem.residual(prediction, inputs, grid)
        squared += residual.pow(2).sum().item()
        count += residual.numel()
        errors.append(relative_l2(prediction, target).cpu())
    model.train(was_training)
    return ValidationResult(torch.cat(errors), squared / count)


def evaluate_model(model: nn.Module, problem: Problem, cfg, device: torch.device) -> EvaluationResult:
    """Evaluate on the test set and, with the same inputs, on all study resolutions."""
    seed = test_seed(int(cfg.seed))
    batch_size = int(cfg.training.batch_size)
    n_test = int(cfg.training.n_test)
    dataset = build_dataset(problem, n_test, cfg.problem.test_resolution, seed)
    predictions, errors = predict_dataset(model, problem, dataset, device, batch_size)

    resolution_errors = {}
    for resolution in cfg.problem.resolution_study:
        study = build_dataset(problem, n_test, resolution, seed)
        _, study_errors = predict_dataset(model, problem, study, device, batch_size)
        resolution_errors[int(resolution)] = summarize_per_variable(problem, study_errors)
    return EvaluationResult(
        dataset, predictions, errors, problem.main_channel, resolution_errors, problem.main_variable
    )


def write_metrics(
    path: Path,
    result: EvaluationResult,
    problem: Problem,
    cfg,
    training_time_s: float,
    best_epoch: int,
    n_parameters: int,
    timestamp: str,
    history: dict | None = None,
) -> dict:
    """Write ``metrics.json`` and return its content.

    ``rel_l2`` and ``resolution_study`` refer to the main variable, the ``*_per_variable``
    entries hold every output channel.
    """
    metrics = {
        "benchmark": problem.name,
        "mode": cfg.training.mode,
        "model": cfg.model.name,
        "seed": int(cfg.seed),
        "timestamp": timestamp,
        "main_variable": problem.main_variable,
        "rel_l2": summarize_errors(result.main_errors),
        "rel_l2_per_variable": summarize_per_variable(problem, result.errors),
        "resolution_study": {str(k): v for k, v in result.main_resolution_errors().items()},
        "resolution_study_per_variable": {str(k): v for k, v in result.resolution_errors.items()},
        "train_resolution": int(cfg.problem.train_resolution),
        "test_resolution": int(cfg.problem.test_resolution),
        "training_time_s": training_time_s,
        "best_epoch": best_epoch,
        "selection": (history or {}).get("selection"),
        "loss_weights": {name: float(value) for name, value in cfg.training.weights.items()},
        "n_parameters": n_parameters,
    }
    Path(path).write_text(json.dumps(metrics, indent=2))
    return metrics
