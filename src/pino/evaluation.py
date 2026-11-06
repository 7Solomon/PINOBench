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
    """Return ``||u - u_hat|| / ||u||`` per example, shape ``(B,)``."""
    batch = target.shape[0]
    difference = (prediction - target).reshape(batch, -1).norm(dim=1)
    return difference / target.reshape(batch, -1).norm(dim=1)


def summarize_errors(errors: Tensor) -> dict[str, float]:
    return {
        "mean": errors.mean().item(),
        "median": errors.median().item(),
        "max": errors.max().item(),
    }


@torch.no_grad()
def predict_dataset(
    model: nn.Module, problem: Problem, dataset: OperatorDataset, device: torch.device, batch_size: int = 32
) -> tuple[Tensor, Tensor]:
    """Return all predictions and per-example errors, both on the CPU."""
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
    errors: Tensor
    resolution_errors: dict[int, dict[str, float]]


def test_seed(seed: int) -> int:
    """Seed of the test inputs, different from the training seed."""
    return seed + 1000


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
        resolution_errors[int(resolution)] = summarize_errors(study_errors)
    return EvaluationResult(dataset, predictions, errors, resolution_errors)


def write_metrics(
    path: Path,
    result: EvaluationResult,
    problem: Problem,
    cfg,
    training_time_s: float,
    best_epoch: int,
    n_parameters: int,
    timestamp: str,
) -> dict:
    """Write ``metrics.json`` and return its content."""
    metrics = {
        "benchmark": problem.name,
        "mode": cfg.training.mode,
        "model": cfg.model.name,
        "seed": int(cfg.seed),
        "timestamp": timestamp,
        "rel_l2": summarize_errors(result.errors),
        "resolution_study": {str(k): v for k, v in result.resolution_errors.items()},
        "train_resolution": int(cfg.problem.train_resolution),
        "test_resolution": int(cfg.problem.test_resolution),
        "training_time_s": training_time_s,
        "best_epoch": best_epoch,
        "n_parameters": n_parameters,
    }
    Path(path).write_text(json.dumps(metrics, indent=2))
    return metrics
