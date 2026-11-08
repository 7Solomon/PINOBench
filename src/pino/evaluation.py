"""Prediction with the hard constraint, error metrics and the resolution study."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from pino.data.dataset import OperatorDataset, build_dataset
from pino.losses import boundary_loss, neumann_loss
from pino.numerics.constraints import apply_hard_constraint
from pino.numerics.grids import resolution_key
from pino.problems.base import Problem


def predict(model: nn.Module, problem: Problem, inputs: Tensor, grid: Tensor) -> Tensor:
    """Evaluate the model on ``inputs`` and apply the problem's output parametrization."""
    g, phi = problem.output_parametrization(inputs, grid)
    return apply_hard_constraint(model(problem.to_fno_input(inputs, grid)), g, phi)


def relative_l2(prediction: Tensor, target: Tensor) -> Tensor:
    """Return ``||u - u_hat|| / ||u||`` per example and channel, shape ``(B, C)``."""
    batch, channels = target.shape[:2]
    difference = (prediction - target).reshape(batch, channels, -1).norm(dim=-1)
    return difference / target.reshape(batch, channels, -1).norm(dim=-1)


def boundary_errors(problem: Problem, prediction: Tensor, target: Tensor, inputs: Tensor, grid: Tensor) -> Tensor:
    """Boundary error per example and channel, ``(B, C)``.

    ``max |u - g|`` over the boundary points of :meth:`Problem.boundary_values`, relative to
    ``max |u_exact|`` over the whole domain. The reference is the analytical solution, not the
    prediction, so a trivial prediction ``u ~ 0`` cannot shrink the error. About zero for the
    hard constraint (``constrained``, ``unscaled``) by construction.
    """
    g, mask = problem.boundary_values(inputs, grid)
    deviation = ((prediction - g).abs() * mask).flatten(2).amax(dim=-1)
    return deviation / target.abs().flatten(2).amax(dim=-1)


def neumann_errors(problem: Problem, prediction: Tensor, inputs: Tensor, grid: Tensor) -> Tensor:
    """Per example and Neumann condition, ``(B, K)``: ``max |q - target|`` relative to the load size."""
    fields = problem.neumann_fields(prediction, inputs, grid)
    scale = problem.neumann_scale(inputs, grid)[:, None]
    errors = [
        ((condition.values(fields[condition.quantity]) - condition.target).abs() / scale).amax(dim=-1)
        for condition in problem.neumann_conditions(inputs, grid)
    ]
    return torch.stack(errors, dim=1)


def stress_errors(problem: Problem, prediction: Tensor, inputs: Tensor, grid: Tensor) -> dict[str, Tensor]:
    """Rel. L2 of the finite-difference stresses against the analytical ones, each ``(B,)``."""
    predicted = problem.stress_fields(prediction, inputs, grid)
    if predicted is None:
        return {}
    exact = problem.exact_stress_fields(inputs, grid)
    return {name: relative_l2(field[:, None], exact[name][:, None])[:, 0] for name, field in predicted.items()}


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
    """Return all predictions, the rel. L2 ``(B, C)``, the boundary errors ``(B, C)`` and, where the
    problem has them, Neumann errors ``(B, K)`` and stress errors (name -> ``(B,)``), on the CPU."""
    was_training = model.training
    model.eval()
    grid = dataset.grid.to(device)
    predictions, errors, bc_errors, neumann, stresses = [], [], [], [], {}
    for start in range(0, len(dataset), batch_size):
        inputs = dataset.inputs[start : start + batch_size].to(device)
        target = dataset.targets[start : start + batch_size].to(device)
        prediction = predict(model, problem, inputs, grid)
        predictions.append(prediction.cpu())
        errors.append(relative_l2(prediction, target).cpu())
        bc_errors.append(boundary_errors(problem, prediction, target, inputs, grid).cpu())
        if problem.has_neumann:
            neumann.append(neumann_errors(problem, prediction, inputs, grid).cpu())
        for name, values in stress_errors(problem, prediction, inputs, grid).items():
            stresses.setdefault(name, []).append(values.cpu())
    model.train(was_training)
    return (
        torch.cat(predictions),
        torch.cat(errors),
        torch.cat(bc_errors),
        torch.cat(neumann) if neumann else None,
        {name: torch.cat(values) for name, values in stresses.items()},
    )


@dataclass
class EvaluationResult:
    """Test-set evaluation at the test resolution and the resolution study."""

    dataset: OperatorDataset
    predictions: Tensor
    #: Relative L2 per example and output channel, ``(B, C)``.
    errors: Tensor
    main_channel: int
    #: Resolution key (see ``resolution_key``) -> variable -> summary.
    resolution_errors: dict[int | str, dict[str, dict[str, float]]]
    main_variable: str
    #: Boundary error per example and output channel, ``(B, C)``, see :func:`boundary_errors`.
    bc_errors: Tensor | None = None
    #: Neumann error per example and condition, ``(B, K)``, and the labels of the conditions.
    neumann_errors: Tensor | None = None
    neumann_labels: tuple[str, ...] = ()
    #: Stress component -> rel. L2 per example of the finite-difference stresses.
    stress_errors: dict[str, Tensor] = field(default_factory=dict)

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
    """Errors against the analytical solution ``(B, C)``, the mean physics loss and, with
    ``output_param = soft``, the mean boundary penalty ``L_bc``; ``L_neumann`` for problems with
    Neumann conditions."""

    errors: Tensor
    physics: float
    bc: float | None = None
    neumann: float | None = None


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
    errors, squared, count, bc_total, neumann_total = [], 0.0, 0, 0.0, 0.0
    for start in range(0, len(dataset), batch_size):
        inputs = dataset.inputs[start : start + batch_size].to(device)
        target = dataset.targets[start : start + batch_size].to(device)
        prediction = predict(model, problem, inputs, grid)
        residual = problem.residual(prediction, inputs, grid)
        squared += residual.pow(2).sum().item()
        count += residual.numel()
        if problem.soft_constraint:
            bc_total += boundary_loss(problem, prediction, inputs, grid).item() * inputs.shape[0]
        if problem.has_neumann:
            neumann_total += neumann_loss(problem, prediction, inputs, grid).item() * inputs.shape[0]
        errors.append(relative_l2(prediction, target).cpu())
    model.train(was_training)
    bc = bc_total / len(dataset) if problem.soft_constraint else None
    neumann = neumann_total / len(dataset) if problem.has_neumann else None
    return ValidationResult(torch.cat(errors), squared / count, bc, neumann)


def evaluate_model(model: nn.Module, problem: Problem, cfg, device: torch.device) -> EvaluationResult:
    """Evaluate on the test set and, with the same inputs, on all study resolutions."""
    seed = test_seed(int(cfg.seed))
    batch_size = int(cfg.training.batch_size)
    n_test = int(cfg.training.n_test)
    dataset = build_dataset(problem, n_test, cfg.problem.test_resolution, seed)
    predictions, errors, bc_errors, neumann, stresses = predict_dataset(model, problem, dataset, device, batch_size)
    labels = tuple(c.label for c in problem.neumann_conditions(dataset.inputs[:1], dataset.grid))

    resolution_errors = {}
    for resolution in cfg.problem.resolution_study:
        study = build_dataset(problem, n_test, resolution, seed)
        _, study_errors, *_ = predict_dataset(model, problem, study, device, batch_size)
        resolution_errors[resolution_key(resolution)] = summarize_per_variable(problem, study_errors)
    return EvaluationResult(
        dataset, predictions, errors, problem.main_channel, resolution_errors, problem.main_variable, bc_errors,
        neumann, labels, stresses,
    )


def boundary_error_summary(problem: Problem, bc_errors: Tensor | None) -> dict[str, dict[str, float]]:
    """``bc_error_mean`` and ``bc_error_max`` over the test examples, per output variable."""
    if bc_errors is None:
        return {}
    names = problem.variable_names["solution"]
    return {
        "bc_error_mean": {name: bc_errors[:, c].mean().item() for c, name in enumerate(names)},
        "bc_error_max": {name: bc_errors[:, c].max().item() for c, name in enumerate(names)},
    }


def stress_rmse(problem: Problem, result: EvaluationResult, batch_size: int = 64) -> dict[str, float]:
    """RMS error of the finite-difference stresses in physical units, mean over the test set.

    Rel. L2 alone says little for a component that is small everywhere, such as ``sigma_yy`` of a
    block heated with a harmonic temperature: there even the exact displacements have a large
    relative error from the finite differences.
    """
    dataset, totals = result.dataset, {}
    for start in range(0, len(dataset.inputs), batch_size):
        inputs = dataset.inputs[start : start + batch_size]
        predicted = problem.stress_fields(result.predictions[start : start + batch_size], inputs, dataset.grid)
        exact = problem.exact_stress_fields(inputs, dataset.grid)
        for name, field in predicted.items():
            values = (field - exact[name]).flatten(1).pow(2).mean(dim=1).sqrt()
            totals[name] = totals.get(name, 0.0) + values.sum().item()
    return {name: float(problem.to_physical("sigma", total / len(dataset.inputs))) for name, total in totals.items()}


def mechanics_summary(result: EvaluationResult, problem: Problem) -> dict:
    """``neumann_error_*`` per condition, ``rel_l2_stress`` and ``stress_rmse`` per component, where they exist."""
    summary: dict = {}
    if result.neumann_errors is not None:
        columns = list(zip(result.neumann_labels, result.neumann_errors.T))
        summary["neumann_error_mean"] = {label: values.mean().item() for label, values in columns}
        summary["neumann_error_max"] = {label: values.max().item() for label, values in columns}
    if result.stress_errors:
        summary["rel_l2_stress"] = {name: summarize_errors(values) for name, values in result.stress_errors.items()}
        summary["stress_rmse"] = stress_rmse(problem, result)
    return summary


class IncompatibleRunError(ValueError):
    """A stored run was trained with a different output wrapper than the current code."""


def check_run_compatibility(stored: dict, problem: Problem, allow_legacy: bool = False) -> None:
    """Refuse to evaluate a checkpoint whose wrapper differs from the current one.

    ``stored`` is the ``history.json`` or ``metrics.json`` of the run. A checkpoint only
    holds the network ``NN``; the prediction ``g + phi NN`` also needs the wrapper it was
    trained with. Runs written before the wrapper version existed are refused unless
    ``allow_legacy`` is set.
    """
    current = problem.variant_metadata()
    if "wrapper_version" not in stored:
        if allow_legacy:
            return
        raise IncompatibleRunError(
            f"The run has no wrapper_version (written before it was introduced). The current "
            f"{problem.name} wrapper is version {current['wrapper_version']}. If you know the run "
            "used this wrapper, pass --allow-legacy."
        )
    differences = {key: (stored.get(key), value) for key, value in current.items() if stored.get(key) != value}
    if differences:
        detail = ", ".join(f"{key}: run {old!r}, code {new!r}" for key, (old, new) in differences.items())
        raise IncompatibleRunError(
            f"The run was trained with a different {problem.name} wrapper ({detail}). Evaluating it "
            "with the current code would give wrong results."
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
        **problem.variant_metadata(),
        "main_variable": problem.main_variable,
        "rel_l2": summarize_errors(result.main_errors),
        "rel_l2_per_variable": summarize_per_variable(problem, result.errors),
        "resolution_study": {str(k): v for k, v in result.main_resolution_errors().items()},
        "resolution_study_per_variable": {str(k): v for k, v in result.resolution_errors.items()},
        **boundary_error_summary(problem, result.bc_errors),
        **mechanics_summary(result, problem),
        "train_resolution": resolution_key(cfg.problem.train_resolution),
        "test_resolution": resolution_key(cfg.problem.test_resolution),
        "training_time_s": training_time_s,
        "best_epoch": best_epoch,
        "selection": (history or {}).get("selection"),
        "loss_weights": {name: float(value) for name, value in cfg.training.weights.items()},
        "training_settings": {key: cfg.training[key] for key in ("epochs", "batch_size", "lr", "n_train") if key in cfg.training},
        "n_parameters": n_parameters,
    }
    Path(path).write_text(json.dumps(metrics, indent=2))
    return metrics
