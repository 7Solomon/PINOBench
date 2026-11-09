"""Load a finished run and evaluate its model on hand-made operator inputs.

Typical use, in a script or a notebook::

    run = load_latest("heat_1d", mode="physics")  # newest run, default inputs, constrained
    run = load_run("workspace/outputs/heat_1d/sine-constrained/physics/<timestamp>")  # a specific run
    prediction = predict_one(run, [1.0, 0.0, 0.5, 0.0, 0.0], resolution=128)
    save_prediction(run, prediction, "half_and_third_mode")

Inputs are given in the problem's own parametrization, the layout of
``Problem.sample_inputs`` for one example (heat_1d: the coefficients ``b_1..b_5``,
chloride_1d: ``(Cs, D)``, darcy_poisson_2d: an ``M x M`` matrix). Inputs outside the
range the model was trained on are evaluated as well, but flagged as extrapolation.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict
from torch import Tensor, nn

from pino.data.dataset import OperatorDataset
from pino.evaluation import (
    EvaluationResult,
    IncompatibleRunError,
    boundary_errors,
    check_run_compatibility,
    predict,
    relative_l2,
)
from pino.losses import overridden_keys, resolve_loss_weights, resolve_training_settings
from pino.models.factory import build_model
from pino.numerics.grids import resolution_key
from pino.paths import OUTPUTS
from pino.problems import build_problem, problem_class
from pino.problems.base import Problem
from pino.trainer import CHECKPOINT, HISTORY, select_device

#: Subdirectory of a run that holds the saved predictions.
PREDICTIONS = "predictions"
#: Runs below this directory are ignored by :func:`find_run`, as by ``summarize.py``.
SUPERSEDED = "_superseded"


def find_run(
    benchmark: str,
    mode: str = "physics",
    output_param: str = "constrained",
    inputs: str | None = None,
    seed: int | None = None,
    outputs: str | Path = OUTPUTS,
) -> Path:
    """Return the newest finished run of a combination, by the ``metrics.json`` timestamp.

    ``inputs`` defaults to the benchmark's original input family; ``seed`` None takes any
    seed. Only runs with ``metrics.json`` and ``best_model.pt`` count; runs below
    ``_superseded/`` are ignored. Raises ``FileNotFoundError`` listing what exists instead.
    """
    inputs = inputs or problem_class(benchmark).default_inputs()
    outputs = Path(outputs)
    candidates, available = [], set()
    for path in (outputs / benchmark).glob("*/*/*/metrics.json"):
        if SUPERSEDED in path.relative_to(outputs).parts or not (path.parent / CHECKPOINT).exists():
            continue
        metrics = json.loads(path.read_text())
        key = (metrics.get("inputs", inputs), metrics.get("output_param", "constrained"), metrics["mode"])
        available.add(f"{key[0]}/{key[1]} {key[2]} seed {metrics['seed']}")
        if key == (inputs, output_param, mode) and (seed is None or int(metrics["seed"]) == seed):
            candidates.append((metrics["timestamp"], path.parent))
    if not candidates:
        wanted = f"{benchmark} {inputs}/{output_param} {mode}" + ("" if seed is None else f" seed {seed}")
        listing = "\n  ".join(sorted(available)) or "(none)"
        raise FileNotFoundError(f"No finished run for {wanted} under {outputs}. Available:\n  {listing}")
    return max(candidates)[1]


def load_latest(benchmark: str, mode: str = "physics", output_param: str = "constrained", inputs: str | None = None,
                seed: int | None = None, outputs: str | Path = OUTPUTS, device: str = "auto") -> "LoadedRun":
    """:func:`load_run` of :func:`find_run`."""
    return load_run(find_run(benchmark, mode, output_param, inputs, seed, outputs), device)


@dataclass
class LoadedRun:
    """A finished run, ready for predictions."""

    run_dir: Path
    cfg: object
    problem: Problem
    model: nn.Module
    device: torch.device
    history: dict = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return str(self.cfg.training.mode)


def load_run(run_dir: str | Path, device: str = "auto", allow_legacy: bool = False) -> LoadedRun:
    """Rebuild problem, model and resolved config of ``run_dir`` and load the best checkpoint.

    Refuses runs whose output wrapper differs from the current code (see
    :func:`pino.evaluation.check_run_compatibility`). The wrapper fields are taken from
    ``history.json``, or from ``metrics.json`` where only that carries them (older runs whose
    metrics were re-evaluated and completed); runs without them anywhere need
    ``allow_legacy``. Loss weights and the problem's training block are resolved as in
    ``train.py``.
    """
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    history = json.loads((run_dir / HISTORY).read_text())
    problem = build_problem(cfg.problem)
    stored = history
    metrics_path = run_dir / "metrics.json"
    if "wrapper_version" not in history and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        if "wrapper_version" in metrics:
            stored = metrics
    check_run_compatibility(stored, problem, allow_legacy)
    overrides = overridden_keys(OmegaConf.load(run_dir / ".hydra" / "overrides.yaml"))
    with open_dict(cfg):
        cfg.training.weights = OmegaConf.create(resolve_loss_weights(cfg.training, cfg.problem, overrides))
        cfg.training.update(resolve_training_settings(cfg.training, cfg.problem, overrides))
    device = select_device(device)
    model = build_model(cfg.model, problem, cfg.problem.train_resolution).to(device)
    model.load_state_dict(torch.load(run_dir / CHECKPOINT, map_location=device, weights_only=False))
    model.eval()
    return LoadedRun(run_dir, cfg, problem, model, device, history)


@dataclass
class Prediction:
    """Prediction for one hand-made input, on the CPU."""

    inputs: Tensor
    resolution: int | str
    grid: Tensor
    prediction: Tensor
    exact: Tensor
    #: Variable -> rel. L2 against the analytical solution.
    rel_l2: dict[str, float]
    #: Variable -> max |u - g| on the boundary points relative to max |u_exact|.
    bc_error: dict[str, float]
    #: Input components outside the trained range, as readable messages; empty if none.
    extrapolation: list[str]

    def as_result(self, problem: Problem) -> EvaluationResult:
        """Wrap as a one-example :class:`EvaluationResult` for the solution plots."""
        dataset = OperatorDataset(self.inputs[None], self.grid, self.exact[None])
        names = problem.variable_names["solution"]
        errors = torch.tensor([[self.rel_l2[name] for name in names]])
        return EvaluationResult(dataset, self.prediction[None], errors, problem.main_channel, {}, problem.main_variable)


def as_inputs(problem: Problem, values) -> Tensor:
    """Bring ``values`` (list, nested list or tensor) to the shape of one sampled input."""
    shape = problem.sample_inputs(1, torch.Generator().manual_seed(0)).shape[1:]
    tensor = torch.as_tensor(values, dtype=torch.get_default_dtype()).flatten()
    if tensor.numel() != shape.numel():
        raise ValueError(
            f"{problem.name} expects {shape.numel()} input values (shape {tuple(shape)}), got {tensor.numel()}."
        )
    return tensor.reshape(shape)


#: Tolerance of the range check as a fraction of each range: the default ranges are the
#: extremes of a finite sample and lie slightly inside the true bounds.
RANGE_TOLERANCE = 0.01


def extrapolation_messages(problem: Problem, inputs: Tensor) -> list[str]:
    """Describe every input component outside :meth:`Problem.input_ranges` (with tolerance)."""
    low, high = problem.input_ranges()
    margin = RANGE_TOLERANCE * (high - low)
    messages = []
    for index in torch.nonzero((inputs < low - margin) | (inputs > high + margin)).tolist():
        i = tuple(index)
        messages.append(
            f"input{list(i)} = {inputs[i].item():.4g} outside the trained range "
            f"[{low[i].item():.4g}, {high[i].item():.4g}]"
        )
    return messages


@torch.no_grad()
def predict_one(run: LoadedRun, values, resolution=None) -> Prediction:
    """Evaluate the model of ``run`` on one input; ``resolution`` defaults to the test grid."""
    problem = run.problem
    inputs = as_inputs(problem, values)
    resolution = run.cfg.problem.test_resolution if resolution is None else resolution
    grid = problem.grid(resolution)
    messages = extrapolation_messages(problem, inputs)
    for message in messages:
        warnings.warn(f"Extrapolation: {message}", stacklevel=3)

    batch, device_grid = inputs[None].to(run.device), grid.to(run.device)
    prediction = predict(run.model, problem, batch, device_grid)
    exact = problem.analytical(batch, device_grid)
    names = problem.variable_names["solution"]
    rel_l2 = relative_l2(prediction, exact)[0]
    bc = boundary_errors(problem, prediction, exact, batch, device_grid)[0]
    return Prediction(
        inputs=inputs,
        resolution=resolution_key(resolution),
        grid=grid,
        prediction=prediction[0].cpu(),
        exact=exact[0].cpu(),
        rel_l2={name: rel_l2[c].item() for c, name in enumerate(names)},
        bc_error={name: bc[c].item() for c, name in enumerate(names)},
        extrapolation=messages,
    )


def save_prediction(run: LoadedRun, prediction: Prediction, name: str) -> Path:
    """Write ``<run_dir>/predictions/<name>/`` with inputs, metrics, tensors and the plot.

    ``summarize.py`` only collects files called ``metrics.json``, so predictions never enter
    the overview.
    """
    from pino.visualization import plot_prediction  # matplotlib only when saving

    directory = run.run_dir / PREDICTIONS / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "inputs.json").write_text(
        json.dumps(
            {
                "inputs": prediction.inputs.tolist(),
                "resolution": prediction.resolution,
                "extrapolation": prediction.extrapolation,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )
    (directory / "result.json").write_text(
        json.dumps({"rel_l2": prediction.rel_l2, "bc_error": prediction.bc_error}, indent=2)
    )
    torch.save(
        {"inputs": prediction.inputs, "grid": prediction.grid, "prediction": prediction.prediction, "exact": prediction.exact},
        directory / "prediction.pt",
    )
    plot_prediction(run.problem, prediction, run.mode, directory / "solution.png")
    return directory


__all__ = [
    "IncompatibleRunError",
    "LoadedRun",
    "Prediction",
    "as_inputs",
    "extrapolation_messages",
    "find_run",
    "load_latest",
    "load_run",
    "predict_one",
    "save_prediction",
]
