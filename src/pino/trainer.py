"""Training loop: Adam, cosine schedule, best checkpoint by relative L2 on the test set."""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from pino.data.dataset import build_dataset
from pino.evaluation import predict, predict_dataset, test_seed
from pino.losses import compute_loss, uses_data
from pino.problems.base import Problem

log = logging.getLogger(__name__)

CHECKPOINT = "best_model.pt"
HISTORY = "history.json"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch on all devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str = "auto") -> torch.device:
    """Resolve the device name, where ``auto`` prefers CUDA and falls back to the CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass
class TrainingResult:
    """History of a run. ``best_state`` holds the weights of the best test error."""

    loss_epochs: list[int] = field(default_factory=list)
    losses: dict[str, list[float]] = field(default_factory=dict)
    eval_epochs: list[int] = field(default_factory=list)
    test_rel_l2: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_rel_l2: float = float("inf")
    training_time_s: float = 0.0
    best_state: dict[str, Tensor] | None = None

    def history(self) -> dict:
        """Return everything except the weights, ready for JSON."""
        return {
            "loss_epochs": self.loss_epochs,
            "losses": self.losses,
            "eval_epochs": self.eval_epochs,
            "test_rel_l2": self.test_rel_l2,
            "best_epoch": self.best_epoch,
            "best_rel_l2": self.best_rel_l2,
            "training_time_s": self.training_time_s,
        }


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    problem: Problem,
    inputs: Tensor,
    grid: Tensor,
    mode: str,
    weights,
    target: Tensor | None = None,
) -> dict[str, float]:
    """Run one optimizer step and return the loss components."""
    optimizer.zero_grad(set_to_none=True)
    prediction = predict(model, problem, inputs, grid)
    loss, components = compute_loss(mode, weights, problem, prediction, inputs, grid, target)
    loss.backward()
    optimizer.step()
    return components


def save_atomic(path: Path, write: Callable[[Path], None]) -> None:
    """Write a file through a temporary sibling, so a failed write keeps the old file intact."""
    temporary = path.with_name(path.name + ".tmp")
    write(temporary)
    os.replace(temporary, path)


def save_progress(run_dir: Path, result: TrainingResult, write_model: bool) -> None:
    """Write ``history.json`` and, after an improvement, ``best_model.pt``."""
    if write_model:
        save_atomic(run_dir / CHECKPOINT, lambda path: torch.save(result.best_state, path))
    save_atomic(run_dir / HISTORY, lambda path: path.write_text(json.dumps(result.history())))


def train(
    model: nn.Module, problem: Problem, cfg, device: torch.device, run_dir: Path | None = None
) -> TrainingResult:
    """Train ``model`` according to ``cfg`` and return its history and best weights.

    With ``run_dir``, the best weights and the history are written whenever the test error
    improves and once more at the end, so an interrupted run keeps its best model.
    """
    training = cfg.training
    seed = int(cfg.seed)
    train_set = build_dataset(
        problem,
        int(training.n_train),
        cfg.problem.train_resolution,
        seed,
        with_targets=uses_data(training.mode),
    )
    test_set = build_dataset(problem, int(training.n_test), cfg.problem.test_resolution, test_seed(seed))
    loader = DataLoader(
        train_set,
        batch_size=int(training.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    grid = train_set.grid.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training.lr))
    epochs = int(training.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    result = TrainingResult()
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        for batch in loader:
            inputs = batch["inputs"].to(device)
            target = batch["target"].to(device) if "target" in batch else None
            components = train_step(model, optimizer, problem, inputs, grid, training.mode, training.weights, target)
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + value
        scheduler.step()

        result.loss_epochs.append(epoch)
        for name, value in totals.items():
            result.losses.setdefault(name, []).append(value / len(loader))

        if epoch % int(training.eval_every) == 0 or epoch == epochs:
            _, errors = predict_dataset(model, problem, test_set, device, int(training.batch_size))
            error = errors[:, problem.main_channel].mean().item()
            result.eval_epochs.append(epoch)
            result.test_rel_l2.append(error)
            improved = error < result.best_rel_l2
            if improved:
                result.best_rel_l2 = error
                result.best_epoch = epoch
                result.best_state = copy.deepcopy(model.state_dict())
            result.training_time_s = time.perf_counter() - start
            if run_dir is not None:
                save_progress(run_dir, result, write_model=improved)
            log.info(
                "epoch %4d/%d  loss %.3e  test rel. L2 %.3f %%  best %.3f %%  %.0f s",
                epoch,
                epochs,
                result.losses["total"][-1],
                100 * error,
                100 * result.best_rel_l2,
                time.perf_counter() - start,
            )
    result.training_time_s = time.perf_counter() - start
    if run_dir is not None:
        save_progress(run_dir, result, write_model=False)
    return result
