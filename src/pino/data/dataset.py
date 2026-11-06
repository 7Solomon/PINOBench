"""Sample operator inputs and, where needed, their analytical solutions."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

from pino.problems.base import Problem


class OperatorDataset(Dataset):
    """Operator inputs with optional targets on one fixed grid.

    Without targets the analytical solution is never computed, so a purely physics-informed
    training run cannot use it by accident.
    """

    def __init__(self, inputs: Tensor, grid: Tensor, targets: Tensor | None = None) -> None:
        self.inputs = inputs
        self.grid = grid
        self.targets = targets

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = {"inputs": self.inputs[index]}
        if self.targets is not None:
            item["target"] = self.targets[index]
        return item


def build_dataset(problem: Problem, n: int, resolution, seed: int, with_targets: bool = True) -> OperatorDataset:
    """Sample ``n`` inputs reproducibly. The same seed yields the same inputs at any resolution."""
    generator = torch.Generator().manual_seed(seed)
    inputs = problem.sample_inputs(n, generator)
    grid = problem.grid(resolution)
    targets = problem.analytical(inputs, grid) if with_targets else None
    return OperatorDataset(inputs, grid, targets)
