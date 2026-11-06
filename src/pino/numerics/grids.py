"""Uniform tensor-product grids including the domain boundaries."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def as_resolution(resolution: int | Sequence[int], n_dims: int) -> tuple[int, ...]:
    """Expand a scalar resolution to ``n_dims`` entries and validate sequences."""
    if isinstance(resolution, int):
        return (resolution,) * n_dims
    resolution = tuple(int(r) for r in resolution)
    if len(resolution) != n_dims:
        raise ValueError(f"Expected {n_dims} resolution entries, got {resolution}.")
    return resolution


def uniform_grid(bounds: Sequence[tuple[float, float]], resolution: Sequence[int]) -> Tensor:
    """Build a coordinate tensor of shape ``(len(bounds), *resolution)`` (``ij`` indexing)."""
    axes = [torch.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, resolution, strict=True)]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)


def grid_spacing(grid: Tensor, axis: int) -> float:
    """Return the uniform spacing of coordinate ``axis`` for a grid from :func:`uniform_grid`."""
    coordinate = grid[axis]
    n = coordinate.shape[axis]
    return float((coordinate.narrow(axis, n - 1, 1).flatten()[0] - coordinate.flatten()[0]) / (n - 1))
