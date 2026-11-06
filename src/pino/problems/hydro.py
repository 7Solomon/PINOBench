"""Groundwater benchmarks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from pino.analytical.porous_media import layered_darcy_head
from pino.numerics.constraints import dirichlet_1d
from pino.numerics.derivatives import face_flux
from pino.numerics.grids import as_resolution, grid_spacing, uniform_grid
from pino.problems.base import Problem
from pino.problems.registry import register_problem


class _Stationary1D(Problem):
    """Shared grid handling of the 1D stationary problems in this module."""

    n_dims = 1
    out_channels = 1
    plot_type = "1d"

    def grid(self, resolution: int | Sequence[int]) -> Tensor:
        return uniform_grid([(0.0, 1.0)], as_resolution(resolution, self.n_dims))


@register_problem("darcy_layered_1d")
class DarcyLayered1D(_Stationary1D):
    """Two-layer Darcy flow ``-(k h')' = 0`` with ``h(0) = 1`` and ``h(1) = 0``.

    Operator inputs are ``(k1, k2, x_s)`` with shape ``(B, 3)``. The network sees
    ``log10 k(x)`` and ``x``.
    """

    in_channels = 2  # log10 k(x), x
    variable_names = {"coordinates": ["x"], "solution": ["h"]}
    units = {"x": "m", "h": "m"}

    def __init__(self, cfg) -> None:
        self.log10_k_range = (math.log10(float(cfg.k_low)), math.log10(float(cfg.k_high)))
        self.interface_range = (float(cfg.interface_low), float(cfg.interface_high))
        self.scales = {"x": float(cfg.length), "h": float(cfg.head_drop)}

    def sample_inputs(self, n: int, generator: torch.Generator) -> Tensor:
        low, high = self.log10_k_range
        k = 10.0 ** (low + (high - low) * torch.rand(n, 2, generator=generator))
        interface_low, interface_high = self.interface_range
        interface = interface_low + (interface_high - interface_low) * torch.rand(n, 1, generator=generator)
        return torch.cat([k, interface], dim=1)

    def analytical(self, inputs: Tensor, grid: Tensor) -> Tensor:
        return layered_darcy_head(inputs[:, 0], inputs[:, 1], inputs[:, 2], grid[0]).unsqueeze(1)

    def face_conductivity(self, inputs: Tensor, grid: Tensor) -> Tensor:
        """Exact face values ``k_f = dx / (l1 / k1 + l2 / k2)``, shape ``(B, 1, N - 1)``.

        ``l1`` and ``l2`` are the parts of the interval ``[x_i, x_{i+1}]`` in each layer.
        With these harmonic means the discrete flux of a piecewise linear head is exact.
        """
        x, dx = grid[0], grid_spacing(grid, 0)
        k1, k2, interface = (inputs[:, i, None] for i in range(3))
        l1 = (interface - x[:-1]).clamp(0.0, dx)
        return (dx / (l1 / k1 + (dx - l1) / k2)).unsqueeze(1)

    def residual(self, u: Tensor, inputs: Tensor, grid: Tensor) -> Tensor:
        """Return the deviation of each face flux from the mean flux, relative to the exact flux.

        ``-(k h')' = 0`` means that the flux ``q = -k h'`` is the same on every face. The
        residual is ``(q_f - mean(q_f)) / k_h`` on the ``N - 1`` faces, where
        ``k_h = 1 / mean(1 / k_f)`` is the effective conductivity. For ``h(0) = 1`` and
        ``h(1) = 0`` the exact discrete flux equals ``k_h``, so the residual is a relative flux
        error. With the exact harmonic face values it vanishes for the analytical solution.

        This replaces the divergence ``-(k h')'`` at the nodes. That second-order form
        penalizes the kink at the layer interface with ``1 / dx^2``: an exact solution with a
        kink smoothed over three nodes (0.08 % rel. L2) scored a loss of 18, while a wrong
        solution with 19 % error scored 3.6. Physics training then converged to the wrong
        solution (19 % rel. L2, against 0.5 % with this flux form).
        """
        k_faces = self.face_conductivity(inputs, grid)
        flux = -face_flux(u, k_faces, grid_spacing(grid, 0))
        effective = 1.0 / (1.0 / k_faces).mean(dim=-1, keepdim=True)
        return (flux - flux.mean(dim=-1, keepdim=True)) / effective

    def boundary(self, inputs: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
        """``h = (1 - x) + x (1 - x) NN``."""
        ones = torch.ones(inputs.shape[0], 1, dtype=inputs.dtype, device=inputs.device)
        return dirichlet_1d(grid[0], ones, torch.zeros_like(ones))

    def to_fno_input(self, inputs: Tensor, grid: Tensor) -> Tensor:
        x = grid[0]
        k = torch.where(x < inputs[:, 2, None], inputs[:, 0, None], inputs[:, 1, None])
        return torch.stack([torch.log10(k), x.expand_as(k)], dim=1)
