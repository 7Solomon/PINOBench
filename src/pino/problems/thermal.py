"""Thermal benchmarks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from pino.analytical.diffusion import heat_sine_series, sine_series
from pino.numerics.constraints import dirichlet_1d_t
from pino.numerics.grids import as_resolution, grid_spacing, uniform_grid
from pino.pdes.diffusion import diffusion_residual
from pino.problems.base import Problem
from pino.problems.registry import register_problem


@register_problem("heat_1d")
class Heat1D(Problem):
    """Heat equation ``dT/dt = a d2T/dx2`` with zero Dirichlet boundaries.

    The operator maps the initial profile ``T0(x) = sum_n b_n sin(n pi x)`` to ``T(x, t)``.
    Operator inputs are the coefficients ``b_n`` with shape ``(B, N)``.
    The grid has shape ``(2, Nx, Nt)`` with the channels ``x`` and ``t``.
    """

    n_dims = 2
    in_channels = 3  # T0(x), x, t / t_end
    out_channels = 1
    plot_type = "1d_t"
    variable_names = {"coordinates": ["x", "t"], "solution": ["T"]}
    units = {"x": "-", "t": "-", "T": "-"}  # non-dimensional
    scales = {"x": 1.0, "t": 1.0, "T": 1.0}

    def __init__(self, cfg) -> None:
        self.a = float(cfg.a)
        self.t_end = float(cfg.t_end)
        self.n_series = int(cfg.n_series)
        self.coeff_low = float(cfg.coeff_low)
        self.coeff_high = float(cfg.coeff_high)
        # Largest decay rate in the solution, the fastest mode's a * (N pi)^2.
        self.residual_scale = self.a * (self.n_series * math.pi) ** 2

    def grid(self, resolution: int | Sequence[int]) -> Tensor:
        return uniform_grid([(0.0, 1.0), (0.0, self.t_end)], as_resolution(resolution, self.n_dims))

    def sample_inputs(self, n: int, generator: torch.Generator) -> Tensor:
        unit = torch.rand(n, self.n_series, generator=generator)
        return self.coeff_low + (self.coeff_high - self.coeff_low) * unit

    def analytical(self, inputs: Tensor, grid: Tensor) -> Tensor:
        return heat_sine_series(inputs, grid[0], grid[1], self.a).unsqueeze(1)

    def residual(self, u: Tensor, inputs: Tensor, grid: Tensor) -> Tensor:
        """Return ``(dT/dt - a d2T/dx2) / (a (N pi)^2)`` on the interior points.

        Normalization: a mode ``n`` with amplitude ``b_n`` produces terms of size
        ``a (n pi)^2 b_n``. Dividing by the largest rate ``a (N pi)^2`` keeps the fastest
        mode's residual at the order of its amplitude, independent of ``a`` and ``N``, and
        makes the residual directly comparable with the size of ``T``.
        """
        dx, dt = grid_spacing(grid, 0), grid_spacing(grid, 1)
        return diffusion_residual(u, dx, dt, self.a) / self.residual_scale

    def boundary(self, inputs: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``g = T0(x)`` and ``phi = tau x (1 - x)`` with ``tau = t / t_end``."""
        x, t = grid[0, :, 0], grid[1, 0, :]
        initial = self._initial_profile(inputs, x)
        # sin(n pi) is only zero up to rounding, the Dirichlet value itself is exactly zero.
        initial[..., 0] = 0.0
        initial[..., -1] = 0.0
        zeros = initial.new_zeros(*initial.shape[:2], t.shape[0])
        return dirichlet_1d_t(x, t, initial, zeros, zeros)

    def to_fno_input(self, inputs: Tensor, grid: Tensor) -> Tensor:
        x, t = grid[0], grid[1]
        batch = inputs.shape[0]
        coordinates = torch.stack([x, t / self.t_end]).expand(batch, -1, -1, -1)
        return torch.cat([self._initial_profile(inputs, x), coordinates], dim=1)

    @staticmethod
    def _initial_profile(inputs: Tensor, x: Tensor) -> Tensor:
        return sine_series(inputs, x).unsqueeze(1)
