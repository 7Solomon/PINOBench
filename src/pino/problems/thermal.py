"""Thermal benchmarks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from pino.analytical.diffusion import heat_ramp_series, heat_sine_series, heat_sine_series_2d, sine_series
from pino.numerics.constraints import dirichlet_1d_t, dirichlet_2d_t
from pino.numerics.grids import as_resolution, uniform_grid
from pino.pdes.diffusion import diffusion_residual
from pino.problems.base import Problem
from pino.problems.registry import register_problem


@register_problem("heat_1d")
class Heat1D(Problem):
    """Heat equation ``dT/dt = a d2T/dx2`` with zero Dirichlet boundaries.

    The operator maps the initial profile ``T0(x) = sum_n b_n sin(n pi x)`` to ``T(x, t)``.
    Operator inputs are the coefficients ``b_n`` with shape ``(B, N)``.
    The grid has shape ``(2, Nx, Nt)`` with the channels ``x`` and ``t``.

    ``inputs = sine_ramp`` adds boundary values linear in time, ``T(0, t) = alpha_L + beta_L t``
    and ``T(1, t) = alpha_R + beta_R t``, with the compatible initial profile
    ``T0 = (1 - x) alpha_L + x alpha_R + sum_n b_n sin(n pi x)``. Operator inputs are then
    ``(b_1, ..., b_N, alpha_L, beta_L, alpha_R, beta_R)`` with shape ``(B, N + 4)``, and the
    network also sees ``T_L(t)`` and ``T_R(t)``.
    """

    # 1: phi = t x (1 - x) (until Phase 2); 2: dirichlet_1d_t, phi = tau x (1 - x).
    WRAPPER_VERSION = "2"
    SUPPORTED_INPUTS = ("sine", "sine_ramp")  # T0 = sine series; plus boundary values linear in t

    n_dims = 2
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
        # Boundary ramps of inputs = sine_ramp: start values alpha and rates beta.
        self.start_range = (float(cfg.get("bc_start_low", -1.0)), float(cfg.get("bc_start_high", 1.0)))
        self.rate_range = (float(cfg.get("bc_rate_low", -10.0)), float(cfg.get("bc_rate_high", 10.0)))
        # Largest decay rate in the solution, the fastest mode's a * (N pi)^2.
        self.residual_scale = self.a * (self.n_series * math.pi) ** 2

    @property
    def in_channels(self) -> int:
        """``T0(x), x, t / t_end``; ``sine_ramp`` adds ``T_L(t)`` and ``T_R(t)``."""
        return 5 if self.inputs == "sine_ramp" else 3

    def grid(self, resolution: int | Sequence[int]) -> Tensor:
        return uniform_grid([(0.0, 1.0), (0.0, self.t_end)], as_resolution(resolution, self.n_dims))

    def sample_inputs(self, n: int, generator: torch.Generator) -> Tensor:
        unit = torch.rand(n, self.n_series, generator=generator)
        coefficients = self.coeff_low + (self.coeff_high - self.coeff_low) * unit
        if self.inputs != "sine_ramp":
            return coefficients
        low = torch.tensor([self.start_range[0], self.rate_range[0]] * 2)
        high = torch.tensor([self.start_range[1], self.rate_range[1]] * 2)
        return torch.cat([coefficients, low + (high - low) * torch.rand(n, 4, generator=generator)], dim=1)

    def analytical(self, inputs: Tensor, grid: Tensor) -> Tensor:
        if self.inputs == "sine_ramp":
            ramps = inputs[:, self.n_series :].unbind(dim=1)
            return heat_ramp_series(inputs[:, : self.n_series], *ramps, grid[0], grid[1], self.a).unsqueeze(1)
        return heat_sine_series(inputs, grid[0], grid[1], self.a).unsqueeze(1)

    def residual(self, u: Tensor, inputs: Tensor, grid: Tensor) -> Tensor:
        """Return ``(dT/dt - a d2T/dx2) / (a (N pi)^2)`` on the interior points.

        Normalization: a mode ``n`` with amplitude ``b_n`` produces terms of size
        ``a (n pi)^2 b_n``. Dividing by the largest rate ``a (N pi)^2`` keeps the fastest
        mode's residual at the order of its amplitude, independent of ``a`` and ``N``, and
        makes the residual directly comparable with the size of ``T``.
        """
        return diffusion_residual(u, grid, self.a) / self.residual_scale

    def boundary(self, inputs: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
        """``dirichlet_1d_t`` with ``T0``, ``phi = tau x (1 - x)``, ``tau = t / t_end``.

        ``g = T0(x)`` for ``sine``; ``sine_ramp`` adds ``(1 - x) beta_L t + x beta_R t``.
        """
        x, t = grid[0, :, 0], grid[1, 0, :]
        initial = self._initial_profile(inputs, x)
        if self.inputs == "sine_ramp":
            alpha_left, beta_left, alpha_right, beta_right = (inputs[:, self.n_series + i, None, None] for i in range(4))
            # sin(n pi) is only zero up to rounding, the end values are exactly the start values.
            initial[..., 0] = alpha_left[..., 0]
            initial[..., -1] = alpha_right[..., 0]
            return dirichlet_1d_t(x, t, initial, alpha_left + beta_left * t, alpha_right + beta_right * t)
        # sin(n pi) is only zero up to rounding, the Dirichlet value itself is exactly zero.
        initial[..., 0] = 0.0
        initial[..., -1] = 0.0
        zeros = initial.new_zeros(*initial.shape[:2], t.shape[0])
        return dirichlet_1d_t(x, t, initial, zeros, zeros)

    def to_fno_input(self, inputs: Tensor, grid: Tensor) -> Tensor:
        x, t = grid[0], grid[1]
        batch = inputs.shape[0]
        coordinates = torch.stack([x, t / self.t_end]).expand(batch, -1, -1, -1)
        channels = [self._initial_profile(inputs, x), coordinates]
        if self.inputs == "sine_ramp":
            alpha_left, beta_left, alpha_right, beta_right = (inputs[:, self.n_series + i, None, None] for i in range(4))
            channels.insert(1, torch.stack([alpha_left + beta_left * t, alpha_right + beta_right * t], dim=1))
        return torch.cat(channels, dim=1)

    def _initial_profile(self, inputs: Tensor, x: Tensor) -> Tensor:
        """``T0`` at ``x``, shape ``(B, 1, *x.shape)``."""
        profile = sine_series(inputs[:, : self.n_series], x)
        if self.inputs == "sine_ramp":
            extra = (1,) * x.dim()
            alpha_left = inputs[:, self.n_series].reshape(-1, *extra)
            alpha_right = inputs[:, self.n_series + 2].reshape(-1, *extra)
            profile = profile + (1.0 - x) * alpha_left + x * alpha_right
        return profile.unsqueeze(1)


@register_problem("heat_2d")
class Heat2D(Problem):
    """Heat equation ``dT/dt = a (d2T/dx2 + d2T/dy2)`` on ``[0, 1]^2`` with ``T = 0`` on the boundary.

    The operator maps ``T0(x, y) = sum_{m,n=1..M} b_mn sin(m pi x) sin(n pi y)`` to ``T(x, y, t)``.
    Operator inputs are the coefficients ``b_mn`` with shape ``(B, M, M)``. The grid has shape
    ``(3, Nx, Ny, Nt)`` with time last. The network sees ``T0`` (broadcast over ``t``), ``x``,
    ``y`` and ``t / t_end``.

    Reference values as in heat_1d: ``x = x_hat L``, ``T = T_hat dT``; time stays
    non-dimensional, ``t_hat = a t / L^2``.
    """

    WRAPPER_VERSION = "1"
    SUPPORTED_INPUTS = ("sine",)  # T0 = double sine series

    n_dims = 3
    in_channels = 4  # T0(x, y), x, y, t / t_end
    out_channels = 1
    plot_type = "2d_t"
    variable_names = {"coordinates": ["x", "y", "t"], "solution": ["T"]}

    def __init__(self, cfg) -> None:
        self.a = float(cfg.a)
        self.t_end = float(cfg.t_end)
        self.n_series = int(cfg.n_series)
        self.coeff_range = (float(cfg.coeff_low), float(cfg.coeff_high))
        # Largest decay rate in the solution, the fastest mode's a pi^2 (M^2 + M^2).
        self.residual_scale = self.a * math.pi**2 * 2 * self.n_series**2
        length, delta = float(cfg.length), float(cfg.temperature_difference)
        self.scales = {"x": length, "y": length, "t": 1.0, "T": delta}
        self.units = {"x": "m", "y": "m", "t": "-", "T": "K"}

    def grid(self, resolution: int | Sequence[int]) -> Tensor:
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, self.t_end)]
        return uniform_grid(bounds, as_resolution(resolution, self.n_dims))

    def sample_inputs(self, n: int, generator: torch.Generator) -> Tensor:
        low, high = self.coeff_range
        return low + (high - low) * torch.rand(n, self.n_series, self.n_series, generator=generator)

    def analytical(self, inputs: Tensor, grid: Tensor) -> Tensor:
        return heat_sine_series_2d(inputs, grid[0], grid[1], grid[2], self.a).unsqueeze(1)

    def residual(self, u: Tensor, inputs: Tensor, grid: Tensor) -> Tensor:
        """Return ``(dT/dt - a Laplace T) / (a pi^2 2 M^2)`` on the interior points.

        As in heat_1d: a mode ``(m, n)`` with amplitude ``b_mn`` produces terms of size
        ``a pi^2 (m^2 + n^2) b_mn``; dividing by the fastest rate ``a pi^2 2 M^2`` (178 for
        ``M = 3``) keeps its residual at the order of its amplitude.
        """
        return diffusion_residual(u, grid, self.a) / self.residual_scale

    def boundary(self, inputs: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
        """``dirichlet_2d_t`` with ``u0 = T0`` and zero edges: ``g = T0``, ``phi = tau x(1-x) y(1-y)``."""
        x, y, t = grid[0, :, 0, 0], grid[1, 0, :, 0], grid[2, 0, 0, :]
        initial = self._initial_field(inputs, grid[0, ..., 0], grid[1, ..., 0])
        # sin(m pi) is only zero up to rounding, the Dirichlet value itself is exactly zero.
        initial[..., [0, -1], :] = 0.0
        initial[..., :, [0, -1]] = 0.0
        batch = inputs.shape[0]
        edge_x, edge_y = inputs.new_zeros(batch, 1, x.shape[0], t.shape[0]), inputs.new_zeros(batch, 1, y.shape[0], t.shape[0])
        return dirichlet_2d_t(x, y, t, initial, edge_x, edge_x, edge_y, edge_y)

    def to_fno_input(self, inputs: Tensor, grid: Tensor) -> Tensor:
        batch = inputs.shape[0]
        initial = self._initial_field(inputs, grid[0, ..., 0], grid[1, ..., 0])[..., None].expand(-1, -1, *grid.shape[1:])
        coordinates = torch.stack([grid[0], grid[1], grid[2] / self.t_end]).expand(batch, -1, -1, -1, -1)
        return torch.cat([initial, coordinates], dim=1)

    def _initial_field(self, inputs: Tensor, x: Tensor, y: Tensor) -> Tensor:
        """``T0`` on the space grid, ``(B, 1, Nx, Ny)``."""
        return heat_sine_series_2d(inputs, x, y, torch.zeros_like(x), self.a).unsqueeze(1)
