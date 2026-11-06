"""Groundwater and unsaturated-flow benchmarks in 1D and 2D."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from pino.analytical.porous_media import (
    gardner_steady_head,
    layered_darcy_head,
    poisson_sine_head_2d,
    sine_source_2d,
)
from pino.numerics.constraints import dirichlet_1d, dirichlet_2d
from pino.numerics.derivatives import conservative_divergence, face_flux, interior, laplacian
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


@register_problem("richards_steady_1d")
class RichardsSteady1D(_Stationary1D):
    """Steady Richards equation ``d/dz [K(h) (dh/dz + 1)] = 0`` with Gardner ``K = Ks exp(alpha h)``.

    ``z`` points up from the water table, ``h(0) = 0``. ``h(1)`` is taken from the analytical
    solution. Operator inputs are ``(alpha, q / Ks)`` with shape ``(B, 2)``. Lengths are
    relative to ``L``, ``alpha`` to ``1 / L`` and conductivities and fluxes to ``Ks``.
    """

    in_channels = 3  # alpha, q / Ks, z
    variable_names = {"coordinates": ["z"], "solution": ["h"]}
    units = {"z": "m", "h": "m"}

    def __init__(self, cfg) -> None:
        self.alpha_range = (float(cfg.alpha_low), float(cfg.alpha_high))
        self.flux_range = (float(cfg.flux_low), float(cfg.flux_high))
        self.scales = {"z": float(cfg.length), "h": float(cfg.length)}

    def sample_inputs(self, n: int, generator: torch.Generator) -> Tensor:
        ranges = torch.tensor([self.alpha_range, self.flux_range])
        return ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * torch.rand(n, 2, generator=generator)

    def analytical(self, inputs: Tensor, grid: Tensor) -> Tensor:
        return gardner_steady_head(inputs[:, 0], inputs[:, 1], grid[0]).unsqueeze(1)

    def residual(self, u: Tensor, inputs: Tensor, grid: Tensor) -> Tensor:
        """Return ``d/dz [K (dh/dz + 1)] / Ks`` at the interior nodes.

        ``K`` on a face is the arithmetic mean of its two nodes. Dividing by ``Ks`` makes the
        residual non-dimensional. ``K / Ks`` lies in ``(0, 1]`` and the flux in ``[-0.9, -0.1]``,
        so no further scaling is needed.
        """
        alpha = inputs[:, 0, None, None]
        conductivity = torch.exp(alpha * u)  # K / Ks
        k_faces = 0.5 * (conductivity[..., 1:] + conductivity[..., :-1])
        return conservative_divergence(u, k_faces, grid_spacing(grid, 0), gravity=1.0)

    def boundary(self, inputs: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
        """``h = h(1) z + z (1 - z) NN``, with ``h(1)`` from the analytical solution."""
        top = torch.ones(1, dtype=inputs.dtype, device=inputs.device)
        head_top = gardner_steady_head(inputs[:, 0], inputs[:, 1], top)
        return dirichlet_1d(grid[0], torch.zeros_like(head_top), head_top)

    def to_fno_input(self, inputs: Tensor, grid: Tensor) -> Tensor:
        z = grid[0]
        batch = inputs.shape[0]
        alpha = inputs[:, 0, None].expand(-1, z.shape[0])
        flux = inputs[:, 1, None].expand(-1, z.shape[0])
        return torch.stack([alpha, flux, z.expand(batch, -1)], dim=1)


@register_problem("darcy_poisson_2d")
class DarcyPoisson2D(Problem):
    """Stationary groundwater flow ``-Laplace h = f`` on ``[0, 1]^2`` with ``h = 0`` on the boundary.

    Constant conductivity ``k = 1``. The source is ``f = sum_{m,n=1..M} a_mn sin(m pi x) sin(n pi y)``
    with ``a_mn ~ U(coeff_low, coeff_high)``. Operator inputs are the coefficients ``(B, M, M)``.
    The network sees ``f``, ``x`` and ``y``.

    Scaling: ``x = x_hat L`` and ``h = h_hat H``. With ``f`` the volumetric source divided by the
    conductivity (unit ``1/m``), ``-Laplace h = f`` becomes ``-Laplace_hat h_hat = f L^2 / H``,
    so the normalized source is ``f_hat = f L^2 / H`` and ``f = f_hat H / L^2``.
    """

    n_dims = 2
    in_channels = 3  # f, x, y
    out_channels = 1
    plot_type = "2d"
    variable_names = {"coordinates": ["x", "y"], "solution": ["h"]}
    units = {"x": "m", "y": "m", "h": "m"}

    def __init__(self, cfg) -> None:
        self.n_series = int(cfg.n_series)
        self.coeff_range = (float(cfg.coeff_low), float(cfg.coeff_high))
        self.scales = {"x": float(cfg.length), "y": float(cfg.length), "h": float(cfg.head_ref)}

    def grid(self, resolution: int | Sequence[int]) -> Tensor:
        return uniform_grid([(0.0, 1.0), (0.0, 1.0)], as_resolution(resolution, self.n_dims))

    def sample_inputs(self, n: int, generator: torch.Generator) -> Tensor:
        low, high = self.coeff_range
        return low + (high - low) * torch.rand(n, self.n_series, self.n_series, generator=generator)

    def analytical(self, inputs: Tensor, grid: Tensor) -> Tensor:
        return poisson_sine_head_2d(inputs, grid[0], grid[1]).unsqueeze(1)

    def source(self, inputs: Tensor, grid: Tensor) -> Tensor:
        """``f`` on the grid, shape ``(B, 1, Nx, Ny)``."""
        return sine_source_2d(inputs, grid[0], grid[1]).unsqueeze(1)

    def residual(self, u: Tensor, inputs: Tensor, grid: Tensor) -> Tensor:
        """Return ``(-Laplace h - f) / ||f||_rms`` on the interior points.

        ``||f||_rms = sqrt(sum a_mn^2) / 2`` is the root mean square of ``f`` over the unit
        square (the sine modes are orthogonal with mean square ``1/4``). The residual is
        therefore an error relative to the size of the source of each sample.
        """
        f = interior(self.source(inputs, grid), [-2, -1])
        size = 0.5 * inputs.pow(2).sum(dim=(1, 2)).sqrt().clamp_min(1e-6)
        return (-laplacian(u, grid) - f) / size[:, None, None, None]

    def boundary(self, inputs: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
        """``h = x (1 - x) y (1 - y) NN`` via ``dirichlet_2d`` with zero edges."""
        x, y = grid[0, :, 0], grid[1, 0, :]
        batch = inputs.shape[0]
        edge_x = inputs.new_zeros(batch, 1, x.shape[0])
        edge_y = inputs.new_zeros(batch, 1, y.shape[0])
        return dirichlet_2d(x, y, edge_x, edge_x, edge_y, edge_y)

    def to_fno_input(self, inputs: Tensor, grid: Tensor) -> Tensor:
        coordinates = grid.expand(inputs.shape[0], -1, -1, -1)
        return torch.cat([self.source(inputs, grid), coordinates], dim=1)
