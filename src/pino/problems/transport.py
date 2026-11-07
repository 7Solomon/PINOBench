"""Transport benchmarks in porous building materials."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from pino.analytical.diffusion import chloride_corner, crank_erfc
from pino.numerics.constraints import dirichlet_1d_t, dirichlet_2d_t
from pino.numerics.grids import as_resolution, uniform_grid
from pino.pdes.diffusion import diffusion_residual
from pino.problems.base import Problem
from pino.problems.registry import register_problem

SECONDS_PER_YEAR = 365.25 * 24 * 3600


@register_problem("chloride_1d")
class Chloride1D(Problem):
    """Chloride ingress ``dC/dt = D d2C/dx2`` into concrete, solved by Crank's erfc solution.

    ``C = Cs erfc(x / (2 sqrt(D t)))`` on ``x in [0, 1]`` and ``t in [t0, t_end]``. The domain
    starts at ``t0 > 0`` because the erfc profile is a step at ``t = 0``. Prescribed are
    ``C(0, t) = Cs``, ``C(1, t)`` and the profile ``C(x, t0)``, all from the analytical solution.

    Operator inputs are the scalars ``(Cs, D)`` with shape ``(B, 2)``. The network sees ``Cs``
    and ``D`` broadcast over the grid, plus ``x`` and ``tau = (t - t0) / (t_end - t0)``.

    Non-dimensionalization: ``x = x_hat L``, ``D = D_hat D_ref``, ``t = t_hat L^2 / D_ref``.
    """

    WRAPPER_VERSION = "1"
    SUPPORTED_INPUTS = ("uniform",)  # Cs, D uniform

    n_dims = 2
    in_channels = 4  # Cs, D, x, tau
    out_channels = 1
    plot_type = "1d_t"
    variable_names = {"coordinates": ["x", "t"], "solution": ["C"]}
    units = {"x": "mm", "t": "a", "C": "M.-%/B"}

    def __init__(self, cfg) -> None:
        self.surface_range = (float(cfg.surface_low), float(cfg.surface_high))
        self.diffusivity_range = (float(cfg.diffusivity_low), float(cfg.diffusivity_high))
        self.t_start = float(cfg.t_start)
        self.t_end = float(cfg.t_end)
        length_m = float(cfg.length_mm) / 1000.0
        time_ref_years = length_m**2 / float(cfg.diffusivity_ref) / SECONDS_PER_YEAR
        self.scales = {"x": float(cfg.length_mm), "t": time_ref_years, "C": 1.0}

    def grid(self, resolution: int | Sequence[int]) -> Tensor:
        return uniform_grid([(0.0, 1.0), (self.t_start, self.t_end)], as_resolution(resolution, self.n_dims))

    def sample_inputs(self, n: int, generator: torch.Generator) -> Tensor:
        ranges = torch.tensor([self.surface_range, self.diffusivity_range])
        return ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * torch.rand(n, 2, generator=generator)

    def analytical(self, inputs: Tensor, grid: Tensor) -> Tensor:
        return crank_erfc(inputs[:, 0], inputs[:, 1], grid[0], grid[1]).unsqueeze(1)

    def residual(self, u: Tensor, inputs: Tensor, grid: Tensor) -> Tensor:
        """Return ``(dC/dt - D d2C/dx2) / (Cs / t0)`` on the interior points.

        Normalization per sample: for the erfc profile ``dC/dt`` peaks at
        ``Cs / (t sqrt(2 pi e)) ~ 0.24 Cs / t`` (at ``x = sqrt(2 D t)``), independent of ``D``.
        Both terms of the residual are therefore at most ``0.24 Cs / t0`` for every sample, and
        dividing by ``Cs / t0`` makes the residual of all samples comparable, whatever ``D``.
        Dividing by ``D`` instead would weight samples with small ``D`` up to 5 times more.
        """
        surface, diffusivity = (inputs[:, i, None, None, None] for i in range(2))
        return diffusion_residual(u, grid, diffusivity) * (self.t_start / surface)

    def boundary(self, inputs: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
        """``dirichlet_1d_t`` with ``C(x, t0)``, ``C(0, t) = Cs`` and ``C(1, t)`` from the solution."""
        x, t = grid[0, :, 0], grid[1, 0, :]
        surface, diffusivity = inputs[:, 0], inputs[:, 1]
        initial = crank_erfc(surface, diffusivity, x, t[:1]).unsqueeze(1)
        left = surface[:, None, None].expand(-1, 1, t.shape[0])
        right = crank_erfc(surface, diffusivity, torch.ones_like(t), t).unsqueeze(1)
        return dirichlet_1d_t(x, t, initial, left, right)

    def to_fno_input(self, inputs: Tensor, grid: Tensor) -> Tensor:
        batch = inputs.shape[0]
        tau = (grid[1] - self.t_start) / (self.t_end - self.t_start)
        coordinates = torch.stack([grid[0], tau]).expand(batch, -1, -1, -1)
        scalars = inputs[:, :, None, None].expand(-1, -1, *grid.shape[1:])
        return torch.cat([scalars, coordinates], dim=1)


@register_problem("chloride_corner_2d")
class ChlorideCorner2D(Problem):
    """Chloride ingress into a component corner with the exposed faces ``x = 0`` and ``y = 0``.

    ``dC/dt = D (d2C/dx2 + d2C/dy2)`` on ``[0, 1]^2`` and ``t in [t0, t_end]``, solved by
    ``C = Cs [1 - erf(x / (2 sqrt(D t))) erf(y / (2 sqrt(D t)))]`` (``C0 = 0``). Prescribed are
    ``C = Cs`` on ``x = 0`` and ``y = 0``, the analytical values on ``x = 1`` and ``y = 1`` (not
    zero, e.g. ``C(1, y) ~ Cs erfc(y / (2 sqrt(D t)))``) and the profile at ``t0``.

    Operator inputs are the scalars ``(Cs, D)`` with shape ``(B, 2)``. The network sees ``Cs``
    and ``D`` broadcast over the grid, plus ``x``, ``y`` and ``tau = (t - t0) / (t_end - t0)``.

    Non-dimensionalization as in chloride_1d: ``x = x_hat L``, ``D = D_hat D_ref``,
    ``t = t_hat L^2 / D_ref``.
    """

    WRAPPER_VERSION = "1"
    SUPPORTED_INPUTS = ("uniform",)  # Cs, D uniform

    n_dims = 3
    in_channels = 5  # Cs, D, x, y, tau
    out_channels = 1
    plot_type = "2d_t"
    variable_names = {"coordinates": ["x", "y", "t"], "solution": ["C"]}
    units = {"x": "mm", "y": "mm", "t": "a", "C": "M.-%/B"}

    def __init__(self, cfg) -> None:
        self.surface_range = (float(cfg.surface_low), float(cfg.surface_high))
        self.diffusivity_range = (float(cfg.diffusivity_low), float(cfg.diffusivity_high))
        self.t_start = float(cfg.t_start)
        self.t_end = float(cfg.t_end)
        length_m = float(cfg.length_mm) / 1000.0
        time_ref_years = length_m**2 / float(cfg.diffusivity_ref) / SECONDS_PER_YEAR
        length = float(cfg.length_mm)
        self.scales = {"x": length, "y": length, "t": time_ref_years, "C": 1.0}

    def grid(self, resolution: int | Sequence[int]) -> Tensor:
        bounds = [(0.0, 1.0), (0.0, 1.0), (self.t_start, self.t_end)]
        return uniform_grid(bounds, as_resolution(resolution, self.n_dims))

    def sample_inputs(self, n: int, generator: torch.Generator) -> Tensor:
        ranges = torch.tensor([self.surface_range, self.diffusivity_range])
        return ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * torch.rand(n, 2, generator=generator)

    def analytical(self, inputs: Tensor, grid: Tensor) -> Tensor:
        return chloride_corner(inputs[:, 0], inputs[:, 1], grid[0], grid[1], grid[2]).unsqueeze(1)

    def residual(self, u: Tensor, inputs: Tensor, grid: Tensor) -> Tensor:
        """Return ``(dC/dt - D Laplace C) / (Cs / t0)`` on the interior points.

        Normalization per sample as in chloride_1d: away from the corner the solution is the 1D
        erfc profile, whose ``dC/dt`` peaks at ``~0.24 Cs / t`` independently of ``D``; near the
        corner both faces contribute, at most twice that. Dividing by ``Cs / t0`` makes the
        residual of all samples comparable, whatever ``D``.
        """
        surface, diffusivity = (inputs[:, i, None, None, None, None] for i in range(2))
        return diffusion_residual(u, grid, diffusivity) * (self.t_start / surface)

    def boundary(self, inputs: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
        """``dirichlet_2d_t`` with the analytical profile at ``t0`` and analytical edge values."""
        x, y, t = grid[0, :, 0, 0], grid[1, 0, :, 0], grid[2, 0, 0, :]
        surface, diffusivity = inputs[:, 0], inputs[:, 1]

        def solution(xs, ys, ts):
            return chloride_corner(surface, diffusivity, xs, ys, ts).unsqueeze(1)

        initial = solution(x[:, None], y[None, :], t[0])
        zero_x, one_x = torch.zeros_like(x)[:, None], torch.ones_like(x)[:, None]
        zero_y, one_y = torch.zeros_like(y)[:, None], torch.ones_like(y)[:, None]
        bottom, top = solution(x[:, None], zero_x, t[None, :]), solution(x[:, None], one_x, t[None, :])
        left, right = solution(zero_y, y[:, None], t[None, :]), solution(one_y, y[:, None], t[None, :])
        return dirichlet_2d_t(x, y, t, initial, bottom, top, left, right)

    def to_fno_input(self, inputs: Tensor, grid: Tensor) -> Tensor:
        batch = inputs.shape[0]
        tau = (grid[2] - self.t_start) / (self.t_end - self.t_start)
        coordinates = torch.stack([grid[0], grid[1], tau]).expand(batch, -1, -1, -1, -1)
        scalars = inputs[:, :, None, None, None].expand(-1, -1, *grid.shape[1:])
        return torch.cat([scalars, coordinates], dim=1)
