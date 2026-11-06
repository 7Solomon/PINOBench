"""Structural mechanics benchmarks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from pino.analytical.beam import simply_supported_beam
from pino.numerics.constraints import dirichlet_1d
from pino.numerics.derivatives import central_d2, interior
from pino.numerics.grids import as_resolution, grid_spacing, uniform_grid
from pino.problems.base import Problem
from pino.problems.registry import register_problem


@register_problem("beam_1d")
class Beam1D(Problem):
    """Simply supported Euler-Bernoulli beam in mixed form: ``M'' = -q`` and ``w'' = -M / EI``.

    Operator inputs are ``(q0, q1, EI)`` with shape ``(B, 3)`` and the load
    ``q(x) = q0 + q1 x``. Outputs are the channels ``w`` and ``M``. All quantities are
    non-dimensional with the reference length ``L``, load ``q_ref`` and stiffness ``EI_ref``.

    Characteristic sizes per sample, from the uniform load with the same mean
    ``q_mean = q0 + q1 / 2``: ``M_c = q_mean / 8`` and ``w_c = 5 q_mean / (384 EI)``.
    They scale the network output (``output_scale``, only for ``output_param = constrained``),
    the data loss (``data_scale``) and the residuals, so that both outputs count alike
    although ``w`` is about ten times smaller than ``M``.
    """

    WRAPPER_VERSION = "1"

    n_dims = 1
    in_channels = 3  # q(x), EI, x
    out_channels = 2  # w, M
    plot_type = "1d"
    variable_names = {"coordinates": ["x"], "solution": ["w", "M"]}
    units = {"x": "m", "w": "mm", "M": "kNm"}

    def __init__(self, cfg) -> None:
        self.q0_range = (float(cfg.q0_low), float(cfg.q0_high))
        self.q1_range = (float(cfg.q1_low), float(cfg.q1_high))
        self.stiffness_range = (float(cfg.stiffness_low), float(cfg.stiffness_high))
        length, load, stiffness = float(cfg.length), float(cfg.load_ref), float(cfg.stiffness_ref)
        self.scales = {
            "x": length,
            "w": 1000.0 * load * length**4 / stiffness,  # m -> mm
            "M": load * length**2,
        }

    def grid(self, resolution: int | Sequence[int]) -> Tensor:
        return uniform_grid([(0.0, 1.0)], as_resolution(resolution, self.n_dims))

    def sample_inputs(self, n: int, generator: torch.Generator) -> Tensor:
        ranges = torch.tensor([self.q0_range, self.q1_range, self.stiffness_range])
        unit = torch.rand(n, 3, generator=generator)
        return ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * unit

    def analytical(self, inputs: Tensor, grid: Tensor) -> Tensor:
        w, m = simply_supported_beam(inputs[:, 0], inputs[:, 1], inputs[:, 2], grid[0])
        return torch.stack([w, m], dim=1)

    def residual(self, u: Tensor, inputs: Tensor, grid: Tensor) -> Tensor:
        """Return ``[(w'' + M/EI) / (M_c/EI), (M'' + q) / q_mean]`` on the interior points.

        Each equation is divided by the size of its source term, so both residuals are
        relative quantities of order one for a wrong solution.
        """
        x, dx = grid[0], grid_spacing(grid, 0)
        stiffness = inputs[:, 2, None]
        q_mean = self._mean_load(inputs)[:, None]
        w, m = u[:, 0], u[:, 1]
        load = self._load(inputs, x)
        curvature = central_d2(w, dx, -1) + interior(m, [-1]) / stiffness
        equilibrium = central_d2(m, dx, -1) + interior(load, [-1])
        return torch.stack([curvature / (q_mean / 8 / stiffness), equilibrium / q_mean], dim=1)

    def boundary(self, inputs: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
        """``g = 0`` and ``phi = x (1 - x)`` for both ``w`` and ``M``."""
        zeros = torch.zeros(inputs.shape[0], self.out_channels, dtype=inputs.dtype, device=inputs.device)
        return dirichlet_1d(grid[0], zeros, zeros)

    def output_scale(self, inputs: Tensor, grid: Tensor) -> Tensor:
        """``[w_c, M_c]`` per sample: ``w = x (1 - x) w_c NN_w``, ``M = x (1 - x) M_c NN_M``.

        Brings the network output to order one. Only used by ``output_param = constrained``.
        """
        return self.characteristic_sizes(inputs)

    def data_scale(self, inputs: Tensor, grid: Tensor) -> Tensor:
        """Loss weighting with the same ``[w_c, M_c]``; acts on the data loss only."""
        return self.characteristic_sizes(inputs)

    def characteristic_sizes(self, inputs: Tensor) -> Tensor:
        """``[w_c, M_c] = [5 q_mean / (384 EI), q_mean / 8]``, shape ``(B, 2, 1)``."""
        q_mean = self._mean_load(inputs)
        w_c = 5 * q_mean / (384 * inputs[:, 2])
        return torch.stack([w_c, q_mean / 8], dim=1)[..., None]

    def to_fno_input(self, inputs: Tensor, grid: Tensor) -> Tensor:
        x = grid[0]
        batch = inputs.shape[0]
        load = self._load(inputs, x)
        stiffness = inputs[:, 2, None].expand(-1, x.shape[0])
        return torch.stack([load, stiffness, x.expand(batch, -1)], dim=1)

    @staticmethod
    def _load(inputs: Tensor, x: Tensor) -> Tensor:
        return inputs[:, 0, None] + inputs[:, 1, None] * x

    @staticmethod
    def _mean_load(inputs: Tensor) -> Tensor:
        # Guards against vanishing scales for the rare samples with almost no load.
        return (inputs[:, 0] + 0.5 * inputs[:, 1]).clamp_min(1e-2)
