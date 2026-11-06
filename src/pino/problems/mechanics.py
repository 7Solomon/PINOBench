"""Structural mechanics benchmarks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from pino.analytical.beam import simply_supported_beam, simply_supported_beam_sine
from pino.analytical.diffusion import sine_series
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
    ``inputs = rich`` replaces the linear load by a sine series ``q = sum_{n=1..N} q_n sin(n pi x)``
    with ``q_n ~ U(coeff_low, coeff_high) / n^coeff_decay``. Operator inputs are then
    ``(q_1, ..., q_N, EI)`` with shape ``(B, N + 1)``. The network channels stay ``q(x), EI, x``.
    The characteristic sizes use the load RMS ``q_rms = sqrt(sum q_n^2 / 2)`` instead:
    ``M_c = q_rms / pi^2`` and ``w_c = q_rms / (pi^4 EI)``, the amplitudes of a single first mode.

    They scale the network output (``output_scale``, only for ``output_param = constrained``),
    the data loss (``data_scale``) and the residuals, so that both outputs count alike
    although ``w`` is about ten times smaller than ``M``.
    """

    WRAPPER_VERSION = "1"
    SUPPORTED_INPUTS = ("simple", "rich")

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
        # Sine-series load of inputs = rich.
        self.n_series = int(cfg.get("n_series", 8))
        self.coeff_range = (float(cfg.get("coeff_low", -1.0)), float(cfg.get("coeff_high", 1.0)))
        self.coeff_decay = float(cfg.get("coeff_decay", 1.0))
        length, load, stiffness = float(cfg.length), float(cfg.load_ref), float(cfg.stiffness_ref)
        self.scales = {
            "x": length,
            "w": 1000.0 * load * length**4 / stiffness,  # m -> mm
            "M": load * length**2,
        }

    def grid(self, resolution: int | Sequence[int]) -> Tensor:
        return uniform_grid([(0.0, 1.0)], as_resolution(resolution, self.n_dims))

    def sample_inputs(self, n: int, generator: torch.Generator) -> Tensor:
        if self.inputs == "rich":
            low, high = self.coeff_range
            decay = torch.arange(1, self.n_series + 1, dtype=torch.get_default_dtype()) ** self.coeff_decay
            coefficients = (low + (high - low) * torch.rand(n, self.n_series, generator=generator)) / decay
            stiffness_low, stiffness_high = self.stiffness_range
            stiffness = stiffness_low + (stiffness_high - stiffness_low) * torch.rand(n, 1, generator=generator)
            return torch.cat([coefficients, stiffness], dim=1)
        ranges = torch.tensor([self.q0_range, self.q1_range, self.stiffness_range])
        unit = torch.rand(n, 3, generator=generator)
        return ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * unit

    def analytical(self, inputs: Tensor, grid: Tensor) -> Tensor:
        if self.inputs == "rich":
            w, m = simply_supported_beam_sine(inputs[:, :-1], inputs[:, -1], grid[0])
        else:
            w, m = simply_supported_beam(inputs[:, 0], inputs[:, 1], inputs[:, 2], grid[0])
        return torch.stack([w, m], dim=1)

    def residual(self, u: Tensor, inputs: Tensor, grid: Tensor) -> Tensor:
        """Return ``[(w'' + M/EI) / (M_c/EI), (M'' + q) / q_ref]`` on the interior points.

        Each equation is divided by the size of its source term, so both residuals are
        relative quantities of order one for a wrong solution. ``q_ref`` is ``q_mean`` for
        ``simple`` and ``q_rms`` for ``rich``; ``M_c`` as in the class docstring. These are
        per-sample loss normalizations only.
        """
        x, dx = grid[0], grid_spacing(grid, 0)
        stiffness = inputs[:, -1, None]
        moment_scale, load_scale = self._residual_scales(inputs)
        w, m = u[:, 0], u[:, 1]
        load = self._load(inputs, x)
        curvature = central_d2(w, dx, -1) + interior(m, [-1]) / stiffness
        equilibrium = central_d2(m, dx, -1) + interior(load, [-1])
        return torch.stack([curvature / (moment_scale / stiffness), equilibrium / load_scale], dim=1)

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
        """``[w_c, M_c]``, shape ``(B, 2, 1)``: ``simple`` from ``q_mean``, ``rich`` from ``q_rms``."""
        if self.inputs == "rich":
            q_rms = self._load_rms(inputs)
            return torch.stack([q_rms / (math.pi**4 * inputs[:, -1]), q_rms / math.pi**2], dim=1)[..., None]
        q_mean = self._mean_load(inputs)
        w_c = 5 * q_mean / (384 * inputs[:, 2])
        return torch.stack([w_c, q_mean / 8], dim=1)[..., None]

    def to_fno_input(self, inputs: Tensor, grid: Tensor) -> Tensor:
        x = grid[0]
        batch = inputs.shape[0]
        load = self._load(inputs, x)
        stiffness = inputs[:, -1, None].expand(-1, x.shape[0])
        return torch.stack([load, stiffness, x.expand(batch, -1)], dim=1)

    def _load(self, inputs: Tensor, x: Tensor) -> Tensor:
        if self.inputs == "rich":
            return sine_series(inputs[:, :-1], x)
        return inputs[:, 0, None] + inputs[:, 1, None] * x

    def _residual_scales(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """``(M_c, q_ref)`` per sample, shape ``(B, 1)`` each."""
        if self.inputs == "rich":
            q_rms = self._load_rms(inputs)[:, None]
            return q_rms / math.pi**2, q_rms
        q_mean = self._mean_load(inputs)[:, None]
        return q_mean / 8, q_mean

    @staticmethod
    def _mean_load(inputs: Tensor) -> Tensor:
        # Guards against vanishing scales for the rare samples with almost no load.
        return (inputs[:, 0] + 0.5 * inputs[:, 1]).clamp_min(1e-2)

    @staticmethod
    def _load_rms(inputs: Tensor) -> Tensor:
        """RMS of the sine-series load over ``[0, 1]``, ``sqrt(sum q_n^2 / 2)``, at least 1e-3."""
        return (0.5 * inputs[:, :-1].pow(2).sum(dim=1)).sqrt().clamp_min(1e-3)
