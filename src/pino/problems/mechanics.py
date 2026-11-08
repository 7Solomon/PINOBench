"""Structural mechanics benchmarks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from pino.analytical.beam import simply_supported_beam, simply_supported_beam_sine
from pino.analytical.elasticity import cosine_series, elastic_block
from pino.analytical.diffusion import sine_series
from pino.numerics.constraints import dirichlet_1d
from pino.numerics.derivatives import central_d2, interior
from pino.numerics.grids import as_resolution, grid_spacing, uniform_grid
from pino.pdes.elasticity import navier_lame_residual, plane_stress
from pino.problems.base import NeumannCondition, Problem
from pino.problems.registry import register_problem


@register_problem("beam_1d")
class Beam1D(Problem):
    """Simply supported Euler-Bernoulli beam in mixed form: ``M'' = -q`` and ``w'' = -M / EI``.

    Operator inputs are ``(q0, q1, EI)`` with shape ``(B, 3)`` and the load
    ``q(x) = q0 + q1 x``. Outputs are the channels ``w`` and ``M``. All quantities are
    non-dimensional with the reference length ``L``, load ``q_ref`` and stiffness ``EI_ref``.

    Characteristic sizes per sample, from the uniform load with the same mean
    ``q_mean = q0 + q1 / 2``: ``M_c = q_mean / 8`` and ``w_c = 5 q_mean / (384 EI)``.
    These sizes belong to ``inputs = linear``, the default. ``inputs = sine`` replaces the linear
    load by a sine series ``q = sum_{n=1..N} q_n sin(n pi x)`` with
    ``q_n ~ U(coeff_low, coeff_high) / n^coeff_decay``. Operator inputs are then
    ``(q_1, ..., q_N, EI)`` with shape ``(B, N + 1)``. The network channels stay ``q(x), EI, x``.
    The characteristic sizes use the load RMS ``q_rms = sqrt(sum q_n^2 / 2)`` instead:
    ``M_c = q_rms / pi^2`` and ``w_c = q_rms / (pi^4 EI)``, the amplitudes of a single first mode.

    They scale the network output (``output_scale``, only for ``output_param = constrained``),
    the data loss (``data_scale``) and the residuals, so that both outputs count alike
    although ``w`` is about ten times smaller than ``M``.
    """

    WRAPPER_VERSION = "1"
    SUPPORTED_INPUTS = ("linear", "sine")

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
        # Sine-series load of inputs = sine.
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
        if self.inputs == "sine":
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
        if self.inputs == "sine":
            w, m = simply_supported_beam_sine(inputs[:, :-1], inputs[:, -1], grid[0])
        else:
            w, m = simply_supported_beam(inputs[:, 0], inputs[:, 1], inputs[:, 2], grid[0])
        return torch.stack([w, m], dim=1)

    def residual(self, u: Tensor, inputs: Tensor, grid: Tensor) -> Tensor:
        """Return ``[(w'' + M/EI) / (M_c/EI), (M'' + q) / q_ref]`` on the interior points.

        Each equation is divided by the size of its source term, so both residuals are
        relative quantities of order one for a wrong solution. ``q_ref`` is ``q_mean`` for
        ``linear`` and ``q_rms`` for ``sine``; ``M_c`` as in the class docstring. These are
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
        """``[w_c, M_c]``, shape ``(B, 2, 1)``: ``linear`` from ``q_mean``, ``sine`` from ``q_rms``."""
        if self.inputs == "sine":
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
        if self.inputs == "sine":
            return sine_series(inputs[:, :-1], x)
        return inputs[:, 0, None] + inputs[:, 1, None] * x

    def _residual_scales(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """``(M_c, q_ref)`` per sample, shape ``(B, 1)`` each."""
        if self.inputs == "sine":
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


@register_problem("elastic_block_2d")
class ElasticBlock2D(Problem):
    """Elastic block in a frictionless box, loaded from above; plane stress, ``E = 1``.

    Domain ``[0, 1] x [0, H]``, ``x`` horizontal and ``y`` upwards. Sliding supports on three
    sides: ``u = 0``, ``sigma_xy = 0`` on ``x = 0, 1`` and ``v = 0``, ``sigma_xy = 0`` on ``y = 0``.
    On top ``sigma_yy = -p(x)``, ``sigma_xy = 0`` with ``p = p_0 + sum_n p_n cos(n pi x)``.

    Operator inputs are ``(p_0, p_1, ..., p_N, nu)`` with shape ``(B, N + 2)``. The network sees
    ``p(x)`` and ``nu`` broadcast over the grid, plus ``x`` and ``y``; it predicts ``u, v``.

    The load enters only through the top edge: ``u = v = 0`` has zero residual, so ``L_neumann``
    is what makes the problem non-trivial. Hard constraint per channel, with the global
    constants ``c`` of the YAML: ``u = x (1 - x) c_u NN_u``, ``v = y c_v NN_v``. Residual and
    ``L_neumann`` are divided by the load RMS per sample, the data loss by ``c``, so that ``u``
    and ``v`` count alike although ``u`` is much smaller.

    Non-dimensionalization: ``x = x_hat L``, ``sigma = sigma_hat p_ref``, ``u = u_hat p_ref L / E``.
    """

    WRAPPER_VERSION = "1"
    SUPPORTED_INPUTS = ("cosine",)  # p(x) = cosine series, nu uniform

    n_dims = 2
    in_channels = 4  # p(x), nu, x, y
    out_channels = 2  # u, v
    plot_type = "2d"
    mechanics_plots = True
    variable_names = {"coordinates": ["x", "y"], "solution": ["u", "v"]}
    units = {"x": "m", "y": "m", "u": "mm", "v": "mm", "sigma": "MPa"}
    #: Displacement unit in the unit of the coordinates: mm in m, for the deformed grid.
    displacement_in_length_unit = 1e-3

    def __init__(self, cfg) -> None:
        self.n_series = int(cfg.n_series)
        self.mean_load_range = (float(cfg.mean_load_low), float(cfg.mean_load_high))
        self.coeff_range = (float(cfg.coeff_low), float(cfg.coeff_high))
        self.poisson_range = (float(cfg.poisson_low), float(cfg.poisson_high))
        self.height = float(cfg.height)
        length, modulus, load = float(cfg.length), float(cfg.youngs_modulus_gpa), float(cfg.load_ref_mpa)
        displacement = length * load / modulus  # m * MPa / GPa = mm
        self.scales = {"x": length, "y": length, "u": displacement, "v": displacement, "sigma": load}

    @property
    def main_variable(self) -> str:
        return "v"

    def grid(self, resolution: int | Sequence[int]) -> Tensor:
        return uniform_grid([(0.0, 1.0), (0.0, self.height)], as_resolution(resolution, self.n_dims))

    def sample_inputs(self, n: int, generator: torch.Generator) -> Tensor:
        low, high = self.mean_load_range
        mean = low + (high - low) * torch.rand(n, 1, generator=generator)
        low, high = self.coeff_range
        decay = torch.arange(1, self.n_series + 1, dtype=torch.get_default_dtype())
        coefficients = (low + (high - low) * torch.rand(n, self.n_series, generator=generator)) / decay
        low, high = self.poisson_range
        poisson = low + (high - low) * torch.rand(n, 1, generator=generator)
        return torch.cat([mean, coefficients, poisson], dim=1)

    def _solution(self, inputs: Tensor, grid: Tensor) -> dict[str, Tensor]:
        return elastic_block(inputs[:, 0], inputs[:, 1:-1], inputs[:, -1], grid[0], grid[1], 1.0, self.height)

    def analytical(self, inputs: Tensor, grid: Tensor) -> Tensor:
        solution = self._solution(inputs, grid)
        return torch.stack([solution["u"], solution["v"]], dim=1)

    def load(self, inputs: Tensor, x: Tensor) -> Tensor:
        """``p`` at ``x``, shape ``(B, *x.shape)``."""
        extra = (1,) * x.dim()
        return inputs[:, 0].reshape(-1, *extra) + cosine_series(inputs[:, 1:-1], x)

    def _load_rms(self, inputs: Tensor) -> Tensor:
        """RMS of ``p`` over the width, ``sqrt(p_0^2 + sum p_n^2 / 2)``, shape ``(B,)``."""
        return (inputs[:, 0].pow(2) + 0.5 * inputs[:, 1:-1].pow(2).sum(dim=1)).sqrt()

    def residual(self, u: Tensor, inputs: Tensor, grid: Tensor) -> Tensor:
        """Navier-Lame in displacement form divided by the load RMS per sample, on interior points.

        The residual of a wrong field is then of the order of the relative load error, whatever
        the size of the load. A normalization of the loss only.
        """
        nu = inputs[:, -1, None, None]
        first, second = navier_lame_residual(u[:, 0], u[:, 1], grid_spacing(grid, 0), grid_spacing(grid, 1), 1.0, nu)
        return torch.stack([first, second], dim=1) / self._load_rms(inputs)[:, None, None, None]

    def boundary(self, inputs: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
        """``g = 0``, ``phi_u = x (1 - x)``, ``phi_v = y``: each output has its own supports."""
        x, y = grid[0], grid[1]
        phi = torch.stack([x * (1.0 - x), y])[None]
        return inputs.new_zeros(inputs.shape[0], 2, *grid.shape[1:]), phi

    def output_scale(self, inputs: Tensor, grid: Tensor) -> Tensor:
        """The global constants ``c_u, c_v``: no per-sample scale."""
        return self.global_output_scale(grid)

    def data_scale(self, inputs: Tensor, grid: Tensor) -> Tensor:
        """The same constants, so that ``u`` and ``v`` count alike in the data loss and ``L_bc``."""
        return self.global_output_scale(grid)

    def neumann_conditions(self, inputs: Tensor, grid: Tensor) -> list[NeumannCondition]:
        """Zero shear on all four edges, the load on top."""
        return [
            NeumannCondition("left", "sigma_xy"),
            NeumannCondition("right", "sigma_xy"),
            NeumannCondition("bottom", "sigma_xy"),
            NeumannCondition("top", "sigma_xy"),
            NeumannCondition("top", "sigma_yy", -self.load(inputs, grid[0, :, 0])),
        ]

    def neumann_fields(self, u: Tensor, inputs: Tensor, grid: Tensor) -> dict[str, Tensor]:
        return self.stress_fields(u, inputs, grid)

    def neumann_scale(self, inputs: Tensor, grid: Tensor) -> Tensor:
        return self._load_rms(inputs)

    def stress_fields(self, u: Tensor, inputs: Tensor, grid: Tensor) -> dict[str, Tensor]:
        nu = inputs[:, -1, None, None]
        return plane_stress(u[:, 0], u[:, 1], grid_spacing(grid, 0), grid_spacing(grid, 1), 1.0, nu)

    def exact_stress_fields(self, inputs: Tensor, grid: Tensor) -> dict[str, Tensor]:
        solution = self._solution(inputs, grid)
        return {name: solution[name] for name in ("sigma_xx", "sigma_yy", "sigma_xy")}

    def to_fno_input(self, inputs: Tensor, grid: Tensor) -> Tensor:
        x, y = grid[0], grid[1]
        batch = inputs.shape[0]
        poisson = inputs[:, -1, None, None].expand(-1, *x.shape)
        return torch.stack([self.load(inputs, x), poisson, x.expand(batch, -1, -1), y.expand(batch, -1, -1)], dim=1)
