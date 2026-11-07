"""Interface between a benchmark and the generic training and evaluation code.

Trainer, losses, evaluation and visualization talk to a benchmark only through this class.

Tensor conventions:
    grid:    ``(n_dims, *resolution)``, coordinates of a tensor-product grid.
    inputs:  ``(B, ...)``, operator inputs in the problem's own parametrization.
    fields:  ``(B, C, *resolution)``, for example solutions and network outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor


#: Output parametrizations, see :meth:`Problem.output_parametrization`.
OUTPUT_PARAMS = ("constrained", "unscaled", "direct", "soft")
#: Output parametrizations without g and phi in the prediction: ``u = c NN``.
UNCONSTRAINED = ("direct", "soft")


class Problem(ABC):
    """A benchmark with an analytical solution.

    Variants: ``inputs`` and ``output_param`` are read from the problem config by
    :meth:`configure_variants`. The defaults, the first entry of ``SUPPORTED_INPUTS`` and
    ``constrained``, are the original behavior of every benchmark.
    """

    #: Version of ``boundary()`` and ``output_scale()``. Increase it by hand whenever g, phi
    #: or the output scaling change, so that old checkpoints are no longer evaluated with
    #: a different wrapper (``scripts/evaluate.py`` refuses them).
    WRAPPER_VERSION: str
    #: Input families a problem implements, named after what is sampled (e.g. ``linear``,
    #: ``sine``). The first one is the original family and the default.
    SUPPORTED_INPUTS: tuple[str, ...]

    inputs: str
    output_param: str = "constrained"
    #: Global output constant ``c`` per channel for ``unscaled`` and ``direct``.
    output_scale_global: tuple[float, ...] | None = None

    #: Registry name, set by ``@register_problem``.
    name: str
    #: Number of grid dimensions, including time.
    n_dims: int
    in_channels: int
    out_channels: int
    #: One of ``"1d"``, ``"1d_t"``, ``"2d"``, ``"2d_t"``; selects the solution plot.
    plot_type: str
    #: ``{"coordinates": [one name per grid dimension], "solution": [one name per channel]}``
    variable_names: dict[str, list[str]]
    #: Physical unit for each variable name.
    units: dict[str, str]
    #: Factor that converts each variable from model units to physical units.
    scales: dict[str, float]

    @property
    def main_variable(self) -> str:
        """Solution variable used for model selection, acceptance and the summary."""
        return self.variable_names["solution"][0]

    @property
    def main_channel(self) -> int:
        return self.variable_names["solution"].index(self.main_variable)

    @abstractmethod
    def grid(self, resolution: int | Sequence[int]) -> Tensor:
        """Return the coordinates of the given resolution, including the boundaries."""

    @abstractmethod
    def sample_inputs(self, n: int, generator: torch.Generator) -> Tensor:
        """Draw ``n`` operator inputs."""

    @abstractmethod
    def analytical(self, inputs: Tensor, grid: Tensor) -> Tensor:
        """Return the exact solution as a field."""

    @abstractmethod
    def residual(self, u: Tensor, inputs: Tensor, grid: Tensor) -> Tensor:
        """Return the normalized PDE residual of ``u`` on the interior points."""

    @abstractmethod
    def boundary(self, inputs: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(g, phi)`` of the hard constraint ``u = g + phi * NN``.

        ``phi`` is only the fixed shape part, e.g. ``x (1 - x)``. A per-sample scaling of the
        output belongs in :meth:`output_scale`.
        """

    def output_scale(self, inputs: Tensor, grid: Tensor) -> Tensor | None:
        """Optional per-sample scale ``s`` of the network output, ``(B, C, 1, ...)``.

        Used only by ``output_param = constrained``. ``None`` (the default) means ``s = 1``.
        """
        return None

    @classmethod
    def default_inputs(cls) -> str:
        """The original input family of the benchmark."""
        return cls.SUPPORTED_INPUTS[0]

    def configure_variants(self, cfg) -> None:
        """Read and validate ``inputs``, ``output_param`` and ``output_scale_global``."""
        inputs = str(cfg.get("inputs") or self.default_inputs())
        output_param = str(cfg.get("output_param", "constrained"))
        if inputs not in self.SUPPORTED_INPUTS:
            raise ValueError(
                f"{self.name} does not implement inputs='{inputs}'. Supported: {self.SUPPORTED_INPUTS}"
            )
        if output_param not in OUTPUT_PARAMS:
            raise ValueError(f"Unknown output_param '{output_param}'. Available: {OUTPUT_PARAMS}")
        scale = cfg.get("output_scale_global")
        if scale is not None:
            scale = tuple(float(value) for value in scale)
            if len(scale) != self.out_channels:
                raise ValueError(
                    f"output_scale_global needs {self.out_channels} entries (one per output), got {scale}"
                )
        self.inputs, self.output_param, self.output_scale_global = inputs, output_param, scale

    def global_output_scale(self, reference: Tensor) -> Tensor:
        """``c`` of shape ``(1, C, 1, ...)``; 1 for every channel if the config sets none."""
        values = self.output_scale_global or (1.0,) * self.out_channels
        c = torch.tensor(values, dtype=reference.dtype, device=reference.device)
        return c.reshape(1, self.out_channels, *(1,) * self.n_dims)

    def output_parametrization(self, inputs: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(g, phi_total)`` with ``u = g + phi_total * NN`` for ``self.output_param``.

        * ``constrained``: ``u = g + phi s NN`` with the per-sample :meth:`output_scale` ``s``
          (or 1). The original behavior.
        * ``unscaled``: ``u = g + phi c NN`` with the global constant ``c``.
        * ``direct``: ``u = c NN``, no boundary values. Only allowed in ``data`` mode.
        * ``soft``: ``u = c NN`` as ``direct``; the boundary and initial values enter the loss
          instead (``L_bc``, see :func:`pino.losses.boundary_loss`). Allowed in every mode.
        """
        if self.output_param in UNCONSTRAINED:
            c = self.global_output_scale(grid)
            return torch.zeros((), dtype=grid.dtype, device=grid.device), c
        g, phi = self.boundary(inputs, grid)
        if self.output_param == "unscaled":
            return g, phi * self.global_output_scale(phi)
        scale = self.output_scale(inputs, grid)
        return g, phi if scale is None else phi * scale

    @property
    def soft_constraint(self) -> bool:
        """Whether boundary and initial values are enforced by the penalty ``L_bc``."""
        return self.output_param == "soft"

    def boundary_values(self, inputs: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(g, mask)``: prescribed values and the points where they are prescribed.

        The points are exactly those where the hard constraint acts, ``phi == 0`` in
        :meth:`boundary`: the Dirichlet boundaries and, for problems in time, the initial time
        slice. ``phi`` is built from products of ``x``, ``1 - x`` and ``tau``, which are exactly
        zero on the boundary nodes of :func:`~pino.numerics.grids.uniform_grid` and nowhere
        else (checked for every problem in ``tests/test_soft.py``). ``mask`` is a bool tensor of
        shape ``(1, 1, *resolution)``, ``g`` has shape ``(B, C, *resolution)``.
        """
        g, phi = self.boundary(inputs, grid)
        mask = (phi == 0).expand(1, 1, *grid.shape[1:])
        return g.expand(inputs.shape[0], self.out_channels, *grid.shape[1:]), mask

    def boundary_scale(self, inputs: Tensor, grid: Tensor) -> Tensor:
        """Normalization ``s`` of ``L_bc``: :meth:`data_scale`, otherwise the global constant ``c``."""
        scale = self.data_scale(inputs, grid)
        return self.global_output_scale(grid) if scale is None else scale

    def check_mode(self, mode: str) -> None:
        """Reject training modes that are ill-posed for the output parametrization."""
        if self.output_param == "direct" and mode != "data":
            raise ValueError(
                f"output_param='direct' is only allowed in mode 'data', not '{mode}': without the "
                "hard constraint and without a boundary loss the PDE alone does not determine the "
                "solution."
            )

    def variant_metadata(self) -> dict[str, str]:
        """Identify the wrapper of a run, stored in ``metrics.json`` and ``history.json``."""
        return {"wrapper_version": self.WRAPPER_VERSION, "inputs": self.inputs, "output_param": self.output_param}

    @abstractmethod
    def to_fno_input(self, inputs: Tensor, grid: Tensor) -> Tensor:
        """Return the network input ``(B, in_channels, *resolution)``, including coordinates."""

    def data_scale(self, inputs: Tensor, grid: Tensor) -> Tensor | None:
        """Typical magnitude of each output channel, ``(B, C, 1, ...)``, for the data loss.

        ``None`` (the default) uses the plain MSE. Problems whose outputs differ in size by
        orders of magnitude return a scale so that all channels count alike.
        """
        return None

    def to_physical(self, variable: str, values):
        """Convert a variable from model units to physical units."""
        return values * self.scales[variable]

    def label(self, variable: str) -> str:
        """Return an axis label such as ``"x [m]"``."""
        return f"{variable} [{self.units[variable]}]"
