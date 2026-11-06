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


class Problem(ABC):
    """A benchmark with an analytical solution."""

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
        """Return ``(g, phi)`` of the hard constraint ``u = g + phi * NN``."""

    @abstractmethod
    def to_fno_input(self, inputs: Tensor, grid: Tensor) -> Tensor:
        """Return the network input ``(B, in_channels, *resolution)``, including coordinates."""

    def to_physical(self, variable: str, values):
        """Convert a variable from model units to physical units."""
        return values * self.scales[variable]

    def label(self, variable: str) -> str:
        """Return an axis label such as ``"x [m]"``."""
        return f"{variable} [{self.units[variable]}]"
