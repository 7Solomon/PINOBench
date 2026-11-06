"""Build a neuraloperator FNO from a config."""

from __future__ import annotations

from collections.abc import Sequence

from neuralop.models import FNO
from torch import nn

from pino.numerics.grids import as_resolution
from pino.problems.base import Problem


def build_model(cfg, problem: Problem, resolution: int | Sequence[int]) -> FNO:
    """Create an FNO whose dimensionality and channels follow the problem.

    Args:
        cfg: Model config. ``n_modes`` is a scalar or one entry per grid dimension.
        problem: Supplies the number of dimensions and channels.
        resolution: Training resolution, used to validate ``n_modes``.

    ``positional_embedding`` is disabled on purpose. neuraloperator's default appends grid
    channels of its own, which are periodic (they exclude the right boundary) and would
    duplicate the coordinates that ``Problem.to_fno_input`` already provides.
    """
    n_modes = as_resolution(cfg.n_modes, problem.n_dims)
    resolution = as_resolution(resolution, problem.n_dims)
    for modes, points in zip(n_modes, resolution):
        if modes > points // 2:
            raise ValueError(
                f"n_modes={modes} exceeds half the training resolution {points}. "
                f"Use at most {points // 2} modes or a finer grid."
            )
    model = FNO(
        n_modes=n_modes,
        in_channels=problem.in_channels,
        out_channels=problem.out_channels,
        hidden_channels=int(cfg.hidden_channels),
        n_layers=int(cfg.n_layers),
        domain_padding=float(cfg.domain_padding),
        positional_embedding=None,
    )
    model.register_forward_pre_hook(_reject_odd_grid)
    return model


def _reject_odd_grid(module: nn.Module, args: tuple) -> None:
    """Refuse inputs on which neuraloperator 2.0.0 silently returns wrong results.

    ``SpectralConv`` undoes its ``fftshift`` with another ``fftshift`` instead of an
    ``ifftshift``. For an odd length this shifts every frequency by one bin in all FFT
    dimensions except the last one, which uses a real FFT and is unaffected. Domain padding
    adds the same number of points on both sides, so the parity of the grid decides.
    """
    shape = args[0].shape[2:]
    odd = [(dim, n) for dim, n in enumerate(shape[:-1]) if n % 2]
    if odd:
        raise ValueError(
            f"Grid {tuple(shape)} has an odd number of points in dimension(s) "
            f"{[dim for dim, _ in odd]}. neuraloperator 2.0.0 computes wrong spectral "
            "convolutions there. Use an even number of points in every dimension except the last."
        )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters, with complex weights counted as two real numbers."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad)
