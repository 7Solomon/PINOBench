"""Build a neuraloperator FNO from a config."""

from __future__ import annotations

from neuralop.models import FNO
from torch import nn

from pino.numerics.grids import as_resolution
from pino.problems.base import Problem


def build_model(cfg, problem: Problem) -> FNO:
    """Create an FNO whose dimensionality and channels follow the problem.

    ``positional_embedding`` is disabled on purpose. neuraloperator's default appends grid
    channels of its own, which are periodic (they exclude the right boundary) and would
    duplicate the coordinates that ``Problem.to_fno_input`` already provides.
    """
    return FNO(
        n_modes=as_resolution(cfg.n_modes, problem.n_dims),
        in_channels=problem.in_channels,
        out_channels=problem.out_channels,
        hidden_channels=int(cfg.hidden_channels),
        n_layers=int(cfg.n_layers),
        domain_padding=float(cfg.domain_padding),
        positional_embedding=None,
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters, with complex weights counted as two real numbers."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad)
