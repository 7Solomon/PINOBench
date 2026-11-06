"""Analytical solutions of the diffusion equation."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _wavenumbers(coefficients: Tensor) -> Tensor:
    """Return n * pi for n = 1..N, where N is the last dimension of ``coefficients``."""
    n = torch.arange(1, coefficients.shape[-1] + 1, dtype=coefficients.dtype, device=coefficients.device)
    return n * math.pi


def sine_series(coefficients: Tensor, x: Tensor) -> Tensor:
    """Evaluate ``sum_n b_n sin(n pi x)``.

    Args:
        coefficients: Series coefficients ``b_n`` of shape ``(B, N)``.
        x: Evaluation points of arbitrary shape ``S``.

    Returns:
        Values of shape ``(B, *S)``.
    """
    return heat_sine_series(coefficients, x, torch.zeros_like(x), diffusivity=0.0)


def heat_sine_series(coefficients: Tensor, x: Tensor, t: Tensor, diffusivity: float) -> Tensor:
    """Evaluate ``T(x, t) = sum_n b_n sin(n pi x) exp(-a (n pi)^2 t)``.

    This solves ``dT/dt = a d2T/dx2`` on ``x in [0, 1]`` with homogeneous Dirichlet
    boundaries and the initial profile ``T(x, 0) = sum_n b_n sin(n pi x)``.

    Args:
        coefficients: Series coefficients ``b_n`` of shape ``(B, N)``.
        x: Spatial points of shape ``S``.
        t: Times of the same shape ``S``.
        diffusivity: Diffusion coefficient ``a``.

    Returns:
        Values of shape ``(B, *S)``.
    """
    k = _wavenumbers(coefficients)
    # Evaluation points get a trailing mode dimension: S -> (*S, N).
    modes = torch.sin(x.unsqueeze(-1) * k) * torch.exp(-diffusivity * k**2 * t.unsqueeze(-1))
    # (B, N) contracted with (*S, N) -> (B, *S).
    return torch.tensordot(coefficients, modes, dims=([-1], [-1]))
