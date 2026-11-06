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


def crank_erfc(surface: Tensor, diffusivity: Tensor, x: Tensor, t: Tensor, initial: float = 0.0) -> Tensor:
    """Evaluate ``C(x, t) = C0 + (Cs - C0) erfc(x / (2 sqrt(D t)))`` (Crank).

    This solves ``dC/dt = D d2C/dx2`` on the half-space ``x >= 0`` with ``C(0, t) = Cs`` and
    ``C(x, 0) = C0``.

    Args:
        surface: Surface concentration ``Cs`` of shape ``(B,)``.
        diffusivity: Diffusion coefficient ``D`` of shape ``(B,)``.
        x: Depths of arbitrary shape ``S``.
        t: Times ``t > 0``, broadcastable against ``x``.
        initial: Initial concentration ``C0``.

    Returns:
        Values of shape ``(B, *S)``.
    """
    x, t = torch.broadcast_tensors(x, t)
    extra = (1,) * x.dim()
    surface = surface.reshape(-1, *extra)
    diffusivity = diffusivity.reshape(-1, *extra)
    return initial + (surface - initial) * torch.erfc(x / (2.0 * torch.sqrt(diffusivity * t)))
