"""Euler-Bernoulli beam, simply supported on ``x in [0, 1]``.

Sign convention: the deflection ``w`` is positive in load direction and ``M = -EI w''``,
so that ``M'' = -q``, ``w'' = -M / EI`` and ``EI w'''' = q``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def simply_supported_beam(q0: Tensor, q1: Tensor, stiffness: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    """Deflection and bending moment under the load ``q(x) = q0 + q1 x``, by superposition.

    Args:
        q0: Uniform load, shape ``(B,)``.
        q1: Slope of the triangular load ``q1 x``, shape ``(B,)``.
        stiffness: Bending stiffness ``EI``, shape ``(B,)``.
        x: Positions ``(N,)``.

    Returns:
        ``(w, M)``, each of shape ``(B, N)``.
    """
    q0, q1, stiffness = q0[:, None], q1[:, None], stiffness[:, None]
    w_uniform = q0 * x * (1 - 2 * x**2 + x**3) / (24 * stiffness)
    w_triangular = q1 * x * (7 - 10 * x**2 + 3 * x**4) / (360 * stiffness)
    m_uniform = q0 * x * (1 - x) / 2
    m_triangular = q1 * x * (1 - x**2) / 6
    return w_uniform + w_triangular, m_uniform + m_triangular


def simply_supported_beam_sine(coefficients: Tensor, stiffness: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    """Deflection and bending moment under the load ``q(x) = sum_n q_n sin(n pi x)``.

    Each sine is an eigenfunction of ``d^2/dx^2`` with the hinged supports, so
    ``M = sum q_n sin(n pi x) / (n pi)^2`` and ``w = sum q_n sin(n pi x) / (EI (n pi)^4)``.

    Args:
        coefficients: Load coefficients ``q_n`` of shape ``(B, N)``.
        stiffness: Bending stiffness ``EI``, shape ``(B,)``.
        x: Positions ``(N_x,)``.

    Returns:
        ``(w, M)``, each of shape ``(B, N_x)``.
    """
    n = torch.arange(1, coefficients.shape[-1] + 1, dtype=coefficients.dtype, device=coefficients.device)
    k = n * math.pi
    modes = torch.sin(x[:, None] * k)  # (N_x, N)
    moment = (coefficients / k**2) @ modes.T
    deflection = (coefficients / k**4) @ modes.T / stiffness[:, None]
    return deflection, moment
