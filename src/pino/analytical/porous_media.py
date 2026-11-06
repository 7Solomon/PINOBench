"""Stationary flow in porous media: layered Darcy flow and Richards' equation (Gardner model)."""

from __future__ import annotations

import torch
from torch import Tensor


def layered_darcy_flux(k1: Tensor, k2: Tensor, interface: Tensor) -> Tensor:
    """Flux ``q = 1 / (x_s / k1 + (1 - x_s) / k2)`` for ``h(0) = 1`` and ``h(1) = 0``."""
    return 1.0 / (interface / k1 + (1.0 - interface) / k2)


def layered_darcy_head(k1: Tensor, k2: Tensor, interface: Tensor, x: Tensor) -> Tensor:
    """Head of ``-(k h')' = 0`` on ``x in [0, 1]`` with ``k = k1`` for ``x < x_s`` and ``k2`` after.

    Args:
        k1, k2, interface: Shape ``(B,)``.
        x: Positions ``(N,)``.

    Returns:
        Piecewise linear head ``(B, N)``, continuous at ``x_s``.
    """
    k1, k2, interface = k1[:, None], k2[:, None], interface[:, None]
    q = layered_darcy_flux(k1, k2, interface)
    head_at_interface = 1.0 - q * interface / k1
    return torch.where(x < interface, 1.0 - q * x / k1, head_at_interface - q * (x - interface) / k2)


def gardner_steady_head(alpha: Tensor, flux: Tensor, z: Tensor) -> Tensor:
    """Pressure head of ``d/dz [K(h) (dh/dz + 1)] = 0`` with ``K = exp(alpha h)`` and ``h(0) = 0``.

    The Kirchhoff variable ``Phi = exp(alpha h)`` solves ``Phi = -q + (1 + q) exp(-alpha z)``,
    where ``q`` is the Darcy flux ``-K (h' + 1)`` relative to ``Ks`` (negative downward).

    Args:
        alpha: Gardner parameter, shape ``(B,)``.
        flux: ``q / Ks`` in ``(-1, 0)``, shape ``(B,)``.
        z: Heights ``(N,)``, pointing up from the water table at ``z = 0``.

    Returns:
        ``h = ln(Phi) / alpha``, shape ``(B, N)``, non-positive everywhere.
    """
    alpha, flux = alpha[:, None], flux[:, None]
    kirchhoff = -flux + (1.0 + flux) * torch.exp(-alpha * z)
    return torch.log(kirchhoff) / alpha
