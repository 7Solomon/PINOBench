"""Stationary flow in porous media: layered Darcy flow."""

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
