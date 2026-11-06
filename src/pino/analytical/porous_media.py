"""Stationary flow in porous media: layered Darcy flow, Richards' equation (Gardner model) and 2D Poisson flow."""

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


def _sine_modes_2d(coefficients: Tensor, x: Tensor, y: Tensor, weight: Tensor) -> Tensor:
    """``sum_mn w_mn a_mn sin(m pi x) sin(n pi y)`` for ``a`` of shape ``(B, M, M)``, values ``(B, *S)``."""
    m = torch.arange(1, coefficients.shape[-1] + 1, dtype=coefficients.dtype, device=coefficients.device)
    sin_x = torch.sin(x.unsqueeze(-1) * m * torch.pi)  # (*S, M)
    sin_y = torch.sin(y.unsqueeze(-1) * m * torch.pi)
    return torch.einsum("bmn,...m,...n->b...", coefficients * weight, sin_x, sin_y)


def sine_source_2d(coefficients: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Source ``f(x, y) = sum_{m,n=1..M} a_mn sin(m pi x) sin(n pi y)``.

    Args:
        coefficients: ``a_mn`` of shape ``(B, M, M)``.
        x, y: Points of the same shape ``S``.

    Returns:
        Values of shape ``(B, *S)``.
    """
    return _sine_modes_2d(coefficients, x, y, torch.ones((), dtype=coefficients.dtype))


def poisson_sine_head_2d(coefficients: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Solution of ``-Laplace h = f`` on ``[0, 1]^2`` with ``h = 0`` on the boundary.

    For the source of :func:`sine_source_2d` each mode is an eigenfunction of the Laplacian,
    so ``h = sum a_mn / (pi^2 (m^2 + n^2)) sin(m pi x) sin(n pi y)``. Shapes as there.
    """
    m = torch.arange(1, coefficients.shape[-1] + 1, dtype=coefficients.dtype, device=coefficients.device)
    weight = 1.0 / (torch.pi**2 * (m[:, None] ** 2 + m[None, :] ** 2))
    return _sine_modes_2d(coefficients, x, y, weight)
