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


def heat_sine_series_2d(coefficients: Tensor, x: Tensor, y: Tensor, t: Tensor, diffusivity: float) -> Tensor:
    """Evaluate ``T = sum_{m,n} b_mn sin(m pi x) sin(n pi y) exp(-a pi^2 (m^2 + n^2) t)``.

    This solves ``dT/dt = a (d2T/dx2 + d2T/dy2)`` on ``[0, 1]^2`` with ``T = 0`` on the boundary
    and the initial field ``T0 = sum b_mn sin(m pi x) sin(n pi y)``.

    Args:
        coefficients: ``b_mn`` of shape ``(B, M, N)``, ``m`` along dimension 1.
        x, y, t: Evaluation points, all of the same shape ``S``.
        diffusivity: Diffusion coefficient ``a``.

    Returns:
        Values of shape ``(B, *S)``.
    """
    km, kn = _wavenumbers(coefficients.transpose(-1, -2)), _wavenumbers(coefficients)
    # Evaluation points get two trailing mode dimensions: S -> (*S, M, N).
    x, y, t = (v[..., None, None] for v in (x, y, t))
    modes = (
        torch.sin(x * km[:, None])
        * torch.sin(y * kn[None, :])
        * torch.exp(-diffusivity * (km[:, None] ** 2 + kn[None, :] ** 2) * t)
    )
    # (B, M, N) contracted with (*S, M, N) -> (B, *S).
    return torch.tensordot(coefficients, modes, dims=([-2, -1], [-2, -1]))


def chloride_corner(surface: Tensor, diffusivity: Tensor, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
    """Evaluate ``C = Cs [1 - erf(x / (2 sqrt(D t))) erf(y / (2 sqrt(D t)))]``.

    Chloride ingress into the corner ``x, y >= 0`` with the exposed faces ``x = 0`` and ``y = 0``
    (``C = Cs`` there) and ``C = 0`` initially. ``1 - C / Cs`` is the product of two solutions
    ``erf(x / (2 sqrt(D t)))`` of the 1D diffusion equation, hence solves the 2D equation.

    Args:
        surface: Surface concentration ``Cs`` of shape ``(B,)``.
        diffusivity: Diffusion coefficient ``D`` of shape ``(B,)``.
        x, y, t: Evaluation points, broadcastable against each other, ``t > 0``.

    Returns:
        Values of shape ``(B, *S)`` with ``S`` the broadcast shape of ``x``, ``y`` and ``t``.
    """
    x, y, t = torch.broadcast_tensors(x, y, t)
    extra = (1,) * x.dim()
    surface = surface.reshape(-1, *extra)
    width = 2.0 * torch.sqrt(diffusivity.reshape(-1, *extra) * t)
    return surface * (1.0 - torch.erf(x / width) * torch.erf(y / width))


def heat_ramp_series(
    coefficients: Tensor,
    start_left: Tensor,
    rate_left: Tensor,
    start_right: Tensor,
    rate_right: Tensor,
    x: Tensor,
    t: Tensor,
    diffusivity: float,
    n_modes: int = 200,
) -> Tensor:
    """Heat equation on ``x in [0, 1]`` with boundary values linear in time.

    Solves ``dT/dt = a d2T/dx2`` with ``T(0, t) = alpha_L + beta_L t``, ``T(1, t) = alpha_R +
    beta_R t`` and the compatible initial profile
    ``T(x, 0) = (1 - x) alpha_L + x alpha_R + sum_n b_n sin(n pi x)``.

    Lifting: ``T = (1 - x) T_L(t) + x T_R(t) + w`` with ``w = 0`` on the boundary and
    ``dw/dt = a d2w/dx2 - [(1 - x) beta_L + x beta_R]``. The source has the sine coefficients
    ``s_n = 2 / (n pi) [beta_L + (-1)^(n+1) beta_R]``, so with ``lambda_n = a (n pi)^2``

        w_n(t) = b_n exp(-lambda_n t) - s_n (1 - exp(-lambda_n t)) / lambda_n.

    The ramp part decays like ``1 / n^3``; truncating it after ``n_modes`` leaves an error of
    about ``|beta| / (a pi^3 n_modes^2)``, 8e-6 for ``beta = 10``, ``a = 1`` and 200 modes.

    Args:
        coefficients: ``b_n`` of shape ``(B, N)``.
        start_left, rate_left, start_right, rate_right: ``alpha_L``, ``beta_L``, ``alpha_R``,
            ``beta_R``, each of shape ``(B,)``.
        x, t: Evaluation points of the same shape ``S``.
        diffusivity: Diffusion coefficient ``a > 0``.
        n_modes: Number of modes of the ramp series.

    Returns:
        Values of shape ``(B, *S)``.
    """
    extra = (1,) * x.dim()
    alpha_left, beta_left = start_left.reshape(-1, *extra), rate_left.reshape(-1, *extra)
    alpha_right, beta_right = start_right.reshape(-1, *extra), rate_right.reshape(-1, *extra)
    lifting = (1.0 - x) * (alpha_left + beta_left * t) + x * (alpha_right + beta_right * t)

    n = torch.arange(1, n_modes + 1, dtype=x.dtype, device=x.device)
    k = n * math.pi
    rate = diffusivity * k**2
    # 2 / (n pi) sin(n pi x) (1 - exp(-lambda t)) / lambda, with expm1 for small lambda t.
    modes = (2.0 / k) * torch.sin(x.unsqueeze(-1) * k) * (-torch.expm1(-rate * t.unsqueeze(-1))) / rate
    response_left = modes.sum(dim=-1)  # the source (1 - x) has s_n proportional to 1
    response_right = (modes * (1.0 - 2.0 * (n.long() % 2 == 0).to(x.dtype))).sum(dim=-1)  # x: (-1)^(n+1)

    series = heat_sine_series(coefficients, x, t, diffusivity)
    return lifting + series - beta_left * response_left - beta_right * response_right
