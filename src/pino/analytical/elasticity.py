"""Analytical solutions of plane-stress elasticity."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def cosine_series(coefficients: Tensor, x: Tensor) -> Tensor:
    """Evaluate ``sum_n p_n cos(n pi x)`` for ``p_n`` of shape ``(B, N)``; values ``(B, *x.shape)``."""
    n = torch.arange(1, coefficients.shape[-1] + 1, dtype=coefficients.dtype, device=coefficients.device)
    return torch.tensordot(coefficients, torch.cos(x.unsqueeze(-1) * n * math.pi), dims=([-1], [-1]))


def block_mode_profile(rate: Tensor, y: Tensor, height: float) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Airy profile ``f(y)`` of a load mode with unit amplitude, and what follows from it.

    The stress function is ``Phi = cos(lambda x) f(y)`` with
    ``f = A cosh(lambda y) + B sinh(lambda y) + C y cosh(lambda y) + D y sinh(lambda y)``.
    The conditions ``sigma_xy(x, 0) = 0`` and ``v(x, 0) = 0`` give ``f'(0) = f'''(0) = 0``, hence
    ``B = C = 0``; ``sigma_xy(x, H) = 0`` and ``sigma_yy(x, H) = -cos(lambda x)`` give
    ``f'(H) = 0`` and ``f(H) = 1 / lambda^2``. With ``T = tanh(lambda H)`` this is

        A cosh(lambda H) = (T + lambda H) / (lambda^2 Q),  D cosh(lambda H) = -lambda T / (lambda^2 Q),
        Q = T + lambda H (1 - T^2).

    ``cosh`` and ``sinh`` are always evaluated relative to ``cosh(lambda H)``, which stays
    bounded for large ``lambda``. ``tests/test_elastic_analytical.py`` checks this against the
    general 4 x 4 system with sympy.

    Args:
        rate: ``lambda``, shape ``(N,)``.
        y: Heights of arbitrary shape ``S``.
        height: Block height ``H``.

    Returns:
        ``f, f', f'', f'''`` and ``F = (2 lambda^2 f' - f''') / lambda^4``, an antiderivative of
        ``f`` with ``F(0) = 0``, each of shape ``(*S, N)``.
    """
    y = y.unsqueeze(-1)
    tanh = torch.tanh(rate * height)
    sech2 = 1.0 / torch.cosh(rate * height) ** 2
    q = tanh + rate * height * sech2
    a = (tanh + rate * height) / (rate**2 * q)
    d = -rate * tanh / (rate**2 * q)
    # cosh(lambda y) / cosh(lambda H) and sinh(lambda y) / cosh(lambda H).
    decay = torch.exp(rate * (y - height)) / (1.0 + torch.exp(-2.0 * rate * height))
    c = decay * (1.0 + torch.exp(-2.0 * rate * y))
    s = decay * (1.0 - torch.exp(-2.0 * rate * y))
    f = a * c + d * y * s
    f1 = a * rate * s + d * (s + y * rate * c)
    f2 = a * rate**2 * c + d * (2.0 * rate * c + y * rate**2 * s)
    f3 = a * rate**3 * s + d * (3.0 * rate**2 * s + y * rate**3 * c)
    return f, f1, f2, f3, (2.0 * rate**2 * f1 - f3) / rate**4


def elastic_block(
    mean_load: Tensor,
    coefficients: Tensor,
    poisson: Tensor,
    x: Tensor,
    y: Tensor,
    youngs_modulus: float = 1.0,
    height: float = 1.0,
) -> dict[str, Tensor]:
    """Plane-stress block on sliding supports under the load ``p(x)`` on top.

    Domain ``[0, 1] x [0, H]``. ``u = 0`` and ``sigma_xy = 0`` on ``x = 0, 1``, ``v = 0`` and
    ``sigma_xy = 0`` on ``y = 0``, ``sigma_yy = -p(x)`` and ``sigma_xy = 0`` on ``y = H``, with
    ``p = p_0 + sum_n p_n cos(n pi x)``.

    Mode 0: ``u = 0``, ``v = -p_0 (1 - nu^2) y / E``, ``sigma_xx = -nu p_0``, ``sigma_yy = -p_0``.
    Modes ``n >= 1`` with ``lambda = n pi`` and the profile of :func:`block_mode_profile`:

        sigma_xx = cos(lambda x) f'',  sigma_yy = -lambda^2 cos(lambda x) f,
        sigma_xy = lambda sin(lambda x) f',
        u = sin(lambda x) (f'' + nu lambda^2 f) / (lambda E),
        v = -cos(lambda x) (lambda^2 F + nu f') / E.

    Evaluated in double precision and returned in the precision of ``x``.

    Args:
        mean_load: ``p_0``, shape ``(B,)``.
        coefficients: ``p_n``, shape ``(B, N)``.
        poisson: ``nu``, shape ``(B,)``.
        x, y: Points of the same shape ``S``.
        youngs_modulus: ``E``.
        height: ``H``.

    Returns:
        ``u``, ``v``, ``sigma_xx``, ``sigma_yy`` and ``sigma_xy``, each of shape ``(B, *S)``.
    """
    dtype = x.dtype
    x, y = x.double(), y.double()
    p0, p, nu = mean_load.double(), coefficients.double(), poisson.double()
    extra = (1,) * x.dim()
    p0, nu_field = p0.reshape(-1, *extra), nu.reshape(-1, *extra)
    n = torch.arange(1, p.shape[-1] + 1, dtype=torch.float64, device=x.device)
    rate = n * math.pi
    f, f1, f2, _, big_f = block_mode_profile(rate, y, height)
    sin, cos = torch.sin(x.unsqueeze(-1) * rate), torch.cos(x.unsqueeze(-1) * rate)

    def series(modes: Tensor) -> Tensor:
        return torch.tensordot(p, modes, dims=([-1], [-1]))

    u = (series(sin * f2 / rate) + nu_field * series(sin * rate * f)) / youngs_modulus
    v = -(series(cos * rate**2 * big_f) + nu_field * series(cos * f1)) / youngs_modulus
    v = v - p0 * (1.0 - nu_field**2) * y / youngs_modulus
    fields = {
        "u": u,
        "v": v,
        "sigma_xx": series(cos * f2) - nu_field * p0,
        "sigma_yy": -series(cos * rate**2 * f) - p0,
        "sigma_xy": series(sin * rate * f1),
    }
    return {name: field.to(dtype) for name, field in fields.items()}
