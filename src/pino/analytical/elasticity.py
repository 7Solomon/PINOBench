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


def thermal_mode_profile(rate: Tensor, y: Tensor, height: float) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Stress profile ``F(y)`` of a temperature mode with unit amplitude, ``E = alpha = 1``.

    The temperature is ``T = cos(lambda x) sinh(lambda y) / sinh(lambda H)``. Goodier's potential
    ``psi = cos(lambda x) y cosh(lambda y) / (2 lambda) * (1 + nu) / sinh(lambda H)`` solves
    ``Laplace psi = (1 + nu) T``, and ``u_i = d psi / d x_i`` satisfies the equilibrium with the
    stresses ``sigma_xx = -E / (1 + nu) psi_yy``, ``sigma_yy = -E / (1 + nu) psi_xx``,
    ``sigma_xy = E / (1 + nu) psi_xy``. These are the stresses of an Airy function. So the whole
    stress state, Goodier's part and the corrections, has one stress function
    ``Phi = cos(lambda x) F(y)``, ``F = A cosh + B sinh + C y cosh + D y sinh`` in ``lambda y``:

        sigma_xx = cos(lambda x) F'',  sigma_yy = -lambda^2 cos(lambda x) F,
        sigma_xy = lambda sin(lambda x) F',

    and the three stress conditions ``F'(0) = F(H) = F'(H) = 0`` are homogeneous. The temperature
    enters only through ``v(x, 0) = 0``, which reads ``F'''(0) = -lambda / sinh(lambda H)``.
    Hence, with ``S = sinh(lambda H)``, ``T_H = tanh(lambda H)``, ``Q = sinh^2 / (sinh cosh + lambda H)``:

        B = 1 / (2 lambda^2 S),  C = -1 / (2 lambda S),  D = Q / (2 lambda S),
        A = (H (1 - T_H Q) / (2 lambda) - T_H / (2 lambda^2)) / S.

    ``F`` does not depend on ``nu``. ``cosh`` and ``sinh`` are always evaluated relative to ``S``,
    which stays bounded for large ``lambda``. ``tests/test_thermoelastic_analytical.py`` checks
    this against the general 4 x 4 system and sympy.

    Args:
        rate: ``lambda``, shape ``(N,)``.
        y: Heights of arbitrary shape ``K``.
        height: Block height ``H``.

    Returns:
        ``F, F', F'', F'''``, ``sinh(lambda y) / S`` and ``cosh(lambda y) / S``, each ``(*K, N)``.
    """
    y = y.unsqueeze(-1)
    tanh = torch.tanh(rate * height)
    # sinh^2 / (sinh cosh + lambda H), divided by cosh^2 in numerator and denominator.
    sech2 = 1.0 / torch.cosh(rate * height) ** 2
    q = tanh**2 / (tanh + rate * height * sech2)
    a = height * (1.0 - tanh * q) / (2.0 * rate) - tanh / (2.0 * rate**2)
    b = 1.0 / (2.0 * rate**2)
    c = -1.0 / (2.0 * rate)
    d = q / (2.0 * rate)
    # sinh(lambda y) / S and cosh(lambda y) / S.
    decay = torch.exp(rate * (y - height)) / (1.0 - torch.exp(-2.0 * rate * height))
    rs = decay * (1.0 - torch.exp(-2.0 * rate * y))
    rc = decay * (1.0 + torch.exp(-2.0 * rate * y))
    f = a * rc + b * rs + c * y * rc + d * y * rs
    f1 = rate * (a * rs + b * rc) + c * (rc + rate * y * rs) + d * (rs + rate * y * rc)
    f2 = rate**2 * (a * rc + b * rs) + c * (2.0 * rate * rs + rate**2 * y * rc) + d * (2.0 * rate * rc + rate**2 * y * rs)
    f3 = rate**3 * (a * rs + b * rc) + c * (3.0 * rate**2 * rc + rate**3 * y * rs) + d * (3.0 * rate**2 * rs + rate**3 * y * rc)
    return f, f1, f2, f3, rs, rc


def thermoelastic_block(
    mean_temperature: Tensor,
    coefficients: Tensor,
    poisson: Tensor,
    x: Tensor,
    y: Tensor,
    height: float = 1.0,
) -> dict[str, Tensor]:
    """Block on sliding supports, heated on top; steady conduction, plane stress, ``E = alpha = 1``.

    Domain ``[0, 1] x [0, H]``. Temperature: ``T(x, H) = t_0 + sum_n t_n cos(n pi x)``,
    ``T(x, 0) = 0``, insulated sides. Mechanics: ``u = 0``, ``sigma_xy = 0`` on ``x = 0, 1``,
    ``v = 0``, ``sigma_xy = 0`` on ``y = 0``, free on top: ``sigma_yy = sigma_xy = 0``.

    Mode 0: ``T = t_0 y / H``, ``u = 0``, ``v = (1 + nu) t_0 y^2 / (2 H)``,
    ``sigma_xx = -t_0 y / H``, ``sigma_yy = sigma_xy = 0``.
    Modes ``n >= 1`` with ``lambda = n pi`` and the profile of :func:`thermal_mode_profile`,
    ``r_s = sinh(lambda y) / sinh(lambda H)``, ``r_c = cosh(lambda y) / sinh(lambda H)``:

        T = cos(lambda x) r_s,
        u = sin(lambda x) (F'' + nu lambda^2 F + r_s) / lambda,
        v = cos(lambda x) ((F''' - (2 + nu) lambda^2 F') / lambda^2 + r_c / lambda),

    and the stresses as in :func:`thermal_mode_profile`. Evaluated in double precision and
    returned in the precision of ``x``.

    Args:
        mean_temperature: ``t_0``, shape ``(B,)``.
        coefficients: ``t_n``, shape ``(B, N)``.
        poisson: ``nu``, shape ``(B,)``.
        x, y: Points of the same shape ``K``.
        height: ``H``.

    Returns:
        ``T``, ``u``, ``v``, ``sigma_xx``, ``sigma_yy`` and ``sigma_xy``, each ``(B, *K)``.
    """
    dtype = x.dtype
    x, y = x.double(), y.double()
    t0, t, nu = mean_temperature.double(), coefficients.double(), poisson.double()
    extra = (1,) * x.dim()
    t0, nu_field = t0.reshape(-1, *extra), nu.reshape(-1, *extra)
    n = torch.arange(1, t.shape[-1] + 1, dtype=torch.float64, device=x.device)
    rate = n * math.pi
    f, f1, f2, f3, rs, rc = thermal_mode_profile(rate, y, height)
    sin, cos = torch.sin(x.unsqueeze(-1) * rate), torch.cos(x.unsqueeze(-1) * rate)

    def series(modes: Tensor) -> Tensor:
        return torch.tensordot(t, modes, dims=([-1], [-1]))

    level = y / height
    fields = {
        "T": series(cos * rs) + t0 * level,
        "u": series(sin * (f2 + rs) / rate) + nu_field * series(sin * rate * f),
        "v": series(cos * ((f3 - 2.0 * rate**2 * f1) / rate**2 + rc / rate)) - nu_field * series(cos * f1)
        + (1.0 + nu_field) * t0 * y * level / 2.0,
        "sigma_xx": series(cos * f2) - t0 * level,
        "sigma_yy": -series(cos * rate**2 * f),
        "sigma_xy": series(sin * rate * f1),
    }
    return {name: field.to(dtype) for name, field in fields.items()}
