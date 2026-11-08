"""Plane-stress elasticity on uniform grids, with an optional thermal strain.

Fields have the shape ``(..., Nx, Ny)``. ``youngs_modulus`` and ``poisson`` are numbers or tensors
that broadcast against the fields, for example ``(B, 1, 1)`` per sample.
"""

from __future__ import annotations

from torch import Tensor

from pino.numerics.derivatives import central_d1, central_d2, central_mixed, gradient, interior


def plane_stress(
    u: Tensor,
    v: Tensor,
    dx: float,
    dy: float,
    youngs_modulus: float | Tensor,
    poisson: float | Tensor,
    thermal_strain: float | Tensor | None = None,
) -> dict[str, Tensor]:
    """Stresses on all grid points; derivatives are one-sided of second order on the boundary.

    ``sigma_xx = E / (1 - nu^2) [(e_xx - e_th) + nu (e_yy - e_th)]``,
    ``sigma_yy = E / (1 - nu^2) [(e_yy - e_th) + nu (e_xx - e_th)]``,
    ``sigma_xy = E / (2 (1 + nu)) (du/dy + dv/dx)``, with ``e_xx = du/dx`` and ``e_yy = dv/dy``.
    """
    e_xx, e_yy = gradient(u, dx, -2), gradient(v, dy, -1)
    if thermal_strain is not None:
        e_xx, e_yy = e_xx - thermal_strain, e_yy - thermal_strain
    factor = youngs_modulus / (1.0 - poisson**2)
    return {
        "sigma_xx": factor * (e_xx + poisson * e_yy),
        "sigma_yy": factor * (e_yy + poisson * e_xx),
        "sigma_xy": youngs_modulus / (2.0 * (1.0 + poisson)) * (gradient(u, dy, -1) + gradient(v, dx, -2)),
    }


def navier_lame_residual(
    u: Tensor,
    v: Tensor,
    dx: float,
    dy: float,
    youngs_modulus: float | Tensor,
    poisson: float | Tensor,
    thermal_strain: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Equilibrium ``div sigma = 0`` in displacement form, on the interior points.

    ``E / (1 - nu^2) [u_xx + (1 - nu) / 2 u_yy + (1 + nu) / 2 v_xy] - E / (1 - nu) d(e_th)/dx``,
    ``E / (1 - nu^2) [v_yy + (1 - nu) / 2 v_xx + (1 + nu) / 2 u_xy] - E / (1 - nu) d(e_th)/dy``.
    A ``thermal_strain`` must be a field on the same grid as ``u``.
    """
    factor = youngs_modulus / (1.0 - poisson**2)
    u_xx, u_yy = interior(central_d2(u, dx, -2), [-1]), interior(central_d2(u, dy, -1), [-2])
    v_xx, v_yy = interior(central_d2(v, dx, -2), [-1]), interior(central_d2(v, dy, -1), [-2])
    u_xy, v_xy = central_mixed(u, dx, dy, -2, -1), central_mixed(v, dx, dy, -2, -1)
    first = factor * (u_xx + 0.5 * (1.0 - poisson) * u_yy + 0.5 * (1.0 + poisson) * v_xy)
    second = factor * (v_yy + 0.5 * (1.0 - poisson) * v_xx + 0.5 * (1.0 + poisson) * u_xy)
    if thermal_strain is not None:
        thermal = youngs_modulus / (1.0 - poisson)
        first = first - thermal * interior(central_d1(thermal_strain, dx, -2), [-1])
        second = second - thermal * interior(central_d1(thermal_strain, dy, -1), [-2])
    return first, second
