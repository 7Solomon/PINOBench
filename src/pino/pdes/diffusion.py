"""Finite-difference residual of the one-dimensional diffusion equation."""

from __future__ import annotations

from torch import Tensor

from pino.numerics.derivatives import central_d1, central_d2, interior


def diffusion_residual(
    u: Tensor,
    dx: float,
    dt: float,
    diffusivity: float | Tensor,
    x_dim: int = -2,
    t_dim: int = -1,
) -> Tensor:
    """Evaluate ``du/dt - D d2u/dx2`` on the interior points in space and time.

    Args:
        u: Field on a uniform space-time grid.
        dx: Spatial grid spacing.
        dt: Temporal grid spacing.
        diffusivity: Diffusion coefficient ``D``. A tensor must already be restricted to
            the interior points and broadcast against the result.
        x_dim: Spatial dimension of ``u``.
        t_dim: Temporal dimension of ``u``.

    Returns:
        Residual, two entries shorter than ``u`` along ``x_dim`` and ``t_dim``.
    """
    u_t = interior(central_d1(u, dt, t_dim), [x_dim])
    u_xx = interior(central_d2(u, dx, x_dim), [t_dim])
    return u_t - diffusivity * u_xx
