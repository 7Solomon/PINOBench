"""Finite-difference residual of the diffusion equation in any number of space dimensions."""

from __future__ import annotations

from torch import Tensor

from pino.numerics.derivatives import central_d1, interior, laplacian
from pino.numerics.grids import grid_spacing


def diffusion_residual(u: Tensor, grid: Tensor, diffusivity: float | Tensor) -> Tensor:
    """Evaluate ``du/dt - D Laplace u`` on the interior points in space and time.

    Args:
        u: Field ``(B, C, *space, Nt)`` on a uniform space-time grid; time is the last dimension.
        grid: Coordinates ``(n_dims, *space, Nt)``, the last axis is time.
        diffusivity: Diffusion coefficient ``D``. A tensor must broadcast against the result.

    Returns:
        Residual, two entries shorter than ``u`` along every space dimension and along time.

    The spatial part is :func:`~pino.numerics.derivatives.laplacian` on the space grid at the
    first time; time is moved in front of the space dimensions for it and back afterwards.
    """
    n_space = grid.shape[0] - 1
    space_dims = list(range(u.dim() - 1 - n_space, u.dim() - 1))
    u_t = interior(central_d1(u, grid_spacing(grid, n_space), -1), space_dims)
    space_grid = grid[:n_space, ..., 0]
    u_xx = laplacian(u.movedim(-1, -1 - n_space), space_grid)
    u_xx = interior(u_xx, [-1 - n_space]).movedim(-1 - n_space, -1)
    return u_t - diffusivity * u_xx
