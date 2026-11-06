"""Second-order central finite differences on uniform grids.

Each operator returns values on the interior points of ``dim`` only, so the result is
two entries shorter along that dimension.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor

from pino.numerics.grids import grid_spacing


def central_d1(u: Tensor, h: float, dim: int) -> Tensor:
    """First derivative ``(u[i+1] - u[i-1]) / (2 h)`` along ``dim``."""
    n = u.shape[dim]
    return (u.narrow(dim, 2, n - 2) - u.narrow(dim, 0, n - 2)) / (2.0 * h)


def central_d2(u: Tensor, h: float, dim: int) -> Tensor:
    """Second derivative ``(u[i+1] - 2 u[i] + u[i-1]) / h^2`` along ``dim``."""
    n = u.shape[dim]
    return (u.narrow(dim, 2, n - 2) - 2.0 * u.narrow(dim, 1, n - 2) + u.narrow(dim, 0, n - 2)) / h**2


def face_flux(u: Tensor, k_faces: Tensor, h: float, dim: int = -1, gravity: float = 0.0) -> Tensor:
    """Flux ``k (du/dx + gravity)`` on the ``N - 1`` faces between neighbouring nodes.

    The face between nodes ``i`` and ``i + 1`` gets ``k_f ((u[i+1] - u[i]) / h + gravity)``.

    Args:
        u: Field on a uniform grid with ``N`` points along ``dim``.
        k_faces: Coefficient on the ``N - 1`` faces along ``dim``, broadcast against ``u``.
            The caller chooses how to average it, for example harmonically or arithmetically.
        h: Grid spacing.
        dim: Dimension of the derivative.
        gravity: Constant added to the gradient, for example ``1`` for Richards' equation
            with ``z`` pointing up.
    """
    n = u.shape[dim]
    gradient = (u.narrow(dim, 1, n - 1) - u.narrow(dim, 0, n - 1)) / h
    return k_faces * (gradient + gravity)


def conservative_divergence(u: Tensor, k_faces: Tensor, h: float, dim: int = -1, gravity: float = 0.0) -> Tensor:
    """Conservative second-order operator ``d/dx [k (du/dx + gravity)]`` on interior nodes.

    Node ``i`` gets ``(F[i] - F[i-1]) / h`` with the face fluxes ``F`` of :func:`face_flux`.
    """
    flux = face_flux(u, k_faces, h, dim, gravity)
    n = flux.shape[dim]
    return (flux.narrow(dim, 1, n - 1) - flux.narrow(dim, 0, n - 1)) / h


def laplacian(u: Tensor, grid: Tensor) -> Tensor:
    """Second-order Laplacian ``sum_d d2u/dx_d^2`` on the interior points of all grid dimensions.

    Works for any number of dimensions: the grid ``(n_dims, *resolution)`` sets the spacing
    of each axis, and the last ``n_dims`` dimensions of ``u`` are the spatial ones. Each
    ``central_d2`` is cut to the interior of the other axes, so the result is two entries
    shorter along every grid dimension.
    """
    n_dims = grid.shape[0]
    dims = list(range(u.dim() - n_dims, u.dim()))
    total = 0.0
    for axis, dim in enumerate(dims):
        others = [d for d in dims if d != dim]
        total = total + interior(central_d2(u, grid_spacing(grid, axis), dim), others)
    return total


def interior(u: Tensor, dims: Iterable[int]) -> Tensor:
    """Remove the first and last entry along each dimension in ``dims``."""
    for dim in dims:
        u = u.narrow(dim, 1, u.shape[dim] - 2)
    return u
