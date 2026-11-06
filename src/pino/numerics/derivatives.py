"""Second-order central finite differences on uniform grids.

Each operator returns values on the interior points of ``dim`` only, so the result is
two entries shorter along that dimension.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor


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


def interior(u: Tensor, dims: Iterable[int]) -> Tensor:
    """Remove the first and last entry along each dimension in ``dims``."""
    for dim in dims:
        u = u.narrow(dim, 1, u.shape[dim] - 2)
    return u
