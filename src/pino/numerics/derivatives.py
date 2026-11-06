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


def interior(u: Tensor, dims: Iterable[int]) -> Tensor:
    """Remove the first and last entry along each dimension in ``dims``."""
    for dim in dims:
        u = u.narrow(dim, 1, u.shape[dim] - 2)
    return u
