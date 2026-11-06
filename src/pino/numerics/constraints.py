"""Hard constraints of the form ``u = g + phi * NN``."""

from __future__ import annotations

from torch import Tensor


def apply_hard_constraint(network_output: Tensor, g: Tensor, phi: Tensor) -> Tensor:
    """Combine a raw network output with a constraint.

    Args:
        network_output: Raw network output ``NN``.
        g: Function that carries the prescribed boundary and initial values.
        phi: Distance-like function that vanishes wherever ``u`` is prescribed.

    Returns:
        ``g + phi * NN``. Where ``phi == 0`` the result equals ``g`` exactly.
    """
    return g + phi * network_output


def dirichlet_1d(x: Tensor, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
    """Constraint for a stationary 1D problem on ``x in [0, 1]`` with Dirichlet values on both ends.

    Args:
        x: Grid points ``(N,)`` with ``x[0] == 0`` and ``x[-1] == 1``.
        left: Values at ``x = 0``, shape ``(B, C)``, one per example and output channel.
        right: Values at ``x = 1``, shape ``(B, C)``.

    Returns:
        ``g = left (1 - x) + right x`` of shape ``(B, C, N)`` and ``phi = x (1 - x)`` of shape
        ``(1, 1, N)``. Both boundary values are reproduced exactly.
    """
    g = left[..., None] * (1.0 - x) + right[..., None] * x
    phi = (x * (1.0 - x))[None, None]
    return g, phi
