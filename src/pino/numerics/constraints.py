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
