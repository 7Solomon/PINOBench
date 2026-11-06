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


def dirichlet_1d_t(
    x: Tensor, t: Tensor, initial: Tensor, left: Tensor, right: Tensor
) -> tuple[Tensor, Tensor]:
    """Constraint for a 1D problem in time with an initial profile and Dirichlet values in time.

    With ``tau = (t - t0) / (t_end - t0)`` the constraint is

        g(x, t) = u0(x) + (1 - x) (u_L(t) - u_L(t0)) + x (u_R(t) - u_R(t0)),
        phi(x, t) = tau x (1 - x).

    ``g`` reproduces ``u0`` at ``t = t0`` and, if the data are compatible in the corners
    (``u0(0) == u_L(t0)`` and ``u0(1) == u_R(t0)``), ``u_L`` and ``u_R`` at both ends.

    Args:
        x: Spatial points ``(Nx,)`` with ``x[0] == 0`` and ``x[-1] == 1``.
        t: Times ``(Nt,)``, ``t[0]`` is the initial time ``t0``.
        initial: ``u0(x)``, shape ``(B, C, Nx)``.
        left: ``u_L(t)`` at ``x = 0``, shape ``(B, C, Nt)``.
        right: ``u_R(t)`` at ``x = 1``, shape ``(B, C, Nt)``.

    Returns:
        ``g`` of shape ``(B, C, Nx, Nt)`` and ``phi`` of shape ``(1, 1, Nx, Nt)``.
    """
    tau = (t - t[0]) / (t[-1] - t[0])
    xs = x[:, None]
    g = (
        initial[..., :, None]
        + (1.0 - xs) * (left - left[..., :1])[..., None, :]
        + xs * (right - right[..., :1])[..., None, :]
    )
    phi = (tau[None, :] * xs * (1.0 - xs))[None, None]
    return g, phi
