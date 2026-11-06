"""Generic hard-constraint helpers in ``numerics/constraints.py``."""

import torch

from pino.numerics.constraints import apply_hard_constraint, dirichlet_1d_t


def test_dirichlet_1d_t_reproduces_initial_and_boundary_values():
    torch.manual_seed(0)
    x = torch.linspace(0, 1, 17, dtype=torch.float64)
    t = torch.linspace(0.01, 0.05, 11, dtype=torch.float64)
    batch, channels = 3, 2
    amplitude = torch.randn(batch, channels, 1, dtype=torch.float64)
    # Time-dependent boundary values with compatible corners: u0(0) = u_L(t0), u0(1) = u_R(t0).
    left = amplitude * (1.0 + torch.sin(20 * t))
    right = -amplitude * torch.cos(30 * t)
    initial = left[..., :1] * (1 - x) ** 2 + right[..., :1] * x**3 + torch.sin(torch.pi * x) * amplitude

    g, phi = dirichlet_1d_t(x, t, initial, left, right)
    assert g.shape == (batch, channels, 17, 11) and phi.shape == (1, 1, 17, 11)

    u = apply_hard_constraint(torch.randn(batch, channels, 17, 11, dtype=torch.float64), g, phi)
    torch.testing.assert_close(u[..., 0], initial, rtol=0, atol=1e-14)
    torch.testing.assert_close(u[..., 0, :], left, rtol=0, atol=1e-14)
    torch.testing.assert_close(u[..., -1, :], right, rtol=0, atol=1e-14)
    # phi vanishes exactly where u is prescribed and is positive everywhere else.
    assert torch.all(phi[..., 0] == 0) and torch.all(phi[..., 0, :] == 0) and torch.all(phi[..., -1, :] == 0)
    assert torch.all(phi[..., 1:-1, 1:] > 0)
    torch.testing.assert_close(phi[0, 0, :, -1], x * (1 - x))
