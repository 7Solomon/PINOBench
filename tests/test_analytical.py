"""The closed-form heat solution must satisfy its PDE, boundaries and initial condition."""

import torch

from pino.analytical.diffusion import heat_sine_series, sine_series

A = 0.7
T_END = 0.1


def _coefficients(batch: int = 8, n_series: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return (2 * torch.rand(batch, n_series, generator=generator) - 1).double()


def test_solution_satisfies_pde_via_autograd():
    generator = torch.Generator().manual_seed(1)
    x = torch.rand(200, generator=generator).double().requires_grad_()
    t = (T_END * torch.rand(200, generator=generator)).double().requires_grad_()
    coefficients = _coefficients()

    u = heat_sine_series(coefficients, x, t, A)
    for sample in u:
        u_t, u_x = torch.autograd.grad(sample.sum(), (t, x), create_graph=True)
        u_xx = torch.autograd.grad(u_x.sum(), x, retain_graph=True)[0]
        assert (u_t - A * u_xx).abs().max() < 1e-10


def test_boundary_conditions():
    t = torch.linspace(0, T_END, 50).double()
    coefficients = _coefficients()
    assert heat_sine_series(coefficients, torch.zeros_like(t), t, A).abs().max() < 1e-12
    assert heat_sine_series(coefficients, torch.ones_like(t), t, A).abs().max() < 1e-12


def test_initial_condition():
    x = torch.linspace(0, 1, 101).double()
    coefficients = _coefficients()
    solution = heat_sine_series(coefficients, x, torch.zeros_like(x), A)
    assert torch.equal(solution, sine_series(coefficients, x))


def test_single_mode_matches_textbook_formula():
    x = torch.linspace(0, 1, 11).double()
    t = torch.full_like(x, 0.05)
    coefficients = torch.tensor([[0.0, 2.0]]).double()
    expected = 2.0 * torch.sin(2 * torch.pi * x) * torch.exp(-A * (2 * torch.pi) ** 2 * t)
    assert torch.allclose(heat_sine_series(coefficients, x, t, A)[0], expected)
