"""heat_2d: double sine series solution, constraint, residual and time resolution."""

import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from pino.analytical.diffusion import heat_sine_series_2d
from pino.evaluation import predict
from pino.models.factory import build_model
from pino.numerics.grids import grid_spacing
from pino.problems import build_problem

CFG = OmegaConf.load(Path(__file__).parents[1] / "configs" / "problem" / "heat_2d.yaml")
MODEL_CFG = OmegaConf.create({"n_modes": 4, "hidden_channels": 4, "n_layers": 2, "domain_padding": 0.25})


@pytest.fixture
def problem():
    return build_problem(CFG)


@pytest.fixture
def inputs(problem):
    return problem.sample_inputs(4, torch.Generator().manual_seed(0)).double()


def test_series_solves_the_heat_equation(inputs):
    x, y = torch.rand(40, dtype=torch.float64, requires_grad=True), torch.rand(40, dtype=torch.float64, requires_grad=True)
    t = (CFG.t_end * torch.rand(40, dtype=torch.float64)).requires_grad_()
    for b in range(inputs.shape[0]):
        T = heat_sine_series_2d(inputs[b : b + 1], x, y, t, CFG.a)
        T_x, T_y, T_t = torch.autograd.grad(T.sum(), (x, y, t), create_graph=True)
        (T_xx,) = torch.autograd.grad(T_x.sum(), x, retain_graph=True)
        (T_yy,) = torch.autograd.grad(T_y.sum(), y)
        torch.testing.assert_close(T_t, CFG.a * (T_xx + T_yy), rtol=1e-9, atol=1e-9)


def test_series_matches_a_single_mode_and_its_initial_and_boundary_values(problem):
    coefficients = torch.zeros(1, 3, 3, dtype=torch.float64)
    coefficients[0, 1, 2] = 0.7  # m = 2, n = 3
    grid = problem.grid(17).double()
    x, y, t = grid
    expected = 0.7 * torch.sin(2 * math.pi * x) * torch.sin(3 * math.pi * y) * torch.exp(-13 * math.pi**2 * t)
    T = problem.analytical(coefficients, grid)[0, 0]
    torch.testing.assert_close(T, expected, rtol=0, atol=1e-13)
    for edge in (T[0], T[-1], T[:, 0], T[:, -1]):
        assert edge.abs().max() < 1e-13


def test_residual_normalization_is_the_fastest_rate(problem):
    assert problem.residual_scale == pytest.approx(CFG.a * math.pi**2 * 18)
    assert problem.residual_scale == pytest.approx(177.65, abs=0.01)


def test_fd_residual_of_the_exact_solution_is_small_and_converges(problem, inputs):
    norms = []
    for n in (16, 32, 64):
        grid = problem.grid(n).double()
        norms.append(problem.residual(problem.analytical(inputs, grid), inputs, grid).pow(2).mean().sqrt().item())
    rates = [math.log2(norms[i] / norms[i + 1]) for i in range(2)]
    assert all(rate > 1.8 for rate in rates), (norms, rates)
    # On the training grid the discretization error is far below the residual of a wrong field.
    grid = problem.grid(CFG.train_resolution).double()
    exact = problem.analytical(inputs, grid)
    wrong = problem.analytical(inputs * 1.05, grid)  # same boundaries, 5 % off in the interior
    assert problem.residual(exact, inputs, grid).abs().mean() < 0.1 * problem.residual(wrong, inputs, grid).abs().mean() + 1e-3


def test_training_grid_resolves_the_fastest_mode(problem):
    """exp(-178 dt) is the decay of the fastest mode per time step; it must stay close to one."""
    dt = grid_spacing(problem.grid(CFG.train_resolution), 2)
    assert math.exp(-problem.residual_scale * dt) > 0.7


def test_constraint_reproduces_initial_and_boundary_values(problem, inputs):
    grid = problem.grid(16).double()
    exact = problem.analytical(inputs, grid)
    u = predict_with_random_network(problem, inputs, grid)
    torch.testing.assert_close(u[..., 0], exact[..., 0], rtol=0, atol=1e-12)
    for edge in (u[..., 0, :, :], u[..., -1, :, :], u[..., :, 0, :], u[..., :, -1, :]):
        assert edge.abs().max() == 0


def predict_with_random_network(problem, inputs, grid):
    g, phi = problem.boundary(inputs, grid)
    return g + phi * torch.randn(inputs.shape[0], 1, *grid.shape[1:], dtype=grid.dtype)


def test_shapes_and_fno_input(problem, inputs):
    grid = problem.grid((16, 12, 10))
    x = problem.to_fno_input(inputs.float(), grid)
    assert x.shape == (4, 4, 16, 12, 10)
    # T0 is broadcast over time, then x, y, t / t_end.
    assert torch.equal(x[:, 0, ..., 0], x[:, 0, ..., -1])
    assert torch.equal(x[0, 1], grid[0]) and torch.equal(x[0, 2], grid[1])
    assert x[0, 3, 0, 0, 0] == 0 and x[0, 3, 0, 0, -1] == pytest.approx(1.0)
    model = build_model(MODEL_CFG, problem, (16, 12, 10))
    assert predict(model, problem, inputs.float(), grid).shape == (4, 1, 16, 12, 10)


def test_other_input_families_are_rejected():
    with pytest.raises(ValueError, match="heat_2d does not implement inputs='rich'"):
        build_problem(OmegaConf.merge(CFG, {"inputs": "rich"}))
