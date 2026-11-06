"""chloride_1d: Crank's erfc solution, constraint and finite-difference residual."""

import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from pino.analytical.diffusion import crank_erfc
from pino.evaluation import predict
from pino.models.factory import build_model
from pino.numerics.grids import grid_spacing
from pino.problems import build_problem

CFG = OmegaConf.load(Path(__file__).parents[1] / "configs" / "problem" / "chloride_1d.yaml")
MODEL_CFG = OmegaConf.create({"n_modes": 4, "hidden_channels": 4, "n_layers": 2, "domain_padding": 0.25})


@pytest.fixture
def problem():
    return build_problem(CFG)


@pytest.fixture
def inputs(problem):
    return problem.sample_inputs(6, torch.Generator().manual_seed(0)).double()


def test_erfc_solves_the_diffusion_equation():
    surface = torch.tensor([0.5, 0.8, 1.0], dtype=torch.float64)
    diffusivity = torch.tensor([0.2, 0.5, 1.0], dtype=torch.float64)
    x = torch.rand(50, dtype=torch.float64, requires_grad=True)
    t = (CFG.t_start + (CFG.t_end - CFG.t_start) * torch.rand(50, dtype=torch.float64)).requires_grad_()
    for i in range(3):
        c = crank_erfc(surface[i : i + 1], diffusivity[i : i + 1], x, t)
        c_x, c_t = torch.autograd.grad(c.sum(), (x, t), create_graph=True)
        (c_xx,) = torch.autograd.grad(c_x.sum(), x)
        torch.testing.assert_close(c_t, diffusivity[i] * c_xx, rtol=1e-10, atol=1e-10)


def test_erfc_reaches_the_surface_value_and_the_initial_value():
    surface, diffusivity = torch.tensor([0.7]), torch.tensor([0.4])
    assert crank_erfc(surface, diffusivity, torch.zeros(3), torch.tensor([0.01, 0.02, 0.05])).eq(0.7).all()
    assert crank_erfc(surface, diffusivity, torch.ones(1), torch.tensor([1e-6])).item() == pytest.approx(0.0, abs=1e-30)


def test_right_boundary_is_far_from_the_front():
    """erfc at x = 1 stays below 1e-2 for every admissible D and t, so the depth is effectively infinite."""
    worst = math.erfc(1.0 / (2.0 * math.sqrt(CFG.diffusivity_high * CFG.t_end)))
    assert worst < 1e-2


def test_boundary_layer_is_resolved_on_the_training_grid(problem):
    """The thinnest profile, at D_min and t0, has width 2 sqrt(D t0) ~ 0.09 and needs >= 5 grid points."""
    width = 2.0 * math.sqrt(CFG.diffusivity_low * CFG.t_start)
    dx = grid_spacing(problem.grid(CFG.train_resolution), 0)
    assert width == pytest.approx(0.0894, abs=1e-3)
    assert width / dx >= 5


def test_reference_time_is_about_1270_years(problem):
    assert problem.scales["t"] == pytest.approx(1267.5, abs=0.1)
    assert problem.to_physical("t", CFG.t_start) == pytest.approx(12.7, abs=0.1)
    assert problem.to_physical("t", CFG.t_end) == pytest.approx(63.4, abs=0.1)


def test_shapes_and_fno_input(problem, inputs):
    grid = problem.grid((16, 12))
    x = problem.to_fno_input(inputs.float(), grid)
    assert x.shape == (6, problem.in_channels, 16, 12)
    assert torch.all(x[:, 0] == inputs[:, 0, None, None].float())
    assert torch.equal(x[0, 2], grid[0])
    assert x[0, 3, 0, 0] == 0 and x[0, 3, 0, -1] == pytest.approx(1.0)
    model = build_model(MODEL_CFG, problem, 16)
    assert predict(model, problem, inputs.float(), grid).shape == (6, 1, 16, 12)


def test_constraint_reproduces_initial_and_boundary_values(problem, inputs):
    grid = problem.grid(32).double()
    exact = problem.analytical(inputs, grid)
    g, phi = problem.boundary(inputs, grid)
    u = g + phi * torch.randn_like(exact)
    torch.testing.assert_close(u[..., 0], exact[..., 0], rtol=0, atol=1e-14)
    torch.testing.assert_close(u[..., 0, :], exact[..., 0, :], rtol=0, atol=1e-14)
    torch.testing.assert_close(u[..., -1, :], exact[..., -1, :], rtol=0, atol=1e-14)


def test_fd_residual_of_the_exact_solution_is_small_and_converges(problem, inputs):
    norms = []
    for resolution in (32, 64, 128, 256):
        grid = problem.grid(resolution).double()
        residual = problem.residual(problem.analytical(inputs, grid), inputs, grid)
        assert residual.shape == (6, 1, resolution - 2, resolution - 2)
        norms.append(residual.pow(2).mean().sqrt().item())
    # Second order: about a factor 4 per halving of the grid spacing.
    assert norms[1] < 2e-3
    for coarse, fine in zip(norms, norms[1:]):
        assert coarse / fine > 3.0
    # A wrong diffusivity is far from the discretization error.
    grid = problem.grid(64).double()
    wrong = inputs.clone()
    wrong[:, 1] *= 1.2
    assert problem.residual(problem.analytical(wrong, grid), inputs, grid).pow(2).mean().sqrt() > 10 * norms[1]
