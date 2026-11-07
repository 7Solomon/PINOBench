"""chloride_corner_2d: product erf solution, boundary values, constraint and boundary layer."""

import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from pino.analytical.diffusion import chloride_corner, crank_erfc
from pino.evaluation import predict
from pino.models.factory import build_model
from pino.numerics.grids import grid_spacing
from pino.problems import build_problem

CFG = OmegaConf.load(Path(__file__).parents[1] / "configs" / "problem" / "chloride_corner_2d.yaml")
MODEL_CFG = OmegaConf.create({"n_modes": 4, "hidden_channels": 4, "n_layers": 2, "domain_padding": 0.25})


@pytest.fixture
def problem():
    return build_problem(CFG)


@pytest.fixture
def inputs(problem):
    return problem.sample_inputs(4, torch.Generator().manual_seed(0)).double()


def test_product_solution_solves_the_2d_diffusion_equation():
    surface = torch.tensor([0.5, 0.8, 1.0], dtype=torch.float64)
    diffusivity = torch.tensor([0.4, 0.7, 1.0], dtype=torch.float64)
    x, y = torch.rand(50, dtype=torch.float64, requires_grad=True), torch.rand(50, dtype=torch.float64, requires_grad=True)
    t = (CFG.t_start + (CFG.t_end - CFG.t_start) * torch.rand(50, dtype=torch.float64)).requires_grad_()
    for i in range(3):
        c = chloride_corner(surface[i : i + 1], diffusivity[i : i + 1], x, y, t)
        c_x, c_y, c_t = torch.autograd.grad(c.sum(), (x, y, t), create_graph=True)
        (c_xx,) = torch.autograd.grad(c_x.sum(), x, retain_graph=True)
        (c_yy,) = torch.autograd.grad(c_y.sum(), y)
        torch.testing.assert_close(c_t, diffusivity[i] * (c_xx + c_yy), rtol=1e-9, atol=1e-9)


def test_exposed_faces_and_far_faces(problem, inputs):
    grid = problem.grid(17).double()
    c = problem.analytical(inputs, grid)[:, 0]
    surface, diffusivity = inputs[:, 0], inputs[:, 1]
    cs = surface[:, None, None]
    torch.testing.assert_close(c[:, 0], cs.expand_as(c[:, 0]), rtol=0, atol=1e-15)  # x = 0
    torch.testing.assert_close(c[:, :, 0], cs.expand_as(c[:, :, 0]), rtol=0, atol=1e-15)  # y = 0
    # x = 1 is not zero: C ~ Cs erfc(y / (2 sqrt(D t))), since erf(1 / (2 sqrt(D t))) ~ 1.
    y, t = grid[1, 0], grid[2, 0]
    far = crank_erfc(surface, diffusivity, y, t)
    assert c[:, -1].abs().max() > 0.4
    torch.testing.assert_close(c[:, -1], far, rtol=0, atol=2e-2)


def test_boundary_layer_is_resolved_on_the_training_grid(problem):
    """The thinnest front, at D_min and t0, has width 2 sqrt(D t0) = 0.179 and needs >= 5 points."""
    width = 2.0 * math.sqrt(CFG.diffusivity_low * CFG.t_start)
    assert width == pytest.approx(0.1789, abs=1e-3)
    grid = problem.grid(CFG.train_resolution)
    for axis in (0, 1):
        assert width / grid_spacing(grid, axis) >= 5


def test_reference_time_in_years(problem):
    assert problem.scales["t"] == pytest.approx(1267.5, abs=0.1)
    assert problem.to_physical("t", CFG.t_start) == pytest.approx(25.35, abs=0.05)
    assert problem.to_physical("t", CFG.t_end) == pytest.approx(63.4, abs=0.1)


def test_fd_residual_of_the_exact_solution_is_small_and_converges(problem, inputs):
    norms = []
    for n in (16, 32, 64):
        grid = problem.grid(n).double()
        norms.append(problem.residual(problem.analytical(inputs, grid), inputs, grid).pow(2).mean().sqrt().item())
    rates = [math.log2(norms[i] / norms[i + 1]) for i in range(2)]
    assert all(rate > 1.7 for rate in rates), (norms, rates)
    assert norms[1] < 0.05


def test_constraint_reproduces_initial_and_boundary_values(problem, inputs):
    grid = problem.grid(16).double()
    exact = problem.analytical(inputs, grid)
    g, phi = problem.boundary(inputs, grid)
    u = g + phi * torch.randn_like(exact)
    tol = {"rtol": 0, "atol": 1e-12}
    torch.testing.assert_close(u[..., 0], exact[..., 0], **tol)
    for index in ((0,), (-1,)):
        torch.testing.assert_close(u[:, :, index[0]], exact[:, :, index[0]], **tol)
        torch.testing.assert_close(u[:, :, :, index[0]], exact[:, :, :, index[0]], **tol)


def test_shapes_and_fno_input(problem, inputs):
    grid = problem.grid((16, 12, 10))
    x = problem.to_fno_input(inputs.float(), grid)
    assert x.shape == (4, 5, 16, 12, 10)
    assert torch.all(x[:, 0] == inputs[:, 0, None, None, None].float())
    assert torch.all(x[:, 1] == inputs[:, 1, None, None, None].float())
    assert torch.equal(x[0, 2], grid[0]) and torch.equal(x[0, 3], grid[1])
    assert x[0, 4, 0, 0, 0] == 0 and x[0, 4, 0, 0, -1] == pytest.approx(1.0)
    model = build_model(MODEL_CFG, problem, (16, 12, 10))
    assert predict(model, problem, inputs.float(), grid).shape == (4, 1, 16, 12, 10)


def test_other_input_families_are_rejected():
    with pytest.raises(ValueError, match="chloride_corner_2d does not implement inputs='rich'"):
        build_problem(OmegaConf.merge(CFG, {"inputs": "rich"}))
