"""darcy_poisson_2d: sine-series solution of -Laplace h = f, constraint and residual."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from pino.analytical.porous_media import poisson_sine_head_2d, sine_source_2d
from pino.evaluation import predict
from pino.models.factory import build_model
from pino.problems import build_problem

CFG = OmegaConf.load(Path(__file__).parents[1] / "configs" / "problem" / "darcy_poisson_2d.yaml")


@pytest.fixture
def problem():
    return build_problem(CFG)


@pytest.fixture
def inputs(problem):
    return problem.sample_inputs(4, torch.Generator().manual_seed(0)).double()


def test_solution_satisfies_the_poisson_equation(inputs):
    x = torch.rand(40, dtype=torch.float64, requires_grad=True)
    y = torch.rand(40, dtype=torch.float64, requires_grad=True)
    for i in range(inputs.shape[0]):
        a = inputs[i : i + 1]
        h = poisson_sine_head_2d(a, x, y).sum()
        h_x, h_y = torch.autograd.grad(h, (x, y), create_graph=True)
        (h_xx,) = torch.autograd.grad(h_x.sum(), x, retain_graph=True)
        (h_yy,) = torch.autograd.grad(h_y.sum(), y)
        torch.testing.assert_close(-(h_xx + h_yy), sine_source_2d(a, x, y)[0], rtol=1e-10, atol=1e-10)


def test_solution_vanishes_on_the_boundary(problem, inputs):
    h = problem.analytical(inputs, problem.grid(32).double())
    for edge in (h[..., 0, :], h[..., -1, :], h[..., :, 0], h[..., :, -1]):
        assert edge.abs().max() < 1e-14


def test_single_mode_has_the_expected_amplitude():
    a = torch.zeros(1, 4, 4, dtype=torch.float64)
    a[0, 1, 2] = 1.0  # m = 2, n = 3
    value = poisson_sine_head_2d(a, torch.tensor([0.25], dtype=torch.float64), torch.tensor([1 / 6], dtype=torch.float64))
    assert value.item() == pytest.approx(1.0 / (torch.pi**2 * 13))


def test_constraint_and_shapes(problem, inputs):
    grid = problem.grid((16, 12)).double()
    g, phi = problem.boundary(inputs, grid)
    assert torch.all(g == 0) and phi.shape == (1, 1, 16, 12)
    x = problem.to_fno_input(inputs, grid)
    assert x.shape == (4, problem.in_channels, 16, 12)
    assert torch.equal(x[1, 1], grid[0]) and torch.equal(x[1, 2], grid[1])
    model = build_model(OmegaConf.create({"n_modes": 4, "hidden_channels": 4, "n_layers": 2, "domain_padding": 0.25}), problem, 16)
    prediction = predict(model, problem, inputs.float(), grid.float())
    assert prediction.shape == (4, 1, 16, 12)
    assert prediction[..., 0, :].abs().max() == 0 and prediction[..., :, -1].abs().max() == 0


def test_fd_residual_of_the_exact_solution_is_small_and_converges(problem, inputs):
    norms = []
    for resolution in (32, 64, 128, 256):
        grid = problem.grid(resolution).double()
        residual = problem.residual(problem.analytical(inputs, grid), inputs, grid)
        assert residual.shape == (4, 1, resolution - 2, resolution - 2)
        norms.append(residual.pow(2).mean().sqrt().item())
    assert norms[1] < 1e-2
    for coarse, fine in zip(norms, norms[1:]):
        assert 3.5 < coarse / fine < 4.5
    # Halving the solution is far from the discretization error.
    grid = problem.grid(64).double()
    wrong = problem.residual(0.5 * problem.analytical(inputs, grid), inputs, grid)
    assert wrong.pow(2).mean().sqrt() > 10 * norms[1]


def test_model_factory_checks_modes_on_both_axes(problem):
    cfg = OmegaConf.create({"n_modes": [16, 40], "hidden_channels": 4, "n_layers": 1, "domain_padding": 0.0})
    with pytest.raises(ValueError, match="n_modes=40"):
        build_model(cfg, problem, (64, 64))
    cfg.n_modes = 16
    assert tuple(build_model(cfg, problem, 64).n_modes) == (16, 16)
