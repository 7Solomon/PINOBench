"""heat_1d with inputs = sine_ramp: boundary values linear in time, lifting solution, constraint."""

import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from pino.analytical.diffusion import heat_ramp_series, heat_sine_series
from pino.evaluation import predict
from pino.models.factory import build_model
from pino.problems import build_problem

CONFIGS = Path(__file__).parents[1] / "configs"
MODEL_CFG = OmegaConf.create({"n_modes": 4, "hidden_channels": 4, "n_layers": 2, "domain_padding": 0.25})


def compose_config(*overrides: str):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides), return_hydra_config=True)


@pytest.fixture(scope="module")
def cfg():
    return compose_config("problem=heat_1d_sine_ramp").problem


@pytest.fixture(scope="module")
def problem(cfg):
    return build_problem(cfg)


@pytest.fixture
def inputs(problem):
    return problem.sample_inputs(6, torch.Generator().manual_seed(0)).double()


def test_config_inherits_from_heat_1d_and_names_the_run_directory(cfg):
    assert cfg.name == "heat_1d" and cfg.inputs == "sine_ramp" and cfg.n_series == 5 and cfg.t_end == 0.1
    assert cfg.loss_weights.bc == 1.0
    run_dir = compose_config("problem=heat_1d_sine_ramp", "training=physics").hydra.run.dir
    assert run_dir.startswith("workspace/outputs/heat_1d/sine_ramp-constrained/physics/")


def test_sampler_follows_the_configured_ranges(problem, cfg):
    sample = problem.sample_inputs(20000, torch.Generator().manual_seed(1))
    assert sample.shape == (20000, cfg.n_series + 4)
    assert sample[:, : cfg.n_series].abs().max() <= 1.0
    starts, rates = sample[:, [cfg.n_series, cfg.n_series + 2]], sample[:, [cfg.n_series + 1, cfg.n_series + 3]]
    assert starts.abs().max() <= 1.0 and starts.abs().max() > 0.99
    assert rates.abs().max() <= 10.0 and rates.abs().max() > 9.9


def _solution(inputs, x, t, problem, n_modes=200):
    n = problem.n_series
    return heat_ramp_series(inputs[:, :n], *inputs[:, n:].unbind(dim=1), x, t, problem.a, n_modes)


def _pde_error(problem, inputs, x0, t0, n_modes):
    worst = 0.0
    for b in range(inputs.shape[0]):
        x, t = x0.clone().requires_grad_(), t0.clone().requires_grad_()
        T = _solution(inputs[b : b + 1], x, t, problem, n_modes)
        T_x, T_t = torch.autograd.grad(T.sum(), (x, t), create_graph=True)
        (T_xx,) = torch.autograd.grad(T_x.sum(), x)
        worst = max(worst, (T_t - problem.a * T_xx).abs().max().item())
    return worst


def test_solution_solves_the_heat_equation_and_converges_in_the_modes(problem, inputs):
    """The pointwise T_xx of the ramp series decays only like 1/n (Gibbs tail of the source).

    The solution itself is accurate to 1e-5 with 200 modes (see the truncation test), but the
    PDE holds pointwise only in the limit, so the check is the convergence of the PDE error.
    """
    generator = torch.Generator().manual_seed(1)
    x0 = 0.05 + 0.9 * torch.rand(30, dtype=torch.float64, generator=generator)
    t0 = 0.02 + 0.08 * torch.rand(30, dtype=torch.float64, generator=generator)
    errors = [_pde_error(problem, inputs, x0, t0, n_modes) for n_modes in (200, 2000, 20000)]
    assert errors[1] < errors[0] / 5 and errors[2] < errors[1] / 5, errors
    assert errors[2] < 2e-3, errors


def test_initial_and_boundary_values(problem, inputs):
    grid = problem.grid(33).double()
    n = problem.n_series
    alpha_l, beta_l, alpha_r, beta_r = (inputs[:, n + i, None] for i in range(4))
    T = problem.analytical(inputs, grid)[:, 0]
    x, t = grid[0, :, 0], grid[1, 0, :]
    torch.testing.assert_close(T[:, 0], alpha_l + beta_l * t, rtol=0, atol=1e-11)
    torch.testing.assert_close(T[:, -1], alpha_r + beta_r * t, rtol=0, atol=1e-11)
    initial = (1 - x) * alpha_l + x * alpha_r + heat_sine_series(inputs[:, :n], x, torch.zeros_like(x), 0.0)
    torch.testing.assert_close(T[:, :, 0], initial, rtol=0, atol=1e-13)


def test_constant_rates_reach_the_steady_linear_ramp():
    """b = 0, equal rates beta: T = alpha + beta t minus a parabola; at large t the known quasi-steady profile."""
    beta = torch.tensor([3.0], dtype=torch.float64)
    x = torch.linspace(0, 1, 21, dtype=torch.float64)
    t = torch.full_like(x, 5.0)  # all modes have decayed
    T = heat_ramp_series(torch.zeros(1, 1, dtype=torch.float64), 0 * beta, beta, 0 * beta, beta, x, t, 1.0, 400)
    # dw/dt ~ 0: a w_xx = beta, w(0) = w(1) = 0, so w = -beta x (1 - x) / (2 a).
    expected = beta * 5.0 - beta * x * (1 - x) / 2.0
    torch.testing.assert_close(T[0], expected, rtol=0, atol=1e-5)


def test_truncation_error_is_negligible(problem, inputs):
    grid = problem.grid(33).double()
    coarse = _solution(inputs, grid[0], grid[1], problem, 200)
    fine = _solution(inputs, grid[0], grid[1], problem, 2000)
    assert (coarse - fine).abs().max() < 2e-5


def test_fd_residual_is_small_and_converges(problem, inputs):
    norms = []
    for n in (32, 64, 128):
        grid = problem.grid(n).double()
        norms.append(problem.residual(problem.analytical(inputs, grid), inputs, grid).pow(2).mean().sqrt().item())
    rates = [math.log2(norms[i] / norms[i + 1]) for i in range(2)]
    assert all(rate > 1.7 for rate in rates), (norms, rates)
    assert norms[1] < 0.01


def test_constraint_reproduces_initial_and_boundary_values(problem, inputs):
    grid = problem.grid(16).double()
    exact = problem.analytical(inputs, grid)
    g, phi = problem.boundary(inputs, grid)
    u = g + phi * torch.randn_like(exact)
    tolerance = {"rtol": 0, "atol": 1e-11}
    torch.testing.assert_close(u[..., 0], exact[..., 0], **tolerance)
    torch.testing.assert_close(u[:, :, 0], exact[:, :, 0], **tolerance)
    torch.testing.assert_close(u[:, :, -1], exact[:, :, -1], **tolerance)
    mask = problem.boundary_values(inputs, grid)[1][0, 0]
    assert mask[0].all() and mask[-1].all() and mask[:, 0].all() and not mask[1:-1, 1:].any()


def test_network_inputs_and_shapes(problem, inputs):
    grid = problem.grid((16, 12))
    x = problem.to_fno_input(inputs.float(), grid)
    assert problem.in_channels == 5 and x.shape == (6, 5, 16, 12)
    # T0 at t = 0 and the boundary values on the edges agree with the exact solution.
    exact = problem.analytical(inputs.float(), grid)
    torch.testing.assert_close(x[:, 0, :, 3], exact[:, 0, :, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(x[:, 1, 5, :], exact[:, 0, 0, :], rtol=0, atol=1e-5)
    torch.testing.assert_close(x[:, 2, 5, :], exact[:, 0, -1, :], rtol=0, atol=1e-5)
    assert torch.equal(x[0, 3], grid[0]) and x[0, 4, 0, -1] == pytest.approx(1.0)
    model = build_model(MODEL_CFG, problem, (16, 12))
    assert predict(model, problem, inputs.float(), grid).shape == (6, 1, 16, 12)


def test_default_family_is_unchanged():
    problem = build_problem(OmegaConf.load(CONFIGS / "problem" / "heat_1d.yaml"))
    assert problem.inputs == "sine" and problem.in_channels == 3
    inputs = problem.sample_inputs(4, torch.Generator().manual_seed(0))
    assert inputs.shape == (4, 5)
    generator = torch.Generator().manual_seed(0)
    torch.testing.assert_close(inputs, -1.0 + 2.0 * torch.rand(4, 5, generator=generator), rtol=0, atol=0)
