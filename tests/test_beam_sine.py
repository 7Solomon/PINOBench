"""beam_1d with inputs = sine: sine-series load, closed-form solution, residual and configs."""

import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from pino.analytical.beam import simply_supported_beam_sine
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
    return compose_config("problem=beam_1d_sine").problem


@pytest.fixture(scope="module")
def problem(cfg):
    return build_problem(cfg)


@pytest.fixture
def inputs(problem):
    return problem.sample_inputs(6, torch.Generator().manual_seed(0)).double()


def test_sine_config_inherits_from_beam_1d(cfg):
    base = OmegaConf.load(CONFIGS / "problem" / "beam_1d.yaml")
    assert cfg.name == "beam_1d" and cfg.inputs == "sine" and cfg.output_param == "constrained"
    assert cfg.stiffness_low == base.stiffness_low and cfg.length == base.length
    assert list(cfg.output_scale_global) == [0.004, 0.04]


def test_run_directory_contains_the_variant():
    hydra_cfg = compose_config("problem=beam_1d_sine", "problem.output_param=unscaled", "training=physics")
    run_dir = hydra_cfg.hydra.run.dir
    assert run_dir.startswith("workspace/outputs/beam_1d/sine-unscaled/physics/")
    default = compose_config().hydra.run.dir
    assert default.startswith("workspace/outputs/heat_1d/sine-constrained/physics/")


def test_sampler_follows_the_configured_family(problem, cfg):
    sample = problem.sample_inputs(20000, torch.Generator().manual_seed(1))
    assert sample.shape == (20000, cfg.n_series + 1)
    n = torch.arange(1, cfg.n_series + 1)
    bound = 1.0 / n
    assert torch.all(sample[:, :-1].abs() <= bound) and torch.all(sample[:, :-1].abs().amax(0) > 0.99 * bound)
    assert sample[:, -1].min() >= cfg.stiffness_low and sample[:, -1].max() <= cfg.stiffness_high


def test_solution_satisfies_fourth_order_and_mixed_form(inputs):
    x = torch.rand(40, dtype=torch.float64).requires_grad_()
    for b in range(inputs.shape[0]):
        coefficients, stiffness = inputs[b : b + 1, :-1], inputs[b : b + 1, -1]
        w, m = simply_supported_beam_sine(coefficients, stiffness, x)
        w_x = torch.autograd.grad(w.sum(), x, create_graph=True)[0]
        w_xx = torch.autograd.grad(w_x.sum(), x, create_graph=True)[0]
        w_xxx = torch.autograd.grad(w_xx.sum(), x, create_graph=True)[0]
        w_xxxx = torch.autograd.grad(w_xxx.sum(), x, create_graph=True)[0]
        m_x = torch.autograd.grad(m.sum(), x, create_graph=True)[0]
        m_xx = torch.autograd.grad(m_x.sum(), x)[0]
        n = torch.arange(1, coefficients.shape[1] + 1, dtype=torch.float64)
        q = (coefficients * torch.sin(x[:, None] * n * math.pi)).sum(1)
        torch.testing.assert_close(stiffness * w_xxxx, q, rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(m_xx, -q, rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(w_xx, -m[0] / stiffness, rtol=1e-8, atol=1e-10)


def test_supports_are_free_of_deflection_and_moment(problem, inputs):
    u = problem.analytical(inputs, problem.grid(32).double())
    assert u[..., 0].abs().max() < 1e-15 and u[..., -1].abs().max() < 1e-15


def test_fd_residual_is_small_and_converges(problem, inputs):
    norms = []
    for resolution in (32, 64, 128, 256):
        grid = problem.grid(resolution).double()
        residual = problem.residual(problem.analytical(inputs, grid), inputs, grid)
        assert residual.shape == (6, 2, resolution - 2)
        norms.append(residual.pow(2).mean().sqrt().item())
    assert norms[1] < 2e-2
    for coarse, fine in zip(norms, norms[1:]):
        assert 3.5 < coarse / fine < 4.5
    grid = problem.grid(64).double()
    wrong = inputs.clone()
    wrong[:, -1] *= 1.3  # wrong stiffness: w off by 30 %
    assert problem.residual(problem.analytical(wrong, grid), inputs, grid).pow(2).mean().sqrt() > 10 * norms[1]


def test_constrained_scaling_and_network_channels(problem, inputs):
    grid = problem.grid(16)
    x = problem.to_fno_input(inputs.float(), grid)
    assert x.shape == (6, 3, 16)  # q(x), EI, x as for linear
    n = torch.arange(1, inputs.shape[1], dtype=torch.float64)
    load = (inputs[:, :-1, None] * torch.sin(n[:, None] * math.pi * grid[0].double())).sum(1)
    torch.testing.assert_close(x[:, 0].double(), load, rtol=1e-5, atol=1e-6)
    q_rms = (0.5 * inputs[:, :-1].pow(2).sum(1)).sqrt()
    scale = problem.output_scale(inputs, grid.double())
    torch.testing.assert_close(scale[:, 1, 0], q_rms / math.pi**2)
    torch.testing.assert_close(scale[:, 0, 0], q_rms / (math.pi**4 * inputs[:, -1]))
    model = build_model(MODEL_CFG, problem, 16)
    u = predict(model, problem, inputs.float(), grid)
    assert torch.all(u[..., 0] == 0) and torch.all(u[..., -1] == 0)


def test_linear_beam_is_unchanged_by_the_sine_code_path():
    linear = build_problem(OmegaConf.load(CONFIGS / "problem" / "beam_1d.yaml"))
    assert linear.inputs == "linear"
    inputs = linear.sample_inputs(4, torch.Generator().manual_seed(0))
    assert inputs.shape == (4, 3)
