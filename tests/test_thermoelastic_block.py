"""thermoelastic_block_2d: coupling, conditions per channel, residual, training and plots."""

import json
import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from pino.evaluation import evaluate_model, predict, write_metrics
from pino.losses import boundary_loss, neumann_loss
from pino.models.factory import build_model
from pino.problems import build_problem
from pino.trainer import train
from pino.visualization import plot_run

CONFIGS = Path(__file__).parents[1] / "configs"
MODEL_CFG = {"name": "tiny", "n_modes": 4, "hidden_channels": 4, "n_layers": 2, "domain_padding": 0.25}
CONDITIONS = {"left:T_x", "right:T_x", "left:sigma_xy", "right:sigma_xy", "bottom:sigma_xy", "top:sigma_xy", "top:sigma_yy"}


def problem_cfg(**overrides):
    cfg = OmegaConf.load(CONFIGS / "problem" / "thermoelastic_block_2d.yaml")
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


@pytest.fixture
def problem():
    return build_problem(problem_cfg())


@pytest.fixture
def inputs(problem):
    return problem.sample_inputs(6, torch.Generator().manual_seed(0)).double()


def test_fd_residual_and_neumann_loss_of_the_exact_solution_converge(problem, inputs):
    residuals, neumann = [], []
    torch.set_default_dtype(torch.float64)  # a float32 grid adds coordinate noise, amplified by 1 / h^2
    for n in (32, 64, 128, 256):
        grid = problem.grid(n)
        exact = problem.analytical(inputs, grid)
        residuals.append(problem.residual(exact, inputs, grid).pow(2).mean().sqrt().item())
        neumann.append(neumann_loss(problem, exact, inputs, grid).sqrt().item())
    torch.set_default_dtype(torch.float32)
    for values in (residuals, neumann):
        rates = [math.log2(values[i] / values[i + 1]) for i in range(3)]
        assert all(rate > 1.5 for rate in rates), (values, rates)


def test_residual_has_three_equations_and_couples_the_temperature(problem, inputs):
    grid = problem.grid(32).double()
    exact = problem.analytical(inputs, grid)
    assert problem.residual(exact, inputs, grid).shape == (6, 3, 30, 30)
    # A wrong temperature with exact displacements shows up in the equilibrium as well.
    wrong = exact.clone()
    wrong[:, 0] += torch.sin(math.pi * grid[0]) * torch.sin(math.pi * grid[1]) * 0.1
    change = (problem.residual(wrong, inputs, grid) - problem.residual(exact, inputs, grid)).abs().amax(dim=(0, 2, 3))
    assert all(change > 0.01)


def test_equations_are_balanced_for_a_wrong_field(problem, inputs):
    """Temperature only from g, no displacements: no equation is 10 times the other."""
    grid = problem.grid(64).double()
    g, _ = problem.boundary(inputs, grid)
    rms = problem.residual(g, inputs, grid).pow(2).mean(dim=(0, 2, 3)).sqrt()
    assert rms.max() / rms.min() < 3, rms


def test_a_heated_block_at_rest_is_not_a_solution(problem):
    """With T correct and u = v = 0 both the residual and L_neumann are far from their exact values."""
    grid = problem.grid(32).double()
    heated = problem.sample_inputs(4, torch.Generator().manual_seed(0)).double()
    exact = problem.analytical(heated, grid)
    at_rest = exact.clone()
    at_rest[:, 1:] = 0.0
    assert problem.residual(at_rest, heated, grid).pow(2).mean() > 100 * problem.residual(exact, heated, grid).pow(2).mean()
    assert neumann_loss(problem, at_rest, heated, grid) > 100 * neumann_loss(problem, exact, heated, grid)


def test_neumann_conditions(problem, inputs):
    grid = problem.grid(16).double()
    assert {c.label for c in problem.neumann_conditions(inputs, grid)} == CONDITIONS
    assert all(c.target == 0.0 for c in problem.neumann_conditions(inputs, grid))
    assert neumann_loss(problem, problem.analytical(inputs, problem.grid(128).double()), inputs, problem.grid(128).double()) < 1e-5


def test_masks_per_channel(problem, inputs):
    grid = problem.grid(16)
    g, mask = problem.boundary_values(inputs.float(), grid)
    assert mask.shape == (1, 3, 16, 16) and g.shape == (6, 3, 16, 16)
    expected_t = torch.zeros(16, 16, dtype=torch.bool)
    expected_t[:, 0], expected_t[:, -1] = True, True
    expected_u = torch.zeros(16, 16, dtype=torch.bool)
    expected_u[0], expected_u[-1] = True, True
    expected_v = torch.zeros(16, 16, dtype=torch.bool)
    expected_v[:, 0] = True
    assert torch.equal(mask[0, 0], expected_t) and torch.equal(mask[0, 1], expected_u) and torch.equal(mask[0, 2], expected_v)


def test_soft_penalty_sees_each_channel_only_on_its_own_edges():
    problem = build_problem(problem_cfg(output_param="soft"))
    grid = problem.grid(16)
    inputs = problem.sample_inputs(2, torch.Generator().manual_seed(0))
    exact = problem.analytical(inputs, grid)
    assert boundary_loss(problem, exact, inputs, grid) < 1e-10
    field = exact.clone()
    field[:, 0, :, 1:-1] += 1.0  # T in the interior of y, also on the left and right edge
    field[:, 1, 1:-1] += 1.0  # u in the interior of x
    field[:, 2, :, 1:] += 1.0  # v away from the bottom
    assert boundary_loss(problem, field, inputs, grid) < 1e-10
    field[:, 0, 3, -1] += 1.0  # T on the top edge
    assert boundary_loss(problem, field, inputs, grid) > 1e-3


def test_constrained_satisfies_the_edges_for_any_network(problem):
    grid = problem.grid(16)
    inputs = problem.sample_inputs(3, torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    model = build_model(OmegaConf.create(MODEL_CFG), problem, 16)
    u = predict(model, problem, inputs, grid)
    exact = problem.analytical(inputs, grid)
    assert u[:, 0, :, 0].abs().max() == 0
    torch.testing.assert_close(u[:, 0, :, -1], exact[:, 0, :, -1], rtol=0, atol=1e-5)
    assert u[:, 1, 0].abs().max() == 0 and u[:, 1, -1].abs().max() == 0 and u[:, 2, :, 0].abs().max() == 0
    assert u[:, 1, 5, 5].abs().max() > 0 and u[:, 2, 5, -1].abs().max() > 0


def test_scales_and_main_variable(problem):
    assert problem.main_variable == "v" and problem.main_channel == 2
    assert problem.to_physical("T", 1.0) == 20.0
    assert problem.to_physical("u", 1.0) == pytest.approx(0.4)  # 2 m * 1e-5 / K * 20 K in mm
    assert problem.to_physical("sigma", 1.0) == pytest.approx(6.0)  # 30 GPa * 1e-5 / K * 20 K in MPa
    assert list(problem.global_output_scale(torch.zeros(1)).flatten()) == pytest.approx([0.45, 0.026, 0.22])


def test_network_inputs(problem, inputs):
    grid = problem.grid((16, 12))
    x = problem.to_fno_input(inputs.float(), grid)
    assert x.shape == (6, 4, 16, 12)
    assert torch.equal(x[:, 0, :, 0], x[:, 0, :, -1])  # T_top(x) broadcast over y
    assert torch.all(x[:, 1] == inputs[:, -1, None, None].float())


def test_stress_metric_is_small_where_the_stress_is_large(problem, inputs):
    grid = problem.grid(64).double()
    exact = problem.analytical(inputs, grid)
    fd = problem.stress_fields(exact, inputs, grid)
    analytical = problem.exact_stress_fields(inputs, grid)
    relative = (fd["sigma_xx"] - analytical["sigma_xx"]).flatten(1).norm(dim=1) / analytical["sigma_xx"].flatten(1).norm(dim=1)
    assert relative.max() < 0.01
    # sigma_yy and sigma_xy are about 100 times smaller: only their absolute error is small.
    for name in ("sigma_yy", "sigma_xy"):
        assert (fd[name] - analytical[name]).pow(2).mean().sqrt() < 0.005, name


def _cfg(mode: str, output_param: str):
    training = OmegaConf.load(CONFIGS / "training" / f"{mode}.yaml")
    training.update({"epochs": 2, "n_train": 4, "n_val": 2, "n_test": 3, "batch_size": 2, "eval_every": 1})
    problem = problem_cfg(output_param=output_param)
    problem.update({"train_resolution": 16, "test_resolution": 16, "resolution_study": [16, 32]})
    return OmegaConf.create({"seed": 0, "problem": problem, "training": training, "model": MODEL_CFG})


@pytest.mark.parametrize("mode,output_param", [("physics", "constrained"), ("data", "constrained"), ("hybrid", "constrained"), ("physics", "soft"), ("data", "soft"), ("data", "direct")])
def test_training_metrics_and_plots(mode, output_param, tmp_path):
    cfg = _cfg(mode, output_param)
    problem = build_problem(cfg.problem)
    model = build_model(cfg.model, problem, 16)
    result = train(model, problem, cfg, torch.device("cpu"), tmp_path)
    history = json.loads((tmp_path / "history.json").read_text())
    assert ("neumann" in history["losses"]) == (mode != "data")
    model.load_state_dict(result.best_state)
    evaluation = evaluate_model(model, problem, cfg, torch.device("cpu"))
    metrics = write_metrics(tmp_path / "metrics.json", evaluation, problem, cfg, 1.0, 1, 1, "now", history)
    assert set(metrics["neumann_error_mean"]) == CONDITIONS
    assert set(metrics["rel_l2_stress"]) == set(metrics["stress_rmse"]) == {"sigma_xx", "sigma_yy", "sigma_xy"}
    assert metrics["main_variable"] == "v" and set(metrics["rel_l2_per_variable"]) == {"T", "u", "v"}
    plot_run(problem, evaluation, history, mode, 16, tmp_path)
    for name in ("training.png", "solution.png", "deformation.png", "stress.png", "resolution.png", "error_hist.png"):
        assert (tmp_path / name).stat().st_size > 0
