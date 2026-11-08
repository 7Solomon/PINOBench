"""elastic_block_2d: Neumann loss, constraints per channel, residual, training and plots."""

import json
import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from pino.evaluation import evaluate_model, predict, write_metrics
from pino.losses import boundary_loss, compute_loss, neumann_active, neumann_loss, resolve_loss_weights
from pino.models.factory import build_model
from pino.problems import build_problem
from pino.trainer import selection_criterion, train
from pino.visualization import plot_run, plot_training

CONFIGS = Path(__file__).parents[1] / "configs"
MODEL_CFG = {"name": "tiny", "n_modes": 4, "hidden_channels": 4, "n_layers": 2, "domain_padding": 0.25}


def problem_cfg(**overrides):
    cfg = OmegaConf.load(CONFIGS / "problem" / "elastic_block_2d.yaml")
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


@pytest.fixture
def problem():
    return build_problem(problem_cfg())


@pytest.fixture
def inputs(problem):
    return problem.sample_inputs(6, torch.Generator().manual_seed(0)).double()


def test_zero_displacement_has_zero_residual_but_large_neumann_loss(problem, inputs):
    """The load enters only through the top edge: without L_neumann u = v = 0 is a solution."""
    grid = problem.grid(64).double()
    zero = torch.zeros(inputs.shape[0], 2, 64, 64, dtype=torch.float64)
    assert problem.residual(zero, inputs, grid).abs().max() == 0
    assert neumann_loss(problem, zero, inputs, grid) > 0.05
    exact = problem.analytical(inputs, grid)
    assert problem.residual(exact, inputs, grid).pow(2).mean() < 1e-4
    assert neumann_loss(problem, exact, inputs, grid) < 1e-5


def test_fd_residual_and_neumann_loss_of_the_exact_solution_converge(problem, inputs):
    residuals, neumann = [], []
    for n in (32, 64, 128, 256):
        grid = problem.grid(n).double()
        exact = problem.analytical(inputs, grid)
        residuals.append(problem.residual(exact, inputs, grid).pow(2).mean().sqrt().item())
        neumann.append(neumann_loss(problem, exact, inputs, grid).sqrt().item())
    for values in (residuals, neumann):
        rates = [math.log2(values[i] / values[i + 1]) for i in range(3)]
        assert all(rate > 1.5 for rate in rates), (values, rates)


def test_one_neumann_condition_by_hand(problem, inputs):
    """The conditions of the task, with -p(x) as the target of sigma_yy on top."""
    grid = problem.grid(16).double()
    conditions = problem.neumann_conditions(inputs, grid)
    assert [c.label for c in conditions] == ["left:sigma_xy", "right:sigma_xy", "bottom:sigma_xy", "top:sigma_xy", "top:sigma_yy"]
    target = conditions[-1].target
    x = grid[0, :, 0]
    load = inputs[:, 0, None] + sum(inputs[:, n, None] * torch.cos(n * math.pi * x) for n in range(1, 7))
    torch.testing.assert_close(target, -load, rtol=1e-12, atol=1e-12)


def test_masks_per_channel(problem, inputs):
    grid = problem.grid(16)
    g, mask = problem.boundary_values(inputs.float(), grid)
    assert mask.shape == (1, 2, 16, 16) and g.shape == (6, 2, 16, 16)
    expected_u = torch.zeros(16, 16, dtype=torch.bool)
    expected_u[0], expected_u[-1] = True, True
    expected_v = torch.zeros(16, 16, dtype=torch.bool)
    expected_v[:, 0] = True
    assert torch.equal(mask[0, 0], expected_u) and torch.equal(mask[0, 1], expected_v)


def test_soft_penalty_sees_each_channel_only_on_its_own_supports():
    problem = build_problem(problem_cfg(output_param="soft"))
    grid = problem.grid(16)
    inputs = problem.sample_inputs(2, torch.Generator().manual_seed(0))
    field = torch.zeros(2, 2, 16, 16)
    field[:, 0, 5:9, :] = 1.0  # u in the interior of x, also on the top and bottom edge
    field[:, 1, :, 5:] = 1.0  # v away from the bottom, also on the left and right edge
    assert boundary_loss(problem, field, inputs, grid) == 0
    field[:, 0, 0, 3] = 1.0  # u on the left edge
    assert boundary_loss(problem, field, inputs, grid) > 0


def test_constrained_satisfies_the_supports_for_any_network(problem):
    grid = problem.grid(16)
    inputs = problem.sample_inputs(3, torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    model = build_model(OmegaConf.create(MODEL_CFG), problem, 16)
    u = predict(model, problem, inputs, grid)
    assert u[:, 0, 0].abs().max() == 0 and u[:, 0, -1].abs().max() == 0 and u[:, 1, :, 0].abs().max() == 0
    assert u[:, 0, 5, 5].abs().max() > 0 and u[:, 1, 5, -1].abs().max() > 0


def test_scales_and_main_variable(problem):
    assert problem.main_variable == "v" and problem.main_channel == 1
    assert problem.scales["u"] == pytest.approx(2.0 * 10.0 / 30.0)
    assert problem.to_physical("sigma", 1.0) == 10.0
    assert list(problem.global_output_scale(torch.zeros(1)).flatten()) == pytest.approx([0.019, 0.42])
    with pytest.raises(ValueError, match="does not implement inputs='rich'"):
        build_problem(problem_cfg(inputs="rich"))


def test_network_inputs(problem, inputs):
    grid = problem.grid((16, 12))
    x = problem.to_fno_input(inputs.float(), grid)
    assert x.shape == (6, 4, 16, 12)
    assert torch.equal(x[:, 0, :, 0], x[:, 0, :, -1])  # p(x) broadcast over y
    assert torch.all(x[:, 1] == inputs[:, -1, None, None].float())
    assert torch.equal(x[0, 2], grid[0]) and torch.equal(x[0, 3], grid[1])


def test_neumann_is_inactive_in_data_unless_set_explicitly():
    for mode, expected in (("physics", 10.0), ("hybrid", 10.0), ("data", 0.0)):
        assert OmegaConf.load(CONFIGS / "training" / f"{mode}.yaml").weights.neumann == expected
    block = OmegaConf.create({"loss_weights": {"neumann": 100.0}})
    assert resolve_loss_weights(OmegaConf.load(CONFIGS / "training" / "physics.yaml"), block)["neumann"] == 100.0
    assert resolve_loss_weights(OmegaConf.load(CONFIGS / "training" / "data.yaml"), block)["neumann"] == 0.0
    training = OmegaConf.load(CONFIGS / "training" / "physics.yaml")
    training.weights.neumann = 3.0
    assert resolve_loss_weights(training, block, {"training.weights.neumann"})["neumann"] == 3.0
    problem = build_problem(problem_cfg())
    assert neumann_active(problem, OmegaConf.create({"neumann": 1.0}))
    assert not neumann_active(problem, OmegaConf.create({"neumann": 0.0}))
    heat = build_problem(OmegaConf.load(CONFIGS / "problem" / "heat_1d.yaml"))
    assert not heat.has_neumann and not neumann_active(heat, OmegaConf.create({"neumann": 1.0}))


def test_compute_loss_adds_the_weighted_neumann_term(problem, inputs):
    grid = problem.grid(16).double()
    field = problem.analytical(inputs, grid) * 0.5
    weights = OmegaConf.create({"physics": 1.0, "data": 0.0, "bc": 1.0, "neumann": 7.0})
    total, components = compute_loss("physics", weights, problem, field, inputs, grid)
    assert components["neumann"] > 0 and "bc" not in components
    assert components["total"] == pytest.approx(components["physics"] + 7.0 * components["neumann"])
    weights.neumann = 0.0
    assert "neumann" not in compute_loss("physics", weights, problem, field, inputs, grid)[1]


def test_selection_includes_the_neumann_term_in_physics():
    assert selection_criterion("physics", neumann=True) == "val_objective"
    assert selection_criterion("data", neumann=True) == "val_rel_l2_all"


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
    assert len(history["val_neumann"]) == 2  # for observation in every mode
    if mode == "physics":
        assert history["selection"] == "val_objective"
    model.load_state_dict(result.best_state)
    evaluation = evaluate_model(model, problem, cfg, torch.device("cpu"))
    metrics = write_metrics(tmp_path / "metrics.json", evaluation, problem, cfg, 1.0, 1, 1, "now", history)
    assert set(metrics["neumann_error_mean"]) == {"left:sigma_xy", "right:sigma_xy", "bottom:sigma_xy", "top:sigma_xy", "top:sigma_yy"}
    assert set(metrics["rel_l2_stress"]) == {"sigma_xx", "sigma_yy", "sigma_xy"}
    assert metrics["main_variable"] == "v" and set(metrics["rel_l2_per_variable"]) == {"u", "v"}
    plot_run(problem, evaluation, history, mode, 16, tmp_path)
    for name in ("training.png", "solution.png", "deformation.png", "stress.png", "resolution.png", "error_hist.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_stress_metric_is_small_for_the_exact_displacements(problem, inputs):
    grid = problem.grid(64).double()
    exact = problem.analytical(inputs, grid)
    fd = problem.stress_fields(exact, inputs, grid)
    analytical = problem.exact_stress_fields(inputs, grid)
    for name, field in fd.items():
        relative = (field - analytical[name]).flatten(1).norm(dim=1) / analytical[name].flatten(1).norm(dim=1)
        assert relative.max() < 0.03, name
