"""output_param = soft: boundary mask, L_bc, weights, boundary error and training in all modes."""

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from pino.data.dataset import build_dataset
from pino.evaluation import boundary_errors, evaluate_model, predict, write_metrics
from pino.losses import boundary_loss, compute_loss, resolve_loss_weights
from pino.models.factory import build_model
from pino.problems import available_problems, build_problem
from pino.trainer import selection_criterion, train
from pino.visualization import plot_training

CONFIGS = Path(__file__).parents[1] / "configs"
MODEL_CFG = OmegaConf.create({"n_modes": 4, "hidden_channels": 4, "n_layers": 2, "domain_padding": 0.25})
#: Last grid axis is time for these plot types; time has a boundary only at its start.
TIME_DEPENDENT = ("1d_t", "2d_t")


def problem_cfg(name: str, **overrides):
    cfg = OmegaConf.load(CONFIGS / "problem" / f"{name}.yaml")
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def expected_mask(problem, grid: torch.Tensor) -> torch.Tensor:
    """Boundary points from the coordinates: both ends of every space axis, the first time slice."""
    n_space = problem.n_dims - (1 if problem.plot_type in TIME_DEPENDENT else 0)
    mask = torch.zeros(grid.shape[1:], dtype=torch.bool)
    for axis in range(problem.n_dims):
        index = [slice(None)] * problem.n_dims
        ends = (0, -1) if axis < n_space else (0,)
        for end in ends:
            index[axis] = end
            mask[tuple(index)] = True
    return mask


@pytest.mark.parametrize("name", available_problems())
@pytest.mark.parametrize("resolution", [8, 17])
def test_boundary_mask_hits_exactly_the_boundary_and_initial_points(name, resolution):
    problem = build_problem(problem_cfg(name))
    grid = problem.grid(resolution)
    inputs = problem.sample_inputs(2, torch.Generator().manual_seed(0))
    g, mask = problem.boundary_values(inputs, grid)
    assert mask.dtype == torch.bool and mask.shape == (1, 1, *grid.shape[1:])
    assert torch.equal(mask[0, 0], expected_mask(problem, grid))
    assert g.shape == (2, problem.out_channels, *grid.shape[1:])


@pytest.mark.parametrize("name", available_problems())
def test_boundary_values_match_the_analytical_solution(name):
    """g on the mask is what L_bc pulls towards; it must be the exact boundary data."""
    problem = build_problem(problem_cfg(name))
    grid = problem.grid(16)
    inputs = problem.sample_inputs(3, torch.Generator().manual_seed(1)).double()
    exact = problem.analytical(inputs, grid.double())
    g, mask = problem.boundary_values(inputs, grid.double())
    deviation = ((exact - g).abs() * mask).amax()
    assert deviation < 1e-10 * exact.abs().amax()


@pytest.mark.parametrize("name", available_problems())
def test_soft_prediction_is_c_times_network_without_boundary_values(name):
    problem = build_problem(problem_cfg(name, output_param="soft"))
    grid = problem.grid(16)
    inputs = problem.sample_inputs(2, torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    model = build_model(MODEL_CFG, problem, 16)
    with torch.no_grad():
        raw = model(problem.to_fno_input(inputs, grid))
        assert torch.equal(predict(model, problem, inputs, grid), problem.global_output_scale(grid) * raw)


@pytest.mark.parametrize("mode", ["physics", "data", "hybrid"])
def test_soft_is_allowed_in_every_mode(mode):
    build_problem(problem_cfg("heat_1d", output_param="soft")).check_mode(mode)


def test_boundary_loss_sees_only_the_boundary_and_is_normalized():
    problem = build_problem(problem_cfg("beam_1d", output_param="soft"))
    grid = problem.grid(16)
    inputs = problem.sample_inputs(4, torch.Generator().manual_seed(0))
    exact = problem.analytical(inputs, grid)
    assert boundary_loss(problem, exact, inputs, grid) < 1e-12
    # Interior changes are invisible to L_bc.
    interior = exact.clone()
    interior[..., 1:-1] += 1.0
    assert boundary_loss(problem, interior, inputs, grid) < 1e-12
    # Shift w by a in both end points: L_bc = mean over 2 x 2 points x B of ((u - g) / s)^2.
    shifted = exact.clone()
    shifted[:, 0, [0, -1]] += 0.1
    scale = problem.data_scale(inputs, grid)
    expected = ((0.1 / scale[:, 0, 0]) ** 2).sum() * 2 / (4 * inputs.shape[0])
    torch.testing.assert_close(boundary_loss(problem, shifted, inputs, grid), expected)


def test_boundary_loss_uses_the_global_constant_without_data_scale():
    problem = build_problem(problem_cfg("darcy_layered_1d", output_param="soft", output_scale_global=[2.0]))
    grid = problem.grid(8)
    inputs = problem.sample_inputs(3, torch.Generator().manual_seed(0))
    shifted = problem.analytical(inputs, grid) + 1.0
    torch.testing.assert_close(boundary_loss(problem, shifted, inputs, grid), torch.tensor(0.25))


def test_compute_loss_adds_bc_only_for_soft():
    grid_inputs = {}
    for output_param in ("constrained", "soft"):
        problem = build_problem(problem_cfg("heat_1d", output_param=output_param))
        grid = problem.grid(8)
        inputs = problem.sample_inputs(2, torch.Generator().manual_seed(0))
        u = torch.zeros(2, 1, 8, 8)
        weights = OmegaConf.create({"physics": 1.0, "data": 0.0, "bc": 10.0})
        total, components = compute_loss("physics", weights, problem, u, inputs, grid)
        grid_inputs[output_param] = components
    assert "bc" not in grid_inputs["constrained"]
    soft = grid_inputs["soft"]
    assert soft["bc"] > 0
    assert soft["total"] == pytest.approx(soft["physics"] + 10.0 * soft["bc"])


def test_bc_weight_from_problem_config_acts_in_every_mode_and_override_wins():
    problem = OmegaConf.create({"loss_weights": {"bc": 100.0, "data": 300.0}})
    for mode in ("physics", "data", "hybrid"):
        training = OmegaConf.load(CONFIGS / "training" / f"{mode}.yaml")
        weights = resolve_loss_weights(training, problem)
        assert weights["bc"] == 100.0
        # data stays fixed by the mode outside hybrid.
        assert weights["data"] == (300.0 if mode == "hybrid" else float(training.weights.data))
    training = OmegaConf.load(CONFIGS / "training" / "physics.yaml")
    training.weights.bc = 3.0
    assert resolve_loss_weights(training, problem, {"training.weights.bc"})["bc"] == 3.0


def test_selection_uses_the_full_objective_for_soft_physics():
    assert selection_criterion("physics", soft=True) == "val_objective"
    assert selection_criterion("data", soft=True) == "val_rel_l2_all"
    assert selection_criterion("physics", soft=False) == "val_physics"


def test_boundary_error_is_relative_to_the_exact_maximum():
    problem = build_problem(problem_cfg("darcy_layered_1d"))
    grid = problem.grid(8)
    inputs = problem.sample_inputs(2, torch.Generator().manual_seed(0))
    exact = problem.analytical(inputs, grid)  # max |h| = 1 at x = 0
    assert torch.all(boundary_errors(problem, exact, exact, inputs, grid) < 1e-6)
    wrong = exact.clone()
    wrong[0, 0, -1] += 0.03  # at x = 1, exact maximum is 1
    wrong[1, 0, 3] += 0.5  # interior, not a boundary error
    errors = boundary_errors(problem, wrong, exact, inputs, grid)
    torch.testing.assert_close(errors[:, 0], torch.tensor([0.03, 0.0]), atol=1e-6, rtol=1e-4)


def _cfg(name: str, mode: str, output_param: str):
    training = OmegaConf.load(CONFIGS / "training" / f"{mode}.yaml")
    training.update({"epochs": 2, "n_train": 4, "n_val": 2, "n_test": 2, "batch_size": 2, "eval_every": 1})
    problem = problem_cfg(name, output_param=output_param)
    problem.update({"train_resolution": 16, "test_resolution": 16, "resolution_study": [16]})
    model = {"name": "tiny", **OmegaConf.to_container(MODEL_CFG)}
    return OmegaConf.create({"seed": 0, "problem": problem, "training": training, "model": model})


@pytest.mark.parametrize("mode", ["physics", "data", "hybrid"])
@pytest.mark.parametrize("name", ["beam_1d", "darcy_layered_1d", "heat_1d"])
def test_soft_trains_in_every_mode(name, mode, tmp_path):
    cfg = _cfg(name, mode, "soft")
    problem = build_problem(cfg.problem)
    model = build_model(MODEL_CFG, problem, 16)
    result = train(model, problem, cfg, torch.device("cpu"), tmp_path)
    history = json.loads((tmp_path / "history.json").read_text())
    assert len(history["losses"]["bc"]) == 2 and history["output_param"] == "soft"
    assert len(history["val_bc"]) == 2 and len(history["val_objective"]) == 2
    assert history["selection"] == ("val_objective" if mode == "physics" else "val_rel_l2_all")
    if mode == "physics":
        assert history["best_score"] == min(history["val_objective"])
    model.load_state_dict(result.best_state)
    evaluation = evaluate_model(model, problem, cfg, torch.device("cpu"))
    metrics = write_metrics(tmp_path / "metrics.json", evaluation, problem, cfg, 1.0, 1, 1, "now", history)
    names = problem.variable_names["solution"]
    assert set(metrics["bc_error_mean"]) == set(names) and set(metrics["bc_error_max"]) == set(names)
    assert all(metrics["bc_error_max"][n] >= metrics["bc_error_mean"][n] > 0 for n in names)
    plot_training(history, "test", tmp_path / "training.png")


def test_constrained_has_no_bc_term_and_zero_boundary_error(tmp_path):
    cfg = _cfg("heat_1d", "physics", "constrained")
    problem = build_problem(cfg.problem)
    model = build_model(MODEL_CFG, problem, 16)
    result = train(model, problem, cfg, torch.device("cpu"), tmp_path)
    history = result.history()
    assert "bc" not in history["losses"] and "val_bc" not in history
    model.load_state_dict(result.best_state)
    evaluation = evaluate_model(model, problem, cfg, torch.device("cpu"))
    assert evaluation.bc_errors.abs().max() < 1e-6
