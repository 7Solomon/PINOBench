"""Loss weights, checkpoint selection on the validation set and the short training loop."""

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from pino import evaluation
from pino.evaluation import predict, validate, validation_seed
from pino.data.dataset import build_dataset
from pino.losses import overridden_keys, resolve_loss_weights
from pino.models.factory import build_model
from pino.problems import build_problem
from pino.trainer import HISTORY, selection_criterion, train

CONFIGS = Path(__file__).parents[1] / "configs"
MODEL_CFG = OmegaConf.create({"n_modes": 4, "hidden_channels": 4, "n_layers": 2, "domain_padding": 0.25})


def training_cfg(mode: str):
    return OmegaConf.load(CONFIGS / "training" / f"{mode}.yaml")


def problem_cfg(name: str):
    return OmegaConf.load(CONFIGS / "problem" / f"{name}.yaml")


def test_overridden_keys_parses_hydra_overrides():
    keys = overridden_keys(["problem=beam_1d", "training.weights.data=300", "++seed=1", "+foo.bar=2"])
    assert keys == {"problem", "training.weights.data", "seed", "foo.bar"}


def test_problem_loss_weights_replace_training_weights_in_hybrid_only():
    darcy = problem_cfg("darcy_layered_1d")
    assert resolve_loss_weights(training_cfg("hybrid"), darcy) == {"physics": 1.0, "data": 300.0, "bc": 100.0}
    # Single-term modes keep their fixed weights.
    assert resolve_loss_weights(training_cfg("physics"), darcy) == {"physics": 1.0, "data": 0.0, "bc": 100.0}
    assert resolve_loss_weights(training_cfg("data"), darcy) == {"physics": 0.0, "data": 1.0, "bc": 100.0}


def test_problems_without_loss_weights_keep_the_global_weights():
    for name in ("richards_steady_1d", "chloride_1d", "darcy_poisson_2d"):
        assert "loss_weights" not in problem_cfg(name)
        assert resolve_loss_weights(training_cfg("hybrid"), problem_cfg(name)) == {"physics": 1.0, "data": 1.0, "bc": 10.0}


def test_soft_pilot_bc_weights_act_in_every_mode():
    """beam_1d and heat_1d set only bc; physics and data keep the training weights."""
    for name, bc in (("beam_1d", 100.0), ("heat_1d", 1.0)):
        for mode in ("physics", "data", "hybrid"):
            training = training_cfg(mode)
            expected = {**{k: float(v) for k, v in training.weights.items()}, "bc": bc}
            assert resolve_loss_weights(training, problem_cfg(name)) == expected


def test_command_line_override_wins_over_problem_loss_weights():
    training = training_cfg("hybrid")
    training.weights.data = 50.0  # as Hydra would apply training.weights.data=50
    weights = resolve_loss_weights(training, problem_cfg("darcy_layered_1d"), {"training.weights.data"})
    assert weights == {"physics": 1.0, "data": 50.0, "bc": 100.0}
    # Overriding the whole block keeps every training weight, also bc.
    weights = resolve_loss_weights(training, problem_cfg("darcy_layered_1d"), {"training.weights"})
    assert weights == {"physics": 1.0, "data": 50.0, "bc": 10.0}


def test_unknown_loss_term_is_rejected():
    problem = OmegaConf.create({"loss_weights": {"boundary": 1.0}})
    with pytest.raises(KeyError):
        resolve_loss_weights(training_cfg("hybrid"), problem)


def test_selection_criterion_depends_on_mode():
    assert selection_criterion("physics") == "val_physics"
    assert selection_criterion("data") == "val_rel_l2_all"
    assert selection_criterion("hybrid") == "val_rel_l2_all"


def test_validation_test_and_training_inputs_differ():
    problem = build_problem(problem_cfg("beam_1d"))
    seeds = {42, evaluation.test_seed(42), validation_seed(42)}
    assert len(seeds) == 3
    inputs = [build_dataset(problem, 8, 16, seed, with_targets=False).inputs for seed in seeds]
    assert not torch.equal(inputs[0], inputs[1]) and not torch.equal(inputs[1], inputs[2])


def test_validate_returns_physics_loss_and_errors_per_variable():
    problem = build_problem(problem_cfg("beam_1d"))
    model = build_model(MODEL_CFG, problem, 16)
    dataset = build_dataset(problem, 5, 16, 0)
    result = validate(model, problem, dataset, torch.device("cpu"), batch_size=2)
    assert result.errors.shape == (5, 2)
    grid = dataset.grid
    with torch.no_grad():
        expected = problem.residual(predict(model, problem, dataset.inputs, grid), dataset.inputs, grid).pow(2).mean()
    assert result.physics == pytest.approx(expected.item(), rel=1e-5)


@pytest.mark.parametrize("mode", ["physics", "hybrid"])
def test_short_training_selects_by_the_validation_criterion(tmp_path, mode):
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "problem": problem_cfg("beam_1d"),
            "training": {**OmegaConf.to_container(training_cfg(mode)), "epochs": 4, "eval_every": 1, "n_train": 8, "n_val": 4},
        }
    )
    cfg.problem.train_resolution = 16
    problem = build_problem(cfg.problem)
    torch.manual_seed(0)
    model = build_model(MODEL_CFG, problem, 16)
    result = train(model, problem, cfg, torch.device("cpu"), tmp_path)

    scores = getattr(result, selection_criterion(mode))
    assert result.selection == selection_criterion(mode)
    assert result.best_score == min(scores)
    assert result.best_epoch == result.eval_epochs[scores.index(min(scores))]
    assert set(result.val_rel_l2) == {"w", "M"} and len(result.val_physics) == 4
    history = json.loads((tmp_path / HISTORY).read_text())
    assert history["best_epoch"] == result.best_epoch and (tmp_path / "best_model.pt").exists()


def test_problem_training_block_sets_budget_unless_overridden():
    from pino.losses import resolve_training_settings

    heat_2d = OmegaConf.create({"training": {"epochs": 200, "batch_size": 8}})  # as in heat_2d.yaml
    settings = resolve_training_settings(training_cfg("physics"), heat_2d)
    assert settings["epochs"] == 200 and settings["batch_size"] == 8 and settings["n_train"] == 1000
    assert resolve_training_settings(training_cfg("physics"), heat_2d, {"training.epochs"})["epochs"] == 500
    # Selecting the training group (training=hybrid) is not an override of its keys.
    assert resolve_training_settings(training_cfg("hybrid"), heat_2d, {"training", "problem"})["batch_size"] == 8
    # Problems without the block keep the training config.
    assert resolve_training_settings(training_cfg("data"), problem_cfg("heat_1d"))["epochs"] == 500
    with pytest.raises(KeyError):
        resolve_training_settings(training_cfg("data"), OmegaConf.create({"training": {"mode": "data"}}))
