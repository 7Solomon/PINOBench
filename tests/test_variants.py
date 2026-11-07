"""Output parametrizations, input variants and the wrapper-version check."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from pino.evaluation import IncompatibleRunError, check_run_compatibility, predict
from pino.models.factory import build_model
from pino.numerics.constraints import dirichlet_1d
from pino.problems import available_problems, build_problem, problem_class
from pino.trainer import train

ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs"
MODEL_CFG = OmegaConf.create({"n_modes": 4, "hidden_channels": 4, "n_layers": 2, "domain_padding": 0.25})


def problem_cfg(name: str, **overrides):
    cfg = OmegaConf.load(CONFIGS / "problem" / f"{name}.yaml")
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


EXPECTED_INPUTS = {
    "beam_1d": ("linear", "sine"),
    "chloride_1d": ("uniform",),
    "chloride_corner_2d": ("uniform",),
    "darcy_layered_1d": ("two_layer",),
    "darcy_poisson_2d": ("sine",),
    "heat_1d": ("sine",),
    "heat_2d": ("sine",),
    "richards_steady_1d": ("uniform",),
}


def test_every_problem_names_its_input_families():
    assert set(available_problems()) == set(EXPECTED_INPUTS)
    for name, families in EXPECTED_INPUTS.items():
        assert problem_class(name).SUPPORTED_INPUTS == families


@pytest.mark.parametrize("name", available_problems())
def test_defaults_are_the_original_family_and_constrained(name):
    default = problem_class(name).default_inputs()
    assert default == EXPECTED_INPUTS[name][0]
    # The YAML names the default family explicitly; it also appears in the run directory.
    assert problem_cfg(name).inputs == default
    problem = build_problem(problem_cfg(name))
    assert (problem.inputs, problem.output_param) == (default, "constrained")
    assert isinstance(problem.WRAPPER_VERSION, str)
    assert problem.variant_metadata() == {
        "wrapper_version": problem.WRAPPER_VERSION,
        "inputs": default,
        "output_param": "constrained",
    }


def test_missing_inputs_key_falls_back_to_the_original_family():
    cfg = problem_cfg("beam_1d")
    del cfg["inputs"]
    assert build_problem(cfg).inputs == "linear"


def test_problem_without_supported_inputs_is_rejected():
    from pino.problems.registry import register_problem

    with pytest.raises(TypeError, match="SUPPORTED_INPUTS"):
        register_problem("no_inputs")(type("NoInputs", (), {"WRAPPER_VERSION": "1"}))


@pytest.mark.parametrize("name", available_problems())
def test_constrained_equals_boundary_times_output_scale(name):
    problem = build_problem(problem_cfg(name))
    grid = problem.grid(16)
    inputs = problem.sample_inputs(3, torch.Generator().manual_seed(0))
    g, phi = problem.boundary(inputs, grid)
    scale = problem.output_scale(inputs, grid)
    g_total, phi_total = problem.output_parametrization(inputs, grid)
    assert torch.equal(g_total, g)
    assert torch.equal(phi_total, phi if scale is None else phi * scale)


def test_beam_constrained_is_bit_identical_to_the_original_wrapper():
    """Before the split, Beam1D.boundary returned phi * data_scale with the same w_c, M_c."""
    problem = build_problem(problem_cfg("beam_1d"))
    grid = problem.grid(64)
    inputs = problem.sample_inputs(16, torch.Generator().manual_seed(1))
    q_mean = (inputs[:, 0] + 0.5 * inputs[:, 1]).clamp_min(1e-2)
    original_scale = torch.stack([5 * q_mean / (384 * inputs[:, 2]), q_mean / 8], dim=1)[..., None]
    zeros = torch.zeros(16, 2)
    g_old, phi_old = dirichlet_1d(grid[0], zeros, zeros)
    g, phi = problem.output_parametrization(inputs, grid)
    assert torch.equal(g, g_old) and torch.equal(phi, phi_old * original_scale)
    # data_scale (loss only) is unchanged as well.
    assert torch.equal(problem.data_scale(inputs, grid), original_scale)


def test_unscaled_uses_the_global_constant_per_channel():
    problem = build_problem(problem_cfg("beam_1d", output_param="unscaled"))
    grid = problem.grid(32)
    inputs = problem.sample_inputs(4, torch.Generator().manual_seed(0))
    g, phi = problem.output_parametrization(inputs, grid)
    x = grid[0]
    c = torch.tensor([0.0075, 0.075])[None, :, None]
    assert torch.equal(phi, (x * (1 - x))[None, None] * c)
    # Independent of the inputs: the same for every sample.
    assert phi.shape == (1, 2, 32)
    model = build_model(MODEL_CFG, problem, 32)
    u = predict(model, problem, inputs, grid)
    assert torch.all(u[..., 0] == 0) and torch.all(u[..., -1] == 0)


def test_unscaled_without_global_constant_uses_one():
    problem = build_problem(problem_cfg("darcy_layered_1d", output_param="unscaled"))
    grid = problem.grid(16)
    inputs = problem.sample_inputs(2, torch.Generator().manual_seed(0))
    assert torch.equal(problem.output_parametrization(inputs, grid)[1], problem.boundary(inputs, grid)[1])


def test_direct_has_no_boundary_values():
    problem = build_problem(problem_cfg("beam_1d", output_param="direct"))
    grid = problem.grid(16)
    inputs = problem.sample_inputs(2, torch.Generator().manual_seed(0))
    g, phi = problem.output_parametrization(inputs, grid)
    assert torch.all(g == 0) and phi.shape == (1, 2, 1)
    torch.manual_seed(0)
    model = build_model(MODEL_CFG, problem, 16)
    with torch.no_grad():
        raw = model(problem.to_fno_input(inputs, grid))
        assert torch.equal(predict(model, problem, inputs, grid), phi * raw)
    # The supports are no longer enforced.
    assert predict(model, problem, inputs, grid)[..., 0].abs().max() > 0


@pytest.mark.parametrize("mode", ["physics", "hybrid"])
def test_direct_is_rejected_outside_data_mode(mode, tmp_path):
    problem = build_problem(problem_cfg("beam_1d", output_param="direct"))
    problem.check_mode("data")
    with pytest.raises(ValueError, match="only allowed in mode 'data'"):
        problem.check_mode(mode)
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "problem": problem_cfg("beam_1d", output_param="direct"),
            "training": {**OmegaConf.to_container(OmegaConf.load(CONFIGS / "training" / f"{mode}.yaml")), "epochs": 1, "n_train": 4, "n_val": 2},
        }
    )
    with pytest.raises(ValueError, match="only allowed in mode 'data'"):
        train(build_model(MODEL_CFG, problem, 64), problem, cfg, torch.device("cpu"), tmp_path)


def test_invalid_and_unsupported_variants_are_rejected():
    with pytest.raises(ValueError, match="Unknown output_param"):
        build_problem(problem_cfg("beam_1d", output_param="free"))
    with pytest.raises(ValueError, match="beam_1d does not implement inputs='fancy'"):
        build_problem(problem_cfg("beam_1d", inputs="fancy"))
    with pytest.raises(ValueError, match="heat_1d does not implement inputs='linear'"):
        build_problem(problem_cfg("heat_1d", inputs="linear"))
    # The old generic names are gone.
    with pytest.raises(ValueError, match="does not implement inputs='simple'"):
        build_problem(problem_cfg("beam_1d", inputs="simple"))
    with pytest.raises(ValueError, match="needs 2 entries"):
        build_problem(problem_cfg("beam_1d", output_scale_global=[1.0]))


def test_run_compatibility_check():
    problem = build_problem(problem_cfg("heat_1d"))
    check_run_compatibility(problem.variant_metadata(), problem)
    with pytest.raises(IncompatibleRunError, match="wrapper_version: run '1', code '2'"):
        check_run_compatibility({**problem.variant_metadata(), "wrapper_version": "1"}, problem)
    with pytest.raises(IncompatibleRunError, match="output_param: run 'unscaled'"):
        check_run_compatibility({**problem.variant_metadata(), "output_param": "unscaled"}, problem)
    legacy = {"best_epoch": 10}
    with pytest.raises(IncompatibleRunError, match="--allow-legacy"):
        check_run_compatibility(legacy, problem)
    check_run_compatibility(legacy, problem, allow_legacy=True)


def _fake_run(run_dir: Path, history: dict) -> None:
    """A minimal run directory as written by scripts/train.py, with a tiny model."""
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "problem": {**OmegaConf.to_container(problem_cfg("beam_1d")), "resolution_study": [16], "train_resolution": 16, "test_resolution": 16},
            "model": {"name": "tiny", **OmegaConf.to_container(MODEL_CFG)},
            "training": {**OmegaConf.to_container(OmegaConf.load(CONFIGS / "training" / "data.yaml")), "n_test": 4},
        }
    )
    (run_dir / ".hydra").mkdir(parents=True)
    OmegaConf.save(cfg, run_dir / ".hydra" / "config.yaml")
    OmegaConf.save(OmegaConf.create([]), run_dir / ".hydra" / "overrides.yaml")
    problem = build_problem(cfg.problem)
    torch.save(build_model(cfg.model, problem, 16).state_dict(), run_dir / "best_model.pt")
    (run_dir / "history.json").write_text(json.dumps(history))


def _evaluate(run_dir: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate.py"), str(run_dir), "--device", "cpu", *flags],
        capture_output=True,
        text=True,
        # Import this checkout's package, also when the installed one lives elsewhere.
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])},
    )


def test_evaluate_script_refuses_mismatched_and_legacy_runs(tmp_path):
    problem = build_problem(problem_cfg("beam_1d"))
    base = {"training_time_s": 1.0, "best_epoch": 1, "selection": "val_rel_l2_all", "loss_epochs": [1],
            "losses": {"data": [1.0]}, "eval_epochs": [1], "val_physics": [1.0], "val_rel_l2_all": [1.0],
            "val_rel_l2": {"w": [1.0], "M": [1.0]}, "best_score": 1.0}

    mismatch = tmp_path / "mismatch"
    _fake_run(mismatch, {**base, **problem.variant_metadata(), "wrapper_version": "0"})
    result = _evaluate(mismatch)
    assert result.returncode != 0 and "different beam_1d wrapper" in result.stderr
    assert not (mismatch / "metrics.json").exists()

    legacy = tmp_path / "legacy"
    _fake_run(legacy, base)
    result = _evaluate(legacy)
    assert result.returncode != 0 and "--allow-legacy" in result.stderr
    assert not (legacy / "metrics.json").exists()
    result = _evaluate(legacy, "--allow-legacy")
    assert result.returncode == 0, result.stderr
    assert json.loads((legacy / "metrics.json").read_text())["wrapper_version"] == problem.WRAPPER_VERSION

    matching = tmp_path / "matching"
    _fake_run(matching, {**base, **problem.variant_metadata()})
    assert _evaluate(matching).returncode == 0
