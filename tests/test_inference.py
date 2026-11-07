"""pino.inference and scripts/predict.py: load a run, predict one input, flag extrapolation, save."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.figure
import pytest
import torch
from omegaconf import OmegaConf

from pino.evaluation import IncompatibleRunError, evaluate_model
from pino.inference import as_inputs, extrapolation_messages, find_run, load_latest, load_run, predict_one, save_prediction
from pino.models.factory import build_model
from pino.problems import build_problem
from pino.visualization import plot_prediction

ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs"
MODEL_CFG = {"name": "tiny", "n_modes": 4, "hidden_channels": 4, "n_layers": 2, "domain_padding": 0.25}
#: One problem per plot type, with a small grid.
CASES = {"beam_1d": 16, "heat_1d": 16, "darcy_poisson_2d": 16, "heat_2d": [8, 8, 8]}


def make_run(run_dir: Path, name: str, resolution, **history_overrides) -> None:
    """A run directory as written by scripts/train.py, with a tiny random model."""
    problem_cfg = OmegaConf.load(CONFIGS / "problem" / f"{name}.yaml")
    problem_cfg.update({"train_resolution": resolution, "test_resolution": resolution, "resolution_study": [resolution]})
    training = OmegaConf.load(CONFIGS / "training" / "data.yaml")
    training.update({"n_test": 4, "batch_size": 4})
    cfg = OmegaConf.create({"seed": 0, "device": "cpu", "problem": problem_cfg, "model": MODEL_CFG, "training": training})
    (run_dir / ".hydra").mkdir(parents=True)
    OmegaConf.save(cfg, run_dir / ".hydra" / "config.yaml")
    OmegaConf.save(OmegaConf.create([f"problem={name}", "training=data"]), run_dir / ".hydra" / "overrides.yaml")
    problem = build_problem(cfg.problem)
    torch.manual_seed(0)
    torch.save(build_model(cfg.model, problem, resolution).state_dict(), run_dir / "best_model.pt")
    history = {"training_time_s": 1.0, "best_epoch": 1, **problem.variant_metadata(), **history_overrides}
    (run_dir / "history.json").write_text(json.dumps(history))


@pytest.mark.parametrize("name", CASES)
def test_prediction_of_a_test_example_matches_the_test_evaluation(name, tmp_path):
    make_run(tmp_path, name, CASES[name])
    run = load_run(tmp_path, device="cpu")
    evaluation = evaluate_model(run.model, run.problem, run.cfg, torch.device("cpu"))
    inputs = evaluation.dataset.inputs[2]
    prediction = predict_one(run, inputs.tolist())
    names = run.problem.variable_names["solution"]
    for c, variable in enumerate(names):
        assert prediction.rel_l2[variable] == pytest.approx(evaluation.errors[2, c].item(), rel=1e-5)
        assert prediction.bc_error[variable] == pytest.approx(evaluation.bc_errors[2, c].item(), rel=1e-5, abs=1e-7)
    torch.testing.assert_close(prediction.prediction, evaluation.predictions[2], rtol=1e-5, atol=1e-6)
    assert prediction.extrapolation == []


def test_other_resolution_and_matrix_inputs(tmp_path):
    make_run(tmp_path, "darcy_poisson_2d", 16)
    run = load_run(tmp_path, device="cpu")
    flat = [0.5] + [0.0] * 15  # a_11 only, given flat for the 4 x 4 coefficient matrix
    prediction = predict_one(run, flat, resolution=32)
    assert prediction.prediction.shape == (1, 32, 32) and prediction.resolution == 32
    assert as_inputs(run.problem, flat).shape == (4, 4)
    with pytest.raises(ValueError, match="expects 16 input values"):
        as_inputs(run.problem, [1.0, 2.0])


def test_inputs_outside_the_trained_range_are_flagged(tmp_path):
    make_run(tmp_path, "heat_1d", 16)
    run = load_run(tmp_path, device="cpu")
    with pytest.warns(UserWarning, match="Extrapolation"):
        prediction = predict_one(run, [2.0, 0.0, 0.0, 0.0, -0.5])
    assert len(prediction.extrapolation) == 1 and "input[0] = 2" in prediction.extrapolation[0]
    low, high = run.problem.input_ranges()
    assert torch.all(low > -1.0001) and torch.all(low < -0.999) and torch.all(high < 1.0001) and torch.all(high > 0.999)
    assert extrapolation_messages(run.problem, torch.zeros(5)) == []
    # The bounds of the family itself are not extrapolation, although the sampled extremes
    # lie slightly inside them.
    assert extrapolation_messages(run.problem, torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0])) == []


def test_load_run_refuses_a_different_wrapper(tmp_path):
    make_run(tmp_path, "heat_1d", 16, wrapper_version="1")
    with pytest.raises(IncompatibleRunError, match="wrapper_version: run '1', code '2'"):
        load_run(tmp_path, device="cpu")


@pytest.mark.parametrize("name", CASES)
def test_single_example_plot_returns_a_figure_or_writes_a_file(name, tmp_path):
    make_run(tmp_path, name, CASES[name])
    run = load_run(tmp_path, device="cpu")
    prediction = predict_one(run, run.problem.sample_inputs(1, torch.Generator().manual_seed(3))[0].tolist())
    figure = plot_prediction(run.problem, prediction, run.mode)
    assert isinstance(figure, matplotlib.figure.Figure)
    plot_prediction(run.problem, prediction, run.mode, tmp_path / "solution.png")
    assert (tmp_path / "solution.png").stat().st_size > 0


def test_saved_prediction_stays_out_of_the_summary(tmp_path):
    make_run(tmp_path / "run", "heat_1d", 16)
    run = load_run(tmp_path / "run", device="cpu")
    directory = save_prediction(run, predict_one(run, [1.0, 0.0, 0.5, 0.0, 0.0]), "half")
    assert directory == tmp_path / "run" / "predictions" / "half"
    assert sorted(p.name for p in directory.iterdir()) == ["inputs.json", "prediction.pt", "result.json", "solution.png"]
    assert json.loads((directory / "inputs.json").read_text())["inputs"] == [1.0, 0.0, 0.5, 0.0, 0.0]
    saved = torch.load(directory / "prediction.pt")
    assert saved["prediction"].shape == saved["exact"].shape == (1, 16, 16)
    spec = importlib.util.spec_from_file_location("summarize", ROOT / "scripts" / "summarize.py")
    summarize = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summarize)
    assert summarize.collect(tmp_path) == []  # no metrics.json anywhere, predictions included


def _predict_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "predict.py"), *args, "--device", "cpu"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])},
    )


def test_predict_script(tmp_path):
    make_run(tmp_path, "heat_1d", 16)
    result = _predict_script(str(tmp_path), "--inputs", "1,0,0.5,0,0", "--resolution", "32", "--name", "cli")
    assert result.returncode == 0, result.stderr
    assert "rel_l2" in result.stdout and (tmp_path / "predictions" / "cli" / "solution.png").exists()
    ranges = _predict_script(str(tmp_path), "--show-ranges")
    assert ranges.returncode == 0 and "[4]" in ranges.stdout
    wrong = _predict_script(str(tmp_path), "--inputs", "1,2", "--no-save")
    assert wrong.returncode != 0 and "expects 5 input values" in wrong.stderr


def _metrics(run_dir: Path, **fields) -> None:
    metrics = {"benchmark": "heat_1d", "mode": "data", "seed": 0, "timestamp": "2026-01-01T00:00:00",
               "wrapper_version": "2", "inputs": "sine", "output_param": "constrained", **fields}
    (run_dir / "metrics.json").write_text(json.dumps(metrics))


def test_find_run_takes_the_newest_finished_run_of_a_combination(tmp_path):
    base = tmp_path / "outputs" / "heat_1d"
    for name, fields in {
        "sine-constrained/data/a": {"timestamp": "2026-01-01T00:00:00"},
        "sine-constrained/data/b": {"timestamp": "2026-03-01T00:00:00", "seed": 1},
        "sine-constrained/data/c": {"timestamp": "2026-02-01T00:00:00"},
        "sine-soft/data/d": {"timestamp": "2026-04-01T00:00:00", "output_param": "soft"},
    }.items():
        make_run(base / name, "heat_1d", 16)
        _metrics(base / name, **fields)
    # Newer, but superseded or without checkpoint: ignored.
    make_run(tmp_path / "outputs" / "_superseded" / "x" / "heat_1d" / "sine-constrained" / "data" / "e", "heat_1d", 16)
    unfinished = base / "sine-constrained" / "data" / "f"
    make_run(unfinished, "heat_1d", 16)
    _metrics(unfinished, timestamp="2027-01-01T00:00:00")
    (unfinished / "best_model.pt").unlink()

    outputs = tmp_path / "outputs"
    assert find_run("heat_1d", "data", outputs=outputs) == base / "sine-constrained" / "data" / "b"
    assert find_run("heat_1d", "data", seed=0, outputs=outputs) == base / "sine-constrained" / "data" / "c"
    assert find_run("heat_1d", "data", "soft", outputs=outputs) == base / "sine-soft" / "data" / "d"
    assert load_latest("heat_1d", "data", outputs=outputs, device="cpu").run_dir == base / "sine-constrained" / "data" / "b"
    with pytest.raises(FileNotFoundError, match="sine/soft data seed 0"):
        find_run("heat_1d", "physics", outputs=outputs)


def test_wrapper_fields_from_metrics_when_the_history_lacks_them(tmp_path):
    """Older runs: history.json without wrapper fields, metrics.json completed after a re-check."""
    make_run(tmp_path, "heat_1d", 16)
    history = json.loads((tmp_path / "history.json").read_text())
    for key in ("wrapper_version", "inputs", "output_param"):
        del history[key]
    (tmp_path / "history.json").write_text(json.dumps(history))
    with pytest.raises(IncompatibleRunError, match="--allow-legacy"):
        load_run(tmp_path, device="cpu")
    _metrics(tmp_path)
    load_run(tmp_path, device="cpu")
    _metrics(tmp_path, wrapper_version="1")
    with pytest.raises(IncompatibleRunError, match="wrapper_version: run '1'"):
        load_run(tmp_path, device="cpu")


def test_predict_script_with_a_benchmark_name(tmp_path):
    run_dir = tmp_path / "outputs" / "heat_1d" / "sine-constrained" / "data" / "a"
    make_run(run_dir, "heat_1d", 16)
    _metrics(run_dir)
    outputs = str(tmp_path / "outputs")
    result = _predict_script("heat_1d", "--mode", "data", "--outputs", outputs, "--inputs", "1,0,0,0,0", "--no-save")
    assert result.returncode == 0, result.stderr
    assert f"run: {run_dir}" in result.stdout
    missing = _predict_script("heat_1d", "--outputs", outputs, "--inputs", "1,0,0,0,0")
    assert missing.returncode != 0 and "No finished run for heat_1d sine/constrained physics" in missing.stderr
    unknown = _predict_script("heat_9d", "--outputs", outputs)
    assert unknown.returncode != 0 and "Unknown problem" in unknown.stderr
