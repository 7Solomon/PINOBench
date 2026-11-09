"""pino.weights_text and scripts/export_weights.py: the text format of a checkpoint's weights."""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from pino.models.factory import build_model
from pino.problems import build_problem
from pino.weights_text import default_path, load_weights, select, to_text

ROOT = Path(__file__).parents[1]
MODEL_CFG = {"n_modes": 4, "hidden_channels": 4, "n_layers": 2, "domain_padding": 0.25}


def tiny_model():
    problem_cfg = OmegaConf.load(ROOT / "configs" / "problem" / "heat_1d.yaml")
    torch.manual_seed(0)
    return build_model(OmegaConf.create(MODEL_CFG), build_problem(problem_cfg), 16)


@pytest.fixture
def run_dir(tmp_path):
    """A checkpoint as written by the trainer, with neuraloperator's ``_metadata`` entry."""
    directory = tmp_path / "outputs" / "heat_1d" / "sine-constrained" / "hybrid" / "stamp"
    directory.mkdir(parents=True)
    torch.save(tiny_model().state_dict(), directory / "best_model.pt")
    return directory


def parse(text):
    """Read the text back: name -> tensor, checking the announced total."""
    blocks = re.findall(r"# tensor (\S+) shape=\[([^\]]*)\] dtype=(\S+)\n([^\n#]*)", text)
    state = {}
    for name, shape, dtype, body in blocks:
        values = torch.tensor([float(value) for value in body.split()], dtype=torch.float32)
        shape = [int(n) for n in shape.split(",")] if shape else []
        state[name] = torch.view_as_complex(values.reshape(*shape, 2)) if dtype == "complex64" else values.reshape(shape)
    total = int(text.splitlines()[0].removeprefix("# total_numbers "))
    assert total == sum(len(body.split()) for *_, body in blocks)
    return state


def run_script(*args, cwd):
    script = ROOT / "scripts" / "export_weights.py"
    return subprocess.run([sys.executable, str(script), *args], cwd=cwd, capture_output=True, text=True)


def test_format_of_real_and_complex_tensors():
    state = {"a.weight": torch.tensor([[1.0, 0.123456], [-2.5, 3.0]]), "b": torch.tensor([1 + 2j], dtype=torch.complex64)}
    text, total = to_text(state)
    assert total == 6
    assert text == (
        "# total_numbers 6\n"
        "# tensor a.weight shape=[2, 2] dtype=float32\n"
        "1 0.1235 -2.5 3\n"
        "# tensor b shape=[1] dtype=complex64\n"
        "1 2\n"
    )


def test_metadata_is_skipped(run_dir):
    assert "_metadata" in torch.load(run_dir / "best_model.pt", weights_only=False)
    state = load_weights(run_dir)
    assert "_metadata" not in state and all(torch.is_tensor(value) for value in state.values())


def test_nine_digits_restore_every_value(run_dir):
    state = load_weights(run_dir)
    assert any(value.is_complex() for value in state.values())
    restored = parse(to_text(state, digits=9)[0])
    assert list(restored) == list(state)
    for name, value in state.items():
        assert torch.equal(restored[name], value), name


def test_only_selects_by_glob(run_dir):
    state = load_weights(run_dir)
    chosen = select(state, "lifting.*, projection.fcs.0.bias")
    assert chosen and all(name.startswith("lifting.") or name == "projection.fcs.0.bias" for name in chosen)
    assert select(state, None) == state


def test_default_path_drops_mode_and_timestamp(run_dir):
    assert default_path(run_dir, "weights") == Path("weights/heat_1d/sine-constrained.txt")


def test_script_writes_the_file_and_refuses_to_replace_it(run_dir, tmp_path):
    first = run_script(str(run_dir), cwd=tmp_path)
    assert first.returncode == 0, first.stderr
    path = tmp_path / "workspace" / "weights" / "heat_1d" / "sine-constrained.txt"
    assert path.read_text().startswith("# total_numbers ")

    second = run_script(str(run_dir), cwd=tmp_path)
    assert second.returncode != 0 and "already exists" in second.stderr
    assert run_script(str(run_dir), "--force", cwd=tmp_path).returncode == 0

    empty = run_script(str(run_dir), "--only", "nothing.*", "--out", str(tmp_path / "x.txt"), cwd=tmp_path)
    assert empty.returncode != 0 and not (tmp_path / "x.txt").exists()
