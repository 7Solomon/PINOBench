"""Write the weights of a checkpoint as plain text.

Format::

    # total_numbers <count of all numbers below>
    # tensor <name> shape=[a, b, ...] dtype=<float32|complex64|...>
    v1 v2 v3 ...

The values are in row-major order, a complex number is written as ``re im``. Entries of the
checkpoint that are no float or complex tensors are skipped, such as neuraloperator's
``_metadata``.

Fewer digits make the file shorter but round the weights: only ``digits=9`` restores every
float32 value exactly.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import torch
from torch import Tensor

from pino.paths import WEIGHTS
from pino.trainer import CHECKPOINT

DEFAULT_DIGITS = 4


def load_weights(run_dir: str | Path) -> dict[str, Tensor]:
    """Return the float and complex tensors of a run's checkpoint, without building the model."""
    state = torch.load(Path(run_dir) / CHECKPOINT, map_location="cpu", weights_only=False)
    return {name: value for name, value in state.items() if torch.is_tensor(value) and _is_weight(value)}


def _is_weight(tensor: Tensor) -> bool:
    return tensor.is_floating_point() or tensor.is_complex()


def select(state: dict[str, Tensor], only: str | None = None) -> dict[str, Tensor]:
    """Keep the tensors matching ``only``, comma-separated globs; ``None`` keeps all of them."""
    if not only:
        return state
    patterns = [pattern.strip() for pattern in only.split(",")]
    return {name: value for name, value in state.items() if any(fnmatch.fnmatch(name, p) for p in patterns)}


def to_text(state: dict[str, Tensor], digits: int = DEFAULT_DIGITS) -> tuple[str, int]:
    """Return the text and the count of numbers in it."""
    lines, total = [], 0
    for name, tensor in state.items():
        values = (torch.view_as_real(tensor) if tensor.is_complex() else tensor).detach().flatten().tolist()
        total += len(values)
        lines.append(f"# tensor {name} shape={list(tensor.shape)} dtype={str(tensor.dtype).replace('torch.', '')}")
        lines.append(" ".join(f"{value:.{digits}g}" for value in values))
    return f"# total_numbers {total}\n" + "\n".join(lines) + "\n", total


def default_path(run_dir: str | Path, weights: str | Path = WEIGHTS) -> Path:
    """``<weights>/<benchmark>/<inputs>-<output_param>.txt`` of a run below the outputs directory.

    Mode, seed and timestamp are not part of the name, so runs of one variant share this path.
    """
    benchmark, variant = Path(run_dir).resolve().parts[-4:-2]
    return Path(weights) / benchmark / f"{variant}.txt"
