"""Measure GPU memory and time per training step of the FNO for a problem, grids and batch sizes.

Usage: ``python scripts/measure_step.py --problem chloride_corner_2d --mode hybrid \
--grids 32x32x32 64x64x32 64x64x64 --batches 4 8 16``

Runs real training steps (prediction with the output parametrization, loss of the mode,
backward, Adam) on random inputs of the problem and reports the peak allocated memory and the
mean time per step after a warm-up.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from pino.losses import uses_data
from pino.models.factory import build_model, count_parameters
from pino.problems import build_problem
from pino.trainer import train_step

CONFIGS = Path(__file__).parents[1] / "configs"


def measure(problem, model_cfg, mode: str, resolution: tuple[int, ...], batch: int, steps: int) -> tuple[float, float]:
    device = torch.device("cuda")
    model = build_model(model_cfg, problem, resolution).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    weights = OmegaConf.load(CONFIGS / "training" / f"{mode}.yaml").weights
    grid = problem.grid(resolution).to(device)
    inputs = problem.sample_inputs(batch, torch.Generator().manual_seed(0)).to(device)
    target = problem.analytical(inputs, grid) if uses_data(mode) else None
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):  # warm-up
        train_step(model, optimizer, problem, inputs, grid, mode, weights, target)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        train_step(model, optimizer, problem, inputs, grid, mode, weights, target)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30, (time.perf_counter() - start) / steps, count_parameters(model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="chloride_corner_2d")
    parser.add_argument("--model", default="fno_small")
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--n-modes", type=int, nargs="*", help="override n_modes, one value per axis")
    parser.add_argument("--grids", nargs="+", default=["32x32x32", "64x64x32", "64x64x64"])
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    problem = build_problem(OmegaConf.load(CONFIGS / "problem" / f"{args.problem}.yaml"))
    model_cfg = OmegaConf.load(CONFIGS / "model" / f"{args.model}.yaml")
    if args.n_modes:
        model_cfg.n_modes = args.n_modes
    print(f"{args.problem}, {args.model}, n_modes {model_cfg.n_modes}, mode {args.mode}, "
          f"{problem.in_channels} input channels, {torch.cuda.get_device_name()}")
    print("| Grid | Batch | Memory [GiB] | Time per step [ms] | Parameters |")
    print("|---|---|---|---|---|")
    for key in args.grids:
        resolution = tuple(int(n) for n in key.split("x"))
        for batch in args.batches:
            try:
                memory, seconds, parameters = measure(problem, model_cfg, args.mode, resolution, batch, args.steps)
                print(f"| {key} | {batch} | {memory:.2f} | {1000 * seconds:.0f} | {parameters:,} |", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"| {key} | {batch} | out of memory | - | - |", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
