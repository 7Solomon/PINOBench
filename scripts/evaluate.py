"""Evaluate a finished run again and rewrite its metrics and plots.

Usage: ``python scripts/evaluate.py outputs/heat_1d/physics/<timestamp> [--device cpu] [--allow-legacy]``

The run's ``wrapper_version``, ``inputs`` and ``output_param`` must match the current code.
Runs written before these fields existed are refused unless ``--allow-legacy`` is given.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict

from pino.evaluation import IncompatibleRunError, check_run_compatibility, evaluate_model, write_metrics
from pino.losses import overridden_keys, resolve_loss_weights, resolve_training_settings
from pino.numerics.grids import resolution_key
from pino.models.factory import build_model, count_parameters
from pino.problems import build_problem
from pino.trainer import select_device, set_seed
from pino.visualization import plot_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Hydra run directory of a finished training")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="evaluate runs without wrapper_version, assuming they used the current wrapper",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / ".hydra" / "config.yaml")
    history = json.loads((args.run_dir / "history.json").read_text())
    set_seed(int(cfg.seed))
    device = select_device(args.device)

    problem = build_problem(cfg.problem)
    try:
        check_run_compatibility(history, problem, args.allow_legacy)
    except IncompatibleRunError as error:
        raise SystemExit(f"Refusing to evaluate {args.run_dir}: {error}") from None
    overrides = overridden_keys(OmegaConf.load(args.run_dir / ".hydra" / "overrides.yaml"))
    with open_dict(cfg):
        cfg.training.weights = OmegaConf.create(resolve_loss_weights(cfg.training, cfg.problem, overrides))
        cfg.training.update(resolve_training_settings(cfg.training, cfg.problem, overrides))
    model = build_model(cfg.model, problem, cfg.problem.train_resolution).to(device)
    state = torch.load(args.run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state)

    result = evaluate_model(model, problem, cfg, device)
    metrics = write_metrics(
        args.run_dir / "metrics.json",
        result,
        problem,
        cfg,
        history["training_time_s"],
        history["best_epoch"],
        count_parameters(model),
        datetime.now().isoformat(timespec="seconds"),
        history,
    )
    plot_run(problem, result, history, cfg.training.mode, resolution_key(cfg.problem.train_resolution), args.run_dir)
    print(json.dumps(metrics["rel_l2"], indent=2))


if __name__ == "__main__":
    main()
