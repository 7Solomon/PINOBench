"""Evaluate a finished run again and rewrite its metrics and plots.

Usage: ``python scripts/evaluate.py workspace/outputs/heat_1d/physics/<timestamp> [--device cpu] [--allow-legacy]``

The run's ``wrapper_version``, ``inputs`` and ``output_param`` must match the current code.
Runs written before these fields existed are refused unless ``--allow-legacy`` is given.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from pino.evaluation import evaluate_model, write_metrics
from pino.inference import IncompatibleRunError, load_run
from pino.models.factory import count_parameters
from pino.numerics.grids import resolution_key
from pino.trainer import set_seed
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

    try:
        run = load_run(args.run_dir, args.device, args.allow_legacy)
    except IncompatibleRunError as error:
        raise SystemExit(f"Refusing to evaluate {args.run_dir}: {error}") from None
    cfg, problem, model, device, history = run.cfg, run.problem, run.model, run.device, run.history
    set_seed(int(cfg.seed))

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
