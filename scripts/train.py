"""Train one model: ``python scripts/train.py problem=heat_1d model=fno_small training=physics``."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from pino.evaluation import evaluate_model, write_metrics
from pino.models.factory import build_model, count_parameters
from pino.problems import build_problem
from pino.trainer import select_device, set_seed, train
from pino.visualization import plot_run

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    set_seed(int(cfg.seed))
    device = select_device(cfg.device)

    problem = build_problem(cfg.problem)
    model = build_model(cfg.model, problem, cfg.problem.train_resolution).to(device)
    n_parameters = count_parameters(model)
    log.info("%s, mode %s, %d parameters on %s", problem.name, cfg.training.mode, n_parameters, device)

    outcome = train(model, problem, cfg, device, run_dir)
    history = outcome.history()

    model.load_state_dict(outcome.best_state)
    result = evaluate_model(model, problem, cfg, device)
    metrics = write_metrics(
        run_dir / "metrics.json",
        result,
        problem,
        cfg,
        outcome.training_time_s,
        outcome.best_epoch,
        n_parameters,
        datetime.now().isoformat(timespec="seconds"),
    )
    plot_run(problem, result, history, cfg.training.mode, int(cfg.problem.train_resolution), run_dir)
    log.info("rel. L2 %s, time %.0f s, results in %s", metrics["rel_l2"], outcome.training_time_s, run_dir)


if __name__ == "__main__":
    main()
