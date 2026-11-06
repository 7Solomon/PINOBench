"""Collect ``metrics.json`` files into ``summary.csv``, ``summary.md`` and ``summary.png``.

Rows are grouped by benchmark x ``inputs`` x ``output_param`` x mode and aggregated over the
seeds: for every seed the most recent run is used, and the mean rel. L2 of each run (main
variable) is reported as mean, minimum and maximum over the seeds. Runs without variant
fields count as ``simple/constrained``. Runs below ``outputs/_superseded/`` are ignored.

Usage: ``python scripts/summarize.py [--outputs outputs]``
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from pino.visualization import plot_summary

COLUMNS = {
    "benchmark": "Benchmark",
    "variant": "Variante",
    "mode": "Modus",
    "variable": "Variable",
    "n_seeds": "Seeds",
    "rel_l2_mean": "rel. L2 Mittel",
    "rel_l2_min": "Min (Seeds)",
    "rel_l2_max": "Max (Seeds)",
    "per_variable": "Alle Variablen (Mittel)",
    "training_time_s": "Zeit [s]",
    "n_parameters": "Parameter",
}
SUPERSEDED = "_superseded"
MODE_ORDER = {"physics": 0, "data": 1, "hybrid": 2}
DEFAULT_VARIANT = "simple/constrained"


def variant(metrics: dict) -> str:
    return f"{metrics.get('inputs', 'simple')}/{metrics.get('output_param', 'constrained')}"


def latest_per_seed(outputs: Path) -> dict[tuple[str, str, str], dict[int, dict]]:
    """Group key -> seed -> most recent metrics."""
    groups: dict[tuple[str, str, str], dict[int, dict]] = {}
    for path in outputs.rglob("metrics.json"):
        if SUPERSEDED in path.relative_to(outputs).parts:
            continue
        metrics = json.loads(path.read_text())
        seeds = groups.setdefault((metrics["benchmark"], variant(metrics), metrics["mode"]), {})
        seed = int(metrics["seed"])
        if seed not in seeds or metrics["timestamp"] > seeds[seed]["timestamp"]:
            seeds[seed] = metrics
    return groups


def collect(outputs: Path) -> list[dict]:
    """Return one row per benchmark, variant and mode, aggregated over the seeds."""
    rows = []
    for (benchmark, variant_name, mode), seeds in latest_per_seed(outputs).items():
        runs = [seeds[seed] for seed in sorted(seeds)]
        errors = [run["rel_l2"]["mean"] for run in runs]
        names = list(runs[0].get("rel_l2_per_variable", {}))
        per_variable = {name: mean(run["rel_l2_per_variable"][name]["mean"] for run in runs) for name in names}
        rows.append(
            {
                "benchmark": benchmark,
                "variant": variant_name,
                "mode": mode,
                "label": benchmark if variant_name == DEFAULT_VARIANT else f"{benchmark}\n{variant_name}",
                "variable": runs[0].get("main_variable", "-"),
                "n_seeds": len(runs),
                "seeds": sorted(seeds),
                "rel_l2_mean": mean(errors),
                "rel_l2_min": min(errors),
                "rel_l2_max": max(errors),
                "per_variable": per_variable,
                "training_time_s": mean(run["training_time_s"] for run in runs),
                "n_parameters": runs[0]["n_parameters"],
            }
        )
    return sorted(
        rows,
        key=lambda r: (r["benchmark"], r["variant"] != DEFAULT_VARIANT, r["variant"], MODE_ORDER.get(r["mode"], 99)),
    )


def format_row(row: dict) -> list[str]:
    return [
        row["benchmark"],
        row["variant"],
        row["mode"],
        row["variable"],
        str(row["n_seeds"]),
        f"{100 * row['rel_l2_mean']:.2f} %",
        f"{100 * row['rel_l2_min']:.2f} %",
        f"{100 * row['rel_l2_max']:.2f} %",
        ", ".join(f"{name} {100 * value:.2f} %" for name, value in row["per_variable"].items()),
        f"{row['training_time_s']:.0f}",
        f"{row['n_parameters']:,}".replace(",", " "),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    rows = collect(args.outputs)
    if not rows:
        raise SystemExit(f"No metrics.json found under {args.outputs}.")

    with (args.outputs / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, "per_variable": json.dumps(row["per_variable"])} for row in rows)

    lines = [
        "| " + " | ".join(COLUMNS.values()) + " |",
        "|" + " --- |" * len(COLUMNS),
        *("| " + " | ".join(format_row(row)) + " |" for row in rows),
    ]
    table = "\n".join(lines) + "\n"
    (args.outputs / "summary.md").write_text(table)
    plot_summary(rows, args.outputs / "summary.png")
    print(table)


if __name__ == "__main__":
    main()
