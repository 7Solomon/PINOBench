"""Collect ``metrics.json`` files into ``summary.csv``, ``summary.md`` and ``summary.png``.

If a benchmark and mode were run several times, the most recent run is used. All error
columns refer to the main variable of each benchmark.

Usage: ``python scripts/summarize.py [--outputs outputs]``
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pino.visualization import plot_summary

COLUMNS = {
    "benchmark": "Benchmark",
    "mode": "Modus",
    "variable": "Variable",
    "rel_l2_mean": "rel. L2 Mittel",
    "rel_l2_median": "rel. L2 Median",
    "rel_l2_max": "rel. L2 Max",
    "training_time_s": "Zeit [s]",
    "n_parameters": "Parameter",
}
MODE_ORDER = {"physics": 0, "data": 1, "hybrid": 2}


def collect(outputs: Path) -> list[dict]:
    """Return one row per benchmark and mode, taken from the most recent run."""
    latest: dict[tuple[str, str], dict] = {}
    for path in outputs.rglob("metrics.json"):
        metrics = json.loads(path.read_text())
        key = (metrics["benchmark"], metrics["mode"])
        if key not in latest or metrics["timestamp"] > latest[key]["timestamp"]:
            latest[key] = metrics
    rows = [
        {
            "benchmark": m["benchmark"],
            "mode": m["mode"],
            "variable": m.get("main_variable", "-"),
            "rel_l2_mean": m["rel_l2"]["mean"],
            "rel_l2_median": m["rel_l2"]["median"],
            "rel_l2_max": m["rel_l2"]["max"],
            "training_time_s": m["training_time_s"],
            "n_parameters": m["n_parameters"],
        }
        for m in latest.values()
    ]
    return sorted(rows, key=lambda r: (r["benchmark"], MODE_ORDER.get(r["mode"], 99)))


def format_row(row: dict) -> list[str]:
    return [
        row["benchmark"],
        row["mode"],
        row["variable"],
        f"{100 * row['rel_l2_mean']:.2f} %",
        f"{100 * row['rel_l2_median']:.2f} %",
        f"{100 * row['rel_l2_max']:.2f} %",
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
        writer = csv.DictWriter(file, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

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
