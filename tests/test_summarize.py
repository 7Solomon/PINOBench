"""scripts/summarize.py: grouping by benchmark x variant x mode and aggregation over seeds."""

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("summarize", Path(__file__).parents[1] / "scripts" / "summarize.py")
summarize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize)


def write(root: Path, name: str, **fields) -> None:
    metrics = {
        "benchmark": "beam_1d", "mode": "data", "seed": 42, "timestamp": "2026-01-01T00:00:00",
        "main_variable": "w", "rel_l2": {"mean": 0.01, "median": 0.01, "max": 0.02},
        "rel_l2_per_variable": {"w": {"mean": 0.01}, "M": {"mean": 0.03}},
        "training_time_s": 100.0, "n_parameters": 10, **fields,
    }
    path = root / name
    path.mkdir(parents=True)
    (path / "metrics.json").write_text(json.dumps(metrics))


def test_groups_by_variant_and_aggregates_over_seeds(tmp_path):
    write(tmp_path, "a", seed=42, rel_l2={"mean": 0.01, "median": 0, "max": 0})
    write(tmp_path, "b", seed=0, rel_l2={"mean": 0.03, "median": 0, "max": 0}, training_time_s=200.0)
    # Older run of seed 0 is replaced by the newer one above.
    write(tmp_path, "c", seed=0, rel_l2={"mean": 0.50, "median": 0, "max": 0}, timestamp="2025-01-01T00:00:00")
    write(tmp_path, "d", inputs="rich", output_param="unscaled", rel_l2={"mean": 0.2, "median": 0, "max": 0})
    write(tmp_path / "_superseded", "e", rel_l2={"mean": 9.0, "median": 0, "max": 0})

    rows = {row["variant"]: row for row in summarize.collect(tmp_path)}
    assert set(rows) == {"simple/constrained", "rich/unscaled"}
    simple = rows["simple/constrained"]
    assert simple["n_seeds"] == 2 and simple["seeds"] == [0, 42]
    assert abs(simple["rel_l2_mean"] - 0.02) < 1e-12
    assert simple["rel_l2_min"] == 0.01 and simple["rel_l2_max"] == 0.03
    assert simple["training_time_s"] == 150.0 and simple["per_variable"]["M"] == 0.03
    assert rows["rich/unscaled"]["n_seeds"] == 1
