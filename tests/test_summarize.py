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
    write(tmp_path, "d", inputs="sine", output_param="unscaled", rel_l2={"mean": 0.2, "median": 0, "max": 0})
    write(tmp_path / "_superseded", "e", rel_l2={"mean": 9.0, "median": 0, "max": 0})

    rows = {row["variant"]: row for row in summarize.collect(tmp_path)}
    # "a" to "c" carry no variant fields: they count as the original family of beam_1d.
    assert set(rows) == {"linear/constrained", "sine/unscaled"}
    original = rows["linear/constrained"]
    assert original["n_seeds"] == 2 and original["seeds"] == [0, 42]
    assert abs(original["rel_l2_mean"] - 0.02) < 1e-12
    assert original["rel_l2_min"] == 0.01 and original["rel_l2_max"] == 0.03
    assert original["training_time_s"] == 150.0 and original["per_variable"]["M"] == 0.03
    assert rows["sine/unscaled"]["n_seeds"] == 1
    assert original["label"] == "beam_1d" and rows["sine/unscaled"]["label"] == "beam_1d\nsine/unscaled"
    assert original["bc_error_max"] is None


def test_boundary_error_column_takes_the_worst_seed(tmp_path):
    write(tmp_path, "a", seed=0, output_param="soft", bc_error_max={"w": 0.01, "M": 0.5})
    write(tmp_path, "b", seed=1, output_param="soft", bc_error_max={"w": 0.03, "M": 0.5})
    (row,) = summarize.collect(tmp_path)
    assert row["bc_error_max"] == 0.03
    assert summarize.format_row(row)[9] == "3.00 %"


def test_default_variant_is_per_benchmark(tmp_path):
    write(tmp_path, "heat", benchmark="heat_1d", main_variable="T", rel_l2_per_variable={"T": {"mean": 0.01}})
    write(tmp_path, "darcy", benchmark="darcy_layered_1d", inputs="two_layer", output_param="constrained",
          main_variable="h", rel_l2_per_variable={"h": {"mean": 0.01}})
    rows = {row["benchmark"]: row for row in summarize.collect(tmp_path)}
    assert rows["heat_1d"]["variant"] == "sine/constrained" and rows["heat_1d"]["label"] == "heat_1d"
    assert rows["darcy_layered_1d"]["variant"] == "two_layer/constrained"
    assert rows["darcy_layered_1d"]["label"] == "darcy_layered_1d"
