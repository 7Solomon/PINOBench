"""Evaluate a trained model on one hand-made operator input and save the result.

Usage::

    python scripts/predict.py heat_1d --inputs 1,0,0.5,0,0 [--resolution 128] [--name half_mode]
    python scripts/predict.py heat_1d --mode data --output-param soft --inputs 1,0,0.5,0,0
    python scripts/predict.py <run_dir> --inputs 1,0,0.5,0,0
    python scripts/predict.py heat_1d --show-ranges

The first argument is a run directory or a benchmark name. A name selects the newest finished
run of ``--mode`` (default physics), ``--output-param`` (default constrained), ``--family``
(the input family, default the benchmark's original one) and optionally ``--seed``.

``--inputs`` lists the operator inputs in the problem's own parametrization (the layout of
``Problem.sample_inputs``, flattened; heat_1d: b_1..b_5, chloride_1d: Cs,D). ``--resolution``
is a number or per axis, e.g. ``64x64x32``; default is the run's test resolution. Results go
to ``<run_dir>/predictions/<name>/`` (``inputs.json``, ``result.json``, ``prediction.pt``,
``solution.png``); ``--no-save`` only prints them. Inputs outside the trained range are
evaluated but flagged as extrapolation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from pino.inference import IncompatibleRunError, find_run, load_run, predict_one, save_prediction


def parse_resolution(text: str | None):
    if text is None:
        return None
    values = [int(v) for v in text.lower().split("x")]
    return values[0] if len(values) == 1 else values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run directory or benchmark name (then the newest matching run)")
    parser.add_argument("--mode", default="physics", help="with a benchmark name: training mode")
    parser.add_argument("--output-param", default="constrained", help="with a benchmark name: output_param")
    parser.add_argument("--family", help="with a benchmark name: input family (default: the original one)")
    parser.add_argument("--seed", type=int, help="with a benchmark name: seed (default: any)")
    parser.add_argument("--outputs", default="outputs", help="with a benchmark name: outputs directory")
    parser.add_argument("--inputs", help="comma-separated operator inputs, e.g. 1,0,0.5,0,0")
    parser.add_argument("--resolution", help="e.g. 128 or 64x64x32 (default: test resolution of the run)")
    parser.add_argument("--name", help="subdirectory under predictions/ (default: timestamp)")
    parser.add_argument("--no-save", action="store_true", help="print the result only")
    parser.add_argument("--show-ranges", action="store_true", help="print the trained input ranges and exit")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-legacy", action="store_true", help="accept runs without wrapper_version")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        try:
            run_dir = find_run(args.run, args.mode, args.output_param, args.family, args.seed, args.outputs)
        except (FileNotFoundError, KeyError) as error:
            raise SystemExit(str(error).strip('"')) from None
    print(f"run: {run_dir}")
    try:
        run = load_run(run_dir, args.device, args.allow_legacy)
    except IncompatibleRunError as error:
        raise SystemExit(f"Refusing to load {run_dir}: {error}") from None

    if args.show_ranges or args.inputs is None:
        low, high = run.problem.input_ranges()
        print(f"{run.problem.name}: inputs of shape {tuple(low.shape)}, trained ranges (flattened):")
        for i, (lo, hi) in enumerate(zip(low.flatten().tolist(), high.flatten().tolist())):
            print(f"  [{i}] {lo:.4g} .. {hi:.4g}")
        if args.inputs is None:
            return

    values = [float(v) for v in args.inputs.split(",")]
    prediction = predict_one(run, values, parse_resolution(args.resolution))
    print(json.dumps({"rel_l2": prediction.rel_l2, "bc_error": prediction.bc_error,
                      "extrapolation": prediction.extrapolation}, indent=2))
    if not args.no_save:
        name = args.name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"saved to {save_prediction(run, prediction, name)}")


if __name__ == "__main__":
    main()
