"""Write the weights of a trained model to a text file.

Usage::

    python scripts/export_weights.py heat_1d --mode hybrid  # workspace/weights/heat_1d/sine-constrained.txt
    python scripts/export_weights.py heat_1d --mode data --out workspace/weights/heat_1d/sine-constrained-data.txt
    python scripts/export_weights.py <run_dir> --digits 9   # restores every value exactly
    python scripts/export_weights.py heat_1d --only "lifting.*,projection.*" --out head.txt

The first argument is a run directory or a benchmark name. A name selects the newest finished
run of ``--mode`` (default physics), ``--output-param`` (default constrained), ``--family``
(the input family, default the benchmark's original one) and optionally ``--seed``.

The default file name holds neither mode nor seed, and the file does not name its run. The
script prints the run; it refuses to replace an existing file without ``--force``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pino.inference import find_run
from pino.paths import OUTPUTS
from pino.weights_text import DEFAULT_DIGITS, default_path, load_weights, select, to_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run directory or benchmark name (then the newest matching run)")
    parser.add_argument("--mode", default="physics", help="with a benchmark name: training mode")
    parser.add_argument("--output-param", default="constrained", help="with a benchmark name: output_param")
    parser.add_argument("--family", help="with a benchmark name: input family (default: the original one)")
    parser.add_argument("--seed", type=int, help="with a benchmark name: seed (default: any)")
    parser.add_argument("--outputs", default=OUTPUTS, help="with a benchmark name: outputs directory")
    parser.add_argument("--digits", type=int, default=DEFAULT_DIGITS, help="significant digits; 9 is exact")
    parser.add_argument("--only", help="comma-separated globs of tensor names, e.g. 'lifting.*,projection.*'")
    parser.add_argument("--out", type=Path, help="output file (default: below workspace/weights/)")
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        try:
            run_dir = find_run(args.run, args.mode, args.output_param, args.family, args.seed, args.outputs)
        except (FileNotFoundError, KeyError) as error:
            raise SystemExit(str(error).strip('"')) from None

    path = args.out or default_path(run_dir)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} already exists. Use --force to replace it, or --out for another file.")
    state = select(load_weights(run_dir), args.only)
    if not state:
        raise SystemExit(f"No tensor matches --only {args.only!r}.")

    text, total = to_text(state, args.digits)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"run: {run_dir}")
    print(f"wrote {total} numbers in {len(state)} tensors, {len(text)} chars (~{len(text) // 3} tokens) -> {path}")


if __name__ == "__main__":
    main()
