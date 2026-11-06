"""Plots of a training run. ``solution.png`` is dispatched on ``Problem.plot_type``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pino.evaluation import EvaluationResult  # noqa: E402
from pino.problems.base import Problem  # noqa: E402

# Fixed categorical order, validated for color-vision deficiencies.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
REFERENCE_COLOR = "#52514e"
TARGET_REL_L2 = 0.05
TIME_FRACTIONS = (0.1, 0.4, 1.0)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "image.interpolation": "nearest",
    }
)


def run_title(problem: Problem, mode: str, rel_l2: float) -> str:
    """Title format ``Benchmark · Modus · rel. L2``."""
    return f"{problem.name} · {mode} · rel. L2 = {100 * rel_l2:.2f} %"


def _save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _percent_axis(ax) -> None:
    """Label a logarithmic error axis in percent.

    Labels at 1, 2, 3, 5 and 7 per decade keep the scale readable when the data spans less than
    one decade, where the default locator would not place a single labeled tick.
    """
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 3, 5, 7), numticks=30))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:g} %"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def plot_training(history: dict, title: str, path: Path) -> None:
    """Loss components (left) and test error (right) over the epochs."""
    fig, (ax_loss, ax_error) = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)

    names = [name for name in history["losses"] if name != "total"]
    for name, color in zip(names, SERIES_COLORS):
        ax_loss.plot(history["loss_epochs"], history["losses"][name], color=color, label=name)
    ax_loss.set(yscale="log", xlabel="Epoche", ylabel="Loss", title="Loss-Komponenten")
    ax_loss.legend()

    ax_error.plot(history["eval_epochs"], history["test_rel_l2"], color=SERIES_COLORS[0])
    ax_error.axhline(TARGET_REL_L2, color=REFERENCE_COLOR, linestyle="--", linewidth=1)
    ax_error.annotate(
        "5 %",
        xy=(1, TARGET_REL_L2),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 3),
        textcoords="offset points",
        ha="right",
        color=REFERENCE_COLOR,
    )
    ax_error.set(yscale="log", xlabel="Epoche", ylabel="rel. L2 (Testset, Mittel)", title="Testfehler")
    _percent_axis(ax_error)

    fig.suptitle(title)
    _save(fig, path)


def plot_error_histogram(errors: np.ndarray, title: str, path: Path) -> None:
    """Histogram of the per-example error with mean, median and maximum marked."""
    fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
    ax.hist(100 * errors, bins=30, color=SERIES_COLORS[0], edgecolor="white", linewidth=0.6)
    markers = (
        ("Mittelwert", errors.mean(), "-", SERIES_COLORS[1]),
        ("Median", np.median(errors), "--", SERIES_COLORS[2]),
        ("Maximum", errors.max(), ":", SERIES_COLORS[3]),
    )
    for label, value, style, color in markers:
        ax.axvline(100 * value, color=color, linestyle=style, label=f"{label}: {100 * value:.2f} %")
    ax.set(xlabel="rel. L2 [%]", ylabel="Anzahl Testbeispiele", title=title)
    ax.grid(axis="x", visible=False)
    ax.legend()
    _save(fig, path)


def plot_resolution(resolution_errors: dict[int, dict[str, float]], train_resolution: int, title: str, path: Path):
    """Mean test error against the test resolution, with the training resolution marked."""
    resolutions = sorted(resolution_errors)
    errors = [resolution_errors[r]["mean"] for r in resolutions]
    fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
    ax.plot(resolutions, errors, color=SERIES_COLORS[0], marker="o", markersize=5)
    ax.axvline(train_resolution, color=REFERENCE_COLOR, linestyle="--", linewidth=1, label="Trainingsauflösung")
    ax.set(xscale="log", yscale="log", xlabel="Testauflösung (je Dimension)", ylabel="rel. L2 (Mittel)", title=title)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(resolutions))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    _percent_axis(ax)
    ax.legend()
    _save(fig, path)


def _representative_examples(errors: np.ndarray) -> list[tuple[str, int]]:
    """Return the indices of the best, median and worst example."""
    order = np.argsort(errors)
    return [("Bestes", order[0]), ("Median", order[len(order) // 2]), ("Schlechtestes", order[-1])]


def plot_solution_1d_t(problem: Problem, result: EvaluationResult, title: str, path: Path) -> None:
    """Heatmaps over ``(x, t)`` and profiles at three times, one row per representative example."""
    x_name, t_name = problem.variable_names["coordinates"]
    u_name = problem.variable_names["solution"][0]
    x = problem.to_physical(x_name, result.dataset.grid[0, :, 0].numpy())
    t = problem.to_physical(t_name, result.dataset.grid[1, 0, :].numpy())
    extent = (x[0], x[-1], t[0], t[-1])
    errors = result.errors.numpy()

    fig, axes = plt.subplots(
        3, 4, figsize=(14, 8.4), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1, 1, 1.15]}
    )
    for row, (label, index) in zip(axes, _representative_examples(errors)):
        exact = problem.to_physical(u_name, result.dataset.targets[index, 0].numpy())
        predicted = problem.to_physical(u_name, result.predictions[index, 0].numpy())
        error = np.abs(exact - predicted)

        vmin, vmax = min(exact.min(), predicted.min()), max(exact.max(), predicted.max())
        panels = (
            ("Analytisch", exact, "viridis", vmin, vmax),
            ("FNO", predicted, "viridis", vmin, vmax),
            ("|Fehler|", error, "magma", 0.0, error.max()),
        )
        images = []
        for ax, (name, field, cmap, low, high) in zip(row, panels):
            image = ax.imshow(field.T, origin="lower", extent=extent, aspect="auto", cmap=cmap, vmin=low, vmax=high)
            images.append(image)
            ax.set(xlabel=problem.label(x_name), ylabel=problem.label(t_name), title=name)
            ax.grid(False)
        row[0].set_title(f"{label} (rel. L2 = {100 * errors[index]:.2f} %)\nAnalytisch")
        fig.colorbar(images[1], ax=row[:2], label=problem.label(u_name), shrink=0.9)
        fig.colorbar(images[2], ax=row[2], label=f"|Fehler| {problem.label(u_name)}", shrink=0.9)

        profile_axis = row[3]
        for fraction, color in zip(TIME_FRACTIONS, SERIES_COLORS):
            time_index = round(fraction * (len(t) - 1))
            profile_axis.plot(x, exact[:, time_index], color=color, label=f"{t_name} = {t[time_index]:.3g}")
            profile_axis.plot(x, predicted[:, time_index], color=color, linestyle="--")
        profile_axis.plot([], [], color=REFERENCE_COLOR, label="Analytisch")
        profile_axis.plot([], [], color=REFERENCE_COLOR, linestyle="--", label="FNO")
        profile_axis.set(xlabel=problem.label(x_name), ylabel=problem.label(u_name), title="Schnitte")
        profile_axis.legend(fontsize=7)

    fig.suptitle(title)
    _save(fig, path)


#: Solution plot for each ``Problem.plot_type``. Add ``"1d"``, ``"2d"`` and ``"2d_t"`` here.
SOLUTION_PLOTTERS: dict[str, Callable[[Problem, EvaluationResult, str, Path], None]] = {
    "1d_t": plot_solution_1d_t,
}


def plot_solution(problem: Problem, result: EvaluationResult, title: str, path: Path) -> None:
    """Dispatch the solution plot on ``problem.plot_type``."""
    if problem.plot_type not in SOLUTION_PLOTTERS:
        raise NotImplementedError(f"No solution plot for plot_type '{problem.plot_type}'.")
    SOLUTION_PLOTTERS[problem.plot_type](problem, result, title, path)


def plot_run(problem: Problem, result: EvaluationResult, history: dict, mode: str, train_resolution: int, run_dir) -> None:
    """Write all four plots of a run to ``run_dir``."""
    run_dir = Path(run_dir)
    errors = result.errors.numpy()
    title = run_title(problem, mode, float(errors.mean()))
    plot_training(history, title, run_dir / "training.png")
    plot_solution(problem, result, title, run_dir / "solution.png")
    plot_error_histogram(errors, title, run_dir / "error_hist.png")
    plot_resolution(result.resolution_errors, train_resolution, title, run_dir / "resolution.png")


def plot_summary(rows: list[dict], path: Path) -> None:
    """Grouped bars of the mean error per benchmark and mode (log axis, 5 % line)."""
    benchmarks = sorted({row["benchmark"] for row in rows})
    modes = [mode for mode in ("physics", "data", "hybrid") if any(row["mode"] == mode for row in rows)]
    width = 0.8 / max(len(modes), 1)

    fig, ax = plt.subplots(figsize=(max(5.0, 2.2 * len(benchmarks) + 2), 3.8), constrained_layout=True)
    for offset, (mode, color) in enumerate(zip(modes, SERIES_COLORS)):
        values = {row["benchmark"]: row["rel_l2_mean"] for row in rows if row["mode"] == mode}
        positions = [i - 0.4 + width * (offset + 0.5) for i, b in enumerate(benchmarks) if b in values]
        heights = [values[b] for b in benchmarks if b in values]
        bars = ax.bar(positions, heights, width=width * 0.9, color=color, label=mode)
        ax.bar_label(bars, labels=[f"{100 * h:.2f} %" for h in heights], fontsize=7, padding=2)
    ax.axhline(TARGET_REL_L2, color=REFERENCE_COLOR, linestyle="--", linewidth=1, label="5 %")
    ax.set_xticks(range(len(benchmarks)), benchmarks)
    ax.set(yscale="log", ylabel="rel. L2 (Testset, Mittel)", title="Übersicht aller Benchmarks")
    ax.grid(axis="x", visible=False)
    _percent_axis(ax)
    ax.legend()
    _save(fig, path)
