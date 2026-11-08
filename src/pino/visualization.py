"""Plots of a training run. ``solution.png`` is dispatched on ``Problem.plot_type``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines  # noqa: E402
import matplotlib.patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pino.evaluation import EvaluationResult  # noqa: E402
from pino.numerics.grids import resolution_size  # noqa: E402
from pino.problems import problem_class  # noqa: E402
from pino.problems.base import Problem  # noqa: E402

# Fixed categorical order, validated for color-vision deficiencies.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
REFERENCE_COLOR = "#52514e"
TARGET_REL_L2 = 0.05
TIME_FRACTIONS = (0.1, 0.4, 1.0)
#: Rows of the overview: module of the problem class -> title. Others follow by name.
SUMMARY_GROUPS = {
    "mechanics": "Strukturmechanik",
    "thermal": "Wärmeleitung",
    "transport": "Stofftransport",
    "hydro": "Strömung",
}
#: Mode -> offset within a slot, marker shape, color.
SUMMARY_MODES = {
    "physics": (-0.27, "o", SERIES_COLORS[0]),
    "data": (0.0, "s", SERIES_COLORS[1]),
    "hybrid": (0.27, "^", SERIES_COLORS[2]),
}

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
    """Title format ``Benchmark · Modus · rel. L2``. Multi-output problems name the main variable."""
    variable = f" ({problem.main_variable})" if len(problem.variable_names["solution"]) > 1 else ""
    return f"{problem.name} · {mode} · rel. L2{variable} = {100 * rel_l2:.2f} %"


def _save(fig, path: Path | None):
    """Write and close ``fig``; without ``path`` return it instead (for notebooks)."""
    if path is None:
        return fig
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return None


def _percent_axis(ax) -> None:
    """Label a logarithmic error axis in percent.

    Labels at 1, 2, 3, 5 and 7 per decade keep the scale readable when the data spans less than
    one decade, where the default locator would not place a single labeled tick.
    """
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 3, 5, 7), numticks=30))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:g} %"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def plot_training(history: dict, title: str, path: Path) -> None:
    """Loss components (left) and validation error (right) over the epochs.

    The left panel adds the validation physics loss. The right panel shows the validation
    rel. L2 of every output variable. A vertical line marks the selected checkpoint, chosen by
    ``history["selection"]``: the physics loss in ``physics`` mode, otherwise the mean rel. L2
    over all variables.
    """
    fig, (ax_loss, ax_error) = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    best = history["best_epoch"]

    names = [name for name in history["losses"] if name != "total"]
    for name, color in zip(names, SERIES_COLORS):
        ax_loss.plot(history["loss_epochs"], history["losses"][name], color=color, label=f"{name} (Training)")
    ax_loss.plot(
        history["eval_epochs"],
        history["val_physics"],
        color=SERIES_COLORS[0],
        linestyle="--",
        linewidth=1.2,
        label="physics (Validierung)",
    )
    if history.get("val_bc"):
        color = SERIES_COLORS[names.index("bc") % len(SERIES_COLORS)] if "bc" in names else REFERENCE_COLOR
        ax_loss.plot(
            history["eval_epochs"], history["val_bc"], color=color, linestyle="--", linewidth=1.2, label="bc (Validierung)"
        )
    ax_loss.set(yscale="log", xlabel="Epoche", ylabel="Loss", title="Loss-Komponenten")
    ax_loss.legend()

    for (name, values), color in zip(history["val_rel_l2"].items(), SERIES_COLORS):
        ax_error.plot(history["eval_epochs"], values, color=color, label=name)
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
    criterion = {"val_physics": "physics-Loss", "val_objective": "physics + bc"}.get(
        history["selection"], "rel. L2 aller Variablen"
    )
    for ax in (ax_loss, ax_error):
        ax.axvline(best, color=REFERENCE_COLOR, linestyle=":", linewidth=1)
    ax_error.plot([], [], color=REFERENCE_COLOR, linestyle=":", linewidth=1, label=f"Checkpoint ({criterion})")
    ax_error.set(yscale="log", xlabel="Epoche", ylabel="rel. L2 (Validierung, Mittel)", title="Validierungsfehler")
    _percent_axis(ax_error)
    ax_error.legend()

    fig.suptitle(title)
    return _save(fig, path)


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


def plot_resolution(
    resolution_errors: dict[int | str, dict[str, float]], train_resolution: int | str, title: str, path: Path
):
    """Mean test error against the test resolution, with the training resolution marked.

    Keys are scalar resolutions or labels such as ``"64x64x32"``; the latter are placed at their
    first entry and labeled in full.
    """
    keys = sorted(resolution_errors, key=resolution_size)
    resolutions = [resolution_size(key) for key in keys]
    errors = [resolution_errors[key]["mean"] for key in keys]
    fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
    ax.plot(resolutions, errors, color=SERIES_COLORS[0], marker="o", markersize=5)
    ax.axvline(
        resolution_size(train_resolution), color=REFERENCE_COLOR, linestyle="--", linewidth=1, label="Trainingsauflösung"
    )
    label = "Testauflösung (je Dimension)" if all(isinstance(k, int) for k in keys) else "Testauflösung (Raum × Raum × Zeit)"
    ax.set(xscale="log", yscale="log", xlabel=label, ylabel="rel. L2 (Mittel)", title=title)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(resolutions))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(key) for key in keys]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    _percent_axis(ax)
    ax.legend()
    _save(fig, path)


def _representative_examples(errors: np.ndarray) -> list[tuple[str, int]]:
    """Return the indices of the best, median and worst example; a single example stands alone."""
    if len(errors) == 1:
        return [("Beispiel", 0)]
    order = np.argsort(errors)
    return [("Bestes", order[0]), ("Median", order[len(order) // 2]), ("Schlechtestes", order[-1])]


def plot_solution_1d_t(problem: Problem, result: EvaluationResult, title: str, path: Path | None):
    """Heatmaps over ``(x, t)`` and profiles at three times, one row per representative example."""
    x_name, t_name = problem.variable_names["coordinates"]
    u_name = problem.variable_names["solution"][0]
    x = problem.to_physical(x_name, result.dataset.grid[0, :, 0].numpy())
    t = problem.to_physical(t_name, result.dataset.grid[1, 0, :].numpy())
    extent = (x[0], x[-1], t[0], t[-1])
    errors = result.main_errors.numpy()
    examples = _representative_examples(errors)

    fig, axes = plt.subplots(
        len(examples), 4, figsize=(14, 2.8 * len(examples)), constrained_layout=True, squeeze=False,
        gridspec_kw={"width_ratios": [1, 1, 1, 1.15]},
    )
    for row, (label, index) in zip(axes, examples):
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
    return _save(fig, path)


def plot_solution_1d(problem: Problem, result: EvaluationResult, title: str, path: Path | None):
    """Profiles of every output variable for the best, median and worst example.

    Each example gets two rows: analytical (solid) against FNO (dashed), and below it the
    absolute error. Output variables sit side by side.
    """
    x_name = problem.variable_names["coordinates"][0]
    names = problem.variable_names["solution"]
    x = problem.to_physical(x_name, result.dataset.grid[0].numpy())
    main_errors = result.main_errors.numpy()
    examples = _representative_examples(main_errors)

    fig, axes = plt.subplots(
        2 * len(examples),
        len(names),
        figsize=(4.6 * len(names), 3.5 * len(examples)),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1] * len(examples)},
    )
    for block, (label, index) in enumerate(examples):
        for channel, name in enumerate(names):
            solution_axis, error_axis = axes[2 * block, channel], axes[2 * block + 1, channel]
            exact = problem.to_physical(name, result.dataset.targets[index, channel].numpy())
            predicted = problem.to_physical(name, result.predictions[index, channel].numpy())
            error = 100 * result.errors[index, channel].item()

            solution_axis.plot(x, exact, color=SERIES_COLORS[0], label="Analytisch")
            solution_axis.plot(x, predicted, color=SERIES_COLORS[1], linestyle="--", label="FNO")
            solution_axis.set(ylabel=problem.label(name), title=f"{label} · {name} · rel. L2 = {error:.2f} %")
            error_axis.plot(x, np.abs(exact - predicted), color=REFERENCE_COLOR, linewidth=1.2)
            error_axis.set(ylabel=f"|Fehler| [{problem.units[name]}]")
            if block == 0 and channel == 0:
                solution_axis.legend()
    for axis in axes[-1]:
        axis.set_xlabel(problem.label(x_name))
    fig.suptitle(title)
    return _save(fig, path)


def plot_solution_2d(problem: Problem, result: EvaluationResult, title: str, path: Path | None):
    """Fields over ``(x, y)`` for the best, median and worst example, one row each.

    Panels: analytical and FNO with a shared ``viridis`` scale, the absolute error with its
    own ``magma`` scale, and a line cut through the grid row closest to ``y = 0.5``.
    """
    x_name, y_name = problem.variable_names["coordinates"]
    u_name = problem.variable_names["solution"][0]
    x = problem.to_physical(x_name, result.dataset.grid[0, :, 0].numpy())
    y = problem.to_physical(y_name, result.dataset.grid[1, 0, :].numpy())
    cut = int(np.argmin(np.abs(result.dataset.grid[1, 0, :].numpy() - 0.5)))
    extent = (x[0], x[-1], y[0], y[-1])
    errors = result.main_errors.numpy()
    examples = _representative_examples(errors)

    fig, axes = plt.subplots(
        len(examples), 4, figsize=(14, 3.2 * len(examples)), constrained_layout=True, squeeze=False,
        gridspec_kw={"width_ratios": [1, 1, 1, 1.15]},
    )
    for row, (label, index) in zip(axes, examples):
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
            image = ax.imshow(field.T, origin="lower", extent=extent, aspect="equal", cmap=cmap, vmin=low, vmax=high)
            images.append(image)
            ax.set(xlabel=problem.label(x_name), ylabel=problem.label(y_name), title=name)
            ax.grid(False)
        row[0].set_title(f"{label} (rel. L2 = {100 * errors[index]:.2f} %)\nAnalytisch")
        row[0].axhline(y[cut], color="white", linestyle=":", linewidth=0.8)
        fig.colorbar(images[1], ax=row[:2], label=problem.label(u_name), shrink=0.9)
        fig.colorbar(images[2], ax=row[2], label=f"|Fehler| {problem.label(u_name)}", shrink=0.9)

        profile_axis = row[3]
        profile_axis.plot(x, exact[:, cut], color=SERIES_COLORS[0], label="Analytisch")
        profile_axis.plot(x, predicted[:, cut], color=SERIES_COLORS[1], linestyle="--", label="FNO")
        profile_axis.set(
            xlabel=problem.label(x_name), ylabel=problem.label(u_name), title=f"Schnitt {y_name} = {y[cut]:.3g}"
        )
        profile_axis.legend(fontsize=7)

    fig.suptitle(title)
    return _save(fig, path)


def plot_solution_2d_t(problem: Problem, result: EvaluationResult, title: str, path: Path | None):
    """One figure per representative example: ``solution_best.png``, ``_median`` and ``_worst``.

    The files are written next to ``path``. Each figure is a 3 x 3 grid: rows analytical, FNO and
    the absolute error, columns the first, middle and last time. Analytical and FNO share one
    ``viridis`` scale over all three times, the error has its own ``magma`` scale. A result with a
    single example gives one figure at ``path`` itself (returned when ``path`` is None).
    """
    x_name, y_name, t_name = problem.variable_names["coordinates"]
    u_name = problem.variable_names["solution"][0]
    grid = result.dataset.grid
    x = problem.to_physical(x_name, grid[0, :, 0, 0].numpy())
    y = problem.to_physical(y_name, grid[1, 0, :, 0].numpy())
    t = problem.to_physical(t_name, grid[2, 0, 0, :].numpy())
    times = [0, (len(t) - 1) // 2, len(t) - 1]
    time_unit = "" if problem.units[t_name] == "-" else f" {problem.units[t_name]}"
    extent = (x[0], x[-1], y[0], y[-1])
    errors = result.main_errors.numpy()
    channel = problem.main_channel

    examples = _representative_examples(errors)
    for (label, index), tag in zip(examples, ("best", "median", "worst")):
        exact = problem.to_physical(u_name, result.dataset.targets[index, channel].numpy())
        predicted = problem.to_physical(u_name, result.predictions[index, channel].numpy())
        error = np.abs(exact - predicted)
        vmin, vmax = min(exact.min(), predicted.min()), max(exact.max(), predicted.max())
        error_max = max(float(error.max()), 1e-30)

        fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
        rows = (("Analytisch", exact, "viridis", vmin, vmax), ("FNO", predicted, "viridis", vmin, vmax),
                ("|Fehler|", error, "magma", 0.0, error_max))
        for row_axes, (name, field, cmap, low, high) in zip(axes, rows):
            for ax, time_index in zip(row_axes, times):
                image = ax.imshow(field[:, :, time_index].T, origin="lower", extent=extent, aspect="equal",
                                  cmap=cmap, vmin=low, vmax=high)
                ax.set(xlabel=problem.label(x_name), ylabel=problem.label(y_name),
                       title=f"{name} · {t_name} = {t[time_index]:.3g}{time_unit}")
                ax.grid(False)
            unit = problem.label(u_name) if name != "|Fehler|" else f"|Fehler| {problem.label(u_name)}"
            fig.colorbar(image, ax=row_axes, label=unit, shrink=0.9)
        fig.suptitle(f"{title}\n{label} (rel. L2 = {100 * errors[index]:.2f} %)")
        if len(examples) == 1:
            return _save(fig, path)
        _save(fig, Path(path).with_name(f"solution_{tag}.png"))


#: Solution plot for each ``Problem.plot_type``. ``"2d_t"`` writes three files next to ``path``.
SOLUTION_PLOTTERS: dict[str, Callable[[Problem, EvaluationResult, str, Path], None]] = {
    "1d": plot_solution_1d,
    "1d_t": plot_solution_1d_t,
    "2d": plot_solution_2d,
    "2d_t": plot_solution_2d_t,
}


def plot_solution(problem: Problem, result: EvaluationResult, title: str, path: Path | None):
    """Dispatch the solution plot on ``problem.plot_type``; returns the figure if ``path`` is None."""
    if problem.plot_type not in SOLUTION_PLOTTERS:
        raise NotImplementedError(f"No solution plot for plot_type '{problem.plot_type}'.")
    return SOLUTION_PLOTTERS[problem.plot_type](problem, result, title, path)


def plot_prediction(problem: Problem, prediction, mode: str, path: Path | None = None):
    """Solution plot of a single :class:`pino.inference.Prediction`.

    Without ``path`` the figure is returned, e.g. to show it inline in a notebook. The title
    marks inputs outside the trained range.
    """
    title = run_title(problem, mode, prediction.rel_l2[problem.main_variable])
    title += f" · Auflösung {prediction.resolution}"
    if prediction.extrapolation:
        title += " · EXTRAPOLATION"
    return plot_solution(problem, prediction.as_result(problem), title, path)


def plot_run(problem: Problem, result: EvaluationResult, history: dict, mode: str, train_resolution: int, run_dir) -> None:
    """Write all four plots of a run to ``run_dir``."""
    run_dir = Path(run_dir)
    errors = result.main_errors.numpy()
    title = run_title(problem, mode, float(errors.mean()))
    plot_training(history, title, run_dir / "training.png")
    plot_solution(problem, result, title, run_dir / "solution.png")
    plot_error_histogram(errors, title, run_dir / "error_hist.png")
    plot_resolution(result.main_resolution_errors(), train_resolution, title, run_dir / "resolution.png")


def _problem_group(benchmark: str) -> str:
    """Physics group of a benchmark: the module of its problem class (mechanics, thermal, ...)."""
    try:
        return problem_class(benchmark).__module__.rsplit(".", 1)[-1]
    except KeyError:
        return "other"


def plot_summary(rows: list[dict], path: Path) -> None:
    """Overview of the mean error: one row per physics group, one slot per benchmark and variant.

    A dot plot, because bar lengths mean nothing on a logarithmic axis. Each mode has a color and
    a marker shape, whiskers span the minimum and maximum over the seeds, values are in percent.
    All rows share one error axis and one slot width, so they can be compared directly. The
    groups follow the modules of the problem classes, so a new benchmark needs no entry here.
    """
    groups: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for row in rows:
        group = groups.setdefault(_problem_group(row["benchmark"]), {})
        group.setdefault(row["benchmark"], {}).setdefault(row.get("variant", ""), []).append(row)
    order = [name for name in SUMMARY_GROUPS if name in groups] + sorted(set(groups) - set(SUMMARY_GROUPS))
    slots = [sum(len(variants) for variants in groups[name].values()) + len(groups[name]) - 1 for name in order]

    values = [row["rel_l2_mean"] for row in rows] + [row.get("rel_l2_min", row["rel_l2_mean"]) for row in rows]
    highs = [row.get("rel_l2_max", row["rel_l2_mean"]) for row in rows] + [TARGET_REL_L2]
    limits = (min(values) / 2.0, max(highs) * 2.5)

    width = max(7.0, 1.0 * max(slots) + 1.2)
    fig, axes = plt.subplots(
        len(order), 1, figsize=(width, 2.5 * len(order) + 0.7), constrained_layout=True, squeeze=False,
    )
    for ax, name in zip(axes[:, 0], order):
        position = 0
        for benchmark, variants in groups[name].items():
            first = position
            for variant, entries in variants.items():
                by_mode = {entry["mode"]: entry for entry in entries}
                for mode, (offset, marker, color) in SUMMARY_MODES.items():
                    if mode not in by_mode:
                        continue
                    entry = by_mode[mode]
                    mean = entry["rel_l2_mean"]
                    spread = [[mean - entry.get("rel_l2_min", mean)], [entry.get("rel_l2_max", mean) - mean]]
                    ax.errorbar(position + offset, mean, yerr=spread, fmt=marker, color=color, markersize=6,
                                markeredgecolor="white", markeredgewidth=0.6, elinewidth=1.0, capsize=2, zorder=3)
                    # Above the upper whisker, with a halo so that no line crosses the number.
                    ax.annotate(f"{100 * mean:.2f}", (position + offset, mean + spread[1][0]), xytext=(0, 5),
                                textcoords="offset points", ha="center", fontsize=6.5, color=REFERENCE_COLOR, zorder=4,
                                path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground="white")])
                position += 1
            # Benchmark name above its slots, thin rule between benchmarks.
            ax.annotate(benchmark, ((first + position - 1) / 2, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
            if position < slots[order.index(name)]:
                ax.axvline(position, color="#c9c8c3", linewidth=0.8)
                position += 1
        labels = {}
        position = 0
        for variants in groups[name].values():
            for variant in variants:
                labels[position] = variant.replace("/", "\n")
                position += 1
            position += 1
        ax.set_xticks(list(labels), list(labels.values()), fontsize=7.5)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, max(slots) - 0.4)  # the same slot width in every row
        ax.axhline(TARGET_REL_L2, color=REFERENCE_COLOR, linestyle="--", linewidth=1, zorder=1)
        ax.annotate("5 %", xy=(1, TARGET_REL_L2), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                    textcoords="offset points", ha="right", color=REFERENCE_COLOR, fontsize=7.5)
        ax.set(yscale="log", ylim=limits, ylabel="rel. L2 (Testset)")
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=12))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:g} %"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(axis="x", visible=False)
        ax.set_title(SUMMARY_GROUPS.get(name, name.capitalize()), loc="left", fontweight="bold", fontsize=9.5, pad=16)
    handles = [
        matplotlib.lines.Line2D([], [], linestyle="", marker=marker, color=color, markersize=6, label=mode)
        for mode, (_, marker, color) in SUMMARY_MODES.items()
    ]
    fig.legend(handles=handles, loc="outside upper right", ncols=3, fontsize=8, title="Modus (Werte in %, Whisker: Min–Max über Seeds)",
               title_fontsize=8)
    fig.suptitle("Übersicht aller Benchmarks", x=0.01, ha="left", fontsize=11)
    _save(fig, path)
