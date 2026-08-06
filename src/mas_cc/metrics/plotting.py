"""Figures rendered from what a run already wrote to disk.

Two unrelated pictures live here because both are "render, don't compute":

`plot_streaming_metrics` — one PNG per population-scope metric of one episode.
Per-agent metrics are skipped: with more than a handful of agents a per-agent
overlay is noise, not signal. Rolling smoothing, if wanted, belongs in the
notebook doing the reading, not in this renderer or in the metric itself.
Option-scope metrics are the exception to "one series per figure": their whole
point is that the curves are shares of one another, so they are drawn together
on one axis by default, where crossing curves *are* the finding. Pass
``separate_options=True`` to additionally get one file per option.

`grid_progress_figure` — the liveness picture for a whole sweep: how full each
grid cell is, and whether anything in it failed.

`plot_cell_curves` — one PNG per `cell_metrics` curve for one finished grid
cell (or one plain, non-grid run). A 3-level curve is a percentile band
(``p10``/``p50``/``p90``) drawn as a shaded region with the middle level as
the line; any other level count is drawn as one line per level, the same
convention `plot_streaming_metrics` uses for option-scope series.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from .aggregate import AggregateResult


def _read_series(streaming_csv: Path) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """metric_name -> series label ("" for population scope) -> sorted (round, value)."""

    by_metric: dict[str, dict[str, list[tuple[int, float]]]] = {}
    with streaming_csv.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["agent_id"]:  # per-agent metric, not plotted
                continue
            series = by_metric.setdefault(row["metric_name"], {}).setdefault(
                row.get("series") or "", []
            )
            series.append((int(row["round_index"]), float(row["value"])))
    for series_map in by_metric.values():
        for points in series_map.values():
            points.sort()
    return by_metric


def plot_streaming_metrics(
    streaming_csv: str | Path, plots_dir: str | Path, *, separate_options: bool = False
) -> list[Path]:
    streaming_csv = Path(streaming_csv)
    plots_dir = Path(plots_dir)
    if not streaming_csv.exists():
        return []
    by_metric = _read_series(streaming_csv)
    if not by_metric:
        return []

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def render(destination: Path, title: str, curves: list[tuple[str, list[tuple[int, float]]]]) -> None:
        figure, axis = plt.subplots(figsize=(7, 4), dpi=120)
        for label, points in curves:
            axis.plot(
                [x for x, _ in points],
                [y for _, y in points],
                marker="o",
                markersize=3,
                label=label or None,
            )
        axis.set_xlabel("round_index")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        if any(label for label, _ in curves):
            axis.legend(frameon=False, fontsize="small")
        figure.tight_layout()
        figure.savefig(destination, metadata={"Software": "MAS-CC"})
        plt.close(figure)
        written.append(destination)

    for metric_name, series_map in by_metric.items():
        curves = sorted(series_map.items())
        render(plots_dir / f"{metric_name}.png", metric_name, curves)
        if separate_options and len(curves) > 1:
            for label, points in curves:
                render(
                    plots_dir / f"{metric_name}_{label}.png",
                    f"{metric_name} [{label}]",
                    [(label, points)],
                )
    return written


@dataclass(frozen=True, slots=True)
class CellProgress:
    """One grid cell's fill state, in the coordinates of the swept axes."""

    coordinates: tuple[Any, ...]
    done: int
    total: int
    failed: int = 0


def grid_progress_figure(
    axes: Sequence[tuple[str, Sequence[Any]]],
    cells: Sequence[CellProgress],
    *,
    title: str = "grid progress",
):
    """A 2-D map of the sweep: colour is fill fraction, red is "something failed".

    The colour choice is the whole design. A cell that merely *finished* and a
    cell that finished *with failures* must not look the same, or a green grid
    stops meaning a healthy grid — so failures are pushed out of the colour
    scale entirely (``NaN`` with ``set_bad("red")``) rather than being blended
    into it. Each cell is annotated ``done/total`` so the picture is still
    readable when several cells sit at similar fractions.

    Sweeps of more than two axes are drawn over the first two, with the
    remaining axes summed into them; the spec's grids are at most two axes, and
    a marginal is a more honest degradation than silently dropping cells.

    Returns a matplotlib ``Figure`` rather than a path: the caller either logs
    it to Comet or saves it, and rendering twice for the two cases would let
    the picture on disk drift from the picture on the dashboard.
    """

    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    from matplotlib import pyplot as plt

    used = list(axes[:2])
    if not used:
        raise ValueError("grid_progress_figure needs at least one swept axis")
    row_values = list(used[0][1])
    column_values = list(used[1][1]) if len(used) > 1 else [""]
    done = np.zeros((len(row_values), len(column_values)), dtype=float)
    total = np.zeros_like(done)
    failed = np.zeros_like(done)
    for cell in cells:
        if not cell.coordinates or cell.coordinates[0] not in row_values:
            continue
        row = row_values.index(cell.coordinates[0])
        column = (
            column_values.index(cell.coordinates[1])
            if len(used) > 1 and len(cell.coordinates) > 1 and cell.coordinates[1] in column_values
            else 0
        )
        done[row, column] += cell.done
        total[row, column] += cell.total
        failed[row, column] += cell.failed

    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(total > 0, done / np.where(total > 0, total, 1), 0.0)
    fraction[failed > 0] = np.nan

    colormap = plt.get_cmap("viridis").with_extremes(bad="red")
    figure, axis = plt.subplots(
        figsize=(1.6 * len(column_values) + 3, 1.1 * len(row_values) + 2), dpi=120
    )
    image = axis.imshow(fraction, cmap=colormap, vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_yticks(range(len(row_values)), [str(value) for value in row_values])
    axis.set_ylabel(used[0][0])
    if len(used) > 1:
        axis.set_xticks(range(len(column_values)), [str(value) for value in column_values])
        axis.set_xlabel(used[1][0])
    else:
        axis.set_xticks([])
    for row in range(len(row_values)):
        for column in range(len(column_values)):
            axis.text(
                column, row, f"{int(done[row, column])}/{int(total[row, column])}",
                ha="center", va="center", fontsize="small",
                color="white" if failed[row, column] or fraction[row, column] < 0.6 else "black",
            )
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="episodes done / total")
    figure.tight_layout()
    return figure


def plot_cell_curves(result: "AggregateResult", plots_dir: str | Path) -> list[Path]:
    plots_dir = Path(plots_dir)
    if not result.curves:
        return []

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name, curve in sorted(result.curves.items()):
        figure, axis = plt.subplots(figsize=(7, 4), dpi=120)
        if len(curve.levels) == 3:
            lower, center, upper = (curve.level(level) for level in curve.levels)
            xs = [x for x, _ in center]
            axis.fill_between(
                xs, [y for _, y in lower], [y for _, y in upper],
                alpha=0.2, label=f"{curve.levels[0]}–{curve.levels[2]}",
            )
            axis.plot(xs, [y for _, y in center], marker="o", markersize=3, label=curve.levels[1])
        else:
            for level in curve.levels:
                points = curve.level(level)
                axis.plot(
                    [x for x, _ in points], [y for _, y in points],
                    marker="o", markersize=3, label=None if level == "value" else level,
                )
        axis.set_xlabel("round_index")
        axis.set_ylabel(name)
        axis.set_title(name)
        axis.grid(alpha=0.25)
        if axis.get_legend_handles_labels()[1]:
            axis.legend(frameon=False, fontsize="small")
        figure.tight_layout()
        destination = plots_dir / f"{name}.png"
        figure.savefig(destination, metadata={"Software": "MAS-CC"})
        plt.close(figure)
        written.append(destination)
    return written
