"""One PNG per population-scope metric, rendered from `metrics/streaming.csv`.

Per-agent metrics are skipped here — with more than a handful of agents a
per-agent overlay is noise, not signal. Rolling smoothing, if wanted, belongs
in the notebook doing the reading, not in this renderer or in the metric
itself.

Option-scope metrics are the exception to "one series per figure": their whole
point is that the curves are shares of one another, so they are drawn together
on one axis by default, where crossing curves *are* the finding. Pass
``separate_options=True`` to additionally get one file per option.
"""

from __future__ import annotations

import csv
from pathlib import Path


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
