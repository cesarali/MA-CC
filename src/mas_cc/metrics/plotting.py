"""One PNG per population-scope metric, rendered from `metrics/streaming.csv`.

Per-agent metrics are skipped here — with more than a handful of agents a
per-agent overlay is noise, not signal. Rolling smoothing, if wanted, belongs
in the notebook doing the reading, not in this renderer or in the metric
itself.
"""

from __future__ import annotations

import csv
from pathlib import Path


def plot_streaming_metrics(streaming_csv: str | Path, plots_dir: str | Path) -> list[Path]:
    streaming_csv = Path(streaming_csv)
    plots_dir = Path(plots_dir)
    if not streaming_csv.exists():
        return []
    rows = list(csv.DictReader(streaming_csv.open(encoding="utf-8")))
    population_rows = [row for row in rows if not row["agent_id"]]
    by_metric: dict[str, list[tuple[int, float]]] = {}
    for row in population_rows:
        by_metric.setdefault(row["metric_name"], []).append((int(row["round_index"]), float(row["value"])))

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for metric_name, points in by_metric.items():
        points.sort()
        figure, axis = plt.subplots(figsize=(7, 4), dpi=120)
        axis.plot([x for x, _ in points], [y for _, y in points], marker="o")
        axis.set_xlabel("round_index")
        axis.set_ylabel(metric_name)
        axis.set_title(metric_name)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        destination = plots_dir / f"{metric_name}.png"
        figure.savefig(destination, metadata={"Software": "MAS-CC"})
        plt.close(figure)
        written.append(destination)
    return written
