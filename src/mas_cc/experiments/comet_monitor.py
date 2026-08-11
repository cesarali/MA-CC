"""Master-only Comet logging: one sweep experiment, one experiment per cell.

Three writers are conceivable and only one is used.

| Process | Writes                                                          |
|---------|-----------------------------------------------------------------|
| Worker  | its episode's files to the filesystem. Nothing else. No Comet.   |
| Master  | all Comet logging: heartbeat, progress, grid image, aggregates.  |

`observability.recorder.CometMetricSink` is *per episode* and the orchestrator
keeps it switched off (`comet_enabled=False` in `_execute_episode`): a 20-cell
grid would otherwise become hundreds of remote experiments racing each other's
step counters. What is left is one writer per experiment key, which has three
consequences worth stating because they are the reason for the design:

- **No step-counter races.** One writer, one monotonic step, per experiment.
- **Comet is a view, not the store.** If it fails, the run is unaffected — the
  same numbers can be re-logged later from the episode files on disk.
- **A job killed at 80% still has correct aggregates** for every cell that
  finished, because a cell is logged at its own completion, not at the end.

Two kinds of experiment, each with exactly one meaning for `step`; mixing them
is what makes a Comet project unreadable:

*Sweep experiment* (one per run) — ``step = episodes_done``. The "is it
running" dashboard: heartbeat, progress, ETA, the grid image, and each cell's
headline scalars as it lands.

*Cell experiment* (one per cell, created when the cell completes) —
``step = round``. The aggregate curves and, when requested, their rendered
plots. Because the step axis is unambiguously the round index, Comet overlays
cells against each other natively, which is the comparison the whole sweep
exists to make.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from mas_cc.config import CometObservability
from mas_cc.metrics import AggregateResult, CellProgress, SweepResult, grid_progress_figure
from mas_cc.observability.recorder import CometMetricSink

LOGGER = logging.getLogger("mas_cc.experiment.comet")

_TERMINAL_STATUSES = ("completed", "failed", "skipped_resumed", "skipped_aborted")


@dataclass(frozen=True, slots=True)
class CellLayout:
    """One cell's identity as the master needs it: where it sits, and how big."""

    coordinates: tuple[Any, ...] = ()
    episodes: int = 0


@dataclass(frozen=True, slots=True)
class SweepLayout:
    """The grid's shape, known before anything runs.

    Needed up front because the grid image must show *empty* cells too — a
    picture that only draws cells which have started cannot distinguish "not
    scheduled" from "not started yet", which is exactly the distinction you are
    squinting at from a phone six hours in.
    """

    axes: tuple[tuple[str, tuple[Any, ...]], ...] = ()
    cells: Mapping[str, CellLayout] = field(default_factory=dict)

    @property
    def total_episodes(self) -> int:
        return sum(cell.episodes for cell in self.cells.values())


class MasterMonitor:
    """The only process that talks to Comet, for one experiment or grid run.

    Disabled instances are inert but still safe to call, so the orchestrator
    never branches on whether monitoring is on. Every remote call is wrapped:
    a Comet outage must not fail a run whose real output is already on disk.
    """

    def __init__(
        self,
        enabled: bool,
        *,
        project_name: str,
        run_name: str,
        layout: SweepLayout | None = None,
        total_episodes: int = 0,
        settings: CometObservability | None = None,
        now: Any = time.monotonic,
    ) -> None:
        self.settings = settings or CometObservability()
        self.layout = layout or SweepLayout()
        self.run_name = run_name
        self._project = project_name
        self._enabled = enabled and self.settings.sweep_experiment
        self._sink = CometMetricSink(self._enabled, project_name=project_name, run_name=run_name)
        self._total = max(1, total_episodes or self.layout.total_episodes)
        self._now = now
        self._started = now()
        self._lock = threading.RLock()
        self._counts: Counter[str] = Counter()
        self._done_per_cell: Counter[str] = Counter({cell_id: 0 for cell_id in self.layout.cells})
        self._failed_per_cell: Counter[str] = Counter()
        self._finished = 0
        self._cells_complete = 0
        self._budget: Mapping[str, Any] | None = None
        self._heartbeat: threading.Thread | None = None
        self._stop = threading.Event()
        self._cell_summaries: list[dict[str, Any]] = []

    # -- lifecycle ------------------------------------------------------------

    @property
    def status(self) -> str:
        return self._sink.status

    @property
    def episodes_done(self) -> int:
        return self._finished

    @property
    def analysis_sink(self) -> CometMetricSink | None:
        """The master's own sink when the post-run analysis should publish onto it.

        ``None`` means "open your own experiment", which is the default and the
        right answer for a grid: one analysis over many cells is its own object
        of study. Under ``master_aggregates`` the report joins the run it
        describes instead, so there is a single experiment to open.
        """

        if not self._enabled or not self.settings.master_aggregates:
            return None
        return self._sink

    def describe(self) -> str:
        """One console line saying where this run is being watched, or why it isn't.

        Reports the *connection*, not the config: "logging.comet is true" and
        "Comet is receiving this run" are different claims, and only the second
        one is worth printing. A run that silently uploaded nothing because the
        key was missing looks identical to one that was never meant to upload,
        which is the confusion this line exists to end.
        """

        if not self._enabled:
            return "off  (set logging.comet: true in the config)"
        if self._sink.status != "active":
            return f"{self._sink.status}  ({self._sink.reason or 'no reason reported'})"
        target = self._sink.url or self._sink.reference or "(no link reported)"
        cells = "on" if self.settings.cell_experiments else "off"
        return f"master -> project {self._project!r}, cell experiments {cells}\n                 {target}"

    def start(self, parameters: Mapping[str, Any] | None = None) -> None:
        """Log the run's parameters and start the heartbeat timer."""

        if parameters:
            self._guarded(lambda: self._sink.log_parameters(parameters))
        self._guarded(lambda: self._sink.add_tags(("sweep", "master_only")))
        self._publish()
        if not self._enabled:
            return
        # A daemon thread rather than an asyncio task: the heartbeat must keep
        # ticking while the event loop is blocked, since "the loop is wedged"
        # is precisely one of the failures it exists to reveal.
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop, name=f"comet-heartbeat-{self.run_name}", daemon=True
        )
        self._heartbeat.start()

    def _heartbeat_loop(self) -> None:
        interval = self.settings.heartbeat_seconds
        while not self._stop.wait(interval):
            # Published unconditionally, even when nothing has completed. A
            # metric that only moves on completion cannot tell a dead master
            # from a slow one; a flatlined heartbeat means dead, not slow, and
            # that is the entire point of putting it on a timer.
            self._publish()

    def close(self) -> dict[str, Any]:
        """Stop the heartbeat, render the final grid image, end the experiment."""

        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=self.settings.heartbeat_seconds + 5.0)
            self._heartbeat = None
        self._publish()
        self._log_grid_image()
        summary = self._sink.close()
        return {
            **summary,
            "writer": self.settings.writer,
            "episodes_finished": self._finished,
            "cells_complete": self._cells_complete,
            "cell_experiments": self._cell_summaries,
        }

    # -- events ---------------------------------------------------------------

    def episode_finished(
        self,
        *,
        status: str,
        cell_id: str | None = None,
        budget_status: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one terminal episode; refresh progress and, periodically, the image."""

        with self._lock:
            self._counts[status] += 1
            self._finished += 1
            if cell_id is not None:
                if status in ("completed", "skipped_resumed"):
                    self._done_per_cell[cell_id] += 1
                elif status == "failed":
                    self._failed_per_cell[cell_id] += 1
            if budget_status:
                self._budget = budget_status
            due = self._finished % self.settings.grid_image_every_n_episodes == 0
        self._publish()
        if due:
            self._log_grid_image()

    def cell_completed(
        self,
        cell_id: str,
        result: AggregateResult,
        sweep: SweepResult | None = None,
        *,
        metric_plots: Sequence[str | Path] = (),
    ) -> dict[str, Any]:
        """Publish one finished cell: curves, requested plots, and headline scalars.

        Returns the cell experiment's status summary so the caller can persist
        it — the record of what was published belongs on disk with the run, not
        only on the dashboard it was published to.
        """

        with self._lock:
            self._cells_complete += 1
        summary = self._log_cell_experiment(cell_id, result, metric_plots)
        headline = {
            f"cell_{cell_id}_{name}": float(value)
            for name, value in result.scalars.items()
            if name in ("converged_fraction", "median_consensus_round")
        }
        if sweep is not None:
            headline.update({name: float(value) for name, value in sweep.scalars.items()})
        if headline:
            self._guarded(lambda: self._sink.log_metrics(headline, self._finished))
        self._publish()
        self._log_grid_image()
        self._cell_summaries.append(summary)
        return summary

    # -- the three things the master publishes --------------------------------

    def _progress_metrics(self) -> dict[str, float]:
        elapsed = max(self._now() - self._started, 1e-9) / 60.0
        rate = self._finished / elapsed
        remaining = max(self._total - self._finished, 0)
        metrics: dict[str, float] = {
            f"episodes_{name}": float(self._counts[name]) for name in _TERMINAL_STATUSES
        }
        metrics.update(
            {
                "episodes_done": float(self._finished),
                "progress_fraction": self._finished / self._total,
                "cells_complete": float(self._cells_complete),
                "elapsed_minutes": elapsed,
                "episodes_per_minute": rate,
                # Reported as -1 rather than omitted before the first episode
                # lands: a missing series and an unknown ETA look identical on
                # a dashboard, and only one of them is a reason to worry.
                "eta_minutes": remaining / rate if rate > 0 else -1.0,
            }
        )
        metrics.update(_budget_metrics(self._budget))
        if self.settings.progress_metrics:
            requested = set(self.settings.progress_metrics)
            return {name: value for name, value in metrics.items() if name in requested}
        return metrics

    def _publish(self) -> None:
        with self._lock:
            metrics, step = self._progress_metrics(), self._finished
        self._guarded(lambda: self._sink.log_metrics(metrics, step))

    def _log_grid_image(self) -> None:
        if not self._enabled or not self.layout.axes:
            return

        def render() -> None:
            figure = self.grid_figure()
            if figure is None:
                return
            try:
                self._sink.log_figure(figure, name="grid_progress", step=self._finished)
            finally:
                from matplotlib import pyplot as plt

                plt.close(figure)

        self._guarded(render)

    def save_grid_figure(self, destination: Any) -> Any | None:
        """Write the grid-progress picture beside the run, or ``None`` if it has no axes.

        Local and remote get the *same* figure, and the local one is written
        whether or not Comet is enabled. That is what makes "is the dashboard
        right?" a checkable question rather than a matter of trust, and it
        leaves a grid run watchable by tailing a directory on a cluster where
        the dashboard is unreachable.
        """

        figure = self.grid_figure()
        if figure is None:
            return None
        from pathlib import Path

        from matplotlib import pyplot as plt

        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.savefig(path, metadata={"Software": "MAS-CC"})
        finally:
            plt.close(figure)
        return path

    def grid_figure(self):
        """The current grid-progress figure, or ``None`` if the run has no axes.

        Public because it is useful on its own: the same picture can be saved
        beside the run for a job that is watched by tailing a directory rather
        than by opening a dashboard.
        """

        if not self.layout.axes:
            return None
        with self._lock:
            cells = [
                CellProgress(
                    coordinates=cell.coordinates,
                    done=self._done_per_cell[cell_id],
                    total=cell.episodes,
                    failed=self._failed_per_cell[cell_id],
                )
                for cell_id, cell in self.layout.cells.items()
            ]
            title = f"{self.run_name} — {self._finished}/{self._total} episodes"
        return grid_progress_figure(self.layout.axes, cells, title=title)

    def _log_cell_experiment(
        self,
        cell_id: str,
        result: AggregateResult,
        metric_plots: Sequence[str | Path] = (),
    ) -> dict[str, Any]:
        """One Comet experiment for one completed cell, stepped by round index."""

        if not self._enabled:
            return {
                "cell_id": cell_id,
                "status": "disabled",
                "curves": len(result.curves),
                "metric_plots": 0,
            }
        if self.settings.master_aggregates:
            return self._log_cell_onto_master(cell_id, result, metric_plots)
        if not self.settings.cell_experiments:
            return {
                "cell_id": cell_id,
                "status": "disabled",
                "curves": len(result.curves),
                "metric_plots": 0,
            }
        sink = CometMetricSink(
            True, project_name=self._project, run_name=f"{self.run_name}/{cell_id}"
        )
        coordinates = self.layout.cells.get(cell_id, CellLayout()).coordinates
        axis_names = [name for name, _ in self.layout.axes]
        self._guarded(
            lambda: sink.log_parameters(
                {
                    "sweep_id": self.run_name,
                    "cell_id": cell_id,
                    "episodes": result.episodes,
                    **{
                        axis_names[index] if index < len(axis_names) else f"axis_{index}": value
                        for index, value in enumerate(coordinates)
                    },
                }
            )
        )
        self._guarded(lambda: sink.add_tags(("cell", self.run_name)))
        uploaded_plots = self._publish_cell_series(sink, result, metric_plots)
        summary = sink.close()
        return {
            "cell_id": cell_id,
            "curves": len(result.curves),
            "metric_plots": uploaded_plots,
            **summary,
        }

    def _log_cell_onto_master(
        self,
        cell_id: str,
        result: AggregateResult,
        metric_plots: Sequence[str | Path] = (),
    ) -> dict[str, Any]:
        """Publish a finished cell onto the master experiment, no child created.

        The master's own progress series are stepped by ``episodes_done`` and
        these curves are stepped by ``round``. Comet keeps one step axis per
        metric name, so the two coexist; what they must not do is *share* a
        name, which is why a grid's cells are prefixed with their cell id here.
        A single-cell run keeps the bare metric names, so its charts read the
        same as they would on a dedicated cell experiment.
        """

        prefix = "" if cell_id in ("run", "") else f"{cell_id}_"
        uploaded_plots = self._publish_cell_series(
            self._sink, result, metric_plots, prefix=prefix
        )
        return {
            "cell_id": cell_id,
            "curves": len(result.curves),
            "metric_plots": uploaded_plots,
            "status": self._sink.status,
            "reference": self._sink.reference,
            "url": self._sink.url,
            "published_to": "master",
        }

    def _publish_cell_series(
        self,
        sink: CometMetricSink,
        result: AggregateResult,
        metric_plots: Sequence[str | Path] = (),
        *,
        prefix: str = "",
    ) -> int:
        """Log one cell's curves, scalars and plots to `sink`; return plots sent."""

        # Curves are transposed to one log_metrics call per round rather than
        # one per point: the step axis is the round, and Comet's line charts
        # only overlay cells cleanly when every series shares that axis.
        by_round: dict[int, dict[str, float]] = {}
        for name, curve in result.curves.items():
            for round_index, values in curve.points.items():
                row = by_round.setdefault(round_index, {})
                for level, value in zip(curve.levels, values):
                    key = name if level == "value" else f"{name}_{level}"
                    row[f"{prefix}{key}"] = float(value)
        for round_index in sorted(by_round):
            values = by_round[round_index]
            self._guarded(lambda: sink.log_metrics(values, round_index))
        last = max(by_round) if by_round else 0
        if result.scalars:
            scalars = {f"{prefix}{name}": float(value) for name, value in result.scalars.items()}
            self._guarded(lambda: sink.log_metrics(scalars, last))
        uploaded_plots = 0
        if self.settings.metric_plots:
            for plot in metric_plots:
                path = Path(plot)
                if not path.is_file():
                    LOGGER.warning("metric plot does not exist; skipping %s", path)
                    continue
                self._guarded(
                    lambda path=path: sink.log_image(
                        path, name=f"metric_plot_{prefix}{path.stem}", step=last
                    )
                )
                uploaded_plots += 1
        return uploaded_plots

    def _guarded(self, call: Any) -> None:
        """Run a remote call, downgrading any failure to a log line.

        Comet is a view of a run whose real output is already on disk, so a
        remote failure is never allowed to fail the run. The exception text is
        not persisted: SDK errors can carry request URLs or headers the
        connector saw alongside the credential.
        """

        try:
            call()
        except Exception as exc:  # a dashboard outage must not kill a day-long job
            LOGGER.warning("Comet call failed (%s); continuing", type(exc).__name__)


def _budget_metrics(budget_status: Mapping[str, Any] | None) -> dict[str, float]:
    """Flatten the spend side of `RuntimeBudgetGuard.status()` into scalars.

    Only the used/reserved totals are sent; the approved limits are static
    configuration and already recorded on disk, so re-sending them on every
    heartbeat would just be noise on the remote charts.
    """

    if not budget_status:
        return {}
    used = budget_status.get("used_and_reserved") or {}
    metrics: dict[str, float] = {}
    for key in ("requests", "input_tokens", "output_tokens"):
        value = used.get(key)
        if isinstance(value, (int, float)):
            metrics[f"budget_{key}"] = float(value)
    cost = used.get("cost")
    if isinstance(cost, Mapping) and isinstance(cost.get("amount"), (int, float)):
        metrics["budget_cost"] = float(cost["amount"])
    return metrics


def sweep_parameters(config: Any, axes: Sequence[tuple[str, Sequence[Any]]] = ()) -> dict[str, Any]:
    """The scalar parameter set describing what this sweep is.

    Flattened to scalars here rather than in the sink, so what reaches Comet is
    decided in one readable place: the grid spec, the per-cell repetitions, the
    seed, the game, and the aggregation settings that determine what every
    logged curve *means*. Nothing prompt-shaped, and nothing nested.
    """

    aggregation = config.aggregation
    parameters: dict[str, Any] = {
        # Identity first: opening a six-hour-old experiment and having to guess
        # what it was is the failure this section prevents.
        "experiment_name": config.experiment.name,
        "experiment_description": config.experiment.description,
        "experiment_tags": ",".join(config.experiment.tags),
        "game": config.game.type,
        "population_size": config.game.population_size,
        "horizon": config.game.horizon,
        "provider": config.llm_provider.type,
        "model": config.llm_provider.model,
        "seed": config.execution.seed,
        "repetitions_per_cell": config.execution.repetitions,
        "parallelism": config.execution.parallelism,
        "prompt_family": config.prompt.prompt_family,
        "prompt_version": config.prompt.prompt_version,
        "forward_fill": aggregation.forward_fill,
        "relabel_by_winner": aggregation.relabel_by_winner,
        "percentiles": ",".join(str(p) for p in aggregation.percentiles),
        "rolling_window": aggregation.rolling_window,
        "cell_metrics": ",".join(aggregation.resolved_cell_metrics()),
        "sweep_metrics": ",".join(aggregation.sweep_metrics),
        "horizons": ",".join(str(h) for h in aggregation.horizons),
        "null_permutations": aggregation.null_permutations,
        "grid_axes": ",".join(path for path, _ in axes) or "(none - single cell)",
    }
    # The game's own knobs, flattened. These are what actually differ between
    # two runs of the same experiment name, so a dashboard without them cannot
    # answer "what was different about this one".
    for key, value in config.game.options.items():
        if isinstance(value, (str, int, float, bool)):
            parameters[f"game.options.{key}"] = value
        elif isinstance(value, (list, tuple)):
            parameters[f"game.options.{key}"] = ",".join(str(item) for item in value)
    cells = 1
    for path, values in axes:
        levels = tuple(values)
        parameters[f"grid.{path}"] = ",".join(str(value) for value in levels)
        cells *= max(1, len(levels))
    parameters["cells"] = cells
    parameters["total_episodes"] = cells * config.execution.repetitions
    return parameters


__all__ = ["CellLayout", "MasterMonitor", "SweepLayout", "sweep_parameters"]
