"""The master's cell-completion work: aggregate, persist, publish, re-estimate.

This is the half of master-only logging that produces numbers rather than
sending them. It runs entirely from what the workers already wrote to disk, and
writes its own output back to disk *before* anything is published, so:

- the aggregates survive Comet being down, unreachable, or switched off;
- they can be recomputed from the same files at any later time, with different
  percentiles or a different window, without re-running one episode;
- a run killed partway still has complete, correct output for every cell that
  finished, because a cell is finalized at its own completion rather than at
  the end of the sweep.

`aggregate_grid_directory` is the recomputation entry point, and it is the same
code path the live run uses — not a reimplementation of it, which is the only
way "re-aggregate later" stays trustworthy.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from mas_cc.config import AggregationConfig, GridSpec, RunConfig
from mas_cc.metrics import (
    AggregateResult,
    SweepResult,
    aggregate_cell,
    create_cell_metrics,
    create_sweep_metrics,
    plot_cell_curves,
    excluded_cell_episodes,
    read_cell_episodes,
    sweep_over_cells,
)

from .comet_monitor import MasterMonitor

LOGGER = logging.getLogger("mas_cc.experiment.aggregation")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


class GridAggregator:
    """Aggregates one cell at a time and keeps the grid estimate up to date.

    Sweep metrics are re-run on *every* cell completion, not once at the end.
    They read only the per-cell count tables, so the cost is negligible and the
    payoff is real: a partial MI estimate appears early and tightens as the
    grid fills, which is how you notice a broken condition six hours in rather
    than the next morning.
    """

    def __init__(
        self,
        root: str | Path,
        aggregation: AggregationConfig,
        *,
        monitor: MasterMonitor | None = None,
        ground_truth: float | None = None,
        seed: int = 1,
    ) -> None:
        self.root = Path(root)
        self.config = aggregation
        self.monitor = monitor
        self.policy = aggregation.policy()
        if ground_truth is None:
            ground_truth = _recorded_ground_truth(self.root)
        elif "mi_ground_truth_gap" in aggregation.sweep_metrics:
            # Persisted so a later recomputation still has the answer key. It
            # is derived from the resolved grid and the game object, neither of
            # which a run directory carries, so without this file re-running
            # aggregation would quietly drop the gap it once reported.
            _write_json(
                self.root / "sweep_ground_truth.json",
                {"terminal_mutual_information": ground_truth},
            )
        self._cell_metrics = create_cell_metrics(
            aggregation.resolved_cell_metrics(), horizons=aggregation.horizons
        )
        self._sweep_metrics = create_sweep_metrics(
            aggregation.sweep_metrics,
            horizons=aggregation.horizons,
            permutations=aggregation.null_permutations,
            seed=seed,
            ground_truth=ground_truth,
        )
        self.results: dict[str, AggregateResult] = {}
        self.sweep: SweepResult = SweepResult()
        # Cells complete concurrently and each aggregation runs on its own
        # worker thread (see the orchestrator's `asyncio.to_thread` call), so
        # two of them can reach the grid-level estimate at once. Serializing
        # here keeps `sweep_metrics.json` a whole file rather than two
        # interleaved ones.
        self._lock = threading.Lock()

    def cell_directory(self, cell_id: str | None) -> Path:
        """Where one cell's episodes live.

        ``None`` means "this whole run is one cell", which is exactly what a
        non-grid experiment is: N episodes of one resolved config. It gets the
        same aggregation rather than a second, parallel implementation.
        """

        return self.root if cell_id is None else self.root / "cells" / cell_id

    def aggregate(self, cell_id: str | None) -> AggregateResult:
        """Aggregate one finished cell, persist it, and refresh the sweep estimate."""

        directory = self.cell_directory(cell_id)
        episodes = read_cell_episodes(directory)
        # An episode that died partway through is left out of the curves (see
        # `read_cell_episodes`), which means the aggregate's N can be smaller
        # than the configured repetitions. Recording *which* episodes and why,
        # next to the numbers they are absent from, is what stops that looking
        # like an unexplained sample size later.
        excluded = excluded_cell_episodes(directory)
        result = aggregate_cell(episodes, self._cell_metrics, self.policy)
        label = cell_id or "run"
        if excluded:
            LOGGER.warning(
                "cell %s aggregated %d episode(s); excluded %d incomplete: %s",
                label,
                len(episodes),
                len(excluded),
                ", ".join(f"{episode} ({status})" for episode, status in excluded),
            )
        # The cell's own file first, and unconditionally: it must survive the
        # sweep estimate failing, Comet being unreachable, and the run being
        # killed on the next episode.
        _write_json(
            directory / "aggregate.json",
            {
                **result.to_dict(),
                "cell_id": cell_id,
                "aggregation": self.config.to_dict(),
                "episodes_aggregated": len(episodes),
                "episodes_excluded": [
                    {"episode_id": episode, "status": status} for episode, status in excluded
                ],
            },
        )
        # Best-effort: a plotting failure (e.g. a matplotlib backend issue on a
        # headless node) must not take down a run whose real output — the JSON
        # just written above — is already correct and on disk.
        plot_paths: list[Path] = []
        try:
            plot_paths = plot_cell_curves(result, directory / "metrics" / "plots")
        except Exception as exc:
            LOGGER.warning("could not render cell curve plots for %s (%s)", label, type(exc).__name__)
        with self._lock:
            self.results[label] = result
            sweep = self._refresh_sweep()
        if self.monitor is not None:
            # Published for a plain experiment too, under the label "run". N
            # episodes of one config are one cell, and their aggregate curves
            # are exactly as worth looking at as a grid cell's - they just have
            # nothing to be compared against. Logging them on their own
            # experiment rather than onto the sweep experiment keeps `step`
            # meaning one thing per experiment: rounds here, episodes there.
            self.monitor.cell_completed(label, result, sweep, metric_plots=plot_paths)
        return result

    def finish(self) -> SweepResult:
        """Final sweep pass over every cell aggregated so far."""

        with self._lock:
            return self._refresh_sweep()

    def _refresh_sweep(self) -> SweepResult:
        """Re-estimate the grid from the cells finished so far. Caller holds the lock."""

        self.sweep = sweep_over_cells(self.results, self._sweep_metrics)
        if self._sweep_metrics:
            _write_json(
                self.root / "sweep_metrics.json",
                {
                    **self.sweep.to_dict(),
                    "cells_complete": len(self.results),
                    "aggregation": self.config.to_dict(),
                },
            )
        return self.sweep


def grid_ground_truth(grid: GridSpec, game: Any, horizons: Sequence[int]) -> float | None:
    """The closed-form terminal ``I(C;O)`` for this sweep, when one exists.

    Only a synthetic game has an answer key, and it is a property of the
    *sweep* — which cells ran and with how many episodes each — so it is built
    from the grid's own resolved cells rather than from any single config. A
    real game, a multi-axis grid, or anything the closed form refuses returns
    ``None``, and `mi_ground_truth_gap` then reports nothing: a missing answer
    key must never be published as a zero gap.
    """

    if len(grid.axes) != 1:
        return None
    try:
        from mas_cc.games.synthetic import SyntheticGame
        from mas_cc.games.synthetic.empowerment import (
            AnalysisSpec,
            SweepCell,
            SweepSpec,
            sweep_ground_truth,
        )
    except ImportError:  # analysis stack unavailable; the run itself is unaffected
        return None
    if not isinstance(game, SyntheticGame):
        return None
    try:
        sweep = SweepSpec(
            condition_name=grid.axes[0].path,
            cells=tuple(
                SweepCell(cell.cell_id, cell.config.game, cell.config.execution.repetitions)
                for cell in grid.cells
            ),
        )
        truth = sweep_ground_truth(game, sweep, AnalysisSpec(horizons=tuple(horizons)))
        return float(truth.value("terminal_mutual_information"))
    except Exception as exc:
        # A game whose closed form does not cover this sweep is a normal
        # outcome, not a failure of the run: report no answer key and continue.
        LOGGER.info("no closed-form ground truth for this sweep (%s)", type(exc).__name__)
        return None


def aggregate_grid_directory(
    grid_dir: str | Path,
    aggregation: AggregationConfig | None = None,
    *,
    seed: int = 1,
) -> dict[str, Any]:
    """Recompute every cell's aggregates from a finished run directory.

    The offline counterpart to what the master does live, and deliberately the
    same code: this is how a curve gets a new percentile band, a different
    rolling window, or a corrected fill rule without touching an episode.
    Cells with no episodes on disk are skipped rather than written as empty.
    """

    root = Path(grid_dir)
    if aggregation is None:
        aggregation = _aggregation_from_run(root)
    aggregator = GridAggregator(root, aggregation, seed=seed)
    cells_dir = root / "cells"
    cell_ids: list[str | None] = (
        [path.name for path in sorted(cells_dir.iterdir()) if path.is_dir()]
        if cells_dir.is_dir()
        else [None]
    )
    aggregated: list[str] = []
    for cell_id in cell_ids:
        if not read_cell_episodes(aggregator.cell_directory(cell_id)):
            continue
        aggregator.aggregate(cell_id)
        aggregated.append(cell_id or "run")
    sweep = aggregator.finish()
    return {
        "grid_dir": str(root),
        "cells_aggregated": aggregated,
        "sweep_metrics": dict(sweep.scalars),
        "aggregation": aggregation.to_dict(),
    }


def _recorded_ground_truth(root: Path) -> float | None:
    """The answer key a live run wrote for this sweep, if it had one."""

    path = root / "sweep_ground_truth.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))["terminal_mutual_information"]
    except (KeyError, ValueError):
        return None
    return None if value is None else float(value)


def _aggregation_from_run(root: Path) -> AggregationConfig:
    """The ``aggregation:`` section the run itself recorded, or the defaults.

    Re-reading the run's own settings is what makes a recomputation reproduce
    the original numbers by default; passing a different section is then an
    explicit act, which is the only way the difference stays interpretable.

    Only that one section is re-parsed. A resolved config on disk is a superset
    of a loadable one — it records derived provenance the loader rejects as
    unknown fields — so re-parsing the whole file would fail on exactly the
    provenance that makes the file worth writing.
    """

    import yaml

    from mas_cc.config import parse_aggregation_config

    for name in ("resolved_base_config.yaml", "resolved_config.yaml"):
        path = root / name
        if not path.exists():
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping) and isinstance(raw.get("aggregation"), Mapping):
            return parse_aggregation_config(raw["aggregation"])
    return AggregationConfig()


def aggregation_ground_truth(config: RunConfig, grid: GridSpec | None, game: Any) -> float | None:
    """Answer key for the configured sweep metrics, or ``None`` if not requested."""

    if grid is None or "mi_ground_truth_gap" not in config.aggregation.sweep_metrics:
        return None
    return grid_ground_truth(grid, game, config.aggregation.horizons)


__all__ = [
    "GridAggregator",
    "aggregate_grid_directory",
    "aggregation_ground_truth",
    "grid_ground_truth",
]
