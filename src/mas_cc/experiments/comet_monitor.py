"""One Comet experiment for a whole experiment/grid run, not one per episode.

`observability.recorder.CometMetricSink` is *per episode*: it lives inside a
`RunRecorder` and streams that episode's round metrics. The orchestrator
deliberately keeps it switched off (`comet_enabled=False` in `_execute_episode`),
because a 20-episode grid would otherwise fan out into 20 remote experiments
with no object representing the run itself.

This module supplies the missing level: a single Comet experiment named after
the run, which receives *progress and budget* telemetry as episodes finish -
how many are done, how many failed, how far through each cell is, what has been
spent. It is the thing you watch while a long Slurm job is running, and it is
intentionally coarse: no per-round curves, no prompts, no per-agent values.

The two levels are independent. Episode-level Comet stays off on this path;
turning this on does not turn that on.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from mas_cc.observability.recorder import CometMetricSink

_TERMINAL_STATUSES = ("completed", "failed", "skipped_resumed", "skipped_aborted")


class RunLevelCometMonitor:
    """Progress/budget telemetry for one experiment or grid run.

    Reuses `CometMetricSink` for the connection so the privacy stance (metrics
    only; no code, git, env, args, or host details) is defined in exactly one
    place. Disabled instances are inert but still safe to call, so the
    orchestrator never branches on whether monitoring is on.
    """

    def __init__(
        self,
        enabled: bool,
        *,
        project_name: str,
        run_name: str,
        total_episodes: int,
        cell_ids: tuple[str, ...] = (),
    ) -> None:
        self._sink = CometMetricSink(enabled, project_name=project_name, run_name=run_name)
        self._total = max(1, total_episodes)
        self._counts: Counter[str] = Counter()
        self._per_cell: Counter[str] = Counter({cell_id: 0 for cell_id in cell_ids})
        self._finished = 0

    @property
    def status(self) -> str:
        return self._sink.status

    def episode_finished(
        self,
        *,
        status: str,
        cell_id: str | None = None,
        budget_status: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one terminal episode and push the running totals to Comet.

        `step` is the number of episodes finished so far, which makes the
        remote x-axis monotonic even though episodes complete out of order
        under `execution.parallelism`.
        """

        self._counts[status] += 1
        self._finished += 1
        if cell_id is not None and status == "completed":
            self._per_cell[cell_id] += 1

        metrics: dict[str, float] = {
            f"episodes_{name}": float(self._counts[name]) for name in _TERMINAL_STATUSES
        }
        metrics["episodes_finished"] = float(self._finished)
        metrics["progress_fraction"] = self._finished / self._total
        for cell_id_key, count in self._per_cell.items():
            metrics[f"cell_{cell_id_key}_completed"] = float(count)
        metrics.update(_budget_metrics(budget_status))
        self._sink.log_metrics(metrics, self._finished)

    def close(self) -> dict[str, Any]:
        return self._sink.close()


def _budget_metrics(budget_status: Mapping[str, Any] | None) -> dict[str, float]:
    """Flatten the spend side of `RuntimeBudgetGuard.status()` into scalars.

    Only the used/reserved totals are sent; the approved limits are static
    configuration and already recorded on disk, so re-sending them every
    episode would just be noise on the remote charts.
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


__all__ = ["RunLevelCometMonitor"]
