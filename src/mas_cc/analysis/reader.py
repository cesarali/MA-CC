"""Read a completed grid directory into tidy per-episode / per-round tables.

Consumes only what a grid run already writes to disk - `cells/<cell_id>/overrides.json`
(the condition label, free once `control.*` is a sweepable path - see
`docs/architecture_overview.md`) and each episode's `metrics/streaming.csv`
(the per-round `population_action_share_<value>` StreamingMetric output
`games/naming_convention/metrics.py` already produces). No new `Metric`
classes or `RunRecorder` changes were needed for this - see the plan's
Phase 3 rationale: `Metric` instances are shared across every episode in an
experiment, which rules out a stateful per-round metric, so the rolling/
terminal derivation happens here instead, post-hoc, exactly as the legacy
pipeline's own `derive_episode` was a separate step over a completed episode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_SHARE_PREFIX = "population_action_share_"


@dataclass(frozen=True)
class GridData:
    """Tidy tables assembled from one grid directory's on-disk output."""

    episodes: pd.DataFrame  # cell_id, <condition columns>, episode_id, terminal_outcome
    rounds: pd.DataFrame  # cell_id, <condition columns>, episode_id, round_index, macrostate
    condition_columns: tuple[str, ...]


def _dominant_action(group: pd.DataFrame) -> str:
    """The action whose `population_action_share_*` value is largest in this round.

    Ties are broken by whichever action's metric row comes first - acceptable
    for a first version since a genuine tie is already a boundary case any
    downstream MI estimate should treat as noisy.
    """

    best = group.loc[group["value"].idxmax()]
    return str(best["action_value"])


def read_grid(grid_dir: str | Path) -> GridData:
    root = Path(grid_dir)
    cells_dir = root / "cells"
    if not cells_dir.is_dir():
        raise FileNotFoundError(f"no cells/ directory under {root} - is this a grid output directory?")

    episode_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    condition_columns: set[str] = set()

    for cell_dir in sorted(p for p in cells_dir.iterdir() if p.is_dir()):
        overrides_path = cell_dir / "overrides.json"
        if not overrides_path.exists():
            continue
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))["overrides"]
        condition_columns.update(overrides)

        episodes_dir = cell_dir / "data" / "episodes"
        if not episodes_dir.is_dir():
            continue
        for episode_dir in sorted(p for p in episodes_dir.iterdir() if p.is_dir()):
            streaming_path = episode_dir / "metrics" / "streaming.csv"
            if not streaming_path.exists():
                continue
            streaming = pd.read_csv(streaming_path)
            shares = streaming[
                streaming["agent_id"].isna()
                & streaming["metric_name"].str.startswith(_SHARE_PREFIX)
            ].copy()
            if shares.empty:
                continue
            shares["action_value"] = shares["metric_name"].str.removeprefix(_SHARE_PREFIX)
            episode_id = episode_dir.name

            for round_index, group in shares.groupby("round_index"):
                round_rows.append({
                    "cell_id": cell_dir.name,
                    **overrides,
                    "episode_id": episode_id,
                    "round_index": int(round_index),
                    "macrostate": _dominant_action(group),
                })

            last_round = shares["round_index"].max()
            terminal_group = shares[shares["round_index"] == last_round]
            episode_rows.append({
                "cell_id": cell_dir.name,
                **overrides,
                "episode_id": episode_id,
                "terminal_outcome": _dominant_action(terminal_group),
            })

    return GridData(
        episodes=pd.DataFrame(episode_rows),
        rounds=pd.DataFrame(round_rows),
        condition_columns=tuple(sorted(condition_columns)),
    )


__all__ = ["GridData", "read_grid"]
