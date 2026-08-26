"""Normalize ordinary run artifacts into study-wide scientific tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from mas_cc.storage import canonical_hash

from .discovery import DiscoveredCell


TABLE_SCHEMAS: Mapping[str, tuple[str, ...]] = {
    "cells": (
        "study_id", "source_config_index", "source_run_id", "source_run_path",
        "cell_id", "source_cell_id", "config_hash", "resolved_config_hash",
        "recorded_resolved_config_hash",
        "task_id", "expected_episodes", "completed_episodes", "failed_episodes", "sealed",
    ),
    "episodes": (
        "study_id", "source_config_index", "source_run_id", "source_run_path",
        "cell_id", "source_cell_id", "episode_id", "episode_seed", "status",
        "interaction_count", "usage_requests", "usage_input_tokens", "usage_output_tokens",
        "started_at", "finished_at", "termination_reason", "scientific_schema_version",
    ),
    "rounds": (
        "study_id", "source_config_index", "source_run_id", "source_run_path",
        "cell_id", "source_cell_id", "episode_id", "round_index", "record_source",
    ),
    "micro_slots": (
        "study_id", "source_config_index", "source_run_id", "source_run_path",
        "cell_id", "source_cell_id", "episode_id", "round_index", "micro_slot_index",
        "record_source",
    ),
}


def _nested(mapping: Mapping[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _coordinates(cell: DiscoveredCell) -> dict[str, Any]:
    result = {str(key): value for key, value in cell.overrides.items()}
    leaves: dict[str, list[str]] = {}
    for path in result:
        leaves.setdefault(path.rsplit(".", 1)[-1], []).append(path)
    for leaf, paths in leaves.items():
        if len(paths) == 1 and leaf not in result:
            result[leaf] = result[paths[0]]
    for dotted, label in (
        ("game.options.task_id", "task_id"),
        ("game.population_size", "population_size"),
        ("control.options.sensor_sample_size", "sensor_sample_size"),
        ("control.options.intervention_budget", "intervention_budget"),
        ("control.options.beta", "beta"),
        ("control.options.threshold", "threshold"),
    ):
        value = _nested(cell.resolved_config, dotted)
        if value is not None and label not in result:
            result[label] = value
    return result


def _expected_repetitions(cell: DiscoveredCell) -> int:
    value = _nested(cell.resolved_config, "execution.repetitions")
    return int(value) if value is not None else 0


def _scientific_frame(cell: DiscoveredCell) -> pd.DataFrame:
    direct = cell.path / "scientific_events.parquet"
    if direct.is_file():
        return pd.read_parquet(direct, engine="pyarrow")
    shards = sorted(cell.path.glob(".resume/*/scientific_events.parquet"))
    if not shards:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path, engine="pyarrow") for path in shards], ignore_index=True)


def _provenance(study_id: str, cell: DiscoveredCell) -> dict[str, Any]:
    return {
        "study_id": study_id,
        "source_config_index": cell.run.entry.array_index,
        "source_run_id": cell.run.run_id,
        "source_run_path": str(cell.run.path),
        "cell_id": cell.cell_key,
        "source_cell_id": cell.local_cell_id,
    }


def _episode_rows(study_id: str, cell: DiscoveredCell, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    provenance = _provenance(study_id, cell)
    if not frame.empty and "episode_id" in frame:
        for episode_id, group in frame.groupby("episode_id", sort=True, dropna=False):
            first = group.iloc[0]
            rows.append(
                {
                    **provenance,
                    "episode_id": str(episode_id),
                    "episode_seed": first.get("episode_seed"),
                    "status": str(first.get("status", "completed")),
                    "interaction_count": int(first.get("interaction_count", len(group))),
                    "usage_requests": first.get("usage_requests"),
                    "usage_input_tokens": first.get("usage_input_tokens"),
                    "usage_output_tokens": first.get("usage_output_tokens"),
                    "started_at": first.get("started_at"),
                    "finished_at": first.get("finished_at"),
                    "termination_reason": first.get("termination_reason"),
                    "scientific_schema_version": first.get("schema_version"),
                }
            )
        return rows

    # Full-profile runs retain one manifest per episode rather than a compact table.
    for path in sorted((cell.path / "data" / "episodes").glob("*/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                **provenance,
                "episode_id": str(payload.get("episode_id", path.parent.name)),
                "episode_seed": payload.get("seed"),
                "status": str(payload.get("status", "unknown")),
                "interaction_count": payload.get("interactions", 0),
                "usage_requests": _nested(payload, "usage.requests"),
                "usage_input_tokens": _nested(payload, "usage.input_tokens"),
                "usage_output_tokens": _nested(payload, "usage.output_tokens"),
                "started_at": payload.get("started_at"),
                "finished_at": payload.get("finished_at"),
                "termination_reason": payload.get("termination_reason"),
                "scientific_schema_version": payload.get("scientific_schema_version"),
            }
        )
    return rows


def _jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path} line {number}") from exc
        event = payload.get("event", payload) if isinstance(payload, Mapping) else payload
        if isinstance(event, Mapping):
            yield {**dict(payload), **dict(event)}


def _rich_rows(study_id: str, cell: DiscoveredCell, filename: str, source_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    provenance = _provenance(study_id, cell)
    for path in sorted(cell.path.rglob(filename)):
        for event in _jsonl(path):
            episode_id = str(event.get("episode_id", path.parent.name))
            round_index = event.get("round_index", event.get("interaction_index"))
            slot_index = event.get("micro_slot_index", event.get("within_round_index"))
            rows.append(
                {
                    **event,
                    **provenance,
                    "episode_id": episode_id,
                    "round_index": round_index,
                    "micro_slot_index": slot_index,
                    "record_source": source_label,
                }
            )
    return rows


def _compact_round_rows(study_id: str, cell: DiscoveredCell, frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    provenance = _provenance(study_id, cell)
    result = []
    for row in frame.to_dict(orient="records"):
        result.append(
            {
                **row,
                **provenance,
                "episode_id": str(row.get("episode_id")),
                "round_index": row.get("interaction_index"),
                "record_source": "scientific_events.parquet",
            }
        )
    return result


def build_canonical_tables(
    study_id: str, cells: tuple[DiscoveredCell, ...]
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    cell_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    micro_rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    for cell in cells:
        frame = _scientific_frame(cell)
        frames[cell.cell_key] = frame
        episodes = _episode_rows(study_id, cell, frame)
        episode_rows.extend(episodes)
        seal_path = cell.path / "cell_complete.json"
        seal: Mapping[str, Any] = {}
        if seal_path.is_file():
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
        completed = sum(row["status"] in {"completed", "skipped_resumed"} for row in episodes)
        failed = sum(row["status"] in {"failed", "skipped_aborted", "aborted"} for row in episodes)
        coords = _coordinates(cell)
        cell_rows.append(
            {
                **_provenance(study_id, cell),
                "config_hash": cell.run.entry.config_hash,
                "resolved_config_hash": canonical_hash(cell.resolved_config),
                "recorded_resolved_config_hash": (
                    None
                    if frame.empty or "resolved_config_hash" not in frame
                    else json.dumps(sorted({str(value) for value in frame["resolved_config_hash"].dropna()}))
                ),
                "task_id": coords.get("task_id"),
                **coords,
                "expected_episodes": _expected_repetitions(cell),
                "completed_episodes": completed,
                "failed_episodes": failed,
                "sealed": seal.get("status") == "completed",
            }
        )
        rich_rounds = _rich_rows(study_id, cell, "round_trajectory.jsonl", "round_trajectory.jsonl")
        round_rows.extend(rich_rounds or _compact_round_rows(study_id, cell, frame))
        micro_rows.extend(
            _rich_rows(study_id, cell, "micro_slot_trajectory.jsonl", "micro_slot_trajectory.jsonl")
        )

    tables = {
        "cells": _frame(cell_rows, TABLE_SCHEMAS["cells"]),
        "episodes": _frame(episode_rows, TABLE_SCHEMAS["episodes"]),
        "rounds": _frame(round_rows, TABLE_SCHEMAS["rounds"]),
        "micro_slots": _frame(micro_rows, TABLE_SCHEMAS["micro_slots"]),
    }
    return tables, {"scientific_frames": frames}


def _frame(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(columns))
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = None
    leading = list(columns)
    return frame[leading + sorted(set(frame.columns) - set(leading))]


def parquet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Serialize heterogeneous nested objects without losing their content."""

    result = frame.copy()
    for column in result.columns:
        values = result[column]
        if values.map(lambda value: isinstance(value, (Mapping, list, tuple, set))).any():
            result[column] = values.map(
                lambda value: json.dumps(value, sort_keys=True, default=str)
                if isinstance(value, (Mapping, list, tuple, set))
                else value
            )
    return result


__all__ = ["TABLE_SCHEMAS", "build_canonical_tables", "parquet_safe"]
