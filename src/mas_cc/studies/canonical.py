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
        "study_id",
        "source_extension_index",
        "source_submission_attempt",
        "source_config_index",
        "source_run_id",
        "source_run_path",
        "cell_id",
        "source_cell_id",
        "config_hash",
        "resolved_config_hash",
        "recorded_resolved_config_hash",
        "task_id",
        "expected_episodes",
        "completed_episodes",
        "failed_episodes",
        "sealed",
    ),
    "episodes": (
        "study_id",
        "source_extension_index",
        "source_submission_attempt",
        "source_config_index",
        "source_run_id",
        "source_run_path",
        "cell_id",
        "source_cell_id",
        "episode_id",
        "source_episode_id",
        "cell_key",
        "repetition_index",
        "episode_key",
        "episode_seed",
        "status",
        "interaction_count",
        "usage_requests",
        "usage_input_tokens",
        "usage_output_tokens",
        "started_at",
        "finished_at",
        "termination_reason",
        "scientific_schema_version",
    ),
    "rounds": (
        "study_id",
        "source_extension_index",
        "source_submission_attempt",
        "source_config_index",
        "source_run_id",
        "source_run_path",
        "cell_id",
        "source_cell_id",
        "episode_id",
        "source_episode_id",
        "cell_key",
        "repetition_index",
        "episode_key",
        "round_index",
        "record_source",
    ),
    "micro_slots": (
        "study_id",
        "source_extension_index",
        "source_submission_attempt",
        "source_config_index",
        "source_run_id",
        "source_run_path",
        "cell_id",
        "source_cell_id",
        "episode_id",
        "source_episode_id",
        "cell_key",
        "repetition_index",
        "episode_key",
        "round_index",
        "micro_slot_index",
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
        ("game.options.social_group_size", "social_group_size"),
        ("control.options.sensor_sample_size", "sensor_sample_size"),
        ("control.options.intervention_budget", "intervention_budget"),
        ("control.options.beta", "beta"),
        ("control.options.threshold", "threshold"),
        (
            "game.options.receiver_epistemic_disposition",
            "receiver_epistemic_disposition",
        ),
        (
            "control.options.controller_evidence_strategy",
            "controller_evidence_strategy",
        ),
        ("control.options.message_mode", "message_mode"),
        ("control.options.target", "controller_target_semantics"),
        ("experiment.metadata.controller_semantics", "controller_semantics"),
        ("experiment.metadata.target_semantics", "target_semantics"),
    ):
        value = _nested(cell.resolved_config, dotted)
        if value is not None and label not in result:
            # Target semantics deliberately permits both an integer option
            # index (adversarial control) and the symbolic value ``correct``.
            # Store the coordinate as text so matched truth/false studies have
            # one stable Parquet type without changing either meaning.
            if label == "controller_target_semantics":
                value = str(value)
            result[label] = value
    if "receiver_epistemic_disposition" not in result:
        legacy = _nested(cell.resolved_config, "game.options.social_distrust")
        if isinstance(legacy, bool):
            result["receiver_epistemic_disposition"] = "vigilant" if legacy else "naive"
        elif (
            _nested(cell.resolved_config, "game.type")
            == "relational_imitation_round_feedback"
        ):
            result["receiver_epistemic_disposition"] = "vigilant"
    if result.get("receiver_epistemic_disposition") and result.get(
        "controller_evidence_strategy"
    ):
        result["epistemic_condition"] = (
            f"{result['receiver_epistemic_disposition']}_"
            f"{result['controller_evidence_strategy']}"
        )
        result["derived_epistemic_condition"] = result["epistemic_condition"]
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
    return pd.concat(
        [pd.read_parquet(path, engine="pyarrow") for path in shards], ignore_index=True
    )


def _provenance(study_id: str, cell: DiscoveredCell) -> dict[str, Any]:
    return {
        "study_id": study_id,
        "source_extension_index": cell.run.entry.source_extension_index,
        "source_submission_attempt": cell.run.entry.source_submission_attempt,
        "source_config_index": cell.run.entry.array_index,
        "source_run_id": cell.run.run_id,
        "source_run_path": str(cell.run.path),
        "cell_id": cell.cell_key,
        "source_cell_id": cell.local_cell_id,
    }


def _episode_rows(
    study_id: str, cell: DiscoveredCell, frame: pd.DataFrame
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    provenance = _provenance(study_id, cell)
    if not frame.empty and "episode_id" in frame:
        for episode_id, group in frame.groupby("episode_id", sort=True, dropna=False):
            first = group.iloc[0]
            rows.append(
                {
                    **provenance,
                    "episode_id": str(episode_id),
                    "source_episode_id": str(episode_id),
                    "cell_key": cell.cell_key,
                    "repetition_index": _repetition_index(str(episode_id)),
                    "episode_key": _episode_key(cell.cell_key, _repetition_index(str(episode_id))),
                    "episode_seed": first.get("episode_seed"),
                    "status": str(first.get("status", "completed")),
                    "interaction_count": int(
                        first.get("interaction_count", len(group))
                    ),
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
                "source_episode_id": str(payload.get("episode_id", path.parent.name)),
                "cell_key": cell.cell_key,
                "repetition_index": _repetition_index(str(payload.get("episode_id", path.parent.name))),
                "episode_key": _episode_key(
                    cell.cell_key,
                    _repetition_index(str(payload.get("episode_id", path.parent.name))),
                ),
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


def _repetition_index(episode_id: str) -> int:
    import re

    match = re.search(r"-(\d+)$", episode_id)
    return int(match.group(1)) if match is not None else -1


def _episode_key(cell_key: str, repetition_index: int) -> str | None:
    if repetition_index < 0 or not cell_key or cell_key.startswith("config-"):
        return None
    from .identity import episode_key

    return episode_key(cell_key, repetition_index)


def _jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path} line {number}") from exc
        event = (
            payload.get("event", payload) if isinstance(payload, Mapping) else payload
        )
        if isinstance(event, Mapping):
            yield {**dict(payload), **dict(event)}


def _rich_rows(
    study_id: str, cell: DiscoveredCell, filename: str, source_label: str
) -> list[dict[str, Any]]:
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
                    "source_episode_id": episode_id,
                    "cell_key": cell.cell_key,
                    "repetition_index": _repetition_index(episode_id),
                    "episode_key": _episode_key(
                        cell.cell_key, _repetition_index(episode_id)
                    ),
                    # Full-profile event payloads can carry a game-local task
                    # identity here while the enclosing artifact directory
                    # carries the orchestrator's canonical episode identity.
                    # Selection reconciles the two against completed manifests.
                    "_storage_episode_id": str(path.parent.name),
                    "round_index": round_index,
                    "micro_slot_index": slot_index,
                    "record_source": source_label,
                }
            )
    return rows


def _compact_round_rows(
    study_id: str, cell: DiscoveredCell, frame: pd.DataFrame
) -> list[dict[str, Any]]:
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
                "source_episode_id": str(row.get("episode_id")),
                "cell_key": cell.cell_key,
                "repetition_index": _repetition_index(str(row.get("episode_id"))),
                "episode_key": _episode_key(
                    cell.cell_key, _repetition_index(str(row.get("episode_id")))
                ),
                "round_index": row.get("interaction_index"),
                "record_source": "scientific_events.parquet",
            }
        )
    return result


def _completed_unique_records(
    rows: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    *,
    coordinate_columns: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select the final canonical trajectory of every completed episode.

    Failed attempts can leave trajectory prefixes behind, and safe retry appends
    the replacement attempt to the same JSONL file.  Episode manifests/compact
    scientific rows are the completion authority.  Keeping the last occurrence
    of each physical coordinate therefore selects the completed retry while
    preventing failed prefixes and duplicate coordinates from entering any
    estimator.
    """

    completed = {
        str(row["episode_id"])
        for row in episodes
        if row.get("status") in {"completed", "skipped_resumed"}
    }
    eligible = []
    for original in rows:
        row = dict(original)
        event_episode_id = str(row.get("episode_id"))
        storage_episode_id = str(row.pop("_storage_episode_id", ""))
        if event_episode_id in completed:
            row["episode_id"] = event_episode_id
        elif storage_episode_id in completed:
            row["episode_id"] = storage_episode_id
        else:
            continue
        eligible.append(row)
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in eligible:
        key = (str(row.get("episode_id")),) + tuple(
            row.get(column) for column in coordinate_columns
        )
        latest[key] = row
    retained = list(latest.values())
    return retained, {
        "input_records": len(rows),
        "excluded_incomplete_records": len(rows) - len(eligible),
        "superseded_retry_records": len(eligible) - len(retained),
        "retained_records": len(retained),
    }


def build_canonical_tables(
    study_id: str, cells: tuple[DiscoveredCell, ...]
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    cell_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    micro_rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    record_selection = {
        "rounds": {
            "input_records": 0,
            "excluded_incomplete_records": 0,
            "superseded_retry_records": 0,
            "retained_records": 0,
        },
        "micro_slots": {
            "input_records": 0,
            "excluded_incomplete_records": 0,
            "superseded_retry_records": 0,
            "retained_records": 0,
        },
    }
    for cell in cells:
        frame = _scientific_frame(cell)
        frames[cell.cell_key] = frame
        episodes = _episode_rows(study_id, cell, frame)
        episode_rows.extend(episodes)
        seal_path = cell.path / "cell_complete.json"
        seal: Mapping[str, Any] = {}
        if seal_path.is_file():
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
        completed = sum(
            row["status"] in {"completed", "skipped_resumed"} for row in episodes
        )
        failed = sum(
            row["status"] in {"failed", "skipped_aborted", "aborted"}
            for row in episodes
        )
        coords = _coordinates(cell)
        cell_rows.append(
            {
                **_provenance(study_id, cell),
                "config_hash": cell.run.entry.config_hash,
                "resolved_config_hash": canonical_hash(cell.resolved_config),
                "recorded_resolved_config_hash": (
                    None
                    if frame.empty or "resolved_config_hash" not in frame
                    else json.dumps(
                        sorted(
                            {
                                str(value)
                                for value in frame["resolved_config_hash"].dropna()
                            }
                        )
                    )
                ),
                "task_id": coords.get("task_id"),
                **coords,
                "expected_episodes": _expected_repetitions(cell),
                "completed_episodes": completed,
                "failed_episodes": failed,
                "sealed": seal.get("status") == "completed",
            }
        )
        rich_rounds = _rich_rows(
            study_id, cell, "round_trajectory.jsonl", "round_trajectory.jsonl"
        )
        selected_rounds, round_selection = _completed_unique_records(
            rich_rounds or _compact_round_rows(study_id, cell, frame),
            episodes,
            coordinate_columns=("round_index",),
        )
        round_rows.extend(selected_rounds)
        for key, value in round_selection.items():
            record_selection["rounds"][key] += value
        # Micro-slot records live in different files per artifact profile:
        # `results_only` compaction writes a dedicated `micro_slot_trajectory`,
        # while the `full` profile leaves them interleaved in the generic
        # `trajectory.jsonl`.  Both are harvested, because `h` and `gamma` -
        # and therefore `eta_th` - are read off these rows, and a profile that
        # merely files them elsewhere must not make those quantities vanish.
        # The generic file is filtered to genuine slot events by
        # `within_round_index`, the same marker the game analyzers use.
        discovered_micro_rows = _rich_rows(
            study_id, cell, "micro_slot_trajectory.jsonl", "micro_slot_trajectory.jsonl"
        ) or [
            row
            for row in _rich_rows(
                study_id, cell, "trajectory.jsonl", "trajectory.jsonl"
            )
            if row.get("within_round_index") is not None
        ]
        selected_micro_rows, micro_selection = _completed_unique_records(
            discovered_micro_rows,
            episodes,
            coordinate_columns=("round_index", "micro_slot_index"),
        )
        micro_rows.extend(selected_micro_rows)
        for key, value in micro_selection.items():
            record_selection["micro_slots"][key] += value

    tables = {
        "cells": _frame(cell_rows, TABLE_SCHEMAS["cells"]),
        "episodes": _frame(episode_rows, TABLE_SCHEMAS["episodes"]),
        "rounds": _frame(round_rows, TABLE_SCHEMAS["rounds"]),
        "micro_slots": _frame(micro_rows, TABLE_SCHEMAS["micro_slots"]),
    }
    return tables, {
        "scientific_frames": frames,
        "record_selection": record_selection,
    }


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
        if values.map(
            lambda value: isinstance(value, (Mapping, list, tuple, set))
        ).any():
            result[column] = values.map(
                lambda value: json.dumps(value, sort_keys=True, default=str)
                if isinstance(value, (Mapping, list, tuple, set))
                else value
            )
    return result


__all__ = ["TABLE_SCHEMAS", "build_canonical_tables", "parquet_safe"]
