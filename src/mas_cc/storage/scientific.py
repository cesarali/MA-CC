"""Atomic compact scientific episode artifacts.

The compact schema is deliberately narrower than the recorder's trajectory:
it retains categorical transitions, derived behavioral diagnostics, and the
already-computed metric values needed by aggregation.  Prompt text, messages,
responses, reasoning, and arbitrary event dictionaries never enter it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .checkpoints import canonical_hash

SCIENTIFIC_SCHEMA_VERSION = 1

IDENTITY_COLUMNS = (
    "schema_version",
    "run_id",
    "cell_id",
    "episode_id",
    "episode_seed",
    "interaction_index",
    "resolved_config_hash",
    "prompt_definition_hashes_hash",
    "pricing_snapshot_hash",
    "game_type",
    "dynamics_mode",
    "control_mechanism",
    "task_id",
)

SCIENTIFIC_COLUMNS = (
    "possible_answers",
    "correct_answer",
    "analysis_target",
    "N_t",
    "N_t1",
    "Y_t",
    "U_t",
    "Z_t",
    "Z_t1",
    "Mtruth_t",
    "Mtruth_t1",
    "Morder_t",
    "Morder_t1",
    "Xf_t",
    "Xf_t1",
)

# These are the non-raw fields consumed by the existing HiddenBench summaries
# and controller diagnostics.  They are scalars or small categorical values.
DIAGNOSTIC_COLUMNS = (
    "controller_policy",
    "sensor_sample_size",
    "controller_threshold",
    "controller_beta",
    "controller_advocacy_probability",
    "delta_m_ctrl",
    "delta_m_truth",
    "delta_m_order",
    "delta_H_vote",
    "focal_changed",
    "focal_adopted_target",
    "focal_left_target",
    "u_advocate",
    "sensor_target_share",
    "population_target_share",
    "sensor_target_error",
    "sensor_target_abs_error",
)

TERMINAL_COLUMNS = (
    "status",
    "interaction_count",
    "termination_reason",
    "error_type",
    "error_summary",
    "started_at",
    "finished_at",
    "usage_requests",
    "usage_input_tokens",
    "usage_output_tokens",
)

INTERNAL_COLUMNS = (
    "population_metrics_json",
    "option_metrics_json",
    "final_metrics_json",
)

ALL_COLUMNS = (*IDENTITY_COLUMNS, *SCIENTIFIC_COLUMNS, *DIAGNOSTIC_COLUMNS,
               *TERMINAL_COLUMNS, *INTERNAL_COLUMNS)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _clean(value: Any) -> Any:
    """Turn pandas/NumPy missing values into ordinary ``None`` values."""

    if value is None:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return value
    if isinstance(missing, bool) and missing:
        return None
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_definition_hash(config: Any) -> str:
    """Hash every built-in prompt definition this resolved run can compile."""

    from mas_cc.games.registry import (
        create_default_prompt_registry,
        register_game_prompt_factories,
    )

    registry = register_game_prompt_factories(create_default_prompt_registry())
    families = [config.prompt.prompt_family]
    if config.game.type in {
        "hidden_bench_imitation",
        "hidden_bench_imitation_round_feedback",
    }:
        from mas_cc.games.hidden_bench.imitation.runtime import PROMPT_FAMILIES

        families = list(PROMPT_FAMILIES)
    hashes: dict[str, str] = {}
    for family in families:
        try:
            prompt = registry.get(family, config.prompt.prompt_version)
        except ValueError:
            continue
        hashes[f"{family}@{config.prompt.prompt_version}"] = prompt.definition_hash
    if not hashes:
        hashes[
            f"{config.prompt.prompt_family}@{config.prompt.prompt_version}"
        ] = canonical_hash(config.prompt.to_dict())
    return canonical_hash(hashes)


def _fsync_directory(directory: Path) -> None:
    """Persist a rename when the backing filesystem supports directory fsync."""

    try:
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Some network filesystems do not permit directory fsync. The file is
        # still flushed and published with an atomic rename on that filesystem.
        pass


@dataclass(frozen=True, slots=True)
class ScientificIdentity:
    run_id: str
    cell_id: str
    episode_id: str
    episode_seed: int
    resolved_config_hash: str
    prompt_definition_hashes_hash: str
    pricing_snapshot_hash: str
    game_type: str
    dynamics_mode: str | None = None
    control_mechanism: str | None = None
    task_id: str | None = None

    def row(self, interaction_index: int) -> dict[str, Any]:
        return {
            "schema_version": SCIENTIFIC_SCHEMA_VERSION,
            "run_id": self.run_id,
            "cell_id": self.cell_id,
            "episode_id": self.episode_id,
            "episode_seed": self.episode_seed,
            "interaction_index": int(interaction_index),
            "resolved_config_hash": self.resolved_config_hash,
            "prompt_definition_hashes_hash": self.prompt_definition_hashes_hash,
            "pricing_snapshot_hash": self.pricing_snapshot_hash,
            "game_type": self.game_type,
            "dynamics_mode": self.dynamics_mode,
            "control_mechanism": self.control_mechanism,
            "task_id": self.task_id,
        }


def empty_compact_row(identity: ScientificIdentity, interaction_index: int) -> dict[str, Any]:
    row = {column: None for column in ALL_COLUMNS}
    row.update(identity.row(interaction_index))
    return row


def compact_imitation_event(
    event: Mapping[str, Any], identity: ScientificIdentity
) -> dict[str, Any]:
    """Normalize one rich imitation event into the versioned compact schema."""

    from mas_cc.games.hidden_bench.imitation.analysis import adapt_event

    adapted = adapt_event(
        event,
        episode_id=identity.episode_id,
        cell_id=identity.cell_id,
    )
    enriched = adapted.event
    row = empty_compact_row(identity, adapted.interaction_index)
    row.update(
        {
            "dynamics_mode": enriched.get("dynamics_mode") or identity.dynamics_mode,
            "task_id": str(enriched.get("task_id") or identity.task_id or ""),
            "possible_answers": list(adapted.options),
            "correct_answer": str(enriched["correct_answer"]),
            "analysis_target": adapted.target,
            "N_t": list(adapted.N_t),
            "N_t1": list(adapted.N_t1),
            "Y_t": None if adapted.Y_t is None else list(adapted.Y_t),
            "U_t": adapted.U_t,
            "Z_t": adapted.Z_t,
            "Z_t1": adapted.Z_t1,
            "Mtruth_t": adapted.Mtruth_t,
            "Mtruth_t1": adapted.Mtruth_t1,
            "Morder_t": adapted.Morder_t,
            "Morder_t1": adapted.Morder_t1,
            "Xf_t": adapted.Xf_t,
            "Xf_t1": adapted.Xf_t1,
        }
    )
    for column in DIAGNOSTIC_COLUMNS:
        row[column] = enriched.get(column)
    return row


def _write_parquet_atomic(rows: Sequence[Mapping[str, Any]], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    frame = pd.DataFrame([{column: row.get(column) for column in ALL_COLUMNS} for row in rows])
    # Explicitly provide columns for deterministic empty/all-null fields.
    frame = frame.reindex(columns=ALL_COLUMNS)
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    # Never replace a previously valid artifact with bytes that pyarrow cannot
    # read back. Identity-specific validation is performed by the caller.
    verified = pd.read_parquet(temporary, engine="pyarrow")
    if tuple(verified.columns) != ALL_COLUMNS or len(verified) != len(frame):
        raise ValueError(f"temporary scientific table failed validation: {temporary}")
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)
    return destination


def _identity_mismatch(frame: pd.DataFrame, identity: ScientificIdentity) -> str | None:
    expected = identity.row(1)
    for column in (
        "schema_version",
        "run_id",
        "cell_id",
        "episode_id",
        "episode_seed",
        "resolved_config_hash",
        "prompt_definition_hashes_hash",
        "pricing_snapshot_hash",
        "game_type",
    ):
        values = {_clean(value) for value in frame[column].tolist()}
        if values != {expected[column]}:
            return f"{column} is {sorted(map(str, values))}, expected {expected[column]!r}"
    return None


def validate_episode_artifact(
    path: str | Path,
    identity: ScientificIdentity,
    *,
    expected_interactions: int | None = None,
) -> pd.DataFrame:
    """Read and validate one shard or one episode's rows in a merged file."""

    source = Path(path)
    try:
        frame = pd.read_parquet(source, engine="pyarrow")
    except Exception as exc:
        raise ValueError(f"cannot read scientific artifact {source}: {type(exc).__name__}") from exc
    missing = [column for column in ALL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"scientific artifact {source} is missing columns: {', '.join(missing)}")
    return validate_episode_frame(
        frame,
        identity,
        expected_interactions=expected_interactions,
        source=source,
    )


def validate_episode_frame(
    frame: pd.DataFrame,
    identity: ScientificIdentity,
    *,
    expected_interactions: int | None = None,
    source: str | Path = "in-memory scientific table",
) -> pd.DataFrame:
    """Validate one episode from an already-read compact table."""

    missing = [column for column in ALL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"scientific artifact {source} is missing columns: {', '.join(missing)}")
    frame = frame[frame["episode_id"].astype(str) == identity.episode_id].copy()
    if frame.empty:
        raise ValueError(f"scientific artifact {source} has no rows for {identity.episode_id}")
    mismatch = _identity_mismatch(frame, identity)
    if mismatch:
        raise ValueError(f"incompatible episode checkpoint {identity.episode_id}: {mismatch}")
    statuses = {str(value) for value in frame["status"].tolist()}
    if statuses != {"completed"}:
        raise ValueError(
            f"episode checkpoint {identity.episode_id} is not completed: {sorted(statuses)}"
        )
    indices = sorted(int(value) for value in frame["interaction_index"].tolist())
    if indices != list(range(1, len(indices) + 1)):
        raise ValueError(f"episode checkpoint {identity.episode_id} has non-contiguous interactions")
    recorded_counts = {int(value) for value in frame["interaction_count"].tolist()}
    if recorded_counts != {len(indices)}:
        raise ValueError(f"episode checkpoint {identity.episode_id} row count does not match terminal metadata")
    if expected_interactions is not None and len(indices) != expected_interactions:
        raise ValueError(
            f"episode checkpoint {identity.episode_id} has {len(indices)} interactions; "
            f"expected {expected_interactions}"
        )
    return frame.sort_values("interaction_index").reset_index(drop=True)


def validate_cell_artifact(cell_dir: str | Path) -> pd.DataFrame:
    """Validate a sealed cell without needing its original in-memory identities."""

    directory = Path(cell_dir)
    table = directory / "scientific_events.parquet"
    seal_path = directory / "cell_complete.json"
    if not table.is_file() or not seal_path.is_file():
        raise ValueError(f"cell is not sealed with compact scientific data: {directory}")
    try:
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read cell completion seal {seal_path}") from exc
    if seal.get("status") != "completed":
        raise ValueError(f"cell completion seal is not completed: {seal_path}")
    expected_hash = seal.get("scientific_events_sha256")
    if not isinstance(expected_hash, str) or file_sha256(table) != expected_hash:
        raise ValueError(f"sealed scientific table hash mismatch: {table}")
    try:
        frame = pd.read_parquet(table, engine="pyarrow")
    except Exception as exc:
        raise ValueError(f"cannot read sealed scientific table {table}") from exc
    missing = [column for column in ALL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"sealed scientific table is missing columns: {', '.join(missing)}")
    if frame.empty or set(frame["schema_version"].astype(int)) != {
        SCIENTIFIC_SCHEMA_VERSION
    }:
        raise ValueError(f"sealed scientific table has an incompatible schema: {table}")
    if set(frame["status"].astype(str)) != {"completed"}:
        raise ValueError(f"sealed scientific table contains non-completed rows: {table}")
    episode_ids = set(frame["episode_id"].astype(str))
    sealed_episode_ids = seal.get("episode_ids", ())
    if not isinstance(sealed_episode_ids, (list, tuple)):
        raise ValueError(f"cell completion seal has invalid episode IDs: {seal_path}")
    if episode_ids != {str(item) for item in sealed_episode_ids}:
        raise ValueError(f"sealed scientific table episode IDs do not match its seal: {table}")
    try:
        sealed_row_count = int(seal.get("row_count", -1))
        sealed_episode_count = int(seal.get("episode_count", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cell completion seal has invalid counts: {seal_path}") from exc
    if len(frame) != sealed_row_count or len(episode_ids) != sealed_episode_count:
        raise ValueError(f"sealed scientific table row count does not match its seal: {table}")
    if seal.get("scientific_schema_version") != SCIENTIFIC_SCHEMA_VERSION:
        raise ValueError(f"cell completion seal has an incompatible scientific schema: {seal_path}")
    raw_counts = seal.get("episode_row_counts", {})
    if not isinstance(raw_counts, Mapping):
        raise ValueError(f"cell completion seal has invalid episode counts: {seal_path}")
    sealed_counts = {
        str(key): int(value)
        for key, value in raw_counts.items()
    }
    if set(sealed_counts) != episode_ids:
        raise ValueError(f"sealed scientific table episode counts are incomplete: {table}")
    for episode_id, group in frame.groupby("episode_id", sort=False):
        label = str(episode_id)
        indices = sorted(int(value) for value in group["interaction_index"].tolist())
        if indices != list(range(1, len(indices) + 1)):
            raise ValueError(f"sealed episode {label} has non-contiguous interactions")
        recorded = {int(value) for value in group["interaction_count"].tolist()}
        if recorded != {len(indices)} or sealed_counts[label] != len(indices):
            raise ValueError(f"sealed episode {label} has inconsistent row counts")
        for column in IDENTITY_COLUMNS:
            if column == "interaction_index":
                continue
            values = {_clean(value) for value in group[column].tolist()}
            if len(values) != 1:
                raise ValueError(
                    f"sealed episode {label} has inconsistent identity column {column}"
                )
    return frame


def write_completed_episode(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    identity: ScientificIdentity,
    *,
    termination_reason: str | None,
    started_at: str,
    usage: Mapping[str, int] | None = None,
) -> Path:
    """Write, read back, validate, and publish one completed episode shard."""

    if not rows:
        raise ValueError("a completed scientific episode must contain at least one interaction")
    finished_at = _now()
    terminal_rows: list[dict[str, Any]] = []
    for source in rows:
        row = {column: source.get(column) for column in ALL_COLUMNS}
        row.update(
            {
                "status": "completed",
                "interaction_count": len(rows),
                "termination_reason": termination_reason,
                "started_at": started_at,
                "finished_at": finished_at,
                "usage_requests": int((usage or {}).get("requests", 0)),
                "usage_input_tokens": int((usage or {}).get("input_tokens", 0)),
                "usage_output_tokens": int((usage or {}).get("output_tokens", 0)),
            }
        )
        terminal_rows.append(row)
    destination = Path(path)
    # The temporary file is intentionally visible until validation succeeds;
    # a crash can never make it look like a completed checkpoint.
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    frame = pd.DataFrame(terminal_rows).reindex(columns=ALL_COLUMNS)
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    validate_episode_artifact(temporary, identity, expected_interactions=len(rows))
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)
    validate_episode_artifact(destination, identity, expected_interactions=len(rows))
    return destination


def episode_shard_path(cell_dir: str | Path, episode_id: str) -> Path:
    if not episode_id or Path(episode_id).name != episode_id:
        raise ValueError("episode_id must be one path component")
    return Path(cell_dir) / ".resume" / episode_id / "scientific_events.parquet"


def discover_episode_artifact(
    cell_dir: str | Path, identity: ScientificIdentity
) -> Path | None:
    """Find a durable shard/final table; temporary files are ignored."""

    directory = Path(cell_dir)
    shard = episode_shard_path(directory, identity.episode_id)
    if shard.is_file():
        return shard
    final = directory / "scientific_events.parquet"
    if final.is_file():
        validate_cell_artifact(directory)
        try:
            validate_episode_artifact(final, identity)
        except ValueError as exc:
            # A final table belonging to this cell but lacking the requested
            # episode is not a checkpoint for it.  Identity conflicts are
            # raised so two configs are never silently combined.
            if "has no rows for" in str(exc):
                return None
            raise
        return final
    return None


def merge_episode_artifacts(
    cell_dir: str | Path,
    identities: Sequence[ScientificIdentity],
    *,
    remove_shards: bool = True,
) -> dict[str, Any]:
    """Transactionally seal a cell's completed shards into one compact table."""

    directory = Path(cell_dir)
    destination = directory / "scientific_events.parquet"
    frames: list[pd.DataFrame] = []
    row_counts: dict[str, int] = {}
    for identity in identities:
        source = discover_episode_artifact(directory, identity)
        if source is None:
            raise ValueError(f"no completed scientific shard for {identity.episode_id}")
        frame = validate_episode_artifact(source, identity)
        frames.append(frame)
        row_counts[identity.episode_id] = len(frame)
    if not frames:
        raise ValueError("cannot seal a cell without completed episode shards")
    merged = pd.concat(frames, ignore_index=True).sort_values(
        ["episode_id", "interaction_index"]
    )
    rows = merged.to_dict(orient="records")
    _write_parquet_atomic(rows, destination)
    verified = pd.read_parquet(destination, engine="pyarrow")
    expected_ids = {identity.episode_id for identity in identities}
    if set(verified["episode_id"].astype(str)) != expected_ids or len(verified) != sum(row_counts.values()):
        raise ValueError("sealed scientific table failed episode/row-count validation")
    for identity in identities:
        validate_episode_artifact(destination, identity, expected_interactions=row_counts[identity.episode_id])
    summary = {
        "schema_version": 1,
        "status": "completed",
        "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION,
        "episode_count": len(expected_ids),
        "episode_ids": sorted(expected_ids),
        "row_count": len(verified),
        "episode_row_counts": dict(sorted(row_counts.items())),
        "scientific_events_sha256": file_sha256(destination),
        "sealed_at": _now(),
    }
    manifest = directory / "cell_complete.json"
    temporary_manifest = manifest.with_name(manifest.name + ".tmp")
    temporary_manifest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary_manifest.open("rb") as stream:
        os.fsync(stream.fileno())
    # A malformed seal must never authorize cleanup of the resumable shards.
    json.loads(temporary_manifest.read_text(encoding="utf-8"))
    os.replace(temporary_manifest, manifest)
    _fsync_directory(manifest.parent)
    # Only a durable and re-readable seal authorizes cleanup.
    if remove_shards:
        resume_dir = directory / ".resume"
        if resume_dir.is_dir():
            shutil.rmtree(resume_dir)
    return summary


def read_scientific_tables(root: str | Path, *, include_resume: bool = True) -> pd.DataFrame:
    """Read compact final files (or partial shards) without duplicating rows."""

    source = Path(root)
    if source.is_file():
        return pd.read_parquet(source, engine="pyarrow")
    direct = source / "scientific_events.parquet"
    if direct.is_file():
        return pd.read_parquet(direct, engine="pyarrow")
    cell_files = sorted((source / "cells").glob("*/scientific_events.parquet"))
    paths = cell_files
    if not paths and include_resume:
        paths = sorted(source.rglob(".resume/*/scientific_events.parquet"))
    if not paths:
        raise FileNotFoundError(f"no scientific_events.parquet files under {source}")
    return pd.concat(
        [pd.read_parquet(path, engine="pyarrow") for path in paths], ignore_index=True
    )


def merge_cell_scientific_tables(root: str | Path) -> Path | None:
    """Atomically publish a run-level table from already sealed grid cells."""

    directory = Path(root)
    paths: list[Path] = []
    cells_root = directory / "cells"
    if cells_root.is_dir():
        for cell in sorted(path for path in cells_root.iterdir() if path.is_dir()):
            final = cell / "scientific_events.parquet"
            if final.is_file():
                validate_cell_artifact(cell)
                paths.append(final)
            else:
                paths.extend(sorted(cell.glob(".resume/*/scientific_events.parquet")))
    if not paths:
        direct = directory / "scientific_events.parquet"
        return direct if direct.is_file() else None
    frames = [pd.read_parquet(path, engine="pyarrow") for path in paths]
    merged = pd.concat(frames, ignore_index=True).sort_values(
        ["cell_id", "episode_id", "interaction_index"]
    )
    key_columns = ["cell_id", "episode_id", "interaction_index"]
    if merged.duplicated(key_columns).any():
        raise ValueError("cell scientific tables contain duplicate interaction identities")
    destination = directory / "scientific_events.parquet"
    _write_parquet_atomic(merged.to_dict(orient="records"), destination)
    verified = pd.read_parquet(destination, engine="pyarrow")
    if len(verified) != len(merged):
        raise ValueError("run-level scientific merge failed row-count validation")
    return destination


def compact_row_to_imitation_event(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Reconstruct the normalized event mapping expected by ``adapt_event``."""

    options = tuple(str(item) for item in row["possible_answers"])
    before_counts = tuple(int(item) for item in row["N_t"])
    after_counts = tuple(int(item) for item in row["N_t1"])
    before_votes = [option for option, count in zip(options, before_counts) for _ in range(count)]
    after_votes = [option for option, count in zip(options, after_counts) for _ in range(count)]
    sensor_raw = _clean(row.get("Y_t"))
    sensor = None if sensor_raw is None else tuple(int(item) for item in sensor_raw)
    event: dict[str, Any] = {
        "episode_id": str(row["episode_id"]),
        "interaction_index": int(row["interaction_index"]),
        "seed": int(row["episode_seed"]),
        "task_id": _clean(row.get("task_id")),
        "N": sum(before_counts),
        "possible_answers": list(options),
        "correct_answer": str(row["correct_answer"]),
        "analysis_target": str(row["analysis_target"]),
        "population_state_before": before_votes,
        "population_state_after": after_votes,
        "occupation_counts_before": dict(zip(options, before_counts)),
        "occupation_counts_after": dict(zip(options, after_counts)),
        "focal_opinion_before": str(row["Xf_t"]),
        "focal_opinion_after": str(row["Xf_t1"]),
        "controller_action": _clean(row.get("U_t")),
        "_compact_control_mechanism": _clean(row.get("control_mechanism")),
        "controller_policy": _clean(row.get("controller_policy")),
        "dynamics_mode": _clean(row.get("dynamics_mode")),
        "sensor_sample_size": _clean(row.get("sensor_sample_size")),
        "sensor_count_vector": (
            {} if sensor is None else dict(zip(options, sensor))
        ),
    }
    for column in DIAGNOSTIC_COLUMNS:
        if column not in event:
            event[column] = _clean(row.get(column))
    return event


def iter_compact_imitation_events(root: str | Path) -> Iterable[tuple[Mapping[str, Any], str, str]]:
    frame = read_scientific_tables(root)
    for row in frame.sort_values(["cell_id", "episode_id", "interaction_index"]).to_dict(
        orient="records"
    ):
        if _clean(row.get("possible_answers")) is None:
            continue
        yield compact_row_to_imitation_event(row), str(row["episode_id"]), str(row["cell_id"])


__all__ = [
    "ALL_COLUMNS",
    "SCIENTIFIC_SCHEMA_VERSION",
    "ScientificIdentity",
    "compact_imitation_event",
    "compact_row_to_imitation_event",
    "discover_episode_artifact",
    "empty_compact_row",
    "episode_shard_path",
    "file_sha256",
    "iter_compact_imitation_events",
    "merge_cell_scientific_tables",
    "merge_episode_artifacts",
    "prompt_definition_hash",
    "read_scientific_tables",
    "validate_episode_artifact",
    "validate_cell_artifact",
    "validate_episode_frame",
    "write_completed_episode",
]
