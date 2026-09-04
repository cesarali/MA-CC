"""Plan and submit incremental additions to a standardized study lineage."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import re
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.core.random import Seed
from mas_cc.storage import canonical_hash, file_sha256, validate_cell_artifact

from .execution import ExecutionEntry, plan_cell_execution, write_execution_manifest
from .execution import read_execution_manifest
from .identity import (
    PROTOCOL_FINGERPRINT_VERSION,
    SEED_CONTRACT_LEGACY_GRID_V1,
    SEED_CONTRACT_STABLE_CELL_V1,
    episode_key,
    protocol_fingerprint,
    scientific_cell_key,
)
from .manifest import StudySpec, discover_study
from .submission import (
    SubmissionEntry,
    build_submission_entries,
    read_submission_manifest,
)


LINEAGE_SCHEMA_VERSION = 1
TARGET_MANIFEST_SCHEMA_VERSION = 1
EPISODE_PLAN_COLUMNS = (
    "study_lineage_id",
    "extension_index",
    "submission_attempt",
    "protocol_fingerprint",
    "cell_key",
    "source_config",
    "resolved_coordinates",
    "repetition_index",
    "episode_seed",
    "episode_key",
    "expected_output_dir",
    "status",
)


class CompatibilityStatus(str, Enum):
    COMPLETE_REUSABLE = "COMPLETE_REUSABLE"
    NEEDS_ADDITIONAL_EPISODES = "NEEDS_ADDITIONAL_EPISODES"
    NEW_CELL = "NEW_CELL"
    INCOMPATIBLE = "INCOMPATIBLE"
    CONFLICTED = "CONFLICTED"


@dataclass(frozen=True, slots=True)
class TargetCell:
    config_index: int
    config_path: str
    source_cell_index: int
    source_cell_id: str
    protocol_fingerprint: str
    cell_key: str
    coordinates: Mapping[str, Any]
    repetitions: int
    base_seed: int
    common_random_numbers: bool


@dataclass(frozen=True, slots=True)
class PlannedEpisode:
    study_lineage_id: str
    extension_index: int
    submission_attempt: int
    protocol_fingerprint: str
    cell_key: str
    source_config: str
    resolved_coordinates: Mapping[str, Any]
    repetition_index: int
    episode_seed: int
    episode_key: str
    expected_output_dir: str
    status: str = "PLANNED"


@dataclass(frozen=True, slots=True)
class ExtensionPlan:
    study_dir: Path
    config_dir: Path
    study_lineage_id: str
    extension_index: int
    submission_attempt: int
    target_cells: tuple[TargetCell, ...]
    episodes: tuple[PlannedEpisode, ...]
    classifications: Mapping[str, str]
    retained_episode_count: int
    conflicts: tuple[str, ...]
    incompatible: tuple[str, ...]
    analysis_changed: bool

    @property
    def target_episode_count(self) -> int:
        return sum(cell.repetitions for cell in self.target_cells)

    @property
    def target_cell_count(self) -> int:
        return len(self.target_cells)

    @property
    def missing_episode_count(self) -> int:
        return len(self.episodes)

    def report(self) -> dict[str, Any]:
        values = list(self.classifications.values())
        return {
            "schema_version": 1,
            "study_lineage_id": self.study_lineage_id,
            "extension_index": self.extension_index,
            "submission_attempt": self.submission_attempt,
            "target_cells": self.target_cell_count,
            "target_episodes": self.target_episode_count,
            "reused_episodes": self.retained_episode_count,
            "partially_reused_cells": values.count(
                CompatibilityStatus.NEEDS_ADDITIONAL_EPISODES.value
            ),
            "complete_reusable_cells": values.count(
                CompatibilityStatus.COMPLETE_REUSABLE.value
            ),
            "new_cells": values.count(CompatibilityStatus.NEW_CELL.value),
            "missing_episodes_to_execute": self.missing_episode_count,
            "incompatible_records": len(self.incompatible),
            "conflicted_records": len(self.conflicts),
            "analysis_changed": self.analysis_changed,
            "classifications": dict(self.classifications),
            "conflicts": list(self.conflicts),
            "incompatible": list(self.incompatible),
        }


@dataclass(frozen=True, slots=True)
class ExtensionResult:
    plan: ExtensionPlan
    dry_run: bool
    extension_dir: Path | None
    job_id: str | None
    command: tuple[str, ...]
    execution_plan: Mapping[str, Any] | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + "-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return path


def _cells(spec: StudySpec) -> tuple[TargetCell, ...]:
    result: list[TargetCell] = []
    for config_index, path in enumerate(spec.configs):
        source = load_run_config_or_grid(path)
        if isinstance(source, GridSpec):
            swept_paths = tuple(axis.path for axis in source.axes)
            source_cells = source.cells
        else:
            swept_paths = ()
            source_cells = (
                type("StandaloneCell", (), {
                    "index": 0,
                    "cell_id": "run",
                    "overrides": {},
                    "config": source,
                })(),
            )
        for cell in source_cells:
            resolved = cell.config.to_dict()
            fingerprint = protocol_fingerprint(resolved, swept_paths=swept_paths)
            key = scientific_cell_key(fingerprint, cell.overrides)
            result.append(
                TargetCell(
                    config_index=config_index,
                    config_path=str(path),
                    source_cell_index=int(cell.index),
                    source_cell_id=str(cell.cell_id),
                    protocol_fingerprint=fingerprint,
                    cell_key=key,
                    coordinates=dict(cell.overrides),
                    repetitions=int(cell.config.execution.repetitions),
                    base_seed=int(cell.config.execution.seed),
                    common_random_numbers=bool(
                        cell.config.experiment.metadata.get(
                            "common_random_numbers_across_grid", False
                        )
                    ),
                )
            )
    keys = [cell.cell_key for cell in result]
    if len(set(keys)) != len(keys):
        raise ValueError("target design contains duplicate scientific cell keys")
    return tuple(result)


def _target_manifest(
    spec: StudySpec,
    lineage_id: str,
    extension_index: int,
    cells: Sequence[TargetCell],
) -> dict[str, Any]:
    analysis_hash = (
        file_sha256(spec.analysis_recipe) if spec.analysis_recipe is not None else None
    )
    return {
        "schema_version": TARGET_MANIFEST_SCHEMA_VERSION,
        "study_lineage_id": lineage_id,
        "extension_index": extension_index,
        "study_id": spec.name,
        "config_dir": str(spec.config_dir),
        "analysis_recipe": None if spec.analysis_recipe is None else str(spec.analysis_recipe),
        "analysis_recipe_hash": analysis_hash,
        "execution": dict(spec.execution),
        "protocol_fingerprint_version": PROTOCOL_FINGERPRINT_VERSION,
        "target_cell_count": len(cells),
        "target_episode_count": sum(cell.repetitions for cell in cells),
        "cells": [asdict(cell) for cell in cells],
    }


def _original_target_from_root(study_dir: Path, lineage_id: str) -> dict[str, Any]:
    manifest = _read_json(study_dir / "study_manifest.json")
    entries = read_submission_manifest(study_dir / "submission_manifest.csv")
    cells: list[TargetCell] = []
    for entry in entries:
        source = load_run_config_or_grid(entry.config_path)
        spec = StudySpec(
            str(manifest.get("study_id", study_dir.name)),
            Path(entry.config_path).parent,
            (Path(entry.config_path),),
        )
        for cell in _cells(spec):
            cells.append(replace(cell, config_index=entry.array_index))
    return {
        "schema_version": TARGET_MANIFEST_SCHEMA_VERSION,
        "study_lineage_id": lineage_id,
        "extension_index": 0,
        "study_id": str(manifest.get("study_id", study_dir.name)),
        "config_dir": manifest.get("config_dir"),
        "analysis_recipe": manifest.get("analysis_recipe"),
        "analysis_recipe_hash": (
            file_sha256(Path(str(manifest["analysis_recipe"])))
            if manifest.get("analysis_recipe")
            and Path(str(manifest["analysis_recipe"])).is_file()
            else None
        ),
        "execution": dict(manifest.get("execution", {})),
        "protocol_fingerprint_version": PROTOCOL_FINGERPRINT_VERSION,
        "target_cell_count": len(cells),
        "target_episode_count": sum(cell.repetitions for cell in cells),
        "cells": [asdict(cell) for cell in cells],
    }


def index_existing_study(study_dir: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Add a small lineage index to an existing standardized study."""

    root = Path(study_dir).expanduser().resolve()
    if not (root / "study_manifest.json").is_file() or not (
        root / "submission_manifest.csv"
    ).is_file():
        raise ValueError(f"not a submitted MA-CC study directory: {root}")
    lineage_path = root / "study_lineage.json"
    existing = _read_json(lineage_path) if lineage_path.is_file() else None
    lineage_id = str(existing["study_lineage_id"]) if existing else str(uuid.uuid4())
    target = _original_target_from_root(root, lineage_id)
    result = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "study_lineage_id": lineage_id,
        "study_id": target["study_id"],
        "created_at": existing.get("created_at", _now()) if existing else _now(),
        "latest_extension_index": int(existing.get("latest_extension_index", 0)) if existing else 0,
        "protocol_fingerprint_version": PROTOCOL_FINGERPRINT_VERSION,
        "seed_contract_version": SEED_CONTRACT_LEGACY_GRID_V1,
        "legacy_root_layout": True,
    }
    if not dry_run:
        extension = root / "extensions" / "extension-0000"
        extension.mkdir(parents=True, exist_ok=True)
        _atomic_json(lineage_path, result)
        _atomic_json(extension / "target_manifest.json", target)
        _atomic_json(
            extension / "migration.json",
            {
                "schema_version": 1,
                "status": "INDEXED",
                "indexed_at": _now(),
                "source_submission_manifest": str(root / "submission_manifest.csv"),
                "source_data_moved": False,
            },
        )
    return {**result, "target": target, "dry_run": dry_run}


def _repetition_index(episode_id: str) -> int:
    match = re.search(r"-(\d+)$", episode_id)
    if match is None:
        raise ValueError(f"cannot recover repetition index from legacy episode ID {episode_id!r}")
    return int(match.group(1))


def _retained_episodes(study_dir: Path, target_cells: Sequence[TargetCell]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Read valid sealed compact episodes from the original and extension roots."""

    by_local: dict[tuple[str, str], TargetCell] = {}
    for cell in target_cells:
        by_local[(Path(cell.config_path).name, cell.source_cell_id)] = cell
    retained: dict[str, dict[str, Any]] = {}
    conflicts: list[str] = []
    roots = [study_dir / "runs"]
    extensions = study_dir / "extensions"
    if extensions.is_dir():
        roots.extend(path / "runs" for path in sorted(extensions.glob("extension-*")))
    for root in roots:
        if not root.is_dir():
            continue
        for seal_path in sorted(root.rglob("cell_complete.json")):
            cell_dir = seal_path.parent
            try:
                frame = validate_cell_artifact(cell_dir)
            except ValueError:
                continue
            override_path = cell_dir / "overrides.json"
            raw_overrides = _read_json(override_path) if override_path.is_file() else {}
            local_id = str(raw_overrides.get("cell_id", cell_dir.name))
            config_name = ""
            for parent in cell_dir.parents:
                if parent.parent == root:
                    config_name = parent.name
                    break
            candidates = [
                cell for cell in target_cells
                if cell.source_cell_id == local_id and (
                    not config_name or Path(cell.config_path).stem == config_name
                ) and dict(cell.coordinates) == dict(
                    raw_overrides.get("overrides", raw_overrides)
                )
            ]
            if not candidates:
                # Reordered grids are matched below from resolved coordinates.
                overrides = raw_overrides.get("overrides", raw_overrides)
                candidates = [
                    cell for cell in target_cells
                    if dict(cell.coordinates) == dict(overrides)
                ]
            if len(candidates) != 1:
                continue
            target = candidates[0]
            source = load_run_config_or_grid(target.config_path)
            swept_paths = (
                tuple(axis.path for axis in source.axes)
                if isinstance(source, GridSpec)
                else ()
            )
            resolved_path = cell_dir / "resolved_config.yaml"
            if resolved_path.is_file():
                import yaml

                resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
                current_target_fingerprint = _prior_cell_current_fingerprint(
                    asdict(target)
                )
                if not isinstance(resolved, Mapping) or protocol_fingerprint(
                    resolved, swept_paths=swept_paths
                ) != (current_target_fingerprint or target.protocol_fingerprint):
                    continue
            for episode_id, group in frame.groupby("episode_id", sort=True):
                repetition = _repetition_index(str(episode_id))
                key = episode_key(target.cell_key, repetition)
                first = group.iloc[0]
                record = {
                    "episode_key": key,
                    "cell_key": target.cell_key,
                    "repetition_index": repetition,
                    "episode_seed": int(first["episode_seed"]),
                    # Round-trip through pandas' JSON encoder so ndarray/list
                    # columns and numpy scalar values have stable JSON-native
                    # representations before canonical hashing.
                    "content_hash": canonical_hash(
                        json.loads(group.to_json(orient="records", date_format="iso"))
                    ),
                    "source": str(cell_dir),
                }
                previous = retained.get(key)
                if previous is not None and (
                    previous["episode_seed"] != record["episode_seed"]
                    or previous["content_hash"] != record["content_hash"]
                ):
                    conflicts.append(key)
                else:
                    retained[key] = record
    return retained, conflicts


def _seed(cell: TargetCell, repetition: int, legacy_index: int | None = None) -> int:
    root = Seed(cell.base_seed)
    if cell.common_random_numbers:
        namespace = root
    elif legacy_index is not None:
        namespace = root.derive(f"grid-cell:{legacy_index}")
    else:
        namespace = root.derive(f"scientific-cell:{cell.cell_key}")
    return int(namespace.derive(f"episode:{repetition}"))


def _latest_target(study_dir: Path) -> Mapping[str, Any]:
    targets = sorted((study_dir / "extensions").glob("extension-*/target_manifest.json"))
    if not targets:
        raise ValueError("study lineage has no target manifest")
    return _read_json(targets[-1])


def _target_signature(cells: Sequence[TargetCell], analysis_hash: str | None) -> str:
    return canonical_hash(
        {
            "cells": [
                {"cell_key": cell.cell_key, "repetitions": cell.repetitions}
                for cell in sorted(cells, key=lambda item: item.cell_key)
            ],
            "analysis_recipe_hash": analysis_hash,
        }
    )


def _prior_cell_current_fingerprint(item: Mapping[str, Any]) -> str | None:
    """Re-evaluate an older manifest cell with the current identity policy."""

    try:
        source = load_run_config_or_grid(str(item["config_path"]))
        index = int(item["source_cell_index"])
        if isinstance(source, GridSpec):
            cell = next(value for value in source.cells if int(value.index) == index)
            swept_paths = tuple(axis.path for axis in source.axes)
        else:
            if index != 0:
                return None
            cell = type("StandaloneCell", (), {"config": source})()
            swept_paths = ()
        return protocol_fingerprint(cell.config.to_dict(), swept_paths=swept_paths)
    except (KeyError, OSError, StopIteration, TypeError, ValueError):
        return None


def _align_target_cells_to_prior_identity(
    target_cells: Sequence[TargetCell], previous_target: Mapping[str, Any]
) -> tuple[TargetCell, ...]:
    """Preserve lineage keys when only the identity exclusion policy changed.

    A retained target manifest is authoritative for cell/episode keys.  Older
    manifests can be re-evaluated with the current scientific fingerprint
    policy when their source configs remain available.  If exactly one prior
    cell now matches a target scientifically, carry its persisted identity
    forward instead of manufacturing a new lineage solely because an
    operational retention field was newly classified as non-scientific.
    """

    prior = [item for item in previous_target.get("cells", ()) if isinstance(item, Mapping)]
    evaluated = [(item, _prior_cell_current_fingerprint(item)) for item in prior]
    aligned: list[TargetCell] = []
    for target in target_cells:
        matches = [
            item
            for item, fingerprint in evaluated
            if fingerprint == target.protocol_fingerprint
            and dict(item.get("coordinates", {})) == dict(target.coordinates)
        ]
        if len(matches) == 1:
            match = matches[0]
            aligned.append(
                replace(
                    target,
                    protocol_fingerprint=str(match["protocol_fingerprint"]),
                    cell_key=str(match["cell_key"]),
                )
            )
        else:
            aligned.append(target)
    return tuple(aligned)


def _manifest_target_signature(target: Mapping[str, Any]) -> str:
    return canonical_hash(
        {
            "cells": sorted(
                [
                    {
                        "cell_key": str(cell["cell_key"]),
                        "repetitions": int(cell["repetitions"]),
                    }
                    for cell in target.get("cells", [])
                    if isinstance(cell, Mapping)
                ],
                key=lambda item: item["cell_key"],
            ),
            "analysis_recipe_hash": target.get("analysis_recipe_hash"),
        }
    )


def extension_aggregation_context(
    study_dir: str | Path,
) -> tuple[Mapping[str, Any], tuple[SubmissionEntry, ...]]:
    """Return the latest target and every physical source in one lineage."""

    root = Path(study_dir).expanduser().resolve()
    target = _latest_target(root)
    entries = [
        replace(entry, scientific_cell_key="AUTO")
        for entry in read_submission_manifest(root / "submission_manifest.csv")
    ]
    for manifest_path in sorted(
        (root / "extensions").glob("extension-*/execution_manifest.csv")
    ):
        extension_index = int(manifest_path.parent.name.rsplit("-", 1)[-1])
        attempt_files = sorted((manifest_path.parent / "submissions").glob("attempt-*.json"))
        attempt = max(0, len(attempt_files) - 1)
        for row in read_execution_manifest(manifest_path):
            plan_rows: list[Mapping[str, str]] = []
            if row.episode_plan_path:
                with Path(row.episode_plan_path).open(newline="", encoding="utf-8") as stream:
                    plan_rows = list(csv.DictReader(stream))
            config = Path(row.config_path)
            entries.append(
                SubmissionEntry(
                    array_index=len(entries),
                    config_path=str(config),
                    config_hash=file_sha256(config),
                    resolved_config_hash="",
                    output_dir=row.output_dir,
                    expected_cell_count=1,
                    expected_episode_count=len(plan_rows),
                    execution_seed=0,
                    git_commit="",
                    source_extension_index=extension_index,
                    source_submission_attempt=attempt,
                    scientific_cell_key=row.cell_key,
                )
            )
    return target, tuple(entries)


def consolidate_extension_tables(
    tables: Mapping[str, pd.DataFrame], target: Mapping[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Deduplicate retries and validate the exact latest target episode set."""

    result = {name: frame.copy() for name, frame in tables.items()}
    target_cells = {
        str(cell["cell_key"]): cell
        for cell in target.get("cells", [])
        if isinstance(cell, Mapping)
    }
    expected = {
        episode_key(key, repetition)
        for key, cell in target_cells.items()
        for repetition in range(int(cell["repetitions"]))
    }
    episodes = result["episodes"]
    errors: list[str] = []
    if not episodes.empty:
        completed = episodes[
            episodes["status"].isin(["completed", "skipped_resumed"])
        ].copy()
        missing_keys = completed["episode_key"].isna()
        if bool(missing_keys.any()):
            errors.append("completed extension episodes lack scientific episode keys")
        conflict_count = 0
        for _, group in completed.fillna("").groupby("episode_key", dropna=False):
            signatures = {
                canonical_hash(
                    {
                        "episode_seed": row.get("episode_seed"),
                        "status": row.get("status"),
                        "interaction_count": row.get("interaction_count"),
                    }
                )
                for row in group.to_dict(orient="records")
            }
            conflict_count += int(len(signatures) > 1)
        if conflict_count:
            errors.append(f"disagreeing duplicate episode keys: {conflict_count}")
        episodes = completed.sort_values(
            ["source_extension_index", "source_submission_attempt"]
        ).drop_duplicates("episode_key", keep="first")
        result["episodes"] = episodes.reset_index(drop=True)
    found = set(result["episodes"].get("episode_key", pd.Series(dtype=str)).dropna())
    missing = expected - found
    extra = found - expected
    if missing:
        errors.append(f"missing target episode keys: {len(missing)}")
    if extra:
        errors.append(f"episodes outside latest target: {len(extra)}")

    for name, coordinates in (
        ("rounds", ["episode_key", "round_index"]),
        ("micro_slots", ["episode_key", "round_index", "micro_slot_index"]),
    ):
        frame = result[name]
        if not frame.empty:
            result[name] = frame.sort_values(
                ["source_extension_index", "source_submission_attempt"]
            ).drop_duplicates(coordinates, keep="first").reset_index(drop=True)

    cells = result["cells"]
    if not cells.empty:
        rows = []
        for key, target_cell in target_cells.items():
            matches = cells[cells["cell_id"] == key]
            if matches.empty:
                continue
            row = matches.sort_values(
                ["source_extension_index", "source_submission_attempt"]
            ).iloc[0].to_dict()
            matched_episodes = result["episodes"][result["episodes"]["cell_id"] == key]
            row["expected_episodes"] = int(target_cell["repetitions"])
            row["completed_episodes"] = len(matched_episodes)
            row["failed_episodes"] = 0
            row["sealed"] = len(matched_episodes) == int(target_cell["repetitions"])
            rows.append(row)
        result["cells"] = pd.DataFrame(rows, columns=cells.columns)
    return result, {
        "valid": not errors,
        "complete": not errors,
        "errors": errors,
        "warnings": [],
        "counts": {
            "expected_configs": len({cell["config_index"] for cell in target_cells.values()}),
            "found_configs": len({cell["config_index"] for cell in target_cells.values()}),
            "expected_cells": len(target_cells),
            "found_cells": len(result["cells"]),
            "sealed_cells": int(result["cells"].get("sealed", pd.Series(dtype=bool)).sum()),
            "incomplete_cells": max(0, len(target_cells) - int(result["cells"].get("sealed", pd.Series(dtype=bool)).sum())),
            "expected_episodes": len(expected),
            "completed_episodes": len(found & expected),
            "failed_episodes": 0,
            "aborted_episodes": 0,
            "duplicate_run_identities": 0,
            "duplicate_cell_identities": 0,
            "duplicate_episode_identities": max(0, len(tables["episodes"]) - len(result["episodes"])),
            "missing_scientific_events": 0,
            "round_rows": len(result["rounds"]),
            "micro_slot_rows": len(result["micro_slots"]),
            "artifact_hash_mismatches": 0,
            "config_mismatches": 0,
        },
        "scientific_schema_versions": sorted(
            str(value)
            for value in result["episodes"].get(
                "scientific_schema_version", pd.Series(dtype=object)
            ).dropna().unique()
        ),
        "paired_initialization": {"required": False, "paired_initialization_pass": None, "errors": []},
        "target_extension_index": int(target["extension_index"]),
    }


def plan_extension(
    study_dir: str | Path,
    config_dir: str | Path,
    *,
    extension_index: int | None = None,
    submission_attempt: int = 0,
) -> ExtensionPlan:
    root = Path(study_dir).expanduser().resolve()
    if (root / "study_lineage.json").is_file():
        lineage = _read_json(root / "study_lineage.json")
        previous_target = _latest_target(root)
    else:
        inspection = index_existing_study(root, dry_run=True)
        lineage = inspection
        previous_target = inspection["target"]
    spec = discover_study(config_dir)
    target_cells = _align_target_cells_to_prior_identity(_cells(spec), previous_target)
    current_analysis = file_sha256(spec.analysis_recipe) if spec.analysis_recipe else None
    same_target = _target_signature(target_cells, current_analysis) == _manifest_target_signature(
        previous_target
    )
    next_index = (
        int(lineage.get("latest_extension_index", 0))
        if same_target
        else int(lineage.get("latest_extension_index", 0)) + 1
    )
    selected_index = next_index if extension_index is None else extension_index
    prior_cells = {
        str(item["cell_key"]): item
        for item in previous_target.get("cells", [])
        if isinstance(item, Mapping)
    }
    prior_protocols_by_coordinates: dict[str, set[str]] = {}
    for item in previous_target.get("cells", []):
        if isinstance(item, Mapping):
            coordinate_key = canonical_hash(item.get("coordinates", {}))
            prior_protocols_by_coordinates.setdefault(coordinate_key, set()).add(
                str(item.get("protocol_fingerprint"))
            )
    target_protocols_by_coordinates: dict[str, set[str]] = {}
    for item in target_cells:
        coordinate_key = canonical_hash(item.coordinates)
        target_protocols_by_coordinates.setdefault(coordinate_key, set()).add(
            item.protocol_fingerprint
        )
    retained, conflicts = _retained_episodes(root, target_cells)
    incompatible: list[str] = []
    classifications: dict[str, str] = {}
    episodes: list[PlannedEpisode] = []
    retained_target_count = 0
    extension_root = root / "extensions" / f"extension-{selected_index:04d}"
    for cell in target_cells:
        previous = prior_cells.get(cell.cell_key)
        coordinate_key = canonical_hash(cell.coordinates)
        prior_protocols = prior_protocols_by_coordinates.get(coordinate_key, set())
        target_protocols = target_protocols_by_coordinates.get(coordinate_key, set())
        if (
            previous is None
            and prior_protocols
            and not prior_protocols.issubset(target_protocols)
        ):
            incompatible.append(cell.cell_key)
            classifications[cell.cell_key] = CompatibilityStatus.INCOMPATIBLE.value
            continue
        if previous is not None and previous.get("protocol_fingerprint") != cell.protocol_fingerprint:
            incompatible.append(cell.cell_key)
            classifications[cell.cell_key] = CompatibilityStatus.INCOMPATIBLE.value
            continue
        present = {
            index
            for index in range(cell.repetitions)
            if episode_key(cell.cell_key, index) in retained
        }
        retained_target_count += len(present)
        missing = sorted(set(range(cell.repetitions)) - present)
        if any(episode_key(cell.cell_key, index) in conflicts for index in range(cell.repetitions)):
            classifications[cell.cell_key] = CompatibilityStatus.CONFLICTED.value
            continue
        if not missing:
            classifications[cell.cell_key] = CompatibilityStatus.COMPLETE_REUSABLE.value
        elif present:
            classifications[cell.cell_key] = CompatibilityStatus.NEEDS_ADDITIONAL_EPISODES.value
        else:
            classifications[cell.cell_key] = CompatibilityStatus.NEW_CELL.value
        legacy_index = int(previous["source_cell_index"]) if previous is not None else None
        for repetition in missing:
            seed = _seed(cell, repetition, legacy_index=legacy_index)
            output = extension_root / "runs" / Path(cell.config_path).stem / f"cell-{cell.cell_key[:16]}"
            episodes.append(
                PlannedEpisode(
                    study_lineage_id=str(lineage["study_lineage_id"]),
                    extension_index=selected_index,
                    submission_attempt=submission_attempt,
                    protocol_fingerprint=cell.protocol_fingerprint,
                    cell_key=cell.cell_key,
                    source_config=cell.config_path,
                    resolved_coordinates=cell.coordinates,
                    repetition_index=repetition,
                    episode_seed=seed,
                    episode_key=episode_key(cell.cell_key, repetition),
                    expected_output_dir=str(output),
                )
            )
    previous_analysis = previous_target.get("analysis_recipe_hash")
    return ExtensionPlan(
        study_dir=root,
        config_dir=Path(config_dir).expanduser().resolve(),
        study_lineage_id=str(lineage["study_lineage_id"]),
        extension_index=selected_index,
        submission_attempt=submission_attempt,
        target_cells=target_cells,
        episodes=tuple(episodes),
        classifications=classifications,
        retained_episode_count=retained_target_count,
        conflicts=tuple(sorted(set(conflicts))),
        incompatible=tuple(sorted(set(incompatible))),
        analysis_changed=previous_analysis != current_analysis,
    )


def _write_episode_plan(path: Path, episodes: Sequence[PlannedEpisode]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_PLAN_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for episode in episodes:
            row = asdict(episode)
            row["resolved_coordinates"] = json.dumps(
                row["resolved_coordinates"], sort_keys=True, separators=(",", ":")
            )
            writer.writerow(row)
    return path


def _attempt_number(extension_dir: Path) -> int:
    attempts = sorted((extension_dir / "submissions").glob("attempt-*.json"))
    return len(attempts)


def extend_study(
    study_dir: str | Path,
    config_dir: str | Path,
    *,
    dry_run: bool = False,
    throttle: int | None = None,
    job_script: str | Path | None = None,
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> ExtensionResult:
    """Plan the exact missing episode set and optionally submit one generic array."""

    root = Path(study_dir).expanduser().resolve()
    if not (root / "study_lineage.json").is_file():
        if not dry_run:
            index_existing_study(root)
    if dry_run:
        plan = plan_extension(root, config_dir)
        if plan.conflicts or plan.incompatible:
            raise ValueError("extension contains incompatible or conflicted observations")
        return ExtensionResult(plan, True, None, None, (), None)

    lock_path = root / ".study-extension.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        lineage = _read_json(root / "study_lineage.json")
        previous_target = _latest_target(root)
        target_spec = discover_study(config_dir)
        proposed_cells = _cells(target_spec)
        proposed_analysis = (
            file_sha256(target_spec.analysis_recipe)
            if target_spec.analysis_recipe is not None
            else None
        )
        retry = _target_signature(
            proposed_cells, proposed_analysis
        ) == _manifest_target_signature(previous_target)
        extension_index = int(lineage.get("latest_extension_index", 0)) + (0 if retry else 1)
        extension_dir = root / "extensions" / f"extension-{extension_index:04d}"
        if retry and extension_index > 0:
            state_path = extension_dir / "state.json"
            state = _read_json(state_path) if state_path.is_file() else {}
            if state.get("status") == "SUBMITTED":
                raise ValueError(
                    f"extension {extension_index} already has an active submission"
                )
        elif not retry and extension_dir.exists():
            raise ValueError(f"extension index already allocated: {extension_index}")
        attempt = _attempt_number(extension_dir) if retry else 0
        plan = plan_extension(
            root,
            config_dir,
            extension_index=extension_index,
            submission_attempt=attempt,
        )
        if plan.conflicts or plan.incompatible:
            raise ValueError("extension contains incompatible or conflicted observations")
        spec = discover_study(config_dir)
        extension_dir.mkdir(parents=True, exist_ok=retry)
        target = _target_manifest(spec, plan.study_lineage_id, extension_index, plan.target_cells)
        if not retry:
            _atomic_json(extension_dir / "target_manifest.json", target)
        _atomic_json(extension_dir / "compatibility_report.json", plan.report())
        _atomic_json(
            extension_dir / "state.json",
            {"schema_version": 1, "status": "PLANNED", "planned_at": _now()},
        )
        if not retry:
            _atomic_json(
                root / "study_lineage.json",
                {**dict(lineage), "latest_extension_index": extension_index},
            )
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    if not plan.episodes:
        _atomic_json(
            extension_dir / "state.json",
            {
                "schema_version": 1,
                "status": "COMPLETE_NO_WORK",
                "planned_at": _now(),
                "analysis_changed": plan.analysis_changed,
            },
        )
        return ExtensionResult(plan, False, extension_dir, None, (), None)

    by_cell: dict[str, list[PlannedEpisode]] = {}
    cell_by_key = {cell.cell_key: cell for cell in plan.target_cells}
    for episode in plan.episodes:
        by_cell.setdefault(episode.cell_key, []).append(episode)
    execution_rows: list[ExecutionEntry] = []
    for cell_key, episodes in sorted(by_cell.items()):
        cell = cell_by_key[cell_key]
        plan_path = _write_episode_plan(
            extension_dir / "episode_plans" / f"{cell_key}.csv", episodes
        )
        execution_rows.append(
            ExecutionEntry(
                array_index=len(execution_rows),
                config_index=cell.config_index,
                config_path=cell.config_path,
                cell_index=cell.source_cell_index,
                cell_id=cell.source_cell_id,
                output_dir=episodes[0].expected_output_dir,
                extension_index=plan.extension_index,
                cell_key=cell.cell_key,
                episode_plan_path=str(plan_path),
                study_root=str(root),
            )
        )
    execution_manifest = write_execution_manifest(
        extension_dir / "execution_manifest.csv", execution_rows
    )
    execution_plan = plan_cell_execution(discover_study(config_dir), len(execution_rows))
    if throttle is not None:
        if throttle < 1 or throttle > execution_plan.array_throttle:
            raise ValueError(
                f"requested throttle {throttle} exceeds planned RPM-safe throttle "
                f"{execution_plan.array_throttle}"
            )
        execution_plan = replace(
            execution_plan,
            array_throttle=throttle,
            total_request_concurrency=throttle * execution_plan.request_concurrency_per_shard,
            total_episode_slots=throttle * execution_plan.episode_slots_per_shard,
            estimated_rpm=(
                throttle * execution_plan.request_concurrency_per_shard * 60.0
                / execution_plan.assumed_latency_seconds
            ),
        )
    plan_payload = {
        **execution_plan.to_dict(),
        "extension_index": plan.extension_index,
        "target_episode_count": plan.target_episode_count,
        "delta_episode_count": plan.missing_episode_count,
    }
    _atomic_json(extension_dir / "execution_plan.json", plan_payload)
    attempts = extension_dir / "submissions"
    attempt = plan.submission_attempt
    logs = extension_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    script = Path(job_script or "scripts/Potsdam/SLURM/run_study_cell_array.job").resolve()
    if not script.is_file():
        raise ValueError(f"SLURM study job script does not exist: {script}")
    command = (
        "sbatch",
        f"--partition={execution_plan.partition}",
        f"--qos={execution_plan.qos}",
        "--nodes=1",
        "--ntasks=1",
        f"--array=0-{len(execution_rows) - 1}%{execution_plan.array_throttle}",
        f"--cpus-per-task={execution_plan.cpus_per_task}",
        f"--mem={execution_plan.memory}",
        f"--time={execution_plan.time_limit}",
        f"--output={logs / 'slurm-%A_%a.out'}",
        f"--error={logs / 'slurm-%A_%a.err'}",
        str(script),
        str(execution_manifest),
    )
    runner = subprocess.run if run is None else run
    started = _now()
    try:
        completed = runner(command, check=True, capture_output=True, text=True)
        stdout = completed.stdout.strip()
        match = re.search(r"Submitted\s+batch\s+job\s+(\d+)", stdout, flags=re.I)
        if match is None:
            raise ValueError(f"could not parse SLURM job ID from sbatch output: {stdout!r}")
        job_id = match.group(1)
        status = "SUBMITTED"
        error = None
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        job_id = None
        status = "SUBMISSION_FAILED"
        error = str(exc)
    _atomic_json(
        attempts / f"attempt-{attempt:04d}.json",
        {
            "schema_version": 1,
            "extension_index": plan.extension_index,
            "submission_attempt": attempt,
            "status": status,
            "submitted_at": started,
            "job_id": job_id,
            "command": list(command),
            "error": error,
        },
    )
    _atomic_json(
        extension_dir / "state.json",
        {"schema_version": 1, "status": status, "updated_at": _now()},
    )
    if error is not None:
        raise ValueError(f"SLURM submission failed: {error}")
    return ExtensionResult(plan, False, extension_dir, job_id, command, plan_payload)


__all__ = [
    "CompatibilityStatus",
    "EPISODE_PLAN_COLUMNS",
    "ExtensionPlan",
    "ExtensionResult",
    "PlannedEpisode",
    "TargetCell",
    "extend_study",
    "index_existing_study",
    "plan_extension",
]
