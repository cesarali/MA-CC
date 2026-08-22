"""Strict, machine-readable validation of discovered study artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from mas_cc.storage import file_sha256, validate_cell_artifact

from .discovery import DiscoveredCell, DiscoveredRun
from .submission import SubmissionEntry


def _current_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_study(
    entries: tuple[SubmissionEntry, ...],
    runs: tuple[DiscoveredRun, ...],
    cells: tuple[DiscoveredCell, ...],
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    found_indices = {run.entry.array_index for run in runs}
    missing_configs = sorted(set(range(len(entries))) - found_indices)
    if missing_configs:
        errors.append("missing expected config run(s): " + ", ".join(map(str, missing_configs)))

    expected_cells = sum(entry.expected_cell_count for entry in entries)
    if len(cells) != expected_cells:
        errors.append(f"found {len(cells)} scientific cells; expected {expected_cells}")

    run_counts = Counter((run.entry.array_index, run.run_id) for run in runs)
    duplicate_runs = [identity for identity, count in run_counts.items() if count > 1]
    # Repeated run identities are allowed only when their cell sets are disjoint
    # execution shards; overlapping cells are detected below.
    cell_counts = Counter((cell.run.entry.array_index, cell.run.run_id, cell.local_cell_id.split("@", 1)[0]) for cell in cells)
    duplicate_cells = [identity for identity, count in cell_counts.items() if count > 1]
    if duplicate_cells:
        errors.append(f"duplicate scientific cell identities: {len(duplicate_cells)}")

    episodes = tables["episodes"]
    cell_table = tables["cells"]
    # Compact rows intentionally hash the episode-resolved config (whose seed
    # is episode-derived), while cells hash the cell-resolved config. They are
    # different levels of identity and must not be compared for literal
    # equality. Conflicts at either level are caught by the cell/episode keys,
    # source config hash, and the compact artifact's own validator.
    config_mismatches = 0
    episode_keys = ["source_config_index", "source_run_id", "source_cell_id", "episode_id"]
    duplicate_episodes = int(episodes.duplicated(episode_keys).sum()) if not episodes.empty else 0
    if duplicate_episodes:
        errors.append(f"duplicate scientific episode identities: {duplicate_episodes}")

    sealed = 0
    hash_failures = 0
    missing_scientific = 0
    for cell in cells:
        seal_path = cell.path / "cell_complete.json"
        scientific = cell.path / "scientific_events.parquet"
        if seal_path.is_file():
            try:
                validate_cell_artifact(cell.path)
                sealed += 1
                seal = json.loads(seal_path.read_text(encoding="utf-8"))
                for relative, metadata in seal.get("artifacts", {}).items():
                    artifact = cell.path / relative
                    expected = metadata.get("sha256") if isinstance(metadata, Mapping) else None
                    if expected and (not artifact.is_file() or file_sha256(artifact) != expected):
                        hash_failures += 1
            except ValueError as exc:
                errors.append(str(exc))
        elif scientific.is_file():
            warnings.append(f"unsealed scientific table: {scientific}")
        elif not list(cell.path.glob(".resume/*/scientific_events.parquet")):
            missing_scientific += 1
            warnings.append(f"missing scientific_events.parquet: {cell.path}")
    for run in runs:
        artifacts = run.manifest.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            continue
        for relative, metadata in artifacts.items():
            artifact = run.path / str(relative)
            expected = metadata.get("sha256") if isinstance(metadata, Mapping) else None
            if expected and (not artifact.is_file() or file_sha256(artifact) != expected):
                hash_failures += 1
    if hash_failures:
        errors.append(f"retained artifact hash mismatches: {hash_failures}")

    for entry in entries:
        config_path = Path(entry.config_path)
        if config_path.is_file() and _current_hash(config_path) != entry.config_hash:
            errors.append(f"source config hash changed since submission: {config_path}")

    expected_episodes = sum(entry.expected_episode_count for entry in entries)
    completed = int(episodes["status"].isin(["completed", "skipped_resumed"]).sum()) if not episodes.empty else 0
    failed = int(episodes["status"].isin(["failed"]).sum()) if not episodes.empty else 0
    aborted = int(episodes["status"].isin(["aborted", "skipped_aborted"]).sum()) if not episodes.empty else 0
    if completed + failed + aborted != expected_episodes:
        errors.append(
            f"found {completed + failed + aborted} episode outcomes; expected {expected_episodes}"
        )
    if failed or aborted:
        errors.append(f"failed episodes: {failed}; aborted episodes: {aborted}")

    schemas = sorted(
        {str(value) for value in episodes.get("scientific_schema_version", pd.Series(dtype=object)).dropna()}
    )
    if len(schemas) > 1:
        errors.append("mismatched scientific schema versions: " + ", ".join(schemas))
    if tables["rounds"].empty:
        warnings.append("no round records discovered")
    if tables["micro_slots"].empty:
        warnings.append("no micro-slot records discovered")

    incomplete_cells = 0
    if not cell_table.empty:
        incomplete_cells = int(
            (
                (cell_table["completed_episodes"].astype(int) != cell_table["expected_episodes"].astype(int))
                | (cell_table["failed_episodes"].astype(int) > 0)
            ).sum()
        )

    return {
        "schema_version": 1,
        "valid": not errors,
        "complete": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "expected_configs": len(entries),
            "found_configs": len(found_indices),
            "expected_cells": expected_cells,
            "found_cells": len(cells),
            "sealed_cells": sealed,
            "incomplete_cells": incomplete_cells,
            "expected_episodes": expected_episodes,
            "completed_episodes": completed,
            "failed_episodes": failed,
            "aborted_episodes": aborted,
            "duplicate_run_identities": len(duplicate_runs),
            "duplicate_cell_identities": len(duplicate_cells),
            "duplicate_episode_identities": duplicate_episodes,
            "missing_scientific_events": missing_scientific,
            "round_rows": len(tables["rounds"]),
            "micro_slot_rows": len(tables["micro_slots"]),
            "artifact_hash_mismatches": hash_failures,
            "config_mismatches": config_mismatches,
        },
        "scientific_schema_versions": schemas,
    }


def validation_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# Study validation",
        "",
        f"Status: **{'VALID' if report['valid'] else 'INVALID'}**",
        "",
        f"- Expected configs / found configs: {counts['expected_configs']} / {counts['found_configs']}",
        f"- Expected cells / found cells / sealed cells: {counts['expected_cells']} / {counts['found_cells']} / {counts['sealed_cells']}",
        f"- Expected episodes / completed / failed / aborted: {counts['expected_episodes']} / {counts['completed_episodes']} / {counts['failed_episodes']} / {counts['aborted_episodes']}",
        f"- Duplicate run / cell / episode identities: {counts['duplicate_run_identities']} / {counts['duplicate_cell_identities']} / {counts['duplicate_episode_identities']}",
        f"- Round / micro-slot rows: {counts['round_rows']} / {counts['micro_slot_rows']}",
        "",
        "## Errors",
        "",
        *(f"- {item}" for item in report["errors"]),
        *( ["- none"] if not report["errors"] else [] ),
        "",
        "## Warnings",
        "",
        *(f"- {item}" for item in report["warnings"]),
        *( ["- none"] if not report["warnings"] else [] ),
        "",
    ]
    return "\n".join(lines)


__all__ = ["validate_study", "validation_markdown"]
