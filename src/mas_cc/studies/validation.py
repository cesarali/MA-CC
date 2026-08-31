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


def paired_initialization_diagnostics(
    tables: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Summarize and strictly compare the realized pre-round physical state."""

    rounds = tables.get("rounds", pd.DataFrame())
    required = {
        "cell_id",
        "episode_id",
        "round_index",
        "target_count_before",
        "N",
        "initialization_repetition",
        "physical_initial_state_hash",
        "initial_task_id",
        "initial_vote_vector",
        "initial_active_fact_ids_by_agent",
        "initial_known_fact_ids_by_agent",
    }
    if rounds.empty or not required.issubset(rounds.columns):
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "required": False,
                "paired_initialization_pass": None,
                "errors": [],
            },
        )
    first = rounds[pd.to_numeric(rounds["round_index"], errors="coerce") == 0].copy()
    first["n_0"] = pd.to_numeric(first["target_count_before"], errors="coerce")
    first["x_0"] = first["n_0"] / pd.to_numeric(first["N"], errors="coerce")
    summaries = first.groupby("cell_id", as_index=False).agg(
        mean_x_0=("x_0", "mean"),
        sd_x_0=("x_0", "std"),
        min_x_0=("x_0", "min"),
        max_x_0=("x_0", "max"),
        mean_n_0=("n_0", "mean"),
        unique_initial_state_count=("physical_initial_state_hash", "nunique"),
        repetitions=("initialization_repetition", "nunique"),
    )
    cells = tables.get("cells", pd.DataFrame())
    if not cells.empty:
        coordinates = [
            column
            for column in (
                "cell_id",
                "epistemic_persistence",
                "intervention_budget",
                "target_semantics",
            )
            if column in cells
        ]
        summaries = summaries.merge(cells[coordinates], on="cell_id", how="left")

    audit_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    expected_cells = int(first["cell_id"].nunique())
    for repetition, group in first.groupby("initialization_repetition", sort=True):
        signatures = {
            (
                str(row["physical_initial_state_hash"]),
                str(row["initial_task_id"]),
                json.dumps(row["initial_vote_vector"], sort_keys=True),
                json.dumps(row["initial_active_fact_ids_by_agent"], sort_keys=True),
                json.dumps(row["initial_known_fact_ids_by_agent"], sort_keys=True),
            )
            for row in group.to_dict(orient="records")
        }
        n_values = sorted(set(pd.to_numeric(group["n_0"], errors="coerce")))
        x_values = sorted(set(pd.to_numeric(group["x_0"], errors="coerce")))
        cells_seen = int(group["cell_id"].nunique())
        passed = len(signatures) == 1 and cells_seen == expected_cells
        if not passed:
            failures.append(
                f"repetition {int(repetition)} has {len(signatures)} initial states "
                f"across {cells_seen}/{expected_cells} cells"
            )
        audit_rows.append(
            {
                "initialization_repetition": int(repetition),
                "n_0": n_values[0] if len(n_values) == 1 else None,
                "x_0": x_values[0] if len(x_values) == 1 else None,
                "initial_vote_vector_hash_count": len(signatures),
                "physical_initial_state_hash": (
                    next(iter(signatures))[0] if len(signatures) == 1 else None
                ),
                "cells_expected": expected_cells,
                "cells_seen": cells_seen,
                "paired_initialization_pass": passed,
            }
        )
    unique_states = int(first["physical_initial_state_hash"].nunique())
    repetitions = int(first["initialization_repetition"].nunique())
    if unique_states != repetitions:
        failures.append(
            f"initialization varies across only {unique_states}/{repetitions} repetitions"
        )
    report = {
        "required": True,
        "paired_initialization_pass": not failures,
        "repetitions": repetitions,
        "cells_per_repetition": expected_cells,
        "unique_initial_state_count": unique_states,
        "initialization_varies_across_repetitions": unique_states == repetitions,
        "errors": failures,
    }
    return summaries, pd.DataFrame(audit_rows), report


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
        errors.append(
            "missing expected config run(s): " + ", ".join(map(str, missing_configs))
        )

    expected_cells = sum(entry.expected_cell_count for entry in entries)
    if len(cells) != expected_cells:
        errors.append(f"found {len(cells)} scientific cells; expected {expected_cells}")

    run_counts = Counter((run.entry.array_index, run.run_id) for run in runs)
    duplicate_runs = [identity for identity, count in run_counts.items() if count > 1]
    # Repeated run identities are allowed only when their cell sets are disjoint
    # execution shards; overlapping cells are detected below.
    cell_counts = Counter(
        (
            cell.run.entry.array_index,
            cell.run.run_id,
            cell.local_cell_id.split("@", 1)[0],
        )
        for cell in cells
    )
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
    episode_keys = [
        "source_config_index",
        "source_run_id",
        "source_cell_id",
        "episode_id",
    ]
    duplicate_episodes = (
        int(episodes.duplicated(episode_keys).sum()) if not episodes.empty else 0
    )
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
                    expected = (
                        metadata.get("sha256")
                        if isinstance(metadata, Mapping)
                        else None
                    )
                    if expected and (
                        not artifact.is_file() or file_sha256(artifact) != expected
                    ):
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
            if expected and (
                not artifact.is_file() or file_sha256(artifact) != expected
            ):
                hash_failures += 1
    if hash_failures:
        errors.append(f"retained artifact hash mismatches: {hash_failures}")

    for entry in entries:
        config_path = Path(entry.config_path)
        if config_path.is_file() and _current_hash(config_path) != entry.config_hash:
            errors.append(f"source config hash changed since submission: {config_path}")

    expected_episodes = sum(entry.expected_episode_count for entry in entries)
    completed = (
        int(episodes["status"].isin(["completed", "skipped_resumed"]).sum())
        if not episodes.empty
        else 0
    )
    failed = int(episodes["status"].isin(["failed"]).sum()) if not episodes.empty else 0
    aborted = (
        int(episodes["status"].isin(["aborted", "skipped_aborted"]).sum())
        if not episodes.empty
        else 0
    )
    if completed + failed + aborted != expected_episodes:
        errors.append(
            f"found {completed + failed + aborted} episode outcomes; expected {expected_episodes}"
        )
    if failed or aborted:
        errors.append(f"failed episodes: {failed}; aborted episodes: {aborted}")

    schemas = sorted(
        {
            str(value)
            for value in episodes.get(
                "scientific_schema_version", pd.Series(dtype=object)
            ).dropna()
        }
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
                (
                    cell_table["completed_episodes"].astype(int)
                    != cell_table["expected_episodes"].astype(int)
                )
                | (cell_table["failed_episodes"].astype(int) > 0)
            ).sum()
        )

    _, _, initialization = paired_initialization_diagnostics(tables)
    if initialization["required"] and not initialization["paired_initialization_pass"]:
        errors.extend(initialization["errors"])

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
        "paired_initialization": initialization,
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
        f"- Paired initialization: {report.get('paired_initialization', {}).get('paired_initialization_pass')}",
        "",
        "## Errors",
        "",
        *(f"- {item}" for item in report["errors"]),
        *(["- none"] if not report["errors"] else []),
        "",
        "## Warnings",
        "",
        *(f"- {item}" for item in report["warnings"]),
        *(["- none"] if not report["warnings"] else []),
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "paired_initialization_diagnostics",
    "validate_study",
    "validation_markdown",
]
