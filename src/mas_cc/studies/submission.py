"""Deterministic study manifests and one-shot SLURM config-array submission."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.storage import canonical_hash

from .manifest import StudySpec, discover_study


SUBMISSION_COLUMNS = (
    "array_index",
    "config_path",
    "config_hash",
    "resolved_config_hash",
    "output_dir",
    "expected_cell_count",
    "expected_episode_count",
    "execution_seed",
    "git_commit",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not result:
        raise ValueError(f"cannot derive an output label from {value!r}")
    return result


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


@dataclass(frozen=True, slots=True)
class SubmissionEntry:
    array_index: int
    config_path: str
    config_hash: str
    resolved_config_hash: str
    output_dir: str
    expected_cell_count: int
    expected_episode_count: int
    execution_seed: int
    git_commit: str


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    study_dir: Path
    manifest_path: Path
    job_id: str | None
    entries: tuple[SubmissionEntry, ...]
    command: tuple[str, ...]


def build_submission_entries(
    spec: StudySpec, results_dir: str | Path, *, git_commit: str | None = None
) -> tuple[SubmissionEntry, ...]:
    """Resolve every config and create the stable scientific array mapping."""

    destination = Path(results_dir).expanduser().resolve()
    commit = _git_commit(spec.config_dir) if git_commit is None else git_commit
    entries: list[SubmissionEntry] = []
    labels: set[str] = set()
    for index, path in enumerate(spec.configs):
        source = load_run_config_or_grid(path)
        base = source.base if isinstance(source, GridSpec) else source
        cells = source.cells if isinstance(source, GridSpec) else ()
        cell_count = len(cells) if cells else 1
        episode_count = (
            sum(cell.config.execution.repetitions for cell in cells)
            if cells
            else base.execution.repetitions
        )
        label = _slug(path.stem)
        if label in labels:
            raise ValueError(f"experiment configs map to duplicate output label {label!r}")
        labels.add(label)
        resolved_identity: Mapping[str, Any]
        if isinstance(source, GridSpec):
            resolved_identity = {
                "base": source.base.to_dict(),
                "grid": source.to_dict(),
                "cells": [
                    {"overrides": dict(cell.overrides), "config": cell.config.to_dict()}
                    for cell in source.cells
                ],
            }
        else:
            resolved_identity = source.to_dict()
        entries.append(
            SubmissionEntry(
                array_index=index,
                config_path=str(path),
                config_hash=_file_hash(path),
                resolved_config_hash=canonical_hash(resolved_identity),
                output_dir=str((destination / "runs" / label).resolve()),
                expected_cell_count=cell_count,
                expected_episode_count=episode_count,
                execution_seed=base.execution.seed,
                git_commit=commit,
            )
        )
    return tuple(entries)


def write_submission_manifest(path: str | Path, entries: Sequence[SubmissionEntry]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(entry) for entry in entries)
    return destination


def read_submission_manifest(path: str | Path) -> tuple[SubmissionEntry, ...]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SUBMISSION_COLUMNS:
            raise ValueError(f"submission manifest has incompatible columns: {source}")
        rows = list(reader)
    entries = tuple(
        SubmissionEntry(
            array_index=int(row["array_index"]),
            config_path=row["config_path"],
            config_hash=row["config_hash"],
            resolved_config_hash=row["resolved_config_hash"],
            output_dir=row["output_dir"],
            expected_cell_count=int(row["expected_cell_count"]),
            expected_episode_count=int(row["expected_episode_count"]),
            execution_seed=int(row["execution_seed"]),
            git_commit=row["git_commit"],
        )
        for row in rows
    )
    if [entry.array_index for entry in entries] != list(range(len(entries))):
        raise ValueError("submission manifest array_index values must be contiguous from zero")
    return entries


def resolve_array_entry(path: str | Path, index: int) -> SubmissionEntry:
    entries = read_submission_manifest(path)
    if index < 0 or index >= len(entries):
        raise ValueError(f"SLURM array index {index} is outside 0..{len(entries) - 1}")
    return entries[index]


def array_task_command(entry: SubmissionEntry) -> tuple[str, ...]:
    return (
        "python",
        "-m",
        "mas_cc.cli.main",
        "experiment",
        "run",
        "--config",
        entry.config_path,
        "--output-dir",
        entry.output_dir,
        "--no-progress",
    )


def _study_manifest(spec: StudySpec, entries: Sequence[SubmissionEntry]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study_id": spec.name,
        "config_dir": str(spec.config_dir),
        "study_manifest_source": None if spec.manifest_path is None else str(spec.manifest_path),
        "analysis_recipe": None if spec.analysis_recipe is None else str(spec.analysis_recipe),
        "execution": dict(spec.execution),
        "configs": [asdict(entry) for entry in entries],
        "expected_config_count": len(entries),
        "expected_cell_count": sum(entry.expected_cell_count for entry in entries),
        "expected_episode_count": sum(entry.expected_episode_count for entry in entries),
    }


def submit_study(
    config_dir: str | Path,
    results_dir: str | Path | None = None,
    *,
    throttle: int | None = None,
    job_script: str | Path | None = None,
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> SubmissionResult:
    """Preflight all configs, publish manifests, then call ``sbatch`` exactly once."""

    from mas_cc.cli.experiment import run_experiment_preflight

    spec = discover_study(config_dir)
    runner = subprocess.run if run is None else run
    study_dir = Path(results_dir or (Path("results") / spec.name)).expanduser().resolve()
    entries = build_submission_entries(spec, study_dir)

    # Preflight in a temporary root so a failed member cannot leave a study that
    # looks submitted. The successful reports are copied into provenance below.
    with tempfile.TemporaryDirectory(prefix="mas-cc-study-preflight-") as temporary:
        preflight_root = Path(temporary)
        for entry in entries:
            estimate = run_experiment_preflight(
                entry.config_path, preflight_root / f"config-{entry.array_index:04d}"
            )
            if estimate.launch_status != "permitted":
                raise ValueError(
                    f"study preflight failed for {entry.config_path}: "
                    f"launch status {estimate.launch_status!r}"
                )

        study_dir.mkdir(parents=True, exist_ok=True)
        provenance = study_dir / "preflight"
        if provenance.exists():
            shutil.rmtree(provenance)
        shutil.copytree(preflight_root, provenance)

    manifest_path = write_submission_manifest(study_dir / "submission_manifest.csv", entries)
    (study_dir / "study_manifest.json").write_text(
        json.dumps(_study_manifest(spec, entries), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    script = Path(job_script or "scripts/Potsdam/SLURM/run_config_array.job").resolve()
    if not script.is_file():
        raise ValueError(f"SLURM config-array job script does not exist: {script}")
    configured_throttle = spec.execution.get("throttle")
    limit = throttle if throttle is not None else configured_throttle
    if limit is not None and (isinstance(limit, bool) or int(limit) < 1):
        raise ValueError("SLURM array throttle must be a positive integer")
    array = f"0-{len(entries) - 1}" + ("" if limit is None else f"%{int(limit)}")
    command = ("sbatch", f"--array={array}", str(script), str(manifest_path))
    started = _now()
    try:
        completed = runner(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        (study_dir / "submission.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "failed",
                    "submitted_at": started,
                    "command": list(command),
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise ValueError(f"SLURM submission failed: {exc}") from exc
    stdout = completed.stdout.strip()
    match = re.search(r"Submitted\s+batch\s+job\s+(\d+)", stdout, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"could not parse SLURM job ID from sbatch output: {stdout!r}")
    job_id = match.group(1)
    (study_dir / "submission.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "submitted",
                "submitted_at": started,
                "job_id": job_id,
                "array": array,
                "command": list(command),
                "stdout": stdout,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return SubmissionResult(study_dir, manifest_path, job_id, entries, command)


__all__ = [
    "SUBMISSION_COLUMNS",
    "SubmissionEntry",
    "SubmissionResult",
    "array_task_command",
    "build_submission_entries",
    "read_submission_manifest",
    "resolve_array_entry",
    "submit_study",
    "write_submission_manifest",
]
