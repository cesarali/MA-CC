"""Deterministic study manifests and one-shot SLURM config-array submission."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, replace
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
    execution_plan: Mapping[str, Any] | None = None


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


def prepare_study(
    config_dir: str | Path,
    results_dir: str | Path | None = None,
    *,
    throttle: int | None = None,
    job_script: str | Path | None = None,
    require_results_under: str | Path | None = None,
    execution_site: str = "unspecified",
) -> SubmissionResult:
    """Preflight all configs and publish manifests without contacting a scheduler."""

    from mas_cc.cli.experiment import run_experiment_preflight

    if execution_site not in {"unspecified", "potsdam", "nersc"}:
        raise ValueError("execution_site must be 'unspecified', 'potsdam', or 'nersc'")
    spec = discover_study(config_dir)
    configured_results = spec.execution.get("results_root")
    study_dir = Path(
        results_dir or configured_results or (Path("results") / spec.name)
    ).expanduser().resolve()
    required_results_under = (
        require_results_under
        if require_results_under is not None
        else spec.execution.get("require_results_under")
    )
    if required_results_under is not None:
        required_root = Path(str(required_results_under)).expanduser().resolve()
        try:
            study_dir.relative_to(required_root)
        except ValueError as exc:
            raise ValueError(
                f"study results must be stored under {required_root}, got {study_dir}"
            ) from exc
        if require_results_under is not None:
            spec = replace(
                spec,
                execution={
                    **spec.execution,
                    "results_root": str(study_dir),
                    "require_results_under": str(required_root),
                },
            )
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
    logs_dir = study_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_pattern = logs_dir / "slurm-%A_%a.out"
    stderr_pattern = logs_dir / "slurm-%A_%a.err"
    (study_dir / "study_manifest.json").write_text(
        json.dumps(_study_manifest(spec, entries), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    mode = str(spec.execution.get("mode", "config_array"))
    execution_plan: Mapping[str, Any] | None = None
    execution_manifest: Path | None = None
    if mode in {"auto", "cell_array"}:
        from .execution import (
            build_cell_execution_entries,
            plan_cell_execution,
            write_execution_manifest,
        )

        shards = build_cell_execution_entries(spec, entries)
        plan = plan_cell_execution(spec, len(shards))
        if throttle is not None:
            if throttle < 1:
                raise ValueError("SLURM array throttle must be a positive integer")
            if throttle > plan.array_throttle:
                raise ValueError(
                    f"requested throttle {throttle} exceeds planned RPM-safe throttle "
                    f"{plan.array_throttle}"
                )
            plan = replace(
                plan,
                array_throttle=throttle,
                total_request_concurrency=throttle * plan.request_concurrency_per_shard,
                total_episode_slots=throttle * plan.episode_slots_per_shard,
                estimated_rpm=(
                    throttle * plan.request_concurrency_per_shard * 60.0
                    / plan.assumed_latency_seconds
                ),
            )
        execution_plan = plan.to_dict()
        execution_manifest = write_execution_manifest(
            study_dir / "execution_manifest.csv", shards
        )
        (study_dir / "execution_plan.json").write_text(
            json.dumps(execution_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        script = Path(
            job_script or "scripts/Potsdam/SLURM/run_study_cell_array.job"
        ).resolve()
        array = f"0-{len(shards) - 1}%{plan.array_throttle}"
        command = (
            "sbatch", f"--partition={plan.partition}", f"--qos={plan.qos}",
            "--nodes=1", "--ntasks=1", f"--array={array}",
            f"--cpus-per-task={plan.cpus_per_task}", f"--mem={plan.memory}",
            f"--time={plan.time_limit}", f"--output={stdout_pattern}",
            f"--error={stderr_pattern}", str(script),
            str(execution_manifest),
        )
    else:
        script = Path(job_script or "scripts/Potsdam/SLURM/run_config_array.job").resolve()
        configured_throttle = spec.execution.get("throttle")
        limit = throttle if throttle is not None else configured_throttle
        if limit is not None and (isinstance(limit, bool) or int(limit) < 1):
            raise ValueError("SLURM array throttle must be a positive integer")
        array = f"0-{len(entries) - 1}" + ("" if limit is None else f"%{int(limit)}")
        command = (
            "sbatch", f"--array={array}", f"--output={stdout_pattern}",
            f"--error={stderr_pattern}", str(script), str(manifest_path),
        )
    if not script.is_file():
        raise ValueError(f"SLURM study job script does not exist: {script}")
    prepared = _now()
    (study_dir / "preparation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "prepared",
                "prepared_at": prepared,
                "execution_site": execution_site,
                "array": array,
                "worker_manifest": str(execution_manifest or manifest_path),
                "execution_plan": execution_plan,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return SubmissionResult(
        study_dir, manifest_path, None, entries, command, execution_plan
    )


def submit_study(
    config_dir: str | Path,
    results_dir: str | Path | None = None,
    *,
    throttle: int | None = None,
    job_script: str | Path | None = None,
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> SubmissionResult:
    """Prepare a study, then call ``sbatch`` exactly once."""

    if run is None and os.environ.get("NERSC_HOST") == "perlmutter":
        raise ValueError(
            "batch study submission is disabled on NERSC Perlmutter; use "
            "`mas-cc study prepare` followed by `scripts/nersc/run_study.sh` "
            "so the allocation uses --qos=interactive"
        )
    prepared = prepare_study(
        config_dir,
        results_dir,
        throttle=throttle,
        job_script=job_script,
        execution_site="potsdam",
    )
    runner = subprocess.run if run is None else run
    command = prepared.command
    study_dir = prepared.study_dir
    array = next(
        (
            argument.removeprefix("--array=")
            for argument in command
            if argument.startswith("--array=")
        ),
        "",
    )
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
                "execution_plan": prepared.execution_plan,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return SubmissionResult(
        study_dir,
        prepared.manifest_path,
        job_id,
        prepared.entries,
        command,
        prepared.execution_plan,
    )


__all__ = [
    "SUBMISSION_COLUMNS",
    "SubmissionEntry",
    "SubmissionResult",
    "array_task_command",
    "build_submission_entries",
    "read_submission_manifest",
    "resolve_array_entry",
    "prepare_study",
    "submit_study",
    "write_submission_manifest",
]
