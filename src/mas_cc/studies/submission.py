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

AMAREL_ACCOUNT = "general"
AMAREL_PARTITION = "main"
AMAREL_QOS = "normal"
AMAREL_MAX_WALLTIME_SECONDS = 72 * 60 * 60
AMAREL_RESULTS_ROOT = "/scratch/df630/MA-CC-results"


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


def _slurm_walltime_seconds(value: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", value)
    if match is None:
        raise ValueError(f"invalid SLURM time limit: {value!r}")
    days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid SLURM time limit: {value!r}")
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def _absolute_path(path: str | Path, *, preserve_symlinks: bool = False) -> Path:
    expanded = Path(path).expanduser()
    if preserve_symlinks:
        return Path(os.path.abspath(str(expanded)))
    return expanded.resolve()


def _default_job_script(execution_site: str, *, cell_array: bool) -> Path:
    filename = "run_study_cell_array.job" if cell_array else "run_config_array.job"
    folder = "scripts/Amarel/SLURM" if execution_site == "amarel" else "scripts/Potsdam/SLURM"
    return Path(folder) / filename


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
    source_extension_index: int = 0
    source_submission_attempt: int = 0
    scientific_cell_key: str = ""


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

    destination = Path(os.path.abspath(str(Path(results_dir).expanduser())))
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
            raise ValueError(
                f"experiment configs map to duplicate output label {label!r}"
            )
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


def write_submission_manifest(
    path: str | Path, entries: Sequence[SubmissionEntry]
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=SUBMISSION_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            {column: getattr(entry, column) for column in SUBMISSION_COLUMNS}
            for entry in entries
        )
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
        raise ValueError(
            "submission manifest array_index values must be contiguous from zero"
        )
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


def _study_manifest(
    spec: StudySpec, entries: Sequence[SubmissionEntry]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study_id": spec.name,
        "config_dir": str(spec.config_dir),
        "study_manifest_source": None
        if spec.manifest_path is None
        else str(spec.manifest_path),
        "analysis_recipe": None
        if spec.analysis_recipe is None
        else str(spec.analysis_recipe),
        "execution": dict(spec.execution),
        "preflight": dict(spec.preflight),
        "configs": [asdict(entry) for entry in entries],
        "expected_config_count": len(entries),
        "expected_cell_count": sum(entry.expected_cell_count for entry in entries),
        "expected_episode_count": sum(
            entry.expected_episode_count for entry in entries
        ),
    }


def _validate_required_initializations(spec: StudySpec) -> None:
    """Paired dynamics may start only after every shared state is sealed."""

    from mas_cc.games import create_game
    from mas_cc.games.relational_reasoning.imitation_round_feedback.initialization import (
        initialization_artifact_path,
        paired_initialization_required,
        read_initialization_artifact,
    )
    from mas_cc.studies.initialization import build_initialization_plan

    paired_configs = []
    for path in spec.configs:
        source = load_run_config_or_grid(path)
        base = source.base if isinstance(source, GridSpec) else source
        if paired_initialization_required(base):
            paired_configs.append((path, base))
    if not paired_configs:
        return
    plans = build_initialization_plan(
        [path for path, _ in paired_configs],
        initialization_artifact_path(
            paired_configs[0][1], paired_configs[0][1].execution.seed
        ).parent,
    )
    for _, config in paired_configs:
        game = create_game(config.game)
        for plan in plans:
            episode_config = replace(
                config,
                execution=replace(config.execution, seed=plan.episode_seed),
            )
            path = initialization_artifact_path(episode_config, plan.episode_seed)
            if not path.is_file():
                raise ValueError(
                    "paired initialization is incomplete; missing artifact "
                    f"for repetition {plan.repetition_index}: {path}"
                )
            read_initialization_artifact(path, game, episode_config, plan.episode_seed)


def submit_study(
    config_dir: str | Path,
    results_dir: str | Path | None = None,
    *,
    throttle: int | None = None,
    job_script: str | Path | None = None,
    require_results_under: str | Path | None = None,
    execution_site: str = "potsdam",
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> SubmissionResult:
    """Preflight all configs, publish manifests, then call ``sbatch`` exactly once."""

    from mas_cc.cli.experiment import run_experiment_preflight

    if run is None and os.environ.get("NERSC_HOST") == "perlmutter":
        raise ValueError(
            "batch study submission is disabled on NERSC Perlmutter; use "
            "`mas-cc study prepare` followed by `scripts/nersc/run_study.sh` "
            "so the allocation uses --qos=interactive"
        )
    if execution_site not in {"potsdam", "nersc", "amarel"}:
        raise ValueError("execution_site must be 'potsdam', 'nersc', or 'amarel'")
    spec = discover_study(config_dir)
    from .preflight import validate_study_preflight_contract

    validate_study_preflight_contract(spec)
    _validate_required_initializations(spec)
    runner = subprocess.run if run is None else run
    configured_results = spec.execution.get("results_root")
    amarel_results_root = _absolute_path(
        os.environ.get("AMAREL_RESULTS_ROOT", AMAREL_RESULTS_ROOT),
        preserve_symlinks=execution_site == "amarel",
    )
    if execution_site == "amarel" and results_dir is None:
        configured_results = amarel_results_root / spec.name
    study_dir = _absolute_path(
        results_dir or configured_results or (Path("results") / spec.name),
        preserve_symlinks=execution_site == "amarel",
    )
    required_results_under = (
        require_results_under
        if require_results_under is not None
        else (
            amarel_results_root
            if execution_site == "amarel"
            else spec.execution.get("require_results_under")
        )
    )
    if required_results_under is not None:
        required_root = _absolute_path(
            str(required_results_under),
            preserve_symlinks=execution_site == "amarel",
        )
        try:
            study_dir.relative_to(required_root)
        except ValueError as exc:
            raise ValueError(
                f"study results must be stored under {required_root}, got {study_dir}"
            ) from exc
        if require_results_under is not None or execution_site == "amarel":
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

    manifest_path = write_submission_manifest(
        study_dir / "submission_manifest.csv", entries
    )
    logs_dir = study_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_pattern = logs_dir / "slurm-%A_%a.out"
    stderr_pattern = logs_dir / "slurm-%A_%a.err"
    (study_dir / "study_manifest.json").write_text(
        json.dumps(_study_manifest(spec, entries), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # New studies are born as extension zero. Root-level manifests remain as
    # compatibility projections for existing readers and tools.
    from .extension import index_existing_study

    index_existing_study(study_dir)

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
                    throttle
                    * plan.request_concurrency_per_shard
                    * 60.0
                    / plan.assumed_latency_seconds
                ),
            )
        execution_plan = plan.to_dict()
        execution_manifest = write_execution_manifest(
            study_dir / "execution_manifest.csv", shards
        )
        (study_dir / "execution_plan.json").write_text(
            json.dumps(execution_plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        script = _absolute_path(
            job_script or _default_job_script(execution_site, cell_array=True),
            preserve_symlinks=execution_site == "amarel",
        )
        array = f"0-{len(shards) - 1}%{plan.array_throttle}"
        if execution_site == "amarel":
            if _slurm_walltime_seconds(plan.time_limit) > AMAREL_MAX_WALLTIME_SECONDS:
                raise ValueError("Amarel time limit cannot exceed 3-00:00:00")
            scheduler_options = (
                f"--account={AMAREL_ACCOUNT}",
                f"--partition={AMAREL_PARTITION}",
                f"--qos={AMAREL_QOS}",
                f"--chdir={script.parents[3]}",
                f"--export=ALL,AMAREL_REPO_ROOT={script.parents[3]}",
            )
        else:
            scheduler_options = (
                f"--partition={plan.partition}",
                f"--qos={plan.qos}",
            )
        command = (
            "sbatch",
            *scheduler_options,
            "--nodes=1",
            "--ntasks=1",
            f"--array={array}",
            f"--cpus-per-task={plan.cpus_per_task}",
            f"--mem={plan.memory}",
            f"--time={plan.time_limit}",
            f"--output={stdout_pattern}",
            f"--error={stderr_pattern}",
            str(script),
            str(execution_manifest),
        )
    else:
        script = _absolute_path(
            job_script or _default_job_script(execution_site, cell_array=False),
            preserve_symlinks=execution_site == "amarel",
        )
        configured_throttle = spec.execution.get("throttle")
        limit = throttle if throttle is not None else configured_throttle
        if limit is not None and (isinstance(limit, bool) or int(limit) < 1):
            raise ValueError("SLURM array throttle must be a positive integer")
        array = f"0-{len(entries) - 1}" + ("" if limit is None else f"%{int(limit)}")
        scheduler_options = (
            (
                f"--account={AMAREL_ACCOUNT}",
                f"--partition={AMAREL_PARTITION}",
                f"--qos={AMAREL_QOS}",
                f"--chdir={script.parents[3]}",
                f"--export=ALL,AMAREL_REPO_ROOT={script.parents[3]}",
            )
            if execution_site == "amarel"
            else ()
        )
        command = (
            "sbatch",
            *scheduler_options,
            f"--array={array}",
            f"--output={stdout_pattern}",
            f"--error={stderr_pattern}",
            str(script),
            str(manifest_path),
        )
    if not script.is_file():
        raise ValueError(f"SLURM study job script does not exist: {script}")
    (study_dir / "preparation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "prepared",
                "prepared_at": _now(),
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
                "execution_plan": execution_plan,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return SubmissionResult(
        study_dir, manifest_path, job_id, entries, command, execution_plan
    )


def prepare_study(
    config_dir: str | Path,
    results_dir: str | Path | None = None,
    *,
    throttle: int | None = None,
    job_script: str | Path | None = None,
    require_results_under: str | Path | None = None,
    execution_site: str = "potsdam",
) -> SubmissionResult:
    """Prepare manifests and a scheduler command without contacting SLURM."""

    def _capture(
        command: Sequence[str], **_: Any
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "Submitted batch job 0\n", "")

    result = submit_study(
        config_dir,
        results_dir,
        throttle=throttle,
        job_script=job_script,
        require_results_under=require_results_under,
        execution_site=execution_site,
        run=_capture,
    )
    (result.study_dir / "submission.json").unlink(missing_ok=True)
    return replace(result, job_id=None)


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
