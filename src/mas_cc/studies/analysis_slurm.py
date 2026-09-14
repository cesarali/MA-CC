"""Detached SLURM orchestration for standardized study information analysis."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from mas_cc.storage import canonical_hash, file_sha256

from .canonical import build_canonical_tables
from .discovery import discover_cells, discover_runs
from .submission import read_submission_manifest
from .table_io import read_scientific_table, retained_table_path, write_scientific_table
from .validation import validate_study


LAUNCHER = Path(__file__).resolve().parents[3] / "scripts/Potsdam/SLURM/run_study_analysis.job"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _job_id(stdout: str) -> str:
    value = stdout.strip().split()[-1] if stdout.strip() else ""
    if not value.isdigit():
        raise ValueError(f"could not parse SLURM job ID from: {stdout!r}")
    return value


def _entries(
    root: Path,
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...], Mapping[str, Any] | None]:
    from .aggregation import _align_indexed_lineage_cells

    target = None
    if (root / "study_lineage.json").is_file():
        from .extension import extension_aggregation_context

        target, entries = extension_aggregation_context(root)
    else:
        entries = read_submission_manifest(root / "submission_manifest.csv")
    runs = discover_runs(entries)
    cells = discover_cells(runs)
    if target is not None:
        cells = _align_indexed_lineage_cells(cells, target)
    return entries, runs, cells, target


def create_generation(root: Path, *, allow_incomplete: bool) -> tuple[Path, dict[str, Any]]:
    """Freeze canonical inputs and create a deterministic resumable generation."""

    from mas_cc.analysis.single_affinity import PROVENANCE as theory_provenance
    from .aggregation import (
        _recipe,
        _requested_statistics,
        _resampling,
        _round_events_from_canonical,
        _scientific_identity,
    )

    study_manifest = _read(root / "study_manifest.json")
    entries, runs, cells, target = _entries(root)
    retained_paths = {
        name: retained_table_path(root / "analysis" / "tables", name)
        for name in ("cells", "episodes", "rounds", "micro_slots")
    }
    reused_retained = not cells and all(retained_paths.values())
    if reused_retained:
        canonical = {
            name: read_scientific_table(path)
            for name, path in retained_paths.items()
            if path is not None
        }
        validation_path = root / "analysis" / "validation.json"
        if not validation_path.is_file():
            raise ValueError("retained canonical inputs lack validation metadata")
        validation = _read(validation_path)
    else:
        canonical, _ = build_canonical_tables(str(study_manifest["study_id"]), cells)
        if target is not None:
            from .extension import consolidate_extension_tables

            canonical, validation = consolidate_extension_tables(canonical, target)
        else:
            validation = validate_study(entries, runs, cells, canonical)
    validation["allow_incomplete"] = bool(allow_incomplete)
    if not validation["valid"] and not allow_incomplete:
        raise ValueError("study validation failed before SLURM analysis submission")
    if not validation["valid"]:
        validation["complete"] = False
        validation.setdefault("warnings", []).append(
            "aggregation continued under --allow-incomplete"
        )

    recipe, recipe_path = _recipe(study_manifest)
    statistics = _requested_statistics(recipe)
    settings = _resampling(recipe)
    from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
        ROUND_ANALYSIS_STATISTICS,
    )

    unknown = sorted(set(statistics) - set(ROUND_ANALYSIS_STATISTICS))
    if unknown:
        raise ValueError(
            "unknown study information estimator(s): " + ", ".join(unknown)
        )
    theoretical_reference = recipe.get(
        "theoretical_reference", "single_affinity_revised"
    )
    if theoretical_reference not in {"single_affinity_revised", "none"}:
        raise ValueError(
            "analysis theoretical_reference must be single_affinity_revised or none"
        )
    events = _round_events_from_canonical(canonical["rounds"])
    if theoretical_reference != "none" and any(
        event.event.get("record_type") == "relational_imitation_round_feedback"
        and (
            float(event.event.get("epistemic_persistence", 1.0)) < 1.0
            or event.event.get("social_mode", "peer") == "board"
        )
        for event in events
    ):
        raise ValueError(
            "analysis theoretical_reference must be none for finite-memory studies"
        )
    input_identity = _scientific_identity(entries, cells, canonical)
    prior_analysis_manifest = root / "analysis" / "analysis_manifest.json"
    if reused_retained and prior_analysis_manifest.is_file():
        input_identity = str(
            _read(prior_analysis_manifest).get(
                "scientific_input_identity", input_identity
            )
        )
    recipe_hash = (
        file_sha256(recipe_path)
        if recipe_path is not None
        else canonical_hash(recipe)
    )
    generation_id = canonical_hash(
        {
            "study_id": study_manifest["study_id"],
            "scientific_input_identity": input_identity,
            "analysis_recipe_hash": recipe_hash,
            "allow_incomplete": bool(allow_incomplete),
            "schema_version": 1,
        }
    )[:20]
    generation = root / "analysis" / ".work" / generation_id
    existing_manifest_path = generation / "execution_manifest.json"
    existing_manifest = (
        _read(existing_manifest_path) if existing_manifest_path.is_file() else None
    )
    input_dir = generation / "input"
    groups_dir = generation / "groups"
    final_dir = generation / "final"
    input_dir.mkdir(parents=True, exist_ok=True)
    groups_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in canonical.items():
        destination = input_dir / f"{name}.parquet"
        if destination.is_file():
            prior_input = (existing_manifest or {}).get("canonical_inputs", {}).get(name)
            if (
                not isinstance(prior_input, Mapping)
                or prior_input.get("sha256") != file_sha256(destination)
                or (existing_manifest or {}).get("scientific_input_identity")
                != input_identity
            ):
                raise ValueError(
                    f"existing aggregation generation has an invalid input: {destination}"
                )
        else:
            retained = retained_paths.get(name) if reused_retained else None
            if retained is not None and retained.suffix == ".parquet":
                try:
                    os.link(retained, destination)
                except OSError:
                    shutil.copy2(retained, destination)
            else:
                temporary_dir = generation / f".input-{name}-{os.getpid()}"
                written = write_scientific_table(temporary_dir, name, frame)
                written.replace(destination)
                shutil.rmtree(temporary_dir, ignore_errors=True)
    _atomic_json(input_dir / "validation.json", validation)

    grouped = sorted({str(event.cell_id) for event in events}) if statistics else []
    source_run_ids = {
        str(row["cell_id"]): str(row.get("source_run_id", row["cell_id"]))
        for row in canonical["cells"].to_dict(orient="records")
    }
    policy = recipe.get("slurm", recipe.get("execution", {}))
    if not isinstance(policy, Mapping):
        raise ValueError("analysis SLURM execution policy must be a mapping")
    throttle = int(policy.get("task_throttle", policy.get("workers", 12)))
    cpus = int(policy.get("cpus_per_task", 1))
    memory = str(policy.get("memory", "8G"))
    time_limit = str(policy.get("time_limit", "06:00:00"))
    prepare_memory = str(policy.get("prepare_memory", memory))
    prepare_time = str(policy.get("prepare_time_limit", "01:00:00"))
    finalizer_memory = str(policy.get("finalizer_memory", memory))
    finalizer_time = str(policy.get("finalizer_time_limit", time_limit))
    if throttle < 1 or cpus < 1:
        raise ValueError("analysis task throttle and CPUs per task must be positive")
    analysis_hash = canonical_hash(
        {
            "scientific_input_identity": input_identity,
            "estimator": "existing_round_information_analysis",
            "estimator_version": "round-feedback-v1",
            "statistics": statistics,
            "settings": settings,
            "theoretical_reference": theoretical_reference,
            "theory_provenance": dict(theory_provenance),
        }
    )
    groups = []
    for index, cell_id in enumerate(grouped):
        stem = hashlib.sha256(cell_id.encode("utf-8")).hexdigest()
        groups.append(
            {
                "group_index": index,
                "cell_id": cell_id,
                "group_hash": stem,
                "seed": int(settings["seed"]) + int(stem[:8], 16),
                "source_run_id": source_run_ids.get(cell_id, cell_id),
                "information_path": str(groups_dir / f"{stem}.information.parquet"),
                "support_path": str(groups_dir / f"{stem}.support.parquet"),
                "completion_path": str(groups_dir / f"{stem}.complete.json"),
            }
        )
    manifest = {
        "schema_version": 1,
        "generation_id": generation_id,
        "created_at": _now(),
        "study_id": str(study_manifest["study_id"]),
        "study_dir": str(root),
        "allow_incomplete": bool(allow_incomplete),
        "scientific_input_identity": input_identity,
        "analysis_recipe_hash": recipe_hash,
        "analysis_recipe_path": None if recipe_path is None else str(recipe_path),
        "analysis_hash": analysis_hash,
        "statistics": list(statistics),
        "resampling": settings,
        "canonical_inputs": {
            name: {
                "path": str(input_dir / f"{name}.parquet"),
                "sha256": file_sha256(input_dir / f"{name}.parquet"),
            }
            for name in canonical
        },
        "config_inputs": [
            {"path": str(entry.config_path), "sha256": str(entry.config_hash)}
            for entry in entries
        ],
        "groups": groups,
        "resources": {
            "task_throttle": min(throttle, max(1, len(groups))),
            "cpus_per_task": cpus,
            "memory_per_task": memory,
            "time_limit": time_limit,
            "peak_cpus": min(throttle, len(groups)) * cpus,
            "prepare": {"cpus": 1, "memory": prepare_memory, "time_limit": prepare_time},
            "finalizer": {"cpus": 1, "memory": finalizer_memory, "time_limit": finalizer_time},
        },
        "jobs": {"prepare": None, "array": None, "finalizer": None},
        "progress_path": str(generation / "progress.json"),
        "final_archive": str(root / "analysis" / f"{study_manifest['study_id']}_analysis.zip"),
    }
    manifest_path = existing_manifest_path
    if not manifest_path.is_file():
        _atomic_json(manifest_path, manifest)
    else:
        prior = _read(manifest_path)
        prior.update({key: value for key, value in manifest.items() if key != "jobs"})
        manifest = prior
        _atomic_json(manifest_path, manifest)
    refresh_progress(manifest_path, stage="planned")
    return manifest_path, manifest


def _valid_group(group: Mapping[str, Any]) -> bool:
    completion = Path(str(group["completion_path"]))
    if not completion.is_file():
        return False
    try:
        value = _read(completion)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        value.get("cell_id") == group["cell_id"]
        and value.get("group_hash") == group.get("group_hash")
        and Path(str(group["information_path"])).is_file()
        and Path(str(group["support_path"])).is_file()
        and file_sha256(Path(str(group["information_path"])))
        == value.get("information_sha256")
        and file_sha256(Path(str(group["support_path"])))
        == value.get("support_sha256")
    )


def refresh_progress(manifest_path: Path, *, stage: str, failed_group: str | None = None) -> None:
    manifest = _read(manifest_path)
    groups = manifest["groups"]
    completed = sum(_valid_group(group) for group in groups)
    running = sum(
        Path(str(group["completion_path"])).with_suffix(".running.json").is_file()
        for group in groups
    )
    failed = sorted(
        str(group["cell_id"])
        for group in groups
        if Path(str(group["completion_path"])).with_suffix(".failed.json").is_file()
    )
    if failed_group is not None and failed_group not in failed:
        failed.append(failed_group)
    created = datetime.fromisoformat(str(manifest["created_at"]).replace("Z", "+00:00"))
    elapsed = max(0.0, (datetime.now(timezone.utc) - created).total_seconds())
    job_states = {
        "prepare": (
            "COMPLETED"
            if stage in {"prepared", "information_resampling", "failed", "published"}
            else "SUBMITTED"
        ),
        "array": "FAILED" if stage == "failed" else (
            "COMPLETED" if completed == len(groups) else "RUNNING" if running else "PENDING"
        ),
        "finalizer": "COMPLETED" if stage == "published" else "PENDING",
    }
    _atomic_json(
        Path(str(manifest["progress_path"])),
        {
            "generation_id": manifest["generation_id"],
            "scientific_input_identity": manifest["scientific_input_identity"],
            "analysis_recipe_hash": manifest["analysis_recipe_hash"],
            "stage": stage,
            "jobs": manifest.get("jobs", {}),
            "job_states": job_states,
            "total_groups": len(groups),
            "pending_groups": max(0, len(groups) - completed - running),
            "running_groups": running,
            "completed_groups": completed,
            "failed_groups": failed,
            "updated_at": _now(),
            "elapsed_seconds": elapsed,
            "published": stage == "published",
            "final_archive": manifest["final_archive"],
        },
    )


def prepare(manifest_path: Path) -> None:
    started = time.monotonic()
    manifest = _read(manifest_path)
    for item in manifest["canonical_inputs"].values():
        path = Path(str(item["path"]))
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise ValueError(f"canonical aggregation input changed: {path}")
    for item in manifest.get("config_inputs", ()):
        path = Path(str(item["path"]))
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise ValueError(f"study config changed after planning: {path}")
    recipe_path = manifest.get("analysis_recipe_path")
    if recipe_path is not None:
        path = Path(str(recipe_path))
        if not path.is_file() or file_sha256(path) != manifest["analysis_recipe_hash"]:
            raise ValueError(f"analysis recipe changed after planning: {path}")
    metrics_path = manifest_path.parent / "prepare_metrics.json"
    if not metrics_path.is_file():
        _atomic_json(
            metrics_path,
            {
                "duration_seconds": time.monotonic() - started,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "completed_at": _now(),
            },
        )
    refresh_progress(manifest_path, stage="prepared")


def run_group(manifest_path: Path, group_index: int) -> None:
    from .aggregation import _information_tables, _round_events_from_canonical

    manifest = _read(manifest_path)
    matches = [
        group
        for group in manifest["groups"]
        if int(group["group_index"]) == group_index
    ]
    if len(matches) != 1:
        raise ValueError(f"invalid information group index: {group_index}")
    group = matches[0]
    if _valid_group(group):
        refresh_progress(manifest_path, stage="information_resampling")
        return
    rounds = read_scientific_table(manifest["canonical_inputs"]["rounds"]["path"])
    events = [
        event for event in _round_events_from_canonical(rounds)
        if str(event.cell_id) == str(group["cell_id"])
    ]
    source_ids = {str(group["cell_id"]): str(group["source_run_id"])}
    running_path = Path(str(group["completion_path"])).with_suffix(".running.json")
    started = time.monotonic()
    _atomic_json(running_path, {"cell_id": group["cell_id"], "started_at": _now()})
    try:
        information, support = _information_tables(
            str(manifest["study_id"]),
            events,
            tuple(map(str, manifest["statistics"])),
            manifest["resampling"],
            str(manifest["analysis_hash"]),
            source_ids,
            workers=1,
        )
        destination = Path(str(group["information_path"]))
        temporary = destination.parent / f".group-{group_index}-{os.getpid()}"
        info_path = write_scientific_table(temporary, "information", information)
        support_path = write_scientific_table(temporary, "support", support)
        info_path.replace(destination)
        support_path.replace(Path(str(group["support_path"])))
        shutil.rmtree(temporary, ignore_errors=True)
        _atomic_json(
            Path(str(group["completion_path"])),
            {
                "cell_id": group["cell_id"],
                "group_index": group_index,
                "group_hash": group["group_hash"],
                "seed": group["seed"],
                "information_sha256": file_sha256(destination),
                "support_sha256": file_sha256(Path(str(group["support_path"]))),
                "duration_seconds": time.monotonic() - started,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "completed_at": _now(),
            },
        )
        Path(str(group["completion_path"])).with_suffix(".failed.json").unlink(
            missing_ok=True
        )
    except Exception as exc:
        _atomic_json(
            Path(str(group["completion_path"])).with_suffix(".failed.json"),
            {"cell_id": group["cell_id"], "failed_at": _now(), "error": str(exc)},
        )
        refresh_progress(manifest_path, stage="failed", failed_group=str(group["cell_id"]))
        raise
    finally:
        running_path.unlink(missing_ok=True)
    refresh_progress(manifest_path, stage="information_resampling")


def finalize(manifest_path: Path) -> None:
    from .aggregation import _aggregate_study_local, _package

    manifest = _read(manifest_path)
    started = time.monotonic()
    prepare(manifest_path)
    missing = [group["cell_id"] for group in manifest["groups"] if not _valid_group(group)]
    if missing:
        raise ValueError("cannot finalize; missing/invalid groups: " + ", ".join(missing))
    generation = manifest_path.parent
    final_dir = generation / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    _aggregate_study_local(
        manifest["study_dir"],
        allow_incomplete=bool(manifest["allow_incomplete"]),
        analysis_output_dir=final_dir,
        canonical_snapshot_dir=generation / "input",
        information_fragments_dir=generation / "groups",
    )
    analysis_manifest_path = final_dir / "analysis_manifest.json"
    analysis_manifest = _read(analysis_manifest_path)
    analysis_manifest["aggregation_execution"] = {
        "backend": "slurm",
        "generation_id": manifest["generation_id"],
        "information_groups": len(manifest["groups"]),
        "resources": manifest["resources"],
        "jobs": manifest.get("jobs", {}),
        "group_measurements": [
            {
                key: completion.get(key)
                for key in ("cell_id", "duration_seconds", "max_rss_kib")
            }
            for group in manifest["groups"]
            for completion in [_read(Path(str(group["completion_path"])))]
        ],
        "prepare_measurement": _read(generation / "prepare_metrics.json"),
        "finalizer_duration_seconds": time.monotonic() - started,
        "completed_at": _now(),
    }
    _atomic_json(analysis_manifest_path, analysis_manifest)
    _package(final_dir, str(manifest["study_id"]))

    root = Path(str(manifest["study_dir"]))
    publish = root / f".analysis-publish-{manifest['generation_id']}"
    previous = root / f".analysis-previous-{manifest['generation_id']}"
    final_dir.replace(publish)
    analysis_dir = root / "analysis"
    analysis_dir.replace(previous)
    try:
        publish.replace(analysis_dir)
    except Exception:
        previous.replace(analysis_dir)
        raise
    shutil.rmtree(previous, ignore_errors=True)


def submit_aggregation(
    study_dir: str | Path,
    *,
    allow_incomplete: bool = False,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    root = Path(study_dir).expanduser().resolve()
    if not (root / "study_manifest.json").is_file() or not (
        root / "submission_manifest.csv"
    ).is_file():
        raise ValueError(f"not a submitted MA-CC study directory: {root}")
    manifest_path, manifest = create_generation(root, allow_incomplete=allow_incomplete)
    launcher = LAUNCHER
    if not launcher.is_file():
        raise ValueError(f"missing generic study-analysis launcher: {launcher}")
    logs = root / "logs" / f"analysis-{manifest['generation_id']}"
    logs.mkdir(parents=True, exist_ok=True)
    resources = manifest["resources"]

    def submit(arguments: list[str]) -> str:
        try:
            completed = run(arguments, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise ValueError(f"SLURM aggregation submission failed: {detail}") from exc
        return _job_id(completed.stdout)

    prepare_id = submit(
        [
            "sbatch",
            "--cpus-per-task=1",
            f"--mem={resources['prepare']['memory']}",
            f"--time={resources['prepare']['time_limit']}",
            f"--output={logs}/prepare-%j.out",
            f"--error={logs}/prepare-%j.err",
            str(launcher),
            "prepare",
            str(manifest_path),
        ]
    )
    manifest["jobs"] = {"prepare": prepare_id, "array": None, "finalizer": None}
    _atomic_json(manifest_path, manifest)
    pending = [
        int(group["group_index"])
        for group in manifest["groups"]
        if not _valid_group(group)
    ]
    array_id = None
    dependency = prepare_id
    if pending:
        array_spec = ",".join(map(str, pending)) + f"%{resources['task_throttle']}"
        array_id = submit(
            [
                "sbatch",
                f"--dependency=afterok:{prepare_id}",
                f"--array={array_spec}",
                f"--cpus-per-task={resources['cpus_per_task']}",
                f"--mem={resources['memory_per_task']}",
                f"--time={resources['time_limit']}",
                f"--output={logs}/group-%A_%a.out",
                f"--error={logs}/group-%A_%a.err",
                str(launcher),
                "group",
                str(manifest_path),
            ]
        )
        dependency = array_id
        manifest["jobs"]["array"] = array_id
        _atomic_json(manifest_path, manifest)
    finalizer_id = submit(
        [
            "sbatch",
            f"--dependency=afterok:{dependency}",
            "--cpus-per-task=1",
            f"--mem={resources['finalizer']['memory']}",
            f"--time={resources['finalizer']['time_limit']}",
            f"--output={logs}/finalize-%j.out",
            f"--error={logs}/finalize-%j.err",
            str(launcher),
            "finalize",
            str(manifest_path),
        ]
    )
    manifest["jobs"]["finalizer"] = finalizer_id
    _atomic_json(manifest_path, manifest)
    refresh_progress(manifest_path, stage="submitted")
    return {
        "study_id": manifest["study_id"],
        "study_dir": str(root),
        "analysis_dir": str(root / "analysis"),
        "submitted": True,
        "complete": False,
        "archive": manifest["final_archive"],
        "generation_id": manifest["generation_id"],
        "progress": manifest["progress_path"],
        "jobs": manifest["jobs"],
        "groups": len(manifest["groups"]),
        "pending_groups": len(pending),
        "resources": resources,
    }


__all__ = [
    "create_generation",
    "finalize",
    "prepare",
    "refresh_progress",
    "run_group",
    "submit_aggregation",
]
