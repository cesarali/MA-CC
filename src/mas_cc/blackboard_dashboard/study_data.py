"""Compact, read-only catalog for standardized blackboard studies."""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.core.random import Seed
from mas_cc.studies.discovery import DiscoveredCell, discover_cells, discover_runs
from mas_cc.studies.execution import read_execution_manifest
from mas_cc.studies.submission import read_submission_manifest
from mas_cc.storage.scientific import validate_cell_artifact

from .data import BlackboardRunReader, _event, _jsonl, _safe_json


_TERMINAL_COMPLETE = {"completed", "skipped_resumed"}
_TERMINAL_FAILED = {"failed"}
_TERMINAL_ABORTED = {"aborted", "skipped_aborted"}
_STATUS_ORDER = ("pending", "running", "completed", "failed", "aborted", "unknown")
_JOB_ID = re.compile(r"^[0-9]+(?:_[0-9]+)?$")


@dataclass(frozen=True, slots=True)
class StudyDescriptor:
    study_id: str
    study_root: str
    expected_config_count: int
    expected_cell_count: int
    expected_episode_count: int
    submission_status: str | None
    job_id: str | None


@dataclass(frozen=True, slots=True)
class CellDescriptor:
    qualified_id: str
    config_index: int
    config_name: str
    cell_id: str
    path: str | None
    expected_episodes: int
    parameters: Mapping[str, Any]
    scheduler_array_index: int | None


@dataclass(frozen=True, slots=True)
class EpisodeDescriptor:
    qualified_id: str
    cell_id: str
    episode_id: str
    repetition_index: int
    seed: int | None
    status: str
    current_round: int | None
    current_update: int | None
    elapsed_seconds: float | None
    detail_available: bool
    detail_reason: str | None


@dataclass(frozen=True, slots=True)
class VoteSeries:
    episode_id: str
    points: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    available: bool
    job_id: str | None
    refreshed_at: str
    tasks: Mapping[int, Mapping[str, Any]]
    error: str | None = None


def is_study_root(path: str | Path) -> bool:
    root = Path(path).expanduser().resolve()
    return (root / "study_manifest.json").is_file() and (
        (root / "execution_manifest.csv").is_file()
        or (root / "submission_manifest.csv").is_file()
    )


def _nested(mapping: Mapping[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _parameters(
    config: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    def flatten(value: Any, prefix: str = "") -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                flatten(nested, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[prefix] = value

    flatten(config)
    result.update({str(key): value for key, value in overrides.items()})
    for dotted, label in (
        ("experiment.metadata.arm", "controller_condition"),
        ("experiment.metadata.study", "experiment_block"),
        ("game.options.epistemic_persistence", "rho"),
        ("control.options.intervention_budget", "b"),
        ("game.options.task_id", "task_id"),
        ("game.population_size", "population_size"),
        ("game.options.social_group_size", "social_group_size"),
        ("control.options.sensor_sample_size", "sensor_sample_size"),
        ("control.options.beta", "beta"),
        ("control.options.threshold", "threshold"),
        ("control.options.target", "controller_target"),
        ("experiment.metadata.ground_truth", "ground_truth"),
        ("experiment.metadata.target_semantics", "target_semantics"),
    ):
        value = _nested(config, dotted)
        if value is not None:
            result.setdefault(label, value)
    leaves: dict[str, list[str]] = {}
    for path in tuple(result):
        leaves.setdefault(path.rsplit(".", 1)[-1], []).append(path)
    for leaf, paths in leaves.items():
        if len(paths) == 1:
            result.setdefault(leaf, result[paths[0]])
    return result


def _signature(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return str(path), stat.st_mtime_ns, stat.st_size


def _timestamp_seconds(value: Any) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _counts(row: Mapping[str, Any], suffix: str) -> dict[str, int]:
    direct = row.get(f"occupation_counts_{suffix}")
    if isinstance(direct, Mapping):
        return {str(key): int(value) for key, value in direct.items()}
    values = row.get(f"population_state_{suffix}")
    if isinstance(values, (list, tuple)):
        return dict(Counter(str(value) for value in values))
    return {}


def _vote_point(
    row: Mapping[str, Any],
    *,
    phase: str,
    round_index: int | None,
    suffix: str,
    complete: bool,
) -> dict[str, Any]:
    counts = _counts(row, suffix)
    total = sum(counts.values())
    options = row.get("possible_answers")
    if not isinstance(options, (list, tuple)):
        options = sorted(counts)
    shares = {
        str(option): (counts.get(str(option), 0) / total if total else None)
        for option in options
    }
    truth = row.get("correct_answer")
    target = row.get("controller_target")
    truth_share = row.get(
        "truth_vote_share_before" if suffix == "before" else "truth_vote_share"
    )
    target_share = row.get(
        "controller_target_share_before"
        if suffix == "before"
        else "controller_target_share"
    )
    if truth_share is None and truth is not None and total:
        truth_share = counts.get(str(truth), 0) / total
    if target_share is None and target is not None and total:
        target_share = counts.get(str(target), 0) / total
    return {
        "phase": phase,
        "round_index": round_index,
        "truth_option": truth,
        "controller_target": target,
        "truth_share": truth_share,
        "controller_target_share": target_share,
        "option_counts": counts,
        "option_shares": shares,
        "n_agents": total or row.get("N"),
        "record_complete": complete,
    }


class _SchedulerReader:
    def __init__(self, job_id: str | None, ttl_seconds: float = 10.0) -> None:
        self.job_id = job_id if job_id and _JOB_ID.fullmatch(job_id) else None
        self.ttl_seconds = ttl_seconds
        self._loaded_at = 0.0
        self._snapshot: SchedulerSnapshot | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _state(value: str) -> str:
        state = value.upper().split("+", 1)[0]
        if state in {"RUNNING", "COMPLETING"}:
            return "running"
        if state in {
            "PENDING",
            "CONFIGURING",
            "SUSPENDED",
            "RESV_DEL_HOLD",
            "REQUEUE_HOLD",
        }:
            return "pending" if "HOLD" not in state and state != "SUSPENDED" else "held"
        return state.lower() or "unknown"

    def snapshot(self) -> SchedulerSnapshot:
        now = time.monotonic()
        with self._lock:
            if self._snapshot is not None and now - self._loaded_at < self.ttl_seconds:
                return self._snapshot
            refreshed = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            if self.job_id is None:
                result = SchedulerSnapshot(
                    False, None, refreshed, {}, "job ID unavailable"
                )
            else:
                result = self._query(refreshed)
            self._snapshot = result
            self._loaded_at = now
            return result

    def _query(self, refreshed: str) -> SchedulerSnapshot:
        assert self.job_id is not None
        commands = []
        if shutil.which("squeue"):
            commands.append(["squeue", "-h", "-j", self.job_id, "-o", "%A|%a|%T|%N|%M"])
        if shutil.which("sacct"):
            commands.append(
                [
                    "sacct",
                    "-n",
                    "-X",
                    "-j",
                    self.job_id,
                    "--parsable2",
                    "--format=JobIDRaw,State,NodeList,Elapsed",
                ]
            )
        if not commands:
            return SchedulerSnapshot(
                False, self.job_id, refreshed, {}, "squeue and sacct unavailable"
            )
        tasks: dict[int, dict[str, Any]] = {}
        errors = []
        for command in commands:
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=3, check=False
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(str(exc))
                continue
            if completed.returncode:
                errors.append(
                    completed.stderr.strip()
                    or f"{command[0]} exited {completed.returncode}"
                )
                continue
            for line in completed.stdout.splitlines():
                fields = [item.strip() for item in line.split("|")]
                if command[0] == "squeue" and len(fields) >= 5:
                    array = fields[1]
                    state, node, elapsed = fields[2], fields[3], fields[4]
                elif len(fields) >= 4:
                    job = fields[0].split(".", 1)[0]
                    match = re.fullmatch(r"[0-9]+_([0-9]+)", job)
                    if match is None:
                        continue
                    array = match.group(1)
                    state, node, elapsed = fields[1], fields[2], fields[3]
                else:
                    continue
                if not array.isdigit():
                    continue
                tasks[int(array)] = {
                    "array_index": int(array),
                    "state": self._state(state),
                    "raw_state": state,
                    "node": node or None,
                    "elapsed": elapsed or None,
                }
            if tasks:
                break
        return SchedulerSnapshot(
            bool(tasks), self.job_id, refreshed, tasks, "; ".join(errors) or None
        )


class BlackboardStudyReader:
    """Discover expected cells once and refresh only compact live artifacts."""

    def __init__(self, study_dir: str | Path, *, scheduler: bool = True) -> None:
        self.study_dir = Path(study_dir).expanduser().resolve()
        if not is_study_root(self.study_dir):
            raise ValueError(f"not a standardized study root: {self.study_dir}")
        self.manifest = _safe_json(
            self.study_dir / "study_manifest.json", required=True
        )
        schema = int(self.manifest.get("schema_version", 0))
        if schema != 1:
            raise ValueError(f"unsupported study manifest schema version: {schema}")
        submission_path = self.study_dir / "submission_manifest.csv"
        if not submission_path.is_file():
            raise ValueError(f"required study artifact is missing: {submission_path}")
        self.submissions = read_submission_manifest(submission_path)
        self.executions = (
            read_execution_manifest(self.study_dir / "execution_manifest.csv")
            if (self.study_dir / "execution_manifest.csv").is_file()
            else ()
        )
        self.submission = _safe_json(self.study_dir / "submission.json")
        job = self.submission.get("job_id")
        self._scheduler = (
            _SchedulerReader(str(job) if job is not None else None)
            if scheduler
            else _SchedulerReader(None)
        )
        self._lock = threading.RLock()
        self._jsonl_cache: dict[
            Path, tuple[tuple[str, int, int] | None, list[dict[str, Any]]]
        ] = {}
        self._seal_cache: dict[
            Path, tuple[tuple[Any, ...], bool, str | None, set[str]]
        ] = {}
        self._last_trajectory_signatures: dict[Path, tuple[str, int, int] | None] = {}
        self._resolved_configs: dict[str, Mapping[str, Any]] = {}
        self._cells = self._build_cells()
        self._cell_map = {cell.qualified_id: cell for cell in self._cells}

    def _build_cells(self) -> tuple[CellDescriptor, ...]:
        discovered: dict[tuple[int, str], DiscoveredCell] = {}
        for cell in discover_cells(discover_runs(self.submissions)):
            discovered.setdefault(
                (cell.run.entry.array_index, cell.local_cell_id), cell
            )
        execution_by_cell = {
            (row.config_index, row.cell_id): row for row in self.executions
        }
        descriptors = []
        for config_index, entry in enumerate(self.submissions):
            config_name = Path(entry.config_path).stem
            try:
                source = load_run_config_or_grid(entry.config_path)
            except (OSError, ValueError):
                source = None
            expected = source.cells if isinstance(source, GridSpec) else ()
            if expected:
                specs = [
                    (cell.cell_id, cell.config.to_dict(), dict(cell.overrides))
                    for cell in expected
                ]
            elif source is not None:
                specs = [("run", source.to_dict(), {})]
            else:
                local_ids = [
                    row.cell_id
                    for row in self.executions
                    if row.config_index == config_index
                ]
                if not local_ids:
                    local_ids = sorted(
                        local_id
                        for index, local_id in discovered
                        if index == config_index
                    )
                repetitions = entry.expected_episode_count // max(
                    1, entry.expected_cell_count
                )
                specs = []
                for local_id in local_ids:
                    actual = discovered.get((config_index, local_id))
                    config = (
                        dict(actual.resolved_config)
                        if actual is not None
                        else {
                            "execution": {
                                "repetitions": repetitions,
                                "seed": entry.execution_seed,
                            }
                        }
                    )
                    overrides = dict(actual.overrides) if actual is not None else {}
                    specs.append((local_id, config, overrides))
            for local_id, config, overrides in specs:
                actual = discovered.get((config_index, local_id))
                execution = execution_by_cell.get((config_index, local_id))
                qualified = f"config-{config_index:04d}~{local_id}"
                repetitions = int(_nested(config, "execution.repetitions") or 0)
                descriptors.append(
                    CellDescriptor(
                        qualified_id=qualified,
                        config_index=config_index,
                        config_name=config_name,
                        cell_id=local_id,
                        path=str(actual.path) if actual is not None else None,
                        expected_episodes=repetitions,
                        parameters=_parameters(config, overrides),
                        scheduler_array_index=execution.array_index
                        if execution
                        else entry.array_index,
                    )
                )
                self._resolved_configs[qualified] = config
        return tuple(descriptors)

    def _rows(self, path: Path, *, completed: bool = False) -> list[dict[str, Any]]:
        signature = _signature(path)
        cached = self._jsonl_cache.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
        rows = [_event(row) for row in _jsonl(path, completed=completed)]
        self._jsonl_cache[path] = (signature, rows)
        return rows

    def _seal(self, cell: CellDescriptor) -> tuple[bool, str | None, set[str]]:
        if cell.path is None:
            return False, None, set()
        root = Path(cell.path)
        seal_path, table_path = (
            root / "cell_complete.json",
            root / "scientific_events.parquet",
        )
        signature = (_signature(seal_path), _signature(table_path))
        cached = self._seal_cache.get(root)
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2], set(cached[3])
        if not seal_path.is_file():
            result = (False, None, set())
        else:
            try:
                validate_cell_artifact(root)
                seal = _safe_json(seal_path, required=True)
                result = (
                    True,
                    None,
                    {str(value) for value in seal.get("episode_ids", [])},
                )
            except ValueError as exc:
                result = (False, str(exc), set())
        self._seal_cache[root] = (signature, *result)
        return result

    @staticmethod
    def _planned_seeds(
        cell: CellDescriptor, config: Mapping[str, Any]
    ) -> dict[int, int]:
        base = int(_nested(config, "execution.seed") or 0)
        common = bool(
            _nested(config, "experiment.metadata.common_random_numbers_across_grid")
        )
        match = re.search(r"(\d+)$", cell.cell_id)
        index = int(match.group(1)) if match else 0
        cell_seed = Seed(base) if common else Seed(base).derive(f"grid-cell:{index}")
        return {
            i: int(cell_seed.derive(f"episode:{i}"))
            for i in range(cell.expected_episodes)
        }

    def _episode_records(
        self, cell: CellDescriptor
    ) -> tuple[list[EpisodeDescriptor], dict[str, VoteSeries]]:
        config = self._resolved_configs[cell.qualified_id]
        seeds = self._planned_seeds(cell, config)
        plan = next(
            (
                row
                for row in self.executions
                if row.config_index == cell.config_index and row.cell_id == cell.cell_id
            ),
            None,
        )
        if plan and plan.episode_plan_path and Path(plan.episode_plan_path).is_file():
            with Path(plan.episode_plan_path).open(
                newline="", encoding="utf-8"
            ) as stream:
                for row in csv.DictReader(stream):
                    seeds[int(row["repetition_index"])] = int(row["episode_seed"])
        sealed, seal_error, sealed_ids = self._seal(cell)
        root = Path(cell.path) if cell.path else None
        summary = _safe_json(root / "cell_summary.json") if root else {}
        outcomes = summary.get("outcomes", [])
        failures = summary.get("failures", [])
        explicit: dict[str, Mapping[str, Any]] = {}
        for value in (*outcomes, *failures):
            if isinstance(value, Mapping) and value.get("episode_id"):
                explicit[str(value["episode_id"])] = value
        scheduler = self._scheduler.snapshot()
        scheduler_task = scheduler.tasks.get(cell.scheduler_array_index or -1, {})
        episodes: list[EpisodeDescriptor] = []
        series: dict[str, VoteSeries] = {}
        for repetition in range(cell.expected_episodes):
            local_id = (
                f"{cell.cell_id}-{repetition:04d}"
                if cell.cell_id != "run"
                else f"episode-{repetition:04d}"
            )
            round_path = (
                root / "round_records" / local_id / "round_trajectory.jsonl"
                if root
                else None
            )
            full_path = root / "data" / "episodes" / local_id if root else None
            compact_path = (
                root / ".resume" / local_id / "scientific_events.parquet"
                if root
                else None
            )
            full_manifest = _safe_json(full_path / "manifest.json") if full_path else {}
            record = explicit.get(local_id, full_manifest)
            raw_status = str(record.get("status", ""))
            trajectory_exists = bool(round_path and round_path.is_file())
            completed = sealed and local_id in sealed_ids
            status = (
                "completed"
                if completed or raw_status in _TERMINAL_COMPLETE
                else (
                    "failed"
                    if raw_status in _TERMINAL_FAILED
                    else "aborted"
                    if raw_status in _TERMINAL_ABORTED
                    else "pending"
                )
            )
            if seal_error and (
                trajectory_exists or compact_path and compact_path.is_file()
            ):
                status = "unknown"
            elif status == "pending" and trajectory_exists:
                current = _signature(round_path)
                previous = self._last_trajectory_signatures.get(round_path)
                if previous is not None and current != previous:
                    status = "running"
                elif scheduler_task.get("state") == "running":
                    status = "running"
                else:
                    status = "unknown"
                self._last_trajectory_signatures[round_path] = current
            rows = (
                self._rows(round_path, completed=status == "completed")
                if trajectory_exists
                else []
            )
            if (
                status == "completed"
                and compact_path
                and compact_path.is_file()
                and not sealed
            ):
                status = "unknown"
            points: list[Mapping[str, Any]] = []
            if rows:
                points.append(
                    _vote_point(
                        rows[0],
                        phase="initialization",
                        round_index=None,
                        suffix="before",
                        complete=status == "completed",
                    )
                )
                points.extend(
                    _vote_point(
                        row,
                        phase="round",
                        round_index=int(row.get("round_index", index)),
                        suffix="after",
                        complete=status == "completed",
                    )
                    for index, row in enumerate(rows)
                )
            qualified = f"{cell.qualified_id}~episode-{repetition:04d}"
            series[qualified] = VoteSeries(qualified, tuple(points))
            started = _timestamp_seconds(record.get("started_at"))
            finished = _timestamp_seconds(record.get("finished_at"))
            elapsed = (
                (finished - started)
                if started is not None and finished is not None
                else None
            )
            last = rows[-1] if rows else {}
            detail_available = bool(
                full_path and (full_path / "trajectory.jsonl").is_file()
            )
            episodes.append(
                EpisodeDescriptor(
                    qualified_id=qualified,
                    cell_id=cell.qualified_id,
                    episode_id=local_id,
                    repetition_index=repetition,
                    seed=record.get("seed", seeds.get(repetition)),
                    status=status,
                    current_round=int(last["round_index"])
                    if "round_index" in last
                    else None,
                    current_update=int(
                        last.get("global_update_index", last.get("within_round_index"))
                    )
                    if last
                    and ("global_update_index" in last or "within_round_index" in last)
                    else None,
                    elapsed_seconds=elapsed,
                    detail_available=detail_available,
                    detail_reason=None
                    if detail_available
                    else "Full prompts and microscopic updates were not retained for this episode.",
                )
            )
        return episodes, series

    def _cell_payload(
        self, cell: CellDescriptor, *, include_votes: bool = True
    ) -> dict[str, Any]:
        episodes, votes = self._episode_records(cell)
        counts = {
            name: sum(item.status == name for item in episodes)
            for name in _STATUS_ORDER
        }
        active = [item for item in episodes if item.status == "running"]
        scheduler_index = (
            cell.scheduler_array_index if cell.scheduler_array_index is not None else -1
        )
        scheduler = self._scheduler.snapshot().tasks.get(scheduler_index)
        groups: dict[tuple[str, int | None], list[Mapping[str, Any]]] = {}
        for episode_id, vote_series in votes.items():
            if not vote_series.points:
                continue
            for point in vote_series.points:
                groups.setdefault(
                    (str(point["phase"]), point["round_index"]), []
                ).append(point)
        mean = []
        for (phase, round_index), points in sorted(
            groups.items(),
            key=lambda item: (-1 if item[0][1] is None else int(item[0][1])),
        ):
            truth = [
                float(point["truth_share"])
                for point in points
                if point.get("truth_share") is not None
            ]
            target = [
                float(point["controller_target_share"])
                for point in points
                if point.get("controller_target_share") is not None
            ]
            mean.append(
                {
                    "phase": phase,
                    "round_index": round_index,
                    "truth_share": sum(truth) / len(truth) if truth else None,
                    "controller_target_share": sum(target) / len(target)
                    if target
                    else None,
                    "episodes_available": len(points),
                    "episodes_expected": cell.expected_episodes,
                }
            )
        return {
            **asdict(cell),
            "discovered": cell.path is not None,
            "episodes_discovered": sum(item.status != "pending" for item in episodes),
            "status_counts": counts,
            "status": "failed"
            if counts["failed"]
            else "aborted"
            if counts["aborted"]
            else "running"
            if counts["running"]
            else "unknown"
            if counts["unknown"]
            else "completed"
            if counts["completed"] == cell.expected_episodes and cell.expected_episodes
            else "pending",
            "current_round": max(
                (
                    item.current_round
                    for item in active
                    if item.current_round is not None
                ),
                default=None,
            ),
            "current_update": max(
                (
                    item.current_update
                    for item in active
                    if item.current_update is not None
                ),
                default=None,
            ),
            "scheduler": scheduler,
            "episodes": [asdict(item) for item in episodes],
            "vote_preview": mean[:: max(1, len(mean) // 12)]
            if len(mean) > 12
            else mean,
            "mean_vote_series": mean,
            **(
                {"vote_series": {key: asdict(value) for key, value in votes.items()}}
                if include_votes
                else {}
            ),
        }

    def study(self) -> dict[str, Any]:
        with self._lock:
            cells = [
                self._cell_payload(cell, include_votes=False) for cell in self._cells
            ]
            totals = {
                name: sum(cell["status_counts"][name] for cell in cells)
                for name in _STATUS_ORDER
            }
            scheduler = self._scheduler.snapshot()
            descriptor = StudyDescriptor(
                study_id=str(self.manifest.get("study_id", self.study_dir.name)),
                study_root=str(self.study_dir),
                expected_config_count=int(
                    self.manifest.get("expected_config_count", len(self.submissions))
                ),
                expected_cell_count=int(
                    self.manifest.get("expected_cell_count", len(self._cells))
                ),
                expected_episode_count=int(
                    self.manifest.get(
                        "expected_episode_count",
                        sum(cell.expected_episodes for cell in self._cells),
                    )
                ),
                submission_status=self.submission.get("status"),
                job_id=scheduler.job_id,
            )
            return {
                "schema_version": 1,
                **asdict(descriptor),
                "discovered_cell_count": sum(cell["discovered"] for cell in cells),
                "episode_counts": totals,
                "active_scheduler_tasks": sum(
                    item.get("state") == "running" for item in scheduler.tasks.values()
                ),
                "scheduler": asdict(scheduler),
                "live": bool(totals["running"] or scheduler.tasks),
                "refreshed_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "cells": cells,
            }

    def cells(self) -> dict[str, Any]:
        return self.study()

    def cell(self, qualified_id: str) -> dict[str, Any]:
        with self._lock:
            cell = self._cell_map.get(qualified_id)
            if cell is None:
                raise ValueError("unknown qualified cell identifier")
            payload = self._cell_payload(cell)
            available = sum(
                bool(payload["vote_series"][item["qualified_id"]]["points"])
                for item in payload["episodes"]
            )
            payload["descriptive_mean"] = {
                "label": f"{available}/{cell.expected_episodes} available",
                "complete_episodes": available,
                "expected_episodes": cell.expected_episodes,
            }
            return {"schema_version": 1, **payload}

    def votes(self, qualified_id: str) -> dict[str, Any]:
        payload = self.cell(qualified_id)
        return {
            "schema_version": 1,
            "cell_id": qualified_id,
            "vote_series": payload["vote_series"],
            "descriptive_mean": payload["descriptive_mean"],
        }

    def episode_status(self, qualified_id: str) -> dict[str, Any]:
        cell_id = qualified_id.rsplit("~episode-", 1)[0]
        payload = self.cell(cell_id)
        episode = next(
            (
                item
                for item in payload["episodes"]
                if item["qualified_id"] == qualified_id
            ),
            None,
        )
        if episode is None:
            raise ValueError("unknown qualified episode identifier")
        return {"schema_version": 1, **episode, "scheduler": payload["scheduler"]}

    def episode_reader(self, qualified_id: str) -> BlackboardRunReader:
        status = self.episode_status(qualified_id)
        if not status["detail_available"]:
            raise ValueError(status["detail_reason"])
        cell = self._cell_map[status["cell_id"]]
        assert cell.path is not None
        return BlackboardRunReader(cell.path, status["episode_id"])

    @property
    def descriptor(self) -> StudyDescriptor:
        payload = self.study()
        return StudyDescriptor(
            **{field: payload[field] for field in StudyDescriptor.__dataclass_fields__}
        )


__all__ = [
    "BlackboardStudyReader",
    "CellDescriptor",
    "EpisodeDescriptor",
    "SchedulerSnapshot",
    "StudyDescriptor",
    "VoteSeries",
    "is_study_root",
]
