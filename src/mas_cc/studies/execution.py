"""Execution-only study planning; scientific identities never depend on this module."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.llm_runtime.providers.load_control import ProviderLoadControlConfig

from .manifest import StudySpec
from .submission import SubmissionEntry


EXECUTION_COLUMNS = (
    "array_index", "config_index", "config_path", "cell_index", "cell_id", "output_dir",
    "extension_index", "cell_key", "episode_plan_path", "study_root",
)


@dataclass(frozen=True, slots=True)
class ExecutionEntry:
    array_index: int
    config_index: int
    config_path: str
    cell_index: int
    cell_id: str
    output_dir: str
    extension_index: int = 0
    cell_key: str = ""
    episode_plan_path: str = ""
    study_root: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    mode: str
    shard_count: int
    array_throttle: int
    request_concurrency_per_shard: int
    total_request_concurrency: int
    episode_slots_per_shard: int
    total_episode_slots: int
    target_rpm: int
    assumed_latency_seconds: float
    estimated_rpm: float
    cpus_per_task: int
    memory: str
    time_limit: str
    partition: str
    qos: str
    provider_load_control: dict[str, object]
    graceful_drain: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def graceful_drain_policy(policy: Mapping[str, Any], time_limit: str) -> dict[str, object]:
    raw = dict(policy.get("graceful_drain", {}) or {})
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("graceful_drain.enabled must be a boolean")
    signal_name = str(raw.get("signal", "USR1"))
    lead_time = str(raw.get("lead_time", "04:00:00"))
    def seconds(value: str) -> int:
        days = 0
        if "-" in value:
            day_part, value = value.split("-", 1)
            if not day_part.isdigit():
                raise ValueError("invalid SLURM time or drain lead_time")
            days = int(day_part)
        parts = value.split(":")
        if (len(parts) != 3 or any(not part.isdigit() for part in parts)
                or int(parts[1]) >= 60 or int(parts[2]) >= 60):
            raise ValueError("SLURM time and drain lead_time must be HH:MM:SS")
        return days * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    lead_seconds = seconds(lead_time)
    if enabled and (signal_name != "USR1" or lead_seconds < 1 or lead_seconds >= seconds(time_limit)):
        raise ValueError("graceful_drain requires USR1 and a positive lead_time shorter than time_limit")
    return {"enabled": enabled, "signal": signal_name, "lead_time": lead_time,
            "lead_seconds": lead_seconds}


def build_cell_execution_entries(
    spec: StudySpec, submissions: Sequence[SubmissionEntry]
) -> tuple[ExecutionEntry, ...]:
    cells_per_shard = int(spec.execution.get("cells_per_shard", 1))
    if cells_per_shard < 1:
        raise ValueError("cells_per_shard must be a positive integer")
    rows: list[ExecutionEntry] = []
    for config_index, (path, submission) in enumerate(zip(spec.configs, submissions)):
        source = load_run_config_or_grid(path)
        if not isinstance(source, GridSpec):
            raise ValueError("cell_array execution currently requires grid configs")
        for cell in source.cells:
            rows.append(
                ExecutionEntry(
                    array_index=len(rows) // cells_per_shard,
                    config_index=config_index,
                    config_path=str(path),
                    cell_index=cell.index,
                    cell_id=cell.cell_id,
                    output_dir=str(
                        (Path(submission.output_dir) / "shards" / f"cell-{cell.index:04d}").resolve()
                    ),
                )
            )
    return tuple(rows)


def plan_cell_execution(spec: StudySpec, shard_count: int) -> ExecutionPlan:
    bases = []
    for path in spec.configs:
        source = load_run_config_or_grid(path)
        bases.append(source.base if isinstance(source, GridSpec) else source)
    request_concurrencies = {
        min(
            base.llm_provider.request_concurrency,
            base.execution.parallelism,
            base.execution.repetitions,
        )
        for base in bases
    }
    if len(request_concurrencies) != 1:
        raise ValueError("automatic cell-array planning requires one request concurrency")
    cells_per_shard = int(spec.execution.get("cells_per_shard", 1))
    if cells_per_shard < 1:
        raise ValueError("cells_per_shard must be a positive integer")
    per_shard = request_concurrencies.pop() * cells_per_shard
    episode_slots = max(base.execution.parallelism for base in bases) * cells_per_shard
    policy = spec.execution
    target_rpm = int(policy.get("target_rpm", 900))
    latency = float(policy.get("assumed_latency_seconds", 10.0))
    if target_rpm < 1 or latency <= 0:
        raise ValueError("target_rpm and assumed_latency_seconds must be positive")
    rpm_bound = max(1, math.floor(target_rpm * latency / (60.0 * per_shard)))
    max_nodes = int(policy.get("max_active_nodes", rpm_bound))
    throttle = min(shard_count, rpm_bound, max_nodes)
    configured = policy.get("throttle")
    if configured is not None:
        throttle = min(throttle, int(configured))
    if throttle < 1:
        raise ValueError("planned array throttle must be positive")
    cpus = int(policy.get("cpus_per_task", episode_slots))
    total_concurrency = throttle * per_shard
    control = ProviderLoadControlConfig.from_mapping(
        policy.get("provider_load_control"),
        defaults={
            "initial_concurrency": total_concurrency,
            "minimum_concurrency": min(4, total_concurrency),
            "maximum_concurrency": total_concurrency,
            "target_rpm": target_rpm,
        },
    )
    return ExecutionPlan(
        mode="cell_array",
        shard_count=shard_count,
        array_throttle=throttle,
        request_concurrency_per_shard=per_shard,
        total_request_concurrency=total_concurrency,
        episode_slots_per_shard=episode_slots,
        total_episode_slots=throttle * episode_slots,
        target_rpm=target_rpm,
        assumed_latency_seconds=latency,
        estimated_rpm=throttle * per_shard * 60.0 / latency,
        cpus_per_task=cpus,
        memory=str(policy.get("memory", "8G")),
        time_limit=str(policy.get("time_limit", "04:00:00")),
        partition=str(policy.get("partition", "all")),
        qos=str(policy.get("qos", "normal")),
        provider_load_control=control.to_dict(),
        graceful_drain=graceful_drain_policy(policy, str(policy.get("time_limit", "04:00:00"))),
    )


def plan_config_execution(spec: StudySpec, shard_count: int) -> ExecutionPlan:
    """Plan one array shard per ordinary config under shared load control."""

    bases = []
    for path in spec.configs:
        source = load_run_config_or_grid(path)
        bases.append(source.base if isinstance(source, GridSpec) else source)
    request_concurrencies = {
        min(
            base.llm_provider.request_concurrency,
            base.execution.parallelism,
            base.execution.repetitions,
        )
        for base in bases
    }
    if len(request_concurrencies) != 1:
        raise ValueError("automatic config-array planning requires one request concurrency")
    per_shard = request_concurrencies.pop()
    episode_slots = max(base.execution.parallelism for base in bases)
    policy = spec.execution
    target_rpm = int(policy.get("target_rpm", 900))
    latency = float(policy.get("assumed_latency_seconds", 10.0))
    if target_rpm < 1 or latency <= 0:
        raise ValueError("target_rpm and assumed_latency_seconds must be positive")
    max_nodes = int(policy.get("max_active_nodes", shard_count))
    throttle = min(shard_count, max_nodes)
    configured = policy.get("throttle")
    if configured is not None:
        throttle = min(throttle, int(configured))
    if throttle < 1:
        raise ValueError("planned array throttle must be positive")
    total_concurrency = throttle * per_shard
    control = ProviderLoadControlConfig.from_mapping(
        policy.get("provider_load_control"),
        defaults={
            "initial_concurrency": total_concurrency,
            "minimum_concurrency": min(4, total_concurrency),
            "maximum_concurrency": total_concurrency,
            "target_rpm": target_rpm,
        },
    )
    return ExecutionPlan(
        mode="config_array",
        shard_count=shard_count,
        array_throttle=throttle,
        request_concurrency_per_shard=per_shard,
        total_request_concurrency=total_concurrency,
        episode_slots_per_shard=episode_slots,
        total_episode_slots=throttle * episode_slots,
        target_rpm=target_rpm,
        assumed_latency_seconds=latency,
        estimated_rpm=min(target_rpm, total_concurrency * 60.0 / latency),
        cpus_per_task=int(policy.get("cpus_per_task", episode_slots)),
        memory=str(policy.get("memory", "8G")),
        time_limit=str(policy.get("time_limit", "04:00:00")),
        partition=str(policy.get("partition", "all")),
        qos=str(policy.get("qos", "normal")),
        provider_load_control=control.to_dict(),
        graceful_drain=graceful_drain_policy(policy, str(policy.get("time_limit", "04:00:00"))),
    )


def write_execution_manifest(path: str | Path, entries: Sequence[ExecutionEntry]) -> Path:
    destination = Path(path)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXECUTION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(entry) for entry in entries)
    return destination


def read_execution_manifest(path: str | Path) -> tuple[ExecutionEntry, ...]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = tuple(reader.fieldnames or ())
        legacy = EXECUTION_COLUMNS[:6]
        if columns not in {legacy, EXECUTION_COLUMNS}:
            raise ValueError(f"execution manifest has incompatible columns: {path}")
        rows = list(reader)
    entries = tuple(
        ExecutionEntry(
            array_index=int(row["array_index"]), config_index=int(row["config_index"]),
            config_path=row["config_path"], cell_index=int(row["cell_index"]),
            cell_id=row["cell_id"], output_dir=row["output_dir"],
            extension_index=int(row.get("extension_index") or 0),
            cell_key=row.get("cell_key", ""),
            episode_plan_path=row.get("episode_plan_path", ""),
            study_root=row.get("study_root", ""),
        )
        for row in rows
    )
    indices = [entry.array_index for entry in entries]
    if indices != sorted(indices) or sorted(set(indices)) != list(
        range(max(indices, default=-1) + 1)
    ):
        raise ValueError(
            "execution manifest array indices must be grouped and contiguous from zero"
        )
    return entries
