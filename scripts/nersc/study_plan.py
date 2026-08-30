#!/usr/bin/env python3
"""Validate a prepared study and derive a safe Perlmutter worker count."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


PHYSICAL_CORES_PER_NODE = 128
MEMORY_BYTES_PER_NODE = 512 * 1024**3


@dataclass(frozen=True, slots=True)
class NerscStudyPlan:
    study_dir: str
    manifest: str
    worker_kind: str
    shard_count: int
    requested_nodes: int
    planned_throttle: int
    physical_cpus_per_worker: int
    memory_bytes_per_worker: int
    workers_per_node_ceiling: int
    total_workers: int
    planned_time_limit: str
    source_partition: str
    source_qos: str
    nersc_constraint: str = "cpu"
    nersc_qos: str = "interactive"


def _mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _validate_output_roots(path: Path, study_root: Path) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        output = Path(row.get("output_dir", "")).expanduser().resolve()
        try:
            output.relative_to(study_root)
        except ValueError as exc:
            raise ValueError(
                f"worker output escapes prepared NERSC study root: {output}"
            ) from exc


def _bytes(value: object) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\s*", str(value), re.I)
    if match is None:
        raise ValueError(f"unsupported worker memory value: {value!r}")
    amount = float(match.group(1))
    power = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4}[match.group(2).upper()]
    return math.ceil(amount * 1024**power)


def _config_array_throttle(array: object, shard_count: int) -> int:
    match = re.fullmatch(r"0-(\d+)(?:%(\d+))?", str(array))
    if match is None or int(match.group(1)) + 1 != shard_count:
        raise ValueError(f"invalid prepared array mapping: {array!r}")
    return int(match.group(2) or shard_count)


def build_plan(study_dir: str | Path, nodes: int) -> NerscStudyPlan:
    root = Path(study_dir).expanduser().resolve()
    if nodes < 1 or nodes > 4:
        raise ValueError("NERSC interactive studies require 1 to 4 nodes")
    preparation = _mapping(root / "preparation.json")
    if preparation.get("status") != "prepared":
        raise ValueError(f"study has not been prepared: {root}")
    if preparation.get("execution_site") != "nersc":
        raise ValueError(
            f"study was not prepared for the NERSC scheduler adapter: {root}"
        )

    execution_manifest = root / "execution_manifest.csv"
    if execution_manifest.is_file():
        worker_kind = "cell"
        manifest = execution_manifest
        execution = _mapping(root / "execution_plan.json")
        manifest_rows = _row_count(manifest)
        shard_count = int(execution.get("shard_count", manifest_rows))
        if shard_count != manifest_rows:
            raise ValueError(
                f"execution plan declares {shard_count} shards but manifest has "
                f"{manifest_rows} rows"
            )
        planned_throttle = int(execution["array_throttle"])
        cpus = int(execution["cpus_per_task"])
        memory = _bytes(execution["memory"])
        time_limit = str(execution["time_limit"])
        source_partition = str(execution.get("partition", ""))
        source_qos = str(execution.get("qos", ""))
    else:
        worker_kind = "config"
        manifest = root / "submission_manifest.csv"
        shard_count = _row_count(manifest)
        planned_throttle = _config_array_throttle(preparation.get("array"), shard_count)
        # Config-array mode has no per-shard resource plan, so use one worker per node.
        cpus = PHYSICAL_CORES_PER_NODE
        memory = MEMORY_BYTES_PER_NODE
        time_limit = "04:00:00"
        source_partition = ""
        source_qos = ""

    if not manifest.is_file() or shard_count < 1:
        raise ValueError(f"prepared worker manifest is empty or missing: {manifest}")
    _validate_output_roots(manifest, root)
    if cpus < 1 or cpus > PHYSICAL_CORES_PER_NODE:
        raise ValueError(f"physical CPUs per worker must be within 1..{PHYSICAL_CORES_PER_NODE}")
    if memory < 1 or memory > MEMORY_BYTES_PER_NODE:
        raise ValueError("worker memory exceeds one Perlmutter CPU node")

    worker_ceiling = min(PHYSICAL_CORES_PER_NODE // cpus, MEMORY_BYTES_PER_NODE // memory)
    total_workers = min(shard_count, planned_throttle, nodes * worker_ceiling)
    if total_workers < nodes:
        raise ValueError(
            f"{nodes} nodes would exceed the useful worker count "
            f"({total_workers}); request fewer nodes"
        )
    return NerscStudyPlan(
        study_dir=str(root),
        manifest=str(manifest),
        worker_kind=worker_kind,
        shard_count=shard_count,
        requested_nodes=nodes,
        planned_throttle=planned_throttle,
        physical_cpus_per_worker=cpus,
        memory_bytes_per_worker=memory,
        workers_per_node_ceiling=worker_ceiling,
        total_workers=total_workers,
        planned_time_limit=time_limit,
        source_partition=source_partition,
        source_qos=source_qos,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    args = parser.parse_args()
    try:
        plan = build_plan(args.study_dir, args.nodes)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.format == "tsv":
        print(
            "\t".join(
                (
                    plan.manifest,
                    plan.worker_kind,
                    str(plan.total_workers),
                    str(plan.physical_cpus_per_worker),
                    plan.planned_time_limit,
                    str(plan.shard_count),
                )
            )
        )
    else:
        print(json.dumps(asdict(plan), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
