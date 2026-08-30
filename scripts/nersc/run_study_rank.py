#!/usr/bin/env python3
"""Run one rank's share of a prepared study inside an interactive allocation."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from mas_cc.studies.execution import read_execution_manifest
from mas_cc.studies.submission import read_submission_manifest


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_interactive_cpu_allocation() -> tuple[str, int, int]:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    qos = os.environ.get("SLURM_JOB_QOS", "")
    constraints = os.environ.get("SLURM_JOB_CONSTRAINTS", "")
    if not job_id:
        raise ValueError("study rank must run inside a SLURM allocation")
    if qos != "interactive":
        raise ValueError(f"refusing non-interactive NERSC QoS: {qos or 'unset'}")
    if "cpu" not in constraints:
        inspected = subprocess.run(
            ("scontrol", "show", "job", job_id, "--oneliner"),
            check=False,
            capture_output=True,
            text=True,
        )
        if inspected.returncode != 0 or re.search(
            r"(?:Features|Constraints)=\S*cpu", inspected.stdout
        ) is None:
            raise ValueError("refusing allocation without the Perlmutter cpu constraint")
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ranks = int(os.environ.get("SLURM_NTASKS", "1"))
    if rank < 0 or ranks < 1 or rank >= ranks:
        raise ValueError("invalid SLURM rank metadata")
    return job_id, rank, ranks


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _run_one(manifest: Path, worker_kind: str, index: int, logs: Path) -> dict[str, object]:
    module = (
        "mas_cc.studies.cell_worker"
        if worker_kind == "cell"
        else "mas_cc.studies.array_worker"
    )
    command = (sys.executable, "-m", module, str(manifest), str(index))
    environment = os.environ.copy()
    environment["SLURM_ARRAY_TASK_ID"] = str(index)
    environment["MAS_CC_EXECUTION_SITE"] = "nersc"
    stdout_path = logs / f"{worker_kind}-{index:06d}.out"
    stderr_path = logs / f"{worker_kind}-{index:06d}.err"
    started = _now()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            check=False,
            cwd=Path.cwd(),
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    return {
        "index": index,
        "returncode": completed.returncode,
        "started_at": started,
        "finished_at": _now(),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--worker-kind", choices=("cell", "config"), required=True)
    parser.add_argument("--total-workers", type=int, required=True)
    args = parser.parse_args()
    try:
        job_id, rank, ranks = _require_interactive_cpu_allocation()
        study_dir = args.study_dir.expanduser().resolve()
        manifest = args.manifest.expanduser().resolve()
        if manifest.parent != study_dir:
            raise ValueError("worker manifest must be directly under the prepared study root")
        entries = (
            read_execution_manifest(manifest)
            if args.worker_kind == "cell"
            else read_submission_manifest(manifest)
        )
        total_workers = min(args.total_workers, len(entries))
        if total_workers < 1:
            raise ValueError("total workers must be positive")
        active_ranks = min(ranks, total_workers)
        assigned = [index for index in range(len(entries)) if index % active_ranks == rank]
        local_workers = total_workers // active_ranks + (rank < total_workers % active_ranks)
        if rank >= active_ranks:
            assigned = []
            local_workers = 0
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    logs = study_dir / "logs" / f"nersc-{job_id}"
    logs.mkdir(parents=True, exist_ok=True)
    print(
        f"[nersc] job={job_id} rank={rank}/{ranks} qos=interactive "
        f"workers={local_workers} shards={len(assigned)}",
        flush=True,
    )
    results: list[dict[str, object]] = []
    if local_workers:
        with ThreadPoolExecutor(max_workers=local_workers) as pool:
            futures = {
                pool.submit(_run_one, manifest, args.worker_kind, index, logs): index
                for index in assigned
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"[nersc] rank={rank} shard={result['index']} "
                    f"returncode={result['returncode']}",
                    flush=True,
                )

    results.sort(key=lambda row: int(row["index"]))
    summary = {
        "schema_version": 1,
        "job_id": job_id,
        "qos": "interactive",
        "constraint": "cpu",
        "rank": rank,
        "rank_count": ranks,
        "worker_kind": args.worker_kind,
        "local_worker_limit": local_workers,
        "assigned_indices": assigned,
        "results": results,
        "finished_at": _now(),
    }
    _write_json_atomic(study_dir / "runtime" / f"nersc-rank-{rank:02d}.json", summary)
    failures = [row for row in results if int(row["returncode"]) != 0]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
