#!/usr/bin/env python3
"""Add one task to a paraphrased population file when it is not ready yet."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from freeze_paraphrase_subset import freeze_subset
from hiddenbench_common import (
    DEFAULT_DATA_ROOT,
    PipelineError,
    ValidationError,
    read_json,
    stable_seed,
)
from prepare_hiddenbench import scale_paraphrased


METHOD = "paraphrased_replication"
DEFAULT_SEED = 1729


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse or deterministically build the requested task in a paraphrased "
            "HiddenBench N_<agents>.json population. No LLM calls are made."
        )
    )
    parser.add_argument("--agents", type=int, required=True)
    parser.add_argument(
        "--task",
        help="Canonical task name or numeric task ID; defaults to the first task.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_DATA_ROOT / "annotations" / "paraphrases.json",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _task_id(record: Mapping[str, Any]) -> int:
    return int(record["task_id"])


def _select_task(
    tasks: Sequence[Mapping[str, Any]], selector: str | None
) -> Mapping[str, Any]:
    if not tasks:
        raise ValidationError("The canonical HiddenBench corpus contains no tasks.")
    if selector is None:
        return tasks[0]
    for task in tasks:
        if str(task.get("name")) == selector or str(task.get("task_id")) == selector:
            return task
    raise ValidationError(f"No canonical HiddenBench task matches {selector!r}.")


def _validate_existing(
    payload: Mapping[str, Any], path: Path, num_agents: int
) -> list[Mapping[str, Any]]:
    metadata = payload.get("metadata")
    tasks = payload.get("tasks")
    if not isinstance(metadata, Mapping):
        raise ValidationError(f"Existing scaled population at {path} has no metadata.")
    if (
        metadata.get("scaling_method") != METHOD
        or int(metadata.get("num_agents", -1)) != num_agents
    ):
        raise ValidationError(
            f"Existing scaled population at {path} is not {METHOD} for N={num_agents}."
        )
    if not isinstance(tasks, list):
        raise ValidationError(f"Existing scaled population at {path} has no task list.")
    if not all(isinstance(task, Mapping) for task in tasks):
        raise ValidationError(
            f"Existing scaled population at {path} contains a malformed task."
        )
    return list(tasks)


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_paraphrased_population(
    *,
    data_root: Path,
    annotations_path: Path,
    num_agents: int,
    task_selector: str | None,
    seed: int = DEFAULT_SEED,
) -> tuple[Path, bool]:
    """Ensure one task is present, returning ``(population_path, changed)``."""

    if num_agents < 2:
        raise ValidationError("A paraphrased population requires at least two agents.")
    canonical_path = data_root / "canonical" / "tasks.json"
    if not canonical_path.exists():
        raise ValidationError(
            f"Canonical HiddenBench corpus does not exist at {canonical_path}."
        )
    canonical = read_json(canonical_path)
    canonical_tasks = canonical.get("tasks")
    if not isinstance(canonical_tasks, list):
        raise ValidationError(
            f"Canonical HiddenBench corpus at {canonical_path} has no task list."
        )
    task = _select_task(canonical_tasks, task_selector)
    selected_id = _task_id(task)

    output_path = data_root / "scaled" / METHOD / f"N_{num_agents}.json"
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing_payload: Mapping[str, Any] | None = None
        existing_tasks: list[Mapping[str, Any]] = []
        if output_path.exists():
            value = read_json(output_path)
            if not isinstance(value, Mapping):
                raise ValidationError(f"Scaled population at {output_path} must be an object.")
            existing_payload = value
            existing_tasks = _validate_existing(value, output_path, num_agents)
            if any(_task_id(item) == selected_id for item in existing_tasks):
                return output_path, False

        if not annotations_path.exists():
            raise ValidationError(
                f"No paraphrase annotations exist at {annotations_path}. Generate or restore "
                "paraphrases.json before launching this configuration."
            )
        annotations = read_json(annotations_path)
        if not isinstance(annotations, Mapping):
            raise ValidationError(
                f"Paraphrase annotations at {annotations_path} must be an object."
            )
        source_sha256 = hashlib.sha256(annotations_path.read_bytes()).hexdigest()
        # The global annotation pool may be unfinished. Freeze and validate only
        # the requested task/capacity; no paraphrase text is generated here.
        frozen = freeze_subset(
            annotations,
            canonical,
            task_ids=[selected_id],
            population_sizes=[num_agents],
            source_sha256=source_sha256,
        )
        local_seed = stable_seed(seed, selected_id, num_agents, METHOD)
        scaled = scale_paraphrased(
            task,
            num_agents,
            local_seed,
            frozen,
            allow_reuse=False,
        )

        merged = {_task_id(item): dict(item) for item in existing_tasks}
        merged[selected_id] = scaled
        metadata = (
            dict(existing_payload.get("metadata", {})) if existing_payload else {}
        )
        prepared_ids = {
            int(item) for item in metadata.get("auto_prepared_task_ids", ())
        }
        prepared_ids.add(selected_id)
        payload = {
            "metadata": {
                **metadata,
                "kind": "scaled",
                "scaling_method": METHOD,
                "num_agents": num_agents,
                "base_seed": seed,
                "annotation_file": str(annotations_path),
                "annotation_sha256": source_sha256,
                "auto_prepared_task_ids": sorted(prepared_ids),
                "excluded_tasks": list(metadata.get("excluded_tasks", ())),
            },
            "tasks": [merged[task_id] for task_id in sorted(merged)],
        }
        _atomic_write(output_path, payload)
        return output_path, True


def main() -> None:
    args = parse_args()
    path, changed = ensure_paraphrased_population(
        data_root=args.data_root,
        annotations_path=args.annotations,
        num_agents=args.agents,
        task_selector=args.task,
        seed=args.seed,
    )
    verb = "Prepared" if changed else "Reused"
    print(f"{verb} {path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, PipelineError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
