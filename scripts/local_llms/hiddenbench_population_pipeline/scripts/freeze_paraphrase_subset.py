#!/usr/bin/env python3
"""Freeze a validated task subset from an unfinished global paraphrase pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from hiddenbench_common import DEFAULT_DATA_ROOT, PipelineError, ValidationError, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible frozen paraphrase release containing only complete "
            "HiddenBench tasks with enough accepted variants for the requested populations."
        )
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--canonical",
        type=Path,
        default=DEFAULT_DATA_ROOT / "canonical" / "tasks.json",
    )
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--agents", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _accepted(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    variants = record.get("variants")
    if not isinstance(variants, list):
        raise ValidationError("evidence annotation variants must be a list")
    return [
        variant
        for variant in variants
        if isinstance(variant, Mapping)
        and variant.get("accepted", True)
        and isinstance(variant.get("text"), str)
        and variant["text"].strip()
    ]


def freeze_subset(
    annotations: Mapping[str, Any],
    canonical: Mapping[str, Any],
    *,
    task_ids: Sequence[int],
    population_sizes: Sequence[int],
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate capacity and return a deterministic, task-scoped frozen release."""

    selected_ids = tuple(dict.fromkeys(int(value) for value in task_ids))
    sizes = tuple(sorted(set(int(value) for value in population_sizes)))
    if not selected_ids or not sizes or any(value < 1 for value in sizes):
        raise ValidationError("task_ids and positive population_sizes are required")
    task_pool = annotations.get("tasks")
    canonical_tasks = canonical.get("tasks")
    if not isinstance(task_pool, Mapping) or not isinstance(canonical_tasks, list):
        raise ValidationError("annotations and canonical corpus must contain tasks")
    canonical_by_id = {int(task["task_id"]): task for task in canonical_tasks}
    frozen_tasks: dict[str, Any] = {}
    capacity: dict[str, int] = {}

    for task_id in selected_ids:
        task = canonical_by_id.get(task_id)
        annotation = task_pool.get(str(task_id))
        if task is None or not isinstance(annotation, Mapping):
            raise ValidationError(f"task {task_id} is absent from the canonical corpus or annotations")
        # Work on a detached copy so missing legacy IDs can be normalized into
        # stable pool IDs without mutating the unfinished global annotation map.
        frozen_annotation = json.loads(json.dumps(annotation))
        hidden = task.get("hidden_information")
        if not isinstance(hidden, list) or not hidden:
            raise ValidationError(f"canonical task {task_id} has no hidden evidence types")
        evidence = frozen_annotation.get("evidence_types")
        if not isinstance(evidence, Mapping):
            raise ValidationError(f"task {task_id} has no evidence_types annotation mapping")
        expected = {str(index) for index in range(len(hidden))}
        if set(evidence) != expected:
            raise ValidationError(
                f"task {task_id} evidence types are incomplete: expected {sorted(expected)}"
            )
        required = math.ceil(max(sizes) / len(hidden))
        capacity[str(task_id)] = required
        for index, source in enumerate(hidden):
            record = evidence[str(index)]
            if not isinstance(record, Mapping):
                raise ValidationError(f"task {task_id} evidence type {index} is not an object")
            source_text = str(source["source_text"])
            if str(record.get("source_text")) != source_text:
                raise ValidationError(f"task {task_id} evidence type {index} has mismatched source text")
            for variant_index, variant in enumerate(record.get("variants", [])):
                if (
                    isinstance(variant, dict)
                    and variant.get("accepted", True)
                    and isinstance(variant.get("text"), str)
                    and variant["text"].strip()
                    and not isinstance(variant.get("variant_id"), str)
                ):
                    variant["variant_id"] = f"{task_id}-{index}-{variant_index:03d}"
            accepted = _accepted(record)
            variant_ids = [str(item.get("variant_id", "")) for item in accepted]
            variant_texts = [str(item["text"]).strip().casefold() for item in accepted]
            if len(accepted) < required:
                raise ValidationError(
                    f"task {task_id} evidence type {index} needs {required} accepted variants, "
                    f"but has {len(accepted)}"
                )
            if not all(variant_ids) or len(set(variant_ids)) != len(variant_ids):
                raise ValidationError(f"task {task_id} evidence type {index} has invalid variant IDs")
            if len(set(variant_texts)) != len(variant_texts):
                raise ValidationError(f"task {task_id} evidence type {index} has duplicate paraphrases")
        # Generator and verifier provenance remains attached to every variant.
        frozen_tasks[str(task_id)] = frozen_annotation

    top_level = {
        key: value
        for key, value in annotations.items()
        if key not in {"tasks", "status", "release_provenance"}
    }
    return {
        **top_level,
        "status": "frozen",
        "tasks": frozen_tasks,
        "release_provenance": {
            "source_annotations_sha256": source_sha256,
            "selected_task_ids": list(selected_ids),
            "population_sizes": list(sizes),
            "required_variants_per_evidence_type": capacity,
            "selection": "complete_task_subset",
        },
    }


def main() -> None:
    args = parse_args()
    source_bytes = args.annotations.read_bytes()
    payload = freeze_subset(
        read_json(args.annotations),
        read_json(args.canonical),
        task_ids=args.task_ids,
        population_sizes=args.agents,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )
    write_json(args.output, payload, overwrite=args.overwrite)
    print(f"Wrote frozen {len(payload['tasks'])}-task release to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
