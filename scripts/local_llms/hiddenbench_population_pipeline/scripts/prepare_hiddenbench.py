#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from hiddenbench_common import (
    PipelineError,
    ValidationError,
    allocate_factor_components,
    balanced_type_assignment,
    canonicalize_tasks,
    download_source,
    load_source_tasks,
    read_json,
    select_tasks,
    stable_seed,
    task_id,
    write_json,
)


METHODS = {
    "exact_replication",
    "paraphrased_replication",
    "factorized_evidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download HiddenBench and optionally construct scaled population datasets."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/hiddenbench"),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face revision or commit hash.",
    )
    parser.add_argument(
        "--agents",
        type=int,
        nargs="+",
        required=True,
        help=(
            "Population sizes. Use exactly `--agents 0` to download and canonicalize "
            "the untouched benchmark without creating a scaled dataset."
        ),
    )
    parser.add_argument(
        "--method",
        choices=sorted(METHODS),
        default="exact_replication",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Paraphrase or factorization annotation JSON.",
    )
    parser.add_argument("--task-ids", type=int, nargs="*")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--allow-paraphrase-reuse",
        action="store_true",
        help="Allow a paraphrase variant to be assigned more than once.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def source_hidden(task: Mapping[str, Any]) -> list[str]:
    return [str(item["source_text"]) for item in task["hidden_information"]]


def population_instruction(num_agents: int, num_types: int) -> str:
    return (
        f"There are exactly {num_agents} agents in this experiment. "
        f"Private information derives from {num_types} latent evidence types. "
        "Several agents may carry different realizations or components of the same "
        "type. Ignore any historical participant count left in the scenario wording."
    )


def task_annotation(
    annotations: Mapping[str, Any], task_id_value: int
) -> Mapping[str, Any]:
    tasks = annotations.get("tasks", annotations)
    value = tasks.get(str(task_id_value))
    if value is None:
        raise ValidationError(
            f"No annotation entry exists for task {task_id_value}."
        )
    if not isinstance(value, Mapping):
        raise ValidationError(
            f"Annotation entry for task {task_id_value} must be an object."
        )
    return value


def evidence_annotation(
    task_ann: Mapping[str, Any], evidence_type: int
) -> Mapping[str, Any]:
    evidence_types = task_ann.get("evidence_types", task_ann)
    value = evidence_types.get(str(evidence_type))
    if value is None:
        raise ValidationError(
            f"No annotation exists for evidence type {evidence_type}."
        )
    if not isinstance(value, Mapping):
        raise ValidationError(
            f"Evidence annotation {evidence_type} must be an object."
        )
    return value


def base_scaled_task(
    task: Mapping[str, Any],
    *,
    num_agents: int,
    method: str,
    agents: Sequence[Mapping[str, Any]],
    seed: int,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hidden = source_hidden(task)
    type_counts = Counter(int(agent["evidence_type"]) for agent in agents)
    return {
        "task_id": task_id(task),
        "name": task["name"],
        "source_description": task["source_description"],
        "scenario_description": task["scenario_description"],
        "population_wording_changes": task.get("population_wording_changes", []),
        "population_instruction": population_instruction(
            num_agents, len(hidden)
        ),
        "shared_information": list(task["shared_information"]),
        "possible_answers": list(task["possible_answers"]),
        "correct_answer": task["correct_answer"],
        "source_hidden_information": hidden,
        "rationale": task.get("rationale"),
        "population": {
            "num_agents": num_agents,
            "source_base_agent_count": len(hidden),
            "number_of_evidence_types": len(hidden),
            "method": method,
            "allocation_seed": seed,
            "type_counts": {
                str(key): value for key, value in sorted(type_counts.items())
            },
            "diagnostics": dict(diagnostics or {}),
        },
        "agents": [dict(agent) for agent in agents],
    }


def scale_exact(
    task: Mapping[str, Any], num_agents: int, seed: int
) -> dict[str, Any]:
    hidden = source_hidden(task)
    labels = balanced_type_assignment(
        num_agents, len(hidden), seed=seed
    )
    agents = [
        {
            "agent_id": agent_id,
            "evidence_type": evidence_type,
            "private_information": [hidden[evidence_type]],
            "source_hidden_indices": [evidence_type],
            "transformation": "identity",
        }
        for agent_id, evidence_type in enumerate(labels)
    ]
    return base_scaled_task(
        task,
        num_agents=num_agents,
        method="exact_replication",
        agents=agents,
        seed=seed,
    )


def accepted_variants(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    variants = record.get("variants", [])
    accepted = [
        variant
        for variant in variants
        if isinstance(variant, Mapping)
        and variant.get("accepted", True)
        and isinstance(variant.get("text"), str)
        and variant["text"].strip()
    ]
    if not accepted:
        raise ValidationError("No accepted paraphrase variants are available.")
    return accepted


def scale_paraphrased(
    task: Mapping[str, Any],
    num_agents: int,
    seed: int,
    annotations: Mapping[str, Any],
    *,
    allow_reuse: bool,
) -> dict[str, Any]:
    hidden = source_hidden(task)
    task_ann = task_annotation(annotations, task_id(task))
    labels = balanced_type_assignment(
        num_agents, len(hidden), seed=seed
    )
    rng = random.Random(seed)

    pools: dict[int, list[Mapping[str, Any]]] = {}
    for evidence_type in range(len(hidden)):
        record = evidence_annotation(task_ann, evidence_type)
        pool = list(accepted_variants(record))
        rng.shuffle(pool)
        pools[evidence_type] = pool

    counters: Counter[int] = Counter()
    agents = []
    for agent_id, evidence_type in enumerate(labels):
        pool = pools[evidence_type]
        index = counters[evidence_type]
        counters[evidence_type] += 1

        if index >= len(pool) and not allow_reuse:
            raise ValidationError(
                f"Task {task_id(task)} evidence type {evidence_type} needs "
                f"{index + 1} variants, but the pool contains {len(pool)}. "
                "Generate a larger pool or pass --allow-paraphrase-reuse."
            )

        variant = pool[index % len(pool)]
        agents.append(
            {
                "agent_id": agent_id,
                "evidence_type": evidence_type,
                "variant_id": variant.get(
                    "variant_id", f"{evidence_type}-{index:03d}"
                ),
                "private_information": [variant["text"]],
                "source_hidden_indices": [evidence_type],
                "source_text": hidden[evidence_type],
                "transformation": "validated_paraphrase",
            }
        )

    return base_scaled_task(
        task,
        num_agents=num_agents,
        method="paraphrased_replication",
        agents=agents,
        seed=seed,
        diagnostics={
            "variant_pool_sizes": {
                str(key): len(value) for key, value in pools.items()
            },
            "variant_reuse_allowed": allow_reuse,
        },
    )


def selected_factorization(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if record.get("factorizable") is False:
        raise ValidationError(
            f"Evidence was marked non-factorizable: "
            f"{record.get('non_factorizable_reason')}"
        )
    selected = record.get("selected_factorization")
    if isinstance(selected, Mapping):
        return selected

    alternatives = record.get("alternatives", [])
    accepted = [
        alternative
        for alternative in alternatives
        if isinstance(alternative, Mapping)
        and alternative.get("accepted", True)
    ]
    if not accepted:
        raise ValidationError("No accepted factorization is available.")
    return max(
        accepted,
        key=lambda item: float(item.get("quality_score", 0)),
    )


def scale_factorized(
    task: Mapping[str, Any],
    num_agents: int,
    seed: int,
    annotations: Mapping[str, Any],
) -> dict[str, Any]:
    hidden = source_hidden(task)
    task_ann = task_annotation(annotations, task_id(task))
    components: list[dict[str, Any]] = []

    for evidence_type in range(len(hidden)):
        record = evidence_annotation(task_ann, evidence_type)
        factorization = selected_factorization(record)
        factor_components = factorization.get("components", [])
        if not isinstance(factor_components, list) or len(factor_components) < 2:
            raise ValidationError(
                f"Task {task_id(task)} evidence type {evidence_type} does not "
                "contain at least two factor components."
            )
        for index, component in enumerate(factor_components):
            if not isinstance(component, Mapping):
                raise ValidationError("Each factor component must be an object.")
            text = component.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValidationError("Factor component text is missing.")
            components.append(
                {
                    "evidence_type": evidence_type,
                    "component_id": component.get(
                        "component_id", f"{evidence_type}-{index}"
                    ),
                    "text": text.strip(),
                    "role": component.get("role"),
                    "source_text": hidden[evidence_type],
                    "reconstruction_rule": factorization.get(
                        "reconstruction_rule"
                    ),
                }
            )

    allocation, diagnostics = allocate_factor_components(
        components, num_agents, seed=seed
    )

    agents = []
    for agent_id, packet in enumerate(allocation):
        evidence_types = sorted(
            {int(component["evidence_type"]) for component in packet}
        )
        agents.append(
            {
                "agent_id": agent_id,
                # Retain a scalar for compatibility when there is one type.
                "evidence_type": evidence_types[0] if len(evidence_types) == 1 else -1,
                "evidence_types": evidence_types,
                "component_ids": [
                    component["component_id"] for component in packet
                ],
                "private_information": [
                    component["text"] for component in packet
                ],
                "source_hidden_indices": evidence_types,
                "transformation": "factor_components",
            }
        )

    return base_scaled_task(
        task,
        num_agents=num_agents,
        method="factorized_evidence",
        agents=agents,
        seed=seed,
        diagnostics=diagnostics,
    )


def main() -> None:
    args = parse_args()

    if args.agents == [0]:
        source_path, metadata = download_source(
            args.data_root,
            revision=args.revision,
            overwrite=args.overwrite,
        )
        tasks = load_source_tasks(source_path)
        canonical = {
            "metadata": {
                **metadata,
                "kind": "canonical",
            },
            "tasks": canonicalize_tasks(tasks),
        }
        write_json(
            args.data_root / "canonical" / "tasks.json",
            canonical,
            overwrite=args.overwrite,
        )
        print(f"Preserved source benchmark at {source_path}")
        print(f"Wrote canonical tasks to {args.data_root / 'canonical' / 'tasks.json'}")
        return

    if 0 in args.agents or any(value < 0 for value in args.agents):
        raise ValidationError(
            "Use `--agents 0` alone, or provide only positive population sizes."
        )

    source_path, metadata = download_source(
        args.data_root,
        revision=args.revision,
        overwrite=False,
    )
    source_tasks = load_source_tasks(source_path)
    canonical_tasks = canonicalize_tasks(source_tasks)
    canonical_payload = {
        "metadata": {**metadata, "kind": "canonical"},
        "tasks": canonical_tasks,
    }
    canonical_path = args.data_root / "canonical" / "tasks.json"
    if not canonical_path.exists() or args.overwrite:
        write_json(
            canonical_path,
            canonical_payload,
            overwrite=args.overwrite,
        )

    tasks = select_tasks(canonical_tasks, args.task_ids)

    annotations: Mapping[str, Any] = {}
    if args.method != "exact_replication":
        if args.annotations is None:
            raise ValidationError(
                f"--annotations is required for {args.method}."
            )
        annotations = read_json(args.annotations)
        if annotations.get("status") != "frozen":
            raise ValidationError(
                "Annotations must be a completed frozen release. Finish the "
                "annotation run (or resume it) before scaling populations."
            )

    for num_agents in args.agents:
        scaled_tasks = []
        excluded_tasks: list[dict[str, Any]] = []
        for task in tasks:
            local_seed = stable_seed(
                args.seed, task_id(task), num_agents, args.method
            )
            if args.method == "exact_replication":
                scaled = scale_exact(task, num_agents, local_seed)
            elif args.method == "paraphrased_replication":
                scaled = scale_paraphrased(
                    task,
                    num_agents,
                    local_seed,
                    annotations,
                    allow_reuse=args.allow_paraphrase_reuse,
                )
            elif args.method == "factorized_evidence":
                try:
                    scaled = scale_factorized(
                        task,
                        num_agents,
                        local_seed,
                        annotations,
                    )
                except ValidationError as exc:
                    # A verified non-factorizable clue is an intentional
                    # scientific exclusion, not a reason to contaminate it with
                    # an arbitrary split or abort the entire condition.
                    excluded_tasks.append(
                        {
                            "task_id": task_id(task),
                            "name": task["name"],
                            "reason": str(exc),
                        }
                    )
                    continue
            else:
                raise ValidationError(f"Unsupported method {args.method}.")
            scaled_tasks.append(scaled)

        output_path = (
            args.data_root
            / "scaled"
            / args.method
            / f"N_{num_agents}.json"
        )
        write_json(
            output_path,
            {
                "metadata": {
                    **metadata,
                    "kind": "scaled",
                    "scaling_method": args.method,
                    "num_agents": num_agents,
                    "base_seed": args.seed,
                    "annotation_file": (
                        str(args.annotations) if args.annotations else None
                    ),
                    "excluded_tasks": excluded_tasks,
                },
                "tasks": scaled_tasks,
            },
            overwrite=args.overwrite,
        )
        print(f"Wrote {len(scaled_tasks)} tasks to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
