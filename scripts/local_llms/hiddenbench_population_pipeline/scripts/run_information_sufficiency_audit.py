#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from hiddenbench_common import (
    PipelineError,
    extract_tasks_payload,
    select_tasks,
    source_hidden_texts,
    stable_seed,
    task_id,
    write_json,
)
from hiddenbench_evaluation import call_vote, experiment_agents
from hiddenbench_llm_api import LLMClient, LLMConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the Hidden Profile information gap before running discussion."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/information_sufficiency_audit.json"),
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=10,
        help="The paper validates task conditions across ten sessions.",
    )
    parser.add_argument("--full-threshold", type=float, default=0.80)
    parser.add_argument("--partial-threshold", type=float, default=0.20)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--task-ids", type=int, nargs="*")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def accuracy(records: Sequence[Mapping[str, Any]]) -> float:
    if not records:
        return 0.0
    return statistics.fmean(float(record["correct"]) for record in records)


def evaluate_packet(
    client: LLMClient,
    task: Mapping[str, Any],
    packet: Sequence[str],
    *,
    label: str,
    sessions: int,
    base_seed: int,
    temperature: float | None,
) -> dict[str, Any]:
    decisions = []
    for session_index in range(sessions):
        decision = call_vote(
            client,
            task,
            packet,
            seed=stable_seed(
                base_seed,
                task_id(task),
                label,
                session_index,
            ),
            temperature=temperature,
        )
        decision["session_index"] = session_index
        decisions.append(decision)
    return {
        "label": label,
        "private_information": list(packet),
        "accuracy": accuracy(decisions),
        "decisions": decisions,
    }


def unique_transformed_packets(
    task: Mapping[str, Any],
) -> list[tuple[str, list[str]]]:
    agents = experiment_agents(task)
    result = []
    seen: set[tuple[str, ...]] = set()
    for agent in agents:
        packet = tuple(agent["private_information"])
        if packet in seen:
            continue
        seen.add(packet)
        result.append(
            (f"transformed_agent_{agent['agent_id']}", list(packet))
        )
    return result


def component_singletons(
    task: Mapping[str, Any],
) -> list[tuple[str, list[str]]]:
    if task.get("population", {}).get("method") != "factorized_evidence":
        return []
    result = []
    seen: set[tuple[str, str]] = set()
    for agent in task.get("agents", []):
        component_ids = agent.get("component_ids", [])
        texts = agent.get("private_information", [])
        for component_id, text in zip(component_ids, texts):
            key = (str(component_id), str(text))
            if key not in seen:
                seen.add(key)
                result.append((f"component_{component_id}", [str(text)]))
    return result


def pooled_transformed_information(task: Mapping[str, Any]) -> list[str]:
    result = []
    seen = set()
    for agent in experiment_agents(task):
        for text in agent["private_information"]:
            if text not in seen:
                seen.add(text)
                result.append(text)
    return result


def main() -> None:
    args = parse_args()
    tasks, input_metadata = extract_tasks_payload(args.input)
    tasks = select_tasks(tasks, args.task_ids)
    client = LLMClient(
        LLMConfig.from_env(
            model_env="LLM_AUDIT_MODEL",
            default_model="microsoft/gpt-5.5",
        )
    )
    results = []

    for task in tasks:
        tid = task_id(task)
        print(f"auditing task={tid}", flush=True)
        original_hidden = source_hidden_texts(task)

        shared_only = evaluate_packet(
            client,
            task,
            [],
            label="shared_only",
            sessions=args.sessions,
            base_seed=args.seed,
            temperature=args.temperature,
        )

        original_partials = [
            evaluate_packet(
                client,
                task,
                [text],
                label=f"original_hidden_{index}",
                sessions=args.sessions,
                base_seed=args.seed,
                temperature=args.temperature,
            )
            for index, text in enumerate(original_hidden)
        ]

        original_full = evaluate_packet(
            client,
            task,
            original_hidden,
            label="original_full_profile",
            sessions=args.sessions,
            base_seed=args.seed,
            temperature=args.temperature,
        )

        leave_one_out = [
            evaluate_packet(
                client,
                task,
                [
                    text
                    for current_index, text in enumerate(original_hidden)
                    if current_index != omitted_index
                ],
                label=f"original_leave_out_{omitted_index}",
                sessions=args.sessions,
                base_seed=args.seed,
                temperature=args.temperature,
            )
            for omitted_index in range(len(original_hidden))
        ]

        transformed_packets = []
        transformed_pooled = None
        factor_singletons = []
        if isinstance(task.get("agents"), list):
            transformed_packets = [
                evaluate_packet(
                    client,
                    task,
                    packet,
                    label=label,
                    sessions=args.sessions,
                    base_seed=args.seed,
                    temperature=args.temperature,
                )
                for label, packet in unique_transformed_packets(task)
            ]
            transformed_pooled = evaluate_packet(
                client,
                task,
                pooled_transformed_information(task),
                label="transformed_pooled_profile",
                sessions=args.sessions,
                base_seed=args.seed,
                temperature=args.temperature,
            )
            factor_singletons = [
                evaluate_packet(
                    client,
                    task,
                    packet,
                    label=label,
                    sessions=args.sessions,
                    base_seed=args.seed,
                    temperature=args.temperature,
                )
                for label, packet in component_singletons(task)
            ]

        original_partial_mean = statistics.fmean(
            item["accuracy"] for item in original_partials
        )
        original_partial_max = max(
            item["accuracy"] for item in original_partials
        )
        original_pass = (
            original_full["accuracy"] >= args.full_threshold
            and original_partial_mean <= args.partial_threshold
        )

        transformed_summary = None
        if transformed_pooled is not None and transformed_packets:
            transformed_partial_mean = statistics.fmean(
                item["accuracy"] for item in transformed_packets
            )
            transformed_partial_max = max(
                item["accuracy"] for item in transformed_packets
            )
            transformed_summary = {
                "pooled_accuracy": transformed_pooled["accuracy"],
                "partial_mean_accuracy": transformed_partial_mean,
                "partial_max_accuracy": transformed_partial_max,
                "passes_paper_style_thresholds": (
                    transformed_pooled["accuracy"] >= args.full_threshold
                    and transformed_partial_mean <= args.partial_threshold
                ),
            }

        results.append(
            {
                "task": {
                    "task_id": tid,
                    "name": task["name"],
                    "correct_answer": task["correct_answer"],
                    "population": task.get("population"),
                },
                "summary": {
                    "original_full_accuracy": original_full["accuracy"],
                    "original_partial_mean_accuracy": original_partial_mean,
                    "original_partial_max_accuracy": original_partial_max,
                    "original_passes_paper_style_thresholds": original_pass,
                    "transformed": transformed_summary,
                },
                "conditions": {
                    "shared_only": shared_only,
                    "original_partials": original_partials,
                    "original_full": original_full,
                    "original_leave_one_out": leave_one_out,
                    "transformed_partials": transformed_packets,
                    "transformed_pooled": transformed_pooled,
                    "factor_component_singletons": factor_singletons,
                },
            }
        )

        write_json(
            args.output,
            {
                "metadata": {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "in_progress",
                    "input": str(args.input),
                    "input_metadata": input_metadata,
                    "model": client.config.model,
                    "sessions_per_condition": args.sessions,
                    "full_threshold": args.full_threshold,
                    "partial_threshold": args.partial_threshold,
                },
                "results": results,
            },
            overwrite=True,
        )

    overall = {
        "num_tasks": len(results),
        "original_pass_fraction": statistics.fmean(
            float(item["summary"]["original_passes_paper_style_thresholds"])
            for item in results
        ) if results else 0.0,
        "transformed_pass_fraction": statistics.fmean(
            float(item["summary"]["transformed"]["passes_paper_style_thresholds"])
            for item in results
            if item["summary"]["transformed"] is not None
        ) if any(
            item["summary"]["transformed"] is not None for item in results
        ) else None,
    }

    write_json(
        args.output,
        {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "complete",
                "input": str(args.input),
                "input_metadata": input_metadata,
                "model": client.config.model,
                "sessions_per_condition": args.sessions,
                "full_threshold": args.full_threshold,
                "partial_threshold": args.partial_threshold,
                "criterion_note": (
                    "Paper-style validation: >=80% complete-information accuracy "
                    "and <=20% local-information pre-discussion accuracy by default."
                ),
            },
            "summary": overall,
            "results": results,
        },
        overwrite=True,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
