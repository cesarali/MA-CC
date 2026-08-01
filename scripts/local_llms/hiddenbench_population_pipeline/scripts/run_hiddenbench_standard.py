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
    task_id,
    write_json,
)
from hiddenbench_evaluation import run_standard_session
from hiddenbench_llm_api import LLMClient, LLMConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper-style HiddenBench protocol on the source benchmark or "
            "on any scaled population dataset."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hiddenbench_standard.json"),
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=10,
        help="The paper uses ten sessions per task and condition.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=15,
        help=(
            "Number of sequential public speaking events. The paper denotes this "
            "communication depth by T and uses T=15 in the main condition."
        ),
    )
    parser.add_argument(
        "--speaker-order",
        choices=["round_robin", "random"],
        default="round_robin",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help=(
            "Stop after unanimous interim votes. Disabled by default because the "
            "paper runs the fixed communication depth."
        ),
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--task-ids", type=int, nargs="*")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def aggregate_task_sessions(
    sessions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "average_pre_accuracy": mean(
            [session["metrics"]["pre"]["average_accuracy"] for session in sessions]
        ),
        "average_post_accuracy": mean(
            [session["metrics"]["post"]["average_accuracy"] for session in sessions]
        ),
        "average_full_profile_accuracy": mean(
            [session["metrics"]["full"]["average_accuracy"] for session in sessions]
        ),
        "pre_majority_accuracy": mean(
            [
                float(session["metrics"]["pre"]["majority_correct"])
                for session in sessions
            ]
        ),
        "post_majority_accuracy": mean(
            [
                float(session["metrics"]["post"]["majority_correct"])
                for session in sessions
            ]
        ),
        "full_majority_accuracy": mean(
            [
                float(session["metrics"]["full"]["majority_correct"])
                for session in sessions
            ]
        ),
        "unanimous_post_fraction": mean(
            [
                float(session["metrics"]["post"]["unanimous"])
                for session in sessions
            ]
        ),
    }


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "num_tasks": len(results),
        "average_pre_accuracy": mean(
            [item["summary"]["average_pre_accuracy"] for item in results]
        ),
        "average_post_accuracy": mean(
            [item["summary"]["average_post_accuracy"] for item in results]
        ),
        "average_full_profile_accuracy": mean(
            [
                item["summary"]["average_full_profile_accuracy"]
                for item in results
            ]
        ),
        "average_information_integration_gain": mean(
            [
                item["summary"]["average_post_accuracy"]
                - item["summary"]["average_pre_accuracy"]
                for item in results
            ]
        ),
        "average_collective_reasoning_gap": mean(
            [
                item["summary"]["average_full_profile_accuracy"]
                - item["summary"]["average_post_accuracy"]
                for item in results
            ]
        ),
        "post_majority_accuracy": mean(
            [item["summary"]["post_majority_accuracy"] for item in results]
        ),
    }


def main() -> None:
    args = parse_args()
    tasks, input_metadata = extract_tasks_payload(args.input)
    tasks = select_tasks(tasks, args.task_ids)
    client = LLMClient(
        LLMConfig.from_env(
            model_env="LLM_BENCHMARK_MODEL",
            default_model="microsoft/gpt-5.5",
        )
    )

    results = []
    for task in tasks:
        sessions = []
        for session_index in range(args.sessions):
            print(
                f"task={task_id(task)} session={session_index + 1}/{args.sessions}",
                flush=True,
            )
            sessions.append(
                run_standard_session(
                    client,
                    task,
                    session_index=session_index,
                    communication_rounds=args.rounds,
                    base_seed=args.seed,
                    temperature=args.temperature,
                    early_stop=args.early_stop,
                    speaker_order=args.speaker_order,
                )
            )

        results.append(
            {
                "task": {
                    "task_id": task_id(task),
                    "name": task["name"],
                    "correct_answer": task["correct_answer"],
                    "population": task.get("population"),
                },
                "summary": aggregate_task_sessions(sessions),
                "sessions": sessions,
            }
        )

        # Checkpoint after every task.
        write_json(
            args.output,
            {
                "metadata": {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "in_progress",
                    "input": str(args.input),
                    "input_metadata": input_metadata,
                    "model": client.config.model,
                    "sessions_per_task": args.sessions,
                    "communication_rounds": args.rounds,
                    "speaker_order": args.speaker_order,
                    "early_stop": args.early_stop,
                    "base_seed": args.seed,
                    "protocol_note": (
                        "Paper-style pre-discussion, sequential broadcast discussion, "
                        "post-discussion, and Full Profile voting. One communication "
                        "round is implemented as one public speaking event."
                    ),
                },
                "summary": aggregate_results(results),
                "results": results,
            },
            overwrite=True,
        )

    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "input": str(args.input),
            "input_metadata": input_metadata,
            "model": client.config.model,
            "sessions_per_task": args.sessions,
            "communication_rounds": args.rounds,
            "speaker_order": args.speaker_order,
            "early_stop": args.early_stop,
            "base_seed": args.seed,
            "protocol_note": (
                "Paper-style pre-discussion, sequential broadcast discussion, "
                "post-discussion, and Full Profile voting. One communication "
                "round is implemented as one public speaking event."
            ),
        },
        "summary": aggregate_results(results),
        "results": results,
    }
    write_json(args.output, payload, overwrite=True)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
