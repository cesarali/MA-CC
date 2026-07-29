"""Run one game and write reproducible evaluator-facing artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from .agent import create_agents
from .api_client import LLMClient
from .models import (
    BenchmarkRunSummary,
    GameResult,
    Inventory,
    RunSpec,
    UpdateMode,
)
from .reasoning_game import ReasoningTask
from .sequential_game import SequentialNamingGame
from .synchronous_game import SynchronousParallelNamingGame


@dataclass(frozen=True)
class CompletedRun:
    summary: BenchmarkRunSummary
    game_result: GameResult
    output_files: dict[str, Path]


def make_run_id(spec: RunSpec) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    model_slug = re.sub(r"[^A-Za-z0-9]+", "-", spec.model).strip("-")
    return (
        f"{timestamp}_{model_slug}_{spec.update_mode.value}_n{spec.num_agents}"
        f"_r{spec.synchronous_round_equivalent:g}_rep{spec.replicate}_{uuid.uuid4().hex[:8]}"
    )


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_run(
    run_id: str,
    spec: RunSpec,
    result: GameResult,
    *,
    api_backend: str,
) -> BenchmarkRunSummary:
    responses = [
        response
        for interaction in result.interactions
        for response in (interaction.speaker_response, interaction.listener_response)
    ]
    latencies = [response.latency_seconds for response in responses]
    actual_calls = sum(response.attempts for response in responses)
    retries = sum(response.retries for response in responses)

    def token_sum(field: str) -> int | None:
        values = [getattr(response.usage, field) for response in responses]
        present = [value for value in values if value is not None]
        return sum(present) if present else None

    successes = sum(
        interaction.naming_success is True for interaction in result.interactions
    )
    failures = sum(
        interaction.naming_success is False for interaction in result.interactions
    )
    round_walls = [record.round_wall_seconds for record in result.rounds]
    slowest_pairs = [record.slowest_pair_seconds for record in result.rounds]
    round_equivalent = spec.synchronous_round_equivalent
    expected_calls = 2 * spec.num_interactions

    return BenchmarkRunSummary(
        run_id=run_id,
        status="completed",
        error=None,
        api_backend=api_backend,
        model=spec.model,
        num_agents=spec.num_agents,
        reasoning_fraction=spec.reasoning_fraction,
        update_mode=spec.update_mode.value,
        synchronous_round_equivalent=round_equivalent,
        total_pair_interactions=len(result.interactions),
        expected_api_calls=expected_calls,
        actual_api_calls=actual_calls,
        random_seed=spec.seed,
        concurrency_limit=spec.concurrency,
        replicate=spec.replicate,
        total_wall_seconds=result.wall_seconds,
        seconds_per_pair_interaction=(
            result.wall_seconds / len(result.interactions)
            if result.interactions
            else None
        ),
        seconds_per_synchronous_round_equivalent=(
            result.wall_seconds / round_equivalent if round_equivalent > 0 else None
        ),
        seconds_per_actual_synchronous_round=(
            result.wall_seconds / len(result.rounds) if result.rounds else None
        ),
        successful_calls=len(responses),
        failed_calls=retries,
        retries=retries,
        mean_request_latency_seconds=(statistics.fmean(latencies) if latencies else None),
        median_request_latency_seconds=(statistics.median(latencies) if latencies else None),
        p90_request_latency_seconds=_percentile(latencies, 0.9),
        max_request_latency_seconds=max(latencies) if latencies else None,
        api_calls_per_second=(actual_calls / result.wall_seconds if result.wall_seconds else None),
        prompt_tokens=token_sum("prompt_tokens"),
        completion_tokens=token_sum("completion_tokens"),
        total_tokens=token_sum("total_tokens"),
        successful_naming_interactions=successes,
        failed_naming_interactions=failures,
        initial_count_a=result.initial_counts["A"],
        initial_count_b=result.initial_counts["B"],
        initial_count_ab=result.initial_counts["AB"],
        final_count_a=result.final_counts["A"],
        final_count_b=result.final_counts["B"],
        final_count_ab=result.final_counts["AB"],
        consensus_reached=result.consensus_reached,
        consensus_interaction_index=result.consensus_interaction_index,
        parallel_pairs_per_round=(
            json.dumps([record.parallel_pairs for record in result.rounds])
            if result.rounds
            else None
        ),
        mean_round_wall_seconds=(statistics.fmean(round_walls) if round_walls else None),
        mean_slowest_pair_seconds=(
            statistics.fmean(slowest_pairs) if slowest_pairs else None
        ),
        total_trajectory_seconds=(
            result.wall_seconds if spec.update_mode == UpdateMode.SEQUENTIAL else None
        ),
        independent_trajectories_concurrent=result.trajectory_concurrency > 1,
        concurrent_trajectories=result.trajectory_concurrency,
    )


def _write_csv(path: Path, fieldnames: list[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_run_outputs(
    *,
    output_dir: Path,
    run_id: str,
    spec: RunSpec,
    result: GameResult,
    api_backend: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    interactions_path = output_dir / f"interactions_{run_id}.jsonl"
    states_path = output_dir / f"states_{run_id}.csv"
    rounds_path = output_dir / f"rounds_{run_id}.csv"
    config_path = output_dir / f"config_{run_id}.json"

    with interactions_path.open("w", encoding="utf-8") as handle:
        for interaction in result.interactions:
            handle.write(json.dumps(interaction.to_log_dict(), sort_keys=True) + "\n")

    state_fields = ["interaction_index", "count_a", "count_b", "count_ab", "consensus"]
    _write_csv(
        states_path,
        state_fields,
        [asdict(record) for record in result.states],
    )
    round_fields = [
        "round_index",
        "interactions_completed",
        "parallel_pairs",
        "idle_agent_id",
        "round_wall_seconds",
        "slowest_pair_seconds",
        "count_a",
        "count_b",
        "count_ab",
        "consensus",
    ]
    _write_csv(
        rounds_path,
        round_fields,
        [asdict(record) for record in result.rounds],
    )
    config_payload = {
        "run_id": run_id,
        **asdict(spec),
        "update_mode": spec.update_mode.value,
        "api_backend": api_backend,
        "initial_counts": result.initial_counts,
    }
    config_path.write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "interactions": interactions_path,
        "states": states_path,
        "rounds": rounds_path,
        "config": config_path,
    }


def write_benchmark_summary(
    output_dir: Path,
    summaries: Sequence[BenchmarkRunSummary],
    *,
    append: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "benchmark_summary.csv"
    if not summaries:
        return path
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        rows = [summary.to_dict() for summary in summaries]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
    return path


async def run_single(
    *,
    spec: RunSpec,
    client: LLMClient,
    output_dir: str | Path,
    reasoning_task: ReasoningTask | None = None,
    initial_population: Sequence[Inventory] | None = None,
    write_summary: bool = True,
) -> CompletedRun:
    if spec.reasoning_fraction > 0 and reasoning_task is None:
        raise ValueError(
            "reasoning_fraction > 0 requires an explicit reasoning-task specification."
        )
    agents = create_agents(
        spec.num_agents,
        spec.seed,
        inventories=initial_population,
    )
    common = {
        "agents": agents,
        "client": client,
        "seed": spec.seed,
        "reasoning_fraction": spec.reasoning_fraction,
        "reasoning_task": reasoning_task,
        "temperature": spec.temperature,
        "max_tokens_speaker": spec.max_tokens_speaker,
        "max_tokens_listener": spec.max_tokens_listener,
    }
    if spec.update_mode == UpdateMode.SEQUENTIAL:
        game = SequentialNamingGame(**common)
        result = await game.run(spec.num_interactions)
    else:
        if spec.rounds is None:
            raise ValueError("Synchronous mode requires a round count.")
        expected_interactions = (spec.num_agents // 2) * spec.rounds
        if expected_interactions != spec.num_interactions:
            raise ValueError(
                "Synchronous num_interactions must equal floor(N / 2) * rounds."
            )
        game = SynchronousParallelNamingGame(**common)
        result = await game.run(spec.rounds)

    run_id = make_run_id(spec)
    api_backend = (
        "mock"
        if client.__class__.__name__ == "MockAsyncLLMClient"
        else "university_proxy"
    )
    summary = summarize_run(run_id, spec, result, api_backend=api_backend)
    destination = Path(output_dir)
    output_files = write_run_outputs(
        output_dir=destination,
        run_id=run_id,
        spec=spec,
        result=result,
        api_backend=api_backend,
    )
    if write_summary:
        output_files["summary"] = write_benchmark_summary(destination, [summary], append=True)
    return CompletedRun(summary=summary, game_result=result, output_files=output_files)
