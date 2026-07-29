"""Matched interaction-budget benchmark orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .agent import initial_inventories
from .api_client import LLMClient
from .models import ConfigurationError, RunSpec, UpdateMode
from .reasoning_game import ReasoningTask
from .runner import CompletedRun, run_single, write_benchmark_summary

REQUIRED_BENCHMARK_MODELS = (
    "gwdg/qwen3-30b-a3b-instruct-2507",
    "microsoft/gpt-4o",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    models: tuple[str, ...] = REQUIRED_BENCHMARK_MODELS
    temperature: float = 0.0
    max_tokens_speaker: int = 20
    max_tokens_listener: int = 20
    timeout_seconds: float = 60.0
    max_retries: int = 2
    concurrency: int = 20
    seed: int = 1
    reasoning_fraction: float = 0.0
    update_modes: tuple[UpdateMode, ...] = (
        UpdateMode.SEQUENTIAL,
        UpdateMode.SYNCHRONOUS_PARALLEL,
    )
    agent_sizes: tuple[int, ...] = (5, 10, 20)
    synchronous_round_counts: tuple[int, ...] = (5, 10, 20)
    replicates: int = 1

    def __post_init__(self) -> None:
        if not self.models:
            raise ConfigurationError("At least one benchmark model is required.")
        if not 0 <= self.reasoning_fraction <= 1:
            raise ConfigurationError("reasoning_fraction must be between 0 and 1.")
        if self.concurrency < 1:
            raise ConfigurationError("concurrency must be at least 1.")
        if self.replicates < 1:
            raise ConfigurationError("replicates must be at least 1.")
        if any(size < 2 for size in self.agent_sizes):
            raise ConfigurationError("All agent sizes must be at least 2.")
        if any(rounds < 1 for rounds in self.synchronous_round_counts):
            raise ConfigurationError("All synchronous round counts must be positive.")


@dataclass(frozen=True)
class BenchmarkResult:
    runs: tuple[CompletedRun, ...]
    summary_path: Path


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Could not load benchmark config: {source}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("Benchmark configuration must be a YAML mapping.")
    try:
        return BenchmarkConfig(
            models=tuple(str(model) for model in raw["models"]),
            temperature=float(raw.get("temperature", 0.0)),
            max_tokens_speaker=int(raw.get("max_tokens_speaker", 20)),
            max_tokens_listener=int(raw.get("max_tokens_listener", 20)),
            timeout_seconds=float(raw.get("timeout_seconds", 60)),
            max_retries=int(raw.get("max_retries", 2)),
            concurrency=int(raw.get("concurrency", 20)),
            seed=int(raw.get("seed", 1)),
            reasoning_fraction=float(raw.get("reasoning_fraction", 0.0)),
            update_modes=tuple(UpdateMode(mode) for mode in raw["update_modes"]),
            agent_sizes=tuple(int(size) for size in raw["agent_sizes"]),
            synchronous_round_counts=tuple(
                int(rounds) for rounds in raw["synchronous_round_counts"]
            ),
            replicates=int(raw.get("replicates", 1)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError("Benchmark configuration has invalid or missing fields.") from exc


def matched_interaction_budget(num_agents: int, synchronous_rounds: int) -> int:
    if num_agents < 2 or synchronous_rounds < 0:
        raise ValueError("num_agents must be at least 2 and rounds cannot be negative.")
    return (num_agents // 2) * synchronous_rounds


async def run_matched_benchmark(
    *,
    config: BenchmarkConfig,
    client_factory: Callable[[str, int], LLMClient],
    output_dir: str | Path = "results",
    reasoning_task: ReasoningTask | None = None,
) -> BenchmarkResult:
    if config.reasoning_fraction > 0 and reasoning_task is None:
        raise ConfigurationError(
            "reasoning_fraction > 0 requires an explicit reasoning-task specification."
        )
    destination = Path(output_dir)
    completed: list[CompletedRun] = []
    summary_path = destination / "benchmark_summary.csv"

    for model in config.models:
        for num_agents in config.agent_sizes:
            for synchronous_rounds in config.synchronous_round_counts:
                interactions = matched_interaction_budget(num_agents, synchronous_rounds)
                for replicate in range(config.replicates):
                    run_seed = config.seed + replicate
                    matched_population = initial_inventories(num_agents, run_seed)
                    for update_mode in config.update_modes:
                        spec = RunSpec(
                            model=model,
                            num_agents=num_agents,
                            reasoning_fraction=config.reasoning_fraction,
                            update_mode=update_mode,
                            synchronous_round_equivalent=float(synchronous_rounds),
                            num_interactions=interactions,
                            rounds=(
                                synchronous_rounds
                                if update_mode == UpdateMode.SYNCHRONOUS_PARALLEL
                                else None
                            ),
                            seed=run_seed,
                            concurrency=config.concurrency,
                            replicate=replicate,
                            temperature=config.temperature,
                            max_tokens_speaker=config.max_tokens_speaker,
                            max_tokens_listener=config.max_tokens_listener,
                        )
                        client = client_factory(model, run_seed)
                        try:
                            completed_run = await run_single(
                                spec=spec,
                                client=client,
                                output_dir=destination,
                                reasoning_task=reasoning_task,
                                initial_population=matched_population,
                                write_summary=False,
                            )
                            completed.append(completed_run)
                            summary_path = write_benchmark_summary(
                                destination, [completed_run.summary], append=True
                            )
                        finally:
                            close = getattr(client, "close", None)
                            if callable(close):
                                close()

    return BenchmarkResult(runs=tuple(completed), summary_path=summary_path)


def override_config(config: BenchmarkConfig, **updates: Any) -> BenchmarkConfig:
    """Create a validated CLI override without mutating loaded configuration."""

    values = {
        field: getattr(config, field)
        for field in BenchmarkConfig.__dataclass_fields__  # type: ignore[attr-defined]
    }
    values.update({key: value for key, value in updates.items() if value is not None})
    return BenchmarkConfig(**values)
