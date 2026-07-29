"""Command-line interface for individual runs and matched benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from .analysis.empowerment import AnalysisConfig, analyze_histories
from .api_client import (
    AsyncLLMClient,
    LLMAPIError,
    MockAsyncLLMClient,
    OpenAIAsyncLLMClient,
)
from .benchmark import (
    BenchmarkConfig,
    load_benchmark_config,
    override_config,
    run_matched_benchmark,
)
from .models import ConfigurationError, RunSpec, UpdateMode
from .empowerment_experiment import load_experiment_config, run_experiment
from .reasoning_game import load_reasoning_task
from .runner import run_single


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one Naming Game trajectory.")
    run.add_argument("--update-mode", choices=[mode.value for mode in UpdateMode], required=True)
    run.add_argument("--num-agents", type=int, required=True)
    run.add_argument("--num-interactions", type=int)
    run.add_argument("--rounds", type=int)
    run.add_argument("--reasoning-fraction", type=float, default=0.0)
    run.add_argument("--reasoning-task", type=Path)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--concurrency", type=int, default=20)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-tokens-speaker", type=int, default=20)
    run.add_argument("--max-tokens-listener", type=int, default=20)
    run.add_argument("--timeout-seconds", type=float, default=60.0)
    run.add_argument("--max-retries", type=int, default=2)
    run.add_argument("--model", default="gwdg/qwen3-30b-a3b-instruct-2507")
    run.add_argument("--mock", action="store_true")
    run.add_argument("--mock-latency", type=float, default=0.001)
    run.add_argument("--output-dir", type=Path, default=Path("results"))

    benchmark = subparsers.add_parser("benchmark", help="Run the matched benchmark grid.")
    benchmark.add_argument("--config", type=Path, default=Path("configs/speed_test.yaml"))
    benchmark.add_argument("--models", nargs="+")
    benchmark.add_argument("--agent-sizes", nargs="+", type=int)
    benchmark.add_argument("--synchronous-round-counts", nargs="+", type=int)
    benchmark.add_argument(
        "--update-modes", nargs="+", choices=[mode.value for mode in UpdateMode]
    )
    benchmark.add_argument("--reasoning-fraction", type=float)
    benchmark.add_argument("--reasoning-task", type=Path)
    benchmark.add_argument("--replicates", type=int)
    benchmark.add_argument("--concurrency", type=int)
    benchmark.add_argument("--seed", type=int)
    benchmark.add_argument("--timeout-seconds", type=float)
    benchmark.add_argument("--max-retries", type=int)
    benchmark.add_argument("--mock", action="store_true")
    benchmark.add_argument("--mock-latency", type=float, default=0.001)
    benchmark.add_argument("--output-dir", type=Path, default=Path("results"))

    experiment = subparsers.add_parser(
        "experiment", help="Run committee-empowerment episodes from YAML."
    )
    experiment.add_argument("--config", type=Path, required=True)
    experiment.add_argument("--output-dir", type=Path, default=Path("results/empowerment"))
    experiment.add_argument("--mock", action="store_true")
    experiment.add_argument("--no-resume", action="store_true")

    analyze = subparsers.add_parser(
        "analyze-empowerment", help="Analyze existing empowerment Parquet histories."
    )
    analyze.add_argument("--history-dir", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, default=Path("results/empowerment_analysis"))
    analyze.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5, 10])
    analyze.add_argument("--bootstrap-resamples", type=int, default=1000)
    analyze.add_argument("--null-permutations", type=int, default=1000)
    analyze.add_argument("--seed", type=int, default=1)
    return parser


async def _run_one(args: argparse.Namespace) -> int:
    mode = UpdateMode(args.update_mode)
    pairs_per_round = args.num_agents // 2
    if mode == UpdateMode.SEQUENTIAL:
        if args.num_interactions is None:
            raise ConfigurationError("Sequential mode requires --num-interactions.")
        if args.rounds is not None:
            raise ConfigurationError("Sequential mode does not accept --rounds.")
        interactions = args.num_interactions
        round_equivalent = interactions / pairs_per_round
        rounds = None
    else:
        if args.rounds is None:
            raise ConfigurationError("Synchronous mode requires --rounds.")
        if args.num_interactions is not None:
            raise ConfigurationError("Synchronous mode derives interactions from --rounds.")
        rounds = args.rounds
        interactions = pairs_per_round * rounds
        round_equivalent = float(rounds)

    reasoning_task = (
        load_reasoning_task(args.reasoning_task) if args.reasoning_task else None
    )
    spec = RunSpec(
        model=args.model,
        num_agents=args.num_agents,
        reasoning_fraction=args.reasoning_fraction,
        update_mode=mode,
        synchronous_round_equivalent=round_equivalent,
        num_interactions=interactions,
        rounds=rounds,
        seed=args.seed,
        concurrency=args.concurrency,
        temperature=args.temperature,
        max_tokens_speaker=args.max_tokens_speaker,
        max_tokens_listener=args.max_tokens_listener,
    )
    client = (
        MockAsyncLLMClient(
            model=args.model,
            concurrency=args.concurrency,
            artificial_latency=args.mock_latency,
            seed=args.seed,
        )
        if args.mock
        else AsyncLLMClient(
            model=args.model,
            concurrency=args.concurrency,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
    )
    try:
        completed = await run_single(
            spec=spec,
            client=client,
            output_dir=args.output_dir,
            reasoning_task=reasoning_task,
        )
    finally:
        client.close()
    print(json.dumps(completed.summary.to_dict(), indent=2, sort_keys=True))
    return 0


async def _run_benchmark(args: argparse.Namespace) -> int:
    config = load_benchmark_config(args.config)
    config = override_config(
        config,
        models=tuple(args.models) if args.models else None,
        agent_sizes=tuple(args.agent_sizes) if args.agent_sizes else None,
        synchronous_round_counts=(
            tuple(args.synchronous_round_counts)
            if args.synchronous_round_counts
            else None
        ),
        update_modes=(
            tuple(UpdateMode(mode) for mode in args.update_modes)
            if args.update_modes
            else None
        ),
        reasoning_fraction=args.reasoning_fraction,
        replicates=args.replicates,
        concurrency=args.concurrency,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    reasoning_task = (
        load_reasoning_task(args.reasoning_task) if args.reasoning_task else None
    )

    if args.mock:

        def client_factory(model: str, seed: int) -> MockAsyncLLMClient:
            return MockAsyncLLMClient(
                model=model,
                concurrency=config.concurrency,
                artificial_latency=args.mock_latency,
                seed=seed,
            )

    else:

        def client_factory(model: str, seed: int) -> AsyncLLMClient:
            del seed
            return AsyncLLMClient(
                model=model,
                concurrency=config.concurrency,
                timeout_seconds=config.timeout_seconds,
                max_retries=config.max_retries,
            )

    benchmark = await run_matched_benchmark(
        config=config,
        client_factory=client_factory,
        output_dir=args.output_dir,
        reasoning_task=reasoning_task,
    )
    print(
        json.dumps(
            {
                "runs_completed": len(benchmark.runs),
                "summary": str(benchmark.summary_path),
                "models": list(config.models),
                "concurrency": config.concurrency,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _provider_for(config, provider: str, model: str):
    common = {
        "model": model,
        "concurrency": config.request_concurrency,
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
    }
    if provider == "university":
        return AsyncLLMClient(**common)
    if provider == "openai":
        return OpenAIAsyncLLMClient(**common)
    raise ConfigurationError(f"Unknown provider: {provider!r}.")


async def _run_empowerment_experiment(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    if args.mock:
        client = MockAsyncLLMClient(
            model=config.model,
            concurrency=config.request_concurrency,
            artificial_latency=0,
            seed=config.seed,
        )
    else:
        client = _provider_for(config, config.provider, config.model)
        validate = getattr(client, "validate_model", None)
        try:
            if callable(validate):
                await validate()
        except (ConfigurationError, LLMAPIError):
            client.close()
            if not config.allow_fallback:
                raise
            fallback_model = config.fallback_model or config.model
            client = _provider_for(config, config.fallback_provider, fallback_model)
            fallback_validate = getattr(client, "validate_model", None)
            if callable(fallback_validate):
                await fallback_validate()
    try:
        result = await run_experiment(
            config, client, args.output_dir, resume=not args.no_resume
        )
    finally:
        client.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _analyze_empowerment(args: argparse.Namespace) -> int:
    result = analyze_histories(
        args.history_dir,
        args.output_dir,
        AnalysisConfig(
            horizons_population_rounds=tuple(args.horizons),
            bootstrap_resamples=args.bootstrap_resamples,
            null_permutations=args.null_permutations,
            seed=args.seed,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return asyncio.run(_run_one(args))
        if args.command == "benchmark":
            return asyncio.run(_run_benchmark(args))
        if args.command == "experiment":
            return asyncio.run(_run_empowerment_experiment(args))
        return _analyze_empowerment(args)
    except (ConfigurationError, LLMAPIError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
