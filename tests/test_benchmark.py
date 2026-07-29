import asyncio
import csv

import pytest

from naming_game.api_client import MockAsyncLLMClient
from naming_game.benchmark import (
    BenchmarkConfig,
    matched_interaction_budget,
    run_matched_benchmark,
)
from naming_game.models import ConfigurationError, UpdateMode


def test_matched_interaction_budget_examples():
    assert matched_interaction_budget(5, 5) == 10
    assert matched_interaction_budget(10, 10) == 50
    assert matched_interaction_budget(20, 20) == 200


def test_matched_modes_use_same_interactions_and_expected_calls(tmp_path):
    config = BenchmarkConfig(
        models=("mock/model",),
        concurrency=10,
        seed=1,
        update_modes=(UpdateMode.SEQUENTIAL, UpdateMode.SYNCHRONOUS_PARALLEL),
        agent_sizes=(5,),
        synchronous_round_counts=(3,),
        replicates=1,
    )
    benchmark = asyncio.run(
        run_matched_benchmark(
            config=config,
            client_factory=lambda model, seed: MockAsyncLLMClient(
                model=model, seed=seed, artificial_latency=0
            ),
            output_dir=tmp_path,
        )
    )
    assert len(benchmark.runs) == 2
    summaries = [run.summary for run in benchmark.runs]
    assert {summary.total_pair_interactions for summary in summaries} == {6}
    assert {summary.expected_api_calls for summary in summaries} == {12}
    assert {summary.actual_api_calls for summary in summaries} == {12}
    assert {(summary.initial_count_a, summary.initial_count_b) for summary in summaries}.__len__() == 1


def test_mock_benchmark_creates_every_required_result_file(tmp_path):
    config = BenchmarkConfig(
        models=("mock/model",),
        agent_sizes=(5,),
        synchronous_round_counts=(1,),
        update_modes=(UpdateMode.SEQUENTIAL, UpdateMode.SYNCHRONOUS_PARALLEL),
    )
    benchmark = asyncio.run(
        run_matched_benchmark(
            config=config,
            client_factory=lambda model, seed: MockAsyncLLMClient(
                model=model, seed=seed, artificial_latency=0
            ),
            output_dir=tmp_path,
        )
    )
    assert benchmark.summary_path.exists()
    for completed in benchmark.runs:
        assert set(completed.output_files) == {
            "interactions",
            "states",
            "rounds",
            "config",
        }
        assert all(path.exists() for path in completed.output_files.values())
    with benchmark.summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["model"] for row in rows} == {"mock/model"}


def test_reasoning_fraction_requires_explicit_task(tmp_path):
    config = BenchmarkConfig(
        models=("mock/model",),
        agent_sizes=(5,),
        synchronous_round_counts=(1,),
        reasoning_fraction=0.1,
    )
    with pytest.raises(ConfigurationError, match="reasoning-task"):
        asyncio.run(
            run_matched_benchmark(
                config=config,
                client_factory=lambda model, seed: MockAsyncLLMClient(model=model),
                output_dir=tmp_path,
            )
        )
