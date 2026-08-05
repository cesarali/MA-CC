from dataclasses import replace
from pathlib import Path

import pytest

from mas_cc.config import RunConfig, load_run_config
from mas_cc.experiments import run_experiment_sync
from mas_cc.games import create_game
from mas_cc.llm_runtime.providers import BudgetLimits, OfflinePricingSource
from mas_cc.planning import static_experiment_preflight, static_game_preflight


def _toy_config(**execution_overrides):
    config = load_run_config("configs/runs/toy_game_smoke_test.yaml", environment={})
    return replace(config, execution=replace(config.execution, **execution_overrides))


def _naming_convention_config(**execution_overrides):
    config = load_run_config("configs/runs/naming_convention_smoke_test_v3.yaml", environment={})
    return replace(config, execution=replace(config.execution, **execution_overrides))


def _with_ample_budget(config: RunConfig) -> RunConfig:
    """Enough headroom on every dimension for several episodes at once.

    Individual tests intentionally exercise the real request/output-token
    budget guard instead (see test_run_experiment_denies_launch_...); this
    helper is only for tests where the budget is not the thing under test.
    """

    return replace(
        config,
        budget=replace(
            config.budget,
            max_provider_requests=10_000, max_input_tokens=10_000_000,
            max_output_tokens=10_000, max_cost_per_run=100.0, system_max_cost_per_run=100.0,
        ),
    )


def test_static_experiment_preflight_multiplies_the_per_episode_estimate():
    config = _toy_config()
    game = create_game(config.game)
    plan = game.call_plan(config.game)
    quote = OfflinePricingSource().fetch(config.llm_provider.type, config.llm_provider.model)

    per_episode = static_game_preflight(
        plan, config.prompt, config.llm_provider, pricing_quote=quote, assumed_output_tokens=1,
    )
    estimate = static_experiment_preflight(
        plan, config.prompt, config.llm_provider, episode_count=5, concurrency=2,
        pricing_quote=quote, assumed_output_tokens=1,
    )
    assert estimate.episode_count == 5
    assert estimate.total_provider_requests.expected == per_episode.provider_requests.expected * 5
    assert estimate.total_input_tokens.expected == per_episode.input_tokens.expected * 5
    assert estimate.total_output_tokens.conservative == per_episode.output_tokens.conservative * 5
    assert estimate.launch_status == "permitted"


def test_static_experiment_preflight_denies_when_total_demand_exceeds_a_run_budget():
    config = _toy_config()
    game = create_game(config.game)
    plan = game.call_plan(config.game)
    quote = OfflinePricingSource().fetch(config.llm_provider.type, config.llm_provider.model)
    per_episode = static_game_preflight(
        plan, config.prompt, config.llm_provider, pricing_quote=quote, assumed_output_tokens=1,
    )
    # One episode's conservative demand fits; three episodes' does not.
    tight_budget = BudgetLimits(max_requests=per_episode.provider_requests.conservative + 1)
    single = static_experiment_preflight(
        plan, config.prompt, config.llm_provider, episode_count=1,
        pricing_quote=quote, assumed_output_tokens=1, run_budget=tight_budget,
    )
    triple = static_experiment_preflight(
        plan, config.prompt, config.llm_provider, episode_count=3,
        pricing_quote=quote, assumed_output_tokens=1, run_budget=tight_budget,
    )
    assert single.launch_status == "permitted"
    assert triple.launch_status == "denied"


def test_run_experiment_runs_episodes_concurrently_with_distinct_deterministic_seeds(tmp_path: Path):
    config = _with_ample_budget(_toy_config(repetitions=4, parallelism=2))

    def once(output_dir: Path):
        return run_experiment_sync(config, output_dir, resume=False, show_progress=False)

    first = once(tmp_path / "a")
    second = once(tmp_path / "b")

    assert first.completed == 4
    assert first.failed == 0
    seeds_first = [outcome.seed for outcome in first.outcomes]
    seeds_second = [outcome.seed for outcome in second.outcomes]
    assert seeds_first == seeds_second
    assert len(set(seeds_first)) == 4  # every episode gets an independent seed


def test_run_experiment_resume_skips_only_completed_episodes(tmp_path: Path):
    config = _with_ample_budget(_toy_config(repetitions=2, parallelism=1))

    first = run_experiment_sync(config, tmp_path, resume=True, show_progress=False)
    assert first.completed == 2
    assert first.skipped_resumed == 0

    second = run_experiment_sync(config, tmp_path, resume=True, show_progress=False)
    assert second.completed == 0
    assert second.skipped_resumed == 2

    third = run_experiment_sync(config, tmp_path, resume=False, show_progress=False)
    assert third.completed == 2
    assert third.skipped_resumed == 0


def test_run_experiment_fail_fast_aborts_remaining_episodes(tmp_path: Path, monkeypatch):
    config = _with_ample_budget(_toy_config(repetitions=3, parallelism=1, fail_fast=True))

    calls: list[int] = []

    from mas_cc.games.runner import run_game as real_run_game

    async def flaky_run_game(game, episode_config, provider, **kwargs):
        calls.append(episode_config.execution.seed)
        if len(calls) == 1:
            raise RuntimeError("simulated episode failure")
        return await real_run_game(game, episode_config, provider, **kwargs)

    monkeypatch.setattr("mas_cc.experiments.orchestrator.run_game", flaky_run_game)

    result = run_experiment_sync(config, tmp_path, resume=False, show_progress=False)
    statuses = [outcome.status for outcome in result.outcomes]
    assert statuses == ["failed", "skipped_aborted", "skipped_aborted"]
    assert len(calls) == 1  # aborted episodes never even attempted a provider call


def test_run_experiment_denies_launch_when_total_demand_exceeds_budget(tmp_path: Path):
    config = _toy_config(repetitions=3, parallelism=1)  # unmodified budget fits exactly one episode
    with pytest.raises(ValueError, match="preflight launch status"):
        run_experiment_sync(config, tmp_path, resume=False, show_progress=False)
    assert not any(tmp_path.iterdir())


def test_run_experiment_writes_experiment_summary_and_episode_manifests(tmp_path: Path):
    config = _with_ample_budget(_toy_config(repetitions=2, parallelism=2))
    result = run_experiment_sync(config, tmp_path, resume=False, show_progress=False)

    summary = result.output_dir / "experiment_summary.json"
    assert summary.is_file()
    episodes_dir = result.output_dir / "data" / "episodes"
    episode_dirs = sorted(path.name for path in episodes_dir.iterdir())
    assert len(episode_dirs) == 2
    for name in episode_dirs:
        assert (episodes_dir / name / "manifest.json").is_file()


def test_run_experiment_naming_convention_dispatch_wires_recorder_and_metrics(tmp_path: Path):
    config = _with_ample_budget(_naming_convention_config(repetitions=2, parallelism=2))
    result = run_experiment_sync(config, tmp_path, resume=False, show_progress=False)

    assert result.completed == 2
    episodes_dir = result.output_dir / "data" / "episodes"
    for episode_dir in sorted(episodes_dir.iterdir()):
        assert (episode_dir / "events.jsonl").is_file()
        assert (episode_dir / "checkpoint_manifest.json").is_file()
        assert (episode_dir / "metrics" / "final.csv").is_file()
        comet_summary = (episode_dir / "comet_summary.json").read_text(encoding="utf-8")
        assert '"status": "disabled"' in comet_summary  # per-episode Comet is deliberately off
