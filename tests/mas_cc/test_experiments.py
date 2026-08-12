import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from mas_cc.config import RunConfig, load_run_config
from mas_cc.experiments import run_experiment_sync
from mas_cc.experiments.orchestrator import _LiveSpendWatcher
from mas_cc.games import create_game
from mas_cc.llm_runtime.providers import (
    AccountBudget,
    BudgetLimits,
    OfflinePricingSource,
    ProviderError,
    RuntimeBudgetGuard,
)
from mas_cc.planning import static_experiment_preflight, static_game_preflight


def _toy_config(**execution_overrides):
    config = load_run_config("configs/runs/old/toy_game_smoke_test.yaml", environment={})
    return replace(config, execution=replace(config.execution, **execution_overrides))


def _naming_convention_config(**execution_overrides):
    config = load_run_config("configs/runs/old/naming_convention_smoke_test_v3.yaml", environment={})
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
    analysis_calls: list[Path] = []
    monkeypatch.setattr(
        "mas_cc.experiments.orchestrator.run_configured_analysis",
        lambda _config, run_dir: analysis_calls.append(Path(run_dir)),
    )

    result = run_experiment_sync(config, tmp_path, resume=False, show_progress=False)
    statuses = [outcome.status for outcome in result.outcomes]
    assert statuses == ["failed", "skipped_aborted", "skipped_aborted"]
    assert len(calls) == 1  # aborted episodes never even attempted a provider call
    assert analysis_calls == []  # no secondary missing-trajectory error masks the failure


def test_a_budget_stop_skips_the_rest_of_the_run_even_without_fail_fast(
    tmp_path: Path, monkeypatch
):
    """One exhausted budget must stop the run, not fail every queued episode.

    From results/DIAGNOSIS.md: a 50-cell grid ran `fail_fast: false`, exhausted
    its token budget in cell 0001, and then recorded 4,235 *failed* episodes -
    each one dispatched only to be refused on its first provider call. The
    guard's counters only move one way, so once it denies, nothing later can
    succeed; the honest record is that those episodes never ran.
    """

    config = _with_ample_budget(_toy_config(repetitions=4, parallelism=1, fail_fast=False))

    calls: list[int] = []

    async def exhausted_run_game(game, episode_config, provider, **kwargs):
        calls.append(episode_config.execution.seed)
        raise ProviderError(
            "Approved runtime input-token budget would be exceeded.",
            provider="budget_guard", code="budget_exhausted",
        )

    monkeypatch.setattr("mas_cc.experiments.orchestrator.run_game", exhausted_run_game)
    monkeypatch.setattr(
        "mas_cc.experiments.orchestrator.run_configured_analysis",
        lambda *_args, **_kwargs: None,
    )

    result = run_experiment_sync(config, tmp_path, resume=False, show_progress=False)
    statuses = [outcome.status for outcome in result.outcomes]
    assert statuses == ["failed", "skipped_aborted", "skipped_aborted", "skipped_aborted"]
    assert len(calls) == 1, "no episode may be dispatched after the budget stop"
    # The reason survives into the per-episode record, so the summary says why
    # the run is short rather than leaving a wall of identical failures.
    skipped = [outcome for outcome in result.outcomes if outcome.status == "skipped_aborted"]
    assert {outcome.error_type for outcome in skipped} == {"BudgetStop"}


def test_an_ordinary_failure_without_fail_fast_still_runs_every_episode(
    tmp_path: Path, monkeypatch
):
    """Only budget denials abort a `fail_fast: false` run; bad luck does not."""

    config = _with_ample_budget(_toy_config(repetitions=3, parallelism=1, fail_fast=False))

    calls: list[int] = []
    from mas_cc.games.runner import run_game as real_run_game

    async def flaky_run_game(game, episode_config, provider, **kwargs):
        calls.append(episode_config.execution.seed)
        if len(calls) == 1:
            raise RuntimeError("simulated episode failure")
        return await real_run_game(game, episode_config, provider, **kwargs)

    monkeypatch.setattr("mas_cc.experiments.orchestrator.run_game", flaky_run_game)
    monkeypatch.setattr(
        "mas_cc.experiments.orchestrator.run_configured_analysis",
        lambda *_args, **_kwargs: None,
    )

    result = run_experiment_sync(config, tmp_path, resume=False, show_progress=False)
    assert [outcome.status for outcome in result.outcomes] == [
        "failed", "completed", "completed",
    ]
    assert len(calls) == 3


class _FakeSpendSource:
    """Stands in for the University proxy's `/user/info` accounting."""

    def __init__(self, amounts, unit="proxy_accounting_unit"):
        self.amounts = list(amounts)
        self.unit = unit
        self.calls = 0

    def fetch_account_budget(self, *, unit=None):
        self.calls += 1
        amount = self.amounts[min(self.calls - 1, len(self.amounts) - 1)]
        if amount is None:
            raise RuntimeError("simulated metadata outage")
        return AccountBudget(spent=_spend_money(amount, unit or self.unit))


def _spend_money(amount, unit):
    from mas_cc.llm_runtime.providers import MonetaryAmount

    return MonetaryAmount(
        amount, unit, "fixture", "university", "chat-model", "GET /user/info",
        "2026-08-12T00:00:00Z", "fixture-v1",
    )


def _watcher(source, guard, ceiling):
    return _LiveSpendWatcher(
        source, guard, unit="proxy_accounting_unit", ceiling=ceiling, poll_seconds=0
    )


def test_live_spend_stops_the_run_on_the_delta_since_launch_not_the_account_total():
    """A shared key's pre-existing spend must not count against this run.

    The account already sits at 100 when the run starts, and the ceiling is 5.
    An absolute reading would stop instantly; only the delta is this run's bill.
    """

    guard = RuntimeBudgetGuard(BudgetLimits(max_cost=_spend_money(5, "proxy_accounting_unit")))
    source = _FakeSpendSource([100.0, 102.0, 104.0, 106.0])
    watcher = _watcher(source, guard, ceiling=5.0)

    async def drive():
        await watcher.start()
        await watcher.run()

    asyncio.run(drive())

    assert guard.stop_reason is not None
    assert "106" not in guard.stop_reason, "the reason must quote the run's spend, not the total"
    spend = guard.status()["provider_account_spend"]
    assert spend["baseline_at_launch"] == 100.0
    assert spend["spent_by_this_run"] == pytest.approx(6.0)
    with pytest.raises(ProviderError) as captured:
        guard.reserve(conservative_cost=None, input_tokens=1, output_tokens=1)
    assert captured.value.code == "budget_stopped"


def test_a_failing_spend_poll_never_kills_a_healthy_run():
    """The guard's own cost ceiling is still in force, so a flaky endpoint is survivable."""

    guard = RuntimeBudgetGuard(BudgetLimits(max_cost=_spend_money(5, "proxy_accounting_unit")))
    # Baseline reads fine, then the endpoint fails twice, then recovers past the ceiling.
    source = _FakeSpendSource([10.0, None, None, 20.0])
    watcher = _watcher(source, guard, ceiling=5.0)

    async def drive():
        await watcher.start()
        await watcher.run()

    asyncio.run(drive())

    assert source.calls == 4, "the watcher must retry rather than give up on the first error"
    assert guard.stop_reason is not None


def test_an_account_with_no_spend_reporting_disables_watching_instead_of_failing():
    guard = RuntimeBudgetGuard(BudgetLimits(max_cost=_spend_money(5, "proxy_accounting_unit")))

    class _Silent:
        calls = 0

        def fetch_account_budget(self, *, unit=None):
            type(self).calls += 1
            return None

    source = _Silent()
    watcher = _watcher(source, guard, ceiling=5.0)

    async def drive():
        await watcher.start()
        await watcher.run()  # must return immediately, not loop forever

    asyncio.run(asyncio.wait_for(drive(), timeout=5))
    assert source.calls == 1
    assert guard.stop_reason is None


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


def test_run_experiment_writes_one_prompt_and_response_markdown_per_configured_round(
    tmp_path: Path,
):
    config = _with_ample_budget(_naming_convention_config(repetitions=1, parallelism=1))
    config = replace(
        config,
        logging=replace(
            config.logging,
            options={**config.logging.options, "prompt_examples": {"count": 12}},
        ),
    )

    result = run_experiment_sync(config, tmp_path, resume=False, show_progress=False)

    episode_dir = next((result.output_dir / "data" / "episodes").iterdir())
    prompt_files = sorted((episode_dir / "prompts").glob("round_*.md"))
    assert [path.name for path in prompt_files] == [
        f"round_{round_index:03d}.md" for round_index in range(1, 13)
    ]
    for path in prompt_files:
        markdown = path.read_text(encoding="utf-8")
        assert "## Exact messages sent to the LLM" in markdown
        assert "## Raw response received" in markdown
        assert '{"value":"Q","reason":"seeded mock"}' in markdown
