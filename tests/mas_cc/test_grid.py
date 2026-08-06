import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from mas_cc.config import GridSpec, RunConfig, load_run_config
from mas_cc.config.grid import GridAxis
from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.experiments import run_experiment_grid_sync
from mas_cc.games import create_game
from mas_cc.llm_runtime.providers import BudgetLimits, OfflinePricingSource
from mas_cc.planning import static_experiment_preflight, static_grid_preflight


def _toy_config(**execution_overrides) -> RunConfig:
    config = load_run_config("configs/runs/old/toy_game_smoke_test.yaml", environment={})
    return replace(config, execution=replace(config.execution, **execution_overrides))


def _with_ample_budget(config: RunConfig) -> RunConfig:
    return replace(
        config,
        budget=replace(
            config.budget,
            max_provider_requests=10_000, max_input_tokens=10_000_000,
            max_output_tokens=10_000, max_cost_per_run=100.0, system_max_cost_per_run=100.0,
        ),
    )


def test_grid_spec_expands_the_cartesian_product_in_stable_order():
    base = _toy_config()
    grid = GridSpec(base=base, axes=(GridAxis("game.horizon", (2, 3)), GridAxis("llm_provider.temperature", (0.0, 0.5))))
    cells = grid.cells
    assert [cell.overrides for cell in cells] == [
        {"game.horizon": 2, "llm_provider.temperature": 0.0},
        {"game.horizon": 2, "llm_provider.temperature": 0.5},
        {"game.horizon": 3, "llm_provider.temperature": 0.0},
        {"game.horizon": 3, "llm_provider.temperature": 0.5},
    ]
    assert [cell.config.game.horizon for cell in cells] == [2, 2, 3, 3]
    assert [cell.config.llm_provider.temperature for cell in cells] == [0.0, 0.5, 0.0, 0.5]
    # The base config is untouched; only each cell's own config carries the override.
    assert base.game.horizon == load_run_config("configs/runs/old/toy_game_smoke_test.yaml", environment={}).game.horizon
    assert grid.grid_id == GridSpec(base=base, axes=grid.axes).grid_id  # deterministic


@pytest.mark.parametrize(
    "path", ["llm_provider.type", "llm_provider.model", "game.type", "budget.max_cost_per_run", "pricing.mode", "budget", "pricing"]
)
def test_grid_axis_forbids_sweeping_shared_identity_fields(path):
    with pytest.raises(ConfigurationError, match="cannot sweep"):
        GridAxis(path, (1, 2))


def test_grid_spec_rejects_a_bad_dotted_path_eagerly():
    base = _toy_config()
    with pytest.raises(ConfigurationError):
        GridSpec(base=base, axes=(GridAxis("game.not_a_real_field", (1, 2)),))


def test_static_grid_preflight_sums_per_cell_estimates():
    base = _toy_config()
    grid = GridSpec(base=base, axes=(GridAxis("game.horizon", (2, 3)),))
    quote = OfflinePricingSource().fetch(base.llm_provider.type, base.llm_provider.model)

    game = create_game(base.game)
    manual_total_expected = 0
    for cell in grid.cells:
        plan = game.call_plan(cell.config.game)
        per_cell = static_experiment_preflight(
            plan, cell.config.prompt, cell.config.llm_provider,
            episode_count=cell.config.execution.repetitions, pricing_quote=quote,
            assumed_output_tokens=1,
        )
        manual_total_expected += per_cell.total_provider_requests.expected

    estimate = static_grid_preflight(grid, pricing_quote=quote, assumed_output_tokens=1)
    assert estimate.cell_count == 2
    assert estimate.total_episode_count == 2 * base.execution.repetitions
    assert estimate.total_provider_requests.expected == manual_total_expected
    assert estimate.launch_status == "permitted"


def test_static_grid_preflight_checks_budget_once_against_the_combined_total():
    base = _toy_config()
    grid = GridSpec(base=base, axes=(GridAxis("game.horizon", (2, 3)),))
    quote = OfflinePricingSource().fetch(base.llm_provider.type, base.llm_provider.model)
    game = create_game(base.game)
    one_cell_requests = static_experiment_preflight(
        game.call_plan(grid.cells[0].config.game), grid.cells[0].config.prompt,
        grid.cells[0].config.llm_provider, episode_count=grid.cells[0].config.execution.repetitions,
        pricing_quote=quote, assumed_output_tokens=1,
    ).total_provider_requests.conservative

    tight_budget = BudgetLimits(max_requests=one_cell_requests + 1)
    estimate = static_grid_preflight(
        grid, pricing_quote=quote, assumed_output_tokens=1, run_budget=tight_budget,
    )
    assert estimate.launch_status == "denied"  # sum across both cells exceeds one cell's worth


def test_run_experiment_grid_runs_every_cell_and_resumes_per_episode(tmp_path: Path):
    base = _with_ample_budget(_toy_config(repetitions=2, parallelism=2))
    grid = GridSpec(base=base, axes=(GridAxis("game.horizon", (2, 3)),))

    first = run_experiment_grid_sync(grid, tmp_path, resume=True, show_progress=False)
    assert len(first.cells) == 2
    assert first.completed == 4  # 2 cells x 2 repetitions
    assert first.failed == 0

    second = run_experiment_grid_sync(grid, tmp_path, resume=True, show_progress=False)
    assert second.completed == 0
    assert second.skipped_resumed == 4

    third = run_experiment_grid_sync(grid, tmp_path, resume=False, show_progress=False)
    assert third.completed == 4


def test_run_experiment_grid_fail_fast_aborts_across_every_cell_not_just_one(tmp_path: Path, monkeypatch):
    base = _with_ample_budget(_toy_config(repetitions=1, parallelism=1, fail_fast=True))
    grid = GridSpec(base=base, axes=(GridAxis("game.horizon", (2, 3, 4)),))  # 3 cells x 1 episode = 3 tasks

    from mas_cc.games.runner import run_game as real_run_game

    calls: list[int] = []

    async def flaky_run_game(game, episode_config, provider, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("simulated cell-0 failure")
        return await real_run_game(game, episode_config, provider, **kwargs)

    monkeypatch.setattr("mas_cc.experiments.orchestrator.run_game", flaky_run_game)

    result = run_experiment_grid_sync(grid, tmp_path, resume=False, show_progress=False)
    statuses = [outcome.status for cell in result.cells for outcome in cell.outcomes]
    assert statuses == ["failed", "skipped_aborted", "skipped_aborted"]
    assert len(calls) == 1


def test_run_experiment_grid_shares_one_concurrency_pool_across_cells(tmp_path: Path, monkeypatch):
    """The whole point of 'combined concurrency': parallelism bounds *all* episodes
    from *every* cell together, not parallelism-per-cell."""

    base = _with_ample_budget(_toy_config(repetitions=2, parallelism=2))
    grid = GridSpec(base=base, axes=(GridAxis("game.horizon", (2, 3)),))  # 2 cells x 2 episodes = 4 tasks

    from mas_cc.games.runner import run_game as real_run_game

    active = 0
    max_active = 0
    lock = asyncio.Lock()

    async def tracked_run_game(game, episode_config, provider, **kwargs):
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.02)
            return await real_run_game(game, episode_config, provider, **kwargs)
        finally:
            async with lock:
                active -= 1

    monkeypatch.setattr("mas_cc.experiments.orchestrator.run_game", tracked_run_game)

    result = run_experiment_grid_sync(grid, tmp_path, resume=False, show_progress=False)
    assert result.completed == 4
    assert max_active == 2  # never more than execution.parallelism, even across 2 cells
