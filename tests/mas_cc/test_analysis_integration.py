"""End-to-end: run a real small grid through the orchestrator, then analyze it
through the actual `mas-cc analysis empowerment` CLI path - not just unit
tests of the pieces in isolation."""

import math
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from mas_cc.cli.analysis import run_analysis_empowerment_command
from mas_cc.config import GridSpec, load_run_config
from mas_cc.config.grid import GridAxis
from mas_cc.experiments import run_experiment_grid_sync


def test_control_grid_produces_a_high_mi_estimate_and_a_near_zero_null(tmp_path: Path):
    base = load_run_config("configs/runs/naming_convention_smoke_test.yaml", environment={})
    base = replace(
        base,
        budget=replace(
            base.budget,
            max_provider_requests=10_000, max_input_tokens=10_000_000,
            max_output_tokens=10_000, max_cost_per_run=100.0, system_max_cost_per_run=100.0,
        ),
        game=replace(base.game, population_size=2, horizon=6),
        execution=replace(base.execution, repetitions=8),
        control=replace(
            base.control,
            mechanism="forced_action",
            options={"agent_ids": ["agent-000", "agent-001"], "forced_value": "Q"},
        ),
    )
    # Forcing *both* agents to the swept value makes the terminal outcome
    # deterministic given the condition - the cleanest possible real signal
    # to prove the CLI path end to end, independent of any LLM variability.
    grid = GridSpec(base=base, axes=(GridAxis("control.options.forced_value", ("Q", "M")),))

    grid_dir = tmp_path / "grid"
    result = run_experiment_grid_sync(grid, grid_dir, resume=False, show_progress=False)
    assert result.completed == 16  # 2 cells x 8 repetitions
    assert result.failed == 0

    # The grid writes into results_run_dir(...): <grid_dir>/<game>/<experiment>/<run_id>/
    run_dir = next((grid_dir / "naming_convention").rglob("cells")).parent

    summary = run_analysis_empowerment_command(
        run_dir, tmp_path / "analysis",
        bootstrap_resamples=200, null_permutations=200, seed=3,
    )
    assert summary["condition_column"] == "control.options.forced_value"
    assert summary["episodes"] == 16
    # A perfectly deterministic condition -> outcome mapping: unsmoothed MI
    # should be (near) exactly log2(2), and even the conservative Jeffreys
    # estimate (pulled down by its +0.5 pseudocount at n=8/cell) should still
    # be clearly nonzero.
    assert summary["terminal_mi_jeffreys"] > 0.5

    estimates = pd.read_csv(tmp_path / "analysis" / "mi_estimates.csv")
    terminal_row = estimates[estimates["statistic"] == "terminal"].iloc[0]
    assert terminal_row["unsmoothed"] == pytest.approx(math.log2(2), abs=1e-6)

    nulls = pd.read_csv(tmp_path / "analysis" / "null_results.csv")
    terminal_nulls = nulls[nulls["null_type"] == "condition_label_shuffle"]
    assert terminal_nulls["jeffreys"].mean() < summary["terminal_mi_jeffreys"] / 2

    invariance_path = tmp_path / "analysis" / "label_swap_invariance.csv"
    assert invariance_path.exists()
    invariance = pd.read_csv(invariance_path)
    assert bool(invariance.iloc[0]["invariant_within_tolerance"]) is True
