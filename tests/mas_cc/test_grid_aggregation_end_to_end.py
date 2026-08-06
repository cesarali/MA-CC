"""A real grid through the orchestrator, checked for the master's own output.

`test_aggregate_metrics.py` exercises the aggregation rules against hand-written
episode files. This checks the wiring instead: that a grid run actually fires
the cell-completion event, aggregates from what the workers wrote, and leaves
the sweep estimate on disk — without any Comet involvement at all.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.experiments import run_experiment_grid_sync

CONFIG = "configs/runs/synthetic_games/synthetic_controlled_markov_empowerment.yaml"


def _small_grid(**aggregation_overrides) -> GridSpec:
    """The controlled-Markov empowerment grid, shrunk to test size.

    A synthetic game is used deliberately: the agents are lookup tables, so the
    run is fast, offline, and free — and the swept `control_value` genuinely
    steers the outcome, which is what makes the grid-level MI non-trivial.
    """

    grid = load_run_config_or_grid(CONFIG, environment={})
    base = replace(
        grid.base,
        game=replace(grid.base.game, population_size=4, horizon=8),
        execution=replace(grid.base.execution, repetitions=4, parallelism=4),
        logging=replace(grid.base.logging, comet=False),
        aggregation=replace(grid.base.aggregation, rolling_window=1, **aggregation_overrides),
    )
    return GridSpec(base=base, axes=grid.axes)


@pytest.fixture(scope="module")
def finished_grid(tmp_path_factory) -> Path:
    grid = _small_grid(sweep_metrics=("terminal_mi", "mi_null_band"), null_permutations=20)
    result = run_experiment_grid_sync(
        grid, tmp_path_factory.mktemp("grid"), resume=False, show_progress=False
    )
    assert result.failed == 0
    return result.output_dir


def test_every_completed_cell_leaves_its_own_aggregate_on_disk(finished_grid: Path):
    cells = sorted(path.name for path in (finished_grid / "cells").iterdir() if path.is_dir())
    assert cells == ["cell-0000", "cell-0001"]

    for cell_id in cells:
        payload = json.loads((finished_grid / "cells" / cell_id / "aggregate.json").read_text())
        assert payload["cell_id"] == cell_id
        assert payload["episodes"] == 4
        assert "converged_fraction" in payload["scalars"]
        assert "active_fraction" in payload["curves"]
        # Sufficient statistics for the grid tier travel with the cell.
        assert sum(payload["counts"]["terminal_outcome"].values()) == 4


def test_the_aggregated_curves_carry_their_percentile_levels(finished_grid: Path):
    payload = json.loads((finished_grid / "cells" / "cell-0000" / "aggregate.json").read_text())

    band = payload["curves"]["dominant_action_share"]
    assert band["levels"] == ["p10", "p50", "p90"]
    for values in band["points"].values():
        assert values == sorted(values)  # p10 <= p50 <= p90, by construction
        assert all(0.0 <= value <= 1.0 for value in values)

    active = payload["curves"]["active_fraction"]
    assert active["levels"] == ["value"]


def test_the_sweep_estimate_is_written_once_the_cells_land(finished_grid: Path):
    payload = json.loads((finished_grid / "sweep_metrics.json").read_text())

    assert payload["cells_complete"] == 2
    assert "terminal_mi_estimate" in payload["scalars"]
    assert "terminal_mi_null_p95" in payload["scalars"]


def test_the_gap_metric_compares_the_live_estimate_against_the_closed_form(tmp_path: Path):
    """A synthetic sweep knows its own answer, so the master can grade itself.

    The answer key is a property of the *sweep* — which cells ran and with how
    many episodes each — and is computed from the resolved grid rather than
    read from anywhere, which is what stops a stale hand-written constant from
    quietly becoming the thing the estimate is checked against.
    """

    grid = _small_grid(
        sweep_metrics=("terminal_mi", "mi_ground_truth_gap"), null_permutations=0
    )
    result = run_experiment_grid_sync(grid, tmp_path, resume=False, show_progress=False)

    scalars = json.loads((result.output_dir / "sweep_metrics.json").read_text())["scalars"]

    assert scalars["terminal_mi_ground_truth"] == pytest.approx(0.626, abs=0.01)
    assert scalars["terminal_mi_gap"] == pytest.approx(
        scalars["terminal_mi_estimate"] - scalars["terminal_mi_ground_truth"]
    )

    # Recomputing later must not silently drop the gap: the answer key comes
    # from the resolved grid and the game object, neither of which a run
    # directory carries, so the live run records it alongside the estimates.
    from mas_cc.cli.experiment import run_aggregate_command

    again = run_aggregate_command(result.output_dir)
    assert again["sweep_metrics"]["terminal_mi_ground_truth"] == pytest.approx(0.626, abs=0.01)


def test_a_run_with_no_sweep_metrics_writes_cells_but_no_grid_estimate(tmp_path: Path):
    """Sweep metrics are opt-in: not asking for MI must not produce a file of NaNs."""

    result = run_experiment_grid_sync(
        _small_grid(sweep_metrics=()), tmp_path, resume=False, show_progress=False
    )

    assert (result.output_dir / "cells" / "cell-0000" / "aggregate.json").exists()
    assert not (result.output_dir / "sweep_metrics.json").exists()


def test_recomputing_from_the_directory_reproduces_the_live_aggregates(finished_grid: Path):
    """Spec acceptance check 2, on real recorded episodes rather than fixtures."""

    from mas_cc.cli.experiment import run_aggregate_command

    before = (finished_grid / "cells" / "cell-0000" / "aggregate.json").read_text()

    summary = run_aggregate_command(finished_grid)

    assert summary["cells_aggregated"] == ["cell-0000", "cell-0001"]
    assert (finished_grid / "cells" / "cell-0000" / "aggregate.json").read_text() == before


def test_recomputing_with_a_different_window_changes_the_curves_but_not_the_episodes(
    finished_grid: Path,
):
    """The point of a separate aggregation tier: re-smooth without re-running."""

    from mas_cc.experiments.aggregation import aggregate_grid_directory

    episode_files = sorted(path for path in finished_grid.rglob("metrics/streaming.csv"))
    fingerprint = [path.stat().st_mtime for path in episode_files]
    before = json.loads((finished_grid / "cells" / "cell-0000" / "aggregate.json").read_text())

    aggregate_grid_directory(
        finished_grid, replace(_small_grid(sweep_metrics=()).base.aggregation, rolling_window=4)
    )
    after = json.loads((finished_grid / "cells" / "cell-0000" / "aggregate.json").read_text())

    assert after["curves"] != before["curves"]
    assert after["aggregation"]["rolling_window"] == 4
    assert [path.stat().st_mtime for path in episode_files] == fingerprint
