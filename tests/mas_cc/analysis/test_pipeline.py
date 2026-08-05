import json
import math
from pathlib import Path

import pandas as pd
import pytest

from mas_cc.analysis import analyze_grid, estimate_terminal_mi, read_grid


def _write_streaming_csv(path: Path, *, rounds: list[tuple[float, float]]) -> None:
    """`rounds` is a list of (share_x, share_y) pairs, one per round index."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for round_index, (share_x, share_y) in enumerate(rounds, start=1):
        rows.append({"round_index": round_index, "episode_id": "unused", "agent_id": "", "series": "x", "metric_name": "population_action_share_per_option", "value": share_x})
        rows.append({"round_index": round_index, "episode_id": "unused", "agent_id": "", "series": "y", "metric_name": "population_action_share_per_option", "value": share_y})
        rows.append({"round_index": round_index, "episode_id": "unused", "agent_id": "agent-000", "series": "", "metric_name": "agent_current_action", "value": "x" if share_x > share_y else "y"})
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_grid(root: Path, *, condition_to_episodes: dict[str, list[list[tuple[float, float]]]]) -> None:
    """condition_to_episodes maps a condition value -> list of episodes, each a list of (share_x, share_y) rounds."""

    cells_dir = root / "cells"
    for index, (condition, episodes) in enumerate(condition_to_episodes.items()):
        cell_id = f"cell-{index:04d}"
        cell_dir = cells_dir / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "overrides.json").write_text(
            json.dumps({"cell_id": cell_id, "index": index, "overrides": {"control.options.forced_value": condition}}),
            encoding="utf-8",
        )
        for episode_index, rounds in enumerate(episodes):
            episode_dir = cell_dir / "data" / "episodes" / f"{cell_id}-{episode_index:04d}"
            _write_streaming_csv(episode_dir / "metrics" / "streaming.csv", rounds=rounds)


def test_read_grid_derives_terminal_outcome_and_condition_columns(tmp_path: Path):
    _build_grid(
        tmp_path,
        condition_to_episodes={
            "x-forced": [[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]],
            "y-forced": [[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]],
        },
    )
    data = read_grid(tmp_path)
    assert data.condition_columns == ("control.options.forced_value",)
    assert set(data.episodes["terminal_outcome"]) == {"x", "y"}
    assert len(data.rounds) == 2 * 3  # 2 episodes x 3 rounds each


def test_terminal_mi_recovers_a_perfect_signal(tmp_path: Path):
    # 20 episodes per condition, condition perfectly determines terminal outcome.
    _build_grid(
        tmp_path,
        condition_to_episodes={
            "x-forced": [[(1.0, 0.0)] for _ in range(20)],
            "y-forced": [[(0.0, 1.0)] for _ in range(20)],
        },
    )
    data = read_grid(tmp_path)
    estimate = estimate_terminal_mi(
        data.episodes, "control.options.forced_value", bootstrap_resamples=200, seed=1,
    )
    # Unsmoothed MI on a perfectly diagonal contingency table is exactly log2(2);
    # Jeffreys (+0.5 pseudocount per cell) is deliberately conservative at n=20/cell,
    # so check it against a looser bound instead of equality.
    assert estimate["unsmoothed"] == pytest.approx(math.log2(2), abs=1e-9)
    assert estimate["jeffreys"] > 0.7
    assert estimate["episodes"] == 40
    # A near-perfect signal should not have zero anywhere near its CI.
    assert estimate["ci_low"] > 0.5


def test_terminal_mi_is_near_zero_for_an_independent_signal(tmp_path: Path):
    # Terminal outcome alternates x/y regardless of condition -> no real signal.
    episodes = [[(1.0, 0.0)] if i % 2 == 0 else [(0.0, 1.0)] for i in range(20)]
    _build_grid(tmp_path, condition_to_episodes={"x-forced": episodes, "y-forced": list(episodes)})
    data = read_grid(tmp_path)
    estimate = estimate_terminal_mi(
        data.episodes, "control.options.forced_value", bootstrap_resamples=0, seed=1,
    )
    assert estimate["jeffreys"] == pytest.approx(0.0, abs=1e-6)


def test_analyze_grid_writes_csv_artifacts_and_null_collapses_toward_zero(tmp_path: Path):
    grid_dir = tmp_path / "grid"
    _build_grid(
        grid_dir,
        condition_to_episodes={
            "x-forced": [[(1.0, 0.0)] for _ in range(15)],
            "y-forced": [[(0.0, 1.0)] for _ in range(15)],
        },
    )
    output_dir = tmp_path / "analysis"
    summary = analyze_grid(
        grid_dir, output_dir, bootstrap_resamples=50, null_permutations=100, seed=7,
    )
    assert summary["condition_column"] == "control.options.forced_value"
    assert summary["terminal_mi_jeffreys"] > 0.7  # Jeffreys-smoothed, deliberately conservative at n=15/cell

    assert (output_dir / "mi_estimates.csv").exists()
    assert (output_dir / "null_results.csv").exists()

    nulls = pd.read_csv(output_dir / "null_results.csv")
    terminal_nulls = nulls[nulls["null_type"] == "condition_label_shuffle"]
    # The shuffled-label null should be much smaller than the real signal on average.
    assert terminal_nulls["jeffreys"].mean() < summary["terminal_mi_jeffreys"] / 2


def test_analyze_grid_requires_explicit_condition_column_when_ambiguous(tmp_path: Path):
    grid_dir = tmp_path / "grid"
    cells_dir = grid_dir / "cells"
    for index, overrides in enumerate([
        {"control.options.forced_value": "x-forced", "control.mechanism": "forced_action"},
        {"control.options.forced_value": "y-forced", "control.mechanism": "none"},
    ]):
        cell_id = f"cell-{index:04d}"
        cell_dir = cells_dir / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "overrides.json").write_text(
            json.dumps({"cell_id": cell_id, "index": index, "overrides": overrides}), encoding="utf-8",
        )
        _write_streaming_csv(
            cell_dir / "data" / "episodes" / f"{cell_id}-0000" / "metrics" / "streaming.csv",
            rounds=[(1.0, 0.0)],
        )
    with pytest.raises(ValueError, match="condition_column must be given explicitly"):
        analyze_grid(grid_dir, tmp_path / "analysis")
