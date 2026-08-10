"""CLI entry point for offline empowerment analysis over a completed grid."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mas_cc.analysis import analyze_grid


def run_analysis_empowerment_command(
    grid_dir: Path,
    output_dir: Path | None,
    *,
    condition_column: str | None = None,
    horizons: tuple[int, ...] = (1,),
    bootstrap_resamples: int = 1000,
    null_permutations: int = 1000,
    seed: int = 1,
) -> dict[str, Any]:
    destination = output_dir or (Path(grid_dir) / "analysis")
    return analyze_grid(
        grid_dir, destination,
        condition_column=condition_column, horizons=horizons,
        bootstrap_resamples=bootstrap_resamples, null_permutations=null_permutations, seed=seed,
    )


def run_hidden_bench_imitation_analysis_command(
    run_dir: Path,
    output_dir: Path | None,
    *,
    bootstrap_resamples: int = 1000,
    null_permutations: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
) -> dict[str, Any]:
    """Build the first-control report from persisted imitation trajectories."""

    from mas_cc.games.hidden_bench.imitation.analysis import (
        analyze_hidden_bench_imitation,
    )

    destination = output_dir or (Path(run_dir) / "hidden_bench_imitation_analysis")
    return analyze_hidden_bench_imitation(
        run_dir,
        destination,
        bootstrap_resamples=bootstrap_resamples,
        null_permutations=null_permutations,
        confidence=confidence,
        seed=seed,
    )
