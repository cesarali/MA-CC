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
