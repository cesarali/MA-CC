"""The `results/<game>/<experiment>/<run_id>/` path convention.

Trimmed from the original Phase 8.0 plan: keeps the game/experiment/run_id
nesting and the config/logs/audit/checkpoints/metrics/data separation, drops
the recording-plan/metric-manifest/planned-analysis provenance files that
weren't earned yet. See the `mas-cc-metrics-architecture` memory for the full
rationale.
"""

from __future__ import annotations

from pathlib import Path


def results_run_dir(base: str | Path, *, game: str, experiment: str, run_id: str) -> Path:
    """Return (and create) `<base>/<game>/<experiment>/<run_id>/` with its subdirectories.

    Does not write any files; callers point their existing writers (a
    `RunRecorder`, a metrics writer) at the subdirectories they need.
    """

    run_dir = Path(base) / game / experiment / run_id
    for subdirectory in ("logs", "audit", "checkpoints", "metrics", "metrics/plots", "data"):
        (run_dir / subdirectory).mkdir(parents=True, exist_ok=True)
    return run_dir
