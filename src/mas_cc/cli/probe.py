"""CLI handlers for provider-backed diagnostic probes.

A probe is not a game and not an experiment: it has no population loop, so it
gets its own command rather than a `Game` type that would have to no-op most of
the interface.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from mas_cc.probes.controller_retention.config import load_probe_config
from mas_cc.probes.controller_retention.runner import run_probe


def run_controller_retention_probe(
    config_path: Path,
    output_dir: Path | None = None,
    *,
    mode: str = "run",
    stream: Any = sys.stderr,
) -> tuple[bool, Path, str]:
    """``preflight`` sends nothing, ``analyze`` re-reads a finished run, ``run``
    does the whole thing.  All three write the same artifact tree."""

    config = load_probe_config(config_path)
    result = run_probe(
        config,
        output_dir=output_dir,
        preflight_only=mode == "preflight",
        analyze_only=mode == "analyze",
        stream=stream,
    )
    if mode == "preflight":
        return (
            result.preflight.passed,
            result.output_dir,
            "Probe preflight " + ("passed" if result.preflight.passed else "FAILED"),
        )
    if mode == "run" and not result.completed_successfully:
        return (
            False,
            result.report_path or result.output_dir,
            "Controller-retention probe finished with missing or invalid calls",
        )
    return (
        result.preflight.passed,
        result.report_path or result.output_dir,
        "Controller-retention probe report written",
    )


__all__ = ["run_controller_retention_probe"]
