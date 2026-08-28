"""The one entry point: preflight -> calls -> paired analysis -> plots -> report.

The boundary the TDD asks for, with nothing extra in it:

```text
frozen local vignette builder
  -> canonical game prompt builder
  -> provider call
  -> canonical parser
  -> paired local analysis
```

No population transition loop, no `Game` subclass, no controller policy.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analysis import (
    CellEffect,
    ResponseRow,
    build_response_rows,
    cell_effects,
    model_quality,
    pair_outcomes,
    write_csv,
)
from .config import ProbeConfig
from .execution import (
    RAW_CALLS_FILENAME,
    build_call_specs,
    completed_call_ids,
    read_raw_calls,
    run_calls,
)
from .preflight import Preflight, format_preflight, preflight_payload, run_preflight
from .report import build_report

RESOLVED_CONFIG = "resolved_probe_config.yaml"
PREFLIGHT_JSON = "preflight.json"
RESPONSE_ROWS_CSV = "local_response_rows.csv"
PAIRED_EFFECTS_CSV = "paired_controller_effects.csv"
MODEL_SUMMARY_CSV = "model_summary.csv"
REPORT_MARKDOWN = "controller_retention_probe_report.md"
PLOTS_DIR = "plots"


@dataclass(slots=True)
class ProbeRun:
    config: ProbeConfig
    preflight: Preflight
    payload: dict[str, Any]
    output_dir: Path
    rows: tuple[ResponseRow, ...] = ()
    effects: tuple[CellEffect, ...] = ()
    quality: tuple[Any, ...] = ()
    report_path: Path | None = None

    @property
    def completed_successfully(self) -> bool:
        return bool(self.quality) and all(
            item.successful == item.scheduled for item in self.quality
        )


def prepare(config: ProbeConfig, output_dir: Path | None = None) -> ProbeRun:
    """Run the preflight and write everything that does not need a provider."""

    root = Path(output_dir or config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    preflight = run_preflight(config)
    done = (
        completed_call_ids(root / RAW_CALLS_FILENAME)
        if config.execution.resume
        else set()
    )
    payload = preflight_payload(config, preflight, done)

    import yaml

    (root / RESOLVED_CONFIG).write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    (root / PREFLIGHT_JSON).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return ProbeRun(config=config, preflight=preflight, payload=payload, output_dir=root)


def execute(run: ProbeRun, *, stream: Any = sys.stderr) -> ProbeRun:
    """Schedule and run the outstanding provider calls."""

    if not run.preflight.passed:
        failed = ", ".join(check["check"] for check in run.preflight.failures())
        raise RuntimeError(f"preflight failed; refusing to call any provider: {failed}")
    raw_path = run.output_dir / RAW_CALLS_FILENAME
    if not run.config.execution.resume:
        raw_path.unlink(missing_ok=True)
    specs = build_call_specs(run.config, run.preflight.vignettes)
    done = completed_call_ids(raw_path) if run.config.execution.resume else set()
    expected = {spec.call_id for spec in specs}
    completed = done & expected
    outstanding = tuple(spec for spec in specs if spec.call_id not in completed)
    if stream is not None:
        stream.write(
            f"  scheduling {len(outstanding)} of {len(specs)} calls "
            f"({len(completed)} already complete)\n"
        )
        stream.flush()
    if outstanding:
        run_calls(run.config, outstanding, raw_path, stream=stream)
    return run


def analyze(run: ProbeRun) -> ProbeRun:
    """Build every artifact from whatever calls have actually completed.

    Deliberately usable on a partial run: a probe interrupted halfway should
    still produce readable tables for what it did measure, with `N/A` and a
    reason everywhere else.
    """

    raw_path = run.output_dir / RAW_CALLS_FILENAME
    raw = read_raw_calls(raw_path)
    rows = build_response_rows(
        run.config, run.preflight.vignettes, run.preflight.tasks, raw
    )
    pairs = pair_outcomes(rows)
    effects = cell_effects(pairs)
    scheduled = {
        spec.label: 240 for spec in run.config.models
    }
    quality = model_quality(rows, scheduled)

    run.payload = preflight_payload(
        run.config, run.preflight, completed_call_ids(raw_path)
    )
    (run.output_dir / PREFLIGHT_JSON).write_text(
        json.dumps(run.payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    write_csv(run.output_dir / RESPONSE_ROWS_CSV, [row.to_dict() for row in rows])
    write_csv(
        run.output_dir / PAIRED_EFFECTS_CSV, [effect.as_dict() for effect in effects]
    )
    write_csv(run.output_dir / MODEL_SUMMARY_CSV, [item.to_dict() for item in quality])

    plot_dir = run.output_dir / PLOTS_DIR
    try:
        from .plots import render_all

        produced = render_all(effects, [spec.label for spec in run.config.models], plot_dir)
    except ImportError:
        # matplotlib is optional for the analysis path; the tables still stand.
        produced = {spec.label: [] for spec in run.config.models}

    report = build_report(
        run.config,
        effects,
        quality,
        produced,
        run.payload,
        run.output_dir,
    )
    report_path = run.output_dir / REPORT_MARKDOWN
    report_path.write_text(report, encoding="utf-8")

    run.rows = rows
    run.effects = effects
    run.quality = quality
    run.report_path = report_path
    return run


def run_probe(
    config: ProbeConfig,
    *,
    output_dir: Path | None = None,
    preflight_only: bool = False,
    analyze_only: bool = False,
    stream: Any = sys.stderr,
) -> ProbeRun:
    run = prepare(config, output_dir)
    if stream is not None:
        stream.write(format_preflight(run.payload) + "\n")
        stream.flush()
    if preflight_only:
        return run
    if not analyze_only:
        execute(run, stream=stream)
    return analyze(run)


__all__ = [
    "MODEL_SUMMARY_CSV",
    "PAIRED_EFFECTS_CSV",
    "PLOTS_DIR",
    "PREFLIGHT_JSON",
    "REPORT_MARKDOWN",
    "RESOLVED_CONFIG",
    "RESPONSE_ROWS_CSV",
    "ProbeRun",
    "analyze",
    "execute",
    "prepare",
    "run_probe",
]
