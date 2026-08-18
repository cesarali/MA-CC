"""CLI entry points for the single-model relational support benchmark."""

from __future__ import annotations

from pathlib import Path

from mas_cc.benchmarks.relational_support import load_benchmark_config
from mas_cc.benchmarks.relational_support.analysis import write_summary
from mas_cc.benchmarks.relational_support.runner import build_benchmark, run_benchmark


def _destination(config, output_dir: Path | None) -> Path:
    return Path(output_dir) if output_dir is not None else config.output_dir / config.name


def run_relational_support_preflight(
    config_path: str | Path,
    output_dir: Path | None = None,
    *,
    verify_reproducibility: bool = True,
) -> tuple[bool, Path, str]:
    """Generate, render and validate everything.  Never opens a provider."""

    config = load_benchmark_config(config_path)
    destination = _destination(config, output_dir)
    plan = build_benchmark(config, destination, verify_reproducibility=verify_reproducibility)
    failing = [check["check"] for check in plan.validation["checks"] if not check["passed"]]
    within_limit = plan.request_count <= config.max_requests
    ok = plan.valid and within_limit
    message = (
        f"Benchmark preflight {'passed' if ok else 'FAILED'} - "
        f"{plan.request_count} request(s) planned, limit {config.max_requests}"
    )
    if failing:
        message += f"; failing checks: {failing}"
    elif not within_limit:
        message += "; plan exceeds limits.max_requests"
    return ok, destination, message


def run_relational_support_benchmark(
    config_path: str | Path,
    output_dir: Path | None = None,
    *,
    verify_reproducibility: bool = True,
) -> tuple[bool, Path, str]:
    """Preflight, refuse on any failure, then send one request per item."""

    config = load_benchmark_config(config_path)
    destination = _destination(config, output_dir)
    plan, rows = run_benchmark(
        config, destination, verify_reproducibility=verify_reproducibility
    )
    write_summary(destination)
    errors = sum(1 for row in rows if row.get("error"))
    scored = len(rows) - errors
    correct = sum(1 for row in rows if row.get("correct"))
    accuracy = correct / scored if scored else 0.0
    message = (
        f"Benchmark run complete - {len(rows)} item(s), {errors} provider error(s), "
        f"overall accuracy {accuracy:.3f}"
    )
    return errors == 0, destination, message


def summarize_relational_support_benchmark(
    input_dir: str | Path, output_dir: str | Path | None = None
) -> Path:
    return write_summary(input_dir, output_dir)
