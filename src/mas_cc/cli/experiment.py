"""Phase 9 CLI: price and run a multi-episode experiment, or a grid of them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mas_cc.config import (
    GridSpec,
    RunConfig,
    load_run_config_or_grid,
    resolved_config_yaml,
)
from mas_cc.experiments import (
    ExperimentResult,
    GridResult,
    run_experiment_grid_sync,
    run_experiment_sync,
)
from mas_cc.control import create_control
from mas_cc.experiments.console import format_money
from mas_cc.games import create_game
from mas_cc.llm_runtime.providers import PricingQuote
from mas_cc.planning import (
    ExperimentPreflightEstimate,
    GridPreflightEstimate,
    static_experiment_preflight,
    static_grid_preflight,
)
from mas_cc.storage import canonical_hash
from mas_cc.planning.semantic_storage import estimate_semantic_storage

from .game import _budgets, _quote
from .inspect import _write, _write_manifest


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _output_dir(source: RunConfig | GridSpec, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    base = source.base if isinstance(source, GridSpec) else source
    return Path(base.storage.output_dir)


def resolve_output_dir(
    config_path: str | Path, output_dir: str | Path | None = None
) -> Path:
    """The directory a preflight/run against ``config_path`` will land in.

    Lets the CLI report the resolved destination in its own status line
    without duplicating the flag-or-``storage.output_dir`` fallback logic.
    """

    return _output_dir(load_run_config_or_grid(config_path), output_dir)


def compute_preflight_id(config: RunConfig, quote: PricingQuote) -> str:
    """Bind approval to the resolved config, prompt, and pricing snapshot only.

    Deliberately narrower than V6's Phase 9: no recording-plan, extractor, or
    planned-analysis fingerprints, because Phase 9 as scoped here does not
    produce those (see the Phase 9 plan doc's "Deferred, not deleted" section).
    """

    return canonical_hash(
        {
            "resolved_config_hash": canonical_hash(config.to_dict()),
            "pricing_snapshot_hash": _approval_pricing_hash(quote),
            "episode_count": config.execution.repetitions,
        }
    )


def _approval_pricing_hash(quote: PricingQuote) -> str:
    """Identify approved rates without binding to volatile retrieval metadata.

    Live pricing is fetched once by ``preflight`` and again by ``run``.  Those
    snapshots necessarily have different retrieval/freshness timestamps and
    may have a different live account balance even when the model's approved
    rates are identical.  Binding approval to those volatile fields made every
    live preflight token unusable.  The runtime still performs its independent
    freshness and immediate pricing-term checks before provider calls.
    """

    pricing_terms = None
    if quote.pricing is not None:
        pricing_terms = quote.pricing.to_dict()
        for field in ("source", "retrieved_at", "version"):
            pricing_terms.pop(field, None)
    return canonical_hash(
        {
            "provider": quote.provider,
            "model": quote.model,
            "available": quote.available,
            "pricing_terms": pricing_terms,
        }
    )


def compute_grid_preflight_id(grid: GridSpec, quote: PricingQuote) -> str:
    """Same idea as ``compute_preflight_id``, additionally bound to the grid's own identity."""

    return canonical_hash(
        {
            "grid_id": grid.grid_id,
            "base_config_hash": canonical_hash(grid.base.to_dict()),
            "pricing_snapshot_hash": _approval_pricing_hash(quote),
            "cell_count": len(grid.cells),
        }
    )


def _run_single_preflight(
    config: RunConfig, destination: Path
) -> ExperimentPreflightEstimate:
    game = create_game(config.game)
    if config.game.type == "relational_imitation_round_feedback":
        controller = create_control(config.control)
        validator = getattr(controller, "validate_truthful_report_task", None)
        if validator is not None:
            validator(game.load_task(config.game), config.execution.seed)
    plan = game.call_plan(config.game)
    quote = _quote(config)
    system_budget, run_budget = _budgets(config, quote)

    estimate = static_experiment_preflight(
        plan,
        config.prompt,
        config.llm_provider,
        episode_count=config.execution.repetitions,
        concurrency=config.execution.parallelism,
        assumed_output_tokens=config.llm_provider.max_output_tokens,
        pricing_quote=quote,
        system_budget=system_budget,
        run_budget=run_budget,
        explicit_override=config.pricing.explicit_unknown_price_override,
        allow_stale_pricing=not config.pricing.require_fresh_at_launch,
    )

    preflight_id = compute_preflight_id(config, quote)
    _write(destination / "resolved_config.yaml", resolved_config_yaml(config))
    _write(
        destination / "per_episode_estimate.json", _json(estimate.per_episode.to_dict())
    )
    payload = estimate.to_dict()
    storage_estimate = estimate_semantic_storage(
        config.to_dict(), config.execution.repetitions
    )
    if storage_estimate is not None:
        payload["semantic_storage"] = storage_estimate.to_dict()
    _write(destination / "experiment_estimate.json", _json(payload))
    _write(destination / "pricing_snapshot.json", _json(quote.to_dict()))
    _write(
        destination / "budget_status.json",
        _json(
            {
                "system_budget": system_budget.to_dict(),
                "run_budget": run_budget.to_dict(),
            }
        ),
    )
    _write(destination / "preflight_id.txt", preflight_id + "\n")

    status = "pass" if estimate.launch_status == "permitted" else "fail"
    report = f"""# Phase 9 experiment preflight

- Status: **{status.upper()}** (`{estimate.launch_status}`)
- Experiment: `{config.experiment.name}`; game `{estimate.game_type}`; provider `{estimate.provider}` / `{estimate.model}`.
- Episodes: {estimate.episode_count} (concurrency {estimate.concurrency}).
- Expected total cost: {format_money(estimate.total_costs.expected)}; conservative: {format_money(estimate.total_costs.conservative)}.
- Rough total runtime: {estimate.rough_runtime_seconds:.1f}s.
{f"- Estimated dashboard-semantic storage: {storage_estimate.total_bytes} bytes ({storage_estimate.files_per_episode * storage_estimate.episode_count} episode files)." if storage_estimate is not None else ""}
- Preflight ID: `{preflight_id}` — pass this to `mas-cc experiment run --approve-preflight` to bind the launch to this estimate.

## Warnings

{chr(10).join(f"- {warning}" for warning in estimate.warnings) or "- none"}
"""
    _write(destination / "report.md", report)
    _write_manifest(
        destination,
        phase=9,
        status=status,
        checks={"launch_permitted": estimate.launch_status == "permitted"},
        warnings=list(estimate.warnings),
    )
    return estimate


def _run_grid_preflight(grid: GridSpec, destination: Path) -> GridPreflightEstimate:
    quote = _quote(grid.base)
    system_budget, run_budget = _budgets(grid.base, quote)

    estimate = static_grid_preflight(
        grid,
        concurrency=grid.base.execution.parallelism,
        assumed_output_tokens=grid.base.llm_provider.max_output_tokens,
        pricing_quote=quote,
        system_budget=system_budget,
        run_budget=run_budget,
        explicit_override=grid.base.pricing.explicit_unknown_price_override,
        allow_stale_pricing=not grid.base.pricing.require_fresh_at_launch,
    )

    preflight_id = compute_grid_preflight_id(grid, quote)
    _write(destination / "resolved_base_config.yaml", resolved_config_yaml(grid.base))
    _write(destination / "grid_definition.json", _json(grid.to_dict()))
    payload = estimate.to_dict()
    storage_estimate = estimate_semantic_storage(
        grid.base.to_dict(), estimate.total_episode_count
    )
    if storage_estimate is not None:
        payload["semantic_storage"] = storage_estimate.to_dict()
    _write(destination / "grid_estimate.json", _json(payload))
    _write(destination / "pricing_snapshot.json", _json(quote.to_dict()))
    _write(
        destination / "budget_status.json",
        _json(
            {
                "system_budget": system_budget.to_dict(),
                "run_budget": run_budget.to_dict(),
            }
        ),
    )
    _write(destination / "preflight_id.txt", preflight_id + "\n")

    status = "pass" if estimate.launch_status == "permitted" else "fail"
    axes_lines = "\n".join(
        f"- `{axis.path}`: {list(axis.values)}" for axis in grid.axes
    )
    report = f"""# Phase 9 grid preflight

- Status: **{status.upper()}** (`{estimate.launch_status}`)
- Experiment: `{grid.base.experiment.name}`; game `{estimate.game_type}`; provider `{estimate.provider}` / `{estimate.model}`.
- Cells: {estimate.cell_count}; total episodes: {estimate.total_episode_count} (shared concurrency {estimate.concurrency}).
- Axes:
{axes_lines}
- Expected total cost: {format_money(estimate.total_costs.expected)}; conservative: {format_money(estimate.total_costs.conservative)}.
- Rough total runtime: {estimate.rough_runtime_seconds:.1f}s.
{f"- Estimated dashboard-semantic storage: {storage_estimate.total_bytes} bytes ({storage_estimate.files_per_episode * storage_estimate.episode_count} episode files)." if storage_estimate is not None else ""}
- Preflight ID: `{preflight_id}` — pass this to `mas-cc experiment run --approve-preflight` to bind the launch to this estimate.

## Warnings

{chr(10).join(f"- {warning}" for warning in estimate.warnings) or "- none"}
"""
    _write(destination / "report.md", report)
    _write_manifest(
        destination,
        phase=9,
        status=status,
        checks={"launch_permitted": estimate.launch_status == "permitted"},
        warnings=list(estimate.warnings),
    )
    return estimate


def run_experiment_preflight(
    config_path: str | Path, output_dir: str | Path | None = None
) -> ExperimentPreflightEstimate | GridPreflightEstimate:
    """Estimate the whole experiment's (or grid's) calls/tokens/cost/runtime; no provider I/O."""

    source = load_run_config_or_grid(config_path)
    destination = _output_dir(source, output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if isinstance(source, GridSpec):
        return _run_grid_preflight(source, destination)
    return _run_single_preflight(source, destination)


def _approve(
    base_config: RunConfig,
    quote: PricingQuote,
    approve_preflight: str | Path | None,
    *,
    grid: GridSpec | None,
) -> None:
    if approve_preflight is None:
        return
    path = Path(approve_preflight)
    approved_id = (
        path.read_text(encoding="utf-8").strip()
        if path.is_file()
        else str(approve_preflight).strip()
    )
    current_id = (
        compute_grid_preflight_id(grid, quote)
        if grid is not None
        else compute_preflight_id(base_config, quote)
    )
    if approved_id != current_id:
        raise ValueError(
            "approved preflight ID does not match the current resolved config/pricing; "
            "re-run `mas-cc experiment preflight` and approve the new estimate"
        )


def run_aggregate_command(
    run_dir: str | Path, config_path: str | Path | None = None
) -> dict[str, Any]:
    """Recompute a finished run's aggregates from disk, touching no provider.

    The point of a separate command is that aggregation is *derived*: changing
    a percentile band, a rolling window, or the fill rule is a re-read of files
    that already exist, not a re-run of a day of episodes. Passing `--config`
    swaps in a different `aggregation:` section, which is the deliberate act
    that makes the two sets of curves comparable rather than confusable.
    """

    from mas_cc.experiments.aggregation import aggregate_grid_directory

    aggregation = None
    if config_path is not None:
        source = load_run_config_or_grid(config_path)
        aggregation = (
            source.base if isinstance(source, GridSpec) else source
        ).aggregation
    return aggregate_grid_directory(run_dir, aggregation)


def run_compact_command(
    run_dir: str | Path,
    *,
    profile: str = "results_only",
    delete_raw: bool = False,
    archive: bool = False,
) -> dict[str, Any]:
    """Preview or explicitly compact a legacy run without provider calls."""

    from mas_cc.storage import compact_run_directory

    return compact_run_directory(
        run_dir, profile=profile, delete_raw=delete_raw, archive=archive
    )


def run_experiment_command(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    resume: bool = True,
    show_progress: bool = True,
    approve_preflight: str | Path | None = None,
) -> ExperimentResult | GridResult:
    source = load_run_config_or_grid(config_path)
    destination = _output_dir(source, output_dir)
    if isinstance(source, GridSpec):
        quote = _quote(source.base)
        _approve(source.base, quote, approve_preflight, grid=source)
        return run_experiment_grid_sync(
            source, destination, resume=resume, show_progress=show_progress
        )
    quote = _quote(source)
    _approve(source, quote, approve_preflight, grid=None)
    return run_experiment_sync(
        source, destination, resume=resume, show_progress=show_progress
    )
