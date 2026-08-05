"""Phase 9: concurrent multi-episode experiment orchestration, plus grid sweeps.

An experiment is N episodes of one resolved game/prompt/provider
configuration, run under bounded concurrency, priced before launch, and
resumable by episode. A grid is many such experiments (cells) — one per
combination of swept `grid:` values — sharing one provider client, one
pricing quote, and one combined concurrency/budget pool across every episode
of every cell. See
``tdd/architecture/03082026_MAS_CC_Phase9_Experiment_Orchestration_Plan_v1.md``.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Mapping

from mas_cc.config import GridSpec, RunConfig, resolved_config_yaml
from mas_cc.control import create_control
from mas_cc.core.random import Seed
from mas_cc.games import Game, create_game, game_metrics
from mas_cc.games.naming_convention.runtime import run_naming_convention_game
from mas_cc.games.runner import run_game
from mas_cc.llm_runtime.providers import (
    BudgetGuardedProvider,
    BudgetLimits,
    CachedPricingSource,
    MonetaryAmount,
    OfflinePricingSource,
    PricingQuote,
    RuntimeBudgetGuard,
    UniversityPricingSource,
    create_llm_provider,
    resolve_budget_limits,
)
from mas_cc.observability import DetailedAuditPolicy, RunRecorder, price_snapshot_hash
from mas_cc.planning import (
    ExperimentPreflightEstimate,
    GridPreflightEstimate,
    estimate_input_tokens,
    static_experiment_preflight,
    static_grid_preflight,
)
from mas_cc.storage import results_run_dir

from .console import ExperimentProgress, format_banner, format_grid_banner, format_money, print_banner

LOGGER = logging.getLogger("mas_cc.experiment")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --- Pricing/budget resolution, duplicated in miniature from cli/game.py -----
# Kept local rather than imported: cli/phase7.py's inspection gate is a frozen
# byte-exact artifact producer (see the mas-cc-metrics-architecture memory),
# and this module intentionally does not import from `mas_cc.cli` at all.


def _quote(config: RunConfig) -> PricingQuote:
    pricing = config.pricing
    provider = config.llm_provider
    if pricing.mode == "offline":
        return OfflinePricingSource().fetch(provider.type, provider.model)
    if pricing.mode == "cached":
        if pricing.cache_path is None:
            raise ValueError("pricing.cache_path is required in cached mode")
        return CachedPricingSource(
            pricing.cache_path,
            max_age=timedelta(seconds=pricing.max_age_seconds),
            allow_stale=pricing.fallback_policy == "allow_stale",
        ).fetch(provider.type, provider.model)
    if pricing.mode == "live" and provider.type == "university":
        return UniversityPricingSource(
            provider, freshness=timedelta(seconds=pricing.max_age_seconds)
        ).fetch(provider.type, provider.model)
    if pricing.fallback_policy == "offline":
        return OfflinePricingSource().fetch(provider.type, provider.model)
    raise ValueError(
        f"live pricing is unavailable for provider {provider.type!r}; select an auditable "
        "cached/offline source or an explicit fallback"
    )


def _money_limit(
    amount: float | None, config: RunConfig, quote: PricingQuote, description: str
) -> MonetaryAmount | None:
    if amount is None:
        return None
    return MonetaryAmount(
        amount=amount, unit=config.budget.accounting_unit,
        unit_source="resolved budget configuration", provider=config.llm_provider.type,
        model=config.llm_provider.model, source=description,
        retrieved_at=quote.retrieved_at, version="resolved-config-v1",
    )


def _budgets(config: RunConfig, quote: PricingQuote) -> tuple[BudgetLimits, BudgetLimits]:
    system = BudgetLimits(
        max_cost=_money_limit(
            config.budget.system_max_cost_per_run, config, quote,
            "resolved system-wide per-run limit",
        ),
        allow_unbounded_paid_requests=config.budget.allow_unbounded_paid_requests,
    )
    run = BudgetLimits(
        max_cost=_money_limit(
            config.budget.max_cost_per_run, config, quote, "resolved run-specific limit",
        ),
        max_requests=config.budget.max_provider_requests,
        max_input_tokens=config.budget.max_input_tokens,
        max_output_tokens=config.budget.max_output_tokens,
        allow_unbounded_paid_requests=config.budget.allow_unbounded_paid_requests,
    )
    return system, run


def _pricing_terms(quote: PricingQuote) -> dict[str, Any] | None:
    if quote.pricing is None:
        return None
    terms = quote.pricing.to_dict()
    for provenance_field in ("source", "retrieved_at", "version"):
        terms.pop(provenance_field, None)
    return terms


class _RoundTickingObserver:
    """Adds budget state and progress ticks at the runtime callback boundary."""

    def __init__(self, recorder: RunRecorder, guard: RuntimeBudgetGuard, progress: ExperimentProgress, episode_label: str) -> None:
        self.recorder = recorder
        self.guard = guard
        self.progress = progress
        self.episode_label = episode_label

    def event(self, event_type: str, **payload: Any) -> None:
        self.recorder.event(event_type, **payload)

    def record_attempt(self, **payload: Any) -> None:
        self.recorder.record_attempt(**payload, budget_status=self.guard.status())

    def record_interaction(self, **payload: Any) -> None:
        self.recorder.record_interaction(**payload, budget_status=self.guard.checkpoint_state())
        self.progress.round_tick(self.episode_label, payload.get("round_index"))


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    episode_id: str
    seed: int
    status: str  # "completed" | "failed" | "skipped_resumed" | "skipped_aborted"
    interactions: int | None = None
    termination_reason: str | None = None
    error_type: str | None = None
    error: str | None = None
    cell_id: str | None = None  # None outside a grid

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id, "seed": self.seed, "status": self.status,
            "interactions": self.interactions, "termination_reason": self.termination_reason,
            "error_type": self.error_type, "error": self.error, "cell_id": self.cell_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeOutcome":
        return cls(
            episode_id=str(value["episode_id"]), seed=int(value["seed"]), status=str(value["status"]),
            interactions=value.get("interactions"), termination_reason=value.get("termination_reason"),
            error_type=value.get("error_type"), error=value.get("error"), cell_id=value.get("cell_id"),
        )


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    run_id: str
    experiment_name: str
    game_type: str
    episode_count: int
    output_dir: Path
    outcomes: tuple[EpisodeOutcome, ...]
    preflight: ExperimentPreflightEstimate
    budget_status: Mapping[str, Any]
    started_at: str
    finished_at: str

    @property
    def completed(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "completed")

    @property
    def failed(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "failed")

    @property
    def skipped_resumed(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "skipped_resumed")

    @property
    def skipped_aborted(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "skipped_aborted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "game_type": self.game_type,
            "episode_count": self.episode_count,
            "completed": self.completed,
            "failed": self.failed,
            "skipped_resumed": self.skipped_resumed,
            "skipped_aborted": self.skipped_aborted,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "preflight": self.preflight.to_dict(),
            "actual_budget_status": dict(self.budget_status),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


@dataclass(frozen=True, slots=True)
class _EpisodeTask:
    """One episode to run, fully self-contained — the shared unit across a
    single experiment's episodes and a grid's flattened cross-cell episodes."""

    episode_id: str
    seed: int
    config: RunConfig
    episode_dir: Path
    cell_id: str | None = None


async def _execute_episode(
    game: Game,
    episode_config: RunConfig,
    guarded_provider: Any,
    guard: RuntimeBudgetGuard,
    episode_dir: Path,
    episode_label: str,
    *,
    policy: DetailedAuditPolicy,
    metrics: tuple[Any, ...],
    to_round_view: Callable[[Any], Any] | None,
    price_hash: str,
    checkpoint_enabled: bool,
    progress: ExperimentProgress,
) -> tuple[int | None, str | None]:
    """Run one episode; return ``(interaction_count, termination_reason)``."""

    if episode_config.game.type == "naming_convention":
        recorder = RunRecorder(
            episode_dir, run_id=episode_label, resolved_config=episode_config.to_dict(),
            policy=policy,
            # Per-episode Comet experiments are not wired here: one mas_cc
            # experiment would otherwise fan out into N remote experiments.
            comet_enabled=False, checkpoint_enabled=checkpoint_enabled,
            price_snapshot_hash=price_hash, metrics=metrics, to_round_view=to_round_view,
            binning=episode_config.metrics.binning_policy(episode_config.game.population_size),
        )
        observer = _RoundTickingObserver(recorder, guard, progress, episode_label)
        control = create_control(episode_config.control)
        try:
            result = await run_naming_convention_game(
                game, episode_config, guarded_provider, observer=observer, control=control
            )
        except Exception as exc:
            recorder.event("run_failed", error_type=type(exc).__name__, error=str(exc))
            recorder.finalize(status="failed", budget_status=guard.checkpoint_state())
            raise
        recorder.finalize(status="completed", budget_status=guard.checkpoint_state())
        return len(result.interactions), result.termination_reason

    result = await run_game(game, episode_config, guarded_provider)
    progress.round_tick(episode_label, None, count=len(result.interactions))
    _write(episode_dir / "result.json", _json(result.to_dict()))
    return len(result.interactions), result.termination_reason


async def _run_episode_task(
    task: _EpisodeTask,
    *,
    game: Game,
    guarded_provider: Any,
    guard: RuntimeBudgetGuard,
    semaphore: asyncio.Semaphore,
    abort: asyncio.Event,
    fail_fast: bool,
    resume: bool,
    policy: DetailedAuditPolicy,
    metrics: tuple[Any, ...],
    to_round_view: Callable[[Any], Any] | None,
    price_hash: str,
    checkpoint_enabled: bool,
    progress: ExperimentProgress,
) -> EpisodeOutcome:
    manifest_path = task.episode_dir / "manifest.json"
    label = task.episode_id if task.cell_id is None else f"{task.cell_id}/{task.episode_id}"
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed":
            outcome = replace(EpisodeOutcome.from_dict(manifest), status="skipped_resumed")
            progress.episode_done(label, outcome.status)
            return outcome
    async with semaphore:
        if fail_fast and abort.is_set():
            outcome = EpisodeOutcome(task.episode_id, task.seed, "skipped_aborted", cell_id=task.cell_id)
            progress.episode_done(label, outcome.status)
            return outcome
        try:
            interactions, termination_reason = await _execute_episode(
                game, task.config, guarded_provider, guard, task.episode_dir, label,
                policy=policy, metrics=metrics, to_round_view=to_round_view,
                price_hash=price_hash, checkpoint_enabled=checkpoint_enabled, progress=progress,
            )
            outcome = EpisodeOutcome(
                task.episode_id, task.seed, "completed", interactions=interactions,
                termination_reason=termination_reason, cell_id=task.cell_id,
            )
        except Exception as exc:
            if fail_fast:
                abort.set()
            outcome = EpisodeOutcome(
                task.episode_id, task.seed, "failed", error_type=type(exc).__name__,
                error=str(exc), cell_id=task.cell_id,
            )
            LOGGER.error("episode %s failed: %s: %s", label, type(exc).__name__, exc)
        _write(manifest_path, _json(outcome.to_dict()))
        progress.episode_done(label, outcome.status)
        return outcome


async def _run_task_batch(
    tasks: list[_EpisodeTask], **kwargs: Any
) -> tuple[EpisodeOutcome, ...]:
    return tuple(await asyncio.gather(*(_run_episode_task(task, **kwargs) for task in tasks)))


def _write_experiment_summary(run_dir: Path, config: RunConfig, result: ExperimentResult) -> None:
    _write(run_dir / "experiment_summary.json", _json(result.to_dict()))
    fieldnames = ["episode_id", "seed", "status", "interactions", "termination_reason", "error_type", "error"]
    output_path = run_dir / "experiment_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in outcome.to_dict().items() if key != "cell_id"}
            for outcome in result.outcomes
        )
    manifest = {
        "schema_version": 1, "run_id": result.run_id, "experiment_name": result.experiment_name,
        "game_type": result.game_type, "episode_count": result.episode_count,
        "started_at": result.started_at, "finished_at": result.finished_at,
    }
    _write(run_dir / "manifest.json", _json(manifest))
    _write(run_dir / "resolved_config.yaml", resolved_config_yaml(config))


async def run_experiment(
    config: RunConfig,
    output_dir: str | Path,
    *,
    resume: bool = True,
    show_progress: bool = True,
    explicit_override: bool | None = None,
) -> ExperimentResult:
    """Run ``config.execution.repetitions`` episodes concurrently and summarize them."""

    if config.execution.repetitions < 1:
        raise ValueError("config.execution.repetitions must be at least 1")
    override = (
        config.pricing.explicit_unknown_price_override
        if explicit_override is None else explicit_override
    )

    game = create_game(config.game)
    plan = game.call_plan(config.game)
    quote = _quote(config)
    system_budget, run_budget = _budgets(config, quote)

    preflight = static_experiment_preflight(
        plan, config.prompt, config.llm_provider,
        episode_count=config.execution.repetitions,
        concurrency=config.execution.parallelism,
        assumed_output_tokens=config.llm_provider.max_output_tokens,
        pricing_quote=quote, system_budget=system_budget, run_budget=run_budget,
        explicit_override=override,
        allow_stale_pricing=not config.pricing.require_fresh_at_launch,
    )
    if preflight.launch_status != "permitted":
        raise ValueError(
            f"experiment preflight launch status is {preflight.launch_status!r}; "
            "no provider calls sent"
        )

    # Live University metadata is queried once during preflight and immediately
    # revalidated before launch, never once per episode or per completion.
    runtime_quote = _quote(config) if config.pricing.mode == "live" else quote
    if _pricing_terms(runtime_quote) != _pricing_terms(quote):
        raise ValueError("live pricing changed during immediate pre-launch revalidation")

    run_id = f"{config.experiment.name}-{config.execution.seed}"
    run_dir = results_run_dir(
        output_dir, game=config.game.type, experiment=config.experiment.name, run_id=run_id
    )
    episodes_dir = run_dir / "data" / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    budget_description = "unbounded" if run_budget.max_cost is None and system_budget.max_cost is None else (
        format_money(run_budget.max_cost) if run_budget.max_cost is not None else format_money(system_budget.max_cost)
    )
    scenarios = preflight.per_episode.prompt_scenarios
    definition_hash = scenarios[0]["representative"]["definition_hash"] if scenarios else ""
    print_banner(
        format_banner(
            experiment_name=config.experiment.name, game_type=config.game.type,
            game_version=game.spec.version, provider=config.llm_provider.type,
            model=config.llm_provider.model, episode_count=config.execution.repetitions,
            concurrency=config.execution.parallelism, prompt_family=config.prompt.prompt_family,
            prompt_version=config.prompt.prompt_version, prompt_definition_hash=definition_hash,
            budget_description=budget_description,
            preflight_expected_cost=format_money(preflight.total_costs.expected),
            preflight_conservative_cost=format_money(preflight.total_costs.conservative),
            preflight_status=preflight.launch_status,
        )
    )

    effective_budget = resolve_budget_limits(system_budget, run_budget)
    guard = RuntimeBudgetGuard(effective_budget)
    provider = create_llm_provider(config.llm_provider)
    guarded_provider = BudgetGuardedProvider(
        provider, guard, runtime_quote.pricing,
        input_token_estimator=estimate_input_tokens, input_token_multiplier=1.0,
    )

    root_seed = Seed(config.execution.seed)

    def _episode_task(index: int) -> _EpisodeTask:
        episode_id = f"{run_id}-{index:04d}"
        seed = int(root_seed.derive(f"episode:{index}"))
        return _EpisodeTask(
            episode_id=episode_id, seed=seed,
            config=replace(config, execution=replace(config.execution, seed=seed)),
            episode_dir=episodes_dir / episode_id,
        )

    tasks = [_episode_task(index) for index in range(config.execution.repetitions)]
    metrics, to_round_view = game_metrics(game)
    policy = DetailedAuditPolicy.from_mapping(config.logging.options.get("detailed_prompt_audit"))
    price_hash = price_snapshot_hash(runtime_quote.to_dict())

    progress = ExperimentProgress(
        total_episodes=config.execution.repetitions,
        total_rounds=plan.interactions.expected * config.execution.repetitions,
        show=show_progress,
    )
    semaphore = asyncio.Semaphore(config.execution.parallelism)
    abort = asyncio.Event()
    started_at = _now()

    try:
        outcomes = await _run_task_batch(
            tasks, game=game, guarded_provider=guarded_provider, guard=guard,
            semaphore=semaphore, abort=abort, fail_fast=config.execution.fail_fast,
            resume=resume, policy=policy, metrics=metrics, to_round_view=to_round_view,
            price_hash=price_hash, checkpoint_enabled=config.storage.checkpoints, progress=progress,
        )
    finally:
        guarded_provider.close()
        progress.close()

    result = ExperimentResult(
        run_id=run_id, experiment_name=config.experiment.name, game_type=config.game.type,
        episode_count=config.execution.repetitions, output_dir=run_dir, outcomes=outcomes,
        preflight=preflight, budget_status=guard.status(), started_at=started_at, finished_at=_now(),
    )
    _write_experiment_summary(run_dir, config, result)
    return result


def run_experiment_sync(*args: Any, **kwargs: Any) -> ExperimentResult:
    return asyncio.run(run_experiment(*args, **kwargs))


@dataclass(frozen=True, slots=True)
class GridCellResult:
    cell_id: str
    overrides: Mapping[str, Any]
    outcomes: tuple[EpisodeOutcome, ...]

    @property
    def completed(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "completed")

    @property
    def failed(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "failed")

    @property
    def skipped_resumed(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "skipped_resumed")

    @property
    def skipped_aborted(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "skipped_aborted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "overrides": dict(self.overrides),
            "completed": self.completed,
            "failed": self.failed,
            "skipped_resumed": self.skipped_resumed,
            "skipped_aborted": self.skipped_aborted,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


@dataclass(frozen=True, slots=True)
class GridResult:
    grid_id: str
    run_id: str
    experiment_name: str
    game_type: str
    output_dir: Path
    cells: tuple[GridCellResult, ...]
    preflight: GridPreflightEstimate
    budget_status: Mapping[str, Any]
    started_at: str
    finished_at: str

    @property
    def completed(self) -> int:
        return sum(cell.completed for cell in self.cells)

    @property
    def failed(self) -> int:
        return sum(cell.failed for cell in self.cells)

    @property
    def skipped_resumed(self) -> int:
        return sum(cell.skipped_resumed for cell in self.cells)

    @property
    def skipped_aborted(self) -> int:
        return sum(cell.skipped_aborted for cell in self.cells)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "grid_id": self.grid_id,
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "game_type": self.game_type,
            "cell_count": len(self.cells),
            "completed": self.completed,
            "failed": self.failed,
            "skipped_resumed": self.skipped_resumed,
            "skipped_aborted": self.skipped_aborted,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "preflight": self.preflight.to_dict(),
            "actual_budget_status": dict(self.budget_status),
            "cells": [cell.to_dict() for cell in self.cells],
        }


def _write_grid_summary(grid_dir: Path, grid: GridSpec, result: GridResult) -> None:
    _write(grid_dir / "grid_summary.json", _json(result.to_dict()))
    fieldnames = ["cell_id", "completed", "failed", "skipped_resumed", "skipped_aborted", "overrides"]
    output_path = grid_dir / "grid_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for cell in result.cells:
            writer.writerow(
                {
                    "cell_id": cell.cell_id, "completed": cell.completed, "failed": cell.failed,
                    "skipped_resumed": cell.skipped_resumed, "skipped_aborted": cell.skipped_aborted,
                    "overrides": json.dumps(dict(cell.overrides), sort_keys=True, default=str),
                }
            )
    manifest = {
        "schema_version": 1, "grid_id": result.grid_id, "run_id": result.run_id,
        "experiment_name": result.experiment_name, "game_type": result.game_type,
        "cell_count": len(result.cells), "started_at": result.started_at,
        "finished_at": result.finished_at,
    }
    _write(grid_dir / "manifest.json", _json(manifest))
    _write(grid_dir / "resolved_base_config.yaml", resolved_config_yaml(grid.base))


async def run_experiment_grid(
    grid: GridSpec,
    output_dir: str | Path,
    *,
    resume: bool = True,
    show_progress: bool = True,
    explicit_override: bool | None = None,
) -> GridResult:
    """Run every cell's episodes under one combined concurrency and budget pool.

    Every cell shares one provider client, one pricing quote, and one
    :class:`RuntimeBudgetGuard` — `GridSpec` already forbids sweeping the
    fields (provider identity, budget, pricing) that would make that unsafe.
    ``execution.fail_fast``/``execution.parallelism``/``execution.seed`` are
    read from the *base* config and apply to the whole grid, not per cell.
    """

    cells = grid.cells
    for cell in cells:
        if cell.config.execution.repetitions < 1:
            raise ValueError("every cell's execution.repetitions must be at least 1")

    base = grid.base
    override = (
        base.pricing.explicit_unknown_price_override if explicit_override is None else explicit_override
    )
    game = create_game(base.game)
    quote = _quote(base)
    system_budget, run_budget = _budgets(base, quote)

    preflight = static_grid_preflight(
        grid, concurrency=base.execution.parallelism,
        assumed_output_tokens=base.llm_provider.max_output_tokens,
        pricing_quote=quote, system_budget=system_budget, run_budget=run_budget,
        explicit_override=override, allow_stale_pricing=not base.pricing.require_fresh_at_launch,
    )
    if preflight.launch_status != "permitted":
        raise ValueError(
            f"grid preflight launch status is {preflight.launch_status!r}; no provider calls sent"
        )

    runtime_quote = _quote(base) if base.pricing.mode == "live" else quote
    if _pricing_terms(runtime_quote) != _pricing_terms(quote):
        raise ValueError("live pricing changed during immediate pre-launch revalidation")

    run_id = f"{base.experiment.name}-{base.execution.seed}"
    grid_dir = results_run_dir(
        output_dir, game=base.game.type, experiment=base.experiment.name, run_id=run_id
    )
    cells_dir = grid_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    print_banner(
        format_grid_banner(
            experiment_name=base.experiment.name, game_type=base.game.type,
            game_version=game.spec.version, provider=base.llm_provider.type,
            model=base.llm_provider.model, cell_count=len(cells),
            total_episode_count=preflight.total_episode_count, concurrency=base.execution.parallelism,
            axes=tuple((axis.path, len(axis.values)) for axis in grid.axes),
            preflight_expected_cost=format_money(preflight.total_costs.expected),
            preflight_conservative_cost=format_money(preflight.total_costs.conservative),
            preflight_status=preflight.launch_status,
        )
    )

    effective_budget = resolve_budget_limits(system_budget, run_budget)
    guard = RuntimeBudgetGuard(effective_budget)
    provider = create_llm_provider(base.llm_provider)
    guarded_provider = BudgetGuardedProvider(
        provider, guard, runtime_quote.pricing,
        input_token_estimator=estimate_input_tokens, input_token_multiplier=1.0,
    )

    metrics, to_round_view = game_metrics(game)
    policy = DetailedAuditPolicy.from_mapping(base.logging.options.get("detailed_prompt_audit"))
    price_hash = price_snapshot_hash(runtime_quote.to_dict())

    grid_seed = Seed(base.execution.seed)
    all_tasks: list[_EpisodeTask] = []
    total_rounds = 0
    for cell in cells:
        cell_seed = grid_seed.derive(f"grid-cell:{cell.index}")
        cell_dir = cells_dir / cell.cell_id
        _write(cell_dir / "resolved_config.yaml", resolved_config_yaml(cell.config))
        _write(cell_dir / "overrides.json", _json(cell.to_dict()))
        episodes_dir = cell_dir / "data" / "episodes"
        plan = game.call_plan(cell.config.game)
        total_rounds += plan.interactions.expected * cell.config.execution.repetitions
        for index in range(cell.config.execution.repetitions):
            episode_seed = int(cell_seed.derive(f"episode:{index}"))
            episode_id = f"{cell.cell_id}-{index:04d}"
            all_tasks.append(
                _EpisodeTask(
                    episode_id=episode_id, seed=episode_seed,
                    config=replace(cell.config, execution=replace(cell.config.execution, seed=episode_seed)),
                    episode_dir=episodes_dir / episode_id, cell_id=cell.cell_id,
                )
            )

    progress = ExperimentProgress(
        total_episodes=len(all_tasks), total_rounds=total_rounds, show=show_progress,
    )
    semaphore = asyncio.Semaphore(base.execution.parallelism)
    abort = asyncio.Event()
    started_at = _now()

    try:
        outcomes = await _run_task_batch(
            all_tasks, game=game, guarded_provider=guarded_provider, guard=guard,
            semaphore=semaphore, abort=abort, fail_fast=base.execution.fail_fast,
            resume=resume, policy=policy, metrics=metrics, to_round_view=to_round_view,
            price_hash=price_hash, checkpoint_enabled=base.storage.checkpoints, progress=progress,
        )
    finally:
        guarded_provider.close()
        progress.close()

    by_cell: dict[str, list[EpisodeOutcome]] = {cell.cell_id: [] for cell in cells}
    for outcome in outcomes:
        by_cell[outcome.cell_id].append(outcome)
    cell_results = tuple(
        GridCellResult(cell.cell_id, cell.overrides, tuple(by_cell[cell.cell_id])) for cell in cells
    )

    result = GridResult(
        grid_id=grid.grid_id, run_id=run_id, experiment_name=base.experiment.name,
        game_type=base.game.type, output_dir=grid_dir, cells=cell_results, preflight=preflight,
        budget_status=guard.status(), started_at=started_at, finished_at=_now(),
    )
    _write_grid_summary(grid_dir, grid, result)
    for cell, cell_result in zip(cells, cell_results):
        cell_dir = cells_dir / cell.cell_id
        _write(
            cell_dir / "cell_summary.json",
            _json(
                {
                    "cell_id": cell_result.cell_id, "overrides": dict(cell_result.overrides),
                    "completed": cell_result.completed, "failed": cell_result.failed,
                    "skipped_resumed": cell_result.skipped_resumed,
                    "skipped_aborted": cell_result.skipped_aborted,
                    "outcomes": [outcome.to_dict() for outcome in cell_result.outcomes],
                }
            ),
        )
    return result


def run_experiment_grid_sync(*args: Any, **kwargs: Any) -> GridResult:
    return asyncio.run(run_experiment_grid(*args, **kwargs))
