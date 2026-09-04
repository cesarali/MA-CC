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
import contextlib
import contextvars
import csv
import gzip
import json
import logging
import os
import resource
import shutil
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mas_cc.config import GridSpec, RunConfig, resolved_config_yaml
from mas_cc.control import create_control
from mas_cc.core.random import Seed
from mas_cc.games import Game, create_game, game_metrics
from mas_cc.games.naming_convention.runtime import run_naming_convention_game
from mas_cc.games.runner import run_game
from mas_cc.llm_runtime.providers import (
    AtomicBudgetStateStore,
    BUDGET_STOP_CODES,
    BudgetExpectation,
    BudgetGuardedProvider,
    BudgetLimits,
    CachedPricingSource,
    MonetaryAmount,
    OfflinePricingSource,
    PricingQuote,
    ProviderError,
    RuntimeBudgetGuard,
    UniversityPricingSource,
    create_llm_provider,
    resolve_budget_limits,
)
from mas_cc.llm_runtime.prompts import (
    PromptMarkdownLogger,
    render_prompt_request_markdown,
)
from mas_cc.observability import DetailedAuditPolicy, RunRecorder
from mas_cc.planning import (
    ExperimentPreflightEstimate,
    GameCallPlan,
    GridPreflightEstimate,
    estimate_input_tokens,
    static_experiment_preflight,
    static_grid_preflight,
)
from mas_cc.storage import (
    SCIENTIFIC_SCHEMA_VERSION,
    ScientificIdentity,
    canonical_hash,
    discover_episode_artifact,
    episode_shard_path,
    file_sha256,
    merge_cell_scientific_tables,
    merge_episode_artifacts,
    prompt_definition_hash,
    results_run_dir,
    validate_cell_artifact,
    validate_episode_artifact,
    validate_episode_frame,
    validate_semantic_stream,
)

from .aggregation import GridAggregator, aggregation_ground_truth
from .comet_monitor import CellLayout, MasterMonitor, SweepLayout, sweep_parameters
from .configured_analysis import (
    per_cell_reports_enabled,
    run_configured_analysis,
    run_configured_cell_analysis,
    validate_configured_analysis,
)
from .console import (
    ExperimentProgress,
    format_banner,
    format_grid_banner,
    format_money,
    print_banner,
)

LOGGER = logging.getLogger("mas_cc.experiment")
_TIMING_EPISODE: contextvars.ContextVar[tuple[str | None, str] | None] = (
    contextvars.ContextVar("mas_cc_timing_episode", default=None)
)


class _TimingProvider:
    """Metadata-only provider decorator used by the timing-study profile."""

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self.name = provider.name
        self.model = provider.model
        self.capabilities = provider.capabilities
        self.rows: list[dict[str, Any]] = []

    async def complete(self, request: Any) -> Any:
        context = _TIMING_EPISODE.get()
        started_at = _now()
        started = time.perf_counter()
        try:
            response = await self._provider.complete(request)
        except Exception as exc:
            self.rows.append(
                {
                    "cell_id": None if context is None else context[0],
                    "episode_id": None if context is None else context[1],
                    "started_at": started_at,
                    "finished_at": _now(),
                    "wall_seconds": round(time.perf_counter() - started, 6),
                    "provider_latency_seconds": None,
                    "retries": None,
                    "status_code": getattr(exc, "status_code", None),
                    "error_type": type(exc).__name__,
                }
            )
            raise
        self.rows.append(
            {
                "cell_id": None if context is None else context[0],
                "episode_id": None if context is None else context[1],
                "started_at": started_at,
                "finished_at": _now(),
                "wall_seconds": round(time.perf_counter() - started, 6),
                "provider_latency_seconds": response.latency_seconds,
                "retries": response.retries,
                "status_code": response.status_code,
                "error_type": None,
            }
        )
        return response

    def close(self) -> None:
        self._provider.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _write_timing_study(
    run_dir: Path,
    outcomes: Sequence["EpisodeOutcome"],
    *,
    profile: str,
    run_started_at: str,
    run_finished_at: str,
    cell_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    request_rows: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Write compact timing evidence for normal runs and detailed rows for studies."""

    grouped: dict[str, list[EpisodeOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.cell_id or "run", []).append(outcome)

    lines = [
        "# Timing study",
        "",
        f"- Artifact profile: `{profile}`",
        f"- Run started: `{run_started_at}`",
        f"- Run finished: `{run_finished_at}`",
        f"- Total wall time: `{(_parse_timestamp(run_finished_at) - _parse_timestamp(run_started_at)).total_seconds():.3f}` seconds",
        "",
        "| Cell | Episodes | Completed | Failed | Cell wall (s) | Mean queue (s) | Mean episode (s) | Median (s) | P95 (s) | Episodes/min | Overrides |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cell_id, cell_outcomes in sorted(grouped.items()):
        durations = [
            float(item.duration_seconds)
            for item in cell_outcomes
            if item.duration_seconds is not None and item.status != "skipped_resumed"
        ]
        queue_waits = [
            float(item.queue_wait_seconds)
            for item in cell_outcomes
            if item.queue_wait_seconds is not None and item.status != "skipped_resumed"
        ]
        starts = [
            _parse_timestamp(item.started_at)
            for item in cell_outcomes
            if item.started_at
        ]
        finishes = [
            _parse_timestamp(item.finished_at)
            for item in cell_outcomes
            if item.finished_at
        ]
        wall = (
            (max(finishes) - min(starts)).total_seconds()
            if starts and finishes
            else 0.0
        )
        completed = sum(
            item.status in {"completed", "skipped_resumed"} for item in cell_outcomes
        )
        failed = sum(item.status == "failed" for item in cell_outcomes)
        mean = sum(durations) / len(durations) if durations else 0.0
        mean_queue = sum(queue_waits) / len(queue_waits) if queue_waits else 0.0
        median = _percentile(durations, 0.5) or 0.0
        p95 = _percentile(durations, 0.95) or 0.0
        rate = completed / (wall / 60.0) if wall > 0 else 0.0
        overrides = json.dumps(
            dict((cell_overrides or {}).get(cell_id, {})), sort_keys=True
        )
        lines.append(
            f"| `{cell_id}` | {len(cell_outcomes)} | {completed} | {failed} | {wall:.3f} | "
            f"{mean_queue:.3f} | {mean:.3f} | {median:.3f} | {p95:.3f} | {rate:.3f} | `{overrides}` |"
        )
    lines.extend(
        [
            "",
            "Episode duration is measured after an episode acquires an execution-parallelism slot and excludes queue wait time. Cell wall time spans the first episode start through the last episode finish.",
            "",
        ]
    )
    request_durations = [float(row["wall_seconds"]) for row in request_rows]
    provider_latencies = [
        float(row["provider_latency_seconds"])
        for row in request_rows
        if row.get("provider_latency_seconds") is not None
    ]
    retries = sum(int(row.get("retries") or 0) for row in request_rows)
    errors = sum(bool(row.get("error_type")) for row in request_rows)
    rate_limits = sum(row.get("status_code") == 429 for row in request_rows)
    rate_wall = (
        _parse_timestamp(run_finished_at) - _parse_timestamp(run_started_at)
    ).total_seconds()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    lines.extend(
        [
            "## Operational summary",
            "",
            f"- Requests observed: `{len(request_rows)}`",
            f"- Request throughput: `{(len(request_rows) / rate_wall if rate_wall > 0 else 0.0):.3f}` requests/second",
            f"- Mean request wall time: `{(sum(request_durations) / len(request_durations) if request_durations else 0.0):.3f}` seconds",
            f"- P50/P95 request wall time: `{(_percentile(request_durations, 0.5) or 0.0):.3f}` / `{(_percentile(request_durations, 0.95) or 0.0):.3f}` seconds",
            f"- P50/P95 provider-reported latency: `{(_percentile(provider_latencies, 0.5) or 0.0):.3f}` / `{(_percentile(provider_latencies, 0.95) or 0.0):.3f}` seconds",
            f"- Retries: `{retries}`; request errors: `{errors}`; HTTP 429 responses: `{rate_limits}`",
            f"- Process user/system CPU: `{usage.ru_utime:.3f}` / `{usage.ru_stime:.3f}` seconds",
            f"- Peak resident memory: `{int(usage.ru_maxrss)}` KiB",
            "",
        ]
    )
    _write(run_dir / "timing_study.md", "\n".join(lines))

    if profile != "timing_study":
        return
    fields = [
        "cell_id",
        "episode_id",
        "seed",
        "status",
        "interactions",
        "queued_at",
        "started_at",
        "finished_at",
        "queue_wait_seconds",
        "duration_seconds",
        "termination_reason",
        "error_type",
        "error",
    ]
    with (run_dir / "timing_study.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for outcome in outcomes:
            row = outcome.to_dict()
            writer.writerow({field: row.get(field) for field in fields})
    request_fields = [
        "cell_id",
        "episode_id",
        "started_at",
        "finished_at",
        "wall_seconds",
        "provider_latency_seconds",
        "retries",
        "status_code",
        "error_type",
    ]
    with (run_dir / "request_timing.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=request_fields)
        writer.writeheader()
        for row in request_rows:
            writer.writerow({field: row.get(field) for field in request_fields})


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass


def _record_cell_hashes(cell_dir: Path) -> None:
    """Add every retained cell result to an already-durable completion seal."""

    seal_path = cell_dir / "cell_complete.json"
    if not seal_path.is_file():
        return
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    artifacts = {}
    for path in sorted(cell_dir.rglob("*")):
        if (
            not path.is_file()
            or path == seal_path
            or path.name == "manifest.json"
            or ".resume" in path.parts
            or path.name.endswith(".tmp")
            or path.name.endswith(":Zone.Identifier")
        ):
            continue
        artifacts[str(path.relative_to(cell_dir))] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    seal["artifacts"] = artifacts
    _write_atomic(seal_path, _json(seal))


def _record_results_only_hashes(run_dir: Path) -> None:
    """Finish the run manifest with hashes of every retained result artifact."""

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    cells_dir = run_dir / "cells"
    if cells_dir.is_dir():
        cells = [path for path in cells_dir.iterdir() if path.is_dir()]
        if any(not (cell / "cell_complete.json").is_file() for cell in cells):
            return
        for cell in cells:
            _record_cell_hashes(cell)
    elif not (run_dir / "cell_complete.json").is_file():
        return
    else:
        _record_cell_hashes(run_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = {}
    for path in sorted(run_dir.rglob("*")):
        if (
            not path.is_file()
            or path == manifest_path
            or ".resume" in path.parts
            or path.name.endswith(".tmp")
            or path.name.endswith(":Zone.Identifier")
        ):
            continue
        artifacts[str(path.relative_to(run_dir))] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    manifest["artifacts"] = artifacts
    _write_atomic(manifest_path, _json(manifest))


def _wipe_run_dir_if_requested(
    run_dir: Path, *, wipe_and_recompute: bool, label: str
) -> None:
    """Delete a stale run directory before it is recreated.

    ``storage.wipe_and_recompute`` overrides ``resume``: the point of wiping
    is to discard every prior manifest/episode/aggregate so nothing gets
    skipped as already-completed.
    """

    if wipe_and_recompute and run_dir.exists():
        LOGGER.warning(
            "storage.wipe_and_recompute=true: deleting %s (%s)", run_dir, label
        )
        shutil.rmtree(run_dir)


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
        amount=amount,
        unit=config.budget.accounting_unit,
        unit_source="resolved budget configuration",
        provider=config.llm_provider.type,
        model=config.llm_provider.model,
        source=description,
        retrieved_at=quote.retrieved_at,
        version="resolved-config-v1",
    )


def _budgets(
    config: RunConfig, quote: PricingQuote
) -> tuple[BudgetLimits, BudgetLimits]:
    system = BudgetLimits(
        max_cost=_money_limit(
            config.budget.system_max_cost_per_run,
            config,
            quote,
            "resolved system-wide per-run limit",
        ),
        allow_unbounded_paid_requests=config.budget.allow_unbounded_paid_requests,
    )
    run = BudgetLimits(
        max_cost=_money_limit(
            config.budget.max_cost_per_run,
            config,
            quote,
            "resolved run-specific limit",
        ),
        max_requests=config.budget.max_provider_requests,
        max_input_tokens=config.budget.max_input_tokens,
        max_output_tokens=config.budget.max_output_tokens,
        allow_unbounded_paid_requests=config.budget.allow_unbounded_paid_requests,
    )
    return system, run


def _configure_durable_budget(
    config: RunConfig,
    run_dir: Path,
    guard: RuntimeBudgetGuard,
    *,
    price_hash: str,
    resume: bool,
) -> AtomicBudgetStateStore | None:
    if not config.storage.retention_policy.compact_scientific:
        return None
    store = AtomicBudgetStateStore(
        run_dir / "budget_state.json",
        resolved_budget_hash=canonical_hash(config.budget.to_dict()),
        pricing_snapshot_hash=price_hash,
    )
    if resume and config.storage.checkpoint_mode == "episode":
        store.restore(guard)
    guard.set_durable_state_sink(store.write)
    return store


def _expectation(preflight: Any) -> BudgetExpectation:
    """Preflight's conservative totals, handed to the guard as advisory only.

    Both the experiment and grid estimates expose the same three
    `EstimateRange` totals, so one helper covers both callers.
    """

    return BudgetExpectation(
        requests=preflight.total_provider_requests.conservative,
        input_tokens=preflight.total_input_tokens.conservative,
        output_tokens=preflight.total_output_tokens.conservative,
    )


class _LiveSpendWatcher:
    """Stops a run when the provider's own accounting says it has spent enough.

    The guard's internal cost counter is only as good as the price table it was
    built from. This is the independent check: it asks the provider what the
    account has actually been charged, and compares the *delta since launch*
    against `budget.max_cost_per_run`. Measuring the delta rather than the
    absolute balance is what keeps a per-run ceiling meaningful on a key that
    other people are also spending against.

    Failures to poll are logged and retried, never fatal - losing sight of the
    live number leaves the run under the guard's own cost ceiling, which is
    still enforced, so a flaky metadata endpoint must not kill a healthy run.
    """

    def __init__(
        self,
        source: UniversityPricingSource,
        guard: RuntimeBudgetGuard,
        *,
        unit: str,
        ceiling: float | None,
        poll_seconds: int,
    ) -> None:
        self._source = source
        self._guard = guard
        self._unit = unit
        self._ceiling = ceiling
        self._poll_seconds = poll_seconds
        self._baseline: float | None = None

    async def _read_spend(self) -> float | None:
        budget = await asyncio.to_thread(
            self._source.fetch_account_budget, unit=self._unit
        )
        if budget is None or budget.spent is None:
            return None
        if budget.spent.unit != self._unit:
            # Comparing two different accounting units would produce a
            # confident wrong answer, so the watcher declines to compare at all.
            LOGGER.warning(
                "live spend is reported in %r but the budget is in %r; not comparing",
                budget.spent.unit,
                self._unit,
            )
            return None
        return budget.spent.amount

    async def start(self) -> None:
        """Record the launch baseline. Raises nothing; disables itself on error."""

        try:
            self._baseline = await self._read_spend()
        except Exception as exc:
            LOGGER.warning(
                "could not read the launch spend baseline (%s); live spend watching is off",
                type(exc).__name__,
            )
            self._baseline = None
        if self._baseline is None:
            LOGGER.info("provider reports no account spend; live spend watching is off")
        else:
            LOGGER.info(
                "live spend baseline at launch: %.4f %s", self._baseline, self._unit
            )

    async def run(self) -> None:
        if self._baseline is None:
            return
        while True:
            await asyncio.sleep(self._poll_seconds)
            try:
                current = await self._read_spend()
            except Exception as exc:
                LOGGER.warning(
                    "live spend poll failed (%s); retrying", type(exc).__name__
                )
                continue
            if current is None:
                continue
            spent = max(0.0, current - self._baseline)
            self._guard.record_account_spend(
                {
                    "unit": self._unit,
                    "baseline_at_launch": self._baseline,
                    "latest_total": current,
                    "spent_by_this_run": spent,
                    "observed_at": _now(),
                }
            )
            LOGGER.info("live spend so far this run: %.4f %s", spent, self._unit)
            if self._ceiling is not None and spent >= self._ceiling:
                self._guard.request_stop(
                    f"provider-reported spend for this run ({spent:.4f} {self._unit}) "
                    f"reached the {self._ceiling:.4f} {self._unit} ceiling"
                )
                return


def _spend_watcher(
    config: RunConfig, guard: RuntimeBudgetGuard, limits: BudgetLimits
) -> _LiveSpendWatcher | None:
    """Build the watcher when the config asks for it and the provider can serve it."""

    poll_seconds = config.budget.live_spend_poll_seconds
    if poll_seconds is None:
        return None
    if config.llm_provider.type != "university":
        # Only the university proxy exposes a spend endpoint today. Say so
        # rather than silently running without the protection that was asked for.
        LOGGER.warning(
            "budget.live_spend_poll_seconds is set but provider %r reports no account "
            "spend; the run relies on its own cost ceiling",
            config.llm_provider.type,
        )
        return None
    return _LiveSpendWatcher(
        UniversityPricingSource(
            config.llm_provider,
            freshness=timedelta(seconds=config.pricing.max_age_seconds),
        ),
        guard,
        unit=config.budget.accounting_unit,
        ceiling=None if limits.max_cost is None else limits.max_cost.amount,
        poll_seconds=poll_seconds,
    )


def _pricing_terms(quote: PricingQuote) -> dict[str, Any] | None:
    if quote.pricing is None:
        return None
    terms = quote.pricing.to_dict()
    for provenance_field in ("source", "retrieved_at", "version"):
        terms.pop(provenance_field, None)
    return terms


def _pricing_identity(quote: PricingQuote) -> str:
    """Stable identity of approved rates, excluding retrieval-time metadata."""

    return canonical_hash(
        {
            "provider": quote.provider,
            "model": quote.model,
            "available": quote.available,
            "pricing_terms": _pricing_terms(quote),
        }
    )


def _steps_per_episode(plan: GameCallPlan) -> int:
    """Return the user-visible steps represented by one episode."""

    return int(
        plan.metadata.get("interactions_per_episode", plan.interactions.expected)
    )


def _provider_registry():
    """The kernel's provider registry plus the game-layer synthetic agent.

    `llm_runtime` is a portable kernel that ships no game-specific content, so
    `synthetic_agent` is registered at this application boundary instead - the
    same convention as `register_game_prompt_factories`. Without it a synthetic
    game can only be driven by `mas-cc synthetic`, and never by an
    `experiment run` grid, which is what the empowerment analysis reads.

    The import is deferred because `mas_cc.games.synthetic` pulls in the
    analysis stack, and this module is imported whenever the CLI starts.
    """

    from mas_cc.games.synthetic import create_synthetic_provider_registry

    return create_synthetic_provider_registry()


def _master_monitor(
    config: RunConfig,
    run_id: str,
    total_episodes: int,
    layout: SweepLayout | None = None,
) -> MasterMonitor:
    """Build the run's one sweep-level Comet experiment from `logging.comet`.

    On this path `logging.comet` used to be inert - episode recorders hard-code
    `comet_enabled=False` so that N episodes do not become N remote
    experiments. The flag now means "publish master-level progress"; the shape
    of that publishing is `observability.comet`, and the master is the only
    writer either way.
    """

    return MasterMonitor(
        config.logging.comet,
        project_name=str(config.logging.options.get("comet_project", "mas-cc")),
        run_name=run_id,
        layout=layout,
        total_episodes=total_episodes,
        settings=config.observability.comet,
    )


class _CellCompletion:
    """Fires once per cell, on the episode that brings it to its expected count.

    Counting terminal episodes rather than watching the filesystem keeps the
    trigger exact under `execution.parallelism`: episodes finish out of order,
    and "the directory looks full" is a race while "the last of N reported in"
    is not. Skipped-on-resume episodes count — a resumed cell is complete, and
    its aggregates are recomputed from the files that already exist.
    """

    def __init__(self, expected: Mapping[str, int]) -> None:
        self._expected = dict(expected)
        self._seen: dict[str, int] = {cell_id: 0 for cell_id in expected}

    def record(self, cell_id: str) -> bool:
        if cell_id not in self._expected:
            return False
        self._seen[cell_id] += 1
        return self._seen[cell_id] == self._expected[cell_id]


class _RoundTickingObserver:
    """Adds budget/progress ticks and optional prompt Markdown at runtime."""

    def __init__(
        self,
        recorder: RunRecorder,
        guard: RuntimeBudgetGuard,
        progress: ExperimentProgress,
        episode_label: str,
        *,
        prompt_logger: PromptMarkdownLogger | None = None,
        prompt_example_rounds: int = 0,
        prompt_sampler: "_CellPromptSampler | None" = None,
        prompt_cell_dir: Path | None = None,
        prompt_episode_id: str | None = None,
        prompt_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.recorder = recorder
        self.guard = guard
        self.progress = progress
        self.episode_label = episode_label
        self.prompt_logger = prompt_logger
        self.prompt_example_rounds = prompt_example_rounds
        self.prompt_sampler = prompt_sampler
        self.prompt_cell_dir = prompt_cell_dir
        self.prompt_episode_id = prompt_episode_id or episode_label
        self.prompt_context = dict(prompt_context or {})
        self._logged_rounds: set[int] = set()
        # Compact runs stream every canonical micro/round row immediately.
        # Keeping the rich game result as well duplicates prompts, responses,
        # transitions, and each progressively growing state history in RAM.
        self.retain_result_history = not recorder.retention_policy.compact_scientific

    def event(self, event_type: str, **payload: Any) -> None:
        self.recorder.event(event_type, **payload)

    def record_attempt(self, **payload: Any) -> None:
        self.recorder.record_attempt(**payload, budget_status=self.guard.status())
        round_index = payload.get("round_index")
        attempt = payload.get("attempt")
        if (
            self.prompt_logger is not None
            and attempt == 1
            and round_index not in self._logged_rounds
            and len(self._logged_rounds) < self.prompt_example_rounds
        ):
            agent_id = payload["request"].metadata.get("agent_id")
            self.prompt_logger.log(
                payload["prompt"],
                interaction_id=f"round_{round_index:03d}",
                title=f"Round {round_index} — agent {agent_id}",
                metadata={"round_index": round_index, "agent_id": str(agent_id)},
                response=payload.get("response"),
                validation_error=payload.get("validation_error"),
            )
            self._logged_rounds.add(round_index)
        if (
            self.prompt_sampler is not None
            and self.prompt_cell_dir is not None
            and payload.get("valid")
        ):
            request = payload["request"]
            agent_id = request.metadata.get("agent_id")
            self.prompt_sampler.capture(
                self.prompt_cell_dir,
                self.prompt_episode_id,
                int(round_index),
                render_prompt_request_markdown(
                    payload["prompt"],
                    title=f"Round {round_index} — agent {agent_id}",
                    metadata={"round_index": round_index, "agent_id": str(agent_id)},
                    response=None,
                    validation_error=None,
                ),
                metadata={
                    **self.prompt_context,
                    "agent_id": str(agent_id),
                    "update_index": int(
                        request.metadata.get(
                            "global_update_index",
                            request.metadata.get("interaction_index", 0),
                        )
                    ),
                    "repair_guidance_included": bool(
                        request.metadata.get("validation_repair", False)
                    ),
                    "prompt_definition_hash": payload["prompt"].definition_hash,
                    "prompt_content_hash": payload["prompt"].instance_hash,
                },
            )

    def record_interaction(self, **payload: Any) -> None:
        self.recorder.record_interaction(
            **payload, budget_status=self.guard.checkpoint_state()
        )
        self.progress.round_tick(self.episode_label, payload.get("round_index"))

    def record_semantic_initialization(self, **payload: Any) -> None:
        self.recorder.record_semantic_initialization(**payload)

    def record_semantic_round_start(self, **payload: Any) -> None:
        self.recorder.record_semantic_round_start(**payload)

    def record_trajectory(self, **payload: Any) -> None:
        self.recorder.record_trajectory(**payload)

    def record_round_trajectory(self, **payload: Any) -> None:
        self.recorder.record_round_trajectory(**payload)

    def record_round_boundary(self, **payload: Any) -> None:
        self.recorder.record_round_boundary(
            **payload, budget_status=self.guard.checkpoint_state()
        )


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
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    queued_at: str | None = None
    queue_wait_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        value = {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "status": self.status,
            "interactions": self.interactions,
            "termination_reason": self.termination_reason,
            "error_type": self.error_type,
            "error": self.error,
            "cell_id": self.cell_id,
        }
        if self.started_at is not None:
            value.update(
                started_at=self.started_at,
                finished_at=self.finished_at,
                duration_seconds=self.duration_seconds,
                queued_at=self.queued_at,
                queue_wait_seconds=self.queue_wait_seconds,
            )
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeOutcome":
        return cls(
            episode_id=str(value["episode_id"]),
            seed=int(value["seed"]),
            status=str(value["status"]),
            interactions=value.get("interactions"),
            termination_reason=value.get("termination_reason"),
            error_type=value.get("error_type"),
            error=value.get("error"),
            cell_id=value.get("cell_id"),
            started_at=value.get("started_at"),
            finished_at=value.get("finished_at"),
            duration_seconds=value.get("duration_seconds"),
            queued_at=value.get("queued_at"),
            queue_wait_seconds=value.get("queue_wait_seconds"),
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
        return sum(
            1 for outcome in self.outcomes if outcome.status == "skipped_resumed"
        )

    @property
    def skipped_aborted(self) -> int:
        return sum(
            1 for outcome in self.outcomes if outcome.status == "skipped_aborted"
        )

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
    cell_dir: Path | None = None
    scientific_identity: ScientificIdentity | None = None

    @property
    def scientific_path(self) -> Path | None:
        if self.scientific_identity is None:
            return None
        return episode_shard_path(
            self.cell_dir or self.episode_dir.parent.parent.parent, self.episode_id
        )

    @property
    def manifest_path(self) -> Path:
        if self.scientific_path is not None:
            return self.scientific_path.parent / "manifest.json"
        return self.episode_dir / "manifest.json"


class _CellPromptSampler:
    """Bounded deterministic prompt candidates, rendered once per cell."""

    def __init__(self, count: int) -> None:
        self.count = max(0, int(count))
        self._lock = threading.Lock()

    @staticmethod
    def _path(cell_dir: Path, episode_id: str) -> Path:
        return cell_dir / ".resume" / episode_id / "prompt_candidates.json.gz"

    def capture(
        self,
        cell_dir: Path,
        episode_id: str,
        round_index: int,
        markdown: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self.count == 0:
            return
        path = self._path(cell_dir, episode_id)
        with self._lock:
            candidates: list[dict[str, Any]] = []
            if path.is_file():
                try:
                    with gzip.open(path, "rt", encoding="utf-8") as stream:
                        candidates = list(json.load(stream))
                except (OSError, ValueError, TypeError):
                    candidates = []
            item = {
                "round_index": int(round_index),
                "markdown": markdown,
                **dict(metadata or {}),
            }
            if self.count == 3 and item.get("rounds"):
                rounds = int(item["rounds"])
                targets = {
                    "beginning": 0,
                    "middle": (rounds - 1) // 2,
                    "end": rounds - 1,
                }
                labels = [
                    label for label, target in targets.items() if target == round_index
                ]
                if not labels:
                    return
                item["sample_point"] = labels[0]
                previous = next(
                    (
                        value
                        for value in candidates
                        if value.get("sample_point") == item["sample_point"]
                    ),
                    None,
                )
                ordering = (
                    int(item.get("update_index", 0)),
                    str(item.get("agent_id", "")),
                )
                if previous is not None and ordering > (
                    int(previous.get("update_index", 0)),
                    str(previous.get("agent_id", "")),
                ):
                    return
                candidates = [
                    value
                    for value in candidates
                    if value.get("sample_point") != item["sample_point"]
                ]
                candidates.append(item)
                candidates.sort(
                    key=lambda value: (
                        ("beginning", "middle", "end").index(value["sample_point"]),
                        int(value.get("update_index", 0)),
                        str(value.get("agent_id", "")),
                    )
                )
                candidates = candidates[:3]
            elif len(candidates) < self.count:
                candidates.append(item)
            elif self.count == 1:
                # A single example is deliberately the initial prompt shape.
                return
            else:
                # Preserve the earliest count-1 and continually replace the
                # final slot, yielding an early/late sample for count=2.
                candidates[-1] = item
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8") as stream:
                json.dump(candidates, stream, ensure_ascii=False)
            temporary.replace(path)

    def render(
        self, cell_dir: Path, completed_episode_ids: Sequence[str]
    ) -> Path | None:
        if self.count == 0:
            return None
        selected: list[dict[str, Any]] = []
        if self.count == 3:
            for label in ("beginning", "middle", "end"):
                for episode_id in sorted(completed_episode_ids):
                    path = self._path(cell_dir, episode_id)
                    if not path.is_file():
                        continue
                    with gzip.open(path, "rt", encoding="utf-8") as stream:
                        candidates = list(json.load(stream))
                    match = next(
                        (
                            item
                            for item in candidates
                            if item.get("sample_point") == label
                        ),
                        None,
                    )
                    if match is not None:
                        selected.append({**match, "episode_id": episode_id})
                        break
        else:
            for episode_id in sorted(completed_episode_ids):
                path = self._path(cell_dir, episode_id)
                if not path.is_file():
                    continue
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    candidates = list(json.load(stream))[: self.count]
                if candidates:
                    selected = [
                        {**item, "episode_id": episode_id} for item in candidates
                    ]
                    break
        if selected:
            sections = [
                "# Prompt examples",
                "",
                "Deterministically selected from completed episode(s): "
                + ", ".join(
                    f"`{episode_id}`"
                    for episode_id in dict.fromkeys(
                        str(item["episode_id"]) for item in selected
                    )
                )
                + ".",
                "",
            ]
            samples = []
            for index, item in enumerate(selected[: self.count], start=1):
                sections.extend(
                    [
                        f"## Example {index} (round {item['round_index']})",
                        "",
                        str(item["markdown"]).rstrip(),
                        "",
                    ]
                )
                samples.append(
                    {key: value for key, value in item.items() if key not in {"rounds"}}
                )
            destination = cell_dir / "prompt_examples.md"
            _write(destination, "\n".join(sections).rstrip() + "\n")
            payload = {
                "schema_version": 1,
                "sample_count": len(samples),
                "samples": samples,
            }
            encoded = (
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            )
            _write(cell_dir / "dashboard_prompt_examples.json", encoded)
            return destination
        return None


def _scientific_identity(
    task: _EpisodeTask, *, run_id: str, price_hash: str
) -> ScientificIdentity:
    options = task.config.game.options
    return ScientificIdentity(
        run_id=run_id,
        cell_id=task.cell_id or "run",
        episode_id=task.episode_id,
        episode_seed=task.seed,
        resolved_config_hash=canonical_hash(task.config.to_dict()),
        prompt_definition_hashes_hash=prompt_definition_hash(task.config),
        pricing_snapshot_hash=price_hash,
        game_type=task.config.game.type,
        dynamics_mode=(
            None
            if options.get("dynamics_mode") is None
            else str(options["dynamics_mode"])
        ),
        control_mechanism=task.config.control.mechanism,
        task_id=None if options.get("task_id") is None else str(options["task_id"]),
    )


def _with_scientific_identity(
    task: _EpisodeTask, *, run_id: str, price_hash: str
) -> _EpisodeTask:
    if not task.config.storage.retention_policy.compact_scientific:
        return task
    return replace(
        task,
        scientific_identity=_scientific_identity(
            task, run_id=run_id, price_hash=price_hash
        ),
    )


def _partition_resume_tasks(
    tasks: Sequence[_EpisodeTask], *, resume: bool, price_hash: str
) -> tuple[list[_EpisodeTask], list[EpisodeOutcome]]:
    """Validate checkpoints and return only episodes that need provider work."""

    scheduled: list[_EpisodeTask] = []
    resumed: list[EpisodeOutcome] = []
    sealed_frames: dict[Path, Any] = {}
    for task in tasks:
        if not resume or task.config.storage.checkpoint_mode != "episode":
            scheduled.append(task)
            continue
        if task.scientific_identity is not None:
            cell_dir = task.cell_dir or task.episode_dir
            shard = episode_shard_path(cell_dir, task.episode_id)
            final = cell_dir / "scientific_events.parquet"
            if shard.is_file():
                frame = validate_episode_artifact(shard, task.scientific_identity)
            elif final.is_file():
                if cell_dir not in sealed_frames:
                    sealed_frames[cell_dir] = validate_cell_artifact(cell_dir)
                frame = validate_episode_frame(
                    sealed_frames[cell_dir], task.scientific_identity, source=final
                )
            else:
                scheduled.append(task)
                continue
            if task.config.storage.retention_policy.semantic_dashboard:
                validate_semantic_stream(
                    (task.cell_dir or task.episode_dir)
                    / "round_records"
                    / task.episode_id
                )
            resumed.append(
                EpisodeOutcome(
                    task.episode_id,
                    task.seed,
                    "skipped_resumed",
                    interactions=len(frame),
                    termination_reason=str(frame.iloc[-1]["termination_reason"]),
                    cell_id=task.cell_id,
                )
            )
            continue
        manifest_path = task.manifest_path
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "completed":
                expected = {
                    "episode_id": task.episode_id,
                    "cell_id": task.cell_id,
                    "seed": task.seed,
                    "resolved_config_hash": canonical_hash(task.config.to_dict()),
                    "prompt_definition_hashes_hash": prompt_definition_hash(
                        task.config
                    ),
                    "pricing_snapshot_hash": price_hash,
                    "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION,
                }
                for field, value in expected.items():
                    if field in manifest and manifest[field] != value:
                        raise ValueError(
                            f"incompatible episode checkpoint {task.episode_id}: "
                            f"{field} does not match"
                        )
                resumed.append(
                    replace(
                        EpisodeOutcome.from_dict(manifest), status="skipped_resumed"
                    )
                )
                continue
        scheduled.append(task)
    return scheduled, resumed


class _ResultsOnlyFinalizer:
    """Seal compact cells before aggregate/analysis consumers are invoked."""

    def __init__(
        self,
        tasks: Sequence[_EpisodeTask],
        prompt_sampler: _CellPromptSampler | None,
    ) -> None:
        self._tasks: dict[str, list[_EpisodeTask]] = {}
        for task in tasks:
            self._tasks.setdefault(task.cell_id or "run", []).append(task)
        self._sampler = prompt_sampler
        self._locks = {cell_id: threading.Lock() for cell_id in self._tasks}

    def seal(self, cell_id: str | None) -> dict[str, Any] | None:
        label = cell_id or "run"
        tasks = self._tasks.get(label, [])
        if not tasks or tasks[0].scientific_identity is None:
            return None
        cell_dir = tasks[0].cell_dir or tasks[0].episode_dir
        with self._locks[label]:
            final = cell_dir / "scientific_events.parquet"
            seal = cell_dir / "cell_complete.json"
            if final.is_file() and seal.is_file():
                frame = validate_cell_artifact(cell_dir)
                for task in tasks:
                    assert task.scientific_identity is not None
                    validate_episode_frame(
                        frame, task.scientific_identity, source=final
                    )
                return json.loads(seal.read_text(encoding="utf-8"))
            identities: list[ScientificIdentity] = []
            for task in tasks:
                assert task.scientific_identity is not None
                artifact = discover_episode_artifact(cell_dir, task.scientific_identity)
                if artifact is not None:
                    validate_episode_artifact(artifact, task.scientific_identity)
                    identities.append(task.scientific_identity)
                    continue
                manifest = task.manifest_path
                if manifest.is_file():
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    if payload.get("status") == "completed":
                        raise ValueError(
                            f"episode {task.episode_id} is marked completed without a valid shard"
                        )
            if not identities or len(identities) != len(tasks):
                # Successful shards remain durable and resumable, but a cell
                # with failed/missing episodes is not falsely sealed whole.
                return None
            if self._sampler is not None:
                self._sampler.render(cell_dir, [item.episode_id for item in identities])
            summary = merge_episode_artifacts(cell_dir, identities, remove_shards=True)
            prompt_artifact = cell_dir / "dashboard_prompt_examples.json"
            if prompt_artifact.is_file():
                summary = {
                    **summary,
                    "dashboard_prompt_examples": {
                        "path": prompt_artifact.name,
                        "sha256": file_sha256(prompt_artifact),
                        "schema_version": 1,
                    },
                }
                _write(
                    cell_dir / "cell_complete.json",
                    json.dumps(summary, sort_keys=True, indent=2) + "\n",
                )
            episodes_dir = cell_dir / "data" / "episodes"
            if episodes_dir.is_dir():
                shutil.rmtree(episodes_dir)
                data_dir = episodes_dir.parent
                if data_dir.is_dir() and not any(data_dir.iterdir()):
                    data_dir.rmdir()
            return summary


class _PerCellAnalysisReporter:
    """Serialize local configured analysis as grid cells become complete.

    Bootstrap/null analysis is CPU-heavy and cell completions can race under
    the shared episode concurrency pool.  One lock keeps those analyses from
    multiplying CPU and memory pressure while provider work in other cells is
    still free to continue.
    """

    def __init__(
        self, cells: Sequence[Any], cells_dir: Path, comet_sink: Any | None = None
    ) -> None:
        self._configs = {cell.cell_id: cell.config for cell in cells}
        self._cells_dir = cells_dir
        self._comet_sink = comet_sink
        self._lock = threading.Lock()

    def write(self, cell_id: str) -> dict[str, Any] | None:
        with self._lock:
            summary = run_configured_cell_analysis(
                self._configs[cell_id],
                self._cells_dir / cell_id,
                cell_id,
                comet_sink=self._comet_sink,
            )
            if summary is not None:
                LOGGER.info(
                    "cell %s configured analysis ready: %s",
                    cell_id,
                    ", ".join(summary.get("cell_reports", ())),
                )
            return summary


def _observer_runtime(game: Game, episode_config: RunConfig, guarded_provider: Any):
    """This game's observer-aware runtime, or ``None`` if it has none.

    Only a runtime that reports each round to an observer can drive the
    recorder's streaming metrics, and `metrics/streaming.csv` is precisely what
    `mas-cc analysis empowerment` reads back. A game without one still runs -
    through the bare `run_game` path - but its episodes carry no per-round
    metrics, so no grid over it can be analysed afterwards.

    Synthetic games are matched by type rather than by name: they live under
    `games/synthetic/` and are not named after their directory, so the
    `game.type` string that identifies the convention game cannot identify them.
    The import is deferred for the reason given in `_provider_registry`.
    """

    if episode_config.game.type == "naming_convention":
        control = create_control(episode_config.control)
        return lambda observer: run_naming_convention_game(
            game, episode_config, guarded_provider, observer=observer, control=control
        )

    if episode_config.game.type in {"hidden_bench_vanilla", "hidden_bench_naming"}:
        # Both HiddenBench games share one runtime; it dispatches on the game's
        # own phase, not on its type. `control` is not wired: the message-level
        # interventions HiddenBench wants (reveal-all, secretary, structured
        # exchange) inject a *message*, and `Control.override` returns an
        # *action* - see docs/hidden_bench/README.md for the open decision.
        from mas_cc.games.hidden_bench import run_hidden_bench_game

        return lambda observer: run_hidden_bench_game(
            game, episode_config, guarded_provider, observer=observer
        )

    if episode_config.game.type == "hidden_bench_imitation":
        from mas_cc.games.hidden_bench.imitation import run_hidden_bench_imitation_game

        control = create_control(episode_config.control)
        return lambda observer: run_hidden_bench_imitation_game(
            game,
            episode_config,
            guarded_provider,
            observer=observer,
            control=control,
        )

    if episode_config.game.type == "hidden_bench_imitation_round_feedback":
        from mas_cc.games.hidden_bench.imitation_round_feedback import (
            run_hidden_bench_imitation_round_feedback_game,
        )

        control = create_control(episode_config.control)
        return lambda observer: run_hidden_bench_imitation_round_feedback_game(
            game,
            episode_config,
            guarded_provider,
            observer=observer,
            control=control,
        )

    if episode_config.game.type == "relational_imitation_round_feedback":
        from mas_cc.games.relational_reasoning.imitation_round_feedback import (
            run_relational_imitation_round_feedback_game,
        )

        control = create_control(episode_config.control)
        return lambda observer: run_relational_imitation_round_feedback_game(
            game,
            episode_config,
            guarded_provider,
            observer=observer,
            control=control,
        )

    from mas_cc.games.synthetic import SyntheticGame, run_synthetic_game

    if isinstance(game, SyntheticGame):
        return lambda observer: run_synthetic_game(
            game, episode_config, guarded_provider, observer=observer
        )
    return None


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
    retention_policy: Any,
    scientific_identity: ScientificIdentity | None,
    scientific_path: Path | None,
    prompt_sampler: _CellPromptSampler | None,
    prompt_cell_dir: Path,
    prompt_episode_id: str,
) -> tuple[int | None, str | None]:
    """Run one episode; return ``(interaction_count, termination_reason)``."""

    runtime = _observer_runtime(game, episode_config, guarded_provider)
    if runtime is not None:
        recorder_dir = (
            scientific_path.parent
            if retention_policy.compact_scientific and scientific_path is not None
            else episode_dir
        )
        recorder = RunRecorder(
            recorder_dir,
            run_id=episode_label,
            resolved_config=episode_config.to_dict(),
            policy=policy,
            # Per-episode Comet experiments are not wired here: one mas_cc
            # experiment would otherwise fan out into N remote experiments.
            comet_enabled=False,
            checkpoint_enabled=checkpoint_enabled,
            price_snapshot_hash=price_hash,
            metrics=metrics,
            to_round_view=to_round_view,
            binning=episode_config.metrics.binning_policy(
                episode_config.game.population_size
            ),
            retention_policy=retention_policy,
            scientific_identity=scientific_identity,
            scientific_path=scientific_path,
        )
        prompt_example_rounds = int(
            dict(episode_config.logging.options.get("prompt_examples", {}) or {}).get(
                "count", 0
            )
        )
        prompt_scope = str(
            dict(episode_config.logging.options.get("prompt_examples", {}) or {}).get(
                "scope",
                "cell"
                if episode_config.storage.retention_policy.compact_scientific
                else "episode",
            )
        )
        prompt_logger = (
            PromptMarkdownLogger(episode_dir / "prompts", overwrite=True)
            if prompt_example_rounds > 0 and prompt_scope == "episode"
            else None
        )
        observer = _RoundTickingObserver(
            recorder,
            guard,
            progress,
            episode_label,
            prompt_logger=prompt_logger,
            prompt_example_rounds=prompt_example_rounds,
            prompt_sampler=prompt_sampler if prompt_scope == "cell" else None,
            prompt_cell_dir=prompt_cell_dir,
            prompt_episode_id=prompt_episode_id,
            prompt_context={
                "cell_id": (
                    scientific_identity.cell_id
                    if scientific_identity is not None
                    else prompt_cell_dir.name
                ),
                "source_config": episode_config.experiment.name,
                "rounds": int(
                    episode_config.game.options.get(
                        "rounds", episode_config.game.horizon
                    )
                ),
                "condition": episode_config.experiment.metadata.get("arm"),
                "controller_role": episode_config.control.mechanism,
                "game_parameters": {
                    "population_size": episode_config.game.population_size,
                    "social_group_size": episode_config.game.options.get(
                        "social_group_size"
                    ),
                    "epistemic_persistence": episode_config.game.options.get(
                        "epistemic_persistence"
                    ),
                    "vote_visibility": episode_config.game.options.get(
                        "vote_visibility"
                    ),
                    "board": episode_config.game.options.get("board"),
                },
                "prompt_schema_version": episode_config.prompt.schema_version,
                "prompt_template_version": episode_config.prompt.prompt_version,
                "provider": episode_config.llm_provider.type,
                "model": episode_config.llm_provider.model,
            },
        )
        try:
            result = await runtime(observer)
        except Exception as exc:
            recorder.event(
                "run_failed",
                error_type=type(exc).__name__,
                **({} if retention_policy.semantic_dashboard else {"error": str(exc)}),
            )
            recorder.finalize(
                status="failed", budget_status=guard.checkpoint_state(), error=exc
            )
            raise
        recorder.finalize(
            status="completed",
            budget_status=guard.checkpoint_state(),
            termination_reason=result.termination_reason,
        )
        return int(
            getattr(result, "logical_decisions", len(result.interactions))
        ), result.termination_reason

    result = await run_game(game, episode_config, guarded_provider)
    progress.round_tick(episode_label, None, count=len(result.interactions))
    if retention_policy.compact_scientific:
        recorder_dir = (
            scientific_path.parent if scientific_path is not None else episode_dir
        )
        recorder = RunRecorder(
            recorder_dir,
            run_id=episode_label,
            resolved_config=episode_config.to_dict(),
            policy=policy,
            comet_enabled=False,
            checkpoint_enabled=False,
            price_snapshot_hash=price_hash,
            metrics=metrics,
            to_round_view=to_round_view,
            binning=episode_config.metrics.binning_policy(
                episode_config.game.population_size
            ),
            retention_policy=retention_policy,
            scientific_identity=scientific_identity,
            scientific_path=scientific_path,
        )
        for index, interaction in enumerate(result.interactions, start=1):
            recorder.record_interaction(
                round_index=index,
                interaction=interaction,
                budget_status=guard.checkpoint_state(),
                state=interaction.transition.next_state.to_dict(),
                prompt_definitions={},
            )
        recorder.finalize(
            status="completed",
            budget_status=guard.checkpoint_state(),
            termination_reason=result.termination_reason,
        )
    else:
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
    budget_abort: asyncio.Event,
    fail_fast: bool,
    resume: bool,
    policy: DetailedAuditPolicy,
    metrics: tuple[Any, ...],
    to_round_view: Callable[[Any], Any] | None,
    price_hash: str,
    checkpoint_enabled: bool,
    progress: ExperimentProgress,
    monitor: MasterMonitor | None = None,
    aggregator: GridAggregator | None = None,
    completion: "_CellCompletion | None" = None,
    prompt_sampler: _CellPromptSampler | None = None,
    finalizer: _ResultsOnlyFinalizer | None = None,
    cell_analysis: _PerCellAnalysisReporter | None = None,
) -> EpisodeOutcome:
    manifest_path = task.manifest_path
    label = (
        task.episode_id if task.cell_id is None else f"{task.cell_id}/{task.episode_id}"
    )

    def _finished(outcome: EpisodeOutcome) -> None:
        """Local console bar and remote master monitor, always together."""
        progress.episode_done(label, outcome.status)
        if monitor is not None:
            monitor.episode_finished(
                status=outcome.status,
                cell_id=task.cell_id,
                budget_status=guard.status(),
            )

    def _persist(outcome: EpisodeOutcome) -> None:
        manifest = {
            **outcome.to_dict(),
            "resolved_config_hash": canonical_hash(task.config.to_dict()),
            "prompt_definition_hashes_hash": prompt_definition_hash(task.config),
            "pricing_snapshot_hash": price_hash,
            "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION,
        }
        _write(manifest_path, _json(manifest))

    async def _maybe_close_cell() -> None:
        """Aggregate and publish this cell if that episode was its last.

        Off the event loop: aggregating a cell reads every episode's metric
        files, and blocking the loop for that would stall the episodes still
        running in the other cells.
        """

        if aggregator is None or completion is None or task.cell_id is None:
            return
        if not completion.record(task.cell_id):
            return
        try:
            if finalizer is not None:
                await asyncio.to_thread(finalizer.seal, task.cell_id)
            await asyncio.to_thread(aggregator.aggregate, task.cell_id)
            if cell_analysis is not None:
                await asyncio.to_thread(cell_analysis.write, task.cell_id)
            if finalizer is not None:
                await asyncio.to_thread(
                    _record_cell_hashes, task.cell_dir or task.episode_dir
                )
        except (
            Exception
        ) as exc:  # aggregates are derived; a failure must not lose episodes
            LOGGER.error(
                "aggregating cell %s failed: %s: %s",
                task.cell_id,
                type(exc).__name__,
                exc,
            )

    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed":
            outcome = replace(
                EpisodeOutcome.from_dict(manifest), status="skipped_resumed"
            )
            _finished(outcome)
            await _maybe_close_cell()
            return outcome
    episode_queued_at = _now()
    episode_queued_clock = time.perf_counter()
    async with semaphore:
        episode_started_at = _now()
        episode_started_clock = time.perf_counter()
        queue_wait_seconds = round(episode_started_clock - episode_queued_clock, 6)

        def _timed(outcome: EpisodeOutcome) -> EpisodeOutcome:
            if not task.config.storage.retention_policy.compact_scientific:
                return outcome
            return replace(
                outcome,
                started_at=episode_started_at,
                finished_at=_now(),
                duration_seconds=round(time.perf_counter() - episode_started_clock, 6),
                queued_at=episode_queued_at,
                queue_wait_seconds=queue_wait_seconds,
            )

        # A budget stop aborts regardless of `fail_fast`. Once the guard is
        # exhausted or stopped its counters only move one way, so every queued
        # episode would dispatch, be refused on its first call, and be recorded
        # as a failure. That is how one exhausted token budget turned into
        # 4,235 "failed" episodes (results/DIAGNOSIS.md) and buried the fact
        # that the run had simply run out of money. Skipping is the honest
        # record: these episodes never ran.
        if budget_abort.is_set():
            outcome = EpisodeOutcome(
                task.episode_id,
                task.seed,
                "skipped_aborted",
                cell_id=task.cell_id,
                error_type="BudgetStop",
                error=guard.stop_reason or "runtime budget exhausted",
            )
            outcome = _timed(outcome)
            _persist(outcome)
            _finished(outcome)
            await _maybe_close_cell()
            return outcome
        if fail_fast and abort.is_set():
            outcome = EpisodeOutcome(
                task.episode_id, task.seed, "skipped_aborted", cell_id=task.cell_id
            )
            outcome = _timed(outcome)
            _persist(outcome)
            _finished(outcome)
            await _maybe_close_cell()
            return outcome
        timing_token = _TIMING_EPISODE.set((task.cell_id, task.episode_id))
        try:
            interactions, termination_reason = await _execute_episode(
                game,
                task.config,
                guarded_provider,
                guard,
                task.episode_dir,
                label,
                policy=policy,
                metrics=metrics,
                to_round_view=to_round_view,
                price_hash=price_hash,
                checkpoint_enabled=checkpoint_enabled,
                progress=progress,
                retention_policy=task.config.storage.retention_policy,
                scientific_identity=task.scientific_identity,
                scientific_path=task.scientific_path,
                prompt_sampler=prompt_sampler,
                prompt_cell_dir=task.cell_dir or task.episode_dir,
                prompt_episode_id=task.episode_id,
            )
            outcome = EpisodeOutcome(
                task.episode_id,
                task.seed,
                "completed",
                interactions=interactions,
                termination_reason=termination_reason,
                cell_id=task.cell_id,
            )
        except Exception as exc:
            if fail_fast:
                abort.set()
            if isinstance(exc, ProviderError) and exc.code in BUDGET_STOP_CODES:
                if not budget_abort.is_set():
                    LOGGER.error(
                        "budget stop at episode %s: %s - no further episodes will be started",
                        label,
                        exc,
                    )
                budget_abort.set()
            outcome = EpisodeOutcome(
                task.episode_id,
                task.seed,
                "failed",
                error_type=type(exc).__name__,
                error=(
                    None
                    if task.config.storage.retention_policy.semantic_dashboard
                    else str(exc)
                ),
                cell_id=task.cell_id,
            )
            if task.config.storage.retention_policy.semantic_dashboard:
                LOGGER.error("episode %s failed: %s", label, type(exc).__name__)
            else:
                LOGGER.error(
                    "episode %s failed: %s: %s", label, type(exc).__name__, exc
                )
        finally:
            _TIMING_EPISODE.reset(timing_token)
        outcome = _timed(outcome)
        _persist(outcome)
        _finished(outcome)
    # Outside the semaphore: aggregating a completed cell must not hold a slot
    # that a queued episode of another cell could be running in.
    await _maybe_close_cell()
    return outcome


async def _run_task_batch(
    tasks: list[_EpisodeTask], **kwargs: Any
) -> tuple[EpisodeOutcome, ...]:
    return tuple(
        await asyncio.gather(*(_run_episode_task(task, **kwargs) for task in tasks))
    )


async def _prime_resumed_outcomes(
    outcomes: Sequence[EpisodeOutcome],
    *,
    progress: ExperimentProgress,
    monitor: MasterMonitor,
    guard: RuntimeBudgetGuard,
    completion: _CellCompletion | None = None,
    aggregator: GridAggregator | None = None,
    finalizer: _ResultsOnlyFinalizer | None = None,
    cell_analysis: _PerCellAnalysisReporter | None = None,
) -> None:
    """Reflect validated checkpoints in progress and cell-completion state."""

    for outcome in outcomes:
        label = (
            outcome.episode_id
            if outcome.cell_id is None
            else f"{outcome.cell_id}/{outcome.episode_id}"
        )
        progress.episode_done(label, outcome.status)
        monitor.episode_finished(
            status=outcome.status,
            cell_id=outcome.cell_id,
            budget_status=guard.status(),
        )
        if completion is None or outcome.cell_id is None:
            continue
        if not completion.record(outcome.cell_id):
            continue
        if finalizer is not None:
            await asyncio.to_thread(finalizer.seal, outcome.cell_id)
        if aggregator is not None:
            await asyncio.to_thread(aggregator.aggregate, outcome.cell_id)
            if cell_analysis is not None:
                await asyncio.to_thread(cell_analysis.write, outcome.cell_id)
            if finalizer is not None:
                await asyncio.to_thread(
                    _record_cell_hashes, aggregator.cell_directory(outcome.cell_id)
                )


async def _with_spend_watch(
    watcher: "_LiveSpendWatcher | None", episodes: Any
) -> tuple[EpisodeOutcome, ...]:
    """Run the episodes with the spend poller alongside them.

    The poller is a side channel, never a participant: it is cancelled the
    moment the episodes finish, and it cannot fail the run. Its only effect on
    the run is through `guard.request_stop`.
    """

    if watcher is None:
        return await episodes
    await watcher.start()
    poller = asyncio.create_task(watcher.run())
    try:
        return await episodes
    finally:
        poller.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await poller


def _aggregate_quietly(aggregator: GridAggregator, cell_id: str | None) -> None:
    """Aggregate one cell, logging rather than raising on failure.

    Called from the run's `finally` block, where the episodes are already
    safely on disk: a bug in a derived statistic must not turn a finished run
    into a failed one, and the same aggregation can be re-run from the
    directory afterwards.
    """

    try:
        aggregator.aggregate(cell_id)
    except Exception as exc:
        LOGGER.error("final aggregation failed: %s: %s", type(exc).__name__, exc)


def _write_experiment_summary(
    run_dir: Path, config: RunConfig, result: ExperimentResult
) -> None:
    compact = config.storage.retention_policy.compact_scientific
    if not compact:
        _write(run_dir / "experiment_summary.json", _json(result.to_dict()))
    fieldnames = [
        "episode_id",
        "seed",
        "status",
        "interactions",
        "termination_reason",
        "error_type",
        "error",
    ]
    if compact:
        fieldnames.extend(
            [
                "queued_at",
                "started_at",
                "finished_at",
                "queue_wait_seconds",
                "duration_seconds",
            ]
        )
    output_path = run_dir / ("run_summary.csv" if compact else "experiment_summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in outcome.to_dict().items() if key != "cell_id"}
            for outcome in result.outcomes
        )
    manifest = {
        "schema_version": 1,
        "run_id": result.run_id,
        "experiment_name": result.experiment_name,
        "game_type": result.game_type,
        "episode_count": result.episode_count,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "artifact_profile": config.storage.artifact_profile,
        "checkpoint_mode": config.storage.checkpoint_mode,
        "completed": result.completed + result.skipped_resumed,
        "failed": result.failed,
        "skipped_aborted": result.skipped_aborted,
        "budget_summary": dict(result.budget_status),
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
    validate_configured_analysis(config)
    override = (
        config.pricing.explicit_unknown_price_override
        if explicit_override is None
        else explicit_override
    )

    game = create_game(config.game)
    if config.game.type == "relational_imitation_round_feedback":
        controller = create_control(config.control)
        validator = getattr(controller, "validate_truthful_report_task", None)
        if validator is not None:
            validator(game.load_task(config.game), config.execution.seed)
    plan = game.call_plan(config.game)
    quote = _quote(config)
    system_budget, run_budget = _budgets(config, quote)

    preflight = static_experiment_preflight(
        plan,
        config.prompt,
        config.llm_provider,
        episode_count=config.execution.repetitions,
        concurrency=config.execution.parallelism,
        assumed_output_tokens=config.llm_provider.max_output_tokens,
        pricing_quote=quote,
        system_budget=system_budget,
        run_budget=run_budget,
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
        raise ValueError(
            "live pricing changed during immediate pre-launch revalidation"
        )

    run_id = f"{config.experiment.name}-{config.execution.seed}"
    run_dir = results_run_dir(
        output_dir,
        game=config.game.type,
        experiment=config.experiment.name,
        run_id=run_id,
    )
    _wipe_run_dir_if_requested(
        run_dir, wipe_and_recompute=config.storage.wipe_and_recompute, label=run_id
    )
    episodes_dir = run_dir / "data" / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    budget_description = (
        "unbounded"
        if run_budget.max_cost is None and system_budget.max_cost is None
        else (
            format_money(run_budget.max_cost)
            if run_budget.max_cost is not None
            else format_money(system_budget.max_cost)
        )
    )
    scenarios = preflight.per_episode.prompt_scenarios
    definition_hash = (
        scenarios[0]["representative"]["definition_hash"] if scenarios else ""
    )
    print_banner(
        format_banner(
            experiment_name=config.experiment.name,
            game_type=config.game.type,
            game_version=game.spec.version,
            provider=config.llm_provider.type,
            model=config.llm_provider.model,
            episode_count=config.execution.repetitions,
            concurrency=config.execution.parallelism,
            prompt_family=config.prompt.prompt_family,
            prompt_version=config.prompt.prompt_version,
            prompt_definition_hash=definition_hash,
            budget_description=budget_description,
            preflight_expected_cost=format_money(preflight.total_costs.expected),
            preflight_conservative_cost=format_money(
                preflight.total_costs.conservative
            ),
            preflight_status=preflight.launch_status,
        )
    )
    # Printed unconditionally and up front: this is the one line that answers
    # "where do I look for the results" without waiting for the run to finish
    # or grepping logs.
    print_banner(f"  Results:       {run_dir}")

    effective_budget = resolve_budget_limits(system_budget, run_budget)
    guard = RuntimeBudgetGuard(effective_budget, expectation=_expectation(preflight))
    price_hash = _pricing_identity(runtime_quote)
    _configure_durable_budget(
        config, run_dir, guard, price_hash=price_hash, resume=resume
    )

    root_seed = Seed(config.execution.seed)

    def _episode_task(index: int) -> _EpisodeTask:
        episode_id = f"{run_id}-{index:04d}"
        seed = int(root_seed.derive(f"episode:{index}"))
        return _EpisodeTask(
            episode_id=episode_id,
            seed=seed,
            config=replace(config, execution=replace(config.execution, seed=seed)),
            episode_dir=episodes_dir / episode_id,
            cell_dir=run_dir,
        )

    tasks = [
        _with_scientific_identity(
            _episode_task(index), run_id=run_id, price_hash=price_hash
        )
        for index in range(config.execution.repetitions)
    ]
    scheduled_tasks, resumed_outcomes = _partition_resume_tasks(
        tasks, resume=resume, price_hash=price_hash
    )
    guarded_provider = None
    timing_provider = None
    watcher = None
    if scheduled_tasks:
        provider = create_llm_provider(
            config.llm_provider, registry=_provider_registry()
        )
        guarded_provider = BudgetGuardedProvider(
            provider,
            guard,
            runtime_quote.pricing,
            input_token_estimator=estimate_input_tokens,
            input_token_multiplier=1.0,
        )
        if config.storage.retention_policy.detailed_timing:
            timing_provider = _TimingProvider(guarded_provider)
            guarded_provider = timing_provider
        watcher = _spend_watcher(config, guard, effective_budget)
    metrics, to_round_view = game_metrics(game)
    policy = DetailedAuditPolicy.from_mapping(
        config.logging.options.get("detailed_prompt_audit")
    )
    prompt_options = dict(config.logging.options.get("prompt_examples", {}) or {})
    prompt_sampler = (
        _CellPromptSampler(int(prompt_options.get("count", 0)))
        if prompt_options.get(
            "scope",
            "cell" if config.storage.retention_policy.compact_scientific else "episode",
        )
        == "cell"
        else None
    )
    finalizer = (
        _ResultsOnlyFinalizer(tasks, prompt_sampler)
        if config.storage.retention_policy.compact_scientific
        else None
    )

    progress = ExperimentProgress(
        total_episodes=config.execution.repetitions,
        total_rounds=_steps_per_episode(plan) * config.execution.repetitions,
        show=show_progress,
    )
    monitor = _master_monitor(config, run_id, config.execution.repetitions)
    monitor.start(sweep_parameters(config))
    # Printed after `start`, so it reports whether Comet actually connected
    # rather than what the config asked for. Those differ exactly when it
    # matters - a missing API key - and that case used to be silent.
    print_banner(f"  Comet:         {monitor.describe()}")
    # N episodes of one resolved config *are* one grid cell, so they get the
    # same aggregation rather than a second implementation of it - just at the
    # end, since there is no earlier completion event to hang it on.
    aggregator = GridAggregator(
        run_dir, config.aggregation, monitor=monitor, seed=config.execution.seed
    )
    semaphore = asyncio.Semaphore(config.execution.parallelism)
    abort = asyncio.Event()
    budget_abort = asyncio.Event()
    started_at = _now()
    outcomes: tuple[EpisodeOutcome, ...] = tuple(resumed_outcomes)

    # The master experiment is closed in the outer `finally`, *after* the
    # post-run analysis rather than before it. Under
    # `observability.comet.cell_reporting: master` the analysis publishes onto that
    # same experiment, and a sink closed first would silently swallow it.
    analysis_summary = None
    try:
        try:
            await _prime_resumed_outcomes(
                resumed_outcomes,
                progress=progress,
                monitor=monitor,
                guard=guard,
            )
            fresh_outcomes = await _with_spend_watch(
                watcher,
                _run_task_batch(
                    scheduled_tasks,
                    game=game,
                    guarded_provider=guarded_provider,
                    guard=guard,
                    semaphore=semaphore,
                    abort=abort,
                    budget_abort=budget_abort,
                    fail_fast=config.execution.fail_fast,
                    resume=resume,
                    policy=policy,
                    metrics=metrics,
                    to_round_view=to_round_view,
                    price_hash=price_hash,
                    checkpoint_enabled=config.storage.checkpoints,
                    progress=progress,
                    monitor=monitor,
                    prompt_sampler=prompt_sampler,
                    finalizer=finalizer,
                ),
            )
            by_id = {
                outcome.episode_id: outcome
                for outcome in (*resumed_outcomes, *fresh_outcomes)
            }
            outcomes = tuple(by_id[task.episode_id] for task in tasks)
        finally:
            if guarded_provider is not None:
                guarded_provider.close()
            progress.close()
            if finalizer is not None:
                finalizer.seal(None)
            elif prompt_sampler is not None:
                prompt_sampler.render(
                    run_dir,
                    [
                        outcome.episode_id
                        for outcome in outcomes
                        if outcome.status in {"completed", "skipped_resumed"}
                    ],
                )
            _aggregate_quietly(aggregator, None)

        result = ExperimentResult(
            run_id=run_id,
            experiment_name=config.experiment.name,
            game_type=config.game.type,
            episode_count=config.execution.repetitions,
            output_dir=run_dir,
            outcomes=outcomes,
            preflight=preflight,
            budget_status=guard.status(),
            started_at=started_at,
            finished_at=_now(),
        )
        _write_experiment_summary(run_dir, config, result)
        # Analysis needs at least one completed trajectory. If every episode
        # failed before its first step, preserve the episode/provider failures
        # in the returned result instead of masking them with a secondary
        # missing-file exception.
        if result.completed or result.skipped_resumed:
            analysis_summary = run_configured_analysis(
                config, run_dir, monitor.analysis_sink
            )
    finally:
        comet_summary = monitor.close()
        _write(
            run_dir
            / (
                "comet_summary.json"
                if config.storage.retention_policy.compact_scientific
                else "comet_run_summary.json"
            ),
            _json(comet_summary),
        )

    if config.storage.retention_policy.compact_scientific:
        _write_timing_study(
            run_dir,
            outcomes,
            profile=config.storage.artifact_profile,
            run_started_at=result.started_at,
            run_finished_at=result.finished_at,
            request_rows=() if timing_provider is None else timing_provider.rows,
        )
        _record_results_only_hashes(run_dir)
    _print_comet_destinations(comet_summary, analysis_summary)
    return result


def _print_comet_destinations(
    comet_summary: Mapping[str, Any] | None, analysis: Mapping[str, Any] | None
) -> None:
    """Say where everything was published, not only where the master lives.

    One run writes to up to three Comet experiments - the master, one per
    completed cell, and one for the post-run information analysis - and the
    banner used to name only the first. Aggregate plots and MI estimates went
    to the other two, so a user watching the printed link saw none of them and
    reasonably concluded nothing had been uploaded.

    Under ``cell_reporting: master`` there is only the master, so the cell and
    analysis lines report what landed on it rather than repeating its URL
    three times.
    """

    lines: list[str] = []
    if (comet_summary or {}).get("url"):
        lines.append(f"  Comet master:   {comet_summary['url']}")
    for cell in (comet_summary or {}).get("cell_experiments", ()) or ():
        if not isinstance(cell, Mapping):
            continue
        if cell.get("published_to") == "master":
            lines.append(
                f"  Comet cell {cell.get('cell_id')}: on master  "
                f"({cell.get('curves', 0)} curves, {cell.get('metric_plots', 0)} plot(s))"
            )
        elif cell.get("url"):
            lines.append(f"  Comet cell {cell.get('cell_id')}: {cell['url']}")
    if analysis and isinstance(analysis.get("comet"), Mapping):
        published = analysis["comet"]
        detail = f"({published.get('metrics', 0)} metrics, {published.get('images', 0)} image(s))"
        if published.get("published_to") == "master":
            lines.append(f"  Comet analysis: on master  {detail}")
        elif published.get("url"):
            lines.append(f"  Comet analysis: {published['url']}  {detail}")
    for line in lines:
        print_banner(line)


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
        return sum(
            1 for outcome in self.outcomes if outcome.status == "skipped_resumed"
        )

    @property
    def skipped_aborted(self) -> int:
        return sum(
            1 for outcome in self.outcomes if outcome.status == "skipped_aborted"
        )

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


def _grid_cell_seed(
    grid_seed: Seed, cell_index: int, *, common_random_numbers: bool
) -> Seed:
    """Cell stream, optionally shared for explicitly matched grid designs."""

    return (
        grid_seed
        if common_random_numbers
        else grid_seed.derive(f"grid-cell:{cell_index}")
    )


def _write_grid_summary(grid_dir: Path, grid: GridSpec, result: GridResult) -> None:
    compact = grid.base.storage.retention_policy.compact_scientific
    if not compact:
        _write(grid_dir / "grid_summary.json", _json(result.to_dict()))
    fieldnames = [
        "cell_id",
        "completed",
        "failed",
        "skipped_resumed",
        "skipped_aborted",
        "overrides",
    ]
    output_path = grid_dir / "grid_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for cell in result.cells:
            writer.writerow(
                {
                    "cell_id": cell.cell_id,
                    "completed": cell.completed,
                    "failed": cell.failed,
                    "skipped_resumed": cell.skipped_resumed,
                    "skipped_aborted": cell.skipped_aborted,
                    "overrides": json.dumps(
                        dict(cell.overrides), sort_keys=True, default=str
                    ),
                }
            )
    manifest = {
        "schema_version": 1,
        "grid_id": result.grid_id,
        "run_id": result.run_id,
        "experiment_name": result.experiment_name,
        "game_type": result.game_type,
        "cell_count": len(result.cells),
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "artifact_profile": grid.base.storage.artifact_profile,
        "checkpoint_mode": grid.base.storage.checkpoint_mode,
        "completed": result.completed + result.skipped_resumed,
        "failed": result.failed,
        "skipped_aborted": result.skipped_aborted,
        "budget_summary": dict(result.budget_status),
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
    episode_plan: Mapping[str, Mapping[int, int]] | None = None,
) -> GridResult:
    """Run every cell's episodes under one combined concurrency and budget pool.

    Every cell shares one provider client, one pricing quote, and one
    :class:`RuntimeBudgetGuard` — `GridSpec` already forbids sweeping the
    fields (provider identity, budget, pricing) that would make that unsafe.
    ``execution.fail_fast``/``execution.parallelism``/``execution.seed`` are
    read from the *base* config and apply to the whole grid, not per cell.
    """

    cells = grid.cells
    cells_by_id = {cell.cell_id: cell for cell in cells}
    if episode_plan is not None:
        unknown_cells = sorted(set(episode_plan) - set(cells_by_id))
        if unknown_cells:
            raise ValueError(
                "episode plan contains unknown cell(s): " + ", ".join(unknown_cells)
            )
        normalized_episode_plan: dict[str, dict[int, int]] = {}
        for cell_id, planned in episode_plan.items():
            normalized: dict[int, int] = {}
            for raw_index, raw_seed in planned.items():
                index = int(raw_index)
                seed = int(raw_seed)
                if (
                    index < 0
                    or index >= cells_by_id[cell_id].config.execution.repetitions
                ):
                    raise ValueError(
                        f"episode plan repetition {index} is outside the target range "
                        f"for {cell_id}"
                    )
                normalized[index] = seed
            if not normalized:
                raise ValueError(f"episode plan for {cell_id} is empty")
            normalized_episode_plan[cell_id] = normalized
        cells = tuple(cell for cell in cells if cell.cell_id in normalized_episode_plan)
    else:
        normalized_episode_plan = {}
    validate_configured_analysis(grid.base)
    for cell in cells:
        if cell.config.execution.repetitions < 1:
            raise ValueError("every cell's execution.repetitions must be at least 1")

    base = grid.base
    override = (
        base.pricing.explicit_unknown_price_override
        if explicit_override is None
        else explicit_override
    )
    game = create_game(base.game)
    quote = _quote(base)
    system_budget, run_budget = _budgets(base, quote)

    preflight = static_grid_preflight(
        grid,
        concurrency=base.execution.parallelism,
        assumed_output_tokens=base.llm_provider.max_output_tokens,
        pricing_quote=quote,
        system_budget=system_budget,
        run_budget=run_budget,
        explicit_override=override,
        allow_stale_pricing=not base.pricing.require_fresh_at_launch,
    )
    if preflight.launch_status != "permitted":
        raise ValueError(
            f"grid preflight launch status is {preflight.launch_status!r}; no provider calls sent"
        )

    runtime_quote = _quote(base) if base.pricing.mode == "live" else quote
    if _pricing_terms(runtime_quote) != _pricing_terms(quote):
        raise ValueError(
            "live pricing changed during immediate pre-launch revalidation"
        )

    run_id = f"{base.experiment.name}-{base.execution.seed}"
    grid_dir = results_run_dir(
        output_dir, game=base.game.type, experiment=base.experiment.name, run_id=run_id
    )
    _wipe_run_dir_if_requested(
        grid_dir, wipe_and_recompute=base.storage.wipe_and_recompute, label=run_id
    )
    cells_dir = grid_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    print_banner(
        format_grid_banner(
            experiment_name=base.experiment.name,
            game_type=base.game.type,
            game_version=game.spec.version,
            provider=base.llm_provider.type,
            model=base.llm_provider.model,
            cell_count=len(cells),
            total_episode_count=preflight.total_episode_count,
            concurrency=base.execution.parallelism,
            axes=tuple((axis.path, len(axis.values)) for axis in grid.axes),
            preflight_expected_cost=format_money(preflight.total_costs.expected),
            preflight_conservative_cost=format_money(
                preflight.total_costs.conservative
            ),
            preflight_status=preflight.launch_status,
        )
    )
    # Printed unconditionally and up front: this is the one line that answers
    # "where do I look for the results" without waiting for the run to finish
    # or grepping logs.
    print_banner(f"  Results:       {grid_dir}")

    effective_budget = resolve_budget_limits(system_budget, run_budget)
    guard = RuntimeBudgetGuard(effective_budget, expectation=_expectation(preflight))

    metrics, to_round_view = game_metrics(game)
    policy = DetailedAuditPolicy.from_mapping(
        base.logging.options.get("detailed_prompt_audit")
    )
    price_hash = _pricing_identity(runtime_quote)
    _configure_durable_budget(
        base, grid_dir, guard, price_hash=price_hash, resume=resume
    )

    grid_seed = Seed(base.execution.seed)
    common_random_numbers = bool(
        base.experiment.metadata.get("common_random_numbers_across_grid", False)
    )
    all_tasks: list[_EpisodeTask] = []
    total_rounds = 0
    for cell in cells:
        # Opt-in matched interventions may hold every stochastic stream fixed
        # across grid cells.  The default retains the historical cell-derived
        # streams exactly.
        cell_seed = _grid_cell_seed(
            grid_seed, cell.index, common_random_numbers=common_random_numbers
        )
        cell_dir = cells_dir / cell.cell_id
        _write(cell_dir / "resolved_config.yaml", resolved_config_yaml(cell.config))
        _write(cell_dir / "overrides.json", _json(cell.to_dict()))
        episodes_dir = cell_dir / "data" / "episodes"
        plan = game.call_plan(cell.config.game)
        planned_episodes = normalized_episode_plan.get(cell.cell_id)
        if planned_episodes is None:
            planned_episodes = {
                index: int(cell_seed.derive(f"episode:{index}"))
                for index in range(cell.config.execution.repetitions)
            }
        total_rounds += _steps_per_episode(plan) * len(planned_episodes)
        for index, episode_seed in sorted(planned_episodes.items()):
            episode_id = f"{cell.cell_id}-{index:04d}"
            all_tasks.append(
                _EpisodeTask(
                    episode_id=episode_id,
                    seed=episode_seed,
                    config=replace(
                        cell.config,
                        execution=replace(cell.config.execution, seed=episode_seed),
                    ),
                    episode_dir=episodes_dir / episode_id,
                    cell_id=cell.cell_id,
                    cell_dir=cell_dir,
                )
            )

    all_tasks = [
        _with_scientific_identity(task, run_id=run_id, price_hash=price_hash)
        for task in all_tasks
    ]
    scheduled_tasks, resumed_outcomes = _partition_resume_tasks(
        all_tasks, resume=resume, price_hash=price_hash
    )
    guarded_provider = None
    timing_provider = None
    watcher = None
    if scheduled_tasks:
        provider = create_llm_provider(base.llm_provider, registry=_provider_registry())
        guarded_provider = BudgetGuardedProvider(
            provider,
            guard,
            runtime_quote.pricing,
            input_token_estimator=estimate_input_tokens,
            input_token_multiplier=1.0,
        )
        if base.storage.retention_policy.detailed_timing:
            timing_provider = _TimingProvider(guarded_provider)
            guarded_provider = timing_provider
        watcher = _spend_watcher(base, guard, effective_budget)

    progress = ExperimentProgress(
        total_episodes=len(all_tasks),
        total_rounds=total_rounds,
        show=show_progress,
    )
    # One sweep experiment for the whole grid, plus one experiment per cell at
    # that cell's completion. Episode-level Comet stays off (see
    # `_execute_episode`); the master is the only writer, which is what keeps
    # the step counters unraced and Comet a view rather than the store.
    axes = tuple((axis.path, tuple(axis.values)) for axis in grid.axes)
    layout = SweepLayout(
        axes=axes,
        cells={
            cell.cell_id: CellLayout(
                coordinates=tuple(cell.overrides.get(path) for path, _ in axes),
                episodes=len(normalized_episode_plan.get(cell.cell_id, {}))
                if episode_plan is not None
                else cell.config.execution.repetitions,
            )
            for cell in cells
        },
    )
    monitor = _master_monitor(base, run_id, len(all_tasks), layout)
    monitor.start(sweep_parameters(base, axes))
    print_banner(f"  Comet:         {monitor.describe()}")
    aggregator = GridAggregator(
        grid_dir,
        base.aggregation,
        monitor=monitor,
        ground_truth=aggregation_ground_truth(base, grid, game),
        seed=base.execution.seed,
    )
    completion = _CellCompletion(
        {
            cell.cell_id: (
                len(normalized_episode_plan[cell.cell_id])
                if episode_plan is not None
                else cell.config.execution.repetitions
            )
            for cell in cells
        }
    )
    prompt_options = dict(base.logging.options.get("prompt_examples", {}) or {})
    prompt_sampler = (
        _CellPromptSampler(int(prompt_options.get("count", 0)))
        if prompt_options.get(
            "scope",
            "cell" if base.storage.retention_policy.compact_scientific else "episode",
        )
        == "cell"
        else None
    )
    finalizer = (
        _ResultsOnlyFinalizer(all_tasks, prompt_sampler)
        if base.storage.retention_policy.compact_scientific
        else None
    )
    cell_analysis = (
        _PerCellAnalysisReporter(cells, cells_dir, monitor.analysis_sink)
        if per_cell_reports_enabled(base)
        else None
    )
    semaphore = asyncio.Semaphore(base.execution.parallelism)
    abort = asyncio.Event()
    budget_abort = asyncio.Event()
    started_at = _now()
    outcomes: tuple[EpisodeOutcome, ...] = tuple(resumed_outcomes)

    try:
        await _prime_resumed_outcomes(
            resumed_outcomes,
            progress=progress,
            monitor=monitor,
            guard=guard,
            completion=completion,
            aggregator=aggregator,
            finalizer=finalizer,
            cell_analysis=cell_analysis,
        )
        fresh_outcomes = await _with_spend_watch(
            watcher,
            _run_task_batch(
                scheduled_tasks,
                game=game,
                guarded_provider=guarded_provider,
                guard=guard,
                semaphore=semaphore,
                abort=abort,
                budget_abort=budget_abort,
                fail_fast=base.execution.fail_fast,
                resume=resume,
                policy=policy,
                metrics=metrics,
                to_round_view=to_round_view,
                price_hash=price_hash,
                checkpoint_enabled=base.storage.checkpoints,
                progress=progress,
                monitor=monitor,
                aggregator=aggregator,
                completion=completion,
                prompt_sampler=prompt_sampler,
                finalizer=finalizer,
                cell_analysis=cell_analysis,
            ),
        )
        by_id = {
            (outcome.cell_id, outcome.episode_id): outcome
            for outcome in (*resumed_outcomes, *fresh_outcomes)
        }
        outcomes = tuple(by_id[(task.cell_id, task.episode_id)] for task in all_tasks)
    finally:
        if guarded_provider is not None:
            guarded_provider.close()
        progress.close()
        if prompt_sampler is not None and finalizer is None:
            for cell in cells:
                prompt_sampler.render(
                    cells_dir / cell.cell_id,
                    [
                        outcome.episode_id
                        for outcome in outcomes
                        if outcome.cell_id == cell.cell_id
                        and outcome.status in {"completed", "skipped_resumed"}
                    ],
                )
        aggregator.finish()
        # The same figure the dashboard gets, written locally regardless of
        # whether Comet is on - so the picture is checkable, and a cluster job
        # with no outbound network is still watchable by tailing this file.
        try:
            monitor.save_grid_figure(grid_dir / "grid_progress.png")
        except Exception as exc:
            LOGGER.warning("could not render the grid image (%s)", type(exc).__name__)
        _write(
            grid_dir
            / (
                "comet_summary.json"
                if base.storage.retention_policy.compact_scientific
                else "comet_run_summary.json"
            ),
            _json(monitor.close()),
        )

    by_cell: dict[str, list[EpisodeOutcome]] = {cell.cell_id: [] for cell in cells}
    for outcome in outcomes:
        by_cell[outcome.cell_id].append(outcome)
    cell_results = tuple(
        GridCellResult(cell.cell_id, cell.overrides, tuple(by_cell[cell.cell_id]))
        for cell in cells
    )

    result = GridResult(
        grid_id=grid.grid_id,
        run_id=run_id,
        experiment_name=base.experiment.name,
        game_type=base.game.type,
        output_dir=grid_dir,
        cells=cell_results,
        preflight=preflight,
        budget_status=guard.status(),
        started_at=started_at,
        finished_at=_now(),
    )
    _write_grid_summary(grid_dir, grid, result)
    for cell, cell_result in zip(cells, cell_results):
        cell_dir = cells_dir / cell.cell_id
        failures = [
            outcome.to_dict()
            for outcome in cell_result.outcomes
            if outcome.status in {"failed", "skipped_aborted"}
        ]
        _write(
            cell_dir / "cell_summary.json",
            _json(
                {
                    "cell_id": cell_result.cell_id,
                    "overrides": dict(cell_result.overrides),
                    "completed": cell_result.completed,
                    "failed": cell_result.failed,
                    "skipped_resumed": cell_result.skipped_resumed,
                    "skipped_aborted": cell_result.skipped_aborted,
                    **(
                        {"failures": failures}
                        if base.storage.retention_policy.compact_scientific
                        else {
                            "outcomes": [
                                outcome.to_dict() for outcome in cell_result.outcomes
                            ]
                        }
                    ),
                }
            ),
        )
    if base.storage.retention_policy.compact_scientific:
        merge_cell_scientific_tables(grid_dir)
    run_configured_analysis(base, grid_dir)
    if base.storage.retention_policy.compact_scientific:
        _write_timing_study(
            grid_dir,
            outcomes,
            profile=base.storage.artifact_profile,
            run_started_at=result.started_at,
            run_finished_at=result.finished_at,
            cell_overrides={cell.cell_id: cell.overrides for cell in cells},
            request_rows=() if timing_provider is None else timing_provider.rows,
        )
        _record_results_only_hashes(grid_dir)
    return result


def run_experiment_grid_sync(*args: Any, **kwargs: Any) -> GridResult:
    return asyncio.run(run_experiment_grid(*args, **kwargs))
