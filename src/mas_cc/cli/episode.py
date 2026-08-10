"""``mas-cc game episode`` / ``mas-cc game preflight`` — the config-driven single-episode CLI.

Deliberately narrow: two commands, two flags each (``--config``, optional
``--output-dir``). Every other behavior — progress display, Comet, metrics
printing, Markdown prompt examples, the raw JSONL prompt audit — is read
straight from the resolved config's ``logging``/``metrics``/``control``
sections, never from a CLI flag. This is the single-episode replacement for
``mas-cc game run`` / ``mas-cc inspect phase 7``; those two stay in place
unchanged because the test suite pins them down as phase-regression
fixtures, not because they're still the recommended way to run an episode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mas_cc.config import RunConfig, load_run_config, resolved_config_yaml
from mas_cc.control import create_control
from mas_cc.experiments.console import format_episode_banner, format_money, print_banner
from mas_cc.games import create_game, game_metrics
from mas_cc.games.naming_convention import NamingConventionGame, run_naming_convention_game_sync
from mas_cc.llm_runtime.providers import (
    BudgetGuardedProvider,
    RuntimeBudgetGuard,
    create_llm_provider,
    resolve_budget_limits,
)
from mas_cc.metrics import FinalMetric, StreamingMetric, plot_streaming_metrics
from mas_cc.observability import DetailedAuditPolicy, RunRecorder, price_snapshot_hash
from mas_cc.planning import GamePreflightEstimate, estimate_input_tokens, static_game_preflight
from mas_cc.llm_runtime.prompts import PromptMarkdownLogger
from mas_cc.storage import results_run_dir

from .game import _budgets, _pricing_terms, _quote
from .inspect import _write

_DECISIONS_PER_INTERACTION = 2  # naming_convention is always a simultaneous pair decision


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _destination(config: RunConfig, output_dir: str | Path | None) -> Path:
    base = Path(output_dir) if output_dir is not None else Path(config.storage.output_dir)
    run_id = f"{config.experiment.name}-{config.execution.seed}"
    return results_run_dir(base, game=config.game.type, experiment=config.experiment.name, run_id=run_id)


def _load_naming_convention_game(config: RunConfig) -> NamingConventionGame:
    game = create_game(config.game)
    if not isinstance(game, NamingConventionGame):
        raise ValueError(
            "mas-cc game episode/preflight currently only supports game.type: naming_convention"
        )
    return game


def run_game_preflight(config_path: str | Path, output_dir: str | Path | None = None) -> GamePreflightEstimate:
    """Zero-provider-I/O cost estimate for exactly one episode of this config."""

    source = Path(config_path).resolve()
    config = load_run_config(source)
    game = _load_naming_convention_game(config)
    plan = game.call_plan(config.game)
    quote = _quote(config)
    system_budget, run_budget = _budgets(config, quote)
    estimate = static_game_preflight(
        plan, config.prompt, config.llm_provider,
        assumed_output_tokens=config.llm_provider.max_output_tokens,
        pricing_quote=quote, system_budget=system_budget, run_budget=run_budget,
        explicit_override=config.pricing.explicit_unknown_price_override,
        allow_stale_pricing=not config.pricing.require_fresh_at_launch,
    )

    destination = _destination(config, output_dir) / "preflight"
    destination.mkdir(parents=True, exist_ok=True)
    _write(destination / "resolved_config.yaml", resolved_config_yaml(config))
    _write(destination / "preflight_estimate.json", _json(estimate.to_dict()))
    _write(destination / "pricing_snapshot.json", _json(quote.to_dict()))
    report = f"""# game preflight

- Status: **{"PASS" if estimate.launch_status == "permitted" else "FAIL"}** (`{estimate.launch_status}`)
- Experiment: `{config.experiment.name}`; game `{config.game.type}`; provider `{config.llm_provider.type}` / `{config.llm_provider.model}`.
- Provider requests (lower/expected/conservative): {estimate.provider_requests.lower}/{estimate.provider_requests.expected}/{estimate.provider_requests.conservative}
- Expected cost: {format_money(estimate.costs.expected)}; conservative: {format_money(estimate.costs.conservative)}

## Warnings

{chr(10).join(f"- {warning}" for warning in estimate.warnings) or "- none"}
"""
    _write(destination / "report.md", report)
    for warning in estimate.warnings:
        print(f"  - {warning}")
    return estimate


class _EpisodeProgress:
    """Nested tqdm — rounds (outer) and decisions within the current round (inner).

    Config-driven only (``logging.options.progress``); auto-off when stdout
    isn't a TTY regardless, same convention as the experiment console.
    """

    def __init__(self, *, total_rounds: int, show: bool) -> None:
        # A real TTY is required for tqdm's in-place bars. `conda run` is
        # never a TTY on the console-visible fd - not even with
        # --live-stream/--no-capture-output, which only stops it from
        # buffering the whole run's output until exit; it doesn't attach a
        # pty. So "configured on but no TTY" still gets one flushed text
        # line per round below, instead of silence until the final summary.
        self._configured = show
        self._total_rounds = total_rounds
        self._show = show and sys.stdout.isatty()
        self._round_bar = None
        self._decision_bar = None
        if self._show:
            from tqdm.auto import tqdm

            self._round_bar = tqdm(
                total=total_rounds, desc="Rounds", unit="round",
                dynamic_ncols=True, position=0, mininterval=0.25,
            )
            self._decision_bar = tqdm(
                total=_DECISIONS_PER_INTERACTION, desc="Decisions", unit="decision",
                dynamic_ncols=True, position=1, mininterval=0.25, leave=False,
            )

    def decision_tick(self, round_index: int, agent_id: Any) -> None:
        if self._decision_bar is not None:
            self._decision_bar.set_postfix_str(f"round {round_index} | {agent_id}", refresh=False)
            self._decision_bar.update(1)

    def round_tick(self, round_index: int) -> None:
        if self._round_bar is not None:
            self._round_bar.set_postfix_str(f"round {round_index}", refresh=True)
            self._round_bar.update(1)
        if self._decision_bar is not None:
            self._decision_bar.reset(total=_DECISIONS_PER_INTERACTION)
        elif self._configured:
            print(f"round {round_index}/{self._total_rounds} complete", flush=True)

    def close(self) -> None:
        if self._decision_bar is not None:
            self._decision_bar.close()
        if self._round_bar is not None:
            self._round_bar.close()


class _EpisodeObserver:
    """Wires RunRecorder + progress + Markdown prompt records.

    Two independent Markdown sources, both under ``<destination>/``:

    - ``prompts/`` — up to ``prompt_example_rounds`` *successful* rounds,
      config-driven (``logging.options.prompt_examples.count``), off by
      default.
    - ``failures/`` — every invalid response or provider error, **always
      written, not config-gated**. An unparseable/failing response is
      exactly the situation you need "what was sent and what came back" for,
      and by the time you know you need it, the round has already happened —
      so this one isn't opt-in the way the others are.

    Both are independent of the raw JSONL audit path
    (``logging.options.detailed_prompt_audit``).
    """

    def __init__(
        self, recorder: RunRecorder, guard: RuntimeBudgetGuard, progress: _EpisodeProgress,
        prompt_logger: PromptMarkdownLogger | None, prompt_example_rounds: int,
        failure_logger: PromptMarkdownLogger,
    ) -> None:
        self.recorder = recorder
        self.guard = guard
        self.progress = progress
        self.prompt_logger = prompt_logger
        self.prompt_example_rounds = prompt_example_rounds
        self.failure_logger = failure_logger
        self._logged_rounds: set[int] = set()
        self._failure_count = 0

    def event(self, event_type: str, **payload: Any) -> None:
        self.recorder.event(event_type, **payload)

    def record_attempt(self, **payload: Any) -> None:
        self.recorder.record_attempt(**payload, budget_status=self.guard.status())
        round_index = payload.get("round_index")
        attempt = payload.get("attempt")
        agent_id = payload["request"].metadata.get("agent_id")
        if attempt == 1:
            self.progress.decision_tick(round_index, agent_id)
        if (
            self.prompt_logger is not None
            and attempt == 1
            and round_index not in self._logged_rounds
            and len(self._logged_rounds) < self.prompt_example_rounds
        ):
            self.prompt_logger.log(
                payload["prompt"],
                interaction_id=f"round_{round_index:03d}",
                title=f"Round {round_index} — agent {agent_id}",
                metadata={"round_index": round_index, "agent_id": str(agent_id)},
            )
            self._logged_rounds.add(round_index)
        provider_error = payload.get("provider_error")
        if not payload.get("valid", True) or provider_error is not None:
            self._failure_count += 1
            self.failure_logger.log(
                payload["prompt"],
                interaction_id=f"round_{round_index:03d}_attempt_{attempt}_{self._failure_count:03d}",
                title=f"REJECTED — round {round_index}, attempt {attempt}, agent {agent_id}",
                metadata={
                    "round_index": round_index, "agent_id": str(agent_id), "attempt": attempt,
                    "provider_error": None if provider_error is None else str(provider_error),
                },
                response=payload.get("response"),
                validation_error=payload.get("validation_error"),
            )

    def record_interaction(self, **payload: Any) -> None:
        self.recorder.record_interaction(**payload, budget_status=self.guard.checkpoint_state())
        self.progress.round_tick(payload.get("round_index"))

    def record_trajectory(self, **payload: Any) -> None:
        self.recorder.record_trajectory(**payload)


def _print_final_metrics(result: Any, metrics: tuple[Any, ...], to_round_view: Any) -> None:
    views = tuple(to_round_view(interaction.transition.next_state) for interaction in result.interactions)
    if not views:
        return
    print("Metrics:")
    for metric in metrics:
        if isinstance(metric, StreamingMetric):
            for key, value in metric.compute_round(views[-1]).items():
                label = "population" if key is None else str(key)
                print(f"  {metric.name} [{label}]: {value:g}" if isinstance(value, (int, float)) else f"  {metric.name} [{label}]: {value}")
        elif isinstance(metric, FinalMetric):
            print(f"  {metric.name}: {metric.compute_final(views)}")


def run_game_episode(
    config_path: str | Path, output_dir: str | Path | None = None, *, skip_preflight: bool = False,
) -> bool:
    """Run exactly one episode of this config, config-driven, no other flags.

    Preflight (the cost/token/request estimate, and the gate that refuses to
    launch when it isn't ``permitted``) runs by default. ``skip_preflight``
    only skips *that* — the live price-quote revalidation just below it and
    the runtime ``RuntimeBudgetGuard`` that actually enforces ``budget.*``
    during execution both stay on regardless; this is a way to skip the
    estimate step, not a way to remove spend protection.
    """

    source = Path(config_path).resolve()
    config = load_run_config(source)
    game = _load_naming_convention_game(config)
    plan = game.call_plan(config.game)

    quote = _quote(config)
    system_budget, run_budget = _budgets(config, quote)
    preflight: GamePreflightEstimate | None = None
    if not skip_preflight:
        preflight = static_game_preflight(
            plan, config.prompt, config.llm_provider,
            assumed_output_tokens=config.llm_provider.max_output_tokens, pricing_quote=quote,
            system_budget=system_budget, run_budget=run_budget,
            explicit_override=config.pricing.explicit_unknown_price_override,
            allow_stale_pricing=not config.pricing.require_fresh_at_launch,
        )
        if preflight.launch_status != "permitted":
            raise ValueError(
                f"game preflight launch status is {preflight.launch_status!r}; no provider calls sent"
            )
    runtime_quote = _quote(config) if config.pricing.mode == "live" else quote
    if _pricing_terms(runtime_quote) != _pricing_terms(quote):
        raise ValueError("live pricing changed during immediate pre-launch revalidation")

    destination = _destination(config, output_dir)
    # Written now, before any provider call, not after a successful finish —
    # a failed episode (see the failures/ directory) is exactly when you most
    # need the exact config that produced it, to replay or debug against.
    _write(destination / "resolved_config.yaml", resolved_config_yaml(config))
    budget_description = (
        "unbounded" if run_budget.max_cost is None and system_budget.max_cost is None
        else format_money(run_budget.max_cost if run_budget.max_cost is not None else system_budget.max_cost)
    )
    definition_hash = plan.decision_stages[0].representative_prompt.bound_prompt.definition_hash
    print_banner(
        format_episode_banner(
            experiment_name=config.experiment.name, game_type=config.game.type,
            game_version=game.spec.version, provider=config.llm_provider.type,
            model=config.llm_provider.model, population_size=config.game.population_size,
            horizon=config.game.horizon, prompt_family=config.prompt.prompt_family,
            prompt_version=config.prompt.prompt_version, prompt_definition_hash=definition_hash,
            budget_description=budget_description,
            preflight_expected_cost=(
                "skipped" if preflight is None else format_money(preflight.costs.expected)
            ),
            preflight_conservative_cost=(
                "skipped" if preflight is None else format_money(preflight.costs.conservative)
            ),
            preflight_status="skipped (--no-preflight)" if preflight is None else preflight.launch_status,
            output_dir=str(destination),
        )
    )
    policy = DetailedAuditPolicy.from_mapping(config.logging.options.get("detailed_prompt_audit"))
    metrics, to_round_view = game_metrics(game)
    recorder = RunRecorder(
        destination, run_id=f"{config.experiment.name}-{config.execution.seed}",
        resolved_config=config.to_dict(), policy=policy, comet_enabled=config.logging.comet,
        project_name=str(config.logging.options.get("comet_project", "mas-cc")),
        checkpoint_enabled=config.storage.checkpoints,
        price_snapshot_hash=price_snapshot_hash(runtime_quote.to_dict()),
        metrics=metrics, to_round_view=to_round_view,
        comet_metric_export=config.metrics.comet_export_names() if config.metrics.enabled else (),
        binning=config.metrics.binning_policy(config.game.population_size),
    )
    guard = RuntimeBudgetGuard(resolve_budget_limits(system_budget, run_budget))
    provider = create_llm_provider(config.llm_provider)
    guarded = BudgetGuardedProvider(
        provider, guard, runtime_quote.pricing,
        input_token_estimator=estimate_input_tokens, input_token_multiplier=1.0,
    )

    show_progress = bool(config.logging.options.get("progress", True))
    show_metrics = bool(config.logging.options.get("show_metrics", False))
    example_count = int(dict(config.logging.options.get("prompt_examples", {}) or {}).get("count", 0))

    progress = _EpisodeProgress(total_rounds=config.game.horizon, show=show_progress)
    prompt_logger = PromptMarkdownLogger(destination / "prompts", overwrite=True) if example_count > 0 else None
    failure_logger = PromptMarkdownLogger(destination / "failures", overwrite=True)
    observer = _EpisodeObserver(recorder, guard, progress, prompt_logger, example_count, failure_logger)
    control = create_control(config.control)

    try:
        result = run_naming_convention_game_sync(game, config, guarded, observer=observer, control=control)
    except Exception as exc:
        recorder.event("run_failed", error_type=type(exc).__name__, error=str(exc))
        recorder.finalize(status="failed", budget_status=guard.checkpoint_state())
        if observer._failure_count:
            print(
                f"{observer._failure_count} rejected response(s) recorded: {destination / 'failures'}",
                file=sys.stderr,
            )
        raise
    finally:
        guarded.close()
        progress.close()

    summary = recorder.finalize(status="completed", budget_status=guard.checkpoint_state())

    if show_metrics and metrics and to_round_view is not None:
        _print_final_metrics(result, metrics, to_round_view)

    plotted: list[Path] = []
    if metrics:
        plotted = plot_streaming_metrics(destination / "metrics" / "streaming.csv", destination / "metrics" / "plots")

    print(
        f"Episode complete: {len(result.interactions)} interactions, "
        f"termination={result.termination_reason!r}"
    )
    print(f"Output: {destination}")
    if example_count > 0:
        print(f"Prompt examples ({len(observer._logged_rounds)}): {destination / 'prompts'}")
    if observer._failure_count:
        print(f"Rejected response(s) ({observer._failure_count}): {destination / 'failures'}")
    if plotted:
        print(f"Metric plots ({len(plotted)}): {destination / 'metrics' / 'plots'}")
    comet_names = config.metrics.comet_export_names() if config.metrics.enabled else ()
    if comet_names:
        print(f"Metrics exported to Comet ({len(comet_names)}): {', '.join(comet_names)}")
    print(f"Comet: {summary['comet']['status']}")
    return True
