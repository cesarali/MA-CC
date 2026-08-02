"""Generic game execution and Phase 5 inspection artifacts."""

from __future__ import annotations

import csv
import io
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from mas_cc.config import RunConfig, load_run_config, resolved_config_yaml
from mas_cc.games import create_game, run_game_sync
from mas_cc.llm_providers import (
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
from mas_cc.planning import estimate_input_tokens, static_game_preflight
from mas_cc.prompts import RegexTokenCounter

from .inspect import _write, _write_manifest


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


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


def _budgets(config: RunConfig, quote: PricingQuote) -> tuple[BudgetLimits, BudgetLimits]:
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


def _pricing_terms(quote: PricingQuote) -> dict[str, Any] | None:
    if quote.pricing is None:
        return None
    terms = quote.pricing.to_dict()
    for provenance_field in ("source", "retrieved_at", "version"):
        terms.pop(provenance_field, None)
    return terms


def _trajectory_csv(result: Any) -> str:
    output = io.StringIO(newline="")
    agent_ids = [str(agent.agent_id) for agent in result.initial_state.agents]
    fieldnames = [
        "interaction",
        "participant_1",
        "participant_2",
        "action_1",
        "action_2",
        "matched",
        "payoff",
        *(f"score_{agent_id}" for agent_id in agent_ids),
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for interaction in result.interactions:
        transition = interaction.transition
        row: dict[str, Any] = {
            "interaction": interaction.turn,
            "participant_1": str(interaction.participants[0]),
            "participant_2": str(interaction.participants[1]),
            "action_1": transition.actions[0].value,
            "action_2": transition.actions[1].value,
            "matched": transition.matched,
            "payoff": next(iter(transition.payoffs.values())),
        }
        row.update(
            {
                f"score_{agent.agent_id}": agent.score
                for agent in transition.next_state.agents
            }
        )
        writer.writerow(row)
    return output.getvalue()


def _trajectory_plot(result: Any, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4), dpi=120)
    turns = [0, *(interaction.turn for interaction in result.interactions)]
    for initial_agent in result.initial_state.agents:
        scores = [initial_agent.score]
        scores.extend(
            interaction.transition.next_state.agent(initial_agent.agent_id).score
            for interaction in result.interactions
        )
        axis.step(turns, scores, where="post", marker="o", label=str(initial_agent.agent_id))
    axis.set_xlabel("Interaction")
    axis.set_ylabel("Cumulative payoff")
    axis.set_title("Toy coordination trajectory")
    axis.set_xticks(turns)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, metadata={"Software": "MAS-CC"})
    plt.close(figure)


def run_game_inspection(config_path: str | Path, output_dir: str | Path) -> bool:
    """Run one configured game and write the complete Phase 5 inspection bundle."""

    source = Path(config_path).resolve()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config = load_run_config(source)
    if config.game.type == "naming_convention":
        from .naming_convention import run_naming_convention_inspection

        quote = _quote(config)
        runtime_quote = _quote(config) if config.pricing.mode == "live" else quote
        if _pricing_terms(runtime_quote) != _pricing_terms(quote):
            raise ValueError("live pricing changed during immediate pre-launch revalidation")
        system_budget, run_budget = _budgets(config, quote)
        return run_naming_convention_inspection(
            config,
            source,
            destination,
            quote=quote,
            runtime_quote=runtime_quote,
            system_budget=system_budget,
            run_budget=run_budget,
        )
    game = create_game(config.game)
    plan = game.call_plan(config.game)
    quote = _quote(config)
    system_budget, run_budget = _budgets(config, quote)
    preflight = static_game_preflight(
        plan,
        config.prompt,
        config.llm_provider,
        assumed_output_tokens=1,
        pricing_quote=quote,
        system_budget=system_budget,
        run_budget=run_budget,
        explicit_override=config.pricing.explicit_unknown_price_override,
        allow_stale_pricing=not config.pricing.require_fresh_at_launch,
    )
    if preflight.launch_status != "permitted":
        raise ValueError(
            f"game preflight launch status is {preflight.launch_status!r}; no provider calls sent"
        )

    # Live University metadata is queried once during preflight and immediately
    # revalidated before launch, never once per completion.
    runtime_quote = _quote(config) if config.pricing.mode == "live" else quote
    if _pricing_terms(runtime_quote) != _pricing_terms(quote):
        raise ValueError("live pricing changed during immediate pre-launch revalidation")
    effective_budget = resolve_budget_limits(system_budget, run_budget)
    guard = RuntimeBudgetGuard(effective_budget)
    provider = create_llm_provider(config.llm_provider)
    guarded_provider = BudgetGuardedProvider(
        provider,
        guard,
        runtime_quote.pricing,
        input_token_estimator=estimate_input_tokens,
        input_token_multiplier=1.0,
    )
    try:
        result = run_game_sync(game, config, guarded_provider)
    finally:
        guarded_provider.close()

    _write(destination / "resolved_config.yaml", resolved_config_yaml(config))
    _write(destination / "initial_state.json", _json(result.initial_state.to_dict()))
    decisions = tuple(
        decision for interaction in result.interactions for decision in interaction.decisions
    )
    _write(
        destination / "observations.jsonl",
        "".join(
            json.dumps(decision.request.observation.to_dict(), sort_keys=True) + "\n"
            for decision in decisions
        ),
    )
    _write(
        destination / "bound_prompts.jsonl",
        "".join(
            json.dumps(decision.request.prompt.to_dict(), sort_keys=True) + "\n"
            for decision in decisions
        ),
    )
    _write(
        destination / "compiled_prompts.jsonl",
        "".join(
            json.dumps(
                decision.request.prompt.compile(RegexTokenCounter()).to_dict(),
                sort_keys=True,
            )
            + "\n"
            for decision in decisions
        ),
    )
    _write(
        destination / "interactions.jsonl",
        "".join(
            json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
            for item in result.interactions
        ),
    )
    _write(destination / "final_state.json", _json(result.final_state.to_dict()))
    _write(destination / "game_call_plan.json", _json(plan.to_dict()))
    _write(
        destination / "prompt_scenarios.json",
        _json(
            [
                scenario.to_dict()
                for stage in plan.decision_stages
                for scenario in stage.prompt_scenarios
            ]
        ),
    )
    _write(destination / "trajectory.csv", _trajectory_csv(result))
    _trajectory_plot(result, destination / "trajectory.png")

    second_plan = game.call_plan(config.game)
    guard_status = guard.status()
    checks = {
        "generic_game_protocol_completed": len(result.interactions) == config.game.horizon,
        "finite_horizon_terminated": (
            result.final_state.terminated
            and result.termination_reason == "finite_horizon_reached"
        ),
        "interaction_trace_complete": all(
            len(item.decisions) == 2
            and all(
                decision.completion_request is not None
                and decision.response is not None
                and decision.action.value in {"A", "B"}
                for decision in item.decisions
            )
            for item in result.interactions
        ),
        "transitions_follow_matching_rule": all(
            item.transition.matched
            == (item.transition.actions[0].value == item.transition.actions[1].value)
            for item in result.interactions
        ),
        "call_plan_is_provider_independent_and_stable": plan == second_plan,
        "planned_requests_match_actual": (
            plan.provider_requests.expected
            == sum(len(item.decisions) for item in result.interactions)
        ),
        "phase_4_pricing_composes_with_game_demand": (
            preflight.provider_requests.expected == plan.provider_requests.expected
            and preflight.costs.expected is not None
        ),
        "runtime_budget_guard_reconciled": (
            guard_status["active_reservations"] == 0
            and guard_status["used_and_reserved"]["requests"]
            == plan.provider_requests.expected
        ),
        "trajectory_plot_nonempty": (destination / "trajectory.png").stat().st_size > 0,
    }
    status = "pass" if all(checks.values()) else "fail"
    report = f"""# Phase 5 inspection report

- Status: **{status.upper()}**
- Command: `mas-cc game run --config {config_path} --output-dir {destination}`
- Code paths exercised: resolved configuration loading, lazy game/provider registries, generic game protocol, compositional prompt rendering, normalized provider calls, local response/action validation, pure state transitions, provider-neutral demand planning, Phase 4 pricing composition, runtime budget enforcement, and deterministic trajectory rendering.
- Input: `{source}`
- Expected behavior: two agents choose A or B through the configured provider for exactly {config.game.horizon} interactions; matching actions earn one point; the same seed and resolved inputs reproduce all scientific trajectory artifacts.
- Deviations or warnings: provider timing is intentionally omitted from `interactions.jsonl` so deterministic mock traces remain byte reproducible; runtime timing/audit events enter in Phase 7.

## Results

- Interactions completed: {len(result.interactions)}
- Normalized provider requests: {guard_status['used_and_reserved']['requests']}
- Matching interactions: {result.final_state.data['matches']}
- Termination reason: `{result.termination_reason}`
- Provider-independent expected request demand: {plan.provider_requests.expected}
- Static pricing composition status: `{preflight.pricing['status']}` / launch `{preflight.launch_status}`
- Final scores: {', '.join(f'{agent.agent_id}={agent.score:g}' for agent in result.final_state.agents)}

## Files to inspect manually

- `resolved_config.yaml` — all component references expanded without secrets.
- `initial_state.json` — immutable state before any provider call.
- `observations.jsonl`, `bound_prompts.jsonl`, and `compiled_prompts.jsonl` — the narrow prompt chain for every decision.
- `interactions.jsonl` — one complete observation/prompt/response/action/transition chain per line.
- `final_state.json` — terminated state and cumulative scores.
- `game_call_plan.json` and `prompt_scenarios.json` — provider-independent demand and bound prompt scenarios.
- `trajectory.csv` and `trajectory.png` — tabular and visual score trajectories.
- `manifest.json` — artifact hashes and machine-readable acceptance checks.
"""
    _write(destination / "report.md", report)
    _write_manifest(destination, phase=5, status=status, checks=checks)
    return status == "pass"
