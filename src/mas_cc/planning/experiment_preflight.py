"""Price a whole experiment (N episodes of one resolved config) before launch.

Reuses :func:`static_game_preflight`'s per-episode estimate rather than
re-deriving demand: an experiment's total demand is that estimate multiplied
by the episode count, checked against the same budget resolution path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mas_cc.config import LLMProviderConfig, PromptConfig
from mas_cc.llm_runtime.providers import BudgetLimits, resolve_budget_limits, PricingQuote

from .call_graph import GameCallPlan
from .game_preflight import GamePreflightEstimate, static_game_preflight
from .preflight import EstimateRange, MonetaryEstimateRange


def _scale_range(value: EstimateRange, factor: int) -> EstimateRange:
    return EstimateRange(value.lower * factor, value.expected * factor, value.conservative * factor)


def _scale_money(value: Any, factor: int) -> Any:
    if value is None:
        return None
    return type(value)(
        amount=value.amount * factor, unit=value.unit, unit_source=value.unit_source,
        provider=value.provider, model=value.model, source=value.source,
        retrieved_at=value.retrieved_at, version=value.version,
    )


@dataclass(frozen=True, slots=True)
class ExperimentPreflightEstimate:
    game_type: str
    provider: str
    model: str
    episode_count: int
    concurrency: int
    per_episode: GamePreflightEstimate
    total_provider_requests: EstimateRange
    total_input_tokens: EstimateRange
    total_output_tokens: EstimateRange
    total_costs: MonetaryEstimateRange
    rough_runtime_seconds: float
    launch_status: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "game_type": self.game_type,
            "provider": self.provider,
            "model": self.model,
            "episode_count": self.episode_count,
            "concurrency": self.concurrency,
            "per_episode": self.per_episode.to_dict(),
            "total_provider_requests": self.total_provider_requests.to_dict(),
            "total_input_tokens": self.total_input_tokens.to_dict(),
            "total_output_tokens": self.total_output_tokens.to_dict(),
            "total_costs": self.total_costs.to_dict(),
            "rough_runtime_seconds": self.rough_runtime_seconds,
            "launch_status": self.launch_status,
            "warnings": list(self.warnings),
        }


def apply_total_demand_budget_check(
    launch_status: str,
    warnings: list[str],
    total_costs: MonetaryEstimateRange,
    total_requests: EstimateRange,
    *,
    system_budget: BudgetLimits | None,
    run_budget: BudgetLimits | None,
    explicit_override: bool,
) -> tuple[str, list[str]]:
    """Check one combined conservative total against a resolved budget.

    Shared between :func:`static_experiment_preflight` (one config, N episodes) and
    :func:`static_grid_preflight` (many configs, summed across all their episodes) — the check
    itself doesn't care whether the total came from scaling one estimate or summing several.
    """

    warnings = list(warnings)
    if launch_status != "permitted":
        return launch_status, warnings
    effective = None
    if system_budget is not None or run_budget is not None:
        effective = resolve_budget_limits(system_budget or BudgetLimits(), run_budget)
    if effective is None:
        return launch_status, warnings
    conservative = total_costs.conservative
    if effective.max_cost is not None:
        if conservative is None or conservative.unit != effective.max_cost.unit:
            if not explicit_override:
                launch_status = "explicit-override-required"
                warnings.append(
                    "Total cost and the approved limit cannot be compared safely in one "
                    "accounting unit."
                )
        elif conservative.amount > effective.max_cost.amount:
            launch_status = "denied"
            warnings.append("Conservative total cost exceeds an effective MAS-CC run limit.")
    if effective.max_requests is not None and total_requests.conservative > effective.max_requests:
        launch_status = "denied"
        warnings.append(
            "Conservative total provider requests exceed an effective MAS-CC run limit."
        )
    return launch_status, warnings


def static_experiment_preflight(
    plan: GameCallPlan,
    prompt_config: PromptConfig,
    provider_config: LLMProviderConfig,
    *,
    episode_count: int,
    concurrency: int = 1,
    assumed_output_tokens: int = 1,
    pricing_quote: PricingQuote | None = None,
    system_budget: BudgetLimits | None = None,
    run_budget: BudgetLimits | None = None,
    explicit_override: bool = False,
    allow_stale_pricing: bool = False,
    seconds_per_episode: float | None = None,
) -> ExperimentPreflightEstimate:
    """Price ``episode_count`` independent, identically configured episodes."""

    if episode_count < 1:
        raise ValueError("episode_count must be a positive integer")
    if concurrency < 1:
        raise ValueError("concurrency must be a positive integer")

    per_episode = static_game_preflight(
        plan, prompt_config, provider_config,
        assumed_output_tokens=assumed_output_tokens, pricing_quote=pricing_quote,
        # Budget checks are deferred to the multiplied total below; a single
        # episode's conservative demand may be within budget while N are not.
        explicit_override=explicit_override, allow_stale_pricing=allow_stale_pricing,
    )

    total_requests = _scale_range(per_episode.provider_requests, episode_count)
    total_inputs = _scale_range(per_episode.input_tokens, episode_count)
    total_outputs = _scale_range(per_episode.output_tokens, episode_count)
    total_costs = MonetaryEstimateRange(
        _scale_money(per_episode.costs.lower, episode_count),
        _scale_money(per_episode.costs.expected, episode_count),
        _scale_money(per_episode.costs.conservative, episode_count),
    )

    launch_status, warnings = apply_total_demand_budget_check(
        per_episode.launch_status, list(per_episode.warnings), total_costs, total_requests,
        system_budget=system_budget, run_budget=run_budget, explicit_override=explicit_override,
    )

    if seconds_per_episode is not None:
        rough_runtime = seconds_per_episode * ((episode_count + concurrency - 1) // concurrency)
    else:
        # Fall back to the per-episode runtime estimate already embedded in
        # static_game_preflight's conservative check via game_preflight's own
        # cost path is not runtime-aware, so approximate using one request's
        # rough runtime scaled by demand and divided across concurrent slots.
        from .runtime_estimation import estimate_runtime_seconds

        per_episode_seconds = estimate_runtime_seconds(
            provider_config.type,
            logical_calls=per_episode.provider_requests.expected,
            request_concurrency=(
                1 if provider_config.type == "gemma_local" else provider_config.request_concurrency
            ),
        )
        rough_runtime = per_episode_seconds * ((episode_count + concurrency - 1) // concurrency)

    return ExperimentPreflightEstimate(
        game_type=plan.game_type,
        provider=provider_config.type,
        model=provider_config.model,
        episode_count=episode_count,
        concurrency=concurrency,
        per_episode=per_episode,
        total_provider_requests=total_requests,
        total_input_tokens=total_inputs,
        total_output_tokens=total_outputs,
        total_costs=total_costs,
        rough_runtime_seconds=rough_runtime,
        launch_status=launch_status,
        warnings=tuple(dict.fromkeys(warnings)),
    )
