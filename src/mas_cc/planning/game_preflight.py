"""Compose a provider-neutral game call plan with prompts and provider pricing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mas_cc.config import LLMProviderConfig, PromptConfig
from mas_cc.llm_providers import (
    BudgetLimits,
    CompletionRequest,
    MonetaryAmount,
    OfflinePricingSource,
    PricingQuote,
)
from mas_cc.prompts import PromptComposer, RegexTokenCounter, create_default_prompt_registry

from .call_graph import GameCallPlan, LogicalCallSpec
from .cost_estimation import estimate_cost
from .preflight import EstimateRange, MonetaryEstimateRange, static_preflight
from .token_estimation import estimate_input_tokens


def _sum_money(values: list[MonetaryAmount | None]) -> MonetaryAmount | None:
    present = [value for value in values if value is not None]
    if len(present) != len(values) or not present:
        return None
    template = present[0]
    if any(value.unit != template.unit for value in present):
        return None
    return MonetaryAmount(
        amount=sum(value.amount for value in present),
        unit=template.unit,
        unit_source=template.unit_source,
        provider=template.provider,
        model=template.model,
        source=template.source,
        retrieved_at=template.retrieved_at,
        version=template.version,
    )


@dataclass(frozen=True, slots=True)
class GamePreflightEstimate:
    game_type: str
    provider: str
    model: str
    provider_requests: EstimateRange
    input_tokens: EstimateRange
    output_tokens: EstimateRange
    costs: MonetaryEstimateRange
    prompt_scenarios: tuple[dict[str, Any], ...]
    pricing: dict[str, Any]
    launch_status: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "game_type": self.game_type,
            "provider": self.provider,
            "model": self.model,
            "provider_requests": self.provider_requests.to_dict(),
            "token_estimates": {
                "input": self.input_tokens.to_dict(),
                "output": self.output_tokens.to_dict(),
            },
            "cost_estimates": self.costs.to_dict(),
            "prompt_scenarios": list(self.prompt_scenarios),
            "pricing": dict(self.pricing),
            "launch_status": self.launch_status,
            "warnings": list(self.warnings),
        }


def static_game_preflight(
    plan: GameCallPlan,
    prompt_config: PromptConfig,
    provider_config: LLMProviderConfig,
    *,
    assumed_output_tokens: int = 1,
    pricing_quote: PricingQuote | None = None,
    system_budget: BudgetLimits | None = None,
    run_budget: BudgetLimits | None = None,
    explicit_override: bool = False,
    allow_stale_pricing: bool = False,
) -> GamePreflightEstimate:
    """Price all lower/expected/maximum demand scenarios without provider I/O."""

    if assumed_output_tokens < 1:
        raise ValueError("assumed_output_tokens must be positive")
    composer = PromptComposer(create_default_prompt_registry(), RegexTokenCounter())
    quote = pricing_quote or OfflinePricingSource().fetch(
        provider_config.type, provider_config.model
    )
    pricing = quote.pricing
    lower_inputs = lower_outputs = expected_inputs = expected_outputs = 0
    maximum_inputs = maximum_outputs = 0
    lower_costs: list[MonetaryAmount | None] = []
    expected_costs: list[MonetaryAmount | None] = []
    maximum_costs: list[MonetaryAmount | None] = []
    scenarios: list[dict[str, Any]] = []
    maximum_request: CompletionRequest | None = None
    maximum_per_call_tokens = -1

    for stage in plan.decision_stages:
        if stage.requests_per_interaction == 0:
            continue
        if stage.representative_prompt is None or stage.maximum_prompt is None:
            raise ValueError(
                f"decision stage {stage.name!r} must define representative and maximum prompts"
            )
        representative_prompt = composer.compose(
            prompt_config, stage.representative_prompt.context
        )
        maximum_prompt = composer.compose(prompt_config, stage.maximum_prompt.context)
        representative_request = CompletionRequest(
            representative_prompt.messages,
            temperature=provider_config.temperature,
            max_output_tokens=provider_config.max_output_tokens,
            metadata={"game_type": plan.game_type, "decision_stage": stage.name},
        )
        maximum_stage_request = CompletionRequest(
            maximum_prompt.messages,
            temperature=provider_config.temperature,
            max_output_tokens=provider_config.max_output_tokens,
            metadata={"game_type": plan.game_type, "decision_stage": stage.name},
        )
        representative_input = estimate_input_tokens(representative_request)
        maximum_input = estimate_input_tokens(maximum_stage_request)
        if maximum_input < representative_input:
            raise ValueError(
                f"decision stage {stage.name!r} maximum prompt is smaller than its representative"
            )
        lower_calls = stage.requests(plan.interactions.lower)
        expected_calls = stage.requests(plan.interactions.expected)
        maximum_calls = stage.requests(plan.interactions.maximum, include_retries=True)
        lower_inputs += representative_input * lower_calls
        expected_inputs += representative_input * expected_calls
        maximum_inputs += maximum_input * maximum_calls
        lower_outputs += lower_calls
        expected_outputs += assumed_output_tokens * expected_calls
        maximum_outputs += provider_config.max_output_tokens * maximum_calls
        lower_costs.append(
            estimate_cost(
                pricing,
                input_tokens_per_call=representative_input,
                output_tokens_per_call=1,
                logical_calls=lower_calls,
            )
        )
        expected_costs.append(
            estimate_cost(
                pricing,
                input_tokens_per_call=representative_input,
                output_tokens_per_call=assumed_output_tokens,
                logical_calls=expected_calls,
            )
        )
        maximum_costs.append(
            estimate_cost(
                pricing,
                input_tokens_per_call=maximum_input,
                output_tokens_per_call=provider_config.max_output_tokens,
                logical_calls=maximum_calls,
            )
        )
        scenarios.append(
            {
                "decision_stage": stage.name,
                "representative": {
                    "name": stage.representative_prompt.name,
                    "input_tokens_per_request": representative_input,
                    "base_requests": expected_calls,
                },
                "maximum": {
                    "name": stage.maximum_prompt.name,
                    "input_tokens_per_request": maximum_input,
                    "requests_including_retries": maximum_calls,
                },
            }
        )
        if maximum_input > maximum_per_call_tokens:
            maximum_request = maximum_stage_request
            maximum_per_call_tokens = maximum_input

    request_counts = plan.provider_requests
    if maximum_request is None or request_counts.maximum < 1:
        raise ValueError("game call plan contains no provider-backed decisions")
    conservative_check = static_preflight(
        maximum_request,
        provider_config,
        LogicalCallSpec(request_counts.maximum),
        assumed_output_tokens=provider_config.max_output_tokens,
        pricing_quote=quote,
        conservative_multiplier=1.0,
        system_budget=system_budget,
        run_budget=run_budget,
        explicit_override=explicit_override,
        allow_stale_pricing=allow_stale_pricing,
    )
    warnings = (
        "Lower/expected scenarios use representative contexts; conservative demand uses each stage's maximum context and retry bound.",
        *conservative_check.warnings,
    )
    return GamePreflightEstimate(
        game_type=plan.game_type,
        provider=provider_config.type,
        model=provider_config.model,
        provider_requests=EstimateRange(
            request_counts.lower, request_counts.expected, request_counts.maximum
        ),
        input_tokens=EstimateRange(lower_inputs, expected_inputs, maximum_inputs),
        output_tokens=EstimateRange(lower_outputs, expected_outputs, maximum_outputs),
        costs=MonetaryEstimateRange(
            _sum_money(lower_costs),
            _sum_money(expected_costs),
            _sum_money(maximum_costs),
        ),
        prompt_scenarios=tuple(scenarios),
        pricing={
            "mode": conservative_check.pricing_mode,
            "status": conservative_check.pricing_status,
            "source": conservative_check.pricing_source,
            "version": conservative_check.pricing_version,
            "retrieved_at": conservative_check.pricing_retrieved_at,
            "fresh_until": conservative_check.pricing_fresh_until,
        },
        launch_status=conservative_check.launch_status,
        warnings=tuple(dict.fromkeys(warnings)),
    )
