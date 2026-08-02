"""Static preflight composition; no credentials, clients, or models are loaded."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mas_cc.config import LLMProviderConfig
from mas_cc.llm_providers import (
    BudgetCeiling,
    CompletionRequest,
    PricingCatalog,
    default_pricing_catalog,
)

from .call_graph import LogicalCallSpec
from .cost_estimation import estimate_cost_usd
from .runtime_estimation import estimate_runtime_seconds
from .token_estimation import TOKENIZER_NAME, estimate_input_tokens


@dataclass(frozen=True, slots=True)
class PreflightEstimate:
    provider: str
    model: str
    tokenizer: str
    estimated_input_tokens_per_call: int
    assumed_output_tokens_per_call: int
    logical_calls: int
    estimated_total_input_tokens: int
    estimated_total_output_tokens: int
    expected_cost_usd: float | None
    conservative_cost_bound_usd: float | None
    rough_runtime_seconds: float
    pricing_catalog_version: str
    pricing_source: str | None
    budget_ceiling_usd: float | None
    within_budget: bool | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider": self.provider,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "estimated_input_tokens_per_call": self.estimated_input_tokens_per_call,
            "assumed_output_tokens_per_call": self.assumed_output_tokens_per_call,
            "logical_calls": self.logical_calls,
            "estimated_total_input_tokens": self.estimated_total_input_tokens,
            "estimated_total_output_tokens": self.estimated_total_output_tokens,
            "expected_cost_usd": self.expected_cost_usd,
            "conservative_cost_bound_usd": self.conservative_cost_bound_usd,
            "rough_runtime_seconds": self.rough_runtime_seconds,
            "pricing_catalog_version": self.pricing_catalog_version,
            "pricing_source": self.pricing_source,
            "budget_ceiling_usd": self.budget_ceiling_usd,
            "within_budget": self.within_budget,
            "warnings": list(self.warnings),
        }


def static_preflight(
    request: CompletionRequest,
    config: LLMProviderConfig,
    calls: LogicalCallSpec,
    *,
    assumed_output_tokens: int | None = None,
    pricing_catalog: PricingCatalog | None = None,
    conservative_multiplier: float = 1.5,
    budget: BudgetCeiling | None = None,
) -> PreflightEstimate:
    if conservative_multiplier < 1:
        raise ValueError("conservative_multiplier must be at least 1")
    output_tokens = assumed_output_tokens or request.max_output_tokens
    if output_tokens < 1:
        raise ValueError("assumed_output_tokens must be positive")
    catalog = pricing_catalog or default_pricing_catalog()
    pricing = catalog.find(config.type, config.model)
    input_tokens = estimate_input_tokens(request)
    cost = estimate_cost_usd(
        pricing,
        input_tokens_per_call=input_tokens,
        output_tokens_per_call=output_tokens,
        logical_calls=calls.logical_calls,
    )
    bound = None if cost is None else cost * conservative_multiplier
    seconds_per_call = config.options.get("estimated_latency_seconds")
    if seconds_per_call is not None:
        seconds_per_call = float(seconds_per_call)
    load_seconds = float(config.options.get("estimated_load_seconds", 0.0))
    concurrency = 1 if config.type == "gemma_local" else config.request_concurrency
    runtime = estimate_runtime_seconds(
        config.type,
        logical_calls=calls.logical_calls,
        request_concurrency=concurrency,
        seconds_per_call=seconds_per_call,
        one_time_load_seconds=load_seconds,
    )
    warnings = [
        "Token counts use a deterministic regex estimate, not the provider tokenizer.",
        "Runtime is a rough planning estimate, not a service guarantee.",
    ]
    if pricing is None:
        warnings.append(
            "No versioned price is configured for this provider/model; cost is unknown."
        )
    elif config.type == "gemma_local":
        warnings.append("Local marginal cost excludes hardware and energy.")
    return PreflightEstimate(
        provider=config.type,
        model=config.model,
        tokenizer=TOKENIZER_NAME,
        estimated_input_tokens_per_call=input_tokens,
        assumed_output_tokens_per_call=output_tokens,
        logical_calls=calls.logical_calls,
        estimated_total_input_tokens=input_tokens * calls.logical_calls,
        estimated_total_output_tokens=output_tokens * calls.logical_calls,
        expected_cost_usd=cost,
        conservative_cost_bound_usd=bound,
        rough_runtime_seconds=runtime,
        pricing_catalog_version=catalog.version,
        pricing_source=None if pricing is None else pricing.source,
        budget_ceiling_usd=None if budget is None else budget.usd,
        within_budget=None if budget is None else budget.permits(bound),
        warnings=tuple(warnings),
    )
