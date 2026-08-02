"""Pure, accounting-unit-safe cost arithmetic."""

from __future__ import annotations

from mas_cc.llm_providers.pricing import ModelPricing, MonetaryAmount


def estimate_cost(
    pricing: ModelPricing | None,
    *,
    input_tokens_per_call: int,
    output_tokens_per_call: int,
    logical_calls: int,
    cached_input_tokens_per_call: int = 0,
    cache_creation_tokens_per_call: int = 0,
) -> MonetaryAmount | None:
    if pricing is None:
        return None
    per_call = pricing.cost(
        input_tokens_per_call,
        output_tokens_per_call,
        cached_input_tokens=cached_input_tokens_per_call,
        cache_creation_tokens=cache_creation_tokens_per_call,
    )
    if per_call is None:
        return None
    return MonetaryAmount(
        per_call.amount * logical_calls,
        per_call.unit,
        per_call.unit_source,
        per_call.provider,
        per_call.model,
        per_call.source,
        per_call.retrieved_at,
        per_call.version,
    )


def estimate_cost_usd(
    pricing: ModelPricing | None,
    *,
    input_tokens_per_call: int,
    output_tokens_per_call: int,
    logical_calls: int,
) -> float | None:
    """Phase 4 v1 compatibility helper; non-USD amounts remain unavailable."""

    result = estimate_cost(
        pricing,
        input_tokens_per_call=input_tokens_per_call,
        output_tokens_per_call=output_tokens_per_call,
        logical_calls=logical_calls,
    )
    return None if result is None or result.unit != "USD" else result.amount
