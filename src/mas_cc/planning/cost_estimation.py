"""Pure cost arithmetic for static preflight."""

from __future__ import annotations

from mas_cc.llm_providers.pricing import ModelPricing


def estimate_cost_usd(
    pricing: ModelPricing | None,
    *,
    input_tokens_per_call: int,
    output_tokens_per_call: int,
    logical_calls: int,
) -> float | None:
    if pricing is None:
        return None
    return pricing.cost(
        input_tokens_per_call * logical_calls,
        output_tokens_per_call * logical_calls,
    )
