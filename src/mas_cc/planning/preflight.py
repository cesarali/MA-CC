"""Preflight composition without implicit credentials, clients, or network I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mas_cc.config import LLMProviderConfig
from mas_cc.llm_runtime.providers.budget import BudgetCeiling, BudgetLimits, resolve_budget_limits
from mas_cc.llm_runtime.providers.pricing import (
    MonetaryAmount,
    OfflinePricingSource,
    PricingCatalog,
    PricingQuote,
)

from .call_graph import LogicalCallSpec
from .cost_estimation import estimate_cost
from .runtime_estimation import estimate_runtime_seconds
from .token_estimation import TOKENIZER_NAME, estimate_input_tokens


@dataclass(frozen=True, slots=True)
class EstimateRange:
    lower: int
    expected: int
    conservative: int

    def to_dict(self) -> dict[str, int]:
        return {"lower": self.lower, "expected": self.expected, "conservative": self.conservative}


@dataclass(frozen=True, slots=True)
class MonetaryEstimateRange:
    lower: MonetaryAmount | None
    expected: MonetaryAmount | None
    conservative: MonetaryAmount | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": None if self.lower is None else self.lower.to_dict(),
            "expected": None if self.expected is None else self.expected.to_dict(),
            "conservative": None if self.conservative is None else self.conservative.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PreflightEstimate:
    provider: str
    model: str
    tokenizer: str
    estimated_input_tokens_per_call: int
    assumed_output_tokens_per_call: int
    logical_calls: int
    input_tokens: EstimateRange
    output_tokens: EstimateRange
    costs: MonetaryEstimateRange
    rough_runtime_seconds: float
    pricing_mode: str
    pricing_status: str
    pricing_source: str
    pricing_version: str
    pricing_retrieved_at: str
    pricing_fresh_until: str | None
    provider_account_budget: dict[str, Any] | None
    system_wide_limits: dict[str, Any] | None
    run_specific_limits: dict[str, Any] | None
    effective_limits: dict[str, Any] | None
    launch_status: str
    explicit_override: bool
    warnings: tuple[str, ...]

    @property
    def estimated_total_input_tokens(self) -> int:
        return self.input_tokens.expected

    @property
    def estimated_total_output_tokens(self) -> int:
        return self.output_tokens.expected

    @property
    def expected_cost_usd(self) -> float | None:
        cost = self.costs.expected
        return None if cost is None or cost.unit != "USD" else cost.amount

    @property
    def conservative_cost_bound_usd(self) -> float | None:
        cost = self.costs.conservative
        return None if cost is None or cost.unit != "USD" else cost.amount

    @property
    def pricing_catalog_version(self) -> str:
        return self.pricing_version

    @property
    def budget_ceiling_usd(self) -> float | None:
        if self.effective_limits is None:
            return None
        value = self.effective_limits.get("max_cost")
        return value.get("amount") if isinstance(value, dict) and value.get("unit") == "USD" else None

    @property
    def within_budget(self) -> bool | None:
        return None if self.effective_limits is None else self.launch_status == "permitted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "provider": self.provider,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "logical_calls": self.logical_calls,
            "estimated_input_tokens_per_call": self.estimated_input_tokens_per_call,
            "assumed_output_tokens_per_call": self.assumed_output_tokens_per_call,
            "token_estimates": {"input": self.input_tokens.to_dict(), "output": self.output_tokens.to_dict()},
            "cost_estimates": self.costs.to_dict(),
            "rough_runtime_seconds": self.rough_runtime_seconds,
            "pricing": {
                "mode": self.pricing_mode,
                "status": self.pricing_status,
                "source": self.pricing_source,
                "version": self.pricing_version,
                "retrieved_at": self.pricing_retrieved_at,
                "fresh_until": self.pricing_fresh_until,
            },
            "provider_account_budget": self.provider_account_budget,
            "system_wide_limits": self.system_wide_limits,
            "run_specific_limits": self.run_specific_limits,
            "effective_limits": self.effective_limits,
            "launch_status": self.launch_status,
            "explicit_override": self.explicit_override,
            "warnings": list(self.warnings),
            # Stable Phase 4 v1 fields. Values appear only for genuine USD quotes.
            "estimated_total_input_tokens": self.estimated_total_input_tokens,
            "estimated_total_output_tokens": self.estimated_total_output_tokens,
            "expected_cost_usd": self.expected_cost_usd,
            "conservative_cost_bound_usd": self.conservative_cost_bound_usd,
            "pricing_catalog_version": self.pricing_catalog_version,
            "pricing_source": self.pricing_source,
            "budget_ceiling_usd": self.budget_ceiling_usd,
            "within_budget": self.within_budget,
        }


def _limit_money(amount: float, quote: PricingQuote) -> MonetaryAmount:
    pricing = quote.pricing
    if pricing is None:
        unit, unit_source = "unknown", "price unavailable"
    else:
        unit, unit_source = pricing.unit, pricing.unit_source
    return MonetaryAmount(amount, unit, unit_source, quote.provider, quote.model,
                          quote.source, quote.retrieved_at, quote.version)


def _amount_exceeds(cost: MonetaryAmount | None, limit: MonetaryAmount | None) -> bool | None:
    if limit is None:
        return False
    if cost is None or cost.unit != limit.unit:
        return None
    return cost.amount > limit.amount


def _scale_cost(cost: MonetaryAmount | None, multiplier: float) -> MonetaryAmount | None:
    if cost is None:
        return None
    return MonetaryAmount(
        cost.amount * multiplier, cost.unit, cost.unit_source, cost.provider,
        cost.model, cost.source, cost.retrieved_at, cost.version,
    )


def _greater_cost(first: MonetaryAmount | None, second: MonetaryAmount | None) -> MonetaryAmount | None:
    if first is None:
        return second
    if second is None:
        return first
    if first.unit != second.unit:
        return None
    return first if first.amount >= second.amount else second


def static_preflight(
    request: Any,
    config: LLMProviderConfig,
    calls: LogicalCallSpec,
    *,
    assumed_output_tokens: int | None = None,
    cached_input_tokens_per_call: int = 0,
    cache_creation_tokens_per_call: int = 0,
    pricing_quote: PricingQuote | None = None,
    pricing_catalog: PricingCatalog | None = None,
    conservative_multiplier: float = 1.5,
    budget: BudgetCeiling | None = None,
    system_budget: BudgetLimits | None = None,
    run_budget: BudgetLimits | None = None,
    explicit_override: bool = False,
    allow_stale_pricing: bool = False,
) -> PreflightEstimate:
    if conservative_multiplier < 1:
        raise ValueError("conservative_multiplier must be at least 1")
    output_tokens = assumed_output_tokens or request.max_output_tokens
    if output_tokens < 1:
        raise ValueError("assumed_output_tokens must be positive")
    quote = pricing_quote or OfflinePricingSource(pricing_catalog).fetch(config.type, config.model)
    pricing = quote.pricing
    input_per_call = estimate_input_tokens(request)
    if cached_input_tokens_per_call + cache_creation_tokens_per_call > input_per_call:
        raise ValueError("cached/cache-creation tokens cannot exceed estimated input")

    input_range = EstimateRange(
        lower=input_per_call * calls.logical_calls,
        expected=input_per_call * calls.logical_calls,
        conservative=math.ceil(input_per_call * conservative_multiplier) * calls.logical_calls,
    )
    output_range = EstimateRange(
        lower=calls.logical_calls,
        expected=output_tokens * calls.logical_calls,
        conservative=max(output_tokens, request.max_output_tokens) * calls.logical_calls,
    )
    expected_cost = estimate_cost(
        pricing, input_tokens_per_call=input_per_call, output_tokens_per_call=output_tokens,
        logical_calls=calls.logical_calls,
        cached_input_tokens_per_call=cached_input_tokens_per_call,
        cache_creation_tokens_per_call=cache_creation_tokens_per_call,
    )
    uncached_cost = estimate_cost(
        pricing, input_tokens_per_call=input_per_call, output_tokens_per_call=output_tokens,
        logical_calls=calls.logical_calls,
    )
    costs = MonetaryEstimateRange(
        estimate_cost(pricing, input_tokens_per_call=input_per_call, output_tokens_per_call=1,
                      logical_calls=calls.logical_calls,
                      cached_input_tokens_per_call=cached_input_tokens_per_call,
                      cache_creation_tokens_per_call=cache_creation_tokens_per_call),
        expected_cost,
        _greater_cost(
            _scale_cost(expected_cost, conservative_multiplier),
            _scale_cost(uncached_cost, conservative_multiplier),
        ),
    )

    if budget is not None:
        legacy = BudgetLimits(max_cost=_limit_money(budget.usd, quote))
        if system_budget is not None or run_budget is not None:
            raise ValueError("legacy budget cannot be combined with typed system/run budgets")
        system_budget = legacy
    effective = resolve_budget_limits(system_budget, run_budget) if system_budget is not None else run_budget

    warnings = [
        "Token counts use a deterministic regex estimate, not the provider tokenizer.",
        "Runtime is a rough planning estimate, not a service guarantee.",
    ]
    pricing_status = quote.status
    if quote.is_fresh is False and pricing_status not in {"unavailable", "unit-unknown"}:
        pricing_status = "stale"
    if pricing is not None and pricing.unit == "unknown":
        pricing_status = "unit-unknown"
    elif expected_cost is None and costs.conservative is not None and quote.status == "known":
        pricing_status = "partial"
    launch_status = "permitted"
    unsafe_status = pricing_status in {"unavailable", "unit-unknown"} or (
        pricing_status == "stale" and not allow_stale_pricing
    )
    if unsafe_status or costs.conservative is None:
        launch_status = "permitted" if explicit_override else "explicit-override-required"
        warnings.append(f"Pricing is {pricing_status}; a paid launch fails closed without an explicit override.")
    elif pricing_status == "partial":
        warnings.append("Expected cost is partial; the conservative bound prices all input as uncached.")
    elif pricing_status == "stale":
        warnings.append("Stale pricing is being used under the explicit freshness policy.")
    if pricing is not None and pricing.modality != "chat":
        launch_status = "denied"
        warnings.append(f"Model modality {pricing.modality!r} is not supported for chat requests.")
    if pricing is not None:
        limits = pricing.limits
        if limits.maximum_input_tokens is not None and input_range.conservative // calls.logical_calls > limits.maximum_input_tokens:
            launch_status = "denied"
            warnings.append("Conservative per-call input exceeds the provider maximum.")
        if limits.maximum_output_tokens is not None and output_range.conservative // calls.logical_calls > limits.maximum_output_tokens:
            launch_status = "denied"
            warnings.append("Conservative per-call output exceeds the provider maximum.")
    if effective is not None:
        exceeded = _amount_exceeds(costs.conservative, effective.max_cost)
        if exceeded is True or (effective.max_requests is not None and calls.logical_calls > effective.max_requests) or \
                (effective.max_input_tokens is not None and input_range.conservative > effective.max_input_tokens) or \
                (effective.max_output_tokens is not None and output_range.conservative > effective.max_output_tokens):
            launch_status = "denied"
            warnings.append("Conservative demand exceeds an effective MAS-CC per-run limit.")
        elif exceeded is None and not explicit_override:
            launch_status = "explicit-override-required"
            warnings.append("Cost and approved limit cannot be compared safely in one accounting unit.")
        if explicit_override and not effective.allow_unbounded_paid_requests and costs.conservative is None:
            launch_status = "explicit-override-required"
            warnings.append("The runtime budget policy does not authorize unbounded paid requests.")
    if quote.account_budget is not None and quote.account_budget.remaining is not None:
        exceeded = _amount_exceeds(costs.conservative, quote.account_budget.remaining)
        if exceeded is True:
            launch_status = "denied"
            warnings.append("Conservative cost exceeds the separate provider account balance.")
        elif exceeded is None and not explicit_override:
            launch_status = "explicit-override-required"

    seconds_per_call = config.options.get("estimated_latency_seconds")
    seconds_per_call = None if seconds_per_call is None else float(seconds_per_call)
    concurrency = 1 if config.type == "gemma_local" else config.request_concurrency
    runtime = estimate_runtime_seconds(
        config.type, logical_calls=calls.logical_calls, request_concurrency=concurrency,
        seconds_per_call=seconds_per_call,
        one_time_load_seconds=float(config.options.get("estimated_load_seconds", 0.0)),
    )
    return PreflightEstimate(
        config.type, config.model, TOKENIZER_NAME, input_per_call, output_tokens,
        calls.logical_calls, input_range, output_range, costs, runtime,
        quote.mode, pricing_status, quote.source, quote.version, quote.retrieved_at,
        quote.fresh_until,
        None if quote.account_budget is None else quote.account_budget.to_dict(),
        None if system_budget is None else system_budget.to_dict(),
        None if run_budget is None else run_budget.to_dict(),
        None if effective is None else effective.to_dict(),
        launch_status, explicit_override, tuple(warnings),
    )
