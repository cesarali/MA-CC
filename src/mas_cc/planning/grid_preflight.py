"""Price a whole grid (many cells, each an experiment) before launch.

Every cell shares one provider client, one pricing quote, and one budget
guard (``GridSpec`` forbids sweeping the fields that would break that), so
the combined budget check happens exactly once, against the sum across every
cell's own conservative demand — not once per cell against the full budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from mas_cc.config import GridSpec, LLMProviderConfig
from mas_cc.llm_providers import BudgetLimits, PricingQuote

from .experiment_preflight import (
    ExperimentPreflightEstimate,
    apply_total_demand_budget_check,
    static_experiment_preflight,
)
from .preflight import EstimateRange, MonetaryEstimateRange
from .runtime_estimation import estimate_runtime_seconds


def _add_ranges(values: Iterable[EstimateRange]) -> EstimateRange:
    lower = expected = conservative = 0
    for value in values:
        lower += value.lower
        expected += value.expected
        conservative += value.conservative
    return EstimateRange(lower, expected, conservative)


def _add_money(values: Iterable[Any]) -> Any:
    total = None
    for value in values:
        if value is None:
            return None
        if total is None:
            total = value
            continue
        if total.unit != value.unit:
            return None
        total = type(total)(
            amount=total.amount + value.amount, unit=total.unit, unit_source=total.unit_source,
            provider=total.provider, model=total.model, source=total.source,
            retrieved_at=total.retrieved_at, version=total.version,
        )
    return total


_STATUS_SEVERITY = {"permitted": 0, "explicit-override-required": 1, "denied": 2}


def _worst_status(statuses: Iterable[str]) -> str:
    return max(statuses, key=lambda status: _STATUS_SEVERITY.get(status, 2))


@dataclass(frozen=True, slots=True)
class GridPreflightEstimate:
    grid_id: str
    game_type: str
    provider: str
    model: str
    cell_count: int
    total_episode_count: int
    concurrency: int
    per_cell: tuple[ExperimentPreflightEstimate, ...]
    cell_overrides: tuple[dict[str, Any], ...]
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
            "grid_id": self.grid_id,
            "game_type": self.game_type,
            "provider": self.provider,
            "model": self.model,
            "cell_count": self.cell_count,
            "total_episode_count": self.total_episode_count,
            "concurrency": self.concurrency,
            "cells": [
                {"overrides": overrides, "estimate": estimate.to_dict()}
                for overrides, estimate in zip(self.cell_overrides, self.per_cell)
            ],
            "total_provider_requests": self.total_provider_requests.to_dict(),
            "total_input_tokens": self.total_input_tokens.to_dict(),
            "total_output_tokens": self.total_output_tokens.to_dict(),
            "total_costs": self.total_costs.to_dict(),
            "rough_runtime_seconds": self.rough_runtime_seconds,
            "launch_status": self.launch_status,
            "warnings": list(self.warnings),
        }


def static_grid_preflight(
    grid: GridSpec,
    *,
    concurrency: int | None = None,
    assumed_output_tokens: int | None = None,
    pricing_quote: PricingQuote | None = None,
    system_budget: BudgetLimits | None = None,
    run_budget: BudgetLimits | None = None,
    explicit_override: bool = False,
    allow_stale_pricing: bool = False,
) -> GridPreflightEstimate:
    """Price every cell independently (no per-cell budget check), then sum and check once."""

    from mas_cc.games import create_game  # local: avoids a games<->planning import cycle

    provider_config: LLMProviderConfig = grid.base.llm_provider
    game = create_game(grid.base.game)
    effective_concurrency = concurrency or grid.base.execution.parallelism

    cells = grid.cells
    per_cell: list[ExperimentPreflightEstimate] = []
    for cell in cells:
        plan = game.call_plan(cell.config.game)
        per_cell.append(
            static_experiment_preflight(
                plan, cell.config.prompt, cell.config.llm_provider,
                episode_count=cell.config.execution.repetitions,
                concurrency=cell.config.execution.parallelism,
                assumed_output_tokens=assumed_output_tokens or cell.config.llm_provider.max_output_tokens,
                pricing_quote=pricing_quote,
                # No budget here - see module docstring; checked once, combined, below.
                explicit_override=explicit_override, allow_stale_pricing=allow_stale_pricing,
            )
        )

    total_requests = _add_ranges(estimate.total_provider_requests for estimate in per_cell)
    total_inputs = _add_ranges(estimate.total_input_tokens for estimate in per_cell)
    total_outputs = _add_ranges(estimate.total_output_tokens for estimate in per_cell)
    total_costs = MonetaryEstimateRange(
        _add_money(estimate.total_costs.lower for estimate in per_cell),
        _add_money(estimate.total_costs.expected for estimate in per_cell),
        _add_money(estimate.total_costs.conservative for estimate in per_cell),
    )
    per_cell_status = _worst_status(estimate.launch_status for estimate in per_cell)
    warnings: list[str] = []
    for estimate in per_cell:
        warnings.extend(estimate.warnings)

    launch_status, warnings = apply_total_demand_budget_check(
        per_cell_status, warnings, total_costs, total_requests,
        system_budget=system_budget, run_budget=run_budget, explicit_override=explicit_override,
    )

    total_episode_count = sum(cell.config.execution.repetitions for cell in cells)
    rough_runtime = estimate_runtime_seconds(
        provider_config.type,
        logical_calls=total_requests.expected,
        request_concurrency=(1 if provider_config.type == "gemma_local" else effective_concurrency),
    )

    return GridPreflightEstimate(
        grid_id=grid.grid_id,
        game_type=game.spec.game_type,
        provider=provider_config.type,
        model=provider_config.model,
        cell_count=len(cells),
        total_episode_count=total_episode_count,
        concurrency=effective_concurrency,
        per_cell=tuple(per_cell),
        cell_overrides=tuple(dict(cell.overrides) for cell in cells),
        total_provider_requests=total_requests,
        total_input_tokens=total_inputs,
        total_output_tokens=total_outputs,
        total_costs=total_costs,
        rough_runtime_seconds=rough_runtime,
        launch_status=launch_status,
        warnings=tuple(dict.fromkeys(warnings)),
    )
