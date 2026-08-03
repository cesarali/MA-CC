"""Preflight limits and an atomic runtime budget guard."""

from __future__ import annotations

import asyncio
import math
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from mas_cc.core.exceptions import ProviderError

from .pricing import ModelPricing, MonetaryAmount
from .requests import CompletionRequest
from .responses import CompletionResponse


@dataclass(frozen=True, slots=True)
class BudgetCeiling:
    """Compatibility wrapper for the original Phase 4 USD-only API."""

    usd: float

    def __post_init__(self) -> None:
        if self.usd < 0:
            raise ValueError("budget ceiling cannot be negative")

    def permits(self, conservative_cost_usd: float | None) -> bool | None:
        return None if conservative_cost_usd is None else conservative_cost_usd <= self.usd


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    max_cost: MonetaryAmount | None = None
    max_requests: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    allow_unbounded_paid_requests: bool = False

    def __post_init__(self) -> None:
        for name in ("max_requests", "max_input_tokens", "max_output_tokens"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_cost": None if self.max_cost is None else self.max_cost.to_dict(),
            "max_requests": self.max_requests,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "allow_unbounded_paid_requests": self.allow_unbounded_paid_requests,
        }


def resolve_budget_limits(system: BudgetLimits, run: BudgetLimits | None) -> BudgetLimits:
    """Apply a run limit only when it is at least as strict as the system limit."""

    if run is None:
        return system

    def lower(system_value: int | None, run_value: int | None, name: str) -> int | None:
        if run_value is None:
            return system_value
        if system_value is not None and run_value > system_value:
            raise ValueError(f"run-specific {name} cannot raise the system-wide limit")
        return run_value

    cost = run.max_cost or system.max_cost
    if run.max_cost is not None and system.max_cost is not None:
        if run.max_cost.unit != system.max_cost.unit:
            raise ValueError("run-specific and system cost limits must use the same accounting unit")
        if run.max_cost.amount > system.max_cost.amount:
            raise ValueError("run-specific max_cost cannot raise the system-wide limit")
    if run.allow_unbounded_paid_requests and not system.allow_unbounded_paid_requests:
        raise ValueError("run-specific override cannot weaken the system-wide unknown-price policy")
    return BudgetLimits(
        max_cost=cost,
        max_requests=lower(system.max_requests, run.max_requests, "max_requests"),
        max_input_tokens=lower(system.max_input_tokens, run.max_input_tokens, "max_input_tokens"),
        max_output_tokens=lower(system.max_output_tokens, run.max_output_tokens, "max_output_tokens"),
        allow_unbounded_paid_requests=system.allow_unbounded_paid_requests and run.allow_unbounded_paid_requests,
    )


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    reservation_id: str
    cost: MonetaryAmount | None
    requests: int
    input_tokens: int
    output_tokens: int


class RuntimeBudgetGuard:
    """Atomically reserves conservative request resources before dispatch."""

    def __init__(self, limits: BudgetLimits) -> None:
        self.limits = limits
        self._lock = threading.RLock()
        self._reservations: dict[str, BudgetReservation] = {}
        self._cost = 0.0
        self._requests = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._stops = 0

    def reserve(
        self,
        *,
        conservative_cost: MonetaryAmount | None,
        input_tokens: int,
        output_tokens: int,
    ) -> BudgetReservation:
        if min(input_tokens, output_tokens) < 0:
            raise ValueError("reservation token counts cannot be negative")
        with self._lock:
            if conservative_cost is None and not self.limits.allow_unbounded_paid_requests:
                self._stops += 1
                raise ProviderError(
                    "Paid request cost cannot be bounded under the approved budget.",
                    provider="budget_guard", code="budget_unbounded",
                )
            if conservative_cost is not None and self.limits.max_cost is not None:
                if conservative_cost.unit != self.limits.max_cost.unit:
                    self._stops += 1
                    raise ProviderError(
                        "Request estimate and budget use different accounting units.",
                        provider="budget_guard", code="budget_unit_mismatch",
                    )
                if self._cost + conservative_cost.amount > self.limits.max_cost.amount:
                    self._deny("cost")
            self._check_integer_limit("request", self._requests + 1, self.limits.max_requests)
            self._check_integer_limit("input-token", self._input_tokens + input_tokens, self.limits.max_input_tokens)
            self._check_integer_limit("output-token", self._output_tokens + output_tokens, self.limits.max_output_tokens)
            reservation = BudgetReservation(
                str(uuid.uuid4()), conservative_cost, 1, input_tokens, output_tokens
            )
            self._reservations[reservation.reservation_id] = reservation
            self._cost += 0.0 if conservative_cost is None else conservative_cost.amount
            self._requests += 1
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            return reservation

    def _check_integer_limit(self, resource: str, proposed: int, limit: int | None) -> None:
        if limit is not None and proposed > limit:
            self._deny(resource)

    def _deny(self, resource: str) -> None:
        self._stops += 1
        raise ProviderError(
            f"Approved runtime {resource} budget would be exceeded.",
            provider="budget_guard", code="budget_exhausted",
        )

    def release(self, reservation: BudgetReservation) -> None:
        with self._lock:
            current = self._reservations.pop(reservation.reservation_id, None)
            if current is None:
                raise ValueError("unknown or already reconciled reservation")
            self._subtract(current)

    def reconcile(
        self,
        reservation: BudgetReservation,
        *,
        actual_cost: MonetaryAmount | None,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Replace one reservation with normalized actual usage atomically.

        Unknown actual cost keeps the conservative reservation, while known
        token usage replaces reserved token bounds.
        """

        with self._lock:
            current = self._reservations.get(reservation.reservation_id)
            if current is None:
                raise ValueError("unknown or already reconciled reservation")
            retained_cost = actual_cost or current.cost
            if (
                retained_cost is not None
                and self.limits.max_cost is not None
                and retained_cost.unit != self.limits.max_cost.unit
            ):
                raise ValueError("actual cost and budget use different accounting units")
            self._reservations.pop(reservation.reservation_id)
            self._subtract(current)
            self._cost += 0.0 if retained_cost is None else retained_cost.amount
            self._requests += 1
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens

    def _subtract(self, reservation: BudgetReservation) -> None:
        self._cost -= 0.0 if reservation.cost is None else reservation.cost.amount
        self._requests -= reservation.requests
        self._input_tokens -= reservation.input_tokens
        self._output_tokens -= reservation.output_tokens

    def status(self) -> dict[str, Any]:
        with self._lock:
            unit = None if self.limits.max_cost is None else self.limits.max_cost.unit
            return {
                "schema_version": 2,
                "approved_limits": self.limits.to_dict(),
                "used_and_reserved": {
                    "cost": None if unit is None else {"amount": max(0.0, self._cost), "unit": unit},
                    "requests": self._requests,
                    "input_tokens": self._input_tokens,
                    "output_tokens": self._output_tokens,
                },
                "active_reservations": len(self._reservations),
                "stop_count": self._stops,
            }

    def checkpoint_state(self) -> dict[str, Any]:
        """Return a JSON-safe state for an atomic checkpoint.

        Reservations are retained because a cancelled request may have been
        dispatched and must remain conservatively charged after resume.
        """
        with self._lock:
            return {
                "schema_version": 1,
                "status": self.status(),
                "reservations": [
                    {
                        "reservation_id": item.reservation_id,
                        "cost": None if item.cost is None else item.cost.to_dict(),
                        "requests": item.requests,
                        "input_tokens": item.input_tokens,
                        "output_tokens": item.output_tokens,
                    }
                    for item in self._reservations.values()
                ],
            }

    def restore_checkpoint_state(self, value: dict[str, Any]) -> None:
        """Restore counters only when the checkpoint has compatible limits."""
        if value.get("schema_version") != 1:
            raise ValueError("unsupported budget checkpoint schema version")
        status = value.get("status")
        if not isinstance(status, dict) or status.get("approved_limits") != self.limits.to_dict():
            raise ValueError("budget checkpoint limits do not match the active run")
        used = status.get("used_and_reserved", {})
        with self._lock:
            self._cost = float((used.get("cost") or {}).get("amount", 0.0))
            self._requests = int(used.get("requests", 0))
            self._input_tokens = int(used.get("input_tokens", 0))
            self._output_tokens = int(used.get("output_tokens", 0))
            self._stops = int(status.get("stop_count", 0))
            restored: dict[str, BudgetReservation] = {}
            for item in value.get("reservations", []):
                if not isinstance(item, dict):
                    raise ValueError("budget checkpoint reservations must be mappings")
                raw_cost = item.get("cost")
                cost = None if raw_cost is None else MonetaryAmount(**raw_cost)
                reservation = BudgetReservation(
                    reservation_id=str(item["reservation_id"]), cost=cost,
                    requests=int(item["requests"]), input_tokens=int(item["input_tokens"]),
                    output_tokens=int(item["output_tokens"]),
                )
                restored[reservation.reservation_id] = reservation
            self._reservations = restored


class BudgetGuardedProvider:
    """Provider decorator preserving the normalized completion interface."""

    def __init__(self, provider: Any, guard: RuntimeBudgetGuard, pricing: ModelPricing | None,
                 *, input_token_estimator: Any, input_token_multiplier: float = 1.5) -> None:
        if input_token_multiplier < 1:
            raise ValueError("input_token_multiplier must be at least 1")
        self._provider = provider
        self._guard = guard
        self._pricing = pricing
        self._estimate_input = input_token_estimator
        self._input_token_multiplier = input_token_multiplier
        self.name = provider.name
        self.model = provider.model
        self.capabilities = provider.capabilities

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        input_tokens = math.ceil(self._estimate_input(request) * self._input_token_multiplier)
        cost = None if self._pricing is None else self._pricing.cost(input_tokens, request.max_output_tokens)
        reservation = self._guard.reserve(
            conservative_cost=cost,
            input_tokens=input_tokens,
            output_tokens=request.max_output_tokens,
        )
        try:
            response = await self._provider.complete(request)
        except asyncio.CancelledError:
            # Dispatch may already have happened; retain the conservative reservation.
            raise
        except Exception:
            # Fail closed because transport failures can happen after provider billing.
            raise
        actual_input_tokens = (
            input_tokens
            if response.usage.input_tokens is None
            else response.usage.input_tokens
        )
        actual_output_tokens = (
            request.max_output_tokens
            if response.usage.output_tokens is None
            else response.usage.output_tokens
        )
        actual = None
        if (
            self._pricing is not None
            and response.usage.input_tokens is not None
            and response.usage.output_tokens is not None
        ):
            actual = self._pricing.cost(
                response.usage.input_tokens,
                response.usage.output_tokens,
                cached_input_tokens=response.usage.cached_input_tokens or 0,
            )
        self._guard.reconcile(
            reservation,
            actual_cost=actual,
            input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens,
        )
        return response

    def close(self) -> None:
        self._provider.close()
