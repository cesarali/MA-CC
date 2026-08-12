"""Preflight limits and an atomic runtime budget guard."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ProviderError
from .pricing import ModelPricing, MonetaryAmount
from .requests import CompletionRequest
from .responses import CompletionResponse

LOGGER = logging.getLogger("mas_cc.budget")

# Every denial the guard can raise. A run that sees one of these can never make
# progress again — the counters only ever move one way — so callers treat them
# as "stop the whole run", not as one episode's bad luck.
BUDGET_STOP_CODES = frozenset(
    {"budget_exhausted", "budget_stopped", "budget_unbounded", "budget_unit_mismatch"}
)


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
class BudgetExpectation:
    """What preflight predicted this run would consume.

    Advisory only: the guard never denies against these numbers, it warns the
    first time each is passed. Token and request counts are estimates built
    from a representative prompt, so a run whose prompts grow with history
    (`memory_size: 0`) legitimately blows past them — that is a signal worth
    printing, never a reason to kill a run that is still inside its *cost*
    ceiling.
    """

    requests: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    reservation_id: str
    cost: MonetaryAmount | None
    requests: int
    input_tokens: int
    output_tokens: int


class RuntimeBudgetGuard:
    """Atomically reserves conservative request resources before dispatch."""

    def __init__(
        self, limits: BudgetLimits, *, expectation: BudgetExpectation | None = None
    ) -> None:
        self.limits = limits
        self.expectation = expectation or BudgetExpectation()
        self._lock = threading.RLock()
        self._reservations: dict[str, BudgetReservation] = {}
        self._cost = 0.0
        self._requests = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._stops = 0
        self._stop_reason: str | None = None
        self._warned: set[str] = set()
        self._account_spend: dict[str, Any] | None = None
        self._durable_state_sink: Any | None = None

    def set_durable_state_sink(self, sink: Any | None) -> None:
        """Install an atomic state callback and immediately publish current totals."""

        with self._lock:
            self._durable_state_sink = sink
            self._persist_locked()

    def _persist_locked(self) -> None:
        if self._durable_state_sink is not None:
            self._durable_state_sink(self.durable_state())

    def request_stop(self, reason: str) -> None:
        """Refuse every subsequent reservation, for a reason set outside.

        This is how an out-of-band signal — chiefly the provider's own reported
        account spend — stops a run. It is deliberately one-way and idempotent:
        the first reason is the one that gets reported.
        """

        if not reason:
            raise ValueError("a budget stop must carry a reason")
        with self._lock:
            if self._stop_reason is None:
                self._stop_reason = reason
                LOGGER.warning("runtime budget stop requested: %s", reason)
                self._persist_locked()

    @property
    def stop_reason(self) -> str | None:
        with self._lock:
            return self._stop_reason

    def record_account_spend(self, value: Mapping[str, Any] | None) -> None:
        """Record the provider's latest self-reported spend for the run record."""

        with self._lock:
            self._account_spend = None if value is None else dict(value)
            self._persist_locked()

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
            if self._stop_reason is not None:
                self._stops += 1
                raise ProviderError(
                    f"Runtime budget stop is in effect: {self._stop_reason}",
                    provider="budget_guard", code="budget_stopped",
                )
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
            self._check_expectations()
            self._persist_locked()
            return reservation

    def _check_integer_limit(self, resource: str, proposed: int, limit: int | None) -> None:
        if limit is not None and proposed > limit:
            self._deny(resource)

    def _check_expectations(self) -> None:
        """Warn once per resource that has outgrown what preflight predicted.

        Called with the lock held. This is the whole of the advisory path: it
        never raises, so an underestimated prompt shows up as a line in the log
        while the run keeps going, and the cost ceiling stays the only stop.
        """

        for resource, used, expected in (
            ("request", self._requests, self.expectation.requests),
            ("input-token", self._input_tokens, self.expectation.input_tokens),
            ("output-token", self._output_tokens, self.expectation.output_tokens),
        ):
            if expected is None or resource in self._warned or used <= expected:
                continue
            self._warned.add(resource)
            LOGGER.warning(
                "%s usage passed the preflight estimate (%s used vs %s estimated); "
                "this is advisory - the run continues until its cost ceiling.",
                resource, f"{used:,}", f"{expected:,}",
            )

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
            self._persist_locked()

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
            self._persist_locked()

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
                "stop_reason": self._stop_reason,
                "preflight_expectation": self.expectation.to_dict(),
                "provider_account_spend": self._account_spend,
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

    def durable_state(self) -> dict[str, Any]:
        """Compact monotone state: committed totals plus dead-process ceilings."""

        with self._lock:
            reserved_cost = sum(
                0.0 if item.cost is None else item.cost.amount
                for item in self._reservations.values()
            )
            reserved_requests = sum(item.requests for item in self._reservations.values())
            reserved_input = sum(item.input_tokens for item in self._reservations.values())
            reserved_output = sum(item.output_tokens for item in self._reservations.values())
            unit = None if self.limits.max_cost is None else self.limits.max_cost.unit
            return {
                "schema_version": 1,
                "approved_limits": self.limits.to_dict(),
                "committed_requests": max(0, self._requests - reserved_requests),
                "committed_input_tokens": max(0, self._input_tokens - reserved_input),
                "committed_output_tokens": max(0, self._output_tokens - reserved_output),
                "committed_cost": {
                    "amount": max(0.0, self._cost - reserved_cost),
                    "unit": unit,
                },
                "outstanding_reservation_ceilings": {
                    "requests": reserved_requests,
                    "input_tokens": reserved_input,
                    "output_tokens": reserved_output,
                    "cost": {"amount": reserved_cost, "unit": unit},
                },
                "stop_count": self._stops,
                "stop_reason": self._stop_reason,
                "provider_account_spend": self._account_spend,
            }

    def restore_durable_state(
        self,
        value: Mapping[str, Any],
        *,
        authoritative_cost: MonetaryAmount | None = None,
    ) -> None:
        """Restore without reviving reservations, conservatively charging uncertainty."""

        if value.get("schema_version") != 1:
            raise ValueError("unsupported durable budget schema version")
        if value.get("approved_limits") != self.limits.to_dict():
            raise ValueError("durable budget limits do not match the active run")
        outstanding = value.get("outstanding_reservation_ceilings", {})
        committed_cost = float((value.get("committed_cost") or {}).get("amount", 0.0))
        uncertain_cost = float((outstanding.get("cost") or {}).get("amount", 0.0))
        restored_cost = committed_cost + uncertain_cost
        if authoritative_cost is not None:
            if self.limits.max_cost is not None and authoritative_cost.unit != self.limits.max_cost.unit:
                raise ValueError("authoritative spend and budget use different accounting units")
            restored_cost = max(restored_cost, authoritative_cost.amount)
        with self._lock:
            self._cost = restored_cost
            self._requests = int(value.get("committed_requests", 0)) + int(
                outstanding.get("requests", 0)
            )
            self._input_tokens = int(value.get("committed_input_tokens", 0)) + int(
                outstanding.get("input_tokens", 0)
            )
            self._output_tokens = int(value.get("committed_output_tokens", 0)) + int(
                outstanding.get("output_tokens", 0)
            )
            self._reservations = {}
            self._stops = int(value.get("stop_count", 0))
            self._stop_reason = value.get("stop_reason")
            account = value.get("provider_account_spend")
            self._account_spend = dict(account) if isinstance(account, Mapping) else None

    def restore_checkpoint_state(self, value: dict[str, Any]) -> None:
        """Restore legacy round-checkpoint state with compatible limits."""

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


class AtomicBudgetStateStore:
    """One fsync-and-replace budget checkpoint shared by concurrent episodes."""

    def __init__(
        self,
        path: str | Path,
        *,
        resolved_budget_hash: str,
        pricing_snapshot_hash: str,
    ) -> None:
        self.path = Path(path)
        self.resolved_budget_hash = resolved_budget_hash
        self.pricing_snapshot_hash = pricing_snapshot_hash
        self._lock = threading.Lock()

    def write(self, value: Mapping[str, Any]) -> None:
        payload = {
            **dict(value),
            "resolved_budget_hash": self.resolved_budget_hash,
            "pricing_snapshot_hash": self.pricing_snapshot_hash,
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        with self._lock:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            try:
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass

    def load(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if value.get("resolved_budget_hash") != self.resolved_budget_hash:
            raise ValueError("budget checkpoint resolved-budget hash does not match this run")
        if value.get("pricing_snapshot_hash") != self.pricing_snapshot_hash:
            raise ValueError("budget checkpoint pricing identity does not match this run")
        return value

    def restore(
        self,
        guard: RuntimeBudgetGuard,
        *,
        authoritative_cost: MonetaryAmount | None = None,
    ) -> bool:
        value = self.load()
        if value is None:
            return False
        guard.restore_durable_state(value, authoritative_cost=authoritative_cost)
        return True

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
