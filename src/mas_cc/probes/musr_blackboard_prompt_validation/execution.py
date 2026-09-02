"""Resumable provider execution for frozen blackboard prompt states."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mas_cc.core import Seed
from mas_cc.games import create_game
from mas_cc.llm_runtime.providers import (
    BudgetExpectation,
    BudgetGuardedProvider,
    BudgetLimits,
    CompletionRequest,
    MonetaryAmount,
    ProviderLoadControlConfig,
    RuntimeBudgetGuard,
    SharedProviderCoordinator,
    UniversityPricingSource,
    create_llm_provider,
)
from mas_cc.llm_runtime.prompts import RegexTokenCounter
from mas_cc.runtime.loop_runtime import (
    DecisionLoopExhausted,
    ValidationAttempt,
    run_validated_decision,
)

from .config import BlackboardValidationConfig
from .states import FrozenState


class _TierGate:
    """Process-local concurrency gate with the required exact fallback tiers."""

    def __init__(self, tiers: Sequence[int]) -> None:
        self.tiers = tuple(tiers)
        self.index = 0
        self.active = 0
        self.peak = 0
        self._condition = asyncio.Condition()

    async def acquire(self) -> None:
        async with self._condition:
            while self.active >= self.tiers[self.index]:
                await self._condition.wait()
            self.active += 1
            self.peak = max(self.peak, self.active)

    async def release(self, *, unstable: bool) -> None:
        async with self._condition:
            self.active -= 1
            if unstable and self.index < len(self.tiers) - 1:
                self.index += 1
            self._condition.notify_all()

    @property
    def limit(self) -> int:
        return self.tiers[self.index]


def append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return ()
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def terminal(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["call_id"]): row
        for row in read(path)
        if row.get("event") in {"call_finished", "call_failed"}
    }


@dataclass(frozen=True, slots=True)
class ValidationCall:
    call_id: str
    state_key: str
    repetition: int
    provider_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "state_key": self.state_key,
            "repetition": self.repetition,
            "provider_seed": self.provider_seed,
        }


def call_plan(
    config: BlackboardValidationConfig, states: Mapping[str, FrozenState]
) -> tuple[ValidationCall, ...]:
    root = Seed(config.state_seed)
    calls = []
    for key in sorted(states):
        for repetition in range(config.repetitions):
            calls.append(
                ValidationCall(
                    call_id=f"{config.mode}:{key}:{repetition:02d}",
                    state_key=key,
                    repetition=repetition,
                    provider_seed=int(root.derive(f"provider:{key}:{repetition}")),
                )
            )
    if len(calls) != config.logical_calls or len(
        {call.call_id for call in calls}
    ) != len(calls):
        raise RuntimeError(
            "blackboard validation call plan violates its count or identity contract"
        )
    return tuple(calls)


async def execute(
    config: BlackboardValidationConfig,
    states: Mapping[str, FrozenState],
    calls: Sequence[ValidationCall],
    journal: Path,
    coordinator_root: Path,
) -> dict[str, Any]:
    prior = terminal(journal)
    outstanding = [
        call
        for call in calls
        if call.call_id not in prior
        or prior[call.call_id].get("event") == "call_failed"
    ]
    quote = UniversityPricingSource(config.provider).fetch(
        config.provider.type, config.provider.model
    )
    if quote.status != "known" or quote.pricing is None:
        raise RuntimeError(f"live pricing does not permit launch: {quote.status}")
    limit = MonetaryAmount(
        config.max_cost,
        config.accounting_unit,
        "blackboard validation config",
        config.provider.type,
        config.provider.model,
        "MuSR blackboard prompt validation",
        quote.retrieved_at,
        "blackboard-validation-v1",
    )
    guard = RuntimeBudgetGuard(
        BudgetLimits(
            max_cost=limit,
            max_requests=config.max_provider_requests,
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=config.max_output_tokens_total,
        ),
        expectation=BudgetExpectation(
            len(calls),
            sum(
                states[call.state_key].compiled_prompt.total_token_estimate or 0
                for call in calls
            ),
            len(calls) * config.provider.max_output_tokens,
        ),
    )
    load_config = ProviderLoadControlConfig(
        initial_concurrency=config.max_concurrency,
        minimum_concurrency=config.fallback_concurrency[-1],
        maximum_concurrency=config.max_concurrency,
        target_rpm=config.max_rpm,
        decrease_factor=2 / 3,
        increase_step=10,
    )
    coordinator = SharedProviderCoordinator(coordinator_root, load_config)
    raw = create_llm_provider(config.provider, request_coordinator=coordinator)
    counter = RegexTokenCounter()
    provider = BudgetGuardedProvider(
        raw,
        guard,
        quote.pricing,
        input_token_estimator=lambda request: sum(
            counter.count_tokens(message.content) for message in request.messages
        ),
    )
    tier_gate = _TierGate(config.fallback_concurrency)
    telemetry_lock = asyncio.Lock()
    active = 0
    peak = 0
    dispatches: list[float] = []
    started = time.monotonic()

    async def one(call: ValidationCall) -> None:
        nonlocal active, peak
        frozen = states[call.state_key]
        definition = frozen.definition
        game_config = config.game_config(str(definition["task_id"]))
        game = create_game(game_config)
        base = {
            **call.to_dict(),
            "task_id": definition["task_id"],
            "agent_id": definition["agent_id"],
            "state_id": definition["state_id"],
            "current_vote": definition["current_vote"],
            "original_evidence_ids": definition["original_evidence_ids"],
            "acquired_evidence_ids": definition["acquired_evidence_ids"],
            "total_evidence_ids": definition["total_evidence_ids"],
            "latent_values_covered": definition["latent_values_covered"],
            "latent_coverage_count": definition["latent_coverage_count"],
            "sampled_message_ids": definition["sampled_message_ids"],
            "sampled_message_types": definition["sampled_message_types"],
            "sampled_shared_fact_ids": definition["sampled_shared_fact_ids"],
            "reply_to_structure": definition["reply_to_structure"],
            "state_sha256": definition["state_sha256"],
            "prompt_family": frozen.compiled_prompt.family,
            "prompt_version": frozen.compiled_prompt.version,
            "prompt_definition_hash": frozen.compiled_prompt.definition_hash,
            "prompt_instance_hash": frozen.compiled_prompt.instance_hash,
            "messages": [
                message.to_dict() for message in frozen.compiled_prompt.messages
            ],
            "provider": config.provider.type,
            "model": config.provider.model,
            "temperature": config.provider.temperature,
            "max_output_tokens": config.provider.max_output_tokens,
        }
        append(
            journal,
            {
                "event": "request_started",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **base,
            },
        )
        attempts: list[dict[str, Any]] = []

        def on_attempt(attempt: ValidationAttempt) -> None:
            attempts.append(
                {
                    "attempt": attempt.attempt,
                    "provider_seed": attempt.request.seed,
                    "metadata": dict(attempt.request.metadata),
                    "raw_response": None
                    if attempt.response is None
                    else attempt.response.content,
                    "response": None
                    if attempt.response is None
                    else attempt.response.to_dict(),
                    "redacted_provider_response": None
                    if attempt.response is None
                    else attempt.response.redacted_raw_response(),
                    "action": None
                    if attempt.action is None
                    else attempt.action.to_dict(),
                    "validation_error": attempt.validation_error,
                    "provider_error": attempt.provider_error,
                    "validation_issues": [
                        {
                            "field": issue.field,
                            "message": issue.message,
                            "invalid_value": issue.invalid_value,
                        }
                        for issue in attempt.validation_issues
                    ],
                }
            )

        try:
            await tier_gate.acquire()
            async with telemetry_lock:
                active += 1
                peak = max(peak, active)
                dispatches.append(time.monotonic())
            try:
                result = await run_validated_decision(
                    game=game,
                    state=frozen.state,
                    request=frozen.request,
                    game_config=game_config,
                    provider=provider,
                    prompt=frozen.compiled_prompt,
                    temperature=config.provider.temperature,
                    max_output_tokens=config.provider.max_output_tokens,
                    seed_for_attempt=lambda _attempt: call.provider_seed,
                    metadata_for_attempt=lambda attempt: {
                        "probe": "musr_blackboard_prompt_validation",
                        "call_id": call.call_id,
                        "state_id": definition["state_id"],
                        "attempt": attempt,
                    },
                    on_attempt=on_attempt,
                )
            except DecisionLoopExhausted:
                await tier_gate.release(unstable=False)
                raise
            except Exception:
                await tier_gate.release(unstable=True)
                raise
            else:
                await tier_gate.release(unstable=False)
            finally:
                async with telemetry_lock:
                    active -= 1
            final_attempt = result.attempts[-1]
            response = final_attempt.response
            assert response is not None
            action = result.action
            append(
                journal,
                {
                    "event": "call_finished",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **base,
                    "attempts": attempts,
                    "validation_attempts": len(attempts),
                    "raw_response": response.content,
                    "response": response.to_dict(),
                    "redacted_provider_response": response.redacted_raw_response(),
                    "parsed_action": action.to_dict(),
                    "parsed_semantic_answer": action.value,
                    "correct": action.value == frozen.state.correct_answer,
                    "parse_success": True,
                    "usage": response.usage.to_dict(),
                    "request_id": response.request_id,
                    "latency_seconds": response.latency_seconds,
                    "transport_retries": response.retries,
                },
            )
        except Exception as exc:
            append(
                journal,
                {
                    "event": "call_failed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **base,
                    "attempts": attempts,
                    "validation_attempts": len(attempts),
                    "parse_success": False,
                    "provider_error": f"{type(exc).__name__}: {exc}",
                },
            )

    try:
        await asyncio.gather(*(one(call) for call in outstanding))
    finally:
        provider.close()
    elapsed = time.monotonic() - started
    final = terminal(journal)
    expected = {call.call_id for call in calls}
    timestamps = sorted(dispatches)
    observed_rpm = 0
    left = 0
    for right, timestamp in enumerate(timestamps):
        while timestamp - timestamps[left] >= 60:
            left += 1
        observed_rpm = max(observed_rpm, right - left + 1)
    coordinator_snapshot = coordinator.snapshot()
    return {
        "scheduled": len(calls),
        "previously_terminal": len(prior),
        "attempted_now": len(outstanding),
        "terminal": len(expected.intersection(final)),
        "successful": sum(
            final.get(call_id, {}).get("parse_success") is True for call_id in expected
        ),
        "failed": sum(
            final.get(call_id, {}).get("event") == "call_failed" for call_id in expected
        ),
        "configured_local_workers": config.local_workers,
        "configured_max_concurrency": config.max_concurrency,
        "configured_provider_concurrency": config.provider.request_concurrency,
        "configured_max_rpm": config.max_rpm,
        "fallback_concurrency": list(config.fallback_concurrency),
        "observed_peak_concurrency": peak,
        "final_fallback_concurrency": tier_gate.limit,
        "observed_peak_rolling_60s_dispatches": observed_rpm,
        "observed_sustained_rpm": len(outstanding) / elapsed * 60 if elapsed else 0.0,
        "wall_clock_seconds": elapsed,
        "coordinator_final_limit": coordinator_snapshot["health"]["current_limit"],
        "budget_status": guard.status(),
        "pricing": quote.to_dict(),
    }


__all__ = ["ValidationCall", "append", "call_plan", "execute", "read", "terminal"]
