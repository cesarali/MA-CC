"""Checkpointed execution for the isolated OSS manifest."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mas_cc.core import Seed
from mas_cc.llm_runtime.messages import Message, MessageRole
from mas_cc.llm_runtime.providers import (
    AtomicBudgetStateStore,
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
from mas_cc.musr_team_allocation_generator.io_utils import sha256_object, write_json_atomic

from .isolated_config import IsolatedOSSConfig
from .isolated_design import IsolatedRequest
from .isolated_prompting import IsolatedPrompt, parse_isolated


def append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


@contextmanager
def run_lock(root: Path):
    lock = root / "runtime/run.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"another writer holds {lock}") from exc
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


def completed_ids(root: Path) -> set[str]:
    return {
        path.stem
        for path in (root / "checkpoints").glob("*.json")
        if json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    }


async def execute(
    config: IsolatedOSSConfig,
    tasks: Mapping[str, Any],
    requests: Sequence[IsolatedRequest],
    prompts: Mapping[str, IsolatedPrompt],
    root: Path,
    *,
    execution_profile: str,
) -> dict[str, Any]:
    profile = config.profile(execution_profile)
    finished = completed_ids(root)
    outstanding = [request for request in requests if request.request_id not in finished]
    quote = UniversityPricingSource(config.provider).fetch(config.provider.type, config.provider.model)
    if quote.status != "known" or quote.pricing is None:
        raise RuntimeError(f"live pricing does not permit launch: {quote.status}")
    max_cost = MonetaryAmount(
        config.max_cost, config.accounting_unit, "isolated OSS config",
        config.provider.type, config.provider.model,
        "MuSR isolated OSS evaluation", quote.retrieved_at, "isolated-oss-v1",
    )
    limits = BudgetLimits(
        max_cost=max_cost, max_requests=config.max_provider_attempts,
        max_input_tokens=config.max_input_tokens, max_output_tokens=config.max_output_tokens,
    )
    guard = RuntimeBudgetGuard(
        limits,
        expectation=BudgetExpectation(
            requests=len(requests) * (config.invalid_response_retries + 1),
            input_tokens=sum(prompts[row.request_id].token_estimate for row in requests),
            output_tokens=len(requests) * config.provider.max_output_tokens,
        ),
    )
    pricing_hash = sha256_object(quote.to_dict())
    budget_hash = sha256_object(limits.compatibility_identity())
    store = AtomicBudgetStateStore(
        root / "runtime/budget_state.json",
        resolved_budget_hash=budget_hash,
        pricing_snapshot_hash=pricing_hash,
    )
    store.restore(guard)
    guard.set_durable_state_sink(store.write)
    load = ProviderLoadControlConfig(
        initial_concurrency=profile.concurrency,
        minimum_concurrency=max(1, min(4, profile.concurrency)),
        maximum_concurrency=profile.concurrency,
        target_rpm=profile.requests_per_minute,
    )
    coordinator = SharedProviderCoordinator(root / "runtime/provider-control", load)
    raw = create_llm_provider(config.provider, request_coordinator=coordinator)
    counter = RegexTokenCounter()
    provider = BudgetGuardedProvider(
        raw, guard, quote.pricing,
        input_token_estimator=lambda request: sum(counter.count_tokens(message.content) for message in request.messages),
    )
    semaphore = asyncio.Semaphore(profile.concurrency)
    stop = asyncio.Event()

    async def one(item: IsolatedRequest) -> None:
        if stop.is_set():
            return
        prompt = prompts[item.request_id]
        attempts = []
        messages = prompt.messages
        parsed: dict[str, Any] = {"parse_success": False}
        terminal_status = "invalid"
        try:
            for attempt in range(config.invalid_response_retries + 1):
                completion = CompletionRequest(
                    messages=messages,
                    temperature=config.provider.temperature,
                    max_output_tokens=config.provider.max_output_tokens,
                    seed=int(Seed(config.seed).derive(f"provider:{item.request_id}:{attempt}")),
                    metadata={"probe": "musr_truthful_selective_isolated", "request_id": item.request_id, "attempt": attempt + 1},
                )
                async with semaphore:
                    response = await provider.complete(completion)
                parsed = parse_isolated(tasks[item.task_id], item, response.content)
                attempt_row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "request_id": item.request_id, "attempt": attempt + 1,
                    "request": completion.to_dict(), "response": response.to_dict(),
                    "raw_response": response.content, "parsed": parsed,
                }
                attempts.append(attempt_row); append(root / "attempts/attempt_ledger.jsonl", attempt_row)
                if parsed["parse_success"]:
                    terminal_status = "completed"; break
                if attempt < config.invalid_response_retries:
                    messages = (*prompt.messages, Message(MessageRole.USER, prompt.response_contract.repair_guidance(())))
        except Exception as exc:
            terminal_status = "failed"
            append(root / "attempts/attempt_ledger.jsonl", {
                "timestamp": datetime.now(timezone.utc).isoformat(), "request_id": item.request_id,
                "attempt": len(attempts) + 1, "provider_error": f"{type(exc).__name__}: {exc}",
            })
            if getattr(exc, "retryable", True) is False:
                stop.set()
        write_json_atomic(root / f"checkpoints/{item.request_id}.json", {
            "schema_version": 1, "status": terminal_status, **item.to_dict(),
            "prompt_instance_hash": prompt.instance_hash, "attempts": attempts,
            "parsed": parsed if terminal_status in {"completed", "invalid"} else None,
        })

    with run_lock(root):
        try:
            await asyncio.gather(*(one(item) for item in outstanding))
        finally:
            provider.close()
    return {
        "planned": len(requests), "previously_completed": len(finished),
        "attempted_now": len(outstanding), "completed": len(completed_ids(root)),
        "stopped": stop.is_set(), "budget": guard.status(),
        "coordinator": coordinator.snapshot(),
    }


__all__ = ["append", "completed_ids", "execute", "run_lock"]
