"""Checkpointed execution for the isolated OSS manifest."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import socket
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
from mas_cc.musr_team_allocation_generator.io_utils import (
    sha256_object,
    write_json_atomic,
)

from .isolated_config import IsolatedOSSConfig
from .isolated_design import IsolatedRequest
from .isolated_prompting import IsolatedPrompt, parse_isolated


def _pricing_identity(pricing: Any) -> dict[str, Any]:
    """Identify effective rates without binding resume to retrieval time."""

    value = pricing.to_dict()
    for field in ("retrieved_at", "version", "source", "unit_source"):
        value.pop(field, None)
    return value


def append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def run_lock(root: Path):
    lock = root / "runtime/run.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RuntimeError(f"another writer holds {lock}") from exc
    try:
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{socket.gethostname()}:{os.getpid()}\n".encode())
        os.fsync(descriptor)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def completed_ids(root: Path) -> set[str]:
    return {
        path.stem
        for path in (root / "checkpoints").glob("*.json")
        if json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    }


def _execution_identity(
    config: IsolatedOSSConfig, item: IsolatedRequest, prompt: IsolatedPrompt
) -> str:
    """Bind reusable output to every setting that can change its answer."""

    return sha256_object(
        {
            "request": item.to_dict(),
            "prompt_instance_hash": prompt.instance_hash,
            "provider": {
                "type": config.provider.type,
                "model": config.provider.model,
                "temperature": config.provider.temperature,
                "max_output_tokens": config.provider.max_output_tokens,
                "response_format": config.provider.options.get("response_format"),
            },
            "seed": config.seed,
        }
    )


def _compatible_completed_ids(
    root: Path,
    config: IsolatedOSSConfig,
    requests: Sequence[IsolatedRequest],
    prompts: Mapping[str, IsolatedPrompt],
) -> set[str]:
    finished: set[str] = set()
    for item in requests:
        path = root / f"checkpoints/{item.request_id}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        expected = _execution_identity(config, item, prompts[item.request_id])
        if payload.get("execution_identity_hash") != expected:
            raise RuntimeError(
                f"incompatible completed checkpoint for {item.request_id}; "
                "use the original frozen manifest/model/prompt or a new result root"
            )
        finished.add(item.request_id)
    return finished


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
    if config.provider.request_concurrency < profile.concurrency:
        raise RuntimeError(
            "provider request_concurrency is below the selected execution profile"
        )
    with run_lock(root):
        return await _execute_locked(
            config,
            tasks,
            requests,
            prompts,
            root,
            execution_profile=execution_profile,
        )


async def _execute_locked(
    config: IsolatedOSSConfig,
    tasks: Mapping[str, Any],
    requests: Sequence[IsolatedRequest],
    prompts: Mapping[str, IsolatedPrompt],
    root: Path,
    *,
    execution_profile: str,
) -> dict[str, Any]:
    profile = config.profile(execution_profile)
    finished = _compatible_completed_ids(root, config, requests, prompts)
    outstanding = [
        request for request in requests if request.request_id not in finished
    ]
    quote = UniversityPricingSource(config.provider).fetch(
        config.provider.type, config.provider.model
    )
    if quote.status != "known" or quote.pricing is None:
        raise RuntimeError(f"live pricing does not permit launch: {quote.status}")
    max_cost = MonetaryAmount(
        config.max_cost,
        config.accounting_unit,
        "isolated OSS config",
        config.provider.type,
        config.provider.model,
        "MuSR isolated OSS evaluation",
        quote.retrieved_at,
        "isolated-oss-v1",
    )
    limits = BudgetLimits(
        max_cost=max_cost,
        max_requests=config.max_provider_attempts,
        max_input_tokens=config.max_input_tokens,
        max_output_tokens=config.max_output_tokens,
    )
    guard = RuntimeBudgetGuard(
        limits,
        expectation=BudgetExpectation(
            requests=len(requests) * (config.invalid_response_retries + 1),
            input_tokens=sum(
                prompts[row.request_id].token_estimate for row in requests
            ),
            output_tokens=len(requests) * config.provider.max_output_tokens,
        ),
    )
    pricing_hash = sha256_object(_pricing_identity(quote.pricing))
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
    coordinator = SharedProviderCoordinator(
        root / "runtime/provider-control" / execution_profile, load
    )
    raw = create_llm_provider(config.provider, request_coordinator=coordinator)
    try:
        advertised = await raw.discover_models()
        if config.provider.model not in advertised:
            raise RuntimeError(
                f"configured model {config.provider.model!r} is not advertised"
            )
    except BaseException:
        raw.close()
        raise
    counter = RegexTokenCounter()
    provider = BudgetGuardedProvider(
        raw,
        guard,
        quote.pricing,
        input_token_estimator=lambda request: sum(
            counter.count_tokens(message.content) for message in request.messages
        ),
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
                    seed=int(
                        Seed(config.seed).derive(
                            f"provider:{item.request_id}:{attempt}"
                        )
                    ),
                    metadata={
                        "probe": "musr_truthful_selective_isolated",
                        "request_id": item.request_id,
                        "attempt": attempt + 1,
                    },
                )
                async with semaphore:
                    if stop.is_set():
                        return
                    response = await provider.complete(completion)
                parsed = parse_isolated(tasks[item.task_id], item, response.content)
                attempt_row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "request_id": item.request_id,
                    "attempt": attempt + 1,
                    "request": completion.to_dict(),
                    "response": response.to_dict(),
                    "raw_response": response.content,
                    "parsed": parsed,
                }
                attempts.append(attempt_row)
                append(root / "attempts/attempt_ledger.jsonl", attempt_row)
                if parsed["parse_success"]:
                    terminal_status = "completed"
                    break
                if attempt < config.invalid_response_retries:
                    messages = (
                        *prompt.messages,
                        Message(
                            MessageRole.USER,
                            prompt.response_contract.repair_guidance(()),
                        ),
                    )
        except Exception as exc:
            terminal_status = "failed"
            append(
                root / "attempts/attempt_ledger.jsonl",
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "request_id": item.request_id,
                    "attempt": len(attempts) + 1,
                    "provider_error": f"{type(exc).__name__}: {exc}",
                },
            )
            if getattr(exc, "retryable", True) is False:
                stop.set()
        write_json_atomic(
            root / f"checkpoints/{item.request_id}.json",
            {
                "schema_version": 1,
                "status": terminal_status,
                **item.to_dict(),
                "prompt_instance_hash": prompt.instance_hash,
                "execution_identity_hash": _execution_identity(config, item, prompt),
                "attempts": attempts,
                "parsed": parsed
                if terminal_status in {"completed", "invalid"}
                else None,
            },
        )

    try:
        await asyncio.gather(*(one(item) for item in outstanding))
    finally:
        provider.close()
    return {
        "planned": len(requests),
        "previously_completed": len(finished),
        "attempted_now": len(outstanding),
        "completed": len(_compatible_completed_ids(root, config, requests, prompts)),
        "stopped": stop.is_set(),
        "budget": guard.status(),
        "coordinator": coordinator.snapshot(),
    }


__all__ = ["append", "completed_ids", "execute", "run_lock"]
