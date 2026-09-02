"""Append-only, resumable staged provider execution."""

from __future__ import annotations

import asyncio, json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mas_cc.llm_runtime.providers import (
    BudgetExpectation,
    BudgetGuardedProvider,
    BudgetLimits,
    CompletionRequest,
    MonetaryAmount,
    RuntimeBudgetGuard,
    UniversityPricingSource,
    create_llm_provider,
)
from mas_cc.llm_runtime.prompts import RegexTokenCounter

from .config import SolvabilityConfig
from .design import CallSpec
from .prompting import RenderedCall, parse


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


async def execute(
    config: SolvabilityConfig,
    tasks: Mapping[str, Any],
    specs: Sequence[CallSpec],
    rendered: Mapping[str, RenderedCall],
    journal: Path,
    *,
    probe_name: str = "musr_prompt_solvability",
    retry_failed: bool = False,
) -> dict[str, Any]:
    done = terminal(journal)
    outstanding = [
        spec
        for spec in specs
        if spec.call_id not in done
        or (retry_failed and done[spec.call_id].get("event") == "call_failed")
    ]
    quote = UniversityPricingSource(config.provider).fetch(
        config.provider.type, config.provider.model
    )
    if quote.status != "known" or quote.pricing is None:
        raise RuntimeError(f"live pricing does not permit launch: {quote.status}")
    limit = MonetaryAmount(
        config.max_cost,
        config.accounting_unit,
        "probe config",
        config.provider.type,
        config.provider.model,
        "MuSR prompt solvability calibration",
        quote.retrieved_at,
        "probe-v1",
    )
    guard = RuntimeBudgetGuard(
        BudgetLimits(
            max_cost=limit,
            max_requests=config.max_requests,
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=config.max_output_tokens_total,
        ),
        expectation=BudgetExpectation(
            len(specs),
            sum(rendered[s.call_id].token_estimate for s in specs),
            len(specs) * config.provider.max_output_tokens,
        ),
    )
    raw = create_llm_provider(config.provider)
    counter = RegexTokenCounter()
    provider = BudgetGuardedProvider(
        raw,
        guard,
        quote.pricing,
        input_token_estimator=lambda request: sum(
            counter.count_tokens(m.content) for m in request.messages
        ),
    )
    semaphore = asyncio.Semaphore(config.workers)

    async def one(spec: CallSpec) -> None:
        prompt = rendered[spec.call_id]
        base = {
            "event": "request_started",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **spec.to_dict(),
            **prompt.to_dict(),
            "provider": config.provider.type,
            "model": config.provider.model,
            "temperature": config.provider.temperature,
            "max_output_tokens": config.provider.max_output_tokens,
        }
        append(journal, base)
        request = CompletionRequest(
            prompt.messages,
            config.provider.temperature,
            config.provider.max_output_tokens,
            spec.provider_seed,
            {"probe": probe_name, "call_id": spec.call_id},
        )
        try:
            async with semaphore:
                response = await provider.complete(request)
            parsed = parse(tasks[spec.task_id], spec, response.content)
            append(
                journal,
                {
                    "event": "call_finished",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **spec.to_dict(),
                    **prompt.to_dict(),
                    **parsed,
                    "raw_response": response.content,
                    "response": response.to_dict(),
                    "redacted_provider_response": response.redacted_raw_response(),
                    "provider": response.provider,
                    "model": response.model,
                    "usage": response.usage.to_dict(),
                    "latency_seconds": response.latency_seconds,
                    "transport_retries": response.retries,
                    "request_id": response.request_id,
                },
            )
        except Exception as exc:
            append(
                journal,
                {
                    "event": "call_failed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **spec.to_dict(),
                    **prompt.to_dict(),
                    "provider": config.provider.type,
                    "model": config.provider.model,
                    "parse_success": False,
                    "provider_error": f"{type(exc).__name__}: {exc}",
                },
            )

    try:
        await asyncio.gather(*(one(spec) for spec in outstanding))
    finally:
        provider.close()
    final = terminal(journal)
    expected = {s.call_id for s in specs}
    return {
        "scheduled": len(specs),
        "previously_terminal": len(done),
        "attempted_now": len(outstanding),
        "terminal": len(expected & set(final)),
        "successful": sum(
            final.get(x, {}).get("parse_success") is True for x in expected
        ),
        "failed": sum(
            final.get(x, {}).get("parse_success") is not True
            for x in expected
            if x in final
        ),
        "budget_status": guard.status(),
        "pricing": quote.to_dict(),
    }


__all__ = ["append", "execute", "read", "terminal"]
