"""Bounded, resumable OSS execution for isolated truthful-selective prompts."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mas_cc.core import Seed
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
from mas_cc.llm_runtime.messages import Message, MessageRole
from mas_cc.probes.musr_prompt_solvability.execution import append, terminal

from .config import TruthfulSelectiveConfig
from .design import CallSpec
from .prompting import RenderedCall, parse


async def execute(
    config: TruthfulSelectiveConfig,
    tasks: Mapping[str, Any],
    specs: Sequence[CallSpec],
    rendered: Mapping[str, RenderedCall],
    journal: Path,
) -> dict[str, Any]:
    done = terminal(journal)
    outstanding = [
        spec
        for spec in specs
        if spec.call_id not in done or done[spec.call_id].get("event") == "call_failed"
    ]
    quote = UniversityPricingSource(config.provider).fetch(
        config.provider.type, config.provider.model
    )
    if quote.status != "known" or quote.pricing is None:
        raise RuntimeError(f"live pricing does not permit launch: {quote.status}")
    guard = RuntimeBudgetGuard(
        BudgetLimits(
            max_cost=MonetaryAmount(
                config.max_cost,
                config.accounting_unit,
                "truthful-selective config",
                config.provider.type,
                config.provider.model,
                "MuSR truthful selective local stress test",
                quote.retrieved_at,
                "truthful-selective-v1",
            ),
            max_requests=config.max_requests,
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=config.max_output_tokens_total,
        ),
        expectation=BudgetExpectation(
            len(specs),
            sum(rendered[spec.call_id].token_estimate for spec in specs),
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
            counter.count_tokens(message.content) for message in request.messages
        ),
    )
    semaphore = asyncio.Semaphore(config.workers)
    parser_retry_bound = 1

    async def one(spec: CallSpec) -> None:
        prompt = rendered[spec.call_id]
        base = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **spec.to_dict(),
            **prompt.to_dict(),
            "provider": config.provider.type,
            "model": config.provider.model,
            "temperature": config.provider.temperature,
            "max_output_tokens": config.provider.max_output_tokens,
        }
        append(journal, {"event": "request_started", **base})
        attempts: list[dict[str, Any]] = []
        try:
            messages = prompt.messages
            response = None
            parsed: dict[str, Any] = {"parse_success": False}
            for attempt_index in range(parser_retry_bound + 1):
                request = CompletionRequest(
                    messages,
                    config.provider.temperature,
                    config.provider.max_output_tokens,
                    int(
                        Seed(spec.provider_seed).derive(
                            f"validation-attempt:{attempt_index}"
                        )
                    ),
                    {
                        "probe": "musr_truthful_selective",
                        "call_id": spec.call_id,
                        "validation_attempt": attempt_index + 1,
                        "validation_repair": attempt_index > 0,
                    },
                )
                async with semaphore:
                    response = await provider.complete(request)
                parsed = parse(tasks[spec.task_id], spec, response.content)
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "request": request.to_dict(),
                        "raw_response": response.content,
                        "response": response.to_dict(),
                        "parse_success": parsed["parse_success"],
                        "parse_error": parsed.get("parse_error"),
                    }
                )
                if parsed["parse_success"]:
                    break
                if attempt_index < parser_retry_bound:
                    messages = (
                        *prompt.messages,
                        Message(
                            MessageRole.USER,
                            prompt.response_contract.repair_guidance(()),
                        ),
                    )
            assert response is not None
            if not parsed["parse_success"]:
                append(
                    journal,
                    {
                        "event": "call_failed",
                        **base,
                        **parsed,
                        "attempts": attempts,
                        "validation_attempts": len(attempts),
                        "raw_response": response.content,
                        "response": response.to_dict(),
                        "usage": response.usage.to_dict(),
                        "provider_error": "response validation exhausted",
                    },
                )
                return
            append(
                journal,
                {
                    "event": "call_finished",
                    **base,
                    **parsed,
                    "attempts": attempts,
                    "validation_attempts": len(attempts),
                    "raw_response": response.content,
                    "response": response.to_dict(),
                    "redacted_provider_response": response.redacted_raw_response(),
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
                    **base,
                    "attempts": attempts,
                    "validation_attempts": len(attempts),
                    "parse_success": False,
                    "provider_error": f"{type(exc).__name__}: {exc}",
                },
            )

    try:
        await asyncio.gather(*(one(spec) for spec in outstanding))
    finally:
        provider.close()
    final = terminal(journal)
    expected = {spec.call_id for spec in specs}
    return {
        "scheduled": len(specs),
        "previously_terminal": len(done),
        "attempted_now": len(outstanding),
        "terminal": len(expected & set(final)),
        "successful": sum(
            final.get(key, {}).get("parse_success") is True for key in expected
        ),
        "failed": sum(
            final.get(key, {}).get("parse_success") is not True
            for key in expected
            if key in final
        ),
        "provider_attempts": sum(
            len(final.get(key, {}).get("attempts", ())) for key in expected
        ),
        "parser_retries": sum(
            max(0, len(final.get(key, {}).get("attempts", ())) - 1) for key in expected
        ),
        "transport_retries": sum(
            int(final.get(key, {}).get("transport_retries") or 0) for key in expected
        ),
        "budget_status": guard.status(),
        "pricing": quote.to_dict(),
    }


__all__ = ["execute"]
