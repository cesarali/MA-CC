"""Bounded, resumable provider execution with complete prompt/response retention."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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

from .config import LocalEvidenceProbeConfig
from .design import CallSpec
from .preflight import ProbePlan
from .prompting import parse_response


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_journal(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_journal(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return ()
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def terminal_rows(paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_journal(path):
            if row.get("event") in {"call_finished", "call_failed"}:
                found[str(row["call_id"])] = row
    return found


async def execute_plan(config: LocalEvidenceProbeConfig, plan: ProbePlan, root: Path) -> dict[str, Any]:
    paths = (
        root / "prompt_equivalence/raw_calls.jsonl",
        root / "evidence_dose/raw_calls.jsonl",
    )
    completed = terminal_rows(paths)
    outstanding = [spec for spec in plan.calls if spec.call_id not in completed]
    semaphore = asyncio.Semaphore(config.workers)
    quote = UniversityPricingSource(config.provider).fetch(
        config.provider.type, config.provider.model
    )
    if quote.status != "known" or quote.pricing is None:
        raise RuntimeError(f"live pricing does not permit launch: {quote.status}")
    cost_limit = MonetaryAmount(
        amount=config.max_cost,
        unit=config.accounting_unit,
        unit_source="probe config",
        provider=config.provider.type,
        model=config.provider.model,
        source="musr local evidence probe",
        retrieved_at=quote.retrieved_at,
        version="probe-v1",
    )
    guard = RuntimeBudgetGuard(
        BudgetLimits(
            max_cost=cost_limit,
            max_requests=config.max_requests,
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=config.max_output_tokens_total,
        ),
        expectation=BudgetExpectation(
            requests=len(plan.calls),
            input_tokens=sum(item.token_estimate for item in plan.rendered.values()),
            output_tokens=len(plan.calls) * config.provider.max_output_tokens,
        ),
    )
    raw_provider = create_llm_provider(config.provider)
    counter = RegexTokenCounter()
    provider = BudgetGuardedProvider(
        raw_provider,
        guard,
        quote.pricing,
        input_token_estimator=lambda request: sum(
            counter.count_tokens(message.content) for message in request.messages
        ),
    )
    card_groups = {
        card: latent
        for latent, cards in (plan.task.supporting_fact_groups or {}).items()
        for card in cards
    }

    async def one(spec: CallSpec) -> None:
        rendered = plan.rendered[spec.call_id]
        journal = paths[0] if spec.experiment == "prompt_equivalence" else paths[1]
        started = {
            "event": "request_started",
            "timestamp": _now(),
            **spec.to_dict(),
            **rendered.to_dict(),
            "provider": config.provider.type,
            "model": config.provider.model,
            "temperature": config.provider.temperature,
            "max_output_tokens": config.provider.max_output_tokens,
            "distinct_latent_facts": len(
                {card_groups[card] for card in spec.evidence_ids}
            ),
        }
        append_journal(journal, started)
        request = CompletionRequest(
            messages=rendered.messages,
            temperature=config.provider.temperature,
            max_output_tokens=config.provider.max_output_tokens,
            seed=spec.requested_seed,
            metadata={"probe": "musr_local_evidence", "call_id": spec.call_id},
        )
        try:
            async with semaphore:
                response = await provider.complete(request)
            parsed = parse_response(spec, response.content)
            append_journal(
                journal,
                {
                    "event": "call_finished",
                    "timestamp": _now(),
                    **spec.to_dict(),
                    **rendered.to_dict(),
                    **parsed,
                    "correct": parsed["parsed_semantic_answer"] == plan.task.correct_relation,
                    "raw_response": response.content,
                    "response": response.to_dict(),
                    "redacted_provider_response": response.redacted_raw_response(),
                    "provider": response.provider,
                    "model": response.model,
                    "usage": response.usage.to_dict(),
                    "latency_seconds": response.latency_seconds,
                    "transport_retries": response.retries,
                    "request_id": response.request_id,
                    "distinct_latent_facts": len(
                        {card_groups[card] for card in spec.evidence_ids}
                    ),
                },
            )
        except Exception as exc:  # preserve failure and let the other calls finish
            append_journal(
                journal,
                {
                    "event": "call_failed",
                    "timestamp": _now(),
                    **spec.to_dict(),
                    **rendered.to_dict(),
                    "provider": config.provider.type,
                    "model": config.provider.model,
                    "provider_error": f"{type(exc).__name__}: {exc}",
                    "parse_success": False,
                },
            )

    try:
        await asyncio.gather(*(one(spec) for spec in outstanding))
    finally:
        provider.close()
    terminal = terminal_rows(paths)
    expected = {spec.call_id for spec in plan.calls}
    return {
        "scheduled": len(plan.calls),
        "previously_terminal": len(completed),
        "attempted_now": len(outstanding),
        "terminal": len(expected & set(terminal)),
        "successful": sum(terminal.get(call_id, {}).get("parse_success") is True for call_id in expected),
        "failed": sum(terminal.get(call_id, {}).get("parse_success") is not True for call_id in expected if call_id in terminal),
        "budget_status": guard.status(),
        "pricing": quote.to_dict(),
    }


__all__ = ["append_journal", "execute_plan", "read_journal", "terminal_rows"]
