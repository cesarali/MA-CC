"""Analysis and smoke-report generation for blackboard prompt validation."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mas_cc.musr_team_allocation_generator.validation_study import wilson_interval
from mas_cc.probes.musr_symbolic_ambiguity.analysis import write_csv

from .config import BlackboardValidationConfig
from .execution import read, terminal


def _summarize(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for values, group in sorted(
        groups.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        truth = sum(bool(row["correct"]) for row in group)
        low, high = wilson_interval(truth, len(group))
        output.append(
            {
                **dict(zip(keys, values, strict=True)),
                "n": len(group),
                "truth": truth,
                "truth_rate": truth / len(group),
                "ci95_low": low,
                "ci95_high": high,
                "parse_rate": sum(bool(row["parse_success"]) for row in group)
                / len(group),
            }
        )
    return output


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *(
                "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
                for row in rows
            ),
        ]
    )


def build_outputs(
    root: Path,
    config: BlackboardValidationConfig,
    execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    journal = (
        root / "behavioral/smoke_raw_calls.jsonl"
        if config.mode == "smoke"
        else root / "behavioral/full_raw_calls.jsonl"
    )
    latest = terminal(journal)
    rows = []
    for call_id in sorted(latest):
        event = latest[call_id]
        if event.get("event") != "call_finished":
            continue
        action = event.get("parsed_action") or {}
        metadata = action.get("metadata") or {}
        usage = event.get("usage") or {}
        rows.append(
            {
                "call_id": call_id,
                "task_id": event["task_id"],
                "agent_id": event["agent_id"],
                "state_id": event["state_id"],
                "repetition": event["repetition"],
                "current_vote": event.get("current_vote"),
                "parsed_semantic_answer": event.get("parsed_semantic_answer"),
                "correct": bool(event.get("correct")),
                "parse_success": bool(event.get("parse_success")),
                "latent_coverage_count": int(event["latent_coverage_count"]),
                "exact_evidence_card_count": len(event["total_evidence_ids"]),
                "acquired_evidence_card_count": len(event["acquired_evidence_ids"]),
                "message_count": len(event["sampled_message_ids"]),
                "sampled_message_types": "|".join(event["sampled_message_types"]),
                "output_shared_fact_id": metadata.get("shared_fact_id"),
                "output_public_message": json.dumps(
                    metadata.get("public_message"), sort_keys=True
                ),
                "validation_attempts": event.get("validation_attempts"),
                "transport_retries": event.get("transport_retries"),
                "request_id": event.get("request_id"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            }
        )
    state = _summarize(rows, ("state_id",)) if rows else []
    by_task = _summarize(rows, ("task_id", "state_id")) if rows else []
    by_agent = _summarize(rows, ("task_id", "agent_id", "state_id")) if rows else []
    by_coverage = _summarize(rows, ("latent_coverage_count",)) if rows else []
    by_cards = _summarize(rows, ("exact_evidence_card_count",)) if rows else []
    by_messages = _summarize(rows, ("message_count",)) if rows else []
    tables = root / "analysis/tables"
    write_csv(root / "behavioral/observation_level_results.csv", rows)
    write_csv(tables / "truth_by_state.csv", state)
    write_csv(tables / "truth_by_task_state.csv", by_task)
    write_csv(tables / "truth_by_agent_state.csv", by_agent)
    write_csv(tables / "truth_by_latent_coverage.csv", by_coverage)
    write_csv(tables / "truth_by_evidence_card_count.csv", by_cards)
    write_csv(tables / "truth_by_message_count.csv", by_messages)

    display = [
        {
            "State": row["state_id"],
            "n": row["n"],
            "Truth rate": f"{float(row['truth_rate']):.1%}",
            "95% CI": f"[{float(row['ci95_low']):.1%}, {float(row['ci95_high']):.1%}]",
            "Parse rate": f"{float(row['parse_rate']):.1%}",
        }
        for row in state
    ]
    complete = len(rows) == config.logical_calls
    parse_ok = complete and all(bool(row["parse_success"]) for row in rows)
    if config.mode == "smoke":
        decision = "PASS" if parse_ok else "FAIL"
        decision_text = "PASS means the harness works end to end; it is not a scientific benchmark result."
    else:
        by_name = {str(row["state_id"]): row for row in state}
        if not rows:
            decision = "NOT RUN"
            decision_text = "The full 360-call design is frozen but intentionally has not been executed."
        else:
            ordered = all(
                float(by_name[right]["truth_rate"])
                >= float(by_name[left]["truth_rate"])
                for left, right in (("S0", "S1"), ("S1", "S2"))
            )
            decision = (
                "PASS"
                if parse_ok and ordered
                else "BORDERLINE PASS"
                if complete and parse_ok
                else "FAIL"
            )
            decision_text = "This descriptive harness decision checks parse completeness and the expected S0-to-S2 ordering."
    execution = dict(execution or {})
    report = f"""# MuSR Blackboard Prompt Validation 01 — {config.mode.upper()}

## A. Motivation

This validates whether controlled private, intermediate-board, and near-full-board states work through the real blackboard prompt path. The smoke test is an engineering check, not a scientific estimate.

## B. Frozen benchmark

The six-task symbolic benchmark, k=4 private views, M<=0.45, Hbar>=0.90, margin>=2, P2 source prompt, F9 reference packet, and `gwdg/openai-gpt-oss-120b` remain unchanged. No task or evidence was regenerated.

## C. Actual blackboard runtime prompt

Every S0/S1/S2 call uses `RelationalImitationRoundFeedbackGame.ballot_request` at the `focal_update` stage. This selects the `relational_blackboard_ballot` family and its real response contract. S0 is therefore an empty-board later-round state rather than the static P2 round-zero prompt.

## D. S0/S1/S2 state construction

S0 contains the original private evidence, no acquired evidence, and no visible board message. S1 contains two acquired exact cards, six represented latent values, and one live semantic-only REPORT message. S2 contains five acquired cards, all nine represented latent values, and one live semantic-only REPORT whose `reply_to` points to its archived parent. Older exact-evidence messages have expired, while their evidence persists in private memory.

## E. Parallel execution

Configured local workers: {config.local_workers}. Configured global maximum concurrency: {config.max_concurrency}. Provider-instance concurrency: {config.provider.request_concurrency}. Effective request concurrency is the lower of those limits. Global RPM cap: {config.max_rpm}. Fallback tiers: {list(config.fallback_concurrency)}. Observed peak concurrency: {execution.get("observed_peak_concurrency", "not run")}. Observed peak rolling-minute dispatches: {execution.get("observed_peak_rolling_60s_dispatches", "not run")}. Observed sustained RPM: {execution.get("observed_sustained_rpm", "not run")}.

## F. Behavioral results

{_markdown_table(display, ("State", "n", "Truth rate", "95% CI", "Parse rate")) if display else "No behavioral observations have been run."}

## G. Task heterogeneity

Task-by-state and task-agent-by-state results are stored in `analysis/tables/truth_by_task_state.csv` and `truth_by_agent_state.csv`. Smoke values are too small for scientific interpretation.

## H. Static-vs-blackboard comparison

Static comparison is {"enabled" if config.static_comparison else "disabled"}. It adds no calls in the current configuration.

## I. Evidence-response analysis

Truth is summarized against latent coverage, exact evidence-card count, and visible-message count in separate machine-readable tables.

## J. Blackboard semantic checks

The sanity tables verify REPORT rendering, valid `reply_to` targets, semantic-only non-acquisition, exact evidence acquisition, message expiry, persistent acquired evidence, private-reason exclusion, output-schema acceptance, intended S0/S1/S2 coverage, and hidden-data exclusion.

## K. PASS / BORDERLINE PASS / FAIL

**{decision} — {decision_text}**

## L. Limitations

The development smoke uses only two tasks, one fixed agent per task, three states, and two repetitions. Repeated prompts are not independent semantic worlds. The frozen benchmark itself previously failed its strict Private/Full heterogeneity check; this harness test does not override that result.
"""
    report_path = root / "analysis/blackboard_prompt_validation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return {
        "report": str(report_path),
        "decision": decision,
        "observations": len(rows),
        "state_summary": state,
    }


__all__ = ["build_outputs"]
