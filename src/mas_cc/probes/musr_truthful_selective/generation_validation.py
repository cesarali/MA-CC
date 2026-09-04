"""Deterministic and Terra-assisted semantic audit of generated evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mas_cc.core import Seed
from mas_cc.musr_team_allocation_generator.evidence_generation import (
    extract_json_object,
)
from mas_cc.musr_team_allocation_generator.io_utils import (
    sha256_object,
    write_json_atomic,
)
from mas_cc.musr_team_allocation_generator.provider_adapter import MuSRGenerationModel
from mas_cc.musr_team_allocation_generator.schemas import LatentProblem
from mas_cc.musr_team_allocation_generator.symbolic_facts import (
    CanonicalFact,
    render_canonical_equality_evidence,
)

_ALLOWED_VERDICTS = {"PASS", "FAIL"}
VALIDATOR_VERSION = "truthful_selective_semantic_v2"
_NUMERIC_PATTERN = re.compile(
    r"\b(?:level|rating|skill(?:\s+level)?|cooperation(?:\s+level)?)\s*"
    r"(?:is|was|=|of|:)?\s*"
    r"(?:[1-3]|one|two|three)\b",
    re.IGNORECASE,
)
_CERTAINTY_PATTERN = re.compile(
    r"\b(?:certainly|definitely|guarantees?|proves?|must be|always|exactly)\b",
    re.IGNORECASE,
)


def validation_prompt(
    problem: LatentProblem,
    fact: CanonicalFact,
    statements: Sequence[str],
) -> str:
    return f"""Audit one generated evidence card for an exact Team Allocation proposition.
This is a strict semantic validation task, not creative rewriting.

Exact intended proposition:
{fact.canonical_text}

Typed exact predicate:
{json.dumps(fact.to_dict(), ensure_ascii=False, sort_keys=True)}

Visible evidence card:
{json.dumps(list(statements), ensure_ascii=False)}

Hidden world context contains three people and two tasks, but the card must not reveal hidden numeric levels, candidate scores, the gold answer, or a recommended allocation.

Decide all fields independently:
- faithfulness: the card supports the intended proposition rather than an unrelated claim;
- polarity_preserved: it does not reverse <=/>=, equality, positive/negative, stronger/weaker, or at-least/at-most meaning;
- no_strengthening: it does not turn a weak comparison or bound into equality, certainty, an exact hidden value, or a stronger absolute claim;
- no_unsupported_implication: it adds no conclusion beyond plausible evidence for the exact proposition;
- no_hidden_state_leakage: it reveals no hidden numeric level, candidate score, gold answer, or option identity;
- coherent: both visible statements describe one coherent evidential event.

Return JSON only:
{{
  "faithfulness": "PASS or FAIL",
  "polarity_preserved": "PASS or FAIL",
  "no_strengthening": "PASS or FAIL",
  "no_unsupported_implication": "PASS or FAIL",
  "no_hidden_state_leakage": "PASS or FAIL",
  "coherent": "PASS or FAIL",
  "reason": "brief concrete explanation"
}}
"""


def deterministic_checks(
    problem: LatentProblem,
    fact: CanonicalFact,
    statements: Sequence[str],
) -> dict[str, Any]:
    combined = " ".join(statements)
    lowered = combined.casefold()
    option_phrases = [
        " ".join((allocation.singleton, *allocation.pair)).casefold()
        for allocation in problem.candidate_allocations
    ]
    answer_leakage = any(
        phrase in lowered
        for phrase in (
            "best allocation",
            "correct allocation",
            "correct assignment",
            "gold answer",
            "option a",
            "option b",
            "option c",
            "allocation_0",
            "allocation_1",
            "allocation_2",
            "should be assigned",
        )
    ) or any(phrase and phrase in lowered for phrase in option_phrases)
    score_leakage = bool(
        re.search(r"\b(?:score|points?)\s*(?:is|of|=|:)\s*\d+\b", combined, re.I)
    )
    numeric_level_leakage = bool(_NUMERIC_PATTERN.search(combined))
    certainty_language = bool(_CERTAINTY_PATTERN.search(combined))
    duplicates = len({" ".join(item.casefold().split()) for item in statements}) != len(
        statements
    )
    return {
        "nonempty": bool(statements) and all(str(item).strip() for item in statements),
        "no_duplicate_statements": not duplicates,
        "no_answer_or_option_leakage": not answer_leakage,
        "no_score_leakage": not score_leakage,
        "no_explicit_numeric_level": not numeric_level_leakage,
        "no_certainty_language": not certainty_language,
    }


def parse_semantic_audit(content: str) -> dict[str, Any]:
    value = extract_json_object(content)
    fields = (
        "faithfulness",
        "polarity_preserved",
        "no_strengthening",
        "no_unsupported_implication",
        "no_hidden_state_leakage",
        "coherent",
    )
    output = {}
    for field in fields:
        verdict = str(value.get(field, "")).strip().upper()
        if verdict not in _ALLOWED_VERDICTS:
            raise ValueError(f"semantic audit {field} must be PASS or FAIL")
        output[field] = verdict
    reason = str(value.get("reason", "")).strip()
    if not reason:
        raise ValueError("semantic audit requires a non-empty reason")
    output["reason"] = reason
    output["passed"] = all(output[field] == "PASS" for field in fields)
    return output


async def audit_cards(
    model: MuSRGenerationModel,
    *,
    task_root: Path,
    problem: LatentProblem,
    facts: Sequence[CanonicalFact],
    cards: Sequence[Mapping[str, Any]],
    seed: Seed,
) -> dict[str, Any]:
    by_id = {fact.fact_id: fact for fact in facts}
    roles = {
        str(row["fact_id"]): str(row["role"])
        for row in json.loads(
            (task_root / "facts/all_true_facts.json").read_text(encoding="utf-8")
        )
    }
    validation_path = task_root / "generation/semantic_validation.json"
    existing_rows = (
        json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_path.is_file()
        else []
    )
    existing = {
        str(row["fact_id"]): row for row in existing_rows if row.get("passed") is True
    }
    requested_ids = {str(card["fact_id"]) for card in cards}
    preserved = [
        row
        for row in existing_rows
        if str(row.get("fact_id")) in requested_ids and row.get("passed") is True
    ]
    if len(preserved) != len(existing_rows):
        write_json_atomic(validation_path, preserved)
    rows = []
    for card in cards:
        fact_id = str(card["fact_id"])
        fact = by_id[fact_id]
        statements = tuple(str(item) for item in card["generated_card_text"])
        content_hash = sha256_object(
            {
                "validator_version": VALIDATOR_VERSION,
                "fact": fact.to_dict(),
                "statements": list(statements),
            }
        )
        if fact_id in existing:
            cached = dict(existing[fact_id])
            legacy_content_matches = (
                cached.get("validated_content_sha256") is None
                and cached.get("generated_card_text") == list(statements)
                and cached.get("canonical_exact_fact") == fact.to_dict()
            )
            if (
                cached.get("validated_content_sha256") == content_hash
                or legacy_content_matches
            ):
                cached["validated_content_sha256"] = content_hash
                cached["validator_version"] = VALIDATOR_VERSION
                rows.append(cached)
                continue
        deterministic = deterministic_checks(problem, fact, statements)
        if fact.operator == "eq" and statements == render_canonical_equality_evidence(
            problem, fact
        ):
            semantic = {
                "faithfulness": "PASS",
                "polarity_preserved": "PASS",
                "no_strengthening": "PASS",
                "no_unsupported_implication": "PASS",
                "no_hidden_state_leakage": "PASS",
                "coherent": "PASS",
                "reason": "Exact deterministic equality template directly states a shared unnamed category.",
                "passed": True,
                "validation_method": "deterministic_canonical_equality_v1",
            }
            model_id = None
        else:
            response = await model.inference(
                validation_prompt(problem, fact, statements),
                seed=int(seed.derive(fact_id)),
                purpose="evidence_semantic_validation",
                metadata={
                    "task_id": task_root.name,
                    "fact_id": fact_id,
                    "role": roles[fact_id],
                },
            )
            model_id = response.model
            try:
                semantic = parse_semantic_audit(response.content)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                semantic = {
                    "faithfulness": "FAIL",
                    "polarity_preserved": "FAIL",
                    "no_strengthening": "FAIL",
                    "no_unsupported_implication": "FAIL",
                    "no_hidden_state_leakage": "FAIL",
                    "coherent": "FAIL",
                    "reason": f"unparseable semantic audit: {exc}",
                    "passed": False,
                }
        passed = all(deterministic.values()) and bool(semantic["passed"])
        rows.append(
            {
                "task_id": task_root.name,
                "fact_id": fact_id,
                "role": roles[fact_id],
                "canonical_exact_fact": fact.to_dict(),
                "generated_card_text": list(statements),
                "deterministic_checks": deterministic,
                "semantic_audit": semantic,
                "passed": passed,
                "validated_content_sha256": content_hash,
                "validator_version": VALIDATOR_VERSION,
                "audit_prompt_hash": sha256_object(
                    validation_prompt(problem, fact, statements)
                ),
                "model_id": model_id,
                "audit_seed": int(seed.derive(fact_id)),
            }
        )
        write_json_atomic(validation_path, rows)
    summary = {
        "validation_method": (
            "deterministic leakage/format checks plus independent Terra semantic "
            "self-audit; strong screening, not a formal proof of natural-language entailment"
        ),
        "task_id": task_root.name,
        "cards": len(rows),
        "passed": sum(bool(row["passed"]) for row in rows),
        "failed": sum(not bool(row["passed"]) for row in rows),
        "all_passed": all(bool(row["passed"]) for row in rows),
        "by_role": {
            role: {
                "cards": sum(row["role"] == role for row in rows),
                "passed": sum(row["role"] == role and row["passed"] for row in rows),
            }
            for role in sorted(set(roles.values()))
        },
    }
    write_json_atomic(validation_path, rows)
    write_json_atomic(
        task_root / "generation/semantic_validation_summary.json", summary
    )
    return summary


__all__ = [
    "audit_cards",
    "deterministic_checks",
    "parse_semantic_audit",
    "validation_prompt",
]
