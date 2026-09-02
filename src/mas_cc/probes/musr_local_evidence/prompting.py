"""Exact validation and production game-initialization prompt adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mas_cc.games.relational_reasoning.data import RelationalTask
from mas_cc.games.relational_reasoning.imitation_round_feedback.prompts import (
    build_relational_ballot_prompt,
    parse_relational_ballot,
    render_own_fact,
)
from mas_cc.llm_runtime.messages import Message
from mas_cc.llm_runtime.prompts import RegexTokenCounter
from mas_cc.musr_team_allocation_generator.evidence_generation import (
    extract_json_object,
)
from mas_cc.musr_team_allocation_generator.validation_study import (
    VALIDATION_PROMPT_VERSION,
    validation_prompt,
)

from .design import CallSpec


@dataclass(frozen=True, slots=True)
class RenderedCall:
    messages: tuple[Message, ...]
    prompt_family: str
    prompt_version: str
    definition_hash: str
    instance_hash: str
    token_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [message.to_dict() for message in self.messages],
            "prompt_family": self.prompt_family,
            "prompt_version": self.prompt_version,
            "prompt_definition_hash": self.definition_hash,
            "prompt_instance_hash": self.instance_hash,
            "estimated_input_tokens": self.token_estimate,
        }


def displayed_options(
    task: RelationalTask, mapping: Mapping[str, str]
) -> list[dict[str, str]]:
    return [
        {
            "label": letter,
            "semantic_option_id": semantic,
            "display_text": task.answer_display_texts[semantic],
        }
        for letter, semantic in mapping.items()
    ]


def render_call(task: RelationalTask, spec: CallSpec) -> RenderedCall:
    counter = RegexTokenCounter()
    if spec.prompt_family == "validation":
        options = displayed_options(task, spec.option_mapping)
        base = json.loads(
            Path(task.source_path.split("|", 1)[0]).read_text(encoding="utf-8")
        )
        text = validation_prompt(base, options, spec.evidence_ids, condition="partial")
        tokens = counter.count_tokens(text)
        return RenderedCall(
            messages=(Message("user", text),),
            prompt_family="musr_team_allocation_validation",
            prompt_version=VALIDATION_PROMPT_VERSION,
            definition_hash=_sha(text.split("Available evidence:", 1)[0]),
            instance_hash=_sha(text),
            token_estimate=tokens,
        )
    prompt = build_relational_ballot_prompt(
        identity=f"Agent {spec.agent_number}",
        question=task.question,
        option_letters=spec.option_mapping,
        known_facts=tuple(
            render_own_fact(card, task.fact_text(card)) for card in spec.evidence_ids
        ),
        fact_ids=spec.evidence_ids,
        current_vote=None,
        social_sources=(),
        receiver_epistemic_disposition="vigilant",
        social_context=False,
        answer_display_texts=task.answer_display_texts,
    )
    compiled = prompt.compile(counter)
    return RenderedCall(
        messages=compiled.messages,
        prompt_family=compiled.family,
        prompt_version=str(compiled.version),
        definition_hash=compiled.definition_hash,
        instance_hash=compiled.instance_hash,
        token_estimate=compiled.total_token_estimate or 0,
    )


def parse_response(spec: CallSpec, content: str) -> dict[str, Any]:
    try:
        if spec.prompt_family == "validation":
            parsed = extract_json_object(content)
            label = str(parsed.get("option_label", "")).strip().upper()
            if label not in spec.option_mapping:
                raise ValueError("option_label must be A, B, or C")
            return {
                "parse_success": True,
                "selected_letter": label,
                "parsed_semantic_answer": spec.option_mapping[label],
                "rationale": str(parsed.get("rationale", "")),
                "reason": None,
                "shared_fact_id": None,
                "parse_error": None,
            }
        ballot = parse_relational_ballot(
            content, tuple(spec.option_mapping), spec.option_mapping
        )
        if (
            ballot.vote is None
            or ballot.reason is None
            or not ballot.shared_fact_present
        ):
            raise ValueError(
                "game ballot did not contain a valid vote, reason, and shared_fact_id"
            )
        semantic = spec.option_mapping.get(ballot.vote, ballot.vote)
        if semantic not in spec.option_mapping.values():
            raise ValueError("game vote did not resolve to a semantic allocation")
        if (
            ballot.shared_fact_id is not None
            and ballot.shared_fact_id not in spec.evidence_ids
        ):
            raise ValueError("game ballot cited evidence outside the available set")
        return {
            "parse_success": True,
            "selected_letter": ballot.vote,
            "parsed_semantic_answer": semantic,
            "rationale": None,
            "reason": ballot.reason,
            "shared_fact_id": ballot.shared_fact_id,
            "parse_error": None,
        }
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "parse_success": False,
            "selected_letter": None,
            "parsed_semantic_answer": None,
            "rationale": None,
            "reason": None,
            "shared_fact_id": None,
            "parse_error": str(exc),
        }


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = ["RenderedCall", "displayed_options", "parse_response", "render_call"]
