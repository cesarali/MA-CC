"""Production-compatible rendering and parsing for solvability ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mas_cc.games.relational_reasoning.data import RelationalTask
from mas_cc.games.relational_reasoning.imitation_round_feedback.prompts import (
    build_relational_ballot_prompt,
    parse_relational_ballot,
    render_own_fact,
)
from mas_cc.llm_runtime.prompts import RegexTokenCounter

from .design import CallSpec


@dataclass(frozen=True, slots=True)
class RenderedCall:
    messages: tuple[Any, ...]
    family: str
    version: int
    definition_hash: str
    instance_hash: str
    token_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [message.to_dict() for message in self.messages],
            "prompt_family": self.family,
            "prompt_version": self.version,
            "prompt_definition_hash": self.definition_hash,
            "prompt_instance_hash": self.instance_hash,
            "estimated_input_tokens": self.token_estimate,
        }


def render(task: RelationalTask, spec: CallSpec) -> RenderedCall:
    prompt = build_relational_ballot_prompt(
        identity=f"Agent {spec.agent_id or 1}",
        question=task.question,
        option_letters=spec.option_mapping,
        known_facts=tuple(
            render_own_fact(card, task.fact_text(card)) for card in spec.evidence_ids
        ),
        fact_ids=spec.evidence_ids,
        current_vote=None,
        social_sources=(),
        social_context=False,
        receiver_epistemic_disposition="vigilant",
        answer_display_texts=task.answer_display_texts,
        local_prompt_variant=spec.prompt_variant,
    )
    compiled = prompt.compile(RegexTokenCounter())
    return RenderedCall(
        compiled.messages,
        compiled.family,
        compiled.version,
        compiled.definition_hash,
        compiled.instance_hash,
        compiled.total_token_estimate or 0,
    )


def parse(task: RelationalTask, spec: CallSpec, response: str) -> dict[str, Any]:
    try:
        ballot = parse_relational_ballot(
            response, tuple(spec.option_mapping), spec.option_mapping
        )
        if (
            ballot.vote is None
            or ballot.reason is None
            or not ballot.shared_fact_present
        ):
            raise ValueError("response lacks a valid vote, reason, or shared_fact_id")
        semantic = spec.option_mapping.get(ballot.vote, ballot.vote)
        if semantic not in task.semantic_answers:
            raise ValueError("vote does not resolve")
        if (
            ballot.shared_fact_id is not None
            and ballot.shared_fact_id not in spec.evidence_ids
        ):
            raise ValueError("shared_fact_id is unavailable")
        return {
            "parse_success": True,
            "selected_letter": ballot.vote,
            "parsed_semantic_answer": semantic,
            "correct": semantic == task.correct_relation,
            "reason": ballot.reason,
            "shared_fact_id": ballot.shared_fact_id,
            "parse_error": None,
        }
    except (TypeError, ValueError) as exc:
        return {
            "parse_success": False,
            "selected_letter": None,
            "parsed_semantic_answer": None,
            "correct": False,
            "reason": None,
            "shared_fact_id": None,
            "parse_error": str(exc),
        }


__all__ = ["RenderedCall", "parse", "render"]
