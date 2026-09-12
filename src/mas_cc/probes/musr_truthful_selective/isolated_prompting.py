"""Production-compatible isolated prompt rendering for frozen OSS requests."""

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

from .isolated_design import IsolatedRequest


@dataclass(frozen=True, slots=True)
class IsolatedPrompt:
    messages: tuple[Any, ...]
    family: str
    version: int
    definition_hash: str
    instance_hash: str
    token_estimate: int
    response_contract: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [message.to_dict() for message in self.messages],
            "prompt_family": self.family,
            "prompt_version": self.version,
            "prompt_definition_hash": self.definition_hash,
            "prompt_instance_hash": self.instance_hash,
            "estimated_input_tokens": self.token_estimate,
        }


def render_isolated(
    task: RelationalTask, request: IsolatedRequest, *, prompt_variant: str = "P2"
) -> IsolatedPrompt:
    identity = (
        "Agent 1"
        if request.agent_id is None
        else f"Agent {int(request.agent_id.rsplit('_', 1)[1])}"
    )
    private_lines = tuple(
        render_own_fact(fact_id, task.fact_text(fact_id))
        for fact_id in request.private_fact_ids
    )
    canonical = task.controller_report_texts or {}
    report_lines = tuple(
        render_own_fact(fact_id, str(canonical[fact_id]))
        for fact_id in request.report_fact_ids
    )
    citable = tuple(dict.fromkeys(request.evidence_ids))
    prompt = build_relational_ballot_prompt(
        identity=identity,
        question=task.question,
        option_letters=request.option_mapping,
        known_facts=(*private_lines, *report_lines),
        fact_ids=citable,
        current_vote=None,
        social_sources=(),
        social_context=False,
        receiver_epistemic_disposition="vigilant",
        answer_display_texts=task.answer_display_texts,
        local_prompt_variant=prompt_variant,
    )
    compiled = prompt.compile(RegexTokenCounter())
    return IsolatedPrompt(
        compiled.messages,
        compiled.family,
        compiled.version,
        compiled.definition_hash,
        compiled.instance_hash,
        compiled.total_token_estimate or 0,
        compiled.response_contract,
    )


def parse_isolated(
    task: RelationalTask, request: IsolatedRequest, content: str
) -> dict[str, Any]:
    try:
        ballot = parse_relational_ballot(
            content, tuple(request.option_mapping), request.option_mapping
        )
        if (
            ballot.vote is None
            or ballot.reason is None
            or not ballot.shared_fact_present
        ):
            raise ValueError("response lacks a valid vote, reason, or shared_fact_id")
        answer = request.option_mapping.get(ballot.vote, ballot.vote)
        if answer not in task.semantic_answers:
            raise ValueError("vote does not resolve to a semantic allocation")
        if (
            ballot.shared_fact_id is not None
            and ballot.shared_fact_id not in request.evidence_ids
        ):
            raise ValueError("shared_fact_id is unavailable")
        return {
            "parse_success": True,
            "selected_letter": ballot.vote,
            "semantic_answer": answer,
            "gold_selected": answer == task.correct_relation,
            "false_target_selected": answer == task.controller_target,
            "other_selected": answer
            not in {task.correct_relation, task.controller_target},
            "reason": ballot.reason,
            "shared_fact_id": ballot.shared_fact_id,
            "parse_error": None,
        }
    except (TypeError, ValueError) as exc:
        return {
            "parse_success": False,
            "selected_letter": None,
            "semantic_answer": None,
            "gold_selected": None,
            "false_target_selected": None,
            "other_selected": None,
            "reason": None,
            "shared_fact_id": None,
            "parse_error": str(exc),
        }


__all__ = ["IsolatedPrompt", "parse_isolated", "render_isolated"]
