"""HiddenBench discussion and voting templates from the paper appendix."""

from __future__ import annotations

from mas_cc.core import MessageRole

from ..blocks import PromptBlock
from ..context import PromptContext
from ..contracts import ResponseContract
from ..registry import PromptDefinition
from ..versions import PromptVersion


def _scenario(context: PromptContext, _: ResponseContract) -> str:
    return context.task_description


def _information(context: PromptContext, _: ResponseContract) -> str:
    information = context.private_state.get("information", ())
    facts = "\n".join(f"- {fact}" for fact in information)
    return (
        "You have received the following information. Notice that the order of this "
        "information is randomly shuffled; the order of facts does not indicate "
        "importance or relationship. Please reason carefully:\n"
        f"{facts}"
    )


def _brevity(_: PromptContext, __: ResponseContract) -> str:
    return "Keep your response concise—just one or two sentences."


def _transcript(context: PromptContext) -> str:
    return "\n".join(
        f"Agent {entry.get('speaker_id')}: {entry.get('message')}"
        for entry in context.recent_memory
    )


def _discussion_turn(context: PromptContext, _: ResponseContract) -> str:
    if not context.recent_memory:
        return "You are the first to speak."
    return (
        "Previous messages from other people:\n"
        f"{_transcript(context)}\n\n"
        "It's your turn to speak."
    )


def _vote(context: PromptContext, contract: ResponseContract) -> str:
    prefix = ""
    if context.recent_memory:
        prefix = (
            "Previous messages from other people:\n"
            f"{_transcript(context)}\n\n"
        )
    return prefix + contract.instruction()


def discussion_prompt_definition() -> PromptDefinition:
    blocks = (
        PromptBlock("scenario_description", "Scenario description", MessageRole.SYSTEM, _scenario),
        PromptBlock("available_information", "Available shuffled information", MessageRole.SYSTEM, _information),
        PromptBlock("brevity_instruction", "Brevity instruction", MessageRole.SYSTEM, _brevity),
        PromptBlock("discussion_turn", "Public transcript and speaking turn", MessageRole.USER, _discussion_turn),
    )
    return PromptDefinition(
        prompt_version=PromptVersion("hidden_profile_discussion_paper", 1),
        blocks=blocks,
        required_blocks=tuple(block.name for block in blocks),
    )


def vote_prompt_definition() -> PromptDefinition:
    blocks = (
        PromptBlock("scenario_description", "Scenario description", MessageRole.SYSTEM, _scenario),
        PromptBlock("available_information", "Available shuffled information", MessageRole.SYSTEM, _information),
        PromptBlock("brevity_instruction", "Brevity instruction", MessageRole.SYSTEM, _brevity),
        PromptBlock("vote_request", "Public transcript and vote request", MessageRole.USER, _vote),
    )
    return PromptDefinition(
        prompt_version=PromptVersion("hidden_profile_vote_paper", 1),
        blocks=blocks,
        required_blocks=tuple(block.name for block in blocks),
    )
