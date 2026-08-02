"""Small concrete full prompt used by Phase 3/4 fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from mas_cc.core import MessageRole, ValidationIssue

from ..blocks import UNBOUND, PromptBlock, Unbound
from ..full_prompt import FullPrompt
from ..contracts import ResponseContract
from .._values import thaw


@dataclass(frozen=True, slots=True)
class BasicChoiceBlock(PromptBlock[Any]):
    def value_issues(self, value: Any) -> tuple[ValidationIssue, ...]:
        if not isinstance(value, (str, tuple, Mapping)):
            return (
                ValidationIssue(
                    f"prompt.blocks.{self.name}.value",
                    "must be text, a sequence, or a mapping",
                    value,
                ),
            )
        return ()

    def render(self) -> str:
        if self.value is UNBOUND:
            raise ValueError(f"prompt.blocks.{self.name}.value is UNBOUND")
        if isinstance(self.value, str):
            return self.value
        return json.dumps(thaw(self.value), ensure_ascii=False, sort_keys=True)


class BasicChoiceFullPrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return "basic_choice"


def basic_choice_prompt() -> BasicChoiceFullPrompt:
    blocks = (
        BasicChoiceBlock(
            "task", "Task", MessageRole.SYSTEM,
            "Choose the option that best matches your private signal.",
            binding="fixed",
        ),
        BasicChoiceBlock(
            "rules", "Rules", MessageRole.SYSTEM,
            ("Choose exactly one available option.", "Do not reveal hidden metadata."),
            binding="fixed",
        ),
        BasicChoiceBlock(
            "private_state", "Private state", MessageRole.SYSTEM, UNBOUND,
            sensitive=True,
        ),
        BasicChoiceBlock(
            "recent_memory", "Recent memory", MessageRole.SYSTEM, (),
        ),
        BasicChoiceBlock(
            "current_interaction", "Current interaction", MessageRole.SYSTEM, UNBOUND,
        ),
    )
    return BasicChoiceFullPrompt(
        family="basic_choice",
        version=1,
        blocks=blocks,
        response_contract=ResponseContract("choice_only", ("A", "B")),
    )
