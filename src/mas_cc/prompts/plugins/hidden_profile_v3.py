"""Benchmark-only HiddenBench concrete full prompts.

These fixtures intentionally do not imply that a HiddenBench game runner exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mas_cc.core import MessageRole

from ..blocks import UNBOUND, PromptBlock
from ..contracts import ResponseContract
from ..full_prompt import FullPrompt
from .._values import thaw


@dataclass(frozen=True, slots=True)
class HiddenProfileBlock(PromptBlock[Any]):
    def render(self) -> str:
        if self.value is UNBOUND:
            raise ValueError(f"prompt.blocks.{self.name}.value is UNBOUND")
        if isinstance(self.value, str):
            return self.value
        return json.dumps(thaw(self.value), ensure_ascii=False, sort_keys=True)


class HiddenProfileFullPrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return "hidden_profile_benchmark"


def _base_blocks(*, include_transcript: bool) -> tuple[PromptBlock[Any], ...]:
    values: list[PromptBlock[Any]] = [
        HiddenProfileBlock(
            "scenario", "Scenario", MessageRole.SYSTEM,
            UNBOUND,
        ),
        HiddenProfileBlock(
            "private_information", "Private information", MessageRole.SYSTEM,
            UNBOUND, sensitive=True,
        ),
    ]
    if include_transcript:
        values.append(
            HiddenProfileBlock(
                "transcript", "Visible transcript", MessageRole.SYSTEM, (), required=False
            )
        )
    return tuple(values)


def hidden_profile_discussion_prompt() -> HiddenProfileFullPrompt:
    return HiddenProfileFullPrompt(
        "hidden_profile_discussion",
        2,
        _base_blocks(include_transcript=True),
        ResponseContract(
            "discussion_turn",
            options={"decision_instruction": "Write one concise discussion contribution."},
        ),
    )


def hidden_profile_vote_prompt() -> HiddenProfileFullPrompt:
    return HiddenProfileFullPrompt(
        "hidden_profile_vote",
        2,
        _base_blocks(include_transcript=True),
        ResponseContract("json_vote", ("A", "B", "C")),
    )
