"""Isolated Version 1 prompt compatibility API.

This module is unregistered unless ``create_default_prompt_registry`` is called
with ``include_legacy=True``. Phase 5+ runtime code must not import it or the
legacy context/composer modules.
"""

from dataclasses import dataclass
from typing import Callable

from mas_cc.core import MessageRole

from .blocks import RenderedPromptBlock
from .composer import PromptComposer, PromptInstance
from .context import PromptContext
from .contracts import ResponseContract
from .registry import PromptDefinition

LegacyRenderer = Callable[[PromptContext, ResponseContract], str]


@dataclass(frozen=True, slots=True)
class LegacyPromptBlock:
    name: str
    title: str
    role: MessageRole
    renderer: LegacyRenderer
    version: int = 1

    def render(
        self, context: PromptContext, response_contract: ResponseContract
    ) -> RenderedPromptBlock:
        content = self.renderer(context, response_contract)
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"prompt.blocks.{self.name}: renderer returned empty content")
        return RenderedPromptBlock(
            self.name, self.title, self.role, content.strip(), self.version, 0
        )


__all__ = [
    "LegacyPromptBlock",
    "PromptComposer",
    "PromptContext",
    "PromptDefinition",
    "PromptInstance",
]
