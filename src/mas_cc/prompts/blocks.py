"""Reusable prompt block definitions and rendered block records."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mas_cc.core import MessageRole

if TYPE_CHECKING:
    from .context import PromptContext
    from .contracts import ResponseContract

BlockRenderer = Callable[["PromptContext", "ResponseContract"], str]


@dataclass(frozen=True, slots=True)
class PromptBlock:
    """A named, versioned renderer for exactly one prompt concern."""

    name: str
    title: str
    role: MessageRole
    renderer: BlockRenderer
    version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("PromptBlock.name must be non-empty")
        if not isinstance(self.title, str) or not self.title:
            raise ValueError("PromptBlock.title must be non-empty")
        if isinstance(self.role, str):
            object.__setattr__(self, "role", MessageRole(self.role))
        if not callable(self.renderer):
            raise TypeError("PromptBlock.renderer must be callable")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("PromptBlock.version must be a positive integer")

    def render(
        self, context: "PromptContext", response_contract: "ResponseContract"
    ) -> "RenderedPromptBlock":
        content = self.renderer(context, response_contract)
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"prompt.blocks.{self.name}: renderer returned empty content")
        return RenderedPromptBlock(
            name=self.name,
            title=self.title,
            role=self.role,
            content=content.strip(),
            version=self.version,
        )


@dataclass(frozen=True, slots=True)
class RenderedPromptBlock:
    """The inspectable output of rendering one block."""

    name: str
    title: str
    role: MessageRole
    content: str
    version: int
    token_count: int | None = None

    def to_dict(self, *, order: int | None = None) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "title": self.title,
            "role": self.role.value,
            "version": self.version,
            "content": self.content,
            "token_count": self.token_count,
        }
        if order is not None:
            result["order"] = order
        return result
