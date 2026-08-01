"""Prompt-family registry with explicit integer version lookup."""

from __future__ import annotations

from dataclasses import dataclass

from .blocks import PromptBlock
from .versions import PromptVersion


@dataclass(frozen=True, slots=True)
class PromptDefinition:
    """All available blocks for one immutable prompt family version."""

    prompt_version: PromptVersion
    blocks: tuple[PromptBlock, ...]
    required_blocks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        blocks = tuple(self.blocks)
        names = tuple(block.name for block in blocks)
        if not blocks:
            raise ValueError("PromptDefinition.blocks must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("PromptDefinition.blocks contains a duplicate name")
        required = tuple(self.required_blocks)
        unknown = set(required) - set(names)
        if unknown:
            raise ValueError(
                f"PromptDefinition.required_blocks contains unknown block {sorted(unknown)[0]!r}"
            )
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "required_blocks", required)

    def block(self, name: str) -> PromptBlock:
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(name)


class PromptRegistry:
    """In-memory registry; constructing it performs no plugin discovery or I/O."""

    def __init__(self) -> None:
        self._definitions: dict[PromptVersion, PromptDefinition] = {}

    def register(self, definition: PromptDefinition) -> None:
        key = definition.prompt_version
        if key in self._definitions:
            raise ValueError(f"prompt {key} is already registered")
        self._definitions[key] = definition

    def get(self, family: str, version: int) -> PromptDefinition:
        key = PromptVersion(family, version)
        try:
            return self._definitions[key]
        except KeyError as exc:
            available = ", ".join(str(item) for item in sorted(self._definitions)) or "none"
            raise ValueError(f"prompt.version: {key} is not registered; available: {available}") from exc

    def versions(self) -> tuple[PromptVersion, ...]:
        return tuple(sorted(self._definitions))


def create_default_prompt_registry() -> PromptRegistry:
    """Create a fresh registry containing built-in prompt plugins."""

    from .plugins.basic_binary_choice import prompt_definition
    from .plugins.hidden_profile_paper import (
        discussion_prompt_definition,
        vote_prompt_definition,
    )
    from .plugins.social_conventions_paper import prompt_definition as conventions_definition

    registry = PromptRegistry()
    registry.register(prompt_definition())
    registry.register(conventions_definition())
    registry.register(discussion_prompt_definition())
    registry.register(vote_prompt_definition())
    return registry
