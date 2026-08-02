"""Explicit factory registry for immutable full prompts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .blocks import PromptBlock
from .full_prompt import FullPrompt
from .versions import PromptVersion

PromptFactory = Callable[[], FullPrompt]


class PromptRegistry:
    """Connection-free prompt factory lookup with no discovery side effects."""

    def __init__(self) -> None:
        self._factories: dict[PromptVersion, PromptFactory] = {}
        self._legacy_definitions: dict[PromptVersion, PromptDefinition] = {}

    def register(self, prompt: FullPrompt | PromptFactory) -> None:
        factory = prompt if callable(prompt) and not isinstance(prompt, FullPrompt) else lambda: prompt  # type: ignore[assignment,return-value]
        created = factory()
        if not isinstance(created, FullPrompt):
            raise TypeError("prompt factory must return a FullPrompt")
        key = PromptVersion(created.family, created.version)
        if key in self._factories:
            raise ValueError(f"prompt {key} is already registered")
        self._factories[key] = factory

    def get(self, family: str, version: int) -> FullPrompt:
        key = PromptVersion(family, version)
        try:
            prompt = self._factories[key]()
        except KeyError as exc:
            available = ", ".join(str(item) for item in sorted(self._factories)) or "none"
            raise ValueError(
                f"prompt.version: {key} is not registered; available: {available}"
            ) from exc
        if (prompt.family, prompt.version) != (family, version):
            raise RuntimeError(f"prompt factory for {key} returned a different identity")
        return prompt

    def versions(self) -> tuple[PromptVersion, ...]:
        return tuple(sorted(self._factories))

    # The methods below isolate the Version 1 renderer path while historical
    # fixtures transition.  New runtime code never calls them.
    def register_legacy(self, definition: "PromptDefinition") -> None:
        key = definition.prompt_version
        if key in self._legacy_definitions:
            raise ValueError(f"legacy prompt {key} is already registered")
        self._legacy_definitions[key] = definition

    def get_legacy(self, family: str, version: int) -> "PromptDefinition":
        key = PromptVersion(family, version)
        try:
            return self._legacy_definitions[key]
        except KeyError as exc:
            raise ValueError(f"legacy prompt.version: {key} is not registered") from exc


# Kept at its historical module path solely so the isolated compatibility
# plugins can still be imported.  It is intentionally absent from __all__.
if TYPE_CHECKING:
    from .compatibility import PromptDefinition
else:
    @dataclass(frozen=True, slots=True)
    class PromptDefinition:
        prompt_version: PromptVersion
        blocks: tuple[Any, ...]
        required_blocks: tuple[str, ...] = ()

        def __post_init__(self) -> None:
            blocks = tuple(self.blocks)
            names = tuple(block.name for block in blocks)
            if not blocks or len(set(names)) != len(names):
                raise ValueError("legacy PromptDefinition blocks must be non-empty and unique")
            unknown = set(self.required_blocks) - set(names)
            if unknown:
                raise ValueError(
                    f"legacy PromptDefinition has unknown required block {sorted(unknown)[0]!r}"
                )
            object.__setattr__(self, "blocks", blocks)
            object.__setattr__(self, "required_blocks", tuple(self.required_blocks))

        def block(self, name: str) -> Any:
            for block in self.blocks:
                if block.name == name:
                    return block
            raise KeyError(name)


def create_default_prompt_registry(
    *, include_legacy: bool = False
) -> PromptRegistry:
    """Build game-neutral example/benchmark registrations without discovery or I/O."""

    from .plugins.basic_choice_v3 import basic_choice_prompt
    from .plugins.hidden_profile_v3 import (
        hidden_profile_discussion_prompt,
        hidden_profile_vote_prompt,
    )

    registry = PromptRegistry()
    registry.register(basic_choice_prompt)
    registry.register(hidden_profile_discussion_prompt)
    registry.register(hidden_profile_vote_prompt)

    if include_legacy:
        from .plugins.ashery_2025 import prompt_definition as ashery
        from .plugins.basic_binary_choice import prompt_definition as basic
        from .plugins.hidden_profile_paper import (
            discussion_prompt_definition,
            vote_prompt_definition,
        )
        from .plugins.social_conventions_paper import prompt_definition as social

        for definition in (
            ashery(), basic(), social(), discussion_prompt_definition(), vote_prompt_definition()
        ):
            registry.register_legacy(definition)
    return registry
