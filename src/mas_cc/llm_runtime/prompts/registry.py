"""Explicit factory registry for immutable full prompts."""

from __future__ import annotations

from collections.abc import Callable

from .full_prompt import FullPrompt
from .versions import PromptVersion

PromptFactory = Callable[[], FullPrompt]


class PromptRegistry:
    """Connection-free prompt factory lookup with no discovery side effects."""

    def __init__(self) -> None:
        self._factories: dict[PromptVersion, PromptFactory] = {}

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


def create_default_prompt_registry() -> PromptRegistry:
    """Return an empty, connection-free registry with no built-in content.

    The portable prompt kernel ships no game- or paper-specific prompt
    definitions; callers register their own ``FullPrompt`` factories via
    :meth:`PromptRegistry.register`.
    """

    return PromptRegistry()
