"""Lazy registry for provider-neutral game implementations."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

from mas_cc.config import GameConfig
from mas_cc.core.exceptions import ConfigurationError
from mas_cc.core.validation import ValidationIssue
from mas_cc.prompts import PromptRegistry

from .protocols import Game

GameFactory = Callable[[], Game]


@dataclass(slots=True)
class GameRegistry:
    _entries: dict[str, str | GameFactory] = field(default_factory=dict)

    def register(self, name: str, factory: str | GameFactory) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("game registry name must be non-empty")
        if normalized in self._entries:
            raise ValueError(f"game {normalized!r} is already registered")
        self._entries[normalized] = factory

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def create(self, config: GameConfig) -> Game:
        name = config.type.strip().lower()
        try:
            factory = self._entries[name]
        except KeyError as exc:
            raise ConfigurationError(
                [
                    ValidationIssue(
                        "game.type",
                        f"unknown game {config.type!r}; available: {', '.join(self.names())}",
                    )
                ],
                context="game creation",
            ) from exc
        if isinstance(factory, str):
            module_name, separator, attribute = factory.partition(":")
            if not separator:
                raise RuntimeError(f"invalid game registry target {factory!r}")
            factory = getattr(importlib.import_module(module_name), attribute)
        game = factory()
        if not isinstance(game, Game):
            raise TypeError(f"game factory for {name!r} returned an incompatible object")
        if game.spec.game_type != name:
            raise ValueError(
                f"game registry entry {name!r} produced spec {game.spec.game_type!r}"
            )
        return game


def create_default_game_registry() -> GameRegistry:
    registry = GameRegistry()
    registry.register(
        "toy_coordination", "mas_cc.games.toy_coordination.game:ToyCoordinationGame"
    )
    registry.register(
        "naming_convention",
        "mas_cc.games.naming_convention.game:NamingConventionGame",
    )
    return registry


def register_game_prompt_factories(registry: PromptRegistry) -> PromptRegistry:
    """Register concrete prompts at the application boundary that owns the games."""

    from .naming_convention.prompts import naming_convention_prompt
    from .toy_coordination.prompts import toy_coordination_prompt

    registry.register(naming_convention_prompt)
    registry.register(toy_coordination_prompt)
    return registry


def create_game(config: GameConfig, *, registry: GameRegistry | None = None) -> Game:
    return (registry or create_default_game_registry()).create(config)
