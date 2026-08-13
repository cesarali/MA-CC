"""Lazy registry for provider-neutral game implementations."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mas_cc.config import GameConfig
from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.llm_runtime.validation import ValidationIssue
from mas_cc.llm_runtime.prompts import PromptRegistry

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
    registry.register(
        "synthetic_bernoulli",
        "mas_cc.games.synthetic.bernoulli.game:SyntheticBernoulliGame",
    )
    registry.register(
        "synthetic_markov",
        "mas_cc.games.synthetic.markov.game:SyntheticMarkovGame",
    )
    registry.register(
        "synthetic_controlled_markov",
        "mas_cc.games.synthetic.controlled_markov.game:SyntheticControlledMarkovGame",
    )
    registry.register(
        "hidden_bench_vanilla",
        "mas_cc.games.hidden_bench.vanilla.game:HiddenBenchVanillaGame",
    )
    registry.register(
        "hidden_bench_naming",
        "mas_cc.games.hidden_bench.naming.game:HiddenBenchNamingGame",
    )
    registry.register(
        "hidden_bench_imitation",
        "mas_cc.games.hidden_bench.imitation.game:HiddenBenchImitationGame",
    )
    registry.register(
        "hidden_bench_imitation_round_feedback",
        "mas_cc.games.hidden_bench.imitation_round_feedback.game:HiddenBenchImitationRoundFeedbackGame",
    )
    return registry


def create_default_prompt_registry() -> PromptRegistry:
    """Build game-neutral example/benchmark registrations without discovery or I/O.

    The portable prompt kernel (``mas_cc.llm_runtime.prompts``) ships with no
    built-in content; this repository's fixture-specific prompt definitions
    live under ``games/prompt_library/`` and are wired in here.
    """

    from .prompt_library.basic_choice_v3 import basic_choice_prompt
    from .prompt_library.hidden_profile_v3 import (
        hidden_profile_discussion_prompt,
        hidden_profile_vote_prompt,
    )

    registry = PromptRegistry()
    registry.register(basic_choice_prompt)
    registry.register(hidden_profile_discussion_prompt)
    registry.register(hidden_profile_vote_prompt)
    return registry


def register_game_prompt_factories(registry: PromptRegistry) -> PromptRegistry:
    """Register concrete prompts at the application boundary that owns the games."""

    from .hidden_bench.naming.prompts import (
        hidden_bench_naming_commit_prompt,
        hidden_bench_naming_message_prompt,
    )
    from .hidden_bench.vanilla.prompts import (
        hidden_bench_discussion_prompt,
        hidden_bench_vote_prompt,
    )
    from .hidden_bench.imitation.prompts import (
        PromptStyle,
        hidden_bench_imitation_initial_prompt,
        hidden_bench_imitation_message_prompt,
        hidden_bench_imitation_update_prompt,
    )
    from .naming_convention.prompts import naming_convention_prompt
    from .synthetic.prompts import synthetic_agent_decision_prompt
    from .toy_coordination.prompts import toy_coordination_prompt

    registry.register(naming_convention_prompt)
    registry.register(toy_coordination_prompt)
    registry.register(synthetic_agent_decision_prompt)
    # Two families per HiddenBench game: the games switch between them by phase.
    registry.register(hidden_bench_discussion_prompt)
    registry.register(hidden_bench_vote_prompt)
    registry.register(hidden_bench_naming_message_prompt)
    registry.register(hidden_bench_naming_commit_prompt)
    registry.register(hidden_bench_imitation_initial_prompt)
    registry.register(hidden_bench_imitation_message_prompt)
    registry.register(hidden_bench_imitation_update_prompt)
    # The imitation game is the one game whose prompt *text* is a study
    # condition, so it ships two versions of each family and both must be
    # registered.  A run at `prompt_version: 2` otherwise completes its
    # episodes and then dies in config export on "not registered".
    #
    # `inform_asymmetry` is deliberately not a third key: it is a variant of
    # v2, not a version of its own, and the registry holds exactly one prompt
    # per family@version.  The exported definition therefore shows the v2 shape
    # without the asymmetry notice; the bound prompts in the run carry their
    # own definition hashes either way.
    imitation_v2 = PromptStyle(version=2)
    registry.register(lambda: hidden_bench_imitation_initial_prompt(style=imitation_v2))
    registry.register(lambda: hidden_bench_imitation_message_prompt(imitation_v2))
    registry.register(lambda: hidden_bench_imitation_update_prompt(style=imitation_v2))
    return registry


def create_game(config: GameConfig, *, registry: GameRegistry | None = None) -> Game:
    return (registry or create_default_game_registry()).create(config)


def game_metrics(game: Game) -> tuple[tuple[Any, ...], Callable[[Any], Any] | None]:
    """This game's declared metrics and its ``RoundView`` adapter.

    Best effort: a game with no ``metrics`` module records nothing, which is
    how the frozen Phase 7 gate stays unaffected.

    The metrics module is resolved from the game class's *own package* rather
    than from its ``game_type`` string. Those coincide for a game that lives at
    ``games/<game_type>/``, but not for one nested a level deeper (the
    synthetic games sit under ``games/synthetic/``), and a game should not have
    to be named after its directory to have its metrics found.

    Metrics that declare a ``requires_game_family`` are checked against the
    game's own ``spec.game_family`` here - the one place where a concrete game
    and its metric list are both in hand - so an option-share metric attached
    to a game with no option set fails at wiring time rather than silently
    producing a column of zeros.
    """

    package = type(game).__module__.rpartition(".")[0]
    try:
        module = importlib.import_module(f"{package}.metrics")
    except ImportError:
        return (), None
    metrics = tuple(getattr(module, "METRICS", ()))
    mismatched = [
        f"{metric.name} (requires {metric.requires_game_family!r})"
        for metric in metrics
        if getattr(metric, "requires_game_family", None) not in (None, game.spec.game_family)
    ]
    if mismatched:
        raise ValueError(
            f"game {game.spec.game_type!r} is family {game.spec.game_family!r} but declares "
            f"metrics for another family: {', '.join(mismatched)}"
        )
    return metrics, getattr(module, "to_round_view", None)
