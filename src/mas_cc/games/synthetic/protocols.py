"""The synthetic-game contract: a normal `Game`, plus an answer key.

A synthetic game is a real game as far as the rest of the framework is
concerned - same `Game` ABC, same decision loop, same recorder - with two
extra obligations that only make sense when we wrote the dynamics ourselves:

`ground_truth()`   the closed-form value of every quantity the system will
                   later estimate, computed *from the resolved config that
                   ran*. Never a hand-written number in a separate file: the
                   whole point is that changing epsilon in the YAML cannot
                   leave a stale expected value behind for someone to lose a
                   day to.

`simulate()`       the same dynamics again, vectorized, with no prompts and no
                   recorder ("speed mode"). It exists so sweeps that need
                   hundreds of seeds are affordable, and so the *pair* of
                   implementations can be checked against each other - if the
                   full pipeline and the bare sampler disagree on the same
                   seed, the pipeline has a bug, which is the entire reason
                   these games exist.

Both are abstract here rather than conventions, so a fourth synthetic game
cannot quietly ship without an answer key.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from mas_cc.config import GameConfig
from mas_cc.core import AgentId, InteractionId
from mas_cc.games.protocols import Action, DecisionRequest, Game, GameState, Transition
from mas_cc.llm_runtime.prompts import CompiledPrompt
from mas_cc.runtime import ValidationAttempt


@dataclass(frozen=True, slots=True)
class GroundTruthQuantity:
    """One analytically known number, with enough identity to join on.

    ``subject`` is what the quantity is *about* - the agent pair for a
    cross-agent MI, a single agent for a marginal entropy, empty for a
    population-wide quantity. It is part of the identity because a run reports
    many values under one name (`mutual_information` for every pair), and the
    comparison table has to line each estimate up with its own truth.
    """

    name: str
    value: float
    subject: tuple[str, ...] = ()
    units: str = "bits"
    definition: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("GroundTruthQuantity.name must be non-empty")
        object.__setattr__(self, "subject", tuple(str(item) for item in self.subject))
        object.__setattr__(self, "value", float(self.value))

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        return self.name, self.subject

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "subject": list(self.subject),
            "units": self.units,
            "definition": self.definition,
        }


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Every closed-form quantity for one resolved synthetic config."""

    game_type: str
    parameters: Mapping[str, Any]
    quantities: tuple[GroundTruthQuantity, ...]

    def __post_init__(self) -> None:
        quantities = tuple(self.quantities)
        keys = [quantity.key for quantity in quantities]
        if len(set(keys)) != len(keys):
            raise ValueError("GroundTruth quantities must be unique by (name, subject)")
        object.__setattr__(self, "quantities", quantities)

    def value(self, name: str, subject: Sequence[str] = ()) -> float:
        key = (name, tuple(str(item) for item in subject))
        for quantity in self.quantities:
            if quantity.key == key:
                return quantity.value
        raise KeyError(f"no ground-truth quantity {key!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "game_type": self.game_type,
            "parameters": dict(self.parameters),
            "quantities": [quantity.to_dict() for quantity in self.quantities],
        }


@dataclass(frozen=True, slots=True)
class SimulatedEpisodes:
    """Speed-mode output: action indices for every (seed, round, agent).

    Indices rather than labels because everything downstream of this is
    counting, and an integer array of shape ``(seeds, rounds, agents)`` is what
    makes a 500-seed sweep a numpy operation instead of a Python loop.
    ``action_labels`` maps them back to what the recorded episode wrote, which
    is what the fidelity/speed comparison joins on.
    """

    seeds: tuple[int, ...]
    actions: np.ndarray
    action_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "action_labels", tuple(self.action_labels))
        actions = np.asarray(self.actions)
        if actions.ndim != 3:
            raise ValueError("SimulatedEpisodes.actions must be (seeds, rounds, agents)")
        if actions.shape[0] != len(self.seeds):
            raise ValueError("SimulatedEpisodes.actions must have one plane per seed")
        if actions.size and int(actions.max()) >= len(self.action_labels):
            raise ValueError("SimulatedEpisodes.actions indexes an unknown action label")
        object.__setattr__(self, "actions", actions)

    @property
    def rounds(self) -> int:
        return int(self.actions.shape[1])

    @property
    def agents(self) -> int:
        return int(self.actions.shape[2])

    def labelled(self, seed_index: int) -> list[list[str]]:
        """One episode as ``[round][agent]`` labels, for comparing against a recorded run."""

        labels = self.action_labels
        return [[labels[index] for index in row] for row in self.actions[seed_index]]


class SyntheticGame(Game, ABC):
    """A `Game` whose answers we already know."""

    @abstractmethod
    def ground_truth(self, config: GameConfig) -> GroundTruth:
        """Closed-form values for this exact resolved config."""

    @abstractmethod
    def simulate(self, config: GameConfig, seeds: Sequence[int]) -> SimulatedEpisodes:
        """The same dynamics without the pipeline, vectorized across seeds."""


class SyntheticTransition(Transition):
    """A `Transition` that also answers the two questions `RunRecorder` asks.

    `RunRecorder.record_interaction` reads `transition.success` and
    `transition.payoff`. Supplying them here is what lets a synthetic episode
    reuse the real recorder unchanged - which matters, because rehearsing the
    real recorder is most of the point.
    """

    __slots__ = ()

    @property
    def success(self) -> bool:
        """Whether the whole population played the same action this round.

        Synthetic games have no payoff structure to succeed at, so `success`
        carries the only per-round binary event they do have: unanimity.
        """

        return bool(self.matched)

    @property
    def payoff(self) -> float:
        return float(next(iter(self.payoffs.values()), 0.0))


@dataclass(frozen=True, slots=True)
class SyntheticDecision:
    """One agent's observation-to-prompt-to-response-to-action chain."""

    request: DecisionRequest
    action: Action
    compiled_prompt: CompiledPrompt
    attempts: tuple[ValidationAttempt, ...]

    @property
    def validation_attempts(self) -> int:
        return len(self.attempts)

    @property
    def provider_retries(self) -> int:
        return 0

    @property
    def prompt_definition_hash(self) -> str:
        return self.compiled_prompt.definition_hash

    @property
    def prompt_instance_hash(self) -> str:
        return self.compiled_prompt.instance_hash


@dataclass(frozen=True, slots=True)
class SyntheticInteractionRecord:
    interaction_id: InteractionId
    interaction_index: int
    participants: tuple[AgentId, ...]
    decisions: tuple[SyntheticDecision, ...]
    transition: SyntheticTransition

    def __post_init__(self) -> None:
        object.__setattr__(self, "participants", tuple(self.participants))
        object.__setattr__(self, "decisions", tuple(self.decisions))
        if tuple(decision.action.agent_id for decision in self.decisions) != self.participants:
            raise ValueError("synthetic decisions must follow participant order")


@dataclass(frozen=True, slots=True)
class SyntheticGameResult:
    initial_state: GameState
    final_state: GameState
    interactions: tuple[SyntheticInteractionRecord, ...]
    termination_reason: str
    ground_truth: GroundTruth
    logical_decisions: int = 0
    validation_attempts: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "interactions", tuple(self.interactions))

    def action_series(self) -> dict[str, list[str]]:
        """Recorded actions as ``{agent_id: [round_1, round_2, ...]}``.

        The in-memory counterpart of reading `metrics/streaming.csv` back off
        the cluster; both must agree, and the parity check uses whichever is
        at hand.
        """

        series: dict[str, list[str]] = {}
        for interaction in self.interactions:
            for decision in interaction.decisions:
                series.setdefault(str(decision.action.agent_id), []).append(decision.action.value)
        return series
