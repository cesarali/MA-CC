"""Game 3 - Controlled Markov. Game 2 plus an exogenous input, and the positive control.

Everything is Game 2 except that an external controller sets ``u`` for the
episode, and a specified subset of agents is pushed toward one action with a
given strength:

    with probability `strength`, a controlled agent reports target(u)
    otherwise it plays the ordinary Game 2 dynamics

**Why this game has to exist.** Sweeping a config parameter is not steering.
On Game 1 the terminal empowerment ``I(C;O)`` is exactly zero for every sweep,
and on symmetric Game 2 it is zero again - because `epsilon` and `N` change how
noisy the population is, not which way it goes. Those zeros are correct and
diagnostic, but they are all nulls. Empowerment in the intended sense needs an
input that actually moves the population, and that is exactly ``u``. This is
the one place in the harness where a *nonzero* empowerment has a known target.

Two numbers worth having, because they answer different questions:

- **Design MI** - ``I(U;S)`` with ``p(u)`` fixed to the grid actually swept.
  This is what the estimator is targeting and the number it must reproduce.
- **Capacity** - ``max_p(u) I(U;S)``, by Blahut-Arimoto over at most 1024
  states. This is the ceiling: how much control authority exists at all,
  independent of how you chose to sample it. Comparing the two is what
  separates "the controller is weak" from "the grid was badly chosen".

``u`` is held constant for an episode rather than redrawn each round, which is
what makes it a *condition* in the sweep sense and lets the family-2 machinery
in `empowerment.py` treat it with ``c := u``, exactly as the ground-truth
document sets out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from mas_cc.config import GameConfig
from mas_cc.core import AgentId
from mas_cc.games.protocols import GameSpec, GameState

from .. import exact
from ..noise import episode_generator
from ..markov.game import (
    MarkovSpec,
    SyntheticMarkovGame,
    initial_distribution,
    pooled_laws,
    transition_matrix,
)
from ..protocols import GroundTruth, GroundTruthQuantity

CONTROL_STREAM = "controlled_markov.control_application"

NO_PUSH = -1
"""The control value meaning "apply no push this episode".

Kept as an explicit member of the alphabet rather than an absence, because a
grid that includes a genuine do-nothing condition is the cleanest way to see
whether the estimator recovers *zero* influence for it.
"""


@dataclass(frozen=True, slots=True)
class ControlSpec:
    """The controller: who it pushes, how hard, and toward what."""

    value: int
    targets: tuple[int, ...]
    strength: float
    agents: tuple[int, ...]

    @classmethod
    def from_config(cls, config: GameConfig, population_size: int) -> "ControlSpec":
        options = config.options
        raw_targets = options.get("control_targets", (0, 1))
        targets = tuple(int(item) for item in raw_targets)
        value = int(options.get("control_value", 0))
        strength = float(options.get("control_strength", 0.5))
        raw_agents = options.get("controlled_agents")
        if raw_agents is None:
            # Half the population by default: pushing everyone makes the
            # population state a deterministic function of u and the answer
            # stops being interesting, while pushing nobody makes it zero.
            agents = tuple(range(max(1, population_size // 2)))
        else:
            agents = tuple(int(item) for item in raw_agents)
        spec = cls(value=value, targets=targets, strength=strength, agents=agents)
        spec.validate(population_size)
        return spec

    def validate(self, population_size: int) -> None:
        if not self.targets:
            raise ValueError("game.options.control_targets must not be empty")
        if any(target not in (NO_PUSH, 0, 1) for target in self.targets):
            raise ValueError(
                f"game.options.control_targets entries must be 0, 1, or {NO_PUSH} (no push)"
            )
        if not 0 <= self.value < len(self.targets):
            raise ValueError("game.options.control_value must index control_targets")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("game.options.control_strength must lie in [0, 1]")
        if any(not 0 <= agent < population_size for agent in self.agents):
            raise ValueError("game.options.controlled_agents must be valid agent indices")
        if len(set(self.agents)) != len(self.agents):
            raise ValueError("game.options.controlled_agents must be unique")

    @property
    def target_action(self) -> int:
        return self.targets[self.value]

    @property
    def is_active(self) -> bool:
        return self.target_action != NO_PUSH and self.strength > 0.0 and bool(self.agents)

    def chain_control(self, population_size: int) -> tuple | None:
        """``(strength, target)`` per agent, as the exact transition matrix needs it."""

        if not self.is_active:
            return None
        target = float(self.target_action)
        return tuple(
            (self.strength, target) if index in set(self.agents) else None
            for index in range(population_size)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_value": self.value,
            "control_targets": list(self.targets),
            "control_strength": self.strength,
            "controlled_agents": list(self.agents),
            "target_action": self.target_action,
        }


class SyntheticControlledMarkovGame(SyntheticMarkovGame):
    """Game 2 with an exogenous control input held constant across the episode."""

    spec = GameSpec(
        game_type="synthetic_controlled_markov",
        version=1,
        description=(
            "Synthetic coupled-Markov game with an exogenous control input and exact "
            "control-to-population channel quantities."
        ),
        game_family="choice",
        minimum_population=2,
        supported_topologies=("complete",),
    )

    def control(self, config: GameConfig) -> ControlSpec:
        return ControlSpec.from_config(config, config.population_size)

    def chain_control(self, config: GameConfig) -> tuple | None:
        return self.control(config).chain_control(config.population_size)

    def _control_tape(self, seed: int, rules: MarkovSpec) -> np.ndarray:
        return episode_generator(seed, CONTROL_STREAM).random(
            (rules.rounds, rules.population_size)
        )

    def round_control(self, config: GameConfig, state: GameState) -> tuple | None:
        rules = self.rules(config)
        control = self.control(config)
        if not control.is_active:
            return None
        tape = self._control_tape(int(state.data["seed"]), rules)
        controlled = set(control.agents)
        label = rules.actions[control.target_action]
        return tuple(
            label
            if index in controlled and tape[state.turn, index] < control.strength
            else None
            for index in range(rules.population_size)
        )

    def simulation_control(self, config: GameConfig, seeds: Sequence[int]) -> Any:
        """Which (episode, round, agent) cells the controller pushes, and toward what.

        Drawn from its own named stream, so introducing a controller leaves
        every uncontrolled episode's draws bit-identical - a Game 3 config with
        zero strength replays exactly as the Game 2 config it came from.
        """

        rules = self.rules(config)
        control = self.control(config)
        if not control.is_active:
            return None
        tape = np.stack([self._control_tape(seed, rules) for seed in seeds])
        mask = np.zeros(rules.population_size, dtype=bool)
        mask[list(control.agents)] = True
        return (tape < control.strength) & mask[None, None, :], control.target_action

    def apply_control(self, profile: np.ndarray, control: Any, round_index: int) -> np.ndarray:
        applies, target = control
        return np.where(applies[:, round_index, :], np.int8(target), profile).astype(np.int8)

    def ground_truth(self, config: GameConfig) -> GroundTruth:
        """Game 2's quantities, plus what the control input is worth."""

        truth = super().ground_truth(config)
        rules = self.rules(config)
        control = self.control(config)
        quantities = list(truth.quantities)

        # The control channel: p(state | u) for every u the grid can take. This
        # is a property of the *family* of chains, so it is computed here rather
        # than at the sweep - the sweep supplies p(u), this supplies p(s | u).
        channel, macro_channel = control_channel(rules, control, rounds=rules.rounds)
        uniform = np.full(len(control.targets), 1.0 / len(control.targets))
        quantities.append(
            GroundTruthQuantity(
                name="control_design_mutual_information",
                value=exact.design_mutual_information(uniform, macro_channel),
                definition=(
                    "I(U; macrostate at the final round) under a uniform grid over "
                    "control_targets - what the estimator targets for an equal-repetition sweep."
                ),
            )
        )
        capacity, optimal = exact.blahut_arimoto(macro_channel)
        quantities.append(
            GroundTruthQuantity(
                name="control_capacity",
                value=capacity,
                definition=(
                    "max_p(u) I(U; macrostate), by Blahut-Arimoto. The ceiling on control "
                    "authority, independent of how the grid sampled it."
                ),
            )
        )
        for index, weight in enumerate(optimal):
            quantities.append(
                GroundTruthQuantity(
                    name="capacity_achieving_input_probability",
                    value=float(weight),
                    subject=(str(index),),
                    units="probability",
                    definition="The input law achieving control_capacity.",
                )
            )
        quantities.append(
            GroundTruthQuantity(
                name="control_microstate_capacity",
                value=exact.blahut_arimoto(channel)[0],
                definition=(
                    "The same ceiling against the full microstate rather than the "
                    "macrostate; the gap between the two is what coarse-graining discards."
                ),
            )
        )
        return GroundTruth(
            game_type=self.spec.game_type,
            parameters={**truth.parameters, **control.to_dict()},
            quantities=tuple(quantities),
        )


def control_channel(
    rules: MarkovSpec, control: ControlSpec, *, rounds: int
) -> tuple[np.ndarray, np.ndarray]:
    """``P(state | u)`` at ``rounds``, over microstates and over macrostates.

    One row per control value in the alphabet, so this is literally the channel
    from controller to population that the capacity is taken over.
    """

    bits = exact.bit_table(rules.population_size, rules.n_states)
    micro = np.zeros((len(control.targets), rules.n_states), dtype=float)
    macro = np.zeros((len(control.targets), rules.population_size + 1), dtype=float)
    for index in range(len(control.targets)):
        variant = ControlSpec(
            value=index, targets=control.targets,
            strength=control.strength, agents=control.agents,
        )
        matrix = transition_matrix(rules, variant.chain_control(rules.population_size))
        law = exact.propagate(initial_distribution(rules), matrix, rounds)
        micro[index] = law
        macro[index] = exact.macrostate_law(law, bits)
    return micro, macro
