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

**Two control modes, and the gap between them.** ``control_mode`` decides when
``u`` is drawn:

``episode_fixed``  once per episode, held constant. This is what makes ``u`` a
                   *condition* in the sweep sense and lets the family-2
                   machinery in `empowerment.py` treat it with ``c := u``. It is
                   also the default, so every config written before the mode
                   existed replays bit-identically.
``per_round``      redrawn from the policy at every round, which is what the
                   empowerment paper's ``I(a_t; s_* | s_t)`` assumes.

The distinction is not cosmetic. Under ``episode_fixed`` the state at round
``t`` is causally downstream of ``u``, so conditioning on it blocks most of the
effect being measured and the answer drifts with ``t``; under ``per_round``
``u_t`` has not acted yet when ``s_t`` is observed. `effective_empowerment.py`
computes both on the same chain and reports the difference as a number rather
than leaving it an argument.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from mas_cc.config import GameConfig
from mas_cc.games.protocols import GameSpec, GameState

from .. import exact
from ..effective_empowerment import (
    PER_ROUND,
    EmpowermentReport,
    EmpowermentSpec,
    analyse,
    empowerment_parameters,
    empowerment_quantities,
)
from ..noise import episode_generator
from ..markov.game import (
    MarkovSpec,
    SyntheticMarkovGame,
    initial_distribution,
    transition_matrix,
)
from ..protocols import GroundTruth, GroundTruthQuantity

CONTROL_STREAM = "controlled_markov.control_application"

CONTROL_VALUE_STREAM = "controlled_markov.control_value"
"""Which control value is in force each round, under ``control_mode: per_round``.

Its own stream, so an ``episode_fixed`` episode - which never reads it - keeps
every other draw bit-identical to the day before this mode existed.
"""

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
        targets = _resolve_alphabet(options)
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
        """Whether *this* control value pushes - the episode-fixed question."""

        return self.target_action != NO_PUSH and self.strength > 0.0 and bool(self.agents)

    @property
    def can_push(self) -> bool:
        """Whether *any* control value pushes - the per-round question.

        Distinct from `is_active` because a per-round controller cycles through
        the whole alphabet, so an alphabet containing one do-nothing entry is
        still a live controller; an alphabet of nothing but do-nothing entries,
        or zero strength, is not.
        """

        return (
            self.strength > 0.0
            and bool(self.agents)
            and any(target != NO_PUSH for target in self.targets)
        )

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


def _resolve_alphabet(options: Any) -> tuple[int, ...]:
    """The control alphabet, written either as action labels or as target indices.

    ``control_alphabet: [Q, M]`` is the readable spelling and the one the
    extension document uses; ``control_targets: [0, 1]`` is the index spelling
    the game has always taken. They mean the same thing, so declaring both is
    refused rather than silently resolved in favour of one.
    """

    raw_alphabet = options.get("control_alphabet")
    if raw_alphabet is None:
        return tuple(int(item) for item in options.get("control_targets", (0, 1)))
    if "control_targets" in options:
        raise ValueError(
            "game.options declares both control_alphabet and control_targets; they are "
            "the same alphabet in two spellings, so declare exactly one"
        )
    if isinstance(raw_alphabet, (str, bytes)):
        raise ValueError("game.options.control_alphabet must be a list of action labels")
    labels = tuple(str(item) for item in options.get("actions", ("Q", "M")))
    resolved: list[int] = []
    for item in raw_alphabet:
        if item is None or str(item).lower() == "none":
            resolved.append(NO_PUSH)
        elif str(item) in labels:
            resolved.append(labels.index(str(item)))
        else:
            raise ValueError(
                f"game.options.control_alphabet entry {item!r} is not one of the game's "
                f"actions {list(labels)} nor a do-nothing entry (null)"
            )
    return tuple(resolved)


def control_kernels(rules: MarkovSpec, control: ControlSpec) -> np.ndarray:
    """``T_u`` for every value in the control alphabet, stacked.

    The family the empowerment closed forms are taken over. Each kernel is a
    product over agents, but their policy-average is not - a shared control
    value correlates the agents it pushes - which is why callers must mix these
    explicitly rather than averaging the per-agent push probability.
    """

    return np.stack(
        [
            transition_matrix(
                rules,
                replace(control, value=index).chain_control(rules.population_size),
            )
            for index in range(len(control.targets))
        ]
    )


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

    def empowerment_spec(self, config: GameConfig) -> EmpowermentSpec:
        return EmpowermentSpec.from_config(config.options, self.rules(config).rounds)

    def control_policy(self, config: GameConfig) -> np.ndarray:
        """``pi(u)`` over the control alphabet - uniform, and state-independent.

        State-independence is what makes ``U_t`` exactly independent of ``S_t``:
        no mediator, no backdoor, and the per-state empowerment is pure control
        authority rather than a mixture of authority and selection.
        """

        size = len(self.control(config).targets)
        return np.full(size, 1.0 / size)

    def chain_control(self, config: GameConfig) -> tuple | None:
        if self.empowerment_spec(config).control_mode == PER_ROUND:
            # No single control value is in force, so there is no per-agent push
            # probability to report. `exact_transition` supplies the mixture.
            return None
        return self.control(config).chain_control(config.population_size)

    def exact_transition(self, config: GameConfig) -> np.ndarray | None:
        if self.empowerment_spec(config).control_mode != PER_ROUND:
            return None
        return exact.policy_averaged_kernel(
            control_kernels(self.rules(config), self.control(config)),
            self.control_policy(config),
        )

    def _control_tape(self, seed: int, rules: MarkovSpec) -> np.ndarray:
        return episode_generator(seed, CONTROL_STREAM).random(
            (rules.rounds, rules.population_size)
        )

    def _control_values(
        self, config: GameConfig, seed: int, rules: MarkovSpec, control: ControlSpec
    ) -> np.ndarray | None:
        """Which control value is in force at each round, or None if none ever is.

        The episode-fixed branch returns a constant tape and never touches the
        draw stream, so an episode that ran before ``control_mode`` existed
        still replays cell for cell.
        """

        if self.empowerment_spec(config).control_mode != PER_ROUND:
            return (
                np.full(rules.rounds, control.value, dtype=int)
                if control.is_active
                else None
            )
        if not control.can_push:
            return None
        return episode_generator(seed, CONTROL_VALUE_STREAM).integers(
            0, len(control.targets), rules.rounds
        )

    def round_control(self, config: GameConfig, state: GameState) -> tuple | None:
        rules = self.rules(config)
        control = self.control(config)
        seed = int(state.data["seed"])
        values = self._control_values(config, seed, rules, control)
        if values is None:
            return None
        target = control.targets[int(values[state.turn])]
        if target == NO_PUSH:
            # The controller exists but drew its do-nothing value this round.
            # Reported as a row of empty pushes rather than as no controller, so
            # the prompt keeps one shape across the episode.
            return tuple(None for _ in range(rules.population_size))
        tape = self._control_tape(seed, rules)
        controlled = set(control.agents)
        label = rules.actions[target]
        return tuple(
            label
            if index in controlled and tape[state.turn, index] < control.strength
            else None
            for index in range(rules.population_size)
        )

    def simulation_control(self, config: GameConfig, seeds: Sequence[int]) -> Any:
        """Which (episode, round, agent) cells the controller pushes, and toward what.

        Drawn from its own named streams, so introducing a controller leaves
        every uncontrolled episode's draws bit-identical - a Game 3 config with
        zero strength replays exactly as the Game 2 config it came from.
        """

        rules = self.rules(config)
        control = self.control(config)
        seed_list = tuple(int(seed) for seed in seeds)
        values = [
            self._control_values(config, seed, rules, control) for seed in seed_list
        ]
        if not seed_list or values[0] is None:
            return None
        alphabet = np.asarray(control.targets, dtype=np.int8)
        targets = alphabet[np.stack(values)]
        tape = np.stack([self._control_tape(seed, rules) for seed in seed_list])
        mask = np.zeros(rules.population_size, dtype=bool)
        mask[list(control.agents)] = True
        applies = (
            (tape < control.strength)
            & mask[None, None, :]
            & (targets != NO_PUSH)[:, :, None]
        )
        # NO_PUSH rounds are already masked out of `applies`, so the target they
        # carry is never read; clamping keeps it a valid action index anyway.
        return applies, np.maximum(targets, 0)

    def apply_control(self, profile: np.ndarray, control: Any, round_index: int) -> np.ndarray:
        applies, targets = control
        return np.where(
            applies[:, round_index, :], targets[:, round_index, None], profile
        ).astype(np.int8)

    def effective_empowerment(
        self, config: GameConfig
    ) -> tuple[EmpowermentReport, EmpowermentSpec]:
        """The paper's ``E(pi)`` and the episode-fixed curve, on the same chain."""

        rules = self.rules(config)
        spec = self.empowerment_spec(config)
        return (
            analyse(
                control_kernels(rules, self.control(config)),
                initial_distribution(rules),
                exact.bit_table(rules.population_size, rules.n_states),
                spec,
                policy=self.control_policy(config),
            ),
            spec,
        )

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

        # Effective empowerment as the paper defines it, plus the episode-fixed
        # curve it is being contrasted against. Computed on every Game 3 config
        # rather than behind a flag: the two variants disagreeing is the finding,
        # and a finding that only appears when asked for is one nobody sees.
        report, empowerment = self.effective_empowerment(config)
        quantities.extend(empowerment_quantities(report, empowerment))
        return GroundTruth(
            game_type=self.spec.game_type,
            parameters={
                **truth.parameters,
                **control.to_dict(),
                **empowerment_parameters(report, empowerment),
            },
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
    kernels = control_kernels(rules, control)
    micro = np.stack(
        [exact.propagate(initial_distribution(rules), kernel, rounds) for kernel in kernels]
    )
    macro = np.stack([exact.macrostate_law(law, bits) for law in micro])
    return micro, macro
