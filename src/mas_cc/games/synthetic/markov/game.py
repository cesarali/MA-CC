"""Game 2 - Markov. Real dynamics, real coupling, still closed-form.

Binary alphabet {Q, M}, matching the real naming game. Agent *i* observes one
partner's previous action through a coupling graph, pushes it through a fixed
2x2 stochastic kernel, and then flips with private probability ``eps_i``:

    observed  = A_{source(i), t}
    target    ~ kernel[observed]
    A_{i,t+1} = target XOR Bern(eps_i)

The microstate is the full action profile, so with ``N <= 10`` the chain has at
most ``2^10 = 1024`` states and every quantity below is exact linear algebra
with no sampling error to argue about.

**The coupling graph is more diagnostic than the noise level**, because it lets
us design *structural zeros*. A value that is 5% off is hard to call a bug; a
conditional mutual information that must be exactly zero and isn't is not
ambiguous. Two configurations carry that weight:

- **Chain 1 -> 2 -> 3.** Agent 3's next action depends on the profile only
  through agent 2's current one, so ``I(A_3^{t+1}; A_1^t | A_2^t) = 0`` exactly,
  while ``0 < I(A_1;A_3) < I(A_1;A_2)``. Tests conditional MI and the
  data-processing inequality together.
- **Asymmetric lag.** Agent 2 follows agent 1 and nothing follows agent 2, so
  ``TE(1->2) > 0`` and ``TE(2->1) = 0`` exactly. This is the one that catches
  direction errors and off-by-one round alignment - the classic silent failure.

Ground truth is reported as the **episode-pooled** value, not the stationary
one, because that is what the estimator actually targets: it pools counts over
every round and computes one table. Pooling counts is averaging the joint
distributions and *then* taking mutual information, which is not the average of
the per-round values. The stationary value is reported alongside as a separate
quantity, since the two coincide only after the chain has mixed.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

from mas_cc.config import GameConfig
from mas_cc.core import AgentId, InteractionId
from mas_cc.games.protocols import (
    Action,
    AgentState,
    DecisionRequest,
    GameSpec,
    GameState,
    Observation,
)
from mas_cc.llm_runtime.validation import ValidationIssue, ValidationResult
from mas_cc.planning import DecisionStagePlan, GameCallPlan, InteractionCount, PromptScenario

from .. import exact
from ..noise import bernoulli_draws, episode_generator
from ..prompts import bind_markov_prompt
from ..protocols import (
    ExactDynamics,
    GroundTruth,
    GroundTruthQuantity,
    SimulatedEpisodes,
    SyntheticGame,
    SyntheticTransition,
)

POLICY = "markov_kernel_v1"

INITIAL_STREAM = "markov.initial_profile"
KERNEL_STREAM = "markov.kernel_draw"
FLIP_STREAM = "markov.private_noise"

COUPLING_SHORTHANDS = ("self", "chain", "ring")
"""Named coupling graphs.

``self``   every agent observes only itself - no coupling, so every transfer
           entropy between distinct agents is a structural zero.
``chain``  agent i observes agent i-1, and agent 0 observes itself. The
           data-processing configuration.
``ring``   agent i observes agent i-1 cyclically, so influence circulates and
           the population can reach consensus.
"""


def resolve_coupling(value: Any, population_size: int) -> tuple[int, ...]:
    """Who each agent observes, as one source index per agent.

    Deliberately single-source. Both structural-zero configurations the game
    exists for are single-source, and a multi-parent rule would need a
    combination convention (majority? first?) that changes what the closed forms
    mean without making either test sharper.
    """

    if value is None or value == "self":
        return tuple(range(population_size))
    if value == "chain":
        return (0, *range(population_size - 1))
    if value == "ring":
        return tuple((index - 1) % population_size for index in range(population_size))
    if isinstance(value, (str, bytes)):
        raise ValueError(
            f"game.options.coupling must be one of {COUPLING_SHORTHANDS} or a list of "
            f"source indices, got {value!r}"
        )
    sources = tuple(int(item) for item in value)
    if len(sources) != population_size:
        raise ValueError("game.options.coupling must have one source per agent")
    if any(not 0 <= source < population_size for source in sources):
        raise ValueError("game.options.coupling entries must be valid agent indices")
    return sources


@dataclass(frozen=True, slots=True)
class MarkovSpec:
    """Resolved Game 2 parameters, validated once."""

    population_size: int
    rounds: int
    actions: tuple[str, ...]
    epsilons: tuple[float, ...]
    coupling: tuple[int, ...]
    kernel: tuple[tuple[float, float], tuple[float, float]]
    initial_bias: float

    @classmethod
    def from_config(cls, config: GameConfig) -> "MarkovSpec":
        options = config.options
        raw_actions = options.get("actions", ("Q", "M"))
        if isinstance(raw_actions, (str, bytes)):
            raise ValueError("game.options.actions must be a list of two labels")
        spec = cls(
            population_size=config.population_size,
            rounds=config.horizon,
            actions=tuple(str(item) for item in raw_actions),
            epsilons=_resolve_epsilons(options, config.population_size),
            coupling=resolve_coupling(options.get("coupling", "ring"), config.population_size),
            kernel=_resolve_kernel(options.get("kernel")),
            initial_bias=float(options.get("initial_bias", 0.5)),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("synthetic markov games require at least two agents")
        if self.population_size > 10:
            raise ValueError(
                "synthetic markov games keep N <= 10 so the microstate space stays "
                f"at most 2^10 = 1024 exactly-computable states; got {self.population_size}"
            )
        if self.rounds < 1:
            raise ValueError("a positive horizon is required")
        if len(self.actions) != 2 or len(set(self.actions)) != 2:
            raise ValueError("synthetic markov games use exactly two distinct action labels")
        if len(self.epsilons) != self.population_size:
            raise ValueError("game.options.epsilons must have one entry per agent")
        if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in self.epsilons):
            raise ValueError("every epsilon must lie in [0, 1]")
        if not math.isfinite(self.initial_bias) or not 0.0 <= self.initial_bias <= 1.0:
            raise ValueError("game.options.initial_bias must lie in [0, 1]")

    @property
    def n_states(self) -> int:
        return 1 << self.population_size

    def agent_id(self, index: int) -> AgentId:
        return AgentId(f"agent-{index:03d}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "population_size": self.population_size,
            "rounds": self.rounds,
            "actions": list(self.actions),
            "epsilons": list(self.epsilons),
            "coupling": list(self.coupling),
            "kernel": [list(row) for row in self.kernel],
            "initial_bias": self.initial_bias,
        }

    @property
    def key(self) -> tuple:
        """A hashable identity, so the exact chain can be cached per spec."""

        return (
            self.population_size, self.rounds, self.actions, self.epsilons,
            self.coupling, self.kernel, self.initial_bias,
        )


def _resolve_epsilons(options: Mapping[str, Any], population_size: int) -> tuple[float, ...]:
    per_agent = options.get("epsilons")
    if per_agent is not None:
        if isinstance(per_agent, (str, bytes)) or not isinstance(per_agent, Sequence):
            raise ValueError("game.options.epsilons must be a list of numbers")
        return tuple(float(value) for value in per_agent)
    scalar = options.get("epsilon", 0.1)
    if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
        raise ValueError("game.options.epsilon must be a number")
    return (float(scalar),) * population_size


def _resolve_kernel(value: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    """The 2x2 stochastic kernel, defaulting to the identity (pure copying).

    Copying is the default because it makes the coupling graph the only thing
    shaping the dynamics, which is what the structural-zero configurations are
    testing. A non-identity kernel is a second, independent knob.
    """

    if value is None:
        return ((1.0, 0.0), (0.0, 1.0))
    rows = tuple(tuple(float(cell) for cell in row) for row in value)
    if len(rows) != 2 or any(len(row) != 2 for row in rows):
        raise ValueError("game.options.kernel must be a 2x2 matrix")
    for row in rows:
        if any(cell < 0 for cell in row) or abs(sum(row) - 1.0) > 1e-9:
            raise ValueError("each game.options.kernel row must be a probability distribution")
    return rows  # type: ignore[return-value]


# -- the exact chain ------------------------------------------------------


def next_action_probabilities(
    spec: MarkovSpec, forced: tuple[float | None, ...] | None = None
) -> np.ndarray:
    """``p[x, i]`` = P(agent i plays the second action next | microstate x).

    ``forced[i]``, when set, is the probability that a control overrides agent
    i onto the second action regardless of what it observed; Game 3 supplies it
    and Game 2 leaves it None. Keeping the seam here rather than in a subclass
    means both games build their transition matrix through exactly one code
    path, so a control of zero strength is provably the uncontrolled game.
    """

    bits = exact.bit_table(spec.population_size, spec.n_states)
    kernel = np.asarray(spec.kernel, dtype=float)
    probabilities = np.zeros((spec.n_states, spec.population_size), dtype=float)
    for index in range(spec.population_size):
        observed = bits[:, spec.coupling[index]]
        target_high = kernel[observed, 1]
        epsilon = spec.epsilons[index]
        natural = target_high * (1.0 - epsilon) + (1.0 - target_high) * epsilon
        if forced is not None and forced[index] is not None:
            strength, value = forced[index]  # type: ignore[misc]
            natural = strength * value + (1.0 - strength) * natural
        probabilities[:, index] = natural
    return probabilities


def transition_matrix(
    spec: MarkovSpec, forced: tuple[Any, ...] | None = None
) -> np.ndarray:
    """The exact microstate transition matrix.

    Agents decide independently given the current profile, so the joint is a
    product over agents - built vectorized rather than cell by cell, which is
    what keeps a 1024x1024 matrix a few milliseconds.
    """

    probabilities = next_action_probabilities(spec, forced)
    bits = exact.bit_table(spec.population_size, spec.n_states).astype(bool)
    matrix = np.where(
        bits[None, :, :], probabilities[:, None, :], 1.0 - probabilities[:, None, :]
    ).prod(axis=2)
    return matrix / matrix.sum(axis=1, keepdims=True)


def initial_distribution(spec: MarkovSpec) -> np.ndarray:
    """Product-Bernoulli initial profile law, matching what `initialize` draws."""

    bits = exact.bit_table(spec.population_size, spec.n_states)
    bias = spec.initial_bias
    return np.prod(np.where(bits == 1, bias, 1.0 - bias), axis=1)


@dataclass(frozen=True, slots=True)
class PooledLaws:
    """What a pooled estimator actually sees over one episode.

    ``single`` averages the per-round microstate law over the recorded rounds
    and ``pair`` averages the consecutive-round joint. Both are averages of
    *distributions*, taken before any mutual information - because pooling
    counts and then estimating is exactly that, and is not the same as
    averaging per-round mutual informations.
    """

    single: np.ndarray
    pair: np.ndarray
    stationary: np.ndarray
    transition: np.ndarray


def _pooled_from_matrix(spec: MarkovSpec, matrix: np.ndarray) -> PooledLaws:
    law = initial_distribution(spec)
    single = np.zeros_like(law)
    pair = np.zeros((spec.n_states, spec.n_states), dtype=float)
    # Recorded rounds are 1..T, so the first recorded profile is p_0 @ T.
    for round_index in range(spec.rounds):
        law = law @ matrix
        single += law
        if round_index < spec.rounds - 1:
            pair += exact.transition_joint(law, matrix)
    single /= single.sum()
    if pair.sum() > 0:
        pair /= pair.sum()
    return PooledLaws(single, pair, exact.stationary_distribution(matrix), matrix)


@lru_cache(maxsize=8)
def _pooled(key: tuple, forced: tuple | None = None) -> PooledLaws:
    spec = MarkovSpec(*key)
    return _pooled_from_matrix(spec, transition_matrix(spec, forced))


def pooled_laws(
    spec: MarkovSpec, forced: tuple | None = None, matrix: np.ndarray | None = None
) -> PooledLaws:
    """The pooled laws for this spec, optionally under an explicit kernel.

    ``matrix`` exists because not every controller's kernel is expressible as a
    per-agent ``forced`` probability. A control resampled each round pushes every
    controlled agent toward the *same* target, so averaging over the control
    correlates them: the policy-averaged kernel is a mixture of product kernels
    and not itself a product kernel. Passing the mixture in is the only way to
    keep the derived quantities honest for that mode.
    """

    if matrix is None:
        return _pooled(spec.key, forced)
    return _pooled_from_matrix(spec, np.asarray(matrix, dtype=float))


class MarkovDynamics(ExactDynamics):
    """Exact macrostate laws for one resolved Markov condition."""

    def __init__(
        self,
        spec: MarkovSpec,
        forced: tuple | None = None,
        transition: np.ndarray | None = None,
    ) -> None:
        self._spec = spec
        self._forced = forced
        self._laws = pooled_laws(spec, forced, transition)
        self._bits = exact.bit_table(spec.population_size, spec.n_states)

    @property
    def population_size(self) -> int:
        return self._spec.population_size

    @property
    def action_labels(self) -> tuple[str, ...]:
        return self._spec.actions

    @property
    def rounds(self) -> int:
        return self._spec.rounds

    @property
    def transition(self) -> np.ndarray:
        return self._laws.transition

    def microstate_law(self, round_index: int) -> np.ndarray:
        return exact.propagate(initial_distribution(self._spec), self._laws.transition, round_index)

    def macrostate_law(self, round_index: int) -> np.ndarray:
        return exact.macrostate_law(self.microstate_law(round_index), self._bits)

    def macrostate_pair_law(self, round_index: int, horizon: int) -> np.ndarray:
        # Propagate the microstate forward `horizon` steps and lump only at the
        # end; lumping first would be wrong wherever the chain is not lumpable.
        current = self.microstate_law(round_index)
        lagged = np.linalg.matrix_power(self._laws.transition, horizon)
        return exact.macrostate_pair_law(exact.transition_joint(current, lagged), self._bits)

    def pooled_macrostate_pair_law(
        self, horizon: int, window: tuple[int, int]
    ) -> np.ndarray:
        """Advance the microstate law once through the window, lumping at each step.

        The inherited default re-propagates from round zero for every round in
        the window, which is quadratic in episode length. This walks the chain
        forward once and reuses the lagged matrix, and - importantly - still
        lumps only after each microstate joint is formed.
        """

        first, last = window
        lagged = np.linalg.matrix_power(self._laws.transition, horizon)
        law = self.microstate_law(first)
        size = self._spec.population_size + 1
        total = np.zeros((size, size), dtype=float)
        for round_index in range(first, last + 1):
            total += exact.macrostate_pair_law(
                exact.transition_joint(law, lagged), self._bits
            )
            if round_index < last:
                law = law @ self._laws.transition
        return total / max(last - first + 1, 1)

    def pooled_macrostate_law(self, window: tuple[int, int]) -> np.ndarray:
        first, last = window
        law = self.microstate_law(first)
        total = np.zeros(self._spec.population_size + 1, dtype=float)
        for round_index in range(first, last + 1):
            total += exact.macrostate_law(law, self._bits)
            if round_index < last:
                law = law @ self._laws.transition
        return total / max(last - first + 1, 1)

    @property
    def is_lumpable(self) -> bool:
        return exact.lumping_gap(self._laws.transition, self._bits) < 1e-9


def _resolve_epsilon_key(spec: MarkovSpec) -> tuple:
    return spec.key


class SyntheticMarkovGame(SyntheticGame):
    """Coupled binary agents with a fixed kernel and private noise."""

    spec = GameSpec(
        game_type="synthetic_markov",
        version=1,
        description=(
            "Synthetic coupled-Markov game with an exactly computable transition chain."
        ),
        game_family="choice",
        minimum_population=2,
        supported_topologies=("complete",),
    )

    policy = POLICY

    def rules(self, config: GameConfig) -> MarkovSpec:
        if config.type != self.spec.game_type:
            raise ValueError(f"expected game type {self.spec.game_type!r}")
        return MarkovSpec.from_config(config)

    def round_control(self, config: GameConfig, state: GameState) -> tuple | None:
        """This round's forced action label per agent, or None where free.

        Game 2 has no controller and returns None outright, which is what keeps
        the ``forced`` field out of its prompts entirely rather than rendering a
        row of nulls. Game 3 overrides it.
        """

        return None

    # -- Game contract ----------------------------------------------------

    def initialize(self, config: GameConfig, seed: int) -> GameState:
        rules = self.rules(config)
        start = bernoulli_draws(
            seed, INITIAL_STREAM, (rules.population_size,), rules.initial_bias
        )
        agents = tuple(
            AgentState(
                rules.agent_id(index),
                attributes={
                    "available_actions": list(rules.actions),
                    "committed_action": rules.actions[int(start[index])],
                    "epsilon": rules.epsilons[index],
                    "observes": int(rules.coupling[index]),
                },
            )
            for index in range(rules.population_size)
        )
        return GameState(
            game_type=self.spec.game_type,
            turn=0,
            agents=agents,
            data={
                "seed": seed,
                "action_pool": list(rules.actions),
                "topology": config.topology,
                "coupling": list(rules.coupling),
                "round_history": [],
            },
        )

    def select_participants(
        self, state: GameState, config: GameConfig, rng: random.Random
    ) -> tuple[AgentId, ...]:
        if state.terminated:
            raise ValueError("cannot select participants after termination")
        return tuple(agent.agent_id for agent in state.agents)

    def _tapes(self, seed: int, rules: MarkovSpec) -> tuple[np.ndarray, np.ndarray]:
        kernel_draw = episode_generator(seed, KERNEL_STREAM).random(
            (rules.rounds, rules.population_size)
        )
        flip = bernoulli_draws(
            seed, FLIP_STREAM, (rules.rounds, rules.population_size), np.asarray(rules.epsilons)
        )
        return kernel_draw, flip

    def construct_observations(
        self, state: GameState, participants: tuple[AgentId, ...], config: GameConfig
    ) -> tuple[Observation, ...]:
        rules = self.rules(config)
        kernel_draw, flip = self._tapes(int(state.data["seed"]), rules)
        round_index = state.turn + 1
        interaction_id = InteractionId(f"interaction-{round_index:04d}")
        current = [str(agent.attributes["committed_action"]) for agent in state.agents]
        forced = self.round_control(config, state)
        observations: list[Observation] = []
        for index, agent_id in enumerate(participants):
            observed = current[rules.coupling[index]]
            # The realized kernel row for this round: one uniform draw resolves
            # the map for every possible observation, which is the standard
            # common-random-numbers coupling and keeps P(target | observed)
            # exactly equal to the kernel row that actually applies.
            draw = float(kernel_draw[state.turn, index])
            rule = {
                label: rules.actions[1] if draw < rules.kernel[value][1] else rules.actions[0]
                for value, label in enumerate(rules.actions)
            }
            visible: dict[str, Any] = {
                "policy": self.policy,
                "round": round_index,
                "actions": list(rules.actions),
                "observes": str(rules.agent_id(rules.coupling[index])),
                "observed": observed,
                "rule": rule,
                "flip": bool(flip[state.turn, index]),
            }
            if forced is not None:
                visible["forced"] = forced[index]
            observations.append(
                Observation(
                    agent_id=agent_id,
                    interaction_id=interaction_id,
                    participants=participants,
                    visible_state=visible,
                )
            )
        return tuple(observations)

    def build_decision_requests(
        self, state: GameState, observations: tuple[Observation, ...], config: GameConfig
    ) -> tuple[DecisionRequest, ...]:
        rules = self.rules(config)
        return tuple(
            DecisionRequest(
                agent_id=observation.agent_id,
                interaction_id=observation.interaction_id,
                stage="synchronous_report",
                observation=observation,
                prompt=bind_markov_prompt(
                    actions=rules.actions, observation=observation.visible_state
                ),
                provider_required=True,
                retry_bound=0,
            )
            for observation in observations
        )

    def parse_action(self, request: DecisionRequest, response: str) -> Action:
        return Action(
            agent_id=request.agent_id,
            value=response.strip(),
            stage=request.stage,
            metadata={
                "interaction_id": str(request.interaction_id),
                "parser_mode": "exact",
                "observed": request.observation.visible_state["observed"],
            },
        )

    def validate_action(
        self, state: GameState, request: DecisionRequest, action: Action, config: GameConfig
    ) -> ValidationResult:
        rules = self.rules(config)
        issues: list[ValidationIssue] = []
        if action.agent_id != request.agent_id:
            issues.append(ValidationIssue("action.agent_id", "must match the decision agent"))
        if action.value not in rules.actions:
            issues.append(
                ValidationIssue(
                    "action.value", f"must be one of {', '.join(rules.actions)}", action.value
                )
            )
        return ValidationResult(tuple(issues))

    def apply_transition(
        self,
        state: GameState,
        participants: tuple[AgentId, ...],
        actions: tuple[Action, ...],
        config: GameConfig,
    ) -> SyntheticTransition:
        rules = self.rules(config)
        if tuple(action.agent_id for action in actions) != participants:
            raise ValueError("actions must follow participant order")
        if len(actions) != rules.population_size:
            raise ValueError("every agent reports every round")
        by_agent = {action.agent_id: action.value for action in actions}
        round_index = state.turn + 1
        unanimous = len({action.value for action in actions}) == 1
        updated = tuple(
            replace(
                agent,
                attributes={
                    **dict(agent.attributes),
                    "committed_action": by_agent[agent.agent_id],
                },
            )
            for agent in state.agents
        )
        entry = {
            "interaction_index": round_index,
            "selected_agents": [str(agent_id) for agent_id in participants],
            "actions": [action.value for action in actions],
            "committed": [False] * len(actions),
            "success": unanimous,
            "payoff": 0.0,
        }
        terminated = round_index >= rules.rounds
        next_state = GameState(
            game_type=state.game_type,
            turn=round_index,
            agents=updated,
            terminated=terminated,
            data={
                **dict(state.data),
                "round_history": (*state.data.get("round_history", ()), entry),
            },
        )
        return SyntheticTransition(
            interaction_id=InteractionId(f"interaction-{round_index:04d}"),
            actions=actions,
            payoffs={str(agent_id): 0.0 for agent_id in participants},
            next_state=next_state,
            matched=unanimous,
            termination_reason="fixed_horizon_reached" if terminated else None,
        )

    def detect_termination(self, state: GameState, config: GameConfig) -> str | None:
        rules = self.rules(config)
        if state.terminated or state.turn >= rules.rounds:
            return "fixed_horizon_reached"
        return None

    def call_plan(self, config: GameConfig) -> GameCallPlan:
        rules = self.rules(config)
        scenario = PromptScenario(
            "synthetic_round",
            bind_markov_prompt(
                actions=rules.actions,
                observation={
                    "policy": self.policy, "round": 1, "actions": list(rules.actions),
                    "observes": "agent-000", "observed": rules.actions[0],
                    "rule": {rules.actions[0]: rules.actions[0], rules.actions[1]: rules.actions[1]},
                    "flip": False,
                },
            ),
            ("Every round's prompt has the same shape; only the payload line changes.",),
        )
        return GameCallPlan(
            game_type=self.spec.game_type,
            game_version=self.spec.version,
            interactions=InteractionCount(
                fixed=rules.rounds, lower=rules.rounds,
                expected=rules.rounds, maximum=rules.rounds,
            ),
            decision_stages=(
                DecisionStagePlan(
                    name="synchronous_report",
                    requests_per_interaction=rules.population_size,
                    retry_bound=0,
                    lower_prompt=scenario,
                    representative_prompt=scenario,
                    maximum_prompt=scenario,
                    prompt_scenarios=(scenario,),
                    assumptions=("Every agent reports once every round.",),
                ),
            ),
            stopping_condition_assumptions=(
                f"The run stops after exactly {rules.rounds} rounds.",
            ),
            metadata={
                "population_size": rules.population_size,
                "coupling": list(rules.coupling),
                "provider_prices_included": False,
            },
        )

    # -- SyntheticGame contract -------------------------------------------

    def chain_control(self, config: GameConfig) -> tuple | None:
        """Control as the exact chain sees it: ``(strength, target_bit)`` per agent.

        A different shape from `round_control` on purpose. That one is a
        realized draw for one round; this one is the *probability* a control
        applies, which is what enters the transition matrix. Conflating them
        would put a sampled value where an expectation belongs.
        """

        return None

    def exact_transition(self, config: GameConfig) -> np.ndarray | None:
        """An explicit microstate kernel, when the product form cannot express one.

        Game 2 has none: its agents decide independently given the profile, so
        `transition_matrix` builds the kernel as a product over agents. Game 3
        with a control resampled every round does need one, because the
        policy-averaged kernel is a *mixture* of product kernels - a shared
        control value correlates the agents it pushes - and a mixture of
        products is not a product.
        """

        return None

    def ground_truth(self, config: GameConfig) -> GroundTruth:
        rules = self.rules(config)
        forced = self.chain_control(config)
        laws = pooled_laws(rules, forced, self.exact_transition(config))
        bits = exact.bit_table(rules.population_size, rules.n_states)
        size = rules.population_size
        quantities: list[GroundTruthQuantity] = []

        for index in range(size):
            table = np.zeros(2, dtype=float)
            np.add.at(table, bits[:, index], laws.single)
            quantities.append(
                GroundTruthQuantity(
                    name="marginal_entropy",
                    value=exact.entropy(table),
                    subject=(str(rules.agent_id(index)),),
                    definition="H(A_i) under the episode-pooled law.",
                )
            )

        pairwise: list[float] = []
        for left in range(size):
            for right in range(left + 1, size):
                value = exact.cross_agent_mutual_information(laws.single, bits, left, right)
                pairwise.append(value)
                quantities.append(
                    GroundTruthQuantity(
                        name="mutual_information",
                        value=value,
                        subject=(str(rules.agent_id(left)), str(rules.agent_id(right))),
                        definition=(
                            "I(A_i;A_j) under the episode-pooled microstate law - what a "
                            "pooled-count estimator over the whole episode targets."
                        ),
                    )
                )
                quantities.append(
                    GroundTruthQuantity(
                        name="stationary_mutual_information",
                        value=exact.cross_agent_mutual_information(
                            laws.stationary, bits, left, right
                        ),
                        subject=(str(rules.agent_id(left)), str(rules.agent_id(right))),
                        definition=(
                            "I(A_i;A_j) at stationarity; equals the pooled value only "
                            "after the chain has mixed."
                        ),
                    )
                )
        quantities.append(
            GroundTruthQuantity(
                name="mean_pairwise_mutual_information",
                value=sum(pairwise) / len(pairwise),
                definition="Unweighted mean of I(A_i;A_j) over all unordered pairs.",
            )
        )
        quantities.append(
            GroundTruthQuantity(
                name="entropy_rate",
                value=exact.entropy_rate(laws.transition, laws.stationary),
                definition="H(X_{t+1} | X_t) over the full microstate at stationarity.",
            )
        )

        # Transfer entropy for every ordered pair. The zeros here are the
        # diagnostic: a direction with no edge into it must be exactly zero, and
        # an off-by-one in round alignment turns that zero positive.
        for source in range(size):
            for target in range(size):
                if source == target:
                    continue
                quantities.append(
                    GroundTruthQuantity(
                        name="transfer_entropy",
                        value=exact.directed_conditional_mutual_information(
                            laws.pair, bits, target=target, source=source, condition=target
                        ),
                        subject=(str(rules.agent_id(source)), str(rules.agent_id(target))),
                        definition="TE(i->j) = I(A_j^{t+1}; A_i^t | A_j^t), episode-pooled.",
                    )
                )

        # The data-processing zeros, only for populations small enough that the
        # triple set stays readable rather than combinatorial noise.
        if size <= 5:
            for target in range(size):
                for source in range(size):
                    for condition in range(size):
                        if len({target, source, condition}) < 3:
                            continue
                        quantities.append(
                            GroundTruthQuantity(
                                name="conditional_transfer_entropy",
                                value=exact.directed_conditional_mutual_information(
                                    laws.pair, bits,
                                    target=target, source=source, condition=condition,
                                ),
                                subject=(
                                    str(rules.agent_id(source)),
                                    str(rules.agent_id(target)),
                                    str(rules.agent_id(condition)),
                                ),
                                definition=(
                                    "I(A_target^{t+1}; A_source^t | A_condition^t); exactly "
                                    "zero when the condition screens the source off."
                                ),
                            )
                        )

        dynamics = self.exact_dynamics(config)
        outcome = dynamics.terminal_outcome_law()
        # The ordinary choice metrics are time-averages over the episode, so
        # their closed forms come from the episode-pooled macrostate law rather
        # than the final round's - comparing a time-average against a
        # single-instant value would be a category error that happens to look
        # close whenever the chain mixes fast.
        macro = exact.macrostate_law(laws.single, bits)
        share_high = float(sum(macro[k] * k / size for k in range(size + 1)))
        for index, label in enumerate(rules.actions):
            quantities.append(
                GroundTruthQuantity(
                    name="expected_action_share",
                    value=share_high if index == 1 else 1.0 - share_high,
                    subject=(label,),
                    units="probability",
                    definition="E[population_action_share_per_option] under the pooled law.",
                )
            )
        quantities.append(
            GroundTruthQuantity(
                name="unanimity_probability",
                value=float(macro[0] + macro[size]),
                units="probability",
                definition=(
                    "P(all agents report the same action) under the pooled law; the "
                    "expected value of rolling_coordination_rate."
                ),
            )
        )
        quantities.append(
            GroundTruthQuantity(
                name="expected_consensus_by_success_rate",
                # A coupled population *can* sustain unanimity, unlike the
                # memoryless game - but only when nothing can break it, which
                # means no noise anywhere.
                value=1.0 if all(value == 0.0 for value in rules.epsilons) else 0.0,
                units="indicator",
                definition=(
                    "1 when every agent is noiseless and unanimity, once reached, is "
                    "absorbing; 0 when noise makes a sustained window impossible."
                ),
            )
        )
        quantities.append(
            GroundTruthQuantity(
                name="expected_first_chance_unanimity_round",
                value=(
                    math.inf
                    if macro[0] + macro[size] <= 0.0
                    else 1.0 / float(macro[0] + macro[size])
                ),
                units="rounds",
                definition=(
                    "1 / P(unanimity): roughly when a standing-share criterion with no "
                    "persistence requirement fires. High variance; one episode is one draw."
                ),
            )
        )
        for index, label in enumerate(rules.actions):
            quantities.append(
                GroundTruthQuantity(
                    name="terminal_outcome_probability",
                    value=float(outcome[index]),
                    subject=(label,),
                    units="probability",
                    definition=(
                        "P(dominant action in the final round = this label), with ties "
                        "resolved to the first declared action exactly as reader.py does."
                    ),
                )
            )
        quantities.append(
            GroundTruthQuantity(
                name="expected_dominant_action_share",
                value=float(
                    sum(macro[k] * max(k, size - k) / size for k in range(size + 1))
                ),
                units="probability",
                definition="E[max(k, N-k)/N] under the episode-pooled macrostate law.",
            )
        )
        quantities.append(
            GroundTruthQuantity(
                name="macrostate_is_lumpable",
                value=1.0 if dynamics.is_lumpable else 0.0,
                units="indicator",
                definition=(
                    "1 when the macrostate chain is itself Markov. When 0, computing on "
                    "the lumped chain gives a different and wrong answer, and only "
                    "microstate propagation is correct."
                ),
            )
        )
        return GroundTruth(
            game_type=self.spec.game_type,
            parameters=rules.to_dict(),
            quantities=tuple(quantities),
        )

    def exact_dynamics(self, config: GameConfig) -> ExactDynamics:
        return MarkovDynamics(
            self.rules(config), self.chain_control(config), self.exact_transition(config)
        )

    def simulate(self, config: GameConfig, seeds: Sequence[int]) -> SimulatedEpisodes:
        """Speed mode: the same tapes, the same rule, vectorized across seeds.

        The time loop cannot be vectorized - each round genuinely depends on the
        last - but every seed and agent advances together inside it, so the
        Python loop runs once per round rather than once per draw.
        """

        rules = self.rules(config)
        seed_list = tuple(int(seed) for seed in seeds)
        size, rounds = rules.population_size, rules.rounds
        kernel = np.asarray(rules.kernel, dtype=float)
        coupling = np.asarray(rules.coupling, dtype=int)

        start = np.empty((len(seed_list), size), dtype=np.int8)
        kernel_draw = np.empty((len(seed_list), rounds, size), dtype=float)
        flip = np.empty((len(seed_list), rounds, size), dtype=bool)
        for position, seed in enumerate(seed_list):
            start[position] = bernoulli_draws(
                seed, INITIAL_STREAM, (size,), rules.initial_bias
            )
            kernel_draw[position] = episode_generator(seed, KERNEL_STREAM).random(
                (rounds, size)
            )
            flip[position] = bernoulli_draws(
                seed, FLIP_STREAM, (rounds, size), np.asarray(rules.epsilons)
            )

        forced = self.simulation_control(config, seed_list)
        actions = np.empty((len(seed_list), rounds, size), dtype=np.int8)
        profile = start
        for round_index in range(rounds):
            observed = np.take_along_axis(
                profile, np.broadcast_to(coupling, profile.shape), axis=1
            )
            target = (kernel_draw[:, round_index, :] < kernel[observed, 1]).astype(np.int8)
            profile = (target ^ flip[:, round_index, :]).astype(np.int8)
            if forced is not None:
                profile = self.apply_control(profile, forced, round_index)
            actions[:, round_index, :] = profile
        return SimulatedEpisodes(
            seeds=seed_list, actions=actions, action_labels=rules.actions
        )

    def simulation_control(self, config: GameConfig, seeds: Sequence[int]) -> Any:
        """Speed mode's realized control tape, or None when there is no controller.

        Takes the seeds rather than a count so a subclass can draw its own
        per-episode tape here and return it, instead of stashing state on the
        game between calls.
        """

        return None

    def apply_control(self, profile: np.ndarray, control: Any, round_index: int) -> np.ndarray:
        """Overwrite the pushed cells of one round's profile. No-op without a controller."""

        return profile
