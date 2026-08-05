"""Game 1 - Bernoulli. Every quantity is zero or a known constant.

This is not the easy version of the Markov game; it has a different job. There
are no dynamics and no memory, so nothing here converges and nothing here is
interesting as a language game. What it gives us is a **floor** and a
**calibration curve**:

Each round nature draws a latent bit ``Z_t ~ Bern(1/2)``, and agent *i* reports
``A_i,t = Z_t XOR B_i,t`` with ``B_i,t ~ Bern(eps_i)`` private and independent.
Marginals are uniform by construction, so for any pair

    q_ij = eps_i (1 - eps_j) + eps_j (1 - eps_i)
    I(A_i ; A_j) = 1 - H(q_ij)   bits, exactly.

Two anchors carry the weight. At ``eps = 0.5`` the true mutual information is
exactly zero, so running many seeds there measures how much MI the estimator
reports when there is none - the finite-sample bias, and therefore the
magnitude below which a number from a real run means nothing. At ``eps = 0`` it
is exactly one bit. Sweeping between them turns estimator bias into a visible
offset from the diagonal instead of a single number someone has to form a
judgement about.
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

from ..noise import bernoulli_draws
from ..prompts import bind_bernoulli_prompt
from ..protocols import (
    GroundTruth,
    GroundTruthQuantity,
    SimulatedEpisodes,
    SyntheticGame,
    SyntheticTransition,
)

POLICY = "bernoulli_xor_v1"
"""The decoding rule the synthetic agent is told to apply, named in the payload."""

LATENT_STREAM = "bernoulli.latent"
PRIVATE_NOISE_STREAM = "bernoulli.private_noise"


def binary_entropy(probability: float) -> float:
    """H(p) in bits, with the H(0) = H(1) = 0 limits taken rather than nan.

    Those limits are not a convenience: ``eps = 0`` is one of the two anchor
    configs, and returning nan there would break the exact 1-bit end of the
    calibration curve.
    """

    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return float(
        -probability * math.log2(probability) - (1 - probability) * math.log2(1 - probability)
    )


@dataclass(frozen=True, slots=True)
class BernoulliSpec:
    """Resolved Game 1 parameters, validated once."""

    population_size: int
    rounds: int
    actions: tuple[str, ...]
    epsilons: tuple[float, ...]

    @classmethod
    def from_config(cls, config: GameConfig) -> "BernoulliSpec":
        options = config.options
        raw_actions = options.get("actions", ("Q", "M"))
        if isinstance(raw_actions, (str, bytes)):
            raise ValueError("game.options.actions must be a list of two labels")
        actions = tuple(str(item) for item in raw_actions)
        spec = cls(
            population_size=config.population_size,
            rounds=config.horizon,
            actions=actions,
            epsilons=_resolve_epsilons(options, config.population_size),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("synthetic_bernoulli requires at least two agents")
        if self.rounds < 1:
            raise ValueError("synthetic_bernoulli requires a positive horizon")
        if len(self.actions) != 2 or len(set(self.actions)) != 2:
            raise ValueError("synthetic_bernoulli uses exactly two distinct action labels")
        if any(not action.strip() or "\n" in action for action in self.actions):
            raise ValueError("action labels must be non-empty single-line strings")
        if len(self.epsilons) != self.population_size:
            raise ValueError("game.options.epsilons must have one entry per agent")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.epsilons):
            raise ValueError("every epsilon must lie in [0, 1]")

    def agent_id(self, index: int) -> AgentId:
        return AgentId(f"agent-{index:03d}")

    @property
    def agent_ids(self) -> tuple[AgentId, ...]:
        return tuple(self.agent_id(index) for index in range(self.population_size))

    def to_dict(self) -> dict[str, Any]:
        return {
            "population_size": self.population_size,
            "rounds": self.rounds,
            "actions": list(self.actions),
            "epsilons": list(self.epsilons),
        }


def _resolve_epsilons(options: Mapping[str, Any], population_size: int) -> tuple[float, ...]:
    """Per-agent noise, from either the scalar or the per-agent spelling.

    ``epsilon`` is the common case (one noise level for everyone, swept to
    build the calibration curve); ``epsilons`` exists for the asymmetric
    configs where agents differ, and wins when both are given.
    """

    per_agent = options.get("epsilons")
    if per_agent is not None:
        if isinstance(per_agent, (str, bytes)) or not isinstance(per_agent, Sequence):
            raise ValueError("game.options.epsilons must be a list of numbers")
        return tuple(float(value) for value in per_agent)
    scalar = options.get("epsilon", 0.5)
    if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
        raise ValueError("game.options.epsilon must be a number")
    return (float(scalar),) * population_size


@dataclass(frozen=True, slots=True)
class BernoulliTape:
    """Every coin one episode will ever need, drawn up front.

    ``latent`` is nature's shared bit and ``flip`` each agent's private noise.
    Both modes read this same object, which is what makes their trajectories
    comparable bit for bit rather than only in distribution.
    """

    latent: np.ndarray
    flip: np.ndarray

    @property
    def bits(self) -> np.ndarray:
        """Actions as 0/1 indices, shape ``(rounds, agents)``."""

        return (self.latent ^ self.flip).astype(np.int8)


@lru_cache(maxsize=4)
def bernoulli_tape(seed: int, rounds: int, epsilons: tuple[float, ...]) -> BernoulliTape:
    """The episode's coin tape. Cached because fidelity mode reads it cell by cell.

    Deliberately keyed on exactly what it depends on - re-deriving it for the
    same seed anywhere in the process gives the identical arrays, so the
    provider, the game, and speed mode never need to pass the tape around.
    """

    return BernoulliTape(
        latent=bernoulli_draws(seed, LATENT_STREAM, (rounds, 1), 0.5),
        flip=bernoulli_draws(seed, PRIVATE_NOISE_STREAM, (rounds, len(epsilons)), np.asarray(epsilons)),
    )


class SyntheticBernoulliGame(SyntheticGame):
    """Memoryless XOR agents; every agent acts every round."""

    spec = GameSpec(
        game_type="synthetic_bernoulli",
        version=1,
        description=(
            "Synthetic calibration game: A_i = Z XOR B_i with known pairwise mutual information."
        ),
        game_family="choice",
        minimum_population=2,
        supported_topologies=("complete",),
    )

    def rules(self, config: GameConfig) -> BernoulliSpec:
        if config.type != self.spec.game_type:
            raise ValueError(f"expected game type {self.spec.game_type!r}")
        return BernoulliSpec.from_config(config)

    # -- Game contract ----------------------------------------------------

    def initialize(self, config: GameConfig, seed: int) -> GameState:
        rules = self.rules(config)
        agents = tuple(
            AgentState(
                rules.agent_id(index),
                attributes={
                    "available_actions": list(rules.actions),
                    "committed_action": None,
                    "epsilon": rules.epsilons[index],
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
                "epsilons": list(rules.epsilons),
                "round_history": [],
            },
        )

    def select_participants(
        self, state: GameState, config: GameConfig, rng: random.Random
    ) -> tuple[AgentId, ...]:
        """Everyone acts, every round - so `rng` is deliberately unused here.

        Game 1 has no pairing to seed. Games that *do* select participants
        must draw from this `rng` rather than a private one, so that a failing
        run replays exactly; there is simply nothing to draw in this one.
        """

        if state.terminated:
            raise ValueError("cannot select participants after termination")
        return tuple(agent.agent_id for agent in state.agents)

    def construct_observations(
        self, state: GameState, participants: tuple[AgentId, ...], config: GameConfig
    ) -> tuple[Observation, ...]:
        rules = self.rules(config)
        tape = bernoulli_tape(int(state.data["seed"]), rules.rounds, rules.epsilons)
        round_index = state.turn + 1
        interaction_id = InteractionId(f"interaction-{round_index:04d}")
        latent_bit = int(tape.latent[state.turn, 0])
        observations: list[Observation] = []
        for index, agent_id in enumerate(participants):
            observations.append(
                Observation(
                    agent_id=agent_id,
                    interaction_id=interaction_id,
                    participants=participants,
                    visible_state={
                        # Exactly the agent's decision input, and nothing else:
                        # its own private flip is visible to it, the other
                        # agents' flips never are.
                        "policy": POLICY,
                        "round": round_index,
                        "actions": list(rules.actions),
                        "signal": rules.actions[latent_bit],
                        "flip": bool(tape.flip[state.turn, index]),
                    },
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
                prompt=bind_bernoulli_prompt(
                    actions=rules.actions, observation=observation.visible_state
                ),
                provider_required=True,
                # A synthetic agent that produces an invalid action has a bug,
                # and retrying would hide it behind a second attempt that
                # happens to work. Zero retries makes that failure loud.
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
                "signal": request.observation.visible_state["signal"],
                "flip": request.observation.visible_state["flip"],
            },
        )

    def validate_action(
        self, state: GameState, request: DecisionRequest, action: Action, config: GameConfig
    ) -> ValidationResult:
        rules = self.rules(config)
        issues: list[ValidationIssue] = []
        if action.agent_id != request.agent_id:
            issues.append(ValidationIssue("action.agent_id", "must match the decision agent"))
        if action.stage != request.stage:
            issues.append(ValidationIssue("action.stage", "must match the decision stage"))
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
            raise ValueError("every agent reports every round in synthetic_bernoulli")
        if any(action.value not in rules.actions for action in actions):
            raise ValueError("an invalid action cannot enter a transition")

        by_agent = {action.agent_id: action.value for action in actions}
        round_index = state.turn + 1
        unanimous = len({action.value for action in actions}) == 1
        # Only the standing choice is kept. These agents are memoryless by
        # definition - A_i,t depends on nothing before round t - so appending a
        # per-round entry to `AgentState.memory` would carry no information
        # while making every checkpoint serialize a structure that grows with
        # the episode, which turns checkpointing into a quadratic cost for
        # nothing. `round_history` below is the per-round record that is
        # actually read (by the binned trajectory metrics).
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
            bind_bernoulli_prompt(
                actions=rules.actions,
                observation={
                    "policy": POLICY,
                    "round": 1,
                    "actions": list(rules.actions),
                    "signal": rules.actions[0],
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
                    assumptions=("Every agent reports independently every round.",),
                ),
            ),
            stopping_condition_assumptions=(
                f"The run stops after exactly {rules.rounds} rounds.",
            ),
            metadata={
                "population_size": rules.population_size,
                "epsilons": list(rules.epsilons),
                "provider_prices_included": False,
            },
        )

    # -- SyntheticGame contract -------------------------------------------

    def ground_truth(self, config: GameConfig) -> GroundTruth:
        """Closed-form values, derived from this config and nothing else."""

        rules = self.rules(config)
        quantities: list[GroundTruthQuantity] = []
        for index, epsilon in enumerate(rules.epsilons):
            quantities.append(
                GroundTruthQuantity(
                    name="marginal_entropy",
                    value=1.0,
                    subject=(str(rules.agent_id(index)),),
                    definition="H(A_i) = 1 bit; the latent is fair so every marginal is uniform.",
                )
            )
        pairwise: list[float] = []
        for left in range(rules.population_size):
            for right in range(left + 1, rules.population_size):
                disagreement = _disagreement_probability(
                    rules.epsilons[left], rules.epsilons[right]
                )
                value = 1.0 - binary_entropy(disagreement)
                pairwise.append(value)
                quantities.append(
                    GroundTruthQuantity(
                        name="mutual_information",
                        value=value,
                        subject=(str(rules.agent_id(left)), str(rules.agent_id(right))),
                        definition="I(A_i;A_j) = 1 - H(q), q = e_i(1-e_j) + e_j(1-e_i).",
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
                name="unanimity_probability",
                value=(
                    math.prod(1.0 - value for value in rules.epsilons)
                    + math.prod(rules.epsilons)
                ),
                units="probability",
                definition=(
                    "P(all agents report the same action) = prod(1-e_i) + prod(e_i); "
                    "the population is unanimous exactly when no agent flips or all do."
                ),
            )
        )
        return GroundTruth(
            game_type=self.spec.game_type,
            parameters=rules.to_dict(),
            quantities=tuple(quantities),
        )

    def simulate(self, config: GameConfig, seeds: Sequence[int]) -> SimulatedEpisodes:
        """Speed mode: the same tape, the same XOR, no pipeline in between."""

        rules = self.rules(config)
        seed_list = tuple(int(seed) for seed in seeds)
        actions = np.empty(
            (len(seed_list), rules.rounds, rules.population_size), dtype=np.int8
        )
        for position, seed in enumerate(seed_list):
            actions[position] = bernoulli_tape(seed, rules.rounds, rules.epsilons).bits
        return SimulatedEpisodes(
            seeds=seed_list, actions=actions, action_labels=rules.actions
        )


def _disagreement_probability(left: float, right: float) -> float:
    """P(A_i != A_j) = e_i(1-e_j) + e_j(1-e_i); the shared latent cancels."""

    return left * (1.0 - right) + right * (1.0 - left)
