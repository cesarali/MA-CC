"""Game 1 - Bernoulli. Every quantity is zero or a known constant.

This is not the easy version of the Markov game; it has a different job. There
are no dynamics and no memory, so nothing here converges and nothing here is
interesting as a language game. What it gives us is a **floor** and a
**calibration curve**:

Each round nature draws a latent bit ``Z_t ~ Bern(p)``, and agent *i* reports
``A_i,t = Z_t XOR B_i,t`` with ``B_i,t ~ Bern(eps_i)`` private and independent.
With a fair latent (``p = 0.5``, the default) marginals are uniform and

    q_ij = eps_i (1 - eps_j) + eps_j (1 - eps_i)
    I(A_i ; A_j) = 1 - H(q_ij)   bits, exactly.

Two anchors carry the weight. At ``eps = 0.5`` the true mutual information is
exactly zero, so running many seeds there measures how much MI the estimator
reports when there is none - the finite-sample bias, and therefore the
magnitude below which a number from a real run means nothing. At ``eps = 0`` it
is exactly one bit. Sweeping between them turns estimator bias into a visible
offset from the diagonal instead of a single number someone has to form a
judgement about.

``latent_bias`` moves ``p`` off 0.5, at which point the marginals stop being
uniform and the ``1 - H(q)`` shortcut stops being valid - so `ground_truth()`
derives everything from the exact 2x2 joint instead, which agrees with the
shortcut at ``p = 0.5`` and stays exact away from it. Its purpose is the
degenerate config: ``p = 1`` with ``eps = 0`` locks every agent onto one action
forever, giving zero entropy and a mutual information that is a genuine 0/0 -
the case where an estimator must return 0 rather than nan or 1.
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
    ExactDynamics,
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
    latent_bias: float = 0.5

    @classmethod
    def from_config(cls, config: GameConfig) -> "BernoulliSpec":
        options = config.options
        raw_actions = options.get("actions", ("Q", "M"))
        if isinstance(raw_actions, (str, bytes)):
            raise ValueError("game.options.actions must be a list of two labels")
        actions = tuple(str(item) for item in raw_actions)
        bias = options.get("latent_bias", 0.5)
        if isinstance(bias, bool) or not isinstance(bias, (int, float)):
            raise ValueError("game.options.latent_bias must be a number")
        spec = cls(
            population_size=config.population_size,
            rounds=config.horizon,
            actions=actions,
            epsilons=_resolve_epsilons(options, config.population_size),
            latent_bias=float(bias),
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
        if not math.isfinite(self.latent_bias) or not 0.0 <= self.latent_bias <= 1.0:
            raise ValueError("game.options.latent_bias must lie in [0, 1]")

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
            "latent_bias": self.latent_bias,
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


def pair_joint(latent_bias: float, left: float, right: float) -> np.ndarray:
    """The exact 2x2 joint P(A_i = a, A_j = b), marginalizing over the latent.

    ``P(a, b) = sum_z P(Z=z) P(B_i = a XOR z) P(B_j = b XOR z)``.

    Everything about this game's ground truth comes from this table rather than
    from the ``1 - H(q)`` shortcut. The shortcut is only valid when the latent
    is fair, and it stops being valid the moment ``latent_bias`` moves - at
    which point the marginals are no longer uniform and MI is genuinely a
    different number. Deriving from the joint keeps the closed form exact for
    every config the game accepts, including the degenerate ones.
    """

    bias = np.array([1.0 - latent_bias, latent_bias])
    noise = (np.array([1.0 - left, left]), np.array([1.0 - right, right]))
    joint = np.zeros((2, 2), dtype=float)
    for z in (0, 1):
        for a in (0, 1):
            for b in (0, 1):
                joint[a, b] += bias[z] * noise[0][a ^ z] * noise[1][b ^ z]
    return joint


def flip_count_distribution(epsilons: Sequence[float]) -> np.ndarray:
    """P(exactly k of the N agents flipped), for k = 0..N.

    A Poisson binomial, built by folding one agent in at a time, so it stays
    exact when agents have different noise levels rather than only in the
    symmetric case.

    This one distribution determines every population-shape statistic in the
    game, because the agents' actions differ from each other *only* through
    which of them flipped - the latent is common to all of them and cancels.
    That is why unanimity probability and expected dominant share below are
    independent of ``latent_bias``.
    """

    distribution = np.zeros(len(epsilons) + 1, dtype=float)
    distribution[0] = 1.0
    for epsilon in epsilons:
        shifted = np.zeros_like(distribution)
        shifted[:-1] += distribution[:-1] * (1.0 - epsilon)
        shifted[1:] += distribution[:-1] * epsilon
        distribution = shifted
    return distribution


def mutual_information_bits(joint: np.ndarray) -> float:
    """I(X;Y) in bits from an exact joint, with the 0 log 0 limit taken."""

    row = joint.sum(axis=1)
    column = joint.sum(axis=0)
    total = 0.0
    for a in range(joint.shape[0]):
        for b in range(joint.shape[1]):
            cell = joint[a, b]
            if cell <= 0.0 or row[a] <= 0.0 or column[b] <= 0.0:
                continue
            total += cell * math.log2(cell / (row[a] * column[b]))
    # A degenerate config makes MI an exact zero rather than a 0/0: if either
    # variable is constant, every surviving term is p * log2(p/p) = 0. Clamping
    # the tiny negative floating-point residue keeps that visible as 0.0
    # instead of -1e-17, which reads as a bug when it is not one.
    return max(total, 0.0)


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
def bernoulli_tape(
    seed: int, rounds: int, epsilons: tuple[float, ...], latent_bias: float = 0.5
) -> BernoulliTape:
    """The episode's coin tape. Cached because fidelity mode reads it cell by cell.

    Deliberately keyed on exactly what it depends on - re-deriving it for the
    same seed anywhere in the process gives the identical arrays, so the
    provider, the game, and speed mode never need to pass the tape around.
    """

    return BernoulliTape(
        latent=bernoulli_draws(seed, LATENT_STREAM, (rounds, 1), latent_bias),
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
        tape = bernoulli_tape(
            int(state.data["seed"]), rules.rounds, rules.epsilons, rules.latent_bias
        )
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
        bias = rules.latent_bias
        quantities: list[GroundTruthQuantity] = []
        for index, epsilon in enumerate(rules.epsilons):
            # P(A_i = actions[1]) = p(1-e_i) + (1-p)e_i: the agent reports the
            # high action when the latent is high and it did not flip, or the
            # latent is low and it did.
            high = bias * (1.0 - epsilon) + (1.0 - bias) * epsilon
            quantities.append(
                GroundTruthQuantity(
                    name="marginal_entropy",
                    value=binary_entropy(high),
                    subject=(str(rules.agent_id(index)),),
                    definition="H(A_i) for P(A_i=high) = p(1-e_i) + (1-p)e_i.",
                )
            )
        pairwise: list[float] = []
        for left in range(rules.population_size):
            for right in range(left + 1, rules.population_size):
                value = mutual_information_bits(
                    pair_joint(bias, rules.epsilons[left], rules.epsilons[right])
                )
                pairwise.append(value)
                quantities.append(
                    GroundTruthQuantity(
                        name="mutual_information",
                        value=value,
                        subject=(str(rules.agent_id(left)), str(rules.agent_id(right))),
                        definition=(
                            "I(A_i;A_j) from the exact joint over the latent; equals "
                            "1 - H(q) with q = e_i(1-e_j) + e_j(1-e_i) when p = 0.5."
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
        # The metrics below are the ordinary choice-game statistics, not
        # information-theoretic ones. They get closed forms too, so an episode
        # can check `population_action_share_per_option` and
        # `rolling_coordination_rate` against the answer key the same way the
        # mutual information is checked - the recorder is under rehearsal here
        # just as much as the estimators are.
        for index, action in enumerate(rules.actions):
            share = sum(
                (bias * (1.0 - epsilon) + (1.0 - bias) * epsilon) if index == 1
                else (bias * epsilon + (1.0 - bias) * (1.0 - epsilon))
                for epsilon in rules.epsilons
            ) / rules.population_size
            quantities.append(
                GroundTruthQuantity(
                    name="expected_action_share",
                    value=share,
                    subject=(action,),
                    units="probability",
                    definition=(
                        "E[population_action_share_per_option]; the time-average of the "
                        "recorded share converges to this."
                    ),
                )
            )
        flips = flip_count_distribution(rules.epsilons)
        size = rules.population_size
        unanimity = float(flips[0] + flips[size])
        quantities.append(
            GroundTruthQuantity(
                name="unanimity_probability",
                value=unanimity,
                units="probability",
                definition=(
                    "P(all agents report the same action): every agent flipped or none "
                    "did. Independent of the latent bias, and the expected value of "
                    "rolling_coordination_rate."
                ),
            )
        )
        quantities.append(
            GroundTruthQuantity(
                name="expected_dominant_action_share",
                value=float(
                    sum(
                        flips[count] * max(count, size - count) / size
                        for count in range(size + 1)
                    )
                ),
                units="probability",
                definition=(
                    "E[max(f, N-f)/N] over the flip-count distribution; the expected "
                    "value of dominant_action_share."
                ),
            )
        )
        quantities.append(
            GroundTruthQuantity(
                name="expected_consensus_by_success_rate",
                # consensus_flip needs a whole trailing window at or above
                # threshold. Unanimity every round (no agent can ever flip) is
                # the only way a memoryless population sustains that, so this is
                # 1 exactly when every epsilon is 0 - and 0 otherwise, however
                # coordinated the population looks on average.
                value=1.0 if all(value == 0.0 for value in rules.epsilons) else 0.0,
                units="indicator",
                definition=(
                    "1 when first_consensus_time_by_success_rate must fire (every agent "
                    "noiseless, so every round is unanimous), 0 when it must stay None. "
                    "A memoryless population never converges, so nothing in between."
                ),
            )
        )
        quantities.append(
            GroundTruthQuantity(
                name="expected_first_chance_unanimity_round",
                value=math.inf if unanimity <= 0.0 else 1.0 / unanimity,
                units="rounds",
                # This is the answer key for `first_consensus_time_by_action_share`,
                # and it is deliberately a *different* answer from the one above.
                # That metric asks whether enough agents currently hold the same
                # value, with no persistence requirement, so a single lucky round
                # satisfies it - which for a memoryless population happens by
                # chance after ~1/P(unanimity) rounds. It is a geometric waiting
                # time, so a single episode is one draw with standard deviation
                # about equal to its own mean; only the average over many seeds
                # should land near this number.
                definition=(
                    "1 / P(unanimity): the mean round at which a memoryless population "
                    "first hits unanimity by chance, which is when a standing-share "
                    "consensus criterion with no persistence requirement fires. "
                    "Geometric, so a single episode is a high-variance draw."
                ),
            )
        )
        return GroundTruth(
            game_type=self.spec.game_type,
            parameters=rules.to_dict(),
            quantities=tuple(quantities),
        )

    def exact_dynamics(self, config: GameConfig) -> "BernoulliDynamics":
        return BernoulliDynamics(self.rules(config))

    def simulate(self, config: GameConfig, seeds: Sequence[int]) -> SimulatedEpisodes:
        """Speed mode: the same tape, the same XOR, no pipeline in between."""

        rules = self.rules(config)
        seed_list = tuple(int(seed) for seed in seeds)
        actions = np.empty(
            (len(seed_list), rules.rounds, rules.population_size), dtype=np.int8
        )
        for position, seed in enumerate(seed_list):
            actions[position] = bernoulli_tape(
                seed, rules.rounds, rules.epsilons, rules.latent_bias
            ).bits
        return SimulatedEpisodes(
            seeds=seed_list, actions=actions, action_labels=rules.actions
        )


class BernoulliDynamics(ExactDynamics):
    """Game 1's exact macrostate laws, in closed form rather than by propagation.

    Rounds are i.i.d. given the condition - there are no dynamics at all - so
    the macrostate law is the same at every round and the lagged joint is
    simply the outer product. That independence is exactly what makes Game 1's
    empowerment quantities analytic:

        I(C; S_{t+h} | S_t) = I(C;S) - I(S_t; S_{t+h}),  independent of h

    ``S_{t+h}`` is conditionally independent of ``S_t`` given ``C``, but *not*
    unconditionally independent of it - they share ``C`` as a common cause. The
    consequence is a lagged conditional MI that must be **flat in h** at a
    height predicted in advance. Any slope is an artifact of windowing, an
    episode-boundary edge effect, or a null construction leaking into the
    estimate. A flat line of known height is a far stronger check than a single
    value, which is why this class computes the law rather than a number.
    """

    def __init__(self, spec: BernoulliSpec) -> None:
        self._spec = spec
        self._law = self._macrostate_law()

    def _macrostate_law(self) -> np.ndarray:
        """P(K = k), the count playing the second action, marginalizing the latent.

        Given the latent is low, each agent plays high exactly when it flipped;
        given the latent is high, exactly when it did not. So the law is a
        two-component mixture of Poisson binomials, weighted by the latent bias.
        """

        low = flip_count_distribution(self._spec.epsilons)
        high = flip_count_distribution(tuple(1.0 - e for e in self._spec.epsilons))
        bias = self._spec.latent_bias
        return (1.0 - bias) * low + bias * high

    @property
    def population_size(self) -> int:
        return self._spec.population_size

    @property
    def action_labels(self) -> tuple[str, ...]:
        return self._spec.actions

    @property
    def rounds(self) -> int:
        return self._spec.rounds

    def macrostate_law(self, round_index: int) -> np.ndarray:
        return self._law

    def macrostate_pair_law(self, round_index: int, horizon: int) -> np.ndarray:
        # Independent given the condition, at every horizon - which is the
        # entire source of the "flat in h" prediction.
        return np.outer(self._law, self._law)

    @property
    def is_lumpable(self) -> bool:
        # With no dynamics there is no macrostate *process* to be non-Markov;
        # successive macrostates are i.i.d., which is trivially Markov.
        return True
