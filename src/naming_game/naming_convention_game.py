"""LLM naming-convention game adapted from ``external/AI-norms``.

This is the repeated coordination experiment from Ashery et al., not the
inventory-based minimal Naming Game implemented by the other engines in this
package.  A randomly selected pair simultaneously chooses one action from a
shared name pool.  Matching choices reward both agents; different choices
penalize both agents.  Every LLM sees only its own bounded interaction history
and is not told that it belongs to a population.

The module only defines the game.  Importing it never creates an API client,
makes a request, or starts an experiment.
"""

from __future__ import annotations

import ast
import asyncio
import json
import random
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .api_client import LLMClient
from .models import ConfigurationError, LLMResponse


class InvalidConventionResponse(RuntimeError):
    """Raised when an agent never returns an action from the configured pool."""


@dataclass(frozen=True)
class ConventionGameConfig:
    """Parameters of one homogeneous naming-convention population.

    Defaults reproduce the principal settings in ``external/AI-norms``.  The
    reference experiments varied ``actions`` between two, ten, and twenty-six
    names while normally keeping a memory length of five and a population of
    twenty-four agents.
    """

    num_agents: int = 24
    actions: tuple[str, ...] = ("Q", "M")
    memory_size: int = 5
    success_reward: int = 100
    failure_payoff: int = -50
    temperature: float = 0.5
    max_tokens: int = 15
    advertised_rounds: int = 100
    convergence_window: int | None = None
    convergence_threshold: float = 1.0
    invalid_response_retries: int = 2

    def __post_init__(self) -> None:
        normalized_actions = tuple(str(action) for action in self.actions)
        object.__setattr__(self, "actions", normalized_actions)
        if self.num_agents < 2:
            raise ConfigurationError("num_agents must be at least 2.")
        if len(normalized_actions) < 2:
            raise ConfigurationError("The convention pool needs at least two actions.")
        if len(set(normalized_actions)) != len(normalized_actions):
            raise ConfigurationError("Convention actions must be unique.")
        if any(
            not action.strip() or "\n" in action or "\r" in action
            for action in normalized_actions
        ):
            raise ConfigurationError(
                "Convention actions must be non-empty single-line strings."
            )
        if self.memory_size < 0:
            raise ConfigurationError("memory_size cannot be negative.")
        if self.max_tokens < 1:
            raise ConfigurationError("max_tokens must be positive.")
        if self.advertised_rounds < 1:
            raise ConfigurationError("advertised_rounds must be positive.")
        if self.temperature < 0:
            raise ConfigurationError("temperature cannot be negative.")
        if self.convergence_window is not None and self.convergence_window < 1:
            raise ConfigurationError("convergence_window must be positive when set.")
        if not 0.0 < self.convergence_threshold <= 1.0:
            raise ConfigurationError(
                "convergence_threshold must be greater than 0 and at most 1."
            )
        if self.invalid_response_retries < 0:
            raise ConfigurationError("invalid_response_retries cannot be negative.")


@dataclass(frozen=True)
class ConventionHistoryEntry:
    """One private observation in an agent's complete history."""

    interaction_index: int | None
    own_action: str
    partner_action: str
    payoff: int
    score_after: int
    success: bool


@dataclass
class ConventionAgent:
    """Independent state for one LLM agent.

    A single stateless API client can serve the whole population because all
    agent-specific state lives here and is supplied explicitly in each prompt.
    """

    agent_id: int
    history: list[ConventionHistoryEntry] = field(default_factory=list)
    interaction_partners: list[int] = field(default_factory=list)
    score: int = 0
    score_history: list[int] = field(default_factory=list)
    committed_action: str | None = None

    @property
    def committed(self) -> bool:
        return self.committed_action is not None

    def remember(
        self,
        *,
        interaction_index: int | None,
        own_action: str,
        partner_action: str,
        payoff: int,
        partner_id: int | None,
    ) -> None:
        self.score += payoff
        self.score_history.append(self.score)
        if partner_id is not None:
            self.interaction_partners.append(partner_id)
        self.history.append(
            ConventionHistoryEntry(
                interaction_index=interaction_index,
                own_action=own_action,
                partner_action=partner_action,
                payoff=payoff,
                score_after=self.score,
                success=own_action == partner_action,
            )
        )


@dataclass(frozen=True)
class ConventionAgentState:
    """Immutable final-state snapshot returned by a run."""

    agent_id: int
    score: int
    score_history: tuple[int, ...]
    history: tuple[ConventionHistoryEntry, ...]
    interaction_partners: tuple[int, ...]
    committed_action: str | None


@dataclass(frozen=True)
class ConventionDecision:
    """A validated LLM decision or a fixed committed-agent decision."""

    action: str
    reason: str | None
    action_order: tuple[str, ...]
    response: LLMResponse | None
    responses: tuple[LLMResponse, ...]
    committed: bool = False


@dataclass(frozen=True)
class ConventionInteraction:
    """Complete audit record for one simultaneous pair interaction."""

    interaction_index: int
    player_1_id: int
    player_2_id: int
    player_1_action: str
    player_2_action: str
    payoff: int
    success: bool
    player_1_score_after: int
    player_2_score_after: int
    player_1_decision: ConventionDecision
    player_2_decision: ConventionDecision
    wall_seconds: float

    def to_log_dict(self) -> dict[str, Any]:
        def decision_fields(decision: ConventionDecision) -> dict[str, Any]:
            response = decision.response
            return {
                "action": decision.action,
                "reason": decision.reason,
                "action_order": list(decision.action_order),
                "committed": decision.committed,
                "response": response.content if response is not None else None,
                "model": response.model if response is not None else None,
                "latency_seconds": (
                    response.latency_seconds if response is not None else None
                ),
                "retries": response.retries if response is not None else None,
                "validation_attempts": len(decision.responses),
                "token_usage": asdict(response.usage) if response is not None else None,
            }

        return {
            "interaction_index": self.interaction_index,
            "players": [self.player_1_id, self.player_2_id],
            "actions": [self.player_1_action, self.player_2_action],
            "payoff": self.payoff,
            "success": self.success,
            "scores_after": [self.player_1_score_after, self.player_2_score_after],
            "player_1_decision": decision_fields(self.player_1_decision),
            "player_2_decision": decision_fields(self.player_2_decision),
            "wall_seconds": self.wall_seconds,
        }


@dataclass(frozen=True)
class ConventionGameResult:
    """The records and state produced by one call to ``run``."""

    interactions: tuple[ConventionInteraction, ...]
    start_interaction_index: int | None
    end_interaction_index: int
    converged: bool
    convergence_interaction_index: int | None
    convention: str | None
    final_window_success_rate: float | None
    population_size: int
    population_rounds: float
    agent_states: Mapping[int, ConventionAgentState]
    wall_seconds: float


def _visible_history(
    agent: ConventionAgent, memory_size: int
) -> tuple[ConventionHistoryEntry, ...]:
    if memory_size == 0:
        return ()
    return tuple(agent.history[-memory_size:])


def build_convention_messages(
    *,
    agent: ConventionAgent,
    action_order: Sequence[str],
    memory_size: int,
    success_reward: int,
    failure_payoff: int,
    advertised_rounds: int = 100,
) -> list[dict[str, str]]:
    """Build the answer-first prompt used by the reference experiment.

    Partner IDs, population state, global interaction number, and histories of
    other agents are deliberately absent.  Visible rounds are renumbered from
    one inside the bounded memory window, as in ``external/AI-norms``.
    """

    visible = _visible_history(agent, memory_size)
    local_score = sum(entry.payoff for entry in visible)
    action_json = json.dumps(list(action_order), ensure_ascii=False)
    sections = [
        (
            f"Context: Player 1 is playing a multi-round partnership game with "
            f"Player 2 for {advertised_rounds} rounds.\n"
            "At each round, Player 1 and Player 2 simultaneously pick an action "
            f"from the following values: {action_json}.\n"
            "The payoff that both players get is determined by the following rule:\n"
            "1. If Players play the SAME action as each other, they will both be "
            f"REWARDED with payoff +{success_reward} points.\n"
            "2. If Players play DIFFERENT actions to each other, they will both be "
            f"PUNISHED with payoff {failure_payoff} points.\n"
            "The objective of each Player is to maximize their own accumulated "
            "point tally, conditional on the behavior of the other player."
        )
    ]
    if visible:
        sections.append("This is the history of choices in past rounds:")
        for local_round, entry in enumerate(visible, start=1):
            sections.append(
                json.dumps(
                    {
                        "round": local_round,
                        "Player 1": entry.own_action,
                        "Player 2": entry.partner_action,
                        "payoff": entry.payoff,
                    },
                    ensure_ascii=False,
                )
            )

    sections.append(
        f"It is now round {len(visible) + 1}. The current score of Player 1 is "
        f"{local_score}. Answer saying which value Player 1 should pick. Please "
        "think step by step before making a decision. Remember, examining history "
        "explicitly is important. Put the decision before the explanation and "
        "return only JSON using this format: "
        '{"value":"<VALUE_OF_PLAYER_1>","reason":"<YOUR_REASON>"}.'
    )
    return [
        {"role": "system", "content": "\n".join(sections)},
        {"role": "user", "content": "Answer saying which action Player 1 should play."},
    ]


def _parse_mapping(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|python)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)

    candidates = [stripped]
    start, end = stripped.find("{"), stripped.rfind("}")
    if 0 <= start < end:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
        if isinstance(value, dict):
            return value
    raise ValueError("response does not contain a valid JSON object")


def parse_convention_decision(
    content: str, actions: Sequence[str]
) -> tuple[str, str | None]:
    """Extract a valid action without inferring one from free-form reasoning."""

    body = _parse_mapping(content)
    candidate = body.get("value", body.get("action"))
    if not isinstance(candidate, str) or candidate not in actions:
        raise ValueError("response value is not one of the configured actions")
    reason_value = body.get("reason")
    reason = reason_value.strip() if isinstance(reason_value, str) else None
    return candidate, reason or None


class NamingConventionGame:
    """Random-sequential population engine for the LLM convention game.

    Pair interactions are sequential at the population level.  The two choices
    inside a pair are requested concurrently from the same pre-interaction state,
    preserving the simultaneous-choice rule while using ``LLMClient``'s existing
    concurrency controls.
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        config: ConventionGameConfig | None = None,
        seed: int = 1,
        adjacency: Mapping[int, Sequence[int]] | None = None,
    ) -> None:
        self.client = client
        self.config = config or ConventionGameConfig()
        self.seed = seed
        self.rng = random.Random(seed)
        self.base_population_size = self.config.num_agents
        self.agents: dict[int, ConventionAgent] = {
            agent_id: ConventionAgent(agent_id=agent_id)
            for agent_id in range(1, self.config.num_agents + 1)
        }
        self.neighbors = self._build_neighbors(adjacency)
        self.interactions: list[ConventionInteraction] = []

    @property
    def convergence_window(self) -> int:
        return self.config.convergence_window or 3 * self.base_population_size

    def _build_neighbors(
        self, adjacency: Mapping[int, Sequence[int]] | None
    ) -> dict[int, tuple[int, ...]]:
        ids = set(self.agents)
        if adjacency is None:
            return {
                agent_id: tuple(sorted(ids - {agent_id})) for agent_id in sorted(ids)
            }
        if set(adjacency) != ids:
            raise ConfigurationError(
                "adjacency must contain exactly one entry for every agent ID."
            )
        result: dict[int, tuple[int, ...]] = {}
        for agent_id, raw_neighbors in adjacency.items():
            neighbors = tuple(dict.fromkeys(raw_neighbors))
            if not neighbors:
                raise ConfigurationError(f"Agent {agent_id} has no neighbors.")
            if agent_id in neighbors:
                raise ConfigurationError("Agents cannot be their own neighbors.")
            if any(neighbor not in ids for neighbor in neighbors):
                raise ConfigurationError("adjacency refers to an unknown agent ID.")
            result[agent_id] = neighbors
        return result

    def _assert_pristine_histories(self) -> None:
        if self.interactions or any(agent.history for agent in self.agents.values()):
            raise RuntimeError("Initial histories can only be seeded on a fresh game.")

    def seed_consensus_history(
        self, action: str, *, repetitions: int | None = None
    ) -> None:
        """Give every non-committed agent a successful consensus memory."""

        self._assert_pristine_histories()
        self._validate_action(action)
        count = self.config.memory_size if repetitions is None else repetitions
        if count < 0:
            raise ValueError("repetitions cannot be negative.")
        for _ in range(count):
            for agent in self.agents.values():
                if not agent.committed:
                    agent.remember(
                        interaction_index=None,
                        own_action=action,
                        partner_action=action,
                        payoff=self.config.success_reward,
                        partner_id=None,
                    )

    def seed_random_history(self, *, repetitions: int | None = None) -> None:
        """Give each non-committed agent an independent synthetic memory."""

        self._assert_pristine_histories()
        count = self.config.memory_size if repetitions is None else repetitions
        if count < 0:
            raise ValueError("repetitions cannot be negative.")
        for _ in range(count):
            for agent in self.agents.values():
                if agent.committed:
                    continue
                own_action = self.rng.choice(self.config.actions)
                partner_action = self.rng.choice(self.config.actions)
                agent.remember(
                    interaction_index=None,
                    own_action=own_action,
                    partner_action=partner_action,
                    payoff=self._payoff(own_action, partner_action),
                    partner_id=None,
                )

    def introduce_committed_minority(
        self,
        *,
        size: int,
        action: str,
        mode: Literal["swap", "inject"] = "swap",
    ) -> tuple[int, ...]:
        """Install fixed-action agents, preserving the reference swap/inject modes.

        ``swap`` converts existing agents and retains their histories. ``inject``
        adds new, empty-history agents and makes the resulting graph complete,
        matching the reference implementation's injection behavior.
        """

        self._validate_action(action)
        if size < 0:
            raise ValueError("size cannot be negative.")
        if size == 0:
            return ()
        if mode == "swap":
            available = [agent for agent in self.agents.values() if not agent.committed]
            if size > len(available):
                raise ValueError("Cannot swap more agents than are uncommitted.")
            selected = self.rng.sample(available, size)
            for agent in selected:
                agent.committed_action = action
            return tuple(sorted(agent.agent_id for agent in selected))
        if mode != "inject":
            raise ValueError("mode must be either 'swap' or 'inject'.")

        next_id = max(self.agents, default=0) + 1
        new_ids = tuple(range(next_id, next_id + size))
        for agent_id in new_ids:
            self.agents[agent_id] = ConventionAgent(
                agent_id=agent_id, committed_action=action
            )
        all_ids = set(self.agents)
        self.neighbors = {
            agent_id: tuple(sorted(all_ids - {agent_id}))
            for agent_id in sorted(all_ids)
        }
        return new_ids

    def _validate_action(self, action: str) -> None:
        if action not in self.config.actions:
            raise ValueError(f"Unknown convention action: {action!r}.")

    def _payoff(self, first: str, second: str) -> int:
        return (
            self.config.success_reward
            if first == second
            else self.config.failure_payoff
        )

    def _sample_pair(self) -> tuple[ConventionAgent, ConventionAgent]:
        first_id = self.rng.choice(tuple(self.agents))
        second_id = self.rng.choice(self.neighbors[first_id])
        return self.agents[first_id], self.agents[second_id]

    def _random_action_order(self) -> tuple[str, ...]:
        actions = list(self.config.actions)
        self.rng.shuffle(actions)
        return tuple(actions)

    async def _request_decision(
        self, agent: ConventionAgent, action_order: tuple[str, ...]
    ) -> ConventionDecision:
        if agent.committed_action is not None:
            return ConventionDecision(
                action=agent.committed_action,
                reason="fixed committed-minority action",
                action_order=action_order,
                response=None,
                responses=(),
                committed=True,
            )

        messages = build_convention_messages(
            agent=agent,
            action_order=action_order,
            memory_size=self.config.memory_size,
            success_reward=self.config.success_reward,
            failure_payoff=self.config.failure_payoff,
            advertised_rounds=self.config.advertised_rounds,
        )
        responses: list[LLMResponse] = []
        last_error = "unknown validation error"
        for _ in range(self.config.invalid_response_retries + 1):
            response = await self.client.complete(
                messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            responses.append(response)
            try:
                action, reason = parse_convention_decision(
                    response.content, self.config.actions
                )
                return ConventionDecision(
                    action=action,
                    reason=reason,
                    action_order=action_order,
                    response=response,
                    responses=tuple(responses),
                )
            except ValueError as exc:
                last_error = str(exc)
        raise InvalidConventionResponse(
            f"Agent {agent.agent_id} returned no valid convention after "
            f"{len(responses)} attempts: {last_error}."
        )

    async def play_interaction(self) -> ConventionInteraction:
        """Select one ordered pair, obtain simultaneous choices, and update state."""

        player_1, player_2 = self._sample_pair()
        order_1 = self._random_action_order()
        order_2 = self._random_action_order()
        interaction_index = len(self.interactions) + 1
        started = time.perf_counter()

        decision_1, decision_2 = await asyncio.gather(
            self._request_decision(player_1, order_1),
            self._request_decision(player_2, order_2),
        )
        payoff = self._payoff(decision_1.action, decision_2.action)
        player_1.remember(
            interaction_index=interaction_index,
            own_action=decision_1.action,
            partner_action=decision_2.action,
            payoff=payoff,
            partner_id=player_2.agent_id,
        )
        player_2.remember(
            interaction_index=interaction_index,
            own_action=decision_2.action,
            partner_action=decision_1.action,
            payoff=payoff,
            partner_id=player_1.agent_id,
        )
        record = ConventionInteraction(
            interaction_index=interaction_index,
            player_1_id=player_1.agent_id,
            player_2_id=player_2.agent_id,
            player_1_action=decision_1.action,
            player_2_action=decision_2.action,
            payoff=payoff,
            success=decision_1.action == decision_2.action,
            player_1_score_after=player_1.score,
            player_2_score_after=player_2.score,
            player_1_decision=decision_1,
            player_2_decision=decision_2,
            wall_seconds=time.perf_counter() - started,
        )
        self.interactions.append(record)
        return record

    def _window_status(
        self,
        records: Sequence[ConventionInteraction],
        *,
        target_action: str | None,
        window_size: int,
        threshold: float,
    ) -> tuple[bool, float | None, str | None]:
        if len(records) < window_size:
            return False, None, None
        window = records[-window_size:]
        success_rate = sum(record.success for record in window) / window_size
        if success_rate < threshold:
            return False, success_rate, None

        choices = [
            action
            for record in window
            for action in (record.player_1_action, record.player_2_action)
        ]
        convention, count = Counter(choices).most_common(1)[0]
        convention_share = count / len(choices)
        if target_action is not None:
            target_share = choices.count(target_action) / len(choices)
            converged = target_share >= threshold
            return converged, success_rate, target_action if converged else None
        # The reference implementation defines convergence from coordination
        # success alone.  A convention label is returned only when the action
        # observations support one at the same threshold.
        return True, success_rate, convention if convention_share >= threshold else None

    def _agent_states(self) -> dict[int, ConventionAgentState]:
        return {
            agent_id: ConventionAgentState(
                agent_id=agent_id,
                score=agent.score,
                score_history=tuple(agent.score_history),
                history=tuple(agent.history),
                interaction_partners=tuple(agent.interaction_partners),
                committed_action=agent.committed_action,
            )
            for agent_id, agent in self.agents.items()
        }

    async def run(
        self,
        max_interactions: int,
        *,
        stop_on_convergence: bool = True,
        target_action: str | None = None,
        convergence_window: int | None = None,
        convergence_threshold: float | None = None,
    ) -> ConventionGameResult:
        """Run a bounded stage without ever issuing requests at import time.

        Convergence is evaluated only over interactions created by this call.
        This makes a second call suitable for a committed-minority phase without
        accidentally counting the preceding consensus history.  Baseline runs
        normally use a threshold of 1.0 over ``3 * N`` interactions.  To apply
        the paper's committed-minority success criterion, pass
        ``convergence_threshold`` as 0.95. Passing the minority's action as
        ``target_action`` additionally verifies that coordination actually
        flipped to that convention rather than remaining on the prior one.
        """

        if max_interactions < 0:
            raise ValueError("max_interactions cannot be negative.")
        if target_action is not None:
            self._validate_action(target_action)
        window_size = (
            self.convergence_window
            if convergence_window is None
            else convergence_window
        )
        threshold = (
            self.config.convergence_threshold
            if convergence_threshold is None
            else convergence_threshold
        )
        if window_size < 1:
            raise ValueError("convergence_window must be positive.")
        if not 0.0 < threshold <= 1.0:
            raise ValueError(
                "convergence_threshold must be greater than 0 and at most 1."
            )

        start_offset = len(self.interactions)
        stage_records: list[ConventionInteraction] = []
        converged = False
        convention: str | None = None
        convergence_index: int | None = None
        success_rate: float | None = None
        started = time.perf_counter()

        for _ in range(max_interactions):
            stage_records.append(await self.play_interaction())
            converged, success_rate, convention = self._window_status(
                stage_records,
                target_action=target_action,
                window_size=window_size,
                threshold=threshold,
            )
            if converged:
                if convergence_index is None:
                    convergence_index = stage_records[-1].interaction_index
                if stop_on_convergence:
                    break

        if stage_records and success_rate is None:
            tail = stage_records[-min(len(stage_records), window_size) :]
            success_rate = sum(record.success for record in tail) / len(tail)
        end_index = len(self.interactions)
        return ConventionGameResult(
            interactions=tuple(stage_records),
            start_interaction_index=(start_offset + 1 if stage_records else None),
            end_interaction_index=end_index,
            converged=converged,
            convergence_interaction_index=convergence_index,
            convention=convention,
            final_window_success_rate=success_rate,
            population_size=len(self.agents),
            population_rounds=(len(stage_records) / len(self.agents)),
            agent_states=self._agent_states(),
            wall_seconds=time.perf_counter() - started,
        )


__all__ = [
    "ConventionAgent",
    "ConventionAgentState",
    "ConventionDecision",
    "ConventionGameConfig",
    "ConventionGameResult",
    "ConventionHistoryEntry",
    "ConventionInteraction",
    "InvalidConventionResponse",
    "NamingConventionGame",
    "build_convention_messages",
    "parse_convention_decision",
]
