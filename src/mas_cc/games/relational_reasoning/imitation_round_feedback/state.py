"""Configuration and typed records for ``relational_imitation_round_feedback``.

Three state variables per agent, and they are genuinely independent:

    X_i(t)  the currently voted option label       -> ``committed_action``
    H_i(t)  every fact id it has ever received     -> ``known_fact_ids``
    K_i(t)  fact ids available for reasoning now   -> ``active_fact_ids``

``X_i`` moves when the agent votes.  ``H_i`` grows only when another
participant exposes a new fact to *this* agent.  ``K_i`` can also shrink at a
population-round persistence boundary and grow again when a valid fact is
communicated.  Nothing in this module derives a vote from either fact set.

In legacy peer mode, ``(X_i, S_i)`` - vote and publicly exposed fact id - is
the whole socially visible state. In board mode, public message prose is an
intentional additional semantic channel. ``K_i`` remains the exact record of
source-evidence acquisition; it is not exhaustive semantic knowledge there.
``R_i``, the free-form private reason, is never rendered to another agent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mas_cc.config import GameConfig
from mas_cc.core import AgentId
from mas_cc.games.protocols import AgentState, GameState, Transition, _thaw

from ..data import DEFAULT_TASK_DATASET_DIR
from .prompts import (
    IMPLEMENTED_VOTE_VISIBILITIES,
    PROMPT_VERSION,
    VOTE_VISIBILITIES,
    resolve_receiver_epistemic_disposition,
)

GAME_TYPE = "relational_imitation_round_feedback"
ROUND_RECORD_TYPE = "relational_imitation_round_feedback"

INITIAL_VOTE = "initial_vote"
FOCAL_UPDATE = "focal_update"

KNOWN_FACT_IDS = "known_fact_ids"
ACTIVE_FACT_IDS = "active_fact_ids"
COMMITTED_ACTION = "committed_action"
PUBLIC_REASON = "public_reason"
PUBLIC_SHARED_FACT_ID = "public_shared_fact_id"

DYNAMICS_MODES = ("reasoning", "classical")
IMPLEMENTED_DYNAMICS_MODES = ("reasoning",)
"""``classical`` is refused rather than approximated.  A provider-free kernel
for this game would have to decide what a *fact* does to a q-voter jump, and
inventing that silently would produce numbers nobody could interpret."""

INITIALIZATION_MODES = (
    "local_vote",
    "paired_local_vote",
    "uniform_random",
    "explicit",
)

PEER_SOURCE = "peer"
CONTROLLER_SOURCE = "controller"
INITIAL_SOURCE = "initial"
FACT_SOURCES = (INITIAL_SOURCE, PEER_SOURCE, CONTROLLER_SOURCE)
"""Every way a fact is allowed to enter ``K_i``.  §18's "no information
teleportation" invariant is exactly the claim that nothing else ever does."""

SOCIAL_MODE_PEER = "peer"
SOCIAL_MODE_BOARD = "board"
SOCIAL_MODES = (SOCIAL_MODE_PEER, SOCIAL_MODE_BOARD)

BOARD_SAMPLING_UNIFORM = "uniform"
BOARD_SAMPLING_MODES = (BOARD_SAMPLING_UNIFORM,)

MESSAGE_CLAIM = "CLAIM"
MESSAGE_QUESTION = "QUESTION"
MESSAGE_REQUEST = "REQUEST"
MESSAGE_RESULT = "RESULT"
MESSAGE_REPLY = "REPLY"
MESSAGE_CORRECTION = "CORRECTION"
BOARD_MESSAGE_TYPES = (
    MESSAGE_CLAIM,
    MESSAGE_QUESTION,
    MESSAGE_REQUEST,
    MESSAGE_RESULT,
    MESSAGE_REPLY,
    MESSAGE_CORRECTION,
)
REPLY_MESSAGE_TYPES = (MESSAGE_REPLY, MESSAGE_CORRECTION)


@dataclass(frozen=True, slots=True)
class BlackboardMessage:
    """One immutable public message, retained for complete provenance."""

    message_id: str
    author_id: str
    message_type: str
    text: str
    vote: str
    shared_fact_id: str | None
    reply_to: str | None
    round_created: int
    micro_step_created: int
    expires_after_round: int
    author_kind: str = "agent"

    def __post_init__(self) -> None:
        if not self.message_id.strip():
            raise ValueError("blackboard message_id must be non-empty")
        if self.message_type not in BOARD_MESSAGE_TYPES:
            raise ValueError(
                f"blackboard message_type must be one of {list(BOARD_MESSAGE_TYPES)}"
            )
        if not self.text.strip():
            raise ValueError("blackboard message text must be non-empty")
        if self.message_type in REPLY_MESSAGE_TYPES and not self.reply_to:
            raise ValueError(f"{self.message_type} requires reply_to")
        if self.author_kind not in {"agent", "controller"}:
            raise ValueError("blackboard author_kind must be agent or controller")
        if self.expires_after_round < self.round_created:
            raise ValueError("blackboard expiry cannot precede creation")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BlackboardMessage":
        return cls(
            message_id=str(value["message_id"]),
            author_id=str(value["author_id"]),
            message_type=str(value["message_type"]),
            text=str(value["text"]),
            vote=str(value["vote"]),
            shared_fact_id=(
                None
                if value.get("shared_fact_id") is None
                else str(value["shared_fact_id"])
            ),
            reply_to=None if value.get("reply_to") is None else str(value["reply_to"]),
            round_created=int(value["round_created"]),
            micro_step_created=int(value["micro_step_created"]),
            expires_after_round=int(value["expires_after_round"]),
            author_kind=str(value.get("author_kind", "agent")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class BlackboardState:
    """Append-only message history with a round-based live-message view."""

    messages: tuple[BlackboardMessage, ...] = ()

    @classmethod
    def from_sequence(cls, values: Sequence[Mapping[str, Any]]) -> "BlackboardState":
        return cls(tuple(BlackboardMessage.from_mapping(value) for value in values))

    def append(self, message: BlackboardMessage) -> "BlackboardState":
        if self.find(message.message_id) is not None:
            raise ValueError(f"duplicate blackboard message id {message.message_id!r}")
        return BlackboardState((*self.messages, message))

    def live_messages(self, round_idx: int) -> tuple[BlackboardMessage, ...]:
        return tuple(
            message
            for message in self.messages
            if message.round_created <= round_idx <= message.expires_after_round
        )

    def sample_live(
        self,
        round_idx: int,
        count: int,
        rng: Any,
        *,
        exclude_author_id: str | None = None,
    ) -> tuple[BlackboardMessage, ...]:
        eligible = [
            message
            for message in self.live_messages(round_idx)
            if exclude_author_id is None or message.author_id != exclude_author_id
        ]
        return tuple(rng.sample(eligible, min(count, len(eligible))))

    def expire(self, round_idx: int) -> tuple["BlackboardState", tuple[str, ...]]:
        """Report messages ending now while retaining append-only history."""

        expired = tuple(
            message.message_id
            for message in self.messages
            if message.expires_after_round == round_idx
        )
        return self, expired

    def find(self, message_id: str) -> BlackboardMessage | None:
        return next(
            (message for message in self.messages if message.message_id == message_id),
            None,
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self.messages]


# --------------------------------------------------------------------------
# Agent / game state
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RelationalAgentState(AgentState):
    """One agent's historical/active facts and its standing public ballot."""

    @property
    def known_fact_ids(self) -> tuple[str, ...]:
        """``H_i(t)`` - every fact id this agent has legitimately received."""

        return tuple(str(item) for item in self.attributes.get(KNOWN_FACT_IDS, ()))

    @property
    def active_fact_ids(self) -> tuple[str, ...]:
        """``K_i(t)`` - fact ids currently available to this agent's reasoning.

        The fallback keeps old serialized states readable.  Such states were
        created before finite persistence existed, so all historical facts
        were active by definition.
        """

        raw = self.attributes.get(ACTIVE_FACT_IDS)
        active = (
            self.known_fact_ids if raw is None else tuple(str(item) for item in raw)
        )
        unknown = set(active) - set(self.known_fact_ids)
        if unknown:
            raise ValueError(
                f"agent {self.agent_id} has active facts outside known_fact_ids: "
                f"{sorted(unknown)}"
            )
        return active

    @property
    def committed_action(self) -> str | None:
        value = self.attributes.get(COMMITTED_ACTION)
        return None if value is None else str(value)

    @property
    def public_reason(self) -> str | None:
        value = self.attributes.get(PUBLIC_REASON)
        return None if value is None else str(value)

    @property
    def public_shared_fact_id(self) -> str | None:
        value = self.attributes.get(PUBLIC_SHARED_FACT_ID)
        return None if value is None else str(value)

    @property
    def initial_fact_ids(self) -> tuple[str, ...]:
        """``K_i(0)``, kept next to ``K_i(t)`` so acquisition is auditable."""

        return tuple(str(item) for item in self.attributes.get("initial_fact_ids", ()))

    @property
    def fact_provenance(self) -> Mapping[str, Any]:
        """``fact_id -> {source, round_index, within_round_index, from}``."""

        return dict(self.attributes.get("fact_provenance", {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": str(self.agent_id),
            "score": self.score,
            "committed_action": self.committed_action,
            "public_reason": self.public_reason,
            "public_shared_fact_id": self.public_shared_fact_id,
            "known_fact_ids": list(self.known_fact_ids),
            "active_fact_ids": list(self.active_fact_ids),
            "initial_fact_ids": list(self.initial_fact_ids),
            "fact_provenance": _thaw(self.attributes.get("fact_provenance", {})),
            "memory": _thaw(self.memory),
        }


@dataclass(frozen=True, slots=True)
class RelationalGameState(GameState):
    """Whole-population state for one relational reasoning episode."""

    @property
    def phase(self) -> str:
        return str(self.data["phase"])

    @property
    def task(self) -> Mapping[str, Any]:
        return self.data["task"]

    @property
    def possible_answers(self) -> tuple[str, ...]:
        """The **semantic** vote alphabet: compass relations, not letters."""

        return tuple(str(item) for item in self.task["possible_answers"])

    @property
    def option_relations(self) -> Mapping[str, str]:
        """The task's frozen ``label -> relation`` map.  Provenance only.

        Nothing in the dynamics reads this: presentation letters are drawn per
        call, see ``game.RelationalImitationRoundFeedbackGame.option_letters``.
        """

        return {
            str(key): str(value) for key, value in self.task["option_relations"].items()
        }

    @property
    def answer_display_texts(self) -> Mapping[str, str]:
        raw = self.task.get("answer_display_texts", {})
        return {str(key): str(value) for key, value in raw.items()}

    def answer_display_text(self, answer: str) -> str:
        return self.answer_display_texts.get(answer, answer)

    @property
    def correct_answer(self) -> str:
        return str(self.task["correct_answer"])

    @property
    def supporting_fact_ids(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.task["supporting_fact_ids"])

    @property
    def fact_ids(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.task["fact_order"])

    @property
    def initial_votes(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.data.get("initial_votes", ()))

    @property
    def evaluator_history(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.data.get("evaluator_history", ()))

    @property
    def event_history(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.data.get("event_history", ()))

    @property
    def blackboard(self) -> BlackboardState:
        raw = self.data.get("blackboard", ())
        if isinstance(raw, BlackboardState):
            return raw
        return BlackboardState.from_sequence(tuple(raw))

    @property
    def termination_reason(self) -> str | None:
        value = self.data.get("termination_reason")
        return None if value is None else str(value)

    @property
    def epistemic_persistence(self) -> float:
        """The per-round survival probability for currently active facts."""

        rules = self.data.get("rules", {})
        if not isinstance(rules, Mapping):
            return 1.0
        return float(rules.get("epistemic_persistence", 1.0))

    def fact_text(self, fact_id: str) -> str:
        """The frozen deterministic rendering of one fact."""

        return str(self.task["facts"][fact_id]["text"])

    def relational_agent(self, agent_id: AgentId) -> RelationalAgentState:
        agent = self.agent(agent_id)
        if not isinstance(agent, RelationalAgentState):
            raise TypeError("relational state contains a non-relational agent")
        return agent

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_type": self.game_type,
            "turn": self.turn,
            "phase": self.phase,
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "seed": int(self.data["seed"]),
            "dynamics_mode": str(self.data["dynamics_mode"]),
            "task": _thaw(self.task),
            "initial_votes": list(self.initial_votes),
            "agents": [agent.to_dict() for agent in self.agents],
            "evaluator_history": _thaw(self.evaluator_history),
            "event_history": _thaw(self.event_history),
            "blackboard": self.blackboard.to_list(),
            "rules": _thaw(self.data.get("rules", {})),
        }


@dataclass(frozen=True, slots=True)
class RelationalTransition(Transition):
    """A transition the shared recorder can read without a special case.

    ``success``/``payoff`` exist because ``observability/recorder.py`` reads
    exactly those two names off every transition it is handed; without them an
    episode runs but writes no ``metrics/streaming.csv``.
    """

    event: Mapping[str, Any] | None = None

    @property
    def success(self) -> bool:
        return bool(self.matched)

    @property
    def payoff(self) -> float:
        values = list(self.payoffs.values())
        return sum(values) / len(values) if values else 0.0

    def to_dict(self) -> dict[str, Any]:
        value = Transition.to_dict(self)
        value["event"] = None if self.event is None else _thaw(self.event)
        return value


@dataclass(frozen=True, slots=True)
class RelationalRoundRecord:
    """One complete slow-clock transition, persisted separately from micro rows."""

    round_index: int
    event: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": ROUND_RECORD_TYPE,
            "round_index": self.round_index,
            **_thaw(self.event),
        }


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _positive_int(value: Any, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return int(value)


def reasoning_fact_ids(
    agent: RelationalAgentState, epistemic_persistence: float
) -> tuple[str, ...]:
    """Return facts available to reasoning under the configured regime."""

    if epistemic_persistence == 1.0:
        active = agent.active_fact_ids
        if set(active) != set(agent.known_fact_ids):
            raise ValueError(
                f"agent {agent.agent_id} must have active_fact_ids == known_fact_ids "
                "when epistemic_persistence is 1.0"
            )
        return agent.known_fact_ids
    return agent.active_fact_ids


@dataclass(frozen=True, slots=True)
class RelationalRules:
    """Everything the game reads out of ``game.options``, validated once.

    ``horizon`` is derived, not configured: ``game.horizon`` counts *population
    rounds* and each round contains exactly ``N`` microscopic focal updates, so
    the elementary-step horizon is ``rounds * N``.  This mirrors the HiddenBench
    round-feedback game so the two remain comparable at equal ``rounds``.
    """

    n_agents: int
    rounds: int
    horizon: int
    social_group_size: int
    social_mode: str
    board_sampling: str
    board_message_lifetime_rounds: int
    board_exclude_self_authored: bool
    board_allow_no_post: bool
    dynamics_mode: str
    task_family: str
    task_dataset_dir: str
    task_id: str | None
    vote_visibility: str
    prompt_version: int
    receiver_epistemic_disposition: str
    stop_on_consensus: bool
    initialization_mode: str
    initial_votes: tuple[str, ...] | None
    initial_distribution: Mapping[str, float] | None
    invalid_response_retries: int
    expected_validation_failure_rate: float
    epistemic_persistence: float

    @property
    def social_distrust(self) -> bool:
        """Compatibility view of the retired boolean (only its two old arms)."""

        return self.receiver_epistemic_disposition == "vigilant"

    @classmethod
    def from_config(cls, config: GameConfig) -> "RelationalRules":
        if config.type != GAME_TYPE:
            raise ValueError(f"RelationalRules requires game.type {GAME_TYPE}")
        options = config.options

        n_agents = int(options.get("n_agents", config.population_size))
        if n_agents != config.population_size:
            raise ValueError("game.options.n_agents must equal game.population_size")
        rounds = _positive_int(
            options.get("rounds", config.horizon), "game.options.rounds"
        )
        social_group_size = _positive_int(
            options.get("social_group_size", 1), "game.options.social_group_size"
        )
        social_mode = str(options.get("social_mode", SOCIAL_MODE_PEER))
        if social_mode not in SOCIAL_MODES:
            raise ValueError(
                f"game.options.social_mode must be one of {list(SOCIAL_MODES)}"
            )
        if social_mode == SOCIAL_MODE_PEER and social_group_size > n_agents - 1:
            raise ValueError(
                "game.options.social_group_size must be between 1 and "
                "game.population_size - 1"
            )
        board = _mapping(options.get("board"), "game.options.board")
        board_sampling = str(board.get("sampling", BOARD_SAMPLING_UNIFORM))
        if board_sampling not in BOARD_SAMPLING_MODES:
            raise ValueError(
                f"game.options.board.sampling must be one of {list(BOARD_SAMPLING_MODES)}"
            )
        lifetime = _positive_int(
            board.get("message_lifetime_rounds", 1),
            "game.options.board.message_lifetime_rounds",
        )
        exclude_self = board.get("exclude_self_authored", True)
        allow_no_post = board.get("allow_no_post", True)
        if not isinstance(exclude_self, bool):
            raise ValueError(
                "game.options.board.exclude_self_authored must be a boolean"
            )
        if not isinstance(allow_no_post, bool):
            raise ValueError("game.options.board.allow_no_post must be a boolean")

        mode = str(options.get("dynamics_mode", "reasoning"))
        if mode not in DYNAMICS_MODES:
            raise ValueError(
                f"game.options.dynamics_mode must be one of {list(DYNAMICS_MODES)}"
            )
        if mode not in IMPLEMENTED_DYNAMICS_MODES:
            raise ValueError(
                f"game.options.dynamics_mode {mode!r} is not implemented for "
                f"{GAME_TYPE!r}; only {list(IMPLEMENTED_DYNAMICS_MODES)} is available. "
                "A provider-free kernel would have to define what an exposed fact "
                "does to a q-voter jump, which this version deliberately leaves open."
            )

        dataset_dir = options.get("task_dataset_dir", str(DEFAULT_TASK_DATASET_DIR))
        if not isinstance(dataset_dir, (str, Path)):
            raise ValueError("game.options.task_dataset_dir must be a path")
        task_id = options.get("task_id")
        if task_id is not None and not isinstance(task_id, str):
            raise ValueError("game.options.task_id must be a string")
        task_family = str(options.get("task_family", "spatial_relational"))
        if task_family not in {"spatial_relational", "musr_team_allocation"}:
            raise ValueError(
                "game.options.task_family must be spatial_relational or "
                "musr_team_allocation"
            )
        if task_family == "musr_team_allocation" and task_id is None:
            raise ValueError("MuSR Team Allocation requires game.options.task_id")

        visibility = str(options.get("vote_visibility", "public"))
        if visibility not in VOTE_VISIBILITIES:
            raise ValueError(
                f"game.options.vote_visibility must be one of {list(VOTE_VISIBILITIES)}"
            )
        if visibility not in IMPLEMENTED_VOTE_VISIBILITIES:
            raise ValueError(
                f"game.options.vote_visibility {visibility!r} is reserved; only "
                f"{list(IMPLEMENTED_VOTE_VISIBILITIES)} is implemented"
            )

        distrust = options.get("social_distrust")
        if "social_distrust" in options and not isinstance(distrust, bool):
            raise ValueError("game.options.social_distrust must be a boolean")
        if "epistemic_prompt_class" in options:
            raise ValueError(
                "game.options.epistemic_prompt_class is retired; use "
                "game.options.receiver_epistemic_disposition (naive or vigilant)"
            )
        prompt_class = options.get("receiver_epistemic_disposition")
        if prompt_class is not None and not isinstance(prompt_class, str):
            raise ValueError(
                "game.options.receiver_epistemic_disposition must be a string"
            )
        try:
            prompt_class = resolve_receiver_epistemic_disposition(
                prompt_class, distrust
            )
        except ValueError as exc:
            raise ValueError(f"game.options.{exc}") from exc

        prompt_version = options.get("prompt_version", PROMPT_VERSION)
        if isinstance(prompt_version, bool) or prompt_version != PROMPT_VERSION:
            raise ValueError(
                f"the {GAME_TYPE!r} prompt family has one version; "
                f"game.options.prompt_version must be {PROMPT_VERSION}"
            )

        initialization = _mapping(
            options.get("initialization"), "game.options.initialization"
        )
        initialization_mode = str(initialization.get("mode", "local_vote"))
        if initialization_mode not in INITIALIZATION_MODES:
            raise ValueError(
                f"game.options.initialization.mode must be one of {list(INITIALIZATION_MODES)}"
            )
        initial_votes_raw = initialization.get("initial_votes")
        initial_votes: tuple[str, ...] | None = None
        if initial_votes_raw is not None:
            if isinstance(initial_votes_raw, (str, bytes)) or not isinstance(
                initial_votes_raw, Sequence
            ):
                raise ValueError("initialization.initial_votes must be a list")
            initial_votes = tuple(str(item) for item in initial_votes_raw)
            if len(initial_votes) != n_agents:
                raise ValueError(
                    "initialization.initial_votes must contain one vote per agent"
                )
        if initialization_mode == "explicit" and initial_votes is None:
            raise ValueError(
                "initialization.mode 'explicit' requires initialization.initial_votes"
            )
        distribution_raw = initialization.get("initial_distribution")
        distribution: Mapping[str, float] | None = None
        if distribution_raw is not None:
            values = _mapping(
                distribution_raw, "game.options.initialization.initial_distribution"
            )
            distribution = {str(key): float(value) for key, value in values.items()}
            if (
                any(value < 0 for value in distribution.values())
                or sum(distribution.values()) <= 0
            ):
                raise ValueError(
                    "initial_distribution weights must be non-negative and nonzero"
                )

        retries = options.get("invalid_response_retries", 0)
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError(
                "game.options.invalid_response_retries must be an integer >= 0"
            )
        expected_failure = float(options.get("expected_validation_failure_rate", 0.0))
        if not 0 <= expected_failure <= 1:
            raise ValueError("expected_validation_failure_rate must be between 0 and 1")
        persistence_raw = options.get("epistemic_persistence", 1.0)
        if isinstance(persistence_raw, bool) or not isinstance(
            persistence_raw, (int, float)
        ):
            raise ValueError("game.options.epistemic_persistence must be a number")
        persistence = float(persistence_raw)
        if not math.isfinite(persistence) or not 0.0 <= persistence <= 1.0:
            raise ValueError(
                "game.options.epistemic_persistence must be between 0.0 and 1.0"
            )

        return cls(
            n_agents=n_agents,
            rounds=rounds,
            horizon=rounds * n_agents,
            social_group_size=social_group_size,
            social_mode=social_mode,
            board_sampling=board_sampling,
            board_message_lifetime_rounds=lifetime,
            board_exclude_self_authored=exclude_self,
            board_allow_no_post=allow_no_post,
            dynamics_mode=mode,
            task_family=task_family,
            task_dataset_dir=str(dataset_dir),
            task_id=task_id,
            vote_visibility=visibility,
            prompt_version=int(prompt_version),
            receiver_epistemic_disposition=prompt_class,
            stop_on_consensus=bool(options.get("stop_on_consensus", False)),
            initialization_mode=initialization_mode,
            initial_votes=initial_votes,
            initial_distribution=distribution,
            invalid_response_retries=int(retries),
            expected_validation_failure_rate=expected_failure,
            epistemic_persistence=persistence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            field: _thaw(getattr(self, field)) for field in self.__dataclass_fields__
        }


__all__ = [
    "ACTIVE_FACT_IDS",
    "COMMITTED_ACTION",
    "CONTROLLER_SOURCE",
    "DYNAMICS_MODES",
    "FACT_SOURCES",
    "FOCAL_UPDATE",
    "GAME_TYPE",
    "IMPLEMENTED_DYNAMICS_MODES",
    "INITIALIZATION_MODES",
    "INITIAL_SOURCE",
    "INITIAL_VOTE",
    "KNOWN_FACT_IDS",
    "PEER_SOURCE",
    "PUBLIC_REASON",
    "PUBLIC_SHARED_FACT_ID",
    "ROUND_RECORD_TYPE",
    "RelationalAgentState",
    "RelationalGameState",
    "RelationalRoundRecord",
    "RelationalRules",
    "RelationalTransition",
    "reasoning_fact_ids",
]
