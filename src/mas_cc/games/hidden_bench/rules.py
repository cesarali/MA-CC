"""Validated `game.options` for both HiddenBench games.

Same idiom as `naming_convention/game.py::NamingConventionGameSpec`: one frozen
dataclass per game that parses `GameConfig.options` once, validates it loudly,
and is then the only thing the game reads. A misconfigured run fails at
`initialize` with a field name, not three interactions later with a `KeyError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from mas_cc.config import GameConfig
from mas_cc.core import AgentId

from .data import SCHEMES
from .schemas import HiddenBenchDataError

if TYPE_CHECKING:
    from .records import HiddenBenchGameState

PAYOFF_MODES = ("coordination", "correctness", "coordination_plus_correctness")


def _integer(options: Mapping[str, Any], name: str, default: int, *, minimum: int | None = None) -> int:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise HiddenBenchDataError(f"game.options.{name} must be an integer")
    if minimum is not None and value < minimum:
        raise HiddenBenchDataError(f"game.options.{name} must be at least {minimum}")
    return value


def _boolean(options: Mapping[str, Any], name: str, default: bool) -> bool:
    value = options.get(name, default)
    if not isinstance(value, bool):
        raise HiddenBenchDataError(f"game.options.{name} must be a boolean")
    return value


def _number(options: Mapping[str, Any], name: str, default: float) -> float:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HiddenBenchDataError(f"game.options.{name} must be a number")
    return float(value)


def _optional_text(options: Mapping[str, Any], name: str) -> str | None:
    value = options.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise HiddenBenchDataError(f"game.options.{name} must be a string or null")
    return value or None


@dataclass(frozen=True, slots=True)
class _CommonRules:
    """Options both games share: which task, how it is split, who sees what."""

    n_agents: int
    task_set: str
    task_id: str | None
    profile: str
    assignment_scheme: str
    corpus_root: str | None
    shuffle_facts: bool
    extra_prompt: str | None
    dissenter_extra_prompt: str | None
    dissenter_fraction: float
    aggregation: str
    decoy: str | None
    invalid_response_retries: int
    expected_validation_failure_rate: float

    population_size: int = 0
    """`GameConfig.population_size`, carried purely to cross-check `n_agents`."""

    def validate_common(self) -> None:
        if self.n_agents < 2:
            raise HiddenBenchDataError("hidden_bench needs at least two agents")
        if self.population_size and self.n_agents != self.population_size:
            # A silent disagreement here is a real footgun: the game would build
            # `n_agents` agents while metric binning and preflight sized
            # themselves from `population_size`. Sweep `game.population_size` in
            # a grid and leave `n_agents` unset, or set both to the same number.
            raise HiddenBenchDataError(
                f"game.options.n_agents ({self.n_agents}) disagrees with "
                f"game.population_size ({self.population_size}); they describe the same "
                "population and must match"
            )
        if self.task_set not in {"vanilla", "expanded"}:
            raise HiddenBenchDataError("game.options.task_set must be 'vanilla' or 'expanded'")
        if self.profile not in {"hidden", "full"}:
            raise HiddenBenchDataError("game.options.profile must be 'hidden' or 'full'")
        if self.assignment_scheme not in SCHEMES:
            raise HiddenBenchDataError(
                f"game.options.assignment_scheme must be one of {', '.join(SCHEMES)}"
            )
        if self.aggregation not in {"average", "majority"}:
            raise HiddenBenchDataError("game.options.aggregation must be 'average' or 'majority'")
        if not 0.0 <= self.dissenter_fraction <= 1.0:
            raise HiddenBenchDataError("game.options.dissenter_fraction must be in [0, 1]")
        if self.dissenter_fraction and self.dissenter_extra_prompt is None:
            raise HiddenBenchDataError(
                "game.options.dissenter_fraction is set but dissenter_extra_prompt is null; "
                "a dissenter with no persona is indistinguishable from an ordinary agent"
            )
        if not 0.0 <= self.expected_validation_failure_rate <= 1.0:
            raise HiddenBenchDataError(
                "game.options.expected_validation_failure_rate must be in [0, 1]"
            )

    @property
    def expected_attempts_per_request(self) -> float:
        probability = self.expected_validation_failure_rate
        return sum(probability**attempt for attempt in range(self.invalid_response_retries + 1))

    def extra_for(self, agent_id: AgentId, state: "HiddenBenchGameState") -> str | None:
        """This agent's `%extra%` (§1.6): the group default, or a dissenter's.

        Group composition is expressed as "the first `k` agents in the recorded
        speaking order are dissenters" rather than a random draw per decision,
        so the composition of a run is a property of its seed and is written
        into the artifact.
        """

        if not self.dissenter_fraction:
            return self.extra_prompt
        order = [str(item) for item in state.data.get("speaking_order", ())]
        count = round(self.dissenter_fraction * len(order))
        if str(agent_id) in order[:count]:
            return self.dissenter_extra_prompt
        return self.extra_prompt

    @staticmethod
    def _common_kwargs(config: GameConfig) -> dict[str, Any]:
        options = config.options
        return {
            "population_size": config.population_size,
            "n_agents": _integer(options, "n_agents", config.population_size, minimum=2),
            "task_set": str(options.get("task_set", "vanilla")),
            "task_id": _optional_text(options, "task_id"),
            "profile": str(options.get("profile", "hidden")),
            "assignment_scheme": str(options.get("assignment_scheme", "bijective")),
            "corpus_root": _optional_text(options, "corpus_root"),
            "shuffle_facts": _boolean(options, "shuffle_facts", True),
            "extra_prompt": _optional_text(options, "extra_prompt"),
            "dissenter_extra_prompt": _optional_text(options, "dissenter_extra_prompt"),
            "dissenter_fraction": _number(options, "dissenter_fraction", 0.0),
            "aggregation": str(options.get("aggregation", "average")),
            "decoy": _optional_text(options, "decoy"),
            "invalid_response_retries": _integer(options, "invalid_response_retries", 2, minimum=0),
            "expected_validation_failure_rate": _number(
                options, "expected_validation_failure_rate", 0.0
            ),
        }


@dataclass(frozen=True, slots=True)
class VanillaRules(_CommonRules):
    """`hidden_bench_vanilla` options."""

    rounds: int = 10
    rounds_are_speaking_turns: bool = False

    @classmethod
    def from_config(cls, config: GameConfig) -> "VanillaRules":
        options = config.options
        rules = cls(
            **cls._common_kwargs(config),
            rounds=_integer(options, "rounds", 10, minimum=0),
            rounds_are_speaking_turns=_boolean(options, "rounds_are_speaking_turns", False),
        )
        rules.validate_common()
        if rules.rounds_are_speaking_turns and rules.rounds and rules.rounds < rules.n_agents:
            # Not an error - the paper's own runner does exactly this at T=5,
            # N=7 - but it means some agents never speak, and a group-size
            # result computed from it is about airtime, not about group size.
            pass
        return rules

    @property
    def has_discussion(self) -> bool:
        return self.total_discussion_turns > 0

    @property
    def total_discussion_turns(self) -> int:
        """How many speaking turns the discussion contains.

        `rounds_are_speaking_turns: false` (the brief's §5 unit): `T` full
        round-robin passes, so `T x N` turns.
        `rounds_are_speaking_turns: true` (the paper's runner's unit): `T` turns
        total, regardless of `N`.
        """

        return self.rounds if self.rounds_are_speaking_turns else self.rounds * self.n_agents

    def discussion_turns_complete(self, completed_rounds: int, speaker_index: int, n: int) -> bool:
        turns = completed_rounds * n + speaker_index
        return turns >= self.total_discussion_turns

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_agents": self.n_agents,
            "task_set": self.task_set,
            "task_id": self.task_id,
            "profile": self.profile,
            "assignment_scheme": self.assignment_scheme,
            "rounds": self.rounds,
            "rounds_are_speaking_turns": self.rounds_are_speaking_turns,
            "total_discussion_turns": self.total_discussion_turns,
            "extra_prompt": self.extra_prompt,
            "dissenter_extra_prompt": self.dissenter_extra_prompt,
            "dissenter_fraction": self.dissenter_fraction,
            "aggregation": self.aggregation,
            "shuffle_facts": self.shuffle_facts,
            "decoy": self.decoy,
        }


@dataclass(frozen=True, slots=True)
class PayoffRules:
    """§6's three payoff modes, one of which is paying and all of which are measured."""

    mode: str = "coordination"
    match_reward: float = 1.0
    mismatch_penalty: float = -1.0
    correctness_bonus: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in PAYOFF_MODES:
            raise HiddenBenchDataError(
                f"game.options.payoff.mode must be one of {', '.join(PAYOFF_MODES)}"
            )

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "PayoffRules":
        payoff = options.get("payoff", {})
        if not isinstance(payoff, Mapping):
            raise HiddenBenchDataError("game.options.payoff must be a mapping")
        return cls(
            mode=str(payoff.get("mode", "coordination")),
            match_reward=_number(payoff, "match_reward", 1.0),
            mismatch_penalty=_number(payoff, "mismatch_penalty", -1.0),
            correctness_bonus=_number(payoff, "correctness_bonus", 0.5),
        )

    def payoff(self, *, matched: bool, correct: bool) -> float:
        """What this pair member is actually paid.

        `coordination` deliberately does not pay for being right. Keeping
        convention formation and truth-finding as two *separable* observables is
        the point: if correctness is paid, any correlation between them in the
        resulting trajectory is an artefact of the payoff, and the MI/empowerment
        estimators downstream would be measuring the reward function.
        """

        if self.mode == "correctness":
            return self.match_reward if correct else self.mismatch_penalty
        base = self.match_reward if matched else self.mismatch_penalty
        if self.mode == "coordination_plus_correctness" and matched and correct:
            return base + self.correctness_bonus
        return base

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "match_reward": self.match_reward,
            "mismatch_penalty": self.mismatch_penalty,
            "correctness_bonus": self.correctness_bonus,
        }


@dataclass(frozen=True, slots=True)
class NamingRules(_CommonRules):
    """`hidden_bench_naming` options (§6)."""

    rounds: int = 20
    messages_per_turn: int = 1
    pairing: str = "uniform_two_distinct"
    memory_size: int = 0
    allow_relay: bool = True
    stop_on_consensus: bool = True
    consensus_threshold: float = 0.95
    payoff: PayoffRules = field(default_factory=PayoffRules)

    @classmethod
    def from_config(cls, config: GameConfig) -> "NamingRules":
        options = config.options
        rules = cls(
            **cls._common_kwargs(config),
            rounds=_integer(options, "rounds", config.horizon or 20, minimum=1),
            messages_per_turn=_integer(options, "messages_per_turn", 1, minimum=0),
            pairing=str(options.get("pairing", "uniform_two_distinct")),
            # 0 means unbounded: an agent remembers every partner it has met.
            memory_size=_integer(options, "memory_size", 0, minimum=0),
            allow_relay=_boolean(options, "allow_relay", True),
            stop_on_consensus=_boolean(options, "stop_on_consensus", True),
            consensus_threshold=_number(options, "consensus_threshold", 0.95),
            payoff=PayoffRules.from_options(options),
        )
        rules.validate_common()
        if rules.pairing != "uniform_two_distinct":
            # The hook is deliberately here rather than absent: network
            # structure is where this repo is headed, and a graph sampler drops
            # in at this one branch.
            raise HiddenBenchDataError(
                "game.options.pairing currently supports 'uniform_two_distinct' only; "
                "graph/topology sampling is the intended extension point"
            )
        if not 0.0 < rules.consensus_threshold <= 1.0:
            raise HiddenBenchDataError("game.options.consensus_threshold must be in (0, 1]")
        return rules

    @property
    def steps_per_round(self) -> int:
        """`m` exchange steps then one commitment step."""

        return self.messages_per_turn + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_agents": self.n_agents,
            "task_set": self.task_set,
            "task_id": self.task_id,
            "profile": self.profile,
            "assignment_scheme": self.assignment_scheme,
            "rounds": self.rounds,
            "messages_per_turn": self.messages_per_turn,
            "pairing": self.pairing,
            "memory_size": self.memory_size,
            "allow_relay": self.allow_relay,
            "stop_on_consensus": self.stop_on_consensus,
            "consensus_threshold": self.consensus_threshold,
            "payoff": self.payoff.to_dict(),
            "extra_prompt": self.extra_prompt,
            "aggregation": self.aggregation,
            "shuffle_facts": self.shuffle_facts,
            "decoy": self.decoy,
        }
