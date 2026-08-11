"""Configuration and trajectory records for ``hidden_bench_imitation``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mas_cc.config import GameConfig
from mas_cc.games.protocols import _thaw

from ..records import HiddenBenchGameState, HiddenBenchTransition
from .prompts import PROMPT_VERSIONS, SCENARIO_VARIANTS, PromptStyle


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


@dataclass(frozen=True, slots=True)
class ImitationRules:
    n_agents: int
    horizon: int
    dynamics_mode: str
    task_set: str
    task_id: str | None
    profile: str
    assignment_scheme: str
    pairing: str
    messages_per_agent: int
    memory_size: int
    allow_relay: bool
    prompt_version: int
    inform_asymmetry: bool
    scenario_variant: int
    stop_on_consensus: bool
    initialization_mode: str
    initial_votes: tuple[str, ...] | None
    initial_distribution: Mapping[str, float] | None
    classical_kernel: str
    forward_rate: float
    reverse_rate: float
    interaction_factor: str
    interaction_offset: float
    control_strength: float
    shuffle_facts: bool
    invalid_response_retries: int
    expected_validation_failure_rate: float
    corpus_root: str | None

    @classmethod
    def from_config(cls, config: GameConfig) -> "ImitationRules":
        if config.type != "hidden_bench_imitation":
            raise ValueError("ImitationRules requires game.type hidden_bench_imitation")
        options = config.options
        n_agents = int(options.get("n_agents", config.population_size))
        if n_agents != config.population_size:
            raise ValueError("game.options.n_agents must equal game.population_size")
        if "interactions" in options:
            raise ValueError(
                "game.options.interactions is no longer supported; use game.horizon"
            )
        messages = options.get("messages_per_agent", options.get("messages_per_turn", 1))
        memory_size = options.get("memory_size", 0)
        retries = options.get("invalid_response_retries", 0)
        for field, value, minimum in (
            ("messages_per_agent", messages, 0),
            ("memory_size", memory_size, 0),
            ("invalid_response_retries", retries, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"game.options.{field} must be an integer >= {minimum}")
        mode = str(options.get("dynamics_mode", "reasoning"))
        if mode not in {"reasoning", "classical"}:
            raise ValueError("dynamics_mode must be 'reasoning' or 'classical'")
        pairing = str(options.get("pairing", "uniform_two_distinct"))
        if pairing != "uniform_two_distinct":
            raise ValueError("only pairing 'uniform_two_distinct' is implemented")

        # Prompt text is a study condition, not a detail: v2 pushes directly on
        # how much information an agent volunteers, so it defaults to off and is
        # recorded in `rules` on every episode.  See `prompts.py`.
        prompt_version = options.get("prompt_version", 1)
        if isinstance(prompt_version, bool) or prompt_version not in PROMPT_VERSIONS:
            raise ValueError(f"game.options.prompt_version must be one of {list(PROMPT_VERSIONS)}")
        inform_asymmetry = bool(options.get("inform_asymmetry", False))
        if inform_asymmetry and prompt_version == 1:
            raise ValueError("game.options.inform_asymmetry requires prompt_version 2")
        scenario_variant = options.get("scenario_variant", 1)
        if isinstance(scenario_variant, bool) or scenario_variant not in SCENARIO_VARIANTS:
            raise ValueError(
                f"game.options.scenario_variant must be one of {list(SCENARIO_VARIANTS)}"
            )

        initialization = _mapping(options.get("initialization"), "game.options.initialization")
        initial_votes_raw = initialization.get("initial_votes")
        initial_votes: tuple[str, ...] | None = None
        if initial_votes_raw is not None:
            if isinstance(initial_votes_raw, (str, bytes)) or not isinstance(
                initial_votes_raw, Sequence
            ):
                raise ValueError("initialization.initial_votes must be a list")
            initial_votes = tuple(str(item) for item in initial_votes_raw)
            if len(initial_votes) != n_agents:
                raise ValueError("initialization.initial_votes must contain one vote per agent")
        distribution_raw = initialization.get("initial_distribution")
        distribution: Mapping[str, float] | None = None
        if distribution_raw is not None:
            distribution_map = _mapping(
                distribution_raw, "game.options.initialization.initial_distribution"
            )
            distribution = {str(key): float(value) for key, value in distribution_map.items()}
            if any(value < 0 for value in distribution.values()) or sum(distribution.values()) <= 0:
                raise ValueError("initial_distribution weights must be non-negative and nonzero")

        classical = _mapping(options.get("classical"), "game.options.classical")
        kernel = str(classical.get("kernel", "irisarri_multi_opinion"))
        if kernel != "irisarri_multi_opinion":
            raise ValueError("only classical.kernel 'irisarri_multi_opinion' is implemented")
        factor = str(classical.get("interaction_factor", "destination_count_plus_offset"))
        if factor not in {"constant", "destination_count_plus_offset"}:
            raise ValueError(
                "classical.interaction_factor must be 'constant' or "
                "'destination_count_plus_offset'"
            )
        h = float(classical.get("forward_rate", classical.get("h", 1.0)))
        a = float(classical.get("reverse_rate", classical.get("a", 1.0)))
        offset = float(classical.get("interaction_offset", 1.0))
        strength = float(classical.get("control_strength", 2.0))
        if h <= 0 or a <= 0:
            raise ValueError("classical forward_rate and reverse_rate must be positive")
        if offset <= 0:
            raise ValueError("classical interaction_offset must be positive")
        if strength < 0:
            raise ValueError("classical control_strength cannot be negative")
        expected_failure = float(options.get("expected_validation_failure_rate", 0.0))
        if not 0 <= expected_failure <= 1:
            raise ValueError("expected_validation_failure_rate must be between 0 and 1")

        return cls(
            n_agents=n_agents,
            horizon=config.horizon,
            dynamics_mode=mode,
            task_set=str(options.get("task_set", "vanilla")),
            task_id=(None if options.get("task_id") is None else str(options["task_id"])),
            profile=str(options.get("profile", "hidden")),
            assignment_scheme=str(options.get("assignment_scheme", "bijective")),
            pairing=pairing,
            messages_per_agent=int(messages),
            memory_size=int(memory_size),
            allow_relay=bool(options.get("allow_relay", True)),
            prompt_version=int(prompt_version),
            inform_asymmetry=inform_asymmetry,
            scenario_variant=int(scenario_variant),
            stop_on_consensus=bool(options.get("stop_on_consensus", False)),
            initialization_mode=str(initialization.get("mode", "local_vote")),
            initial_votes=initial_votes,
            initial_distribution=distribution,
            classical_kernel=kernel,
            forward_rate=h,
            reverse_rate=a,
            interaction_factor=factor,
            interaction_offset=offset,
            control_strength=strength,
            shuffle_facts=bool(options.get("shuffle_facts", True)),
            invalid_response_retries=int(retries),
            expected_validation_failure_rate=expected_failure,
            corpus_root=(None if options.get("corpus_root") is None else str(options["corpus_root"])),
        )

    @property
    def prompt_style(self) -> PromptStyle:
        return PromptStyle(version=self.prompt_version, inform_asymmetry=self.inform_asymmetry)

    def to_dict(self) -> dict[str, Any]:
        return {
            field: _thaw(getattr(self, field))
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class ImitationGameState(HiddenBenchGameState):
    @property
    def dynamics_mode(self) -> str:
        return str(self.data["dynamics_mode"])

    @property
    def initial_votes(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.data.get("initial_votes", ()))

    def to_dict(self) -> dict[str, Any]:
        value = HiddenBenchGameState.to_dict(self)
        value.update(
            {
                "seed": int(self.data["seed"]),
                "dynamics_mode": self.dynamics_mode,
                "initial_votes": list(self.initial_votes),
                "physical_time": float(self.data.get("physical_time", 0.0)),
                "event_history": _thaw(self.data.get("event_history", ())),
                "rules": _thaw(self.data.get("rules", {})),
            }
        )
        return value


@dataclass(frozen=True, slots=True)
class ImitationTransition(HiddenBenchTransition):
    event: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = HiddenBenchTransition.to_dict(self)
        value["event"] = None if self.event is None else _thaw(self.event)
        return value
