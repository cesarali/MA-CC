"""Typed data structures shared by the game engines and benchmark runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

Name = Literal["A", "B"]
Inventory = frozenset[Name]


class ConfigurationError(ValueError):
    """Raised when a requested experiment is not completely configured."""


class UpdateMode(str, Enum):
    SEQUENTIAL = "sequential"
    SYNCHRONOUS_PARALLEL = "synchronous_parallel"


VALID_INVENTORIES: frozenset[Inventory] = frozenset(
    {frozenset({"A"}), frozenset({"B"}), frozenset({"A", "B"})}
)


def normalize_inventory(value: Any) -> Inventory:
    """Return one of the three valid binary Naming Game inventories."""

    if isinstance(value, str):
        compact = value.strip().replace(" ", "")
        aliases = {
            "A": frozenset({"A"}),
            "B": frozenset({"B"}),
            "{A}": frozenset({"A"}),
            "{B}": frozenset({"B"}),
            "A,B": frozenset({"A", "B"}),
            "{A,B}": frozenset({"A", "B"}),
        }
        inventory = aliases.get(compact)
        if inventory is None:
            raise ValueError(f"Invalid inventory: {value!r}")
    else:
        try:
            inventory = frozenset(value)
        except TypeError as exc:
            raise ValueError(f"Invalid inventory: {value!r}") from exc

    if inventory not in VALID_INVENTORIES:
        raise ValueError(
            "Inventory must be exactly one of {A}, {B}, or {A, B}; "
            f"received {value!r}."
        )
    return inventory  # type: ignore[return-value]


def inventory_values(inventory: Inventory) -> list[Name]:
    return [name for name in ("A", "B") if name in inventory]


def inventory_label(inventory: Inventory) -> str:
    return "{" + ", ".join(inventory_values(inventory)) + "}"


@dataclass(frozen=True)
class AgentSnapshot:
    """The immutable, pair-local view of an agent."""

    agent_id: int
    inventory: Inventory
    evidence: str | None = None


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TokenUsage":
        value = value or {}

        def optional_int(key: str) -> int | None:
            item = value.get(key)
            return item if isinstance(item, int) and not isinstance(item, bool) else None

        return cls(
            prompt_tokens=optional_int("prompt_tokens"),
            completion_tokens=optional_int("completion_tokens"),
            total_tokens=optional_int("total_tokens"),
        )


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    latency_seconds: float
    retries: int = 0
    status_code: int | None = 200
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw_response: Any = None
    finish_reason: str | None = None

    @property
    def attempts(self) -> int:
        return self.retries + 1


@dataclass(frozen=True)
class InteractionResult:
    interaction_index: int
    round_index: int | None
    pair_index: int | None
    interaction_kind: Literal["basic", "reasoning"]
    speaker_id: int
    listener_id: int
    speaker_before: Inventory
    listener_before: Inventory
    selected_name: Name
    listener_reported_known: bool | None
    engine_already_known: bool | None
    naming_success: bool | None
    speaker_after: Inventory
    listener_after: Inventory
    speaker_response: LLMResponse
    listener_response: LLMResponse
    speaker_response_valid: bool
    listener_response_valid: bool
    speaker_validation_error: str | None
    listener_validation_error: str | None
    reason: str | None
    pair_wall_seconds: float

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "interaction_index": self.interaction_index,
            "round_index": self.round_index,
            "pair_index": self.pair_index,
            "interaction_kind": self.interaction_kind,
            "speaker_id": self.speaker_id,
            "listener_id": self.listener_id,
            "speaker_before": inventory_label(self.speaker_before),
            "listener_before": inventory_label(self.listener_before),
            "selected_name": self.selected_name,
            "listener_reported_known": self.listener_reported_known,
            "engine_already_known": self.engine_already_known,
            "naming_success": self.naming_success,
            "speaker_after": inventory_label(self.speaker_after),
            "listener_after": inventory_label(self.listener_after),
            "speaker_response": self.speaker_response.content,
            "listener_response": self.listener_response.content,
            "speaker_response_valid": self.speaker_response_valid,
            "listener_response_valid": self.listener_response_valid,
            "speaker_validation_error": self.speaker_validation_error,
            "listener_validation_error": self.listener_validation_error,
            "reason": self.reason,
            "speaker_latency_seconds": self.speaker_response.latency_seconds,
            "listener_latency_seconds": self.listener_response.latency_seconds,
            "pair_wall_seconds": self.pair_wall_seconds,
            "speaker_retries": self.speaker_response.retries,
            "listener_retries": self.listener_response.retries,
            "speaker_status": self.speaker_response.status_code,
            "listener_status": self.listener_response.status_code,
            "speaker_token_usage": asdict(self.speaker_response.usage),
            "listener_token_usage": asdict(self.listener_response.usage),
        }


@dataclass(frozen=True)
class StateRecord:
    interaction_index: int
    count_a: int
    count_b: int
    count_ab: int
    consensus: bool


@dataclass(frozen=True)
class RoundRecord:
    round_index: int
    interactions_completed: int
    parallel_pairs: int
    idle_agent_id: int | None
    round_wall_seconds: float
    slowest_pair_seconds: float
    count_a: int
    count_b: int
    count_ab: int
    consensus: bool


@dataclass
class GameResult:
    interactions: list[InteractionResult]
    states: list[StateRecord]
    rounds: list[RoundRecord]
    initial_counts: dict[str, int]
    final_counts: dict[str, int]
    wall_seconds: float
    consensus_reached: bool
    consensus_interaction_index: int | None
    trajectory_concurrency: int = 1


@dataclass(frozen=True)
class RunSpec:
    model: str
    num_agents: int
    reasoning_fraction: float
    update_mode: UpdateMode
    synchronous_round_equivalent: float
    num_interactions: int
    rounds: int | None
    seed: int
    concurrency: int
    replicate: int = 0
    temperature: float = 0.0
    max_tokens_speaker: int = 20
    max_tokens_listener: int = 20

    def __post_init__(self) -> None:
        if self.num_agents < 2:
            raise ConfigurationError("num_agents must be at least 2.")
        if not 0.0 <= self.reasoning_fraction <= 1.0:
            raise ConfigurationError("reasoning_fraction must be between 0 and 1.")
        if self.num_interactions < 0:
            raise ConfigurationError("num_interactions cannot be negative.")
        if self.concurrency < 1:
            raise ConfigurationError("concurrency must be at least 1.")


@dataclass
class BenchmarkRunSummary:
    run_id: str
    status: str
    error: str | None
    api_backend: str
    model: str
    num_agents: int
    reasoning_fraction: float
    update_mode: str
    synchronous_round_equivalent: float
    total_pair_interactions: int
    expected_api_calls: int
    actual_api_calls: int
    random_seed: int
    concurrency_limit: int
    replicate: int
    total_wall_seconds: float
    seconds_per_pair_interaction: float | None
    seconds_per_synchronous_round_equivalent: float | None
    seconds_per_actual_synchronous_round: float | None
    successful_calls: int
    failed_calls: int
    retries: int
    mean_request_latency_seconds: float | None
    median_request_latency_seconds: float | None
    p90_request_latency_seconds: float | None
    max_request_latency_seconds: float | None
    api_calls_per_second: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    successful_naming_interactions: int
    failed_naming_interactions: int
    initial_count_a: int
    initial_count_b: int
    initial_count_ab: int
    final_count_a: int
    final_count_b: int
    final_count_ab: int
    consensus_reached: bool
    consensus_interaction_index: int | None
    parallel_pairs_per_round: str | None
    mean_round_wall_seconds: float | None
    mean_slowest_pair_seconds: float | None
    total_trajectory_seconds: float | None
    independent_trajectories_concurrent: bool
    concurrent_trajectories: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def population_counts(inventories: Sequence[Inventory]) -> dict[str, int]:
    counts = {"A": 0, "B": 0, "AB": 0}
    for inventory in inventories:
        normalized = normalize_inventory(inventory)
        if normalized == frozenset({"A"}):
            counts["A"] += 1
        elif normalized == frozenset({"B"}):
            counts["B"] += 1
        else:
            counts["AB"] += 1
    return counts


def has_consensus(inventories: Sequence[Inventory]) -> bool:
    return bool(inventories) and (
        all(inventory == frozenset({"A"}) for inventory in inventories)
        or all(inventory == frozenset({"B"}) for inventory in inventories)
    )
