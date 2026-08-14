"""Configuration and persisted records for round-level HiddenBench feedback."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

from mas_cc.config import GameConfig
from mas_cc.games.protocols import AgentState, _thaw

from ..imitation.state import ImitationRules
from .prompts import (
    IMPLEMENTED_VOTE_VISIBILITIES,
    PROMPT_FAMILY,
    PROMPT_VERSION,
    VOTE_VISIBILITIES,
)

GAME_TYPE = "hidden_bench_imitation_round_feedback"
CLASSICAL_KERNEL = "controlled_imitation_round_reference"
CLASSICAL_KERNEL_RULE = "strict_unanimity_q_voter"
ROUND_RECORD_TYPE = "imitation_round_feedback"

PUBLIC_REASON = "public_reason"
"""The agent attribute holding `R_i` - the latest reason publicly given for the
agent's current `committed_action`.  `(committed_action, public_reason)` is the
whole socially visible state; the inherited HiddenBench evidence stays private
unless the agent itself puts it into a reason."""


def get_public_reason(agent: AgentState) -> str | None:
    """`R_i`, or `None` before the agent has ever published a reason."""

    value = agent.attributes.get(PUBLIC_REASON)
    return None if value is None else str(value)


def set_public_reason(attributes: Mapping[str, Any], reason: str | None) -> dict[str, Any]:
    """Return `attributes` with `R_i` replaced.  Callers own the agent rebuild."""

    return {**dict(attributes), PUBLIC_REASON: reason}


@dataclass(frozen=True, slots=True)
class RoundFeedbackRules(ImitationRules):
    """The inherited imitation rules plus an explicit slow-clock horizon.

    `messages_per_agent` is inherited and **ignored**: reasoning mode makes one
    provider call per focal update and generates no peer dialogue at all, so
    there is nothing for it to count.  It stays in the schema only so the
    parent game's validation and the shipped configs keep loading.
    """

    rounds: int = 1
    vote_visibility: str = "public"

    @classmethod
    def from_config(cls, config: GameConfig) -> "RoundFeedbackRules":
        if config.type != GAME_TYPE:
            raise ValueError(f"RoundFeedbackRules requires game.type {GAME_TYPE}")
        raw_rounds = config.options.get("rounds", config.horizon)
        if isinstance(raw_rounds, bool) or not isinstance(raw_rounds, int) or raw_rounds < 1:
            raise ValueError("game.options.rounds must be a positive integer")

        visibility = str(config.options.get("vote_visibility", "public"))
        if visibility not in VOTE_VISIBILITIES:
            raise ValueError(
                f"game.options.vote_visibility must be one of {list(VOTE_VISIBILITIES)}"
            )
        if visibility not in IMPLEMENTED_VOTE_VISIBILITIES:
            raise ValueError(
                f"game.options.vote_visibility {visibility!r} is reserved; only "
                f"{list(IMPLEMENTED_VOTE_VISIBILITIES)} is implemented"
            )

        classical = config.options.get("classical", {})
        if classical is None:
            classical = {}
        if not isinstance(classical, Mapping):
            raise ValueError("game.options.classical must be a mapping")
        kernel = str(classical.get("kernel", CLASSICAL_KERNEL))
        if kernel != CLASSICAL_KERNEL:
            raise ValueError(f"only classical.kernel {CLASSICAL_KERNEL!r} is implemented")

        # Reuse every existing validation rule and default without weakening the
        # old game's type guard.  The old classical kernel name is supplied only
        # to that validator; this game's returned rule records its own kernel.
        compatible_options = dict(config.options)
        compatible_options["classical"] = {
            **dict(classical),
            "kernel": "irisarri_multi_opinion",
        }
        compatible = replace(
            config,
            type="hidden_bench_imitation",
            horizon=raw_rounds,
            options=compatible_options,
        )
        base = ImitationRules.from_config(compatible)
        if base.dynamics_mode == "reasoning" and base.prompt_version != PROMPT_VERSION:
            # The parent game's v1/v2 dialogue text is not reachable from here:
            # reasoning updates go through the single-version public-ballot
            # family, so a config asking for v2 would silently get v1 text.
            raise ValueError(
                f"the round-feedback reasoning game uses the {PROMPT_FAMILY!r} "
                f"prompt family, which has one version; "
                f"game.options.prompt_version must be {PROMPT_VERSION}"
            )
        values = {field.name: getattr(base, field.name) for field in fields(ImitationRules)}
        values.update(
            horizon=raw_rounds * config.population_size,
            classical_kernel=kernel,
        )
        return cls(**values, rounds=raw_rounds, vote_visibility=visibility)


@dataclass(frozen=True, slots=True)
class RoundFeedbackRecord:
    """One complete slow-clock transition, persisted separately from micro rows."""

    round_index: int
    event: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": ROUND_RECORD_TYPE,
            "round_index": self.round_index,
            **_thaw(self.event),
        }


__all__ = [
    "CLASSICAL_KERNEL",
    "CLASSICAL_KERNEL_RULE",
    "GAME_TYPE",
    "PUBLIC_REASON",
    "ROUND_RECORD_TYPE",
    "RoundFeedbackRecord",
    "RoundFeedbackRules",
    "get_public_reason",
    "set_public_reason",
]
