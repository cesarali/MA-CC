"""Configuration and persisted records for round-level HiddenBench feedback."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

from mas_cc.config import GameConfig
from mas_cc.games.protocols import _thaw

from ..imitation.state import ImitationRules

GAME_TYPE = "hidden_bench_imitation_round_feedback"
CLASSICAL_KERNEL = "controlled_imitation_round_reference"
CLASSICAL_KERNEL_RULE = "strict_unanimity_q_voter"
ROUND_RECORD_TYPE = "imitation_round_feedback"


@dataclass(frozen=True, slots=True)
class RoundFeedbackRules(ImitationRules):
    """The inherited imitation rules plus an explicit slow-clock horizon."""

    rounds: int = 1

    @classmethod
    def from_config(cls, config: GameConfig) -> "RoundFeedbackRules":
        if config.type != GAME_TYPE:
            raise ValueError(f"RoundFeedbackRules requires game.type {GAME_TYPE}")
        raw_rounds = config.options.get("rounds", config.horizon)
        if isinstance(raw_rounds, bool) or not isinstance(raw_rounds, int) or raw_rounds < 1:
            raise ValueError("game.options.rounds must be a positive integer")

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
        values = {field.name: getattr(base, field.name) for field in fields(ImitationRules)}
        values.update(
            horizon=raw_rounds * config.population_size,
            classical_kernel=kernel,
        )
        return cls(**values, rounds=raw_rounds)


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
    "ROUND_RECORD_TYPE",
    "RoundFeedbackRecord",
    "RoundFeedbackRules",
]
