"""Bounded, deterministic selection of expensive prompt audit records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class DetailedAuditPolicy:
    """A count-based policy; it deliberately does not retain prompt content."""

    enabled: bool = False
    log_every_n_rounds: int | None = None
    always_log_first_n_rounds: int = 0
    max_logged_prompts_per_game: int | None = None
    max_logged_prompts_per_run: int | None = None
    always_log_provider_errors: bool = True
    always_log_invalid_responses: bool = True

    def __post_init__(self) -> None:
        for name in (
            "log_every_n_rounds", "always_log_first_n_rounds",
            "max_logged_prompts_per_game", "max_logged_prompts_per_run",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.log_every_n_rounds == 0:
            raise ValueError("log_every_n_rounds must be positive when supplied")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DetailedAuditPolicy":
        values = dict(value or {})
        return cls(
            enabled=bool(values.get("enabled", False)),
            log_every_n_rounds=values.get("log_every_n_rounds"),
            always_log_first_n_rounds=int(values.get("always_log_first_n_rounds", 0)),
            max_logged_prompts_per_game=values.get("max_logged_prompts_per_game"),
            max_logged_prompts_per_run=values.get("max_logged_prompts_per_run"),
            always_log_provider_errors=bool(values.get("always_log_provider_errors", True)),
            always_log_invalid_responses=bool(values.get("always_log_invalid_responses", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "log_every_n_rounds": self.log_every_n_rounds,
            "always_log_first_n_rounds": self.always_log_first_n_rounds,
            "max_logged_prompts_per_game": self.max_logged_prompts_per_game,
            "max_logged_prompts_per_run": self.max_logged_prompts_per_run,
            "always_log_provider_errors": self.always_log_provider_errors,
            "always_log_invalid_responses": self.always_log_invalid_responses,
        }


@dataclass(frozen=True, slots=True)
class AuditSelection:
    selected: bool
    reason: str


class DetailedAuditSelector:
    """Mutable counters only; selection is deterministic for a run configuration."""

    def __init__(self, policy: DetailedAuditPolicy) -> None:
        self.policy = policy
        self._run_count = 0
        self._game_counts: dict[str, int] = {}
        self._omitted: dict[str, int] = {}

    def select(
        self, *, game_id: str, round_index: int, provider_error: bool = False,
        invalid_response: bool = False,
    ) -> AuditSelection:
        if not self.policy.enabled:
            return self._omit("detailed_audit_disabled")
        forced = (
            provider_error and self.policy.always_log_provider_errors
        ) or (invalid_response and self.policy.always_log_invalid_responses)
        scheduled = (
            round_index <= self.policy.always_log_first_n_rounds
            or (
                self.policy.log_every_n_rounds is not None
                and round_index % self.policy.log_every_n_rounds == 0
            )
        )
        if not forced and not scheduled:
            return self._omit("not_selected_by_round_policy")
        game_count = self._game_counts.get(game_id, 0)
        if (
            self.policy.max_logged_prompts_per_game is not None
            and game_count >= self.policy.max_logged_prompts_per_game
        ):
            return self._omit("max_logged_prompts_per_game_reached")
        if (
            self.policy.max_logged_prompts_per_run is not None
            and self._run_count >= self.policy.max_logged_prompts_per_run
        ):
            return self._omit("max_logged_prompts_per_run_reached")
        self._run_count += 1
        self._game_counts[game_id] = game_count + 1
        return AuditSelection(True, "provider_error" if provider_error else "invalid_response" if invalid_response else "round_policy")

    def _omit(self, reason: str) -> AuditSelection:
        self._omitted[reason] = self._omitted.get(reason, 0) + 1
        return AuditSelection(False, reason)

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": self.policy.to_dict(),
            "selected_prompt_records": self._run_count,
            "selected_prompt_records_per_game": dict(sorted(self._game_counts.items())),
            "omitted_prompt_records_by_reason": dict(sorted(self._omitted.items())),
        }
