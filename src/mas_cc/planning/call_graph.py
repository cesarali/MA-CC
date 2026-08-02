"""Static logical-call specification shared by preflight and experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LogicalCallSpec:
    logical_calls: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.logical_calls, bool)
            or not isinstance(self.logical_calls, int)
            or self.logical_calls < 1
        ):
            raise ValueError("logical_calls must be a positive integer")
