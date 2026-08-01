"""Strongly typed identifiers used in persisted records."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True, order=True, slots=True)
class Identifier:
    """A compact, filesystem- and JSON-friendly identifier."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not _ID_PATTERN.fullmatch(self.value):
            raise ValueError(
                f"{type(self).__name__}.value must match {_ID_PATTERN.pattern!r}"
            )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True, slots=True)
class RunId(Identifier):
    """Identifier for one resolved run."""


@dataclass(frozen=True, order=True, slots=True)
class ExperimentId(Identifier):
    """Identifier for an experiment definition."""


@dataclass(frozen=True, order=True, slots=True)
class AgentId(Identifier):
    """Identifier for an agent within a game."""


@dataclass(frozen=True, order=True, slots=True)
class InteractionId(Identifier):
    """Identifier for one interaction."""


@dataclass(frozen=True, order=True, slots=True)
class MessageId(Identifier):
    """Identifier for one message."""
