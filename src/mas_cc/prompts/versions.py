"""Stable prompt-family/version identities."""

from __future__ import annotations

import re
from dataclasses import dataclass

_FAMILY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True, order=True, slots=True)
class PromptVersion:
    family: str
    version: int

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not _FAMILY.fullmatch(self.family):
            raise ValueError("PromptVersion.family must be a lowercase snake_case name")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("PromptVersion.version must be a positive integer")

    def __str__(self) -> str:
        return f"{self.family}@{self.version}"
