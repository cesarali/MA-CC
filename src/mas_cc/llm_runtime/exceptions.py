"""Base exception hierarchy used across llm_runtime."""

from __future__ import annotations

from collections.abc import Iterable


class MasCCError(Exception):
    """Base class for expected llm_runtime failures."""


class ValidationError(MasCCError, ValueError):
    """Raised when one or more named fields fail validation."""

    def __init__(self, issues: Iterable[object], *, context: str = "validation") -> None:
        self.issues = tuple(issues)
        detail = "; ".join(str(issue) for issue in self.issues)
        super().__init__(f"{context} failed: {detail}" if detail else f"{context} failed")


class ConfigurationError(ValidationError):
    """Raised when a configuration cannot be loaded or validated."""
