"""Structured validation results shared by configs, prompts, and games."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from collections.abc import Mapping

from .exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One error tied to an exact dotted configuration or record field."""

    field: str
    message: str
    invalid_value: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field:
            raise ValueError("ValidationIssue.field must be non-empty")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("ValidationIssue.message must be non-empty")
        object.__setattr__(self, "invalid_value", _freeze(self.invalid_value))

    def __str__(self) -> str:
        return f"{self.field}: {self.message}"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """The immutable result of validating an object."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues

    @property
    def valid(self) -> bool:
        """Compatibility spelling useful in serialized reports."""

        return self.is_valid

    @classmethod
    def success(cls) -> "ValidationResult":
        return cls()

    @classmethod
    def failure(cls, *issues: ValidationIssue) -> "ValidationResult":
        return cls(tuple(issues))

    def raise_for_errors(self, *, context: str = "validation") -> None:
        if self.issues:
            raise ValidationError(self.issues, context=context)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value
