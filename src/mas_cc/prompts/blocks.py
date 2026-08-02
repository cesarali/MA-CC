"""Immutable semantic prompt blocks and their rendered records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Generic, Literal, Self, TypeVar, cast

from mas_cc.core import MessageRole, ValidationIssue, ValidationResult

from ._values import freeze, thaw

T = TypeVar("T")
BindingType = Literal["fixed", "dynamic"]


class _Unbound:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNBOUND"

    def __reduce__(self) -> str:
        return "UNBOUND"


UNBOUND = _Unbound()
"""A value was not supplied.  It is intentionally distinct from ``None``."""

Unbound = _Unbound


@dataclass(frozen=True, slots=True)
class PromptBlock(ABC, Generic[T]):
    """One immutable semantic value together with validation and rendering."""

    name: str
    title: str
    role: MessageRole
    value: T | Unbound = UNBOUND
    required: bool = True
    binding: BindingType = "dynamic"
    version: int = 1
    sensitive: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("PromptBlock.name must be a non-empty stable name")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("PromptBlock.title must be non-empty")
        if isinstance(self.role, str):
            object.__setattr__(self, "role", MessageRole(self.role))
        if not isinstance(self.role, MessageRole):
            raise TypeError("PromptBlock.role must be a MessageRole")
        if self.binding not in {"fixed", "dynamic"}:
            raise ValueError("PromptBlock.binding must be 'fixed' or 'dynamic'")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("PromptBlock.version must be a positive integer")
        if not isinstance(self.required, bool):
            raise TypeError("PromptBlock.required must be a boolean")
        if not isinstance(self.sensitive, bool):
            raise TypeError("PromptBlock.sensitive must be a boolean")
        if self.value is not UNBOUND:
            object.__setattr__(self, "value", freeze(self.value))

    @property
    def is_bound(self) -> bool:
        return self.value is not UNBOUND

    def bind(self, value: T) -> Self:
        """Return a separately bound block without mutating this instance."""

        if self.binding == "fixed" and self.is_bound:
            raise ValueError(f"prompt.blocks.{self.name}.value: fixed block cannot be rebound")
        return cast(Self, replace(self, value=freeze(value)))

    def value_issues(self, value: T) -> tuple[ValidationIssue, ...]:
        """Concrete blocks override this for field-specific value validation."""

        return ()

    def validate(self) -> ValidationResult:
        field = f"prompt.blocks.{self.name}.value"
        if self.value is UNBOUND:
            if self.required:
                return ValidationResult.failure(
                    ValidationIssue(field, "required value is UNBOUND")
                )
            return ValidationResult.success()
        return ValidationResult(tuple(self.value_issues(cast(T, self.value))))

    @abstractmethod
    def render(self) -> str:
        """Render the bound value deterministically without side effects."""

    def definition_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "title": self.title,
            "role": self.role.value,
            "version": self.version,
            "required": self.required,
            "binding": self.binding,
            "sensitive": self.sensitive,
            "implementation": f"{type(self).__module__}.{type(self).__qualname__}",
        }
        if self.binding == "fixed":
            result["fixed_value"] = (
                None
                if self.value is UNBOUND
                else "<redacted>"
                if self.sensitive
                else thaw(self.value)
            )
        return result

    def definition_fingerprint_dict(self) -> dict[str, Any]:
        result = self.definition_dict()
        if self.binding == "fixed":
            result["fixed_value"] = None if self.value is UNBOUND else thaw(self.value)
        return result

    def to_dict(self, *, include_value: bool = True) -> dict[str, Any]:
        result = self.definition_dict()
        result["is_bound"] = self.is_bound
        if include_value:
            if self.value is UNBOUND:
                result["value_state"] = "unbound"
            elif self.sensitive:
                result.update({"value_state": "bound", "value": "<redacted>"})
            else:
                result.update({"value_state": "bound", "value": thaw(self.value)})
        return result

    def fingerprint_dict(self) -> dict[str, Any]:
        """Canonical local binding payload; callers must not publish it."""

        return {
            "name": self.name,
            "version": self.version,
            "value_state": "unbound" if self.value is UNBOUND else "bound",
            "value": None if self.value is UNBOUND else thaw(self.value),
        }


@dataclass(frozen=True, slots=True)
class RenderedPromptBlock:
    """Inspectable rendered output of one independently meaningful block."""

    name: str
    title: str
    role: MessageRole
    content: str
    version: int
    order: int
    token_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "role": self.role.value,
            "version": self.version,
            "order": self.order,
            "content": self.content,
            "token_count": self.token_count,
        }
