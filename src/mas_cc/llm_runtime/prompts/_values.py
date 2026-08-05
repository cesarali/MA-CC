"""Private helpers for immutable, canonical prompt values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any


def freeze(value: Any) -> Any:
    """Recursively detach and freeze a user supplied prompt value."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(item) for item in value)
    return value


def thaw(value: Any) -> Any:
    """Return a JSON-safe mutable representation without exposing shared state."""

    if is_dataclass(value) and not isinstance(value, type):
        return thaw(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((thaw(item) for item in value), key=repr)
    return value
