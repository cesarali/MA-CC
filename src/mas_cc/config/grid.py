"""Cartesian-product parameter sweeps over one base :class:`RunConfig`.

A grid varies *design* fields (game, prompt, execution) across cells that all
share one provider client, one pricing quote, and one budget guard — see the
forbidden-prefix check in :func:`parse_grid_axes`. Sweeping the provider
identity or the budget/pricing policy itself is deliberately not supported:
that would require a different provider client (and a different pricing
quote) per cell, which is a materially different, unbuilt feature.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from typing import Any

from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.llm_runtime.validation import ValidationIssue

from .models import RunConfig

_FORBIDDEN_EXACT = ("llm_provider.type", "llm_provider.model", "game.type", "budget", "pricing")
_FORBIDDEN_PREFIXES = ("budget.", "pricing.")


def _apply_path_override(config: RunConfig, path: str, value: Any) -> RunConfig:
    return _replace_nested(config, path.split("."), value, path)


def _replace_nested(obj: Any, parts: list[str], value: Any, full_path: str) -> Any:
    head, *rest = parts
    if dataclasses.is_dataclass(obj):
        if head not in {field.name for field in dataclasses.fields(obj)}:
            raise ConfigurationError(
                [ValidationIssue(f"grid.{full_path}", f"{type(obj).__name__} has no field {head!r}")],
                context="grid expansion",
            )
        current = getattr(obj, head)
        new_value = value if not rest else _replace_nested(current, rest, value, full_path)
        return dataclasses.replace(obj, **{head: new_value})
    if isinstance(obj, Mapping):
        current = obj.get(head)
        new_value = value if not rest else _replace_nested(current, rest, value, full_path)
        updated = dict(obj)
        updated[head] = new_value
        return updated
    raise ConfigurationError(
        [
            ValidationIssue(
                f"grid.{full_path}",
                f"cannot override {'.'.join(parts)!r}: {type(obj).__name__} is neither a "
                "config section nor a mapping",
            )
        ],
        context="grid expansion",
    )


@dataclasses.dataclass(frozen=True, slots=True)
class GridAxis:
    """One swept field: a dotted path into ``RunConfig`` and its candidate values."""

    path: str
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ConfigurationError(
                [ValidationIssue("grid", "axis path must be non-empty")], context="grid parsing"
            )
        if not self.values:
            raise ConfigurationError(
                [ValidationIssue(f"grid.{self.path}", "must list at least one value")],
                context="grid parsing",
            )
        if self.path in _FORBIDDEN_EXACT or self.path.startswith(_FORBIDDEN_PREFIXES):
            raise ConfigurationError(
                [
                    ValidationIssue(
                        f"grid.{self.path}",
                        "cannot sweep the game type, provider identity, or budget/pricing "
                        "policy; these are shared across every grid cell (one Game instance, "
                        "one provider client, one pricing quote, one budget guard) — use a "
                        "separate config/run for a different game, provider, model, or budget",
                    )
                ],
                context="grid parsing",
            )

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "values": list(self.values)}


@dataclasses.dataclass(frozen=True, slots=True)
class GridCell:
    """One resolved config produced by binding every axis to one of its values."""

    index: int
    cell_id: str
    overrides: Mapping[str, Any]
    config: RunConfig

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "cell_id": self.cell_id, "overrides": dict(self.overrides)}


@dataclasses.dataclass(frozen=True, slots=True)
class GridSpec:
    """A base config plus axes; ``cells`` is the cartesian product, in stable order."""

    base: RunConfig
    axes: tuple[GridAxis, ...]

    def __post_init__(self) -> None:
        if not self.axes:
            raise ConfigurationError(
                [ValidationIssue("grid", "must declare at least one axis")], context="grid parsing"
            )
        if len({axis.path for axis in self.axes}) != len(self.axes):
            raise ConfigurationError(
                [ValidationIssue("grid", "axis paths must be unique")], context="grid parsing"
            )
        # Fail fast on a bad path/type once, rather than once per generated cell.
        for axis in self.axes:
            _apply_path_override(self.base, axis.path, axis.values[0])

    @property
    def cells(self) -> tuple[GridCell, ...]:
        paths = [axis.path for axis in self.axes]
        value_lists = [axis.values for axis in self.axes]
        cells: list[GridCell] = []
        for index, combo in enumerate(itertools.product(*value_lists)):
            overrides = dict(zip(paths, combo))
            config = self.base
            for path, value in overrides.items():
                config = _apply_path_override(config, path, value)
            cells.append(GridCell(index, f"cell-{index:04d}", overrides, config))
        return tuple(cells)

    @property
    def grid_id(self) -> str:
        payload = {axis.path: [_jsonable(value) for value in axis.values] for axis in self.axes}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_id": self.grid_id,
            "axes": [axis.to_dict() for axis in self.axes],
            "cell_count": len(self.cells),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_grid_axes(raw: Any) -> tuple[GridAxis, ...]:
    if not isinstance(raw, Mapping) or not raw:
        raise ConfigurationError(
            [ValidationIssue("grid", "must be a non-empty mapping of dotted-path -> list of values")],
            context="grid parsing",
        )
    axes = []
    for path, values in raw.items():
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ConfigurationError(
                [ValidationIssue(f"grid.{path}", "must be a list of values")], context="grid parsing"
            )
        axes.append(GridAxis(str(path), tuple(values)))
    return tuple(axes)
