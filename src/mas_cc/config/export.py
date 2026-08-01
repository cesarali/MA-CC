"""Deterministic, secret-safe serialization of resolved configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from mas_cc.core.exceptions import ConfigurationError
from mas_cc.core.validation import ValidationIssue

from .loader import _validate_secret_fields
from .models import RunConfig


def resolved_config_yaml(config: RunConfig) -> str:
    """Serialize a fully resolved config after a second secret-field audit."""

    values = config.to_dict()
    issues: list[ValidationIssue] = []
    _validate_secret_fields(values, path="", issues=issues)
    if issues:
        raise ConfigurationError(issues, context="resolved configuration export")
    return yaml.safe_dump(values, sort_keys=False, allow_unicode=True)


def write_resolved_config(config: RunConfig, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(resolved_config_yaml(config), encoding="utf-8")
    return destination


def assert_secret_free(values: Mapping[str, Any]) -> None:
    """Raise if a serialization mapping contains a secret-bearing field."""

    issues: list[ValidationIssue] = []
    _validate_secret_fields(values, path="", issues=issues)
    if issues:
        raise ConfigurationError(issues, context="secret audit")
