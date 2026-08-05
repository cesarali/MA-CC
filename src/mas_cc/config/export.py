"""Deterministic, secret-safe serialization of resolved configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.llm_runtime.validation import ValidationIssue

from .loader import _validate_secret_fields
from .models import RunConfig


def resolved_config_yaml(config: RunConfig) -> str:
    """Serialize a fully resolved config after a second secret-field audit."""

    values = config.to_dict()
    if config.prompt.schema_version == 2:
        from mas_cc.games.registry import create_default_prompt_registry

        registry = create_default_prompt_registry()
        try:
            prompt = registry.get(
                config.prompt.prompt_family, config.prompt.prompt_version
            )
        except ValueError:
            from mas_cc.games.registry import register_game_prompt_factories

            prompt = register_game_prompt_factories(registry).get(
                config.prompt.prompt_family, config.prompt.prompt_version
            )
        values["prompt"]["resolved_block_manifest"] = [
            {
                "order": index,
                "name": block.name,
                "version": block.version,
                "role": block.role.value,
                "required": block.required,
                "binding": block.binding,
                "sensitive": block.sensitive,
            }
            for index, block in enumerate(prompt.blocks, start=1)
        ]
        values["prompt"]["definition_hash"] = prompt.definition_hash
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
