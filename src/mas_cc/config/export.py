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
    if (
        config.game.type == "relational_imitation_round_feedback"
        and config.game.options.get("social_mode") == "board"
    ):
        game_options = dict(values["game"]["options"])
        board = dict(game_options.get("board", {}))
        version = int(game_options.get("prompt_version", config.prompt.prompt_version))
        board.setdefault(
            "report_citation_scope",
            "active_or_observed" if version >= 5 else "active_only",
        )
        board.setdefault(
            "no_citable_fact_action",
            "none" if version >= 5 else "model_select",
        )
        game_options["board"] = board
        values["game"] = {**values["game"], "options": game_options}
    if config.prompt.schema_version == 2:
        from mas_cc.games.registry import create_default_prompt_registry

        registry = create_default_prompt_registry()
        if (
            config.prompt.prompt_family == "relational_blackboard_ballot"
            and config.prompt.prompt_version >= 4
        ):
            from mas_cc.games.relational_reasoning.imitation_round_feedback.prompts import (
                relational_blackboard_ballot_prompt,
            )

            board = config.game.options.get("board", {})
            communication_profile = board.get("communication_profile")
            prompt = relational_blackboard_ballot_prompt(
                version=config.prompt.prompt_version,
                allow_participant_requests=bool(
                    communication_profile == "full_communication"
                    if communication_profile is not None
                    else board.get("allow_participant_requests", True)
                ),
                require_grounded_reports=bool(
                    board.get(
                        "require_grounded_reports",
                        config.prompt.prompt_version >= 5,
                    )
                ),
                report_citation_scope=str(
                    board.get(
                        "report_citation_scope",
                        "active_or_observed"
                        if config.prompt.prompt_version >= 5
                        else "active_only",
                    )
                ),
            )
        else:
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
