"""Reusable, validated initial conditions for matched relational studies.

A paired initialization artifact stores the realized local-vote actions once.
Dynamics cells replay those actions instead of asking the provider again.  The
compatibility identity deliberately excludes controller and persistence fields:
neither is present in a local-vote prompt, and both start acting only after the
initial condition has been established.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from mas_cc.config import RunConfig
from mas_cc.core import AgentId
from mas_cc.games.protocols import Action, _thaw
from mas_cc.storage import canonical_hash, prompt_definition_hash

from .state import INITIAL_VOTE, RelationalGameState

INITIALIZATION_ARTIFACT_SCHEMA_VERSION = 1


def paired_initialization_directory(config: RunConfig) -> Path | None:
    environment_value = os.environ.get("MAS_CC_INITIALIZATION_ARTIFACT_DIR")
    if environment_value:
        return Path(environment_value).expanduser()
    initialization = config.game.options.get("initialization", {})
    if not isinstance(initialization, Mapping):
        return None
    value = initialization.get("artifact_dir")
    return None if value in {None, ""} else Path(str(value)).expanduser()


def paired_initialization_required(config: RunConfig) -> bool:
    initialization = config.game.options.get("initialization", {})
    return bool(
        isinstance(initialization, Mapping)
        and initialization.get("mode") == "paired_local_vote"
        and initialization.get("require_artifact", True)
    )


def initialization_artifact_path(config: RunConfig, episode_seed: int) -> Path:
    directory = paired_initialization_directory(config)
    if directory is None:
        raise ValueError(
            "paired_local_vote initialization requires initialization.artifact_dir"
        )
    return directory / f"episode-seed-{int(episode_seed)}.json"


def physical_initial_state_projection(state: RelationalGameState) -> dict[str, Any]:
    """The treatment-independent physical state established before round zero."""

    return {
        "task": _thaw(state.task),
        "agent_order": [str(agent.agent_id) for agent in state.agents],
        "agents": [
            {
                "agent_id": str(agent.agent_id),
                "committed_action": agent.committed_action,
                "public_reason": agent.public_reason,
                "public_shared_fact_id": agent.public_shared_fact_id,
                "known_fact_ids": list(agent.known_fact_ids),
                "active_fact_ids": list(agent.active_fact_ids),
                "initial_fact_ids": list(agent.initial_fact_ids),
                "fact_provenance": _thaw(agent.fact_provenance),
            }
            for agent in state.agents
        ],
    }


def initialization_compatibility_payload(
    game: Any, config: RunConfig, episode_seed: int
) -> dict[str, Any]:
    """Everything that can affect local-vote prompts or their requested model."""

    shell = game.initialize(config.game, episode_seed)
    requests = game.initial_vote_requests(shell, config.game)
    task = game.load_task(config.game)
    return {
        "schema_version": INITIALIZATION_ARTIFACT_SCHEMA_VERSION,
        "game_type": config.game.type,
        "game_version": game.spec.version,
        "episode_seed": int(episode_seed),
        "task_id": task.task_id,
        "task_sha256": canonical_hash(task.to_dict()),
        "population_size": config.game.population_size,
        "receiver_epistemic_disposition": game.rules(
            config.game
        ).receiver_epistemic_disposition,
        "prompt": config.prompt.to_dict(),
        "prompt_definition_hashes_hash": prompt_definition_hash(config),
        "provider": {
            "type": config.llm_provider.type,
            "model": config.llm_provider.model,
            "temperature": config.llm_provider.temperature,
            "max_output_tokens": config.llm_provider.max_output_tokens,
        },
        "initial_request_instance_hashes": [
            request.prompt.compile().instance_hash for request in requests
        ],
    }


def initialization_compatibility_key(
    game: Any, config: RunConfig, episode_seed: int
) -> str:
    return canonical_hash(
        initialization_compatibility_payload(game, config, episode_seed)
    )


def artifact_from_actions(
    game: Any,
    config: RunConfig,
    episode_seed: int,
    actions: Sequence[Action],
    *,
    repetition_index: int,
) -> dict[str, Any]:
    shell = game.initialize(config.game, episode_seed)
    requests = game.initial_vote_requests(shell, config.game)
    if len(actions) != len(requests):
        raise ValueError("paired initialization requires one action per agent")
    for request, action in zip(requests, actions, strict=True):
        validation = game.validate_action(shell, request, action, config.game)
        if not validation.valid:
            raise ValueError(
                f"invalid paired initialization action for {request.agent_id}: "
                + "; ".join(str(issue) for issue in validation.issues)
            )
    initialized = game.apply_initial_votes(shell, tuple(actions))
    physical = physical_initial_state_projection(initialized)
    body = {
        "schema_version": INITIALIZATION_ARTIFACT_SCHEMA_VERSION,
        "repetition_index": int(repetition_index),
        "episode_seed": int(episode_seed),
        "compatibility_key": initialization_compatibility_key(
            game, config, episode_seed
        ),
        "compatibility": initialization_compatibility_payload(
            game, config, episode_seed
        ),
        "actions": [action.to_dict() for action in actions],
        "physical_initial_state": physical,
        "physical_initial_state_hash": canonical_hash(physical),
    }
    return {**body, "artifact_hash": canonical_hash(body)}


def _action_from_dict(value: Mapping[str, Any]) -> Action:
    return Action(
        AgentId(str(value["agent_id"])),
        str(value["value"]),
        str(value.get("stage", INITIAL_VOTE)),
        dict(value.get("metadata", {})),
    )


def validate_initialization_artifact(
    artifact: Mapping[str, Any], game: Any, config: RunConfig, episode_seed: int
) -> tuple[tuple[Action, ...], RelationalGameState]:
    if artifact.get("schema_version") != INITIALIZATION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("paired initialization artifact schema is incompatible")
    expected_key = initialization_compatibility_key(game, config, episode_seed)
    if artifact.get("compatibility_key") != expected_key:
        raise ValueError("paired initialization artifact compatibility key mismatch")
    body = {key: value for key, value in artifact.items() if key != "artifact_hash"}
    if artifact.get("artifact_hash") != canonical_hash(body):
        raise ValueError("paired initialization artifact hash mismatch")
    raw_actions = artifact.get("actions")
    if isinstance(raw_actions, (str, bytes)) or not isinstance(raw_actions, Sequence):
        raise ValueError("paired initialization artifact actions must be a list")
    actions = tuple(_action_from_dict(value) for value in raw_actions)
    shell = game.initialize(config.game, episode_seed)
    requests = game.initial_vote_requests(shell, config.game)
    if [action.agent_id for action in actions] != [
        request.agent_id for request in requests
    ]:
        raise ValueError("paired initialization artifact agent order mismatch")
    for request, action in zip(requests, actions, strict=True):
        validation = game.validate_action(shell, request, action, config.game)
        if not validation.valid:
            raise ValueError(
                f"invalid paired initialization action for {request.agent_id}: "
                + "; ".join(str(issue) for issue in validation.issues)
            )
    initialized = game.apply_initial_votes(shell, actions)
    physical = physical_initial_state_projection(initialized)
    if artifact.get("physical_initial_state_hash") != canonical_hash(physical):
        raise ValueError("paired initialization physical-state hash mismatch")
    if artifact.get("physical_initial_state") != physical:
        raise ValueError("paired initialization physical state does not match actions")
    return actions, initialized


def read_initialization_artifact(
    path: str | Path, game: Any, config: RunConfig, episode_seed: int
) -> tuple[dict[str, Any], tuple[Action, ...], RelationalGameState]:
    source = Path(path)
    try:
        artifact = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read paired initialization artifact {source}: {exc}"
        ) from exc
    if not isinstance(artifact, Mapping):
        raise ValueError("paired initialization artifact must be a JSON object")
    actions, state = validate_initialization_artifact(
        artifact, game, config, episode_seed
    )
    return dict(artifact), actions, state


def write_initialization_artifact(
    path: str | Path, artifact: Mapping[str, Any]
) -> Path:
    """Publish a complete artifact atomically; never expose partial JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    payload = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return destination


__all__ = [
    "INITIALIZATION_ARTIFACT_SCHEMA_VERSION",
    "artifact_from_actions",
    "initialization_artifact_path",
    "initialization_compatibility_key",
    "initialization_compatibility_payload",
    "paired_initialization_directory",
    "paired_initialization_required",
    "physical_initial_state_projection",
    "read_initialization_artifact",
    "validate_initialization_artifact",
    "write_initialization_artifact",
]
