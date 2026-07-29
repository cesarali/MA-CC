"""Explicit task configuration and prompts for the optional reasoning game."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from .models import AgentSnapshot, ConfigurationError, Name, inventory_values


@dataclass(frozen=True)
class ReasoningTask:
    task: str
    claims: Mapping[Name, str]
    evidence_by_agent: Mapping[int, str]
    default_evidence: str | None = None

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ConfigurationError("A reasoning task needs a non-empty task description.")
        if set(self.claims) != {"A", "B"}:
            raise ConfigurationError("A reasoning task must define claims for A and B.")
        object.__setattr__(self, "claims", MappingProxyType(dict(self.claims)))
        object.__setattr__(
            self, "evidence_by_agent", MappingProxyType(dict(self.evidence_by_agent))
        )

    def evidence_for(self, agent_id: int) -> str:
        evidence = self.evidence_by_agent.get(agent_id, self.default_evidence)
        if evidence is None:
            raise ConfigurationError(
                f"Reasoning task has no evidence configured for agent {agent_id}."
            )
        return evidence


def load_reasoning_task(path: str | Path) -> ReasoningTask:
    """Load a non-trivial reasoning task from JSON or YAML configuration."""

    source = Path(path)
    if not source.exists():
        raise ConfigurationError(f"Reasoning-task file does not exist: {source}")
    try:
        if source.suffix.lower() == ".json":
            raw = json.loads(source.read_text(encoding="utf-8"))
        else:
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Could not parse reasoning-task file: {source}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("Reasoning-task configuration must be an object.")

    claims = raw.get("claims")
    if not isinstance(claims, dict):
        raise ConfigurationError("Reasoning-task configuration needs a claims mapping.")
    evidence = raw.get("evidence_by_agent", {})
    if not isinstance(evidence, dict):
        raise ConfigurationError("evidence_by_agent must be a mapping.")
    try:
        evidence_by_agent = {int(key): str(value) for key, value in evidence.items()}
        normalized_claims = {str(key): str(value) for key, value in claims.items()}
        return ReasoningTask(
            task=str(raw.get("task", "")),
            claims=normalized_claims,  # type: ignore[arg-type]
            evidence_by_agent=evidence_by_agent,
            default_evidence=(
                str(raw["default_evidence"])
                if raw.get("default_evidence") is not None
                else None
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("Reasoning-task fields have invalid types.") from exc


def build_reasoning_speaker_messages(
    speaker: AgentSnapshot, task: ReasoningTask
) -> list[dict[str, str]]:
    evidence = task.evidence_for(speaker.agent_id)
    return [
        {
            "role": "system",
            "content": (
                "You are the speaker in a binary reasoning Naming Game. Select one "
                "name present in your state and give a concise evidence-based reason. "
                "Return only JSON with selected_name and reason."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                (
                    "ACTION: speaker_reasoning",
                    f"AGENT_ID: {speaker.agent_id}",
                    f"INVENTORY_JSON: {json.dumps(inventory_values(speaker.inventory))}",
                    f"TASK: {task.task}",
                    f"CLAIM_A: {task.claims['A']}",
                    f"CLAIM_B: {task.claims['B']}",
                    f"OWN_EVIDENCE: {evidence}",
                    'OUTPUT_SCHEMA: {"selected_name":"A","reason":"..."}',
                )
            ),
        },
    ]


def build_reasoning_listener_messages(
    listener: AgentSnapshot,
    task: ReasoningTask,
    selected_name: Name,
    reason: str,
) -> list[dict[str, str]]:
    evidence = task.evidence_for(listener.agent_id)
    return [
        {
            "role": "system",
            "content": (
                "You are the listener in a binary reasoning Naming Game. Assess the "
                "speaker's claim and reason against your own evidence and current state. "
                "Return only JSON with new_inventory containing A, B, or both."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                (
                    "ACTION: listener_reasoning",
                    f"AGENT_ID: {listener.agent_id}",
                    f"INVENTORY_JSON: {json.dumps(inventory_values(listener.inventory))}",
                    f"TASK: {task.task}",
                    f"CLAIM: {task.claims[selected_name]}",
                    f"SPEAKER_REASON: {reason}",
                    f"OWN_EVIDENCE: {evidence}",
                    'OUTPUT_SCHEMA: {"new_inventory":["A"]}',
                )
            ),
        },
    ]

