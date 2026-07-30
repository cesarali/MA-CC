"""Isolated two-call pair interactions shared by both update engines."""

from __future__ import annotations

import json
import math
import random
import re
import time
from typing import Any, Literal

from .api_client import LLMClient
from .models import (
    AgentSnapshot,
    InteractionResult,
    Inventory,
    Name,
    inventory_values,
    normalize_inventory,
    LLMResponse,
)
from .reasoning_game import (
    ReasoningTask,
    build_reasoning_listener_messages,
    build_reasoning_speaker_messages,
)


def build_basic_speaker_messages(speaker: AgentSnapshot) -> list[dict[str, str]]:
    """Build a request containing only the speaker's permitted information."""

    return [
        {
            "role": "system",
            "content": (
                "You are the speaker in a binary Naming Game. Select exactly one name "
                "from your own inventory. Return only the requested short JSON object."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                (
                    "ACTION: speaker_basic",
                    f"AGENT_ID: {speaker.agent_id}",
                    f"INVENTORY_JSON: {json.dumps(inventory_values(speaker.inventory))}",
                    'OUTPUT_SCHEMA: {"selected_name":"A"}',
                )
            ),
        },
    ]


def build_basic_listener_messages(
    listener: AgentSnapshot, selected_name: Name
) -> list[dict[str, str]]:
    """Build a request containing only listener state and the paired message."""

    return [
        {
            "role": "system",
            "content": (
                "You are the listener in a binary Naming Game. Report whether the "
                "transmitted name was already in your own inventory. Return only the "
                "requested short JSON object."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                (
                    "ACTION: listener_basic",
                    f"AGENT_ID: {listener.agent_id}",
                    f"INVENTORY_JSON: {json.dumps(inventory_values(listener.inventory))}",
                    f"TRANSMITTED_NAME: {selected_name}",
                    'OUTPUT_SCHEMA: {"already_known":true}',
                )
            ),
        },
    ]


def _parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response does not contain a JSON object")
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("response contains malformed JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def _repair_speaker_selection(inventory: Inventory, choice_seed: int) -> Name:
    choices = inventory_values(inventory)
    return random.Random(choice_seed).choice(choices)


def _validate_speaker(
    content: str, inventory: Inventory, choice_seed: int
) -> tuple[Name, bool, str | None, str | None]:
    reason: str | None = None
    try:
        body = _parse_json_object(content)
        candidate = body.get("selected_name")
        reason_value = body.get("reason")
        if reason_value is not None:
            if not isinstance(reason_value, str) or not reason_value.strip():
                raise ValueError("reason must be a non-empty string")
            reason = reason_value.strip()
        if candidate not in ("A", "B"):
            raise ValueError("selected_name must be A or B")
        if candidate not in inventory:
            raise ValueError("selected_name was not in the speaker inventory")
        return candidate, True, None, reason
    except ValueError as exc:
        return (
            _repair_speaker_selection(inventory, choice_seed),
            False,
            f"{exc}; locally repaired from the immutable speaker inventory",
            reason,
        )


def _validate_basic_listener(
    content: str, engine_known: bool
) -> tuple[bool | None, bool, str | None]:
    try:
        body = _parse_json_object(content)
        reported = body.get("already_known")
        if not isinstance(reported, bool):
            raise ValueError("already_known must be a JSON boolean")
        if reported != engine_known:
            return (
                reported,
                False,
                "listener disagreed with engine truth; engine result was applied",
            )
        return reported, True, None
    except ValueError as exc:
        return None, False, f"{exc}; engine truth was applied"


def basic_naming_update(
    speaker_inventory: Inventory,
    listener_inventory: Inventory,
    selected_name: Name,
) -> tuple[Inventory, Inventory, bool]:
    """Apply the engine-authoritative binary Naming Game update."""

    speaker_inventory = normalize_inventory(speaker_inventory)
    listener_inventory = normalize_inventory(listener_inventory)
    if selected_name not in speaker_inventory:
        raise ValueError("The transmitted name must be in the speaker inventory.")
    success = selected_name in listener_inventory
    if success:
        singleton = frozenset({selected_name})
        return normalize_inventory(singleton), normalize_inventory(singleton), True
    return (
        speaker_inventory,
        normalize_inventory(listener_inventory | {selected_name}),
        False,
    )


async def execute_pair_interaction(
    *,
    client: LLMClient,
    speaker: AgentSnapshot,
    listener: AgentSnapshot,
    interaction_index: int,
    round_index: int | None,
    pair_index: int | None,
    interaction_kind: Literal["basic", "reasoning"],
    choice_seed: int,
    temperature: float,
    max_tokens_speaker: int,
    max_tokens_listener: int,
    reasoning_task: ReasoningTask | None = None,
) -> InteractionResult:
    """Execute one isolated speaker-call -> listener-call chain."""

    started = time.perf_counter()
    if interaction_kind == "reasoning":
        if reasoning_task is None:
            raise ValueError("A reasoning task is required for reasoning interactions.")
        speaker_messages = build_reasoning_speaker_messages(speaker, reasoning_task)
    else:
        speaker_messages = build_basic_speaker_messages(speaker)

    constrained = getattr(client, "complete_constrained", None)
    decision = None
    # Local constrained decisions are opt-in by provider capability; remote clients
    # retain the established generated JSON path.
    if interaction_kind == "basic" and getattr(client, "provider_name", None) == "gemma_local" and callable(constrained):
        allowed = inventory_values(speaker.inventory)
        decision = await constrained(speaker_messages, choices=allowed, temperature=max(temperature, 1.0))
        selected = decision.selected_choice
        speaker_response = LLMResponse(
            content=json.dumps({"selected_name": selected}), model=decision.model,
            latency_seconds=decision.latency_seconds, usage=decision.usage,
        )
        speaker_valid, speaker_error, reason = True, None, None
    else:
        speaker_response = await client.complete(
            speaker_messages,
            temperature=temperature,
            max_tokens=max_tokens_speaker,
        )
        selected, speaker_valid, speaker_error, reason = _validate_speaker(
            speaker_response.content, speaker.inventory, choice_seed
        )

    if interaction_kind == "basic":
        listener_messages = build_basic_listener_messages(listener, selected)
    else:
        assert reasoning_task is not None
        if reason is None:
            reason = "No valid reason was returned by the speaker."
            speaker_valid = False
            speaker_error = (
                (speaker_error + "; ") if speaker_error else ""
            ) + "missing reason was locally repaired"
        listener_messages = build_reasoning_listener_messages(
            listener, reasoning_task, selected, reason
        )

    listener_response = await client.complete(
        listener_messages,
        temperature=temperature,
        max_tokens=max_tokens_listener,
    )

    if interaction_kind == "basic":
        engine_known = selected in listener.inventory
        listener_report, listener_valid, listener_error = _validate_basic_listener(
            listener_response.content, engine_known
        )
        speaker_after, listener_after, success = basic_naming_update(
            speaker.inventory, listener.inventory, selected
        )
    else:
        engine_known = None
        listener_report = None
        success = None
        speaker_after = speaker.inventory
        try:
            body = _parse_json_object(listener_response.content)
            listener_after = normalize_inventory(body.get("new_inventory"))
            listener_valid = True
            listener_error = None
        except ValueError as exc:
            listener_after = listener.inventory
            listener_valid = False
            listener_error = f"{exc}; listener state was left unchanged"

    probabilities = ({score.choice: score.probability for score in decision.scores} if decision else None)
    return InteractionResult(
        interaction_index=interaction_index,
        round_index=round_index,
        pair_index=pair_index,
        interaction_kind=interaction_kind,
        speaker_id=speaker.agent_id,
        listener_id=listener.agent_id,
        speaker_before=speaker.inventory,
        listener_before=listener.inventory,
        selected_name=selected,
        listener_reported_known=listener_report,
        engine_already_known=engine_known,
        naming_success=success,
        speaker_after=speaker_after,
        listener_after=listener_after,
        speaker_response=speaker_response,
        listener_response=listener_response,
        speaker_response_valid=speaker_valid,
        listener_response_valid=listener_valid,
        speaker_validation_error=speaker_error,
        listener_validation_error=listener_error,
        reason=reason,
        pair_wall_seconds=time.perf_counter() - started,
        decision_method="constrained_sequence" if decision else "generated",
        allowed_choices=tuple(inventory_values(speaker.inventory)) if decision else None,
        choice_log_likelihoods=({score.choice: score.log_likelihood for score in decision.scores} if decision else None),
        choice_probabilities=probabilities,
        selected_choice_probability=(probabilities[selected] if probabilities else None),
        choice_entropy=(-sum(p * math.log(p) for p in probabilities.values() if p > 0) if probabilities else None),
    )
