from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from mas_cc.core import (
    AgentId,
    Message,
    MessageRole,
    Seed,
    Timestamp,
    ValidationIssue,
    ValidationResult,
)
from mas_cc.core.exceptions import ValidationError


def test_identifiers_are_typed_immutable_and_validated():
    identifier = AgentId("agent-01")
    assert str(identifier) == "agent-01"
    with pytest.raises(FrozenInstanceError):
        identifier.value = "agent-02"
    with pytest.raises(ValueError, match="AgentId.value"):
        AgentId("contains spaces")


def test_seed_derivation_is_stable_and_namespaced():
    seed = Seed(1026)
    assert seed.derive("episode-1") == seed.derive("episode-1")
    assert seed.derive("episode-1") != seed.derive("episode-2")
    assert seed.create_random().random() == seed.create_random().random()
    with pytest.raises(ValueError, match="between"):
        Seed(-1)


def test_timestamp_requires_timezone_and_normalizes_to_utc():
    source = datetime(2026, 8, 1, 12, tzinfo=timezone(timedelta(hours=2)))
    timestamp = Timestamp(source)
    assert timestamp.isoformat() == "2026-08-01T10:00:00Z"
    assert Timestamp.parse(timestamp.isoformat()) == timestamp
    with pytest.raises(ValueError, match="timezone"):
        Timestamp(datetime(2026, 8, 1))


def test_message_is_immutable_and_serializes_provider_independently():
    message = Message(
        role="user",
        content="Choose A or B",
        metadata={"prompt_version": 1, "labels": ["A", "B"]},
    )
    assert message.role is MessageRole.USER
    assert message.to_dict() == {
        "role": "user",
        "content": "Choose A or B",
        "metadata": {"prompt_version": 1, "labels": ["A", "B"]},
    }
    with pytest.raises(TypeError):
        message.metadata["prompt_version"] = 2
    with pytest.raises(TypeError):
        message.metadata["labels"][0] = "B"


def test_validation_result_preserves_exact_fields():
    issue = ValidationIssue("game.population_size", "must be at least 2", {"value": [1]})
    result = ValidationResult.failure(issue)
    assert not result.is_valid
    with pytest.raises(ValidationError, match=r"game\.population_size"):
        result.raise_for_errors(context="game config")
    with pytest.raises(TypeError):
        issue.invalid_value["value"][0] = 2
