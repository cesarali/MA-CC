"""TypeSafe System One as a ballot provider: typed Choice in, contract-valid ballot JSON out."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from mas_cc.analysis import systemone
from mas_cc.games import create_game
from mas_cc.games.relational_reasoning.imitation_round_feedback.runtime import (
    run_relational_imitation_round_feedback_game,
)
from mas_cc.llm_runtime.config import LLMProviderConfig
from mas_cc.llm_runtime.providers.adapters.typesafe import TypeSafeChoiceProvider, presented_letters
from mas_cc.llm_runtime.providers.errors import ProviderError
from mas_cc.llm_runtime.providers.registry import create_llm_provider
from mas_cc.llm_runtime.providers.requests import CompletionRequest, Message

_spec = importlib.util.spec_from_file_location("rb_fixtures", Path(__file__).with_name("test_relational_blackboard.py"))
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)

BALLOT = (
    "Return only valid JSON:\n\n{\n  \"vote\": \"<A | B | C>\",\n"
    "  \"private_reason\": \"<a few sentences>\",\n  \"public_message\": {\"type\": \"<REPORT | NONE>\"}\n}"
)


def _request(text: str = BALLOT) -> CompletionRequest:
    return CompletionRequest((Message("system", "You are agent_003."), Message("user", text)), max_output_tokens=64)


def _transport(choice="B", probabilities=None, confidence=0.71):
    calls = []

    def send(payload):
        calls.append(payload)
        return {"model": "jev-1.13.0",
                "answers": {qid: {"choice": choice, "probabilities": probabilities or {"A": 0.2, "B": 0.7, "C": 0.1},
                                  "confidence": confidence} for qid in payload["questions"]},
                "usage": {"input_tokens": 120, "output_tokens": 0}}

    return send, calls


def _provider(send, **options) -> TypeSafeChoiceProvider:
    client = systemone.SystemOneClient(model="jev-1.13.0", key="test-key", transport=send)
    return TypeSafeChoiceProvider(LLMProviderConfig(type="typesafe", model="jev-1.13.0", options=options), client=client)


def test_letters_are_parsed_from_the_ballot_instruction():
    assert presented_letters(_request()) == ("A", "B", "C")
    assert presented_letters(_request("no ballot here"), fallback=("X", "Y")) == ("X", "Y")
    with pytest.raises(ProviderError):
        presented_letters(_request("no ballot here"))


def test_choice_becomes_a_contract_shaped_ballot_with_the_probability_vector():
    send, calls = _transport()
    response = asyncio.run(_provider(send).complete(_request()))
    ballot = json.loads(response.content)
    assert ballot["vote"] == "B"
    assert "P(B)=0.700" in ballot["private_reason"] and "confidence 0.710" in ballot["private_reason"]
    assert ballot["public_message"] == {"type": "NONE", "text": None, "shared_fact_id": None, "reply_to": None}
    assert response.provider == "typesafe" and response.model == "jev-1.13.0"
    assert response.usage.input_tokens == 120 and response.raw_response["typesafe"]["probabilities"]["B"] == 0.7
    payload = calls[0]
    assert payload["model"] == "jev-1.13.0"
    assert [m["role"] for m in payload["state"]["conversation"]] == ["system", "user"]
    question = payload["questions"]["vote"]
    assert question["type"] == "choice" and set(question["criteria"]) == {"A", "B", "C"}


def test_a_letter_outside_the_ballot_is_a_provider_error_not_a_ballot():
    send, _ = _transport(choice="D")
    with pytest.raises(ProviderError) as info:
        asyncio.run(_provider(send).complete(_request()))
    assert info.value.code == "bad_choice"


def test_budget_refusal_surfaces_as_a_provider_error():
    send, calls = _transport()
    client = systemone.SystemOneClient(model="jev-1.13.0", key="k", transport=send,
                                       budget=systemone.Budget(max_input_tokens=1))
    provider = TypeSafeChoiceProvider(LLMProviderConfig(type="typesafe", model="jev-1.13.0"), client=client)
    with pytest.raises(ProviderError) as info:
        asyncio.run(provider.complete(_request()))
    assert info.value.code == "budget" and calls == []


def test_registry_selects_the_provider_and_dry_run_needs_no_network():
    provider = create_llm_provider(LLMProviderConfig(type="typesafe", model="jev-1.13.0", options={"dry_run": True}))
    assert isinstance(provider, TypeSafeChoiceProvider)
    response = asyncio.run(provider.complete(_request()))
    assert json.loads(response.content)["vote"] == "A" and response.raw_response["dry_run"] is True


def test_the_relational_board_game_accepts_dry_run_typesafe_ballots():
    config = fixtures._config()
    provider = create_llm_provider(LLMProviderConfig(type="typesafe", model="jev-1.13.0", options={"dry_run": True}))

    class _Observer:
        attempts: list = []

        def record_attempt(self, **payload):
            self.attempts.append(payload)

    observer = _Observer()
    asyncio.run(run_relational_imitation_round_feedback_game(create_game(config.game), config, provider, observer=observer))
    assert observer.attempts
    assert all(a["attempt"] == 1 for a in observer.attempts), "typesafe ballots were rejected by the contract"
