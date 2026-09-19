"""The System One client is opt-in, cached by content hash, and never called by the finalizer."""

from __future__ import annotations

import json

import pytest

from mas_cc.analysis import systemone


def _fake_transport(log: list):
    def transport(payload: dict) -> dict:
        log.append(payload)
        answers = {}
        for name, question in payload["questions"].items():
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.25}
            elif question["type"] == "choice":
                options = list(question["criteria"])
                answers[name] = {"type": "choice", "choice": options[0], "confidence": 0.9,
                                 "probabilities": {o: (0.9 if i == 0 else 0.1 / max(1, len(options) - 1)) for i, o in enumerate(options)}}
            else:
                levels = question["criteria"]
                answers[name] = {"type": "score", "score": 1.0, "confidence": 0.8,
                                 "legend": {str(i): lvl for i, lvl in enumerate(levels)},
                                 "probabilities": {str(i): (1.0 if i == 1 else 0.0) for i in range(len(levels))}}
        return {"model": "jev-test", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 3}}
    return transport


def test_ask_caches_by_content_hash(tmp_path):
    log: list = []
    client = systemone.SystemOneClient(cache_dir=tmp_path, transport=_fake_transport(log), key="k")
    questions = {
        "cites": systemone.noul("Does the message cite the controller?", true="yes", false="no"),
        "stance": systemone.choice("Which option?", {"A": "argues A", "B": "argues B", "none": "neither"}),
        "push": systemone.score("How strong?", ["none", "weak", "strong"]),
    }
    first = client.ask({"text": "hello"}, questions)
    second = client.ask({"text": "hello"}, questions)
    assert first["cached"] is False and second["cached"] is True
    assert len(log) == 1, "the second identical request must be served from the cache"
    assert first["request_id"] == second["request_id"] == systemone.request_id(client.model, {"text": "hello"}, questions)
    assert client.usage.requests == 1 and client.usage.cached == 1 and client.usage.input_tokens == 10
    on_disk = json.loads((tmp_path / first["request_id"][:2] / f"{first['request_id']}.json").read_text())
    assert on_disk["answers"]["stance"]["choice"] == "A"
    # A different state is a different request.
    client.ask({"text": "goodbye"}, questions)
    assert len(log) == 2


def test_flatten_answers_to_rows():
    rows = [systemone.flatten_answer(n, a) for n, a in {
        "cites": {"type": "noul", "noul": 0.03},
        "stance": {"type": "choice", "choice": "none", "confidence": 1.0, "probabilities": {"none": 1.0, "A": 0.0}},
        "push": {"type": "score", "score": 0.01, "confidence": 0.99, "probabilities": {"0": 1.0, "1": 0.0}},
    }.items()]
    assert rows[0] == {"question": "cites", "answer_type": "noul", "value": 0.03, "label": None, "confidence": None, "probabilities_json": None}
    assert rows[1]["label"] == "none" and rows[1]["confidence"] == 1.0
    assert rows[2]["value"] == 0.01 and json.loads(rows[2]["probabilities_json"]) == {"0": 1.0, "1": 0.0}


def test_validation_and_credential_errors(tmp_path, monkeypatch):
    client = systemone.SystemOneClient(cache_dir=tmp_path, transport=_fake_transport([]), key="k")
    with pytest.raises(ValueError):
        client.ask("s", {})
    with pytest.raises(ValueError):
        client.ask("s", {"q": {"type": "essay", "instructions": "write"}})
    with pytest.raises(ValueError):
        systemone.score("x", ["only-one"])
    monkeypatch.delenv("MA_CC_SYSTEMONE_KEY_FILE", raising=False)
    monkeypatch.delenv("LLM_KEY", raising=False)
    bare = systemone.SystemOneClient(cache_dir=None)
    with pytest.raises(systemone.SystemOneError):
        bare._bearer()


def test_malformed_response_is_an_error(tmp_path):
    client = systemone.SystemOneClient(cache_dir=tmp_path, transport=lambda payload: {"answers": {}}, key="k")
    with pytest.raises(systemone.SystemOneError):
        client.ask("s", {"q": systemone.noul("is it?")})
    assert client.usage.requests == 0


def test_default_url_prefers_explicit_then_cluster_then_public(monkeypatch):
    monkeypatch.delenv("MA_CC_SYSTEMONE_URL", raising=False)
    monkeypatch.delenv("LLM_BASE", raising=False)
    assert systemone.default_url() == systemone.PUBLIC_URL
    monkeypatch.setenv("LLM_BASE", "http://llm.llm.svc.cluster.local:4000/")
    assert systemone.default_url() == "http://llm.llm.svc.cluster.local:4000/typesafe/v1/systemone"
    monkeypatch.setenv("MA_CC_SYSTEMONE_URL", "http://example.test/s1")
    assert systemone.default_url() == "http://example.test/s1"
    assert systemone.SystemOneClient(cache_dir=None).url == "http://example.test/s1"
