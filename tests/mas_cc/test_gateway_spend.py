"""Run cost = spend delta of the submitting key between submit and finalize; never raises."""

from __future__ import annotations

import io
import json
import urllib.request

import pytest

from mas_cc.observability import gateway_spend as gs


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _serve(spends: list[float], monkeypatch):
    calls: list[str] = []

    def fake_urlopen(request, timeout=0):
        calls.append(request.full_url)
        spend = spends.pop(0)
        return _Response(json.dumps({"info": {"spend": spend, "key_alias": "user-test", "user_id": "u@example", "team_id": None,
                                             "max_budget": None}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def test_snapshot_and_delta(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_KEY", "sk-test-key")
    monkeypatch.setenv("LLM_BASE", "http://llm.test:4000/")
    calls = _serve([10.0, 12.5], monkeypatch)
    start = gs.record_run_start(tmp_path)
    assert start["start"]["status"] == "ok" and start["start"]["spend_usd"] == 10.0
    assert (tmp_path / gs.RUN_COST_FILE).is_file()
    record = gs.run_cost(tmp_path)
    assert record["status"] == "ok" and record["usd"] == 2.5
    assert record["start"]["key_fingerprint"] == record["end"]["key_fingerprint"] == gs.key_fingerprint("sk-test-key")
    assert calls == ["http://llm.test:4000/key/info"] * 2
    assert "sk-test-key" not in json.dumps(record)


def test_unavailable_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE", raising=False)
    monkeypatch.setenv("LLM_KEY_FILE", str(tmp_path / "missing"))
    snap = gs.key_spend()
    assert snap["status"] == "unavailable"
    record = gs.run_cost(tmp_path)
    assert record["status"] == "no_start_snapshot" and "usd" not in record


def test_gateway_error_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_KEY", "k")
    monkeypatch.setenv("LLM_BASE", "http://llm.test:4000")

    def boom(request, timeout=0):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    snap = gs.key_spend()
    assert snap["status"] == "unavailable" and "URLError" in snap["reason"]


def test_key_change_between_snapshots_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_BASE", "http://llm.test:4000")
    monkeypatch.setenv("LLM_KEY", "first")
    _serve([1.0, 3.0], monkeypatch)
    gs.record_run_start(tmp_path)
    monkeypatch.setenv("LLM_KEY", "second")
    record = gs.run_cost(tmp_path)
    assert record["status"] == "key_changed" and "usd" not in record
