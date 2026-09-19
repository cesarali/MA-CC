"""Per-decision timing rows: emitted by the relational runtime, written by the recorder, summarised by the report."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from mas_cc.games import create_game
from mas_cc.games.relational_reasoning.imitation_round_feedback.runtime import (
    run_relational_imitation_round_feedback_game,
)
from mas_cc.llm_runtime.providers.responses import CompletionResponse
from mas_cc.observability import otel
from mas_cc.observability.recorder import RunRecorder

_spec = importlib.util.spec_from_file_location("rb_fixtures", Path(__file__).with_name("test_relational_blackboard.py"))
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)

_report_spec = importlib.util.spec_from_file_location(
    "decision_timing_report", Path(__file__).parents[2] / "scripts" / "Cygnus" / "analysis" / "decision_timing_report.py")
report = importlib.util.module_from_spec(_report_spec)
_report_spec.loader.exec_module(report)


class _Observer:
    def __init__(self) -> None:
        self.timing: list[dict] = []
        self.attempts: list[dict] = []

    def record_decision_timing(self, **row):
        self.timing.append(row)

    def record_attempt(self, **payload):
        self.attempts.append(payload)


def test_runtime_emits_one_timing_row_per_logical_decision():
    config = fixtures._config()
    ballots = fixtures._BoardBallots(post=False)
    observer = _Observer()
    result = asyncio.run(run_relational_imitation_round_feedback_game(
        create_game(config.game), config, ballots.provider(config.llm_provider), observer=observer))
    first_attempts = [a for a in observer.attempts if a["attempt"] == 1]
    assert observer.timing, "no timing rows"
    assert len(observer.timing) == len(first_attempts)
    row = observer.timing[0]
    assert {"round_index", "interaction_id", "decision_stage", "agent_id", "outcome", "wall_seconds",
            "provider_latency_seconds", "queue_seconds", "admission_seconds", "provider_retries",
            "validation_attempts", "provider_errors", "status_code", "input_tokens", "output_tokens",
            "provider", "model", "prompt_family"} <= set(row)
    assert all(r["outcome"] == "ok" and r["validation_attempts"] == 1 for r in observer.timing)
    assert all(r["wall_seconds"] >= r["provider_latency_seconds"] >= 0 for r in observer.timing)
    assert len({r["agent_id"] for r in observer.timing}) > 1
    assert all(r["provider"] == "mock" for r in observer.timing)


def test_completion_response_carries_the_wait_fields():
    response = CompletionResponse(content="{}", provider="p", model="m", latency_seconds=1.5,
                                  queue_seconds=0.2, admission_seconds=0.7)
    as_dict = response.to_dict()
    assert as_dict["queue_seconds"] == 0.2 and as_dict["admission_seconds"] == 0.7
    assert CompletionResponse(content="{}", provider="p", model="m").to_dict()["admission_seconds"] is None


def test_recorder_writes_one_json_line(tmp_path):
    stub = SimpleNamespace(_decision_timing_path=tmp_path / "decision_timing.jsonl", schema_version=7, run_id="run-x")
    RunRecorder.record_decision_timing(stub, agent_id="agent_001", decision_stage="vote", wall_seconds=0.5,
                                       provider_latency_seconds=0.4, queue_seconds=0.0, admission_seconds=0.05)
    RunRecorder.record_decision_timing(stub, agent_id="agent_002", decision_stage="vote", wall_seconds=0.7,
                                       provider_latency_seconds=0.6, queue_seconds=0.0, admission_seconds=0.0)
    lines = [json.loads(l) for l in (tmp_path / "decision_timing.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and lines[0]["run_id"] == "run-x" and lines[0]["schema_version"] == 7
    assert lines[1]["agent_id"] == "agent_002" and "recorded_at" in lines[1]


def test_report_splits_wall_time(tmp_path):
    rows = [
        {"round_index": 1, "decision_stage": "vote", "agent_id": "a", "outcome": "ok", "wall_seconds": 10.0,
         "provider_latency_seconds": 8.0, "queue_seconds": 1.0, "admission_seconds": 2.0, "provider_retries": 0, "validation_attempts": 1},
        {"round_index": 1, "decision_stage": "message", "agent_id": "a", "outcome": "ok", "wall_seconds": 4.0,
         "provider_latency_seconds": 3.0, "queue_seconds": 0.0, "admission_seconds": 0.0, "provider_retries": 1, "validation_attempts": 2},
    ]
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "decision_timing.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    frame = report.load(tmp_path)
    assert len(frame) == 2
    first = frame.iloc[0]
    assert first["client_overhead_seconds"] == 2.0 and first["backpressure_seconds"] == 3.0 and first["model_seconds"] == 5.0
    summary = report.summarize(frame, ["decision_stage"])
    assert set(summary["decision_stage"]) == {"vote", "message"} and set(summary["decisions"]) == {1}
    assert report.main([str(tmp_path)]) == 0


def test_otel_is_a_no_op_without_an_endpoint(monkeypatch):
    monkeypatch.delenv("MA_CC_OTEL_ENDPOINT", raising=False)
    monkeypatch.setattr(otel, "_TRACER", None)
    monkeypatch.setattr(otel, "_DISABLED", False)
    with otel.decision_span("x", {"a": 1}) as span:
        assert span is None
    assert otel.enabled() is False


def test_production_observer_forwards_timing_rows():
    from mas_cc.experiments.orchestrator import _RoundTickingObserver

    class _Recorder:
        retention_policy = SimpleNamespace(compact_scientific=True)
        rows: list = []

        def record_decision_timing(self, **row):
            self.rows.append(row)

    recorder = _Recorder()
    observer = _RoundTickingObserver(recorder, guard=None, progress=None, episode_label="e")
    observer.record_decision_timing(agent_id="a", wall_seconds=1.0)
    assert recorder.rows == [{"agent_id": "a", "wall_seconds": 1.0}]
