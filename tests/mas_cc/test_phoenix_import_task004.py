"""Check that retrospective Task 004 traces retain the right prompt owners."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


pytest.importorskip("openinference.semconv.trace")
pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/local/phoenix_import.py"
SPEC = importlib.util.spec_from_file_location("phoenix_import_task004_test", SCRIPT)
assert SPEC and SPEC.loader
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_controller_prompt_and_initial_vote_have_distinct_spans(tmp_path: Path) -> None:
    episode = tmp_path / "false_allfacts_rho075" / "episode-0000"
    episode.mkdir(parents=True)
    _write_jsonl(episode / "api_call_status.jsonl", [
        {"decision_stage": "initial_vote", "round_index": 1,
         "interaction_id": "initial-local-votes", "agent_id": "agent_001",
         "model": "model", "provider": "deepinfra", "valid": True},
        {"decision_stage": "focal_update", "round_index": 1,
         "interaction_id": "interaction-0001", "agent_id": "agent_001",
         "model": "model", "provider": "deepinfra", "valid": True},
        {"decision_stage": "controller_communication", "round_index": 1,
         "interaction_id": "None", "agent_id": "None",
         "model": "model", "provider": "deepinfra", "valid": True},
    ])
    _write_jsonl(episode / "usage_cost.jsonl", [])
    _write_jsonl(episode / "trajectory.jsonl", [
        {"interaction_index": 1, "interaction_id": "interaction-0001",
         "decisions": [{"action": {"agent_id": "agent_001", "value": "ALLOCATION_0",
                                    "metadata": {"reason": "because", "public_message": None}}}]},
    ])
    _write_jsonl(episode / "round_trajectory.jsonl", [
        {"absolute_round": 1, "controller_target": "ALLOCATION_2",
         "correct_answer": "ALLOCATION_0", "epistemic_persistence": 0.75,
         "communication_profile": "report_only", "intervention_budget": 3},
        {"absolute_round": 2, "controller_target": "ALLOCATION_2",
         "correct_answer": "ALLOCATION_0", "epistemic_persistence": 0.75,
         "communication_profile": "report_only", "intervention_budget": 3,
         "controller_llm_attempts": [{"attempt": 1, "request": {"messages": [
             {"role": "user", "content": "controller exact prompt"}]},
             "raw_output": '{"post": ["fact"]}'}]},
    ])
    prompt = """# Round 1 — agent agent_001
- Prompt version: `relational_public_ballot@1`
## Exact messages sent to the LLM
### Message 1 — `user`
```text
initial exact prompt
```
## Response contract
not part of input
"""
    _write_jsonl(episode / "prompts.jsonl", [{"name": "round_001.md", "content": prompt}])

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    counts = IMPORTER.import_episode(provider.get_tracer("test"), episode, "suite")
    spans = exporter.get_finished_spans()
    assert counts == (3, 0, 1, 1)
    assert len(spans) == 7  # episode, initialization, two rounds, two agents, controller
    by_name = {span.name: span for span in spans}
    initial = by_name["agent_001 r1 initial_vote"]
    focal = by_name["agent_001 r1 focal_update"]
    controller = by_name["controller communication r2"]
    assert "initial exact prompt" in initial.attributes["input.value"]
    assert "not part of input" not in initial.attributes["input.value"]
    assert "input.value" not in focal.attributes
    assert "controller exact prompt" in controller.attributes["input.value"]
    assert by_name["episode episode-0000"].attributes["mas_cc.arm"] == "false-controller"
