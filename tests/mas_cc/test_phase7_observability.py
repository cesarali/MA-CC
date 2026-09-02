import json
from pathlib import Path
from types import SimpleNamespace

from mas_cc.cli.phase7 import run_phase_7_inspection
from mas_cc.llm_runtime.messages import Message, MessageRole
from mas_cc.llm_runtime.providers import CompletionRequest, CompletionResponse, ProviderUsage
from mas_cc.llm_runtime.validation import ValidationIssue
from mas_cc.observability import DetailedAuditPolicy, DetailedAuditSelector
from mas_cc.observability.recorder import RunRecorder
from mas_cc.storage import AtomicCheckpointStore, Checkpoint


def test_detailed_audit_selector_is_deterministic_and_enforces_caps():
    selector = DetailedAuditSelector(
        DetailedAuditPolicy(
            enabled=True, always_log_first_n_rounds=1, log_every_n_rounds=3,
            max_logged_prompts_per_game=2, max_logged_prompts_per_run=3,
        )
    )
    selections = [selector.select(game_id="g", round_index=round_index) for round_index in range(1, 7)]
    assert [(item.selected, item.reason) for item in selections] == [
        (True, "round_policy"),
        (False, "not_selected_by_round_policy"),
        (True, "round_policy"),
        (False, "not_selected_by_round_policy"),
        (False, "not_selected_by_round_policy"),
        (False, "max_logged_prompts_per_game_reached"),
    ]
    assert selector.summary()["selected_prompt_records"] == 2


def test_atomic_checkpoint_rejects_changed_configuration(tmp_path: Path):
    store = AtomicCheckpointStore(tmp_path)
    store.write(Checkpoint("run", 2, "configuration-a", {"turn": 2}, {}, {"decision": "hash-a"}))
    assert store.require_compatible(
        resolved_config_hash="configuration-a", prompt_definitions={"decision": "hash-a"}
    ).completed_rounds == 2
    try:
        store.require_compatible(resolved_config_hash="configuration-b", prompt_definitions={"decision": "hash-a"})
    except ValueError as exc:
        assert "configuration" in str(exc)
    else:
        raise AssertionError("incompatible checkpoint unexpectedly accepted")


def test_phase_7_inspection_writes_bounded_local_first_artifacts(tmp_path: Path, monkeypatch):
    class LocalComet:
        def __init__(self, *args, **kwargs):
            pass

        def log_metrics(self, metrics, step):
            pass

        def close(self):
            return {"schema_version": 1, "status": "disabled", "reference": None, "reason": None, "privacy": "aggregate metrics only"}

    monkeypatch.setattr("mas_cc.observability.recorder.CometMetricSink", LocalComet)
    output = tmp_path / "phase_07"
    assert run_phase_7_inspection("configs/runs/old/naming_convention_smoke_test_v3.yaml", output)
    expected = {
        "report.md", "manifest.json", "resolved_config.yaml", "experiment.log",
        "events.jsonl", "api_call_status.jsonl", "audit_traces.jsonl",
        "prompt_block_traces.jsonl", "usage_cost.jsonl", "budget_events.jsonl",
        "checkpoint_manifest.json", "local_metrics.csv", "comet_summary.json",
        "observability_dashboard.png",
    }
    assert {path.name for path in output.iterdir() if path.is_file()} == expected
    attempts = (output / "api_call_status.jsonl").read_text(encoding="utf-8").splitlines()
    audit = (output / "audit_traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(attempts) == 24
    assert len(audit) == 5  # configured per-game cap
    assert not (output / "audit_traces.jsonl").read_text(encoding="utf-8") == ""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "pass"
    assert all(manifest["checks"].values())
    assert "compiled_messages" not in (output / "comet_summary.json").read_text(encoding="utf-8")


def test_malformed_response_diagnostic_is_field_only_and_bounded(tmp_path: Path):
    recorder = RunRecorder(
        tmp_path, run_id="diagnostic", resolved_config={},
        policy=DetailedAuditPolicy(enabled=False),
    )
    recorder.malformed_response_row_cap = 1
    request = CompletionRequest(
        (Message(MessageRole.USER, "original"),),
        metadata={
            "interaction_id": "i1", "agent_id": "a1", "decision_stage": "vote",
            "validation_attempt": 1, "validation_repair": False,
            "repair_schema_version": 1, "effective_messages_hash": "abc",
        },
    )
    response = CompletionResponse(
        content='{"reason":"private","shared_fact_id":"Fact f2"}',
        provider="mock", model="fake", usage=ProviderUsage(10, 5, 15),
        finish_reason="stop",
    )
    prompt = SimpleNamespace(
        family="test", version=1, definition_hash="definition", instance_hash="instance",
    )
    issue = ValidationIssue(
        "response.shared_fact_id", "must be a bare fact identifier", "Fact f2",
    )
    kwargs = dict(
        round_index=1, game_id="relational", request=request, prompt=prompt,
        response=response, attempt=1, valid=False, validation_error=str(issue),
        validation_issues=(issue,),
    )

    recorder.record_attempt(**kwargs)
    recorder.record_attempt(**kwargs)

    rows = [json.loads(line) for line in (
        tmp_path / "runtime" / "malformed_responses.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    assert rows[0]["received_value"] == "Fact f2"
    assert rows[0]["received_type"] == "str"
    assert "private" not in json.dumps(rows)
    assert rows[1] == {
        "row_cap": 1, "run_id": "diagnostic", "schema_version": 1,
        "timestamp": rows[1]["timestamp"], "truncated": True,
    }
