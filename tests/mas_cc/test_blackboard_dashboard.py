from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
from urllib.request import ProxyHandler, build_opener

import pytest

from mas_cc.blackboard_dashboard.data import BlackboardRunReader, _jsonl
from mas_cc.blackboard_dashboard.server import make_handler, serve_dashboard
from mas_cc.cli.main import main


def _line(value):
    return json.dumps(value, sort_keys=True) + "\n"


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    episode = run / "data" / "episodes" / "episode-0000"
    (episode / ".checkpoints").mkdir(parents=True)
    events = []
    for index, focal in enumerate(("agent_001", "agent_002")):
        message = {
            "message_id": f"m{index + 1}",
            "author_id": focal,
            "author_kind": "agent",
            "message_type": "REQUEST" if index == 0 else "REPORT",
            "text": f"message {index + 1}",
            "shared_fact_id": None if index == 0 else "f2",
            "reply_to": None if index == 0 else "m1",
            "round_created": 0,
            "micro_step_created": index + 1,
            "expires_after_round": 0,
        }
        event = {
            "round_index": 0,
            "within_round_index": index,
            "global_update_index": index,
            "interaction_index": index + 1,
            "focal_agent_id": focal,
            "population_state_before": ["A", "B"],
            "population_state_after": ["A", "A"] if index else ["A", "B"],
            "focal_active_fact_ids_after": [f"f{index + 1}"],
            "focal_known_fact_ids_after": [f"f{index + 1}"],
            "new_message": message,
            "new_message_type": message["message_type"],
            "correct_answer": "A",
            "new_peer_fact_ids": ["f2"] if index else [],
            "reactivated_peer_fact_ids": [],
        }
        events.append(
            {"interaction_id": f"interaction-{index + 1:04d}", "event": event}
        )
    (episode / "trajectory.jsonl").write_text(
        "".join(_line(row) for row in events), encoding="utf-8"
    )
    round_record = {
        "round_index": 0,
        "N": 2,
        "agent_ids": ["agent_001", "agent_002"],
        "population_state_before": ["A", "B"],
        "population_state_after": ["A", "A"],
        "initial_active_fact_ids_by_agent": [["f1"], ["f2"]],
        "initial_known_fact_ids_by_agent": [["f1"], ["f2"]],
        "active_fact_ids_by_agent_after": {"agent_001": ["f1"], "agent_002": ["f2"]},
        "known_fact_ids_by_agent_after": {"agent_001": ["f1"], "agent_002": ["f2"]},
        "controller_enabled": True,
        "controller_action": "NO_OP",
        "controller_sensor_Y": {"sample_size": 1},
    }
    (episode / "round_trajectory.jsonl").write_text(
        _line(round_record), encoding="utf-8"
    )
    audits = []
    for index, focal in enumerate(("agent_001", "agent_002")):
        audits.append(
            {
                "agent_id": focal,
                "attempt": 1,
                "decision_stage": "focal_update",
                "interaction_id": f"interaction-{index + 1:04d}",
                "compiled_messages": [{"role": "user", "content": f"prompt {index}"}],
                "observation": {"visible_state": {"current_vote": "A"}},
                "response": {
                    "content": json.dumps({"vote": "A", "private_reason": "because"})
                },
                "valid": True,
            }
        )
    (episode / "audit_traces.jsonl").write_text(
        "".join(_line(row) for row in audits), encoding="utf-8"
    )
    (episode / "api_call_status.jsonl").write_text(
        "".join(_line({"valid": True}) for _ in audits), encoding="utf-8"
    )
    checkpoint = {
        "state": {
            "blackboard": [row["event"]["new_message"] for row in events],
            "task": {
                "correct_answer": "A",
                "supporting_fact_groups": {"latent_1": ["f1"], "latent_2": ["f2"]},
                "facts": {"f1": {"text": "fact one"}, "f2": {"text": "fact two"}},
            },
        }
    }
    (episode / ".checkpoints" / "checkpoint.json").write_text(
        json.dumps(checkpoint), encoding="utf-8"
    )
    (episode / "manifest.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    return run


def test_jsonl_ignores_only_a_partial_trailing_record(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"ok": 1}\n{"partial":', encoding="utf-8")
    assert _jsonl(path) == [{"ok": 1}]


def test_snapshot_reconstructs_cursor_board_coverage_and_agent(tmp_path: Path):
    reader = BlackboardRunReader(_run(tmp_path))
    first = reader.snapshot(0, 1, "agent_001")
    assert first["run"]["completed_updates"] == 2
    assert first["cursor"]["state_semantics"] == "after_selected_update"
    assert first["population"]["vote_counts"] == {"A": 1, "B": 1}
    assert [message["message_id"] for message in first["blackboard"]] == ["m1"]
    assert first["agent"]["parsed_response"]["private_reason"] == "because"
    assert first["agent"]["audit_index"] == 0
    final = reader.snapshot(0, 2, "agent_002")
    assert final["population"]["truth_vote_share"] == 1.0
    assert final["population"]["exact_acquisitions"] == 1
    assert final["agent"]["active_latent_ids"] == ["latent_2"]
    assert final["blackboard"][1]["reply_to"] == "m1"


def test_dashboard_http_api_and_export(tmp_path: Path):
    run = _run(tmp_path)
    reader = BlackboardRunReader(run)
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(reader))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        opener = build_opener(ProxyHandler({}))
        with opener.open(
            base + "/api/snapshot?round=0&step=2&agent=agent_002"
        ) as response:
            payload = json.load(response)
        assert payload["agent"]["agent_id"] == "agent_002"
        with opener.open(base + "/") as response:
            assert b"Blackboard Observatory" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    destination = tmp_path / "export"
    assert (
        main(
            [
                "blackboard",
                "export",
                "--run-dir",
                str(run),
                "--output-dir",
                str(destination),
            ]
        )
        == 0
    )
    assert "dashboard-data" in (destination / "index.html").read_text(encoding="utf-8")
    assert (destination / "app.js").is_file()


def test_dashboard_refuses_non_local_binding(tmp_path: Path):
    with pytest.raises(ValueError, match="localhost"):
        serve_dashboard(_run(tmp_path), host="0.0.0.0", port=0)


def test_modern_dawn_persistence_is_applied_before_first_day_update(tmp_path: Path):
    run = _run(tmp_path)
    episode = run / "data" / "episodes" / "episode-0000"
    first_round = json.loads((episode / "round_trajectory.jsonl").read_text())
    first_round["protocol"] = "night_dawn_autonomous_day_v1"
    first_round["controller_timing"] = "dawn_only"
    second_round = {
        **first_round,
        "round_index": 1,
        "population_state_before": ["A", "A"],
        "population_state_after": ["A", "A"],
        "persistence_deactivated_pairs": [{"agent_id": "agent_002", "fact_id": "f2"}],
    }
    (episode / "round_trajectory.jsonl").write_text(
        _line(first_round) + _line(second_round), encoding="utf-8"
    )
    event = {
        "round_index": 1,
        "within_round_index": 0,
        "global_update_index": 2,
        "interaction_index": 3,
        "focal_agent_id": "agent_001",
        "population_state_after": ["A", "A"],
        "focal_active_fact_ids_after": ["f1"],
        "focal_known_fact_ids_after": ["f1"],
        "correct_answer": "A",
    }
    with (episode / "trajectory.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(_line({"interaction_id": "interaction-0003", "event": event}))
    snapshot = BlackboardRunReader(run).snapshot(1, 1, "agent_002")
    assert snapshot["source"]["protocol"] == "night_dawn_autonomous_day_v1"
    assert snapshot["agent"]["active_fact_ids"] == []
    assert snapshot["agent"]["historical_fact_ids"] == ["f2"]


def test_dashboard_preserves_all_validation_attempts(tmp_path: Path):
    run = _run(tmp_path)
    episode = run / "data" / "episodes" / "episode-0000"
    audits = [
        {
            "agent_id": "agent_002",
            "attempt": 1,
            "decision_stage": "focal_update",
            "interaction_id": "interaction-0002",
            "response": {"content": "bad"},
            "valid": False,
            "validation_error": "invalid JSON",
        },
        {
            "agent_id": "agent_002",
            "attempt": 2,
            "decision_stage": "focal_update",
            "interaction_id": "interaction-0002",
            "response": {"content": '{"vote":"A"}'},
            "valid": True,
        },
    ]
    (episode / "audit_traces.jsonl").write_text(
        "".join(_line(row) for row in audits), encoding="utf-8"
    )
    history = BlackboardRunReader(run).snapshot(0, 2, "agent_002")["agent"][
        "attempt_history"
    ]
    assert [row["valid"] for row in history] == [False, True]
    assert history[0]["validation_error"] == "invalid JSON"


def test_completed_unknown_schema_and_partial_tail_fail_clearly(tmp_path: Path):
    run = _run(tmp_path)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 999,
                "game_type": "relational_imitation_round_feedback",
                "artifact_profile": "full",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported dashboard run schema"):
        BlackboardRunReader(run).timeline()

    run = _run(tmp_path / "partial")
    episode = run / "data" / "episodes" / "episode-0000"
    with (episode / "trajectory.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"partial":')
    with pytest.raises(ValueError, match="partial trailing record"):
        BlackboardRunReader(run).timeline()
