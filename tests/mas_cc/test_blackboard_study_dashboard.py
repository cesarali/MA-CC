from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

import pytest
import yaml

from mas_cc.blackboard_dashboard.server import make_handler
from mas_cc.blackboard_dashboard.study_data import (
    BlackboardStudyReader,
    _SchedulerReader,
    is_study_root,
)
from mas_cc.studies.execution import ExecutionEntry, write_execution_manifest
from mas_cc.studies.submission import SubmissionEntry, write_submission_manifest


def _line(value: object) -> str:
    return json.dumps(value, sort_keys=True) + "\n"


def _study(tmp_path: Path) -> Path:
    root = tmp_path / "study"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    raw = yaml.safe_load(
        Path(
            "configs/runs/relational_reasoning/blackboard_game/blackboard_1/"
            "blackboard_1_false_control.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["llm_provider"]["type"] = "mock"
    raw["llm_provider"]["model"] = "mock-model"
    raw["llm_provider"].pop("credentials_env", None)
    raw["llm_provider"].pop("base_url_env", None)
    raw["execution"]["repetitions"] = 2
    raw["grid"] = {"game.options.epistemic_persistence": [0.74, 0.85]}
    config = config_dir / "blackboard.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    output = root / "runs" / "blackboard"
    entry = SubmissionEntry(
        array_index=0,
        config_path=str(config.resolve()),
        config_hash="config-hash",
        resolved_config_hash="resolved-hash",
        output_dir=str(output.resolve()),
        expected_cell_count=2,
        expected_episode_count=4,
        execution_seed=20260902,
        git_commit="test",
    )
    write_submission_manifest(root / "submission_manifest.csv", (entry,))
    executions = tuple(
        ExecutionEntry(
            array_index=index,
            config_index=0,
            config_path=str(config.resolve()),
            cell_index=index,
            cell_id=f"cell-{index:04d}",
            output_dir=str((output / "shards" / f"cell-{index:04d}").resolve()),
        )
        for index in range(2)
    )
    write_execution_manifest(root / "execution_manifest.csv", executions)
    (root / "study_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study_id": "dashboard-study",
                "expected_config_count": 1,
                "expected_cell_count": 2,
                "expected_episode_count": 4,
            }
        ),
        encoding="utf-8",
    )
    (root / "submission.json").write_text(
        json.dumps({"status": "submitted", "job_id": "12345"}), encoding="utf-8"
    )

    run = output / "shards" / "cell-0000" / "game" / "experiment" / "run-1"
    cell = run / "cells" / "cell-0000"
    episode = cell / "data" / "episodes" / "cell-0000-0000"
    episode.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run-1",
                "experiment_name": "experiment",
                "game_type": "relational_imitation_round_feedback",
                "artifact_profile": "full",
            }
        ),
        encoding="utf-8",
    )
    (run / "resolved_base_config.yaml").write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )
    resolved = dict(raw)
    resolved["game"] = dict(raw["game"])
    resolved["game"]["options"] = dict(raw["game"]["options"])
    resolved["game"]["options"]["epistemic_persistence"] = 0.74
    (cell / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    (cell / "overrides.json").write_text(
        json.dumps(
            {
                "cell_id": "cell-0000",
                "index": 0,
                "overrides": {"game.options.epistemic_persistence": 0.74},
            }
        ),
        encoding="utf-8",
    )
    round_row = {
        "record_type": "relational_imitation_round_feedback",
        "episode_id": "cell-0000-0000",
        "round_index": 0,
        "N": 4,
        "possible_answers": ["A", "B", "C"],
        "correct_answer": "A",
        "controller_target": "B",
        "population_state_before": ["A", "A", "B", "C"],
        "population_state_after": ["A", "B", "B", "C"],
        "agent_ids": ["agent_001", "agent_002", "agent_003", "agent_004"],
        "initial_active_fact_ids_by_agent": [[], [], [], []],
        "initial_known_fact_ids_by_agent": [[], [], [], []],
    }
    (episode / "round_trajectory.jsonl").write_text(_line(round_row), encoding="utf-8")
    (cell / "round_records" / "cell-0000-0000").mkdir(parents=True)
    (cell / "round_records" / "cell-0000-0000" / "round_trajectory.jsonl").write_text(
        _line(round_row), encoding="utf-8"
    )
    trajectory = {
        "round_index": 0,
        "within_round_index": 0,
        "global_update_index": 0,
        "focal_agent_id": "agent_001",
        "population_state_after": ["A", "B", "B", "C"],
        "correct_answer": "A",
    }
    (episode / "trajectory.jsonl").write_text(
        _line({"event": trajectory}), encoding="utf-8"
    )
    (episode / "manifest.json").write_text(
        json.dumps(
            {
                "episode_id": "cell-0000-0000",
                "status": "completed",
                "seed": 77,
                "started_at": "2026-09-03T10:00:00Z",
                "finished_at": "2026-09-03T10:00:05Z",
            }
        ),
        encoding="utf-8",
    )
    (cell / "cell_summary.json").write_text(
        json.dumps(
            {
                "failures": [
                    {"episode_id": "cell-0000-0001", "status": "failed", "seed": 88}
                ]
            }
        ),
        encoding="utf-8",
    )
    return root


def test_study_discovery_status_parameters_and_votes(tmp_path: Path):
    root = _study(tmp_path)
    reader = BlackboardStudyReader(root, scheduler=False)
    summary = reader.study()

    assert is_study_root(root)
    assert summary["study_id"] == "dashboard-study"
    assert summary["expected_cell_count"] == 2
    assert summary["discovered_cell_count"] == 1
    assert summary["episode_counts"] == {
        "pending": 2,
        "running": 0,
        "completed": 1,
        "failed": 1,
        "aborted": 0,
        "unknown": 0,
    }
    assert {cell["qualified_id"] for cell in summary["cells"]} == {
        "config-0000~cell-0000",
        "config-0000~cell-0001",
    }

    cell = reader.cell("config-0000~cell-0000")
    assert cell["parameters"]["controller_condition"] == "false_control"
    assert cell["parameters"]["rho"] == 0.74
    assert cell["parameters"]["b"] == 3
    assert cell["descriptive_mean"]["label"] == "1/2 available"
    episode = cell["episodes"][0]
    assert episode["seed"] == 77
    assert episode["elapsed_seconds"] == 5
    points = cell["vote_series"][episode["qualified_id"]]["points"]
    assert points[0]["phase"] == "initialization"
    assert points[0]["round_index"] is None
    assert points[0]["truth_share"] == 0.5
    assert points[1]["round_index"] == 0
    assert points[1]["controller_target_share"] == 0.5
    assert points[1]["option_counts"] == {"A": 1, "B": 2, "C": 1}


def test_study_api_is_batched_and_rejects_unknown_ids(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(reader))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = build_opener(ProxyHandler({}))
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with opener.open(base + "/api/study") as response:
            payload = json.load(response)
        assert len(payload["cells"]) == 2
        with opener.open(base + "/api/study/cell/config-0000~cell-0000") as response:
            assert json.load(response)["episodes"][0]["detail_available"] is True
        with opener.open(
            base + "/api/study/episode/config-0000~cell-0000~episode-0000/status"
        ) as response:
            assert json.load(response)["status"] == "completed"
        with pytest.raises(HTTPError) as error:
            opener.open(base + "/api/study/cell/..%2F..%2Fetc")
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_study_reader_is_read_only_and_rejects_completed_partial_jsonl(tmp_path: Path):
    root = _study(tmp_path)
    tracked = sorted(path for path in root.rglob("*") if path.is_file())
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    reader = BlackboardStudyReader(root, scheduler=False)
    reader.study()
    reader.cell("config-0000~cell-0000")
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    assert before == after

    trajectory = next(root.rglob("round_records/*/round_trajectory.jsonl"))
    with trajectory.open("a", encoding="utf-8") as stream:
        stream.write('{"partial":')
    reader = BlackboardStudyReader(root, scheduler=False)
    with pytest.raises(ValueError, match="partial trailing record"):
        reader.cell("config-0000~cell-0000")


def test_unrelated_layout_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="standardized study root"):
        BlackboardStudyReader(tmp_path)


def test_scheduler_annotations_are_batched_cached_and_read_only(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return type(
            "Result",
            (),
            {
                "returncode": 0,
                "stdout": "12345|0|RUNNING|node-a|00:12\n12345|1|SUSPENDED|node-b|00:03\n",
                "stderr": "",
            },
        )()

    monkeypatch.setattr(
        "mas_cc.blackboard_dashboard.study_data.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(
        "mas_cc.blackboard_dashboard.study_data.subprocess.run", fake_run
    )
    reader = _SchedulerReader("12345", ttl_seconds=30)
    first = reader.snapshot()
    second = reader.snapshot()
    assert first.tasks[0]["state"] == "running"
    assert first.tasks[1]["state"] == "held"
    assert second == first
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 3
