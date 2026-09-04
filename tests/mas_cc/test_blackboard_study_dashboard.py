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
from mas_cc.blackboard_dashboard.data import BlackboardRunReader
from mas_cc.blackboard_dashboard.study_data import (
    BlackboardStudyReader,
    _SchedulerReader,
    is_study_root,
)
from mas_cc.studies.execution import ExecutionEntry, write_execution_manifest
from mas_cc.studies.submission import SubmissionEntry, write_submission_manifest
from mas_cc.storage.scientific import (
    ScientificIdentity,
    empty_compact_row,
    write_completed_episode,
)
from mas_cc.storage.dashboard_semantic import SemanticDashboardWriter
from mas_cc.experiments.orchestrator import _CellPromptSampler


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
        "interaction_index": 1,
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


def test_nested_paths_and_study_selected_reader_match_direct_reader(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    paths = reader.resolved_paths("config-0000~cell-0000")
    assert paths.shard_root is not None and paths.shard_root.name == "cell-0000"
    assert paths.run_root.name == "run-1"
    assert paths.cell_root.name == "cell-0000"
    assert paths.round_records_root == paths.cell_root / "round_records"
    assert paths.full_episodes_root == paths.cell_root / "data" / "episodes"
    assert paths.resume_root == paths.cell_root / ".resume"
    assert paths.cell_summary_path == paths.cell_root / "cell_summary.json"
    assert paths.cell_seal_path == paths.cell_root / "cell_complete.json"

    qualified = "config-0000~cell-0000~episode-0000"
    selected = reader.episode_reader(qualified)
    direct = BlackboardRunReader(paths.run_root, "cell-0000-0000")
    assert selected.timeline() == direct.timeline()
    assert selected.snapshot(0, 1, "agent_001") == direct.snapshot(0, 1, "agent_001")


def test_compact_completion_counts_before_cell_seal_and_wins_over_failure(
    tmp_path: Path,
):
    root = _study(tmp_path)
    initial = BlackboardStudyReader(root, scheduler=False)
    paths = initial.resolved_paths("config-0000~cell-0000")
    identity = ScientificIdentity(
        run_id="run-1",
        cell_id="cell-0000",
        episode_id="cell-0000-0001",
        episode_seed=88,
        resolved_config_hash="episode-config-hash",
        prompt_definition_hashes_hash="prompt-hash",
        pricing_snapshot_hash="price-hash",
        game_type="relational_imitation_round_feedback",
        dynamics_mode="reasoning",
        control_mechanism="relational_round_budgeted",
        task_id="task_001",
    )
    row = empty_compact_row(identity, 1)
    write_completed_episode(
        paths.resume_root / identity.episode_id / "scientific_events.parquet",
        [row],
        identity,
        termination_reason="horizon",
        started_at="2026-09-03T10:00:00Z",
    )
    (paths.resume_root / identity.episode_id / "manifest.json").write_text(
        json.dumps(
            {
                "episode_id": identity.episode_id,
                "cell_id": identity.cell_id,
                "seed": identity.episode_seed,
                "status": "completed",
                "run_id": identity.run_id,
                "resolved_config_hash": identity.resolved_config_hash,
                "prompt_definition_hashes_hash": identity.prompt_definition_hashes_hash,
                "pricing_snapshot_hash": identity.pricing_snapshot_hash,
            }
        ),
        encoding="utf-8",
    )
    corrected = BlackboardStudyReader(root, scheduler=False).cell(
        "config-0000~cell-0000"
    )
    assert corrected["outcome_counts"]["completed"] == 2
    assert corrected["outcome_counts"]["failed"] == 0
    assert corrected["episodes"][1]["durable_status"] == "completed"
    assert not paths.cell_seal_path.exists()


def test_scheduler_does_not_turn_all_started_episodes_into_active(
    monkeypatch, tmp_path: Path
):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    monkeypatch.setattr(
        reader._scheduler,
        "snapshot",
        lambda: type(
            "Snapshot",
            (),
            {
                "tasks": {0: {"state": "running"}},
                "available": True,
                "job_id": "123",
                "refreshed_at": "now",
                "error": None,
            },
        )(),
    )
    cell = reader.cell("config-0000~cell-0000")
    assert cell["activity_counts"]["advancing"] == 0
    assert cell["activity_counts"]["started_unchanged"] == 1
    assert cell["activity_counts"]["not_started"] == 1


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
        with opener.open(
            base + "/api/study/episode/config-0000~cell-0000~episode-0000/timeline"
        ) as response:
            assert (
                json.load(response)
                == reader.episode_reader(
                    "config-0000~cell-0000~episode-0000"
                ).timeline()
            )
        with opener.open(
            base + "/api/study/episode/config-0000~cell-0000~episode-0000/detail"
        ) as response:
            detail = json.load(response)
        assert detail["timeline"]["available_cursors"]
        assert detail["snapshot"]["cursor"]["global_update_index"] == 0
        with opener.open(
            base
            + "/api/study/episode/config-0000~cell-0000~episode-0000/snapshot?round=0&step=1&agent=agent_001"
        ) as response:
            assert json.load(response) == reader.episode_reader(
                "config-0000~cell-0000~episode-0000"
            ).snapshot(0, 1, "agent_001")
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


def test_study_index_never_loads_trajectory_rows(monkeypatch, tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)

    def fail(*args, **kwargs):
        raise AssertionError("the lightweight index parsed an episode trajectory")

    monkeypatch.setattr(reader, "_rows", fail)
    payload = reader.study()
    assert payload["expected_episode_count"] == 4
    assert len(payload["cells"]) == 2


def test_study_index_cache_invalidates_when_cell_marker_changes(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    first = reader.study()
    assert first["episode_outcomes"]["failed"] == 1

    summary = reader.resolved_paths(
        "config-0000~cell-0000"
    ).cell_summary_path
    summary.write_text(json.dumps({"failures": []}), encoding="utf-8")
    second = reader.study()
    assert second["episode_outcomes"]["failed"] == 0


def test_episode_reader_cache_is_bounded(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    reader._episode_reader_limit = 1
    qualified = "config-0000~cell-0000~episode-0000"
    first = reader.episode_reader(qualified)
    assert reader.episode_reader(qualified) is first
    assert len(reader._episode_readers) == 1


def test_three_prompt_samples_are_capped_and_fall_back_by_episode(tmp_path: Path):
    cell = tmp_path / "cell-0000"
    sampler = _CellPromptSampler(3)
    common = {"rounds": 5, "agent_id": "agent_001", "update_index": 0}
    sampler.capture(cell, "episode-0000", 0, "beginning zero", metadata=common)
    sampler.capture(cell, "episode-0000", 0, "later request", metadata={**common, "update_index": 2})
    for round_index, label in ((0, "beginning one"), (2, "middle"), (4, "end")):
        sampler.capture(
            cell,
            "episode-0001",
            round_index,
            label,
            metadata={**common, "update_index": round_index},
        )
    sampler.render(cell, ["episode-0000", "episode-0001"])

    payload = json.loads((cell / "dashboard_prompt_examples.json").read_text())
    assert [item["sample_point"] for item in payload["samples"]] == [
        "beginning",
        "middle",
        "end",
    ]
    assert payload["samples"][0]["episode_id"] == "episode-0000"
    assert payload["samples"][0]["markdown"] == "beginning zero"
    assert len(payload["samples"]) == 3


def test_prompt_and_analysis_endpoints_are_lazy_and_downloads_allowlisted(
    tmp_path: Path,
):
    root = _study(tmp_path)
    reader = BlackboardStudyReader(root, scheduler=False)
    cell_root = reader.resolved_paths("config-0000~cell-0000").cell_root
    (cell_root / "dashboard_prompt_examples.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "samples": [
                    {
                        "sample_point": "beginning",
                        "episode_id": "cell-0000-0000",
                        "round_index": 0,
                        "update_index": 0,
                        "agent_id": "agent_001",
                        "markdown": "exact prompt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    analysis = root / "analysis"
    (analysis / "reports").mkdir(parents=True)
    (analysis / "validation.json").write_text(
        json.dumps({"valid": True, "complete": True}), encoding="utf-8"
    )
    (analysis / "analysis_manifest.json").write_text(
        json.dumps({"schema_version": 2, "requested_statistics": ["chi"]}),
        encoding="utf-8",
    )
    (analysis / "reports" / "summary.md").write_text("canonical", encoding="utf-8")

    assert reader.prompt_examples("config-0000~cell-0000")["available"] is True
    catalog = reader.analysis_catalog()
    assert catalog["available"] is True
    assert {item["id"] for item in catalog["artifacts"]} >= {
        "validation.json",
        "reports/summary.md",
    }
    assert reader.analysis_file("reports/summary.md").read_text() == "canonical"
    with pytest.raises(ValueError, match="allowlisted"):
        reader.analysis_file("../submission.json")


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


def test_cell_markup_separates_episode_navigation_and_trajectories():
    html = Path("src/mas_cc/blackboard_dashboard/assets/index.html").read_text()
    script = Path("src/mas_cc/blackboard_dashboard/assets/app.js").read_text()
    style = Path("src/mas_cc/blackboard_dashboard/assets/style.css").read_text()
    assert 'id="theme-toggle"' in html
    assert 'id="manual-refresh"' in html
    assert "mas-cc-dashboard-theme" in script
    assert "setInterval(" not in script
    assert "refreshCurrentView" in script
    assert ':root[data-theme="dark"]' in style
    assert 'id="cell-episodes"' in html
    assert 'id="cell-trajectories"' in html
    assert '<details id="all-parameters">' in html
    assert 'id="filter-rho"' in html
    episode_template = script.split("$('episode-table').innerHTML =", 1)[1].split(
        ";", 1
    )[0]
    assert "sparkline" not in episode_template
    assert "Truth trajectory" not in episode_template
    assert "Loading episode detail" in script
    assert "/detail`" in script
    assert "episodeCache" in script
    assert 'class="plot-grid' in script
    assert "Update ${update}:" in script
    assert 'id="round" type="range"' in html
    assert 'id="round-value"' in html


def test_study_opens_running_semantic_episode_read_only(tmp_path: Path):
    root = _study(tmp_path)
    reader = BlackboardStudyReader(root, scheduler=False)
    paths = reader.resolved_paths("config-0000~cell-0000")
    full_episode = paths.full_episodes_root / "cell-0000-0000"
    for path in sorted(full_episode.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    full_episode.rmdir()
    semantic_dir = paths.round_records_root / "cell-0000-0000"
    semantic = SemanticDashboardWriter(
        semantic_dir,
        identity={
            "run_id": "run-1",
            "cell_id": "cell-0000",
            "episode_id": "cell-0000-0000",
            "episode_seed": 77,
        },
        header={
            "game_type": "relational_imitation_round_feedback",
            "protocol_version": "night_dawn_autonomous_day_v1",
            "population_size": 2,
            "rounds": 1,
            "expected_updates": 2,
        },
    )
    semantic.initialization(
        {
            "task": {"correct_answer": "A", "possible_answers": ["A", "B"]},
            "agents": [
                {
                    "agent_id": "agent_001",
                    "committed_action": "A",
                    "active_fact_ids": ["f1"],
                    "known_fact_ids": ["f1"],
                },
                {
                    "agent_id": "agent_002",
                    "committed_action": "B",
                    "active_fact_ids": ["f2"],
                    "known_fact_ids": ["f2"],
                },
            ],
            "blackboard": [],
        }
    )
    semantic.update(
        {
            "round_index": 0,
            "within_round_index": 0,
            "global_update_index": 0,
            "interaction_index": 1,
            "focal_agent_id": "agent_001",
            "focal_vote_before": "A",
            "focal_vote_after": "A",
            "population_state_before": ["A", "B"],
            "population_state_after": ["A", "B"],
            "focal_active_fact_ids_after": ["f1"],
            "focal_known_fact_ids_after": ["f1"],
            "possible_answers": ["A", "B"],
            "correct_answer": "A",
        }
    )
    before = hashlib.sha256(
        (semantic_dir / "dashboard_semantic.jsonl").read_bytes()
    ).hexdigest()
    reader = BlackboardStudyReader(root, scheduler=False)
    status = reader.episode_status("config-0000~cell-0000~episode-0000")
    assert status["detail_available"] is True
    assert status["activity_status"] == "started_unchanged"
    first_reader = reader.episode_reader("config-0000~cell-0000~episode-0000")
    assert first_reader is reader.episode_reader("config-0000~cell-0000~episode-0000")
    assert (
        first_reader.snapshot(0, 1, "agent_001")["source"]["artifact_profile"]
        == "dashboard_semantic"
    )
    after = hashlib.sha256(
        (semantic_dir / "dashboard_semantic.jsonl").read_bytes()
    ).hexdigest()
    assert before == after
