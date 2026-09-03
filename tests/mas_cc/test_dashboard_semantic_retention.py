from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mas_cc.blackboard_dashboard.data import BlackboardRunReader
from mas_cc.config import (
    ARTIFACT_PROFILES,
    RetentionPolicy,
    load_run_config,
    parse_run_config,
)
from mas_cc.storage.dashboard_semantic import (
    SemanticDashboardWriter,
    read_semantic_stream,
    validate_semantic_stream,
)
from mas_cc.observability import DetailedAuditPolicy, RunRecorder
from mas_cc.planning.semantic_storage import estimate_semantic_storage
from mas_cc.studies.identity import protocol_fingerprint
from mas_cc.storage import ScientificIdentity


def _identity() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "cell_id": "cell-0000",
        "episode_id": "cell-0000-0000",
        "episode_seed": 17,
    }


def _writer(directory: Path) -> SemanticDashboardWriter:
    return SemanticDashboardWriter(
        directory,
        identity=_identity(),
        header={
            "game_type": "relational_imitation_round_feedback",
            "protocol_version": "night_dawn_autonomous_day_v1",
            "resolved_config_hash": "config",
            "prompt_definition_hashes_hash": "prompts",
            "task_id": "task_001",
            "population_size": 2,
            "rounds": 1,
            "q": 1,
            "epistemic_persistence": 0.85,
            "controller": {"mechanism": "relational_round_budgeted"},
            "expected_updates": 2,
        },
    )


def _state() -> dict:
    return {
        "task": {"correct_answer": "A", "possible_answers": ["A", "B"]},
        "agents": [
            {
                "agent_id": "agent_001",
                "committed_action": "A",
                "active_fact_ids": ["f1"],
                "known_fact_ids": ["f1"],
                "public_reason": "private text must not survive",
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


def _update(index: int, focal: str, before: list[str], after: list[str]) -> dict:
    message = {
        "message_id": f"m{index + 1}",
        "author_id": focal,
        "author_kind": "agent",
        "message_type": "REQUEST" if index == 0 else "REPORT",
        "text": f"public message {index + 1}",
        "vote": after[index],
        "shared_fact_id": None if index == 0 else "f2",
        "reply_to": None if index == 0 else "m1",
        "round_created": 0,
        "micro_step_created": index + 1,
        "expires_after_round": 0,
    }
    return {
        "round_index": 0,
        "within_round_index": index,
        "global_update_index": index,
        "interaction_index": index + 1,
        "N": len(before),
        "K": 2,
        "focal_agent_id": focal,
        "focal_vote_before": before[index],
        "focal_vote_after": after[index],
        "focal_opinion_before": before[index],
        "focal_opinion_after": after[index],
        "population_state_before": before,
        "population_state_after": after,
        "occupation_counts_before": {"A": before.count("A"), "B": before.count("B")},
        "occupation_counts_after": {"A": after.count("A"), "B": after.count("B")},
        "q_requested": 1,
        "q_effective": 1,
        "sampled_message_ids": [] if index == 0 else ["m1"],
        "sampled_message_authors": [] if index == 0 else ["agent_001"],
        "sampled_message_types": [] if index == 0 else ["REQUEST"],
        "focal_active_fact_ids_before": [f"f{index + 1}"],
        "focal_active_fact_ids_after": [f"f{index + 1}"],
        "focal_known_fact_ids_before": [f"f{index + 1}"],
        "focal_known_fact_ids_after": [f"f{index + 1}"],
        "new_peer_fact_ids": ["f2"] if index else [],
        "reactivated_peer_fact_ids": [],
        "new_message": message,
        "possible_answers": ["A", "B"],
        "correct_answer": "A",
        "controller_enabled": True,
        "controller_action": "NO_OP",
        "controller_target": "B",
    }


def _semantic_episode(tmp_path: Path, *, complete: bool = True) -> Path:
    episode = (
        tmp_path / "run" / "cells" / "cell-0000" / "round_records" / "cell-0000-0000"
    )
    writer = _writer(episode)
    writer.initialization(_state())
    writer.round_start(
        round_index=0,
        state=_state(),
        expired_message_ids=[],
        deactivated_pairs=[],
        controller={"enabled": True, "action": "NO_OP", "target": "B"},
    )
    writer.attempt(
        round_index=0,
        interaction_id="interaction-0001",
        agent_id="agent_001",
        attempt=1,
        valid=False,
        validation_issues=[type("Issue", (), {"field": "response.vote"})()],
        repair=False,
    )
    writer.attempt(
        round_index=0,
        interaction_id="interaction-0001",
        agent_id="agent_001",
        attempt=2,
        valid=True,
        validation_issues=[],
        repair=True,
    )
    writer.update(_update(0, "agent_001", ["A", "B"], ["A", "B"]))
    writer.update(_update(1, "agent_002", ["A", "B"], ["A", "A"]))
    final = _state()
    final["agents"][1]["committed_action"] = "A"
    final["blackboard"] = [
        _update(0, "agent_001", ["A", "B"], ["A", "B"])["new_message"],
        _update(1, "agent_002", ["A", "B"], ["A", "A"])["new_message"],
    ]
    writer.round_end(round_index=0, state=final)
    if complete:
        writer.finalize("completed")
    return episode


def test_profile_is_first_class_and_compact():
    assert "dashboard_semantic" in ARTIFACT_PROFILES
    policy = RetentionPolicy.for_profile("dashboard_semantic")
    assert policy.compact_scientific is True
    assert policy.semantic_dashboard is True
    assert policy.verbose_episode_history is False

    raw = load_run_config(
        "configs/runs/relational_reasoning/misselaneous/relational_blackboard_no_control_smoke.yaml",
        environment={},
    ).to_dict()
    raw["storage"]["artifact_profile"] = "dashboard_semantic"
    assert parse_run_config(raw).storage.artifact_profile == "dashboard_semantic"

    raw["game"]["options"]["social_mode"] = "peer"
    with pytest.raises(Exception, match="social_mode board"):
        parse_run_config(raw)

    raw["game"]["options"]["social_mode"] = "board"
    raw["logging"]["options"]["prompt_examples"] = {"count": 1, "scope": "cell"}
    with pytest.raises(Exception, match="must be 0 for dashboard_semantic"):
        parse_run_config(raw)


def test_semantic_stream_reconstructs_supported_dashboard_without_private_data(
    tmp_path: Path,
):
    episode = _semantic_episode(tmp_path)
    rows = validate_semantic_stream(episode)
    retained = (episode / "dashboard_semantic.jsonl").read_text(encoding="utf-8")
    for forbidden in (
        "private text must not survive",
        "compiled_messages",
        "raw_response",
        "authorization",
        "correction_message",
    ):
        assert forbidden not in retained
    assert [row["record_type"] for row in rows] == [
        "header",
        "initialization",
        "round_start",
        "validation",
        "validation",
        "update",
        "update",
        "round_end",
        "completion",
    ]

    run = tmp_path / "run"
    reader = BlackboardRunReader(run, "cell-0000-0000")
    timeline = reader.timeline()
    assert len(timeline["available_cursors"]) == 2
    first = reader.snapshot(0, 1, "agent_001")
    final = reader.snapshot(0, 2, "agent_002")
    assert first["source"]["artifact_profile"] == "dashboard_semantic"
    assert (
        first["capabilities"]["prompts"]["reason"]
        == "Not retained by dashboard_semantic profile"
    )
    assert first["population"]["vote_counts"] == {"A": 1, "B": 1}
    assert [message["message_id"] for message in final["blackboard"]] == ["m1", "m2"]
    assert final["blackboard"][1]["reply_to"] == "m1"
    assert final["population"]["truth_vote_share"] == 1.0
    assert final["agent"]["active_fact_ids"] == ["f2"]
    assert [item["valid"] for item in first["agent"]["attempt_history"]] == [
        False,
        True,
    ]


def test_live_partial_tail_and_completed_corruption_rules(tmp_path: Path):
    episode = _semantic_episode(tmp_path, complete=False)
    stream = episode / "dashboard_semantic.jsonl"
    with stream.open("a", encoding="utf-8") as output:
        output.write('{"partial":')
    assert (
        read_semantic_stream(stream, completed=False)[-1]["record_type"] == "round_end"
    )
    with pytest.raises(ValueError, match="partial trailing"):
        read_semantic_stream(stream, completed=True)

    completed = _semantic_episode(tmp_path / "completed")
    with (completed / "dashboard_semantic.jsonl").open("a", encoding="utf-8") as output:
        output.write('{"partial":')
    with pytest.raises(ValueError, match="partial trailing"):
        validate_semantic_stream(completed)


def test_semantic_profile_has_bounded_lean_artifact_set(tmp_path: Path):
    episode = _semantic_episode(tmp_path)
    files = {path.name for path in episode.iterdir() if path.is_file()}
    assert files == {"dashboard_semantic.jsonl", "dashboard_semantic_complete.json"}
    assert (episode / "dashboard_semantic.jsonl").stat().st_size > 0
    assert len(files) == 2


def test_unsealed_attempt_is_replaced_and_strict_validation_rejects_duplicate_cursor(
    tmp_path: Path,
):
    episode = _semantic_episode(tmp_path, complete=False)
    old_size = (episode / "dashboard_semantic.jsonl").stat().st_size
    replacement = _writer(episode)
    replacement.initialization(_state())
    assert (episode / "dashboard_semantic.jsonl").stat().st_size < old_size

    replacement.update(_update(0, "agent_001", ["A", "B"], ["A", "B"]))
    replacement.update(_update(0, "agent_001", ["A", "B"], ["A", "B"]))
    replacement.round_end(round_index=0, state=_state())
    with pytest.raises(ValueError, match="indices are not contiguous|duplicated"):
        replacement.finalize("completed")


def test_retention_profile_does_not_change_scientific_protocol_identity():
    raw = load_run_config(
        "configs/runs/relational_reasoning/misselaneous/relational_blackboard_no_control_smoke.yaml",
        environment={},
    ).to_dict()
    semantic = json.loads(json.dumps(raw))
    semantic["storage"]["artifact_profile"] = "dashboard_semantic"
    results = json.loads(json.dumps(raw))
    results["storage"]["artifact_profile"] = "results_only"
    assert protocol_fingerprint(semantic) == protocol_fingerprint(results)


def test_storage_estimate_scales_with_population_rounds_and_repetitions():
    raw = load_run_config(
        "configs/runs/relational_reasoning/misselaneous/relational_blackboard_no_control_smoke.yaml",
        environment={},
    ).to_dict()
    raw["storage"]["artifact_profile"] = "dashboard_semantic"
    estimate = estimate_semantic_storage(raw, 10)
    assert estimate is not None
    assert estimate.total_bytes == estimate.per_episode_bytes * 10
    assert estimate.population_size == raw["game"]["population_size"]
    assert estimate.rounds == raw["game"]["options"]["rounds"]


def test_real_recorder_writes_semantic_only_and_omits_audit_payloads(tmp_path: Path):
    identity = ScientificIdentity(
        "run-1",
        "cell-0000",
        "cell-0000-0000",
        17,
        "config",
        "prompts",
        "pricing",
        "relational_imitation_round_feedback",
        "reasoning",
        None,
        "task_001",
    )
    config = load_run_config(
        "configs/runs/relational_reasoning/misselaneous/relational_blackboard_no_control_smoke.yaml",
        environment={},
    )
    config = replace(
        config,
        storage=replace(config.storage, artifact_profile="dashboard_semantic"),
    )
    scientific = (
        tmp_path / ".resume" / identity.episode_id / "scientific_events.parquet"
    )
    recorder = RunRecorder(
        scientific.parent,
        run_id=identity.episode_id,
        resolved_config=config.to_dict(),
        policy=DetailedAuditPolicy(enabled=True, always_log_first_n_rounds=2),
        retention_policy=config.storage.retention_policy,
        scientific_identity=identity,
        scientific_path=scientific,
    )
    recorder.record_semantic_initialization(state=_state())
    event = _update(0, "agent_001", ["A", "B"], ["A", "B"])
    event["controller_action"] = None
    event["controller_enabled"] = False
    recorder.record_trajectory(record={"event": event, "decisions": []})
    semantic_dir = tmp_path / "round_records" / identity.episode_id
    retained = (semantic_dir / "dashboard_semantic.jsonl").read_text(encoding="utf-8")
    assert "private text must not survive" not in retained
    assert not (scientific.parent / "audit_traces.jsonl").exists()
    assert not (scientific.parent / "api_call_status.jsonl").exists()
    assert not (scientific.parent / "experiment.log").exists()


def test_semantic_and_full_sources_reconstruct_equal_shared_state(tmp_path: Path):
    from tests.mas_cc.test_blackboard_dashboard import _run as full_run

    full = BlackboardRunReader(full_run(tmp_path / "full")).snapshot(0, 2, "agent_002")
    semantic = BlackboardRunReader(
        _semantic_episode(tmp_path / "semantic").parents[3], "cell-0000-0000"
    ).snapshot(0, 2, "agent_002")
    assert semantic["population"]["vote_counts"] == full["population"]["vote_counts"]
    assert (
        semantic["population"]["truth_vote_share"]
        == full["population"]["truth_vote_share"]
    )
    assert [item["message_id"] for item in semantic["blackboard"]] == [
        item["message_id"] for item in full["blackboard"]
    ]
    assert semantic["agent"]["active_fact_ids"] == full["agent"]["active_fact_ids"]
    assert (
        semantic["agent"]["historical_fact_ids"] == full["agent"]["historical_fact_ids"]
    )


def test_measured_semantic_fixture_is_smaller_than_full_fixture(tmp_path: Path):
    from mas_cc.experiments import run_experiment_sync

    base = load_run_config(
        "configs/runs/relational_reasoning/misselaneous/relational_blackboard_no_control_smoke.yaml",
        environment={},
    )
    configs = {
        profile: replace(
            base,
            execution=replace(base.execution, repetitions=1, parallelism=1),
            logging=replace(
                base.logging,
                comet=False,
                options={"prompt_examples": {"count": 0, "scope": "cell"}},
            ),
            storage=replace(base.storage, artifact_profile=profile, overwrite=True),
        )
        for profile in ("full", "dashboard_semantic", "results_only")
    }
    results = {
        profile: run_experiment_sync(
            config, tmp_path / profile, resume=False, show_progress=False
        )
        for profile, config in configs.items()
    }
    full_episode = next((results["full"].output_dir / "data" / "episodes").iterdir())
    semantic_episode = next(
        (results["dashboard_semantic"].output_dir / "round_records").iterdir()
    )
    results_episode = next(
        (results["results_only"].output_dir / "round_records").iterdir()
    )
    full_rounds = [
        json.loads(line)
        for line in (full_episode / "round_trajectory.jsonl").read_text().splitlines()
    ]
    semantic_rounds = [
        json.loads(line)
        for line in (semantic_episode / "round_trajectory.jsonl")
        .read_text()
        .splitlines()
    ]
    results_rounds = [
        json.loads(line)
        for line in (results_episode / "round_trajectory.jsonl")
        .read_text()
        .splitlines()
    ]

    def scientific_rounds(rows):
        return [
            {
                key: value
                for key, value in row.items()
                if key not in {"schema_version", "run_id", "cell_id", "episode_id"}
            }
            for row in rows
        ]

    assert scientific_rounds(full_rounds) == scientific_rounds(semantic_rounds)
    assert scientific_rounds(semantic_rounds) == scientific_rounds(results_rounds)

    semantic_tree = [
        path
        for path in results["dashboard_semantic"].output_dir.rglob("*")
        if path.is_file()
    ]
    forbidden = {
        "audit_traces.jsonl",
        "api_call_status.jsonl",
        "usage_cost.jsonl",
        "budget_events.jsonl",
        "experiment.log",
        "prompt_block_traces.jsonl",
        "prompt_examples.md",
        "prompt_candidates.json.gz",
    }
    assert not forbidden.intersection(path.name for path in semantic_tree)
    retained_text = "".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in semantic_tree
        if path.suffix in {".jsonl", ".json", ".csv", ".md"}
    )
    assert "private smoke reasoning" not in retained_text
    semantic_snapshot = BlackboardRunReader(
        results["dashboard_semantic"].output_dir,
        semantic_episode.name,
    ).snapshot()
    full_snapshot = BlackboardRunReader(
        results["full"].output_dir,
        full_episode.name,
    ).snapshot()
    assert (
        semantic_snapshot["population"]["vote_counts"]
        == full_snapshot["population"]["vote_counts"]
    )
    assert (
        semantic_snapshot["population"]["truth_vote_share"]
        == full_snapshot["population"]["truth_vote_share"]
    )

    def measured(root: Path) -> tuple[int, int]:
        files = [path for path in root.rglob("*") if path.is_file()]
        return sum(path.stat().st_size for path in files), len(files)

    full_bytes, full_count = measured(full_episode)
    semantic_bytes, semantic_count = measured(semantic_episode)
    results_bytes, results_count = measured(results_episode)
    assert semantic_bytes < full_bytes
    assert semantic_count < full_count
    measurements = {
        "full": {"bytes": full_bytes, "files": full_count},
        "dashboard_semantic": {"bytes": semantic_bytes, "files": semantic_count},
        "results_only": {"bytes": results_bytes, "files": results_count},
    }
    (tmp_path / "retention_measurements.json").write_text(
        json.dumps(measurements, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
