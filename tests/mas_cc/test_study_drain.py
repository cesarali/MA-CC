"""Cooperative study drain controls use only local fake scheduler state."""

import json

import pytest

from mas_cc.studies.drain import (DrainController, Drained, drain_status,
                                  request_study_drain, study_status, worker_drain)
from mas_cc.studies.execution import graceful_drain_policy
from mas_cc.studies.site import default_study_launcher


def _study(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    (root / "study_manifest.json").write_text(json.dumps({"study_id": "sample"}))
    controller = DrainController(root, job_id="123", attempt="attempt-a", array_index=2)
    (root / "submission.json").write_text(json.dumps({
        "status": "submitted", "job_id": "123", "submission_attempt": "attempt-a",
        "study_manifest_hash": controller.study_manifest_hash,
    }))
    return root, controller


def test_manual_request_is_atomic_idempotent_and_scoped(tmp_path):
    root, controller = _study(tmp_path)
    first = request_study_drain(root, "123", verify_scheduler=False)
    second = request_study_drain(root, "123", verify_scheduler=False)
    assert first == second
    assert controller.requested
    with pytest.raises(Drained, match="branch"):
        controller.raise_if_safe("branch")
    assert drain_status(root, "123")["shards"]["2"]["state"] == "draining"
    assert json.loads((root / "runtime" / "drain" / "job-123" / "state.json").read_text())[
        "phase"
    ] == "draining"
    with pytest.raises(ValueError, match="does not match"):
        request_study_drain(root, "456", verify_scheduler=False)
    later = DrainController(root, job_id="456", attempt="attempt-b", array_index=2)
    assert not later.requested


def test_stale_or_incompatible_request_is_ignored(tmp_path):
    root, controller = _study(tmp_path)
    controller.request(reason="manual", requested_by="cli")
    other = DrainController(root, job_id="123", attempt="attempt-b", array_index=2)
    assert not other.requested
    with pytest.raises(ValueError, match="another submission"):
        other.request(reason="manual", requested_by="cli")


def test_signal_lead_time_must_fit_allocation():
    assert graceful_drain_policy({"graceful_drain": {
        "enabled": True, "signal": "USR1", "lead_time": "04:00:00"
    }}, "24:00:00")["lead_seconds"] == 4 * 3600
    with pytest.raises(ValueError, match="shorter"):
        graceful_drain_policy({"graceful_drain": {
            "enabled": True, "lead_time": "24:00:00"
        }}, "24:00:00")


def test_worker_loads_attempt_without_sbatch_export(tmp_path, monkeypatch):
    root, _controller = _study(tmp_path)
    manifest = root / "execution_manifest.csv"
    manifest.write_text("array_index\n0\n")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "123")
    monkeypatch.delenv("MAS_CC_SUBMISSION_ATTEMPT", raising=False)
    with worker_drain(manifest, 0) as controller:
        assert controller is not None
        assert controller.attempt == "attempt-a"
    status = drain_status(root, "123")
    assert status["shards"]["0"]["state"] == "finished"


def test_cygnus_cluster_selects_its_generic_launchers(monkeypatch):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cygnus")
    assert str(default_study_launcher(cell_array=True)) == (
        "scripts/Cygnus/SLURM/run_study_cell_array.job"
    )
    assert str(default_study_launcher(cell_array=False)) == (
        "scripts/Cygnus/SLURM/run_config_array.job"
    )


def test_extension_manifest_enables_worker_drain(tmp_path, monkeypatch):
    root = tmp_path / "study"
    root.mkdir()
    (root / "study_manifest.json").write_text(json.dumps({"study_id": "sample"}))
    manifest = root / "extensions" / "extension-0001" / "execution_manifest.csv"
    manifest.parent.mkdir(parents=True)
    episode_plan = manifest.parent / "episode-plan.csv"
    episode_plan.write_text("submission_attempt\n1\n")
    manifest.write_text(
        "array_index,study_root,episode_plan_path\n"
        f"0,{root},{episode_plan}\n"
    )
    (manifest.parent / "execution_plan.json").write_text(json.dumps({"shard_count": 1}))
    submissions = manifest.parent / "submissions"
    submissions.mkdir()
    (submissions / "attempt-0001.json").write_text(json.dumps({
        "status": "SUBMITTED", "job_id": "123", "submission_attempt": 1,
        "study_manifest_hash": DrainController(
            root, job_id="123", attempt="1", array_index=0
        ).study_manifest_hash,
    }))
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "123")
    monkeypatch.delenv("MAS_CC_SUBMISSION_ATTEMPT", raising=False)
    with worker_drain(manifest, 0) as controller:
        assert controller is not None
        assert controller.root == root.resolve()
        assert controller.attempt == "1"
        assert controller.plan_path == manifest.parent / "execution_plan.json"


def test_manual_request_recognizes_active_extension_attempt(tmp_path):
    root, controller = _study(tmp_path)
    attempt = root / "extensions" / "extension-0001" / "submissions" / "attempt-0001.json"
    attempt.parent.mkdir(parents=True)
    attempt.write_text(json.dumps({
        "status": "SUBMITTED",
        "job_id": "456",
        "submission_attempt": 1,
        "study_manifest_hash": controller.study_manifest_hash,
    }))
    (attempt.parent.parent / "execution_plan.json").write_text(json.dumps({"shard_count": 3}))

    request = request_study_drain(root, "456", verify_scheduler=False)

    assert request["submission_attempt"] == "1"
    assert request["job_id"] == "456"
    assert study_status(root, "456")["expected_shards"] == 3
    assert study_status(attempt.parent.parent, "456")["expected_shards"] == 3
