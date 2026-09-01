from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mas_cc.config import load_run_config
from mas_cc.studies.extension import extend_study, index_existing_study, plan_extension
from mas_cc.studies.cell_worker import main as cell_worker_main
from mas_cc.studies.execution import build_cell_execution_entries, write_execution_manifest
from mas_cc.studies.identity import episode_key, protocol_fingerprint, scientific_cell_key
from mas_cc.studies.manifest import StudySpec
from mas_cc.studies.submission import build_submission_entries, write_submission_manifest


def _config(path: Path, *, repetitions: int, values=(1, 2), parallelism: int = 1) -> Path:
    config = load_run_config("configs/runs/old/toy_game_smoke_test.yaml", environment={})
    config = replace(
        config,
        execution=replace(
            config.execution,
            repetitions=repetitions,
            parallelism=parallelism,
            seed=1234,
        ),
        storage=replace(
            config.storage,
            artifact_profile="results_only",
            checkpoint_mode="episode",
            overwrite=False,
        ),
    )
    raw = config.to_dict()
    raw["experiment"]["metadata"]["common_random_numbers_across_grid"] = True
    raw["budget"]["max_cost_per_run"] = None
    raw["budget"]["system_max_cost_per_run"] = None
    raw["budget"]["max_provider_requests"] = 1000
    raw["budget"]["max_input_tokens"] = None
    raw["budget"]["max_output_tokens"] = None
    raw["grid"] = {"game.options.max_steps": list(values)}
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _legacy_study(tmp_path: Path, *, repetitions: int = 2, values=(1, 2)):
    config_dir = tmp_path / "configs-old"
    config_dir.mkdir()
    config = _config(config_dir / "grid.yaml", repetitions=repetitions, values=values)
    (config_dir / "study.yaml").write_text(
        "study: {name: extension-test}\nconfigs: [grid.yaml]\n"
        "execution: {mode: auto, target_rpm: 60, assumed_latency_seconds: 1}\n",
        encoding="utf-8",
    )
    study_dir = tmp_path / "study"
    spec = StudySpec("extension-test", config_dir, (config,))
    entries = build_submission_entries(spec, study_dir, git_commit="test")
    write_submission_manifest(study_dir / "submission_manifest.csv", entries)
    (study_dir / "study_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study_id": "extension-test",
                "config_dir": str(config_dir),
                "analysis_recipe": None,
                "execution": {"mode": "auto", "target_rpm": 60, "assumed_latency_seconds": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return study_dir, config_dir


def test_typed_cell_identity_is_order_independent_and_collision_safe():
    config = {
        "game": {"type": "toy", "options": {"x": 1}},
        "execution": {"repetitions": 2, "parallelism": 1, "seed": 7},
        "logging": {"console": True},
    }
    changed_operation = {
        **config,
        "execution": {**config["execution"], "repetitions": 99, "parallelism": 8},
        "logging": {"console": False},
    }
    first = protocol_fingerprint(config, swept_paths=("game.options.x",))
    second = protocol_fingerprint(changed_operation, swept_paths=("game.options.x",))
    assert first == second
    assert scientific_cell_key(first, {"b": 2, "a": 1}) == scientific_cell_key(
        first, {"a": 1, "b": 2}
    )
    assert scientific_cell_key(first, {"a": 1}) != scientific_cell_key(
        first, {"a": 1.0}
    )
    assert scientific_cell_key(first, {"a": 1}) != scientific_cell_key(
        first, {"a": "1"}
    )


def test_index_existing_dry_run_is_read_only(tmp_path):
    study_dir, _ = _legacy_study(tmp_path)
    report = index_existing_study(study_dir, dry_run=True)
    assert report["dry_run"] is True
    assert report["target"]["target_cell_count"] == 2
    assert not (study_dir / "study_lineage.json").exists()
    assert not (study_dir / "extensions").exists()


def test_extension_dry_run_plans_complete_new_target_without_mutation(tmp_path):
    study_dir, _ = _legacy_study(tmp_path, repetitions=2, values=(1, 2))
    target_dir = tmp_path / "configs-target"
    target_dir.mkdir()
    _config(target_dir / "grid.yaml", repetitions=4, values=(2, 1, 3))
    (target_dir / "study.yaml").write_text(
        "study: {name: extension-test}\nconfigs: [grid.yaml]\n"
        "execution: {mode: auto, target_rpm: 60, assumed_latency_seconds: 1}\n",
        encoding="utf-8",
    )

    result = extend_study(study_dir, target_dir, dry_run=True)
    assert result.plan.target_cell_count == 3
    assert result.plan.target_episode_count == 12
    assert result.plan.missing_episode_count == 12
    assert not (study_dir / "study_lineage.json").exists()


def test_real_extension_writes_exact_episode_plans_and_submits_once(tmp_path):
    study_dir, _ = _legacy_study(tmp_path, repetitions=2, values=(1, 2))
    index_existing_study(study_dir)
    target_dir = tmp_path / "configs-target"
    target_dir.mkdir()
    _config(target_dir / "grid.yaml", repetitions=4, values=(2, 1, 3))
    (target_dir / "study.yaml").write_text(
        "study: {name: extension-test}\nconfigs: [grid.yaml]\n"
        "execution: {mode: auto, target_rpm: 60, assumed_latency_seconds: 1}\n",
        encoding="utf-8",
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, "Submitted batch job 99\n", "")

    result = extend_study(study_dir, target_dir, run=fake_run)
    assert result.job_id == "99"
    assert len(calls) == 1
    assert result.plan.extension_index == 1
    extension = study_dir / "extensions" / "extension-0001"
    assert (extension / "target_manifest.json").is_file()
    assert (extension / "compatibility_report.json").is_file()
    assert (extension / "submissions" / "attempt-0000.json").is_file()
    with (extension / "execution_manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert all(row["cell_key"] for row in rows)
    assert all(row["episode_plan_path"] for row in rows)
    planned = []
    for row in rows:
        with Path(row["episode_plan_path"]).open(newline="") as stream:
            planned.extend(csv.DictReader(stream))
    assert len(planned) == 12
    assert len({row["episode_key"] for row in planned}) == 12
    assert all(
        row["episode_key"]
        == episode_key(row["cell_key"], int(row["repetition_index"]))
        for row in planned
    )


def test_completed_cells_reuse_only_missing_repetition_indices(tmp_path):
    study_dir, config_dir = _legacy_study(tmp_path, repetitions=2, values=(1, 2))
    spec = StudySpec("extension-test", config_dir, (config_dir / "grid.yaml",))
    submissions = build_submission_entries(spec, study_dir, git_commit="test")
    manifest = write_execution_manifest(
        study_dir / "execution_manifest.csv",
        build_cell_execution_entries(spec, submissions),
    )
    assert cell_worker_main([str(manifest), "0"]) == 0
    assert cell_worker_main([str(manifest), "1"]) == 0
    index_existing_study(study_dir)

    target_dir = tmp_path / "configs-target-reuse"
    target_dir.mkdir()
    _config(target_dir / "grid.yaml", repetitions=4, values=(2, 1, 3))
    (target_dir / "study.yaml").write_text(
        "study: {name: extension-test}\nconfigs: [grid.yaml]\n"
        "execution: {mode: auto, target_rpm: 60, assumed_latency_seconds: 1}\n",
        encoding="utf-8",
    )
    plan = plan_extension(study_dir, target_dir)
    assert plan.target_episode_count == 12
    assert plan.retained_episode_count == 4
    assert plan.missing_episode_count == 8
    by_coordinate = {
        tuple(cell.coordinates.values()): cell for cell in plan.target_cells
    }
    for value in (1, 2):
        key = by_coordinate[(value,)].cell_key
        assert {
            episode.repetition_index
            for episode in plan.episodes
            if episode.cell_key == key
        } == {2, 3}
    new_key = by_coordinate[(3,)].cell_key
    assert {
        episode.repetition_index
        for episode in plan.episodes
        if episode.cell_key == new_key
    } == {0, 1, 2, 3}


def test_empty_delta_publishes_target_without_sbatch(tmp_path):
    study_dir, config_dir = _legacy_study(tmp_path, repetitions=2, values=(1, 2))
    index_existing_study(study_dir)
    # No retained episodes exist, so this is not empty yet. Model the reusable
    # registry at the planning boundary to test the no-submission contract.
    import mas_cc.studies.extension as extension_module

    target = plan_extension(study_dir, config_dir)
    retained = {
        episode.episode_key: {
            "episode_seed": episode.episode_seed,
            "content_hash": "same",
        }
        for episode in target.episodes
    }
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(extension_module, "_retained_episodes", lambda *_: (retained, []))
    try:
        result = extend_study(
            study_dir,
            config_dir,
            run=lambda *args, **kwargs: pytest.fail("sbatch must not be called"),
        )
    finally:
        monkeypatch.undo()
    assert result.job_id is None
    assert result.plan.missing_episode_count == 0
    state = json.loads((result.extension_dir / "state.json").read_text())
    assert state["status"] == "COMPLETE_NO_WORK"
