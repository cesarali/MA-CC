from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from mas_cc.cli.experiment import run_experiment_command
from mas_cc.config import load_run_config
from mas_cc.studies.aggregation import _expected_cell_coordinates, _information_tables
from mas_cc.studies.analysis_slurm import create_generation, prepare, submit_aggregation
from mas_cc.studies.manifest import StudySpec
from mas_cc.studies.submission import build_submission_entries, write_submission_manifest
from mas_cc.studies.table_io import write_scientific_table
from mas_cc.storage import file_sha256


def test_information_fragments_reproduce_the_in_process_group_result(tmp_path, monkeypatch):
    events = [
        SimpleNamespace(cell_id="canonical-cell", episode_id="episode-0", U_k="NO_OP"),
        SimpleNamespace(cell_id="canonical-cell", episode_id="episode-0", U_k="ACT"),
    ]

    def fake_estimator(payload):
        return (
            [
                {
                    "statistic": "round_sensing_mi",
                    "estimate": 0.25,
                    "bootstrap_ci_low": 0.1,
                    "bootstrap_ci_high": 0.4,
                    "n_rounds": 2,
                    "n_episodes": 1,
                    "number_of_actions_observed": 2,
                    "round_dual_action_state_fraction": 1.0,
                    "round_singleton_fraction": 0.0,
                    "units": "bits",
                }
            ],
            [],
        )

    monkeypatch.setattr(
        "mas_cc.studies.aggregation._run_information_group", fake_estimator
    )
    settings = {
        "bootstrap_resamples": 3,
        "null_permutations": 0,
        "confidence": 0.95,
        "seed": 11,
    }
    expected = _information_tables(
        "study", events, ("round_sensing_mi",), settings, "hash", {}, workers=1
    )
    groups = tmp_path / "groups"
    stem = hashlib.sha256(b"canonical-cell").hexdigest()
    information_path = write_scientific_table(groups, f"{stem}.information", expected[0])
    support_path = write_scientific_table(groups, f"{stem}.support", expected[1])
    (groups / f"{stem}.complete.json").write_text(
        json.dumps(
            {
                "cell_id": "canonical-cell",
                "group_hash": stem,
                "seed": 11 + int(stem[:8], 16),
                "information_sha256": file_sha256(information_path),
                "support_sha256": file_sha256(support_path),
            }
        ),
        encoding="utf-8",
    )
    actual = _information_tables(
        "study",
        events,
        ("round_sensing_mi",),
        settings,
        "hash",
        {},
        fragments_dir=groups,
    )
    pd.testing.assert_frame_equal(expected[0], actual[0])
    pd.testing.assert_frame_equal(expected[1], actual[1])


def test_information_fragments_validate_generation_hash_and_publish_final_hash(
    tmp_path, monkeypatch
):
    events = [
        SimpleNamespace(cell_id="canonical-cell", episode_id="episode-0", U_k="ACT")
    ]

    monkeypatch.setattr(
        "mas_cc.studies.aggregation._run_information_group",
        lambda payload: (
            [
                {
                    "statistic": "round_sensing_mi",
                    "estimate": 0.25,
                    "n_rounds": 1,
                    "n_episodes": 1,
                    "units": "bits",
                }
            ],
            [],
        ),
    )
    settings = {
        "bootstrap_resamples": 0,
        "null_permutations": 0,
        "confidence": 0.95,
        "seed": 11,
    }
    generation_hash = "generation-hash"
    generated = _information_tables(
        "study",
        events,
        ("round_sensing_mi",),
        settings,
        generation_hash,
        {},
    )
    groups = tmp_path / "groups"
    stem = hashlib.sha256(b"canonical-cell").hexdigest()
    information_path = write_scientific_table(
        groups, f"{stem}.information", generated[0]
    )
    support_path = write_scientific_table(groups, f"{stem}.support", generated[1])
    (groups / f"{stem}.complete.json").write_text(
        json.dumps(
            {
                "cell_id": "canonical-cell",
                "group_hash": stem,
                "seed": 11 + int(stem[:8], 16),
                "information_sha256": file_sha256(information_path),
                "support_sha256": file_sha256(support_path),
            }
        ),
        encoding="utf-8",
    )

    information, support = _information_tables(
        "study",
        events,
        ("round_sensing_mi",),
        settings,
        "final-hash",
        {},
        fragments_dir=groups,
        fragments_analysis_hash=generation_hash,
    )

    assert set(information["analysis_hash"]) == {"final-hash"}
    assert support.empty or set(support["analysis_hash"]) == {"final-hash"}


def test_expected_phase_grid_prefers_persisted_canonical_cell_identity():
    config = Path(
        "configs/runs/relational_reasoning/first_population_studies/"
        "population_study_09h/study09h_task0002_false_high_statistics_persistence.yaml"
    ).resolve()
    canonical_id = "6b8c7c2f3efb-persisted-cell-key"
    cells = pd.DataFrame(
        [
            {
                "cell_id": canonical_id,
                "source_config_index": 0,
                "source_cell_id": "cell-0000",
            }
        ]
    )
    rows = _expected_cell_coordinates(
        [SimpleNamespace(array_index=0, config_path=str(config))], cells
    )
    first = next(row for row in rows if row["source_cell_id"] == "cell-0000")
    assert first["cell_id"] == canonical_id
    assert first["canonical_observations_present"] is True
    assert first["cell_id"] != "config-0000/cell-0000"


def test_one_submission_builds_prepare_array_finalize_dependencies(tmp_path, monkeypatch):
    root = tmp_path / "study"
    root.mkdir()
    (root / "study_manifest.json").write_text("{}", encoding="utf-8")
    (root / "submission_manifest.csv").write_text("array_index\n", encoding="utf-8")
    generation = root / "analysis" / ".work" / "generation"
    generation.mkdir(parents=True)
    groups = [
        {"group_index": 0, "completion_path": str(generation / "0.complete.json")},
        {"group_index": 1, "completion_path": str(generation / "1.complete.json")},
    ]
    manifest = {
        "generation_id": "generation",
        "created_at": "2026-09-11T00:00:00Z",
        "study_id": "study",
        "study_dir": str(root),
        "groups": groups,
        "resources": {
            "task_throttle": 2,
            "cpus_per_task": 1,
            "memory_per_task": "8G",
            "time_limit": "06:00:00",
            "prepare": {"cpus": 1, "memory": "8G", "time_limit": "01:00:00"},
            "finalizer": {"cpus": 4, "memory": "16G", "time_limit": "06:00:00"},
        },
        "jobs": {},
        "progress_path": str(generation / "progress.json"),
        "final_archive": str(root / "analysis" / "study_analysis.zip"),
        "scientific_input_identity": "input",
        "analysis_recipe_hash": "recipe",
    }
    manifest_path = generation / "execution_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "mas_cc.studies.analysis_slurm.create_generation",
        lambda *args, **kwargs: (manifest_path, manifest),
    )
    monkeypatch.setattr("mas_cc.studies.analysis_slurm._valid_group", lambda group: False)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, f"Submitted batch job {40 + len(calls)}\n", ""
        )

    result = submit_aggregation(root, run=fake_run)
    assert len(calls) == 3
    assert any(item == "--dependency=afterok:41" for item in calls[1])
    assert any(item == "--array=0,1%2" for item in calls[1])
    assert any(item == "--dependency=afterok:42" for item in calls[2])
    assert "--cpus-per-task=4" in calls[2]
    assert result["jobs"] == {"prepare": "41", "array": "42", "finalizer": "43"}
    assert str(root / "logs" / "analysis-generation") in " ".join(calls[0])


def test_analysis_launcher_uses_potsdam_environment_and_avoids_nested_blas():
    script = Path("scripts/Potsdam/SLURM/run_study_analysis.job").read_text()
    assert "readonly CONDA_EXE=/home/ojedamarin/.local/share/miniforge3/bin/conda" in script
    assert "run -n MA-CC --live-stream" in script
    assert "export OMP_NUM_THREADS=1" in script


def test_generation_freezes_and_validates_canonical_inputs(tmp_path):
    config = load_run_config("configs/runs/old/toy_game_smoke_test.yaml", environment={})
    config = replace(
        config,
        experiment=replace(config.experiment, name="analysis-generation-smoke"),
        storage=replace(config.storage, artifact_profile="results_only"),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    study_dir = tmp_path / "study"
    entries = build_submission_entries(
        StudySpec("generation-smoke", tmp_path, (config_path,)),
        study_dir,
        git_commit="test",
    )
    run_experiment_command(config_path, entries[0].output_dir, show_progress=False)
    write_submission_manifest(study_dir / "submission_manifest.csv", entries)
    (study_dir / "study_manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "study_id": "generation-smoke", "analysis_recipe": None}
        ),
        encoding="utf-8",
    )

    manifest_path, manifest = create_generation(study_dir, allow_incomplete=False)
    prepare(manifest_path)
    assert manifest["groups"] == []
    assert set(manifest["canonical_inputs"]) >= {"cells", "episodes", "rounds", "micro_slots"}
    assert Path(manifest["progress_path"]).is_file()
    assert json.loads(Path(manifest["progress_path"]).read_text())["stage"] == "prepared"
