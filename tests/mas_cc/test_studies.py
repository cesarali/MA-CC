from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from mas_cc.cli.experiment import run_experiment_command
from mas_cc.analysis.effective_affinity import effective_affinity_analysis
from mas_cc.config import GridSpec, load_run_config, load_run_config_or_grid
from mas_cc.experiments import run_experiment_sync
from mas_cc.studies.aggregation import aggregate_study
from mas_cc.studies.aggregation import (
    _conditioning_json,
    _derived,
    _requested_statistics,
)
from mas_cc.studies.manifest import StudySpec, discover_study
from mas_cc.studies.execution import (
    build_cell_execution_entries,
    plan_cell_execution,
    read_execution_manifest,
    write_execution_manifest,
)
from mas_cc.studies.cell_worker import main as cell_worker_main
from mas_cc.studies.canonical import _coordinates
from mas_cc.studies.table_io import read_scientific_table, write_scientific_table
from mas_cc.studies.submission import (
    array_task_command,
    build_submission_entries,
    resolve_array_entry,
    submit_study,
    write_submission_manifest,
)


def _standalone_config(path: Path, *, name: str = "study-smoke") -> Path:
    config = load_run_config(
        "configs/runs/old/toy_game_smoke_test.yaml", environment={}
    )
    config = replace(
        config,
        experiment=replace(config.experiment, name=name),
        storage=replace(
            config.storage, artifact_profile="results_only", checkpoint_mode="episode"
        ),
    )
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    return path


def test_study_manifest_is_stable_and_excludes_orchestration_yaml(tmp_path):
    (tmp_path / "b.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (tmp_path / "a.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (tmp_path / "analysis.yaml").write_text("version: 1\n", encoding="utf-8")
    discovered = discover_study(tmp_path)
    assert [path.name for path in discovered.configs] == ["a.yaml", "b.yaml"]

    (tmp_path / "study.yaml").write_text(
        "study:\n  name: ordered\nconfigs: [b.yaml, a.yaml]\n", encoding="utf-8"
    )
    discovered = discover_study(tmp_path)
    assert discovered.name == "ordered"
    assert [path.name for path in discovered.configs] == ["b.yaml", "a.yaml"]

    (tmp_path / "study.yaml").write_text(
        "configs: [a.yaml, a.yaml]\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        discover_study(tmp_path)


def test_controller_target_semantics_has_stable_text_type_across_matched_blocks():
    def cell(target):
        return SimpleNamespace(
            overrides={},
            resolved_config={"control": {"options": {"target": target}}},
        )

    false_coordinates = _coordinates(cell(2))
    truth_coordinates = _coordinates(cell("correct"))

    assert false_coordinates["controller_target_semantics"] == "2"
    assert truth_coordinates["controller_target_semantics"] == "correct"
    frame = pd.DataFrame([false_coordinates, truth_coordinates])
    assert frame["controller_target_semantics"].map(type).eq(str).all()


def test_submission_manifest_and_array_mapping_are_deterministic(tmp_path):
    first = _standalone_config(tmp_path / "a.yaml", name="a")
    second = _standalone_config(tmp_path / "b.yaml", name="b")
    spec = StudySpec("study", tmp_path, (first, second))
    entries = build_submission_entries(spec, tmp_path / "results", git_commit="commit")
    again = build_submission_entries(spec, tmp_path / "results", git_commit="commit")
    assert entries == again
    assert [entry.array_index for entry in entries] == [0, 1]
    assert [entry.expected_cell_count for entry in entries] == [1, 1]
    assert [entry.expected_episode_count for entry in entries] == [1, 1]

    manifest = write_submission_manifest(tmp_path / "submission.csv", entries)
    selected = resolve_array_entry(manifest, 1)
    command = array_task_command(selected)
    assert command[-5:] == (
        "--config",
        str(second.resolve()),
        "--output-dir",
        selected.output_dir,
        "--no-progress",
    )
    with pytest.raises(ValueError, match="outside"):
        resolve_array_entry(manifest, 2)


def test_submit_preflights_every_config_and_calls_sbatch_once(tmp_path, monkeypatch):
    _standalone_config(tmp_path / "a.yaml", name="a")
    _standalone_config(tmp_path / "b.yaml", name="b")
    (tmp_path / "study.yaml").write_text(
        "study:\n  name: submitted\nconfigs: [a.yaml, b.yaml]\n", encoding="utf-8"
    )
    preflights: list[Path] = []

    def fake_preflight(config, output):
        preflights.append(Path(config))
        Path(output).mkdir(parents=True)
        return SimpleNamespace(launch_status="permitted")

    monkeypatch.setattr(
        "mas_cc.cli.experiment.run_experiment_preflight", fake_preflight
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, "Submitted batch job 4242\n", "")

    result = submit_study(
        tmp_path,
        tmp_path / "results",
        throttle=2,
        job_script=Path("scripts/Potsdam/SLURM/run_config_array.job"),
        run=fake_run,
    )
    assert len(preflights) == 2
    assert len(calls) == 1
    assert calls[0][0:2] == ("sbatch", "--array=0-1%2")
    assert result.job_id == "4242"
    assert (result.study_dir / "study_manifest.json").is_file()
    with (result.study_dir / "submission_manifest.csv").open(newline="") as stream:
        assert [int(row["array_index"]) for row in csv.DictReader(stream)] == [0, 1]


def test_potsdam_study_launchers_pin_dedicated_conda_environment():
    expected_conda = "/home/ojedamarin/.local/share/miniforge3/bin/conda"
    launchers = (
        Path("scripts/Potsdam/SLURM/run_config_array.job"),
        Path("scripts/Potsdam/SLURM/run_study_cell_array.job"),
    )

    for launcher in launchers:
        script = launcher.read_text(encoding="utf-8")
        assert f"readonly CONDA_EXE={expected_conda}" in script
        assert "run -n MA-CC --live-stream" in script
        assert "\npython -m mas_cc.studies." not in script


def test_study06_auto_plan_uses_cells_and_stays_below_rpm_target():
    spec = discover_study("configs/runs/relational_reasoning/population_study_06")
    submissions = build_submission_entries(
        spec, "results/test-study06", git_commit="test"
    )
    shards = build_cell_execution_entries(spec, submissions)
    plan = plan_cell_execution(spec, len(shards))

    assert len(shards) == 156
    assert [(row.config_index, row.cell_index) for row in shards[:2]] == [
        (0, 0),
        (0, 1),
    ]
    assert shards[120].config_index == 1
    assert shards[120].cell_index == 0
    assert plan.array_throttle == 18
    assert plan.total_request_concurrency == 144
    assert plan.episode_slots_per_shard == 8
    assert plan.total_episode_slots == 144
    assert plan.estimated_rpm == 864
    assert plan.estimated_rpm <= plan.target_rpm < 1000
    assert plan.cpus_per_task == 8
    assert plan.time_limit == "04:00:00"
    assert plan.partition == "all"
    assert plan.qos == "normal"
    assert plan.provider_load_control["mode"] == "shared_adaptive"
    assert plan.provider_load_control["initial_concurrency"] == 144
    assert plan.provider_load_control["minimum_concurrency"] == 4
    assert plan.provider_load_control["maximum_concurrency"] == 144
    assert plan.provider_load_control["target_rpm"] == 900


def test_study07_is_matched_to_study06_and_uses_same_execution_protocol():
    root = Path("configs/runs/relational_reasoning/population_study_07")
    spec = discover_study(root)
    submissions = build_submission_entries(spec, "/tmp/test-study07", git_commit="test")
    shards = build_cell_execution_entries(spec, submissions)
    plan = plan_cell_execution(spec, len(shards))
    fine = load_run_config_or_grid(root / "study07_fine_beta_atlas.yaml")
    truth = load_run_config_or_grid(root / "study07_truth_aligned_b_theta.yaml")

    assert [entry.expected_cell_count for entry in submissions] == [144, 120]
    assert [entry.expected_episode_count for entry in submissions] == [1440, 1200]
    assert len(shards) == 264
    assert plan.array_throttle == 18
    assert plan.total_episode_slots == 144
    assert plan.total_request_concurrency == 144
    assert plan.estimated_rpm == 864
    assert [(axis.path, list(axis.values)) for axis in fine.axes[:2]] == [
        ("control.options.intervention_budget", [4, 8, 12, 16, 20, 24]),
        ("control.options.beta", [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]),
    ]
    assert [(axis.path, list(axis.values)) for axis in truth.axes[:2]] == [
        ("control.options.intervention_budget", [4, 8, 12, 16, 20, 24]),
        ("control.options.threshold", [0.2, 0.35, 0.5, 0.65, 0.8]),
    ]
    assert {cell.config.control.options["target"] for cell in fine.cells} == {2}
    assert {cell.config.control.options["target"] for cell in truth.cells} == {
        "correct"
    }
    assert {cell.config.execution.seed for cell in fine.cells + truth.cells} == {
        20260822
    }
    assert {cell.config.execution.repetitions for cell in fine.cells + truth.cells} == {
        10
    }
    study06_analysis = yaml.safe_load(
        Path(
            "configs/runs/relational_reasoning/population_study_06/analysis.yaml"
        ).read_text()
    )
    study07_analysis = yaml.safe_load((root / "analysis.yaml").read_text())
    for key in ("estimators", "resampling", "derived"):
        assert study07_analysis[key] == study06_analysis[key]


def test_study08_prompt_semantics_design_is_paired_and_uses_study06_topology():
    root = Path("configs/runs/relational_reasoning/population_study_08")
    spec = discover_study(root)
    submissions = build_submission_entries(spec, "/tmp/test-study08", git_commit="test")
    shards = build_cell_execution_entries(spec, submissions)
    plan = plan_cell_execution(spec, len(shards))
    wrong = load_run_config_or_grid(root / "study08_wrong_prompt_b.yaml")
    truth = load_run_config_or_grid(root / "study08_truth_prompt_b.yaml")

    expected_axes = [
        (
            "game.options.receiver_epistemic_disposition",
            ["naive", "vigilant"],
        ),
        ("control.options.controller_evidence_strategy", ["neutral", "strategic"]),
        ("control.options.intervention_budget", [4, 8, 12, 16, 20, 24]),
        ("game.options.task_id", ["task_0001", "task_0002", "task_0003", "task_0004"]),
    ]
    assert [(axis.path, list(axis.values)) for axis in wrong.axes] == expected_axes
    assert [(axis.path, list(axis.values)) for axis in truth.axes] == expected_axes
    assert [entry.expected_cell_count for entry in submissions] == [96, 96]
    assert [entry.expected_episode_count for entry in submissions] == [960, 960]
    assert len(shards) == 192
    assert {cell.config.control.options["target"] for cell in wrong.cells} == {2}
    assert {cell.config.control.options["target"] for cell in truth.cells} == {
        "correct"
    }
    assert {
        cell.config.control.options["message_mode"]
        for cell in wrong.cells + truth.cells
    } == {"recommendation_plus_fact"}
    assert {
        cell.config.experiment.metadata["common_random_numbers_across_grid"]
        for cell in wrong.cells + truth.cells
    } == {True}
    assert {cell.config.execution.seed for cell in wrong.cells + truth.cells} == {
        20260822
    }
    assert {
        cell.config.execution.repetitions for cell in wrong.cells + truth.cells
    } == {10}
    assert plan.array_throttle == 18
    assert plan.total_request_concurrency == 144
    assert plan.estimated_rpm == 864
    assert plan.cpus_per_task == 8
    assert plan.time_limit == "04:00:00"

    study06_analysis = yaml.safe_load(
        Path(
            "configs/runs/relational_reasoning/population_study_06/analysis.yaml"
        ).read_text()
    )
    study08_analysis = yaml.safe_load((root / "analysis.yaml").read_text())
    assert study08_analysis["resampling"] == study06_analysis["resampling"]
    # Study 08 keeps every Study 06 estimator so the two remain comparable, and
    # adds the single-affinity pair on top. Study 06's own recipe is left alone:
    # rewriting a finished study's estimator list would change what its already
    # published numbers mean.
    assert set(study06_analysis["estimators"]) <= set(study08_analysis["estimators"])
    assert {"round_target_susceptibility", "round_target_sensing_mi"} <= set(
        study08_analysis["estimators"]
    )
    assert study08_analysis["derived"] == [
        "round_target_susceptibility",
        "eta_ir",
        "target_sensing_information_nats",
        "controlled_current",
        "affinity_weighted_current_nats",
        "thermodynamic_control_expenditure_nats",
        "eta_th",
        "factorial_contrasts",
    ]
    assert study08_analysis["state_local"] == ["x", "x_phi", "x_kappa"]
    required_plots = {
        f"{target}_{metric}_{resolution}"
        for target in ("truth", "false")
        for metric in ("tpi", "chi", "eta")
        for resolution in ("x_b", "x_phi", "x_kappa")
    }
    assert required_plots <= set(study08_analysis["plots"])
    assert {"b24_tpi_profile", "b24_chi_profile", "b24_eta_profile"} <= set(
        study08_analysis["plots"]
    )


def test_auto_submission_writes_execution_plan_and_explicit_resources(
    tmp_path, monkeypatch
):
    source = load_run_config_or_grid(
        "configs/runs/relational_reasoning/population_study_06/study06_beta_ablation.yaml"
    )
    config = tmp_path / "grid.yaml"
    config.write_text(
        yaml.safe_dump(source.base.to_dict(), sort_keys=False)
        + "grid:\n  control.options.intervention_budget: [8, 16]\n",
        encoding="utf-8",
    )
    (tmp_path / "study.yaml").write_text(
        "study: {name: auto}\nconfigs: [grid.yaml]\n"
        "execution:\n  mode: auto\n  target_rpm: 900\n"
        "  assumed_latency_seconds: 10\n  max_active_nodes: 18\n"
        "  cpus_per_task: 8\n  memory: 8G\n  time_limit: '04:00:00'\n",
        encoding="utf-8",
    )

    def fake_preflight(config_path, output):
        Path(output).mkdir(parents=True)
        return SimpleNamespace(launch_status="permitted")

    monkeypatch.setattr(
        "mas_cc.cli.experiment.run_experiment_preflight", fake_preflight
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, "Submitted batch job 4243\n", "")

    result = submit_study(tmp_path, tmp_path / "results", run=fake_run)
    execution_rows = read_execution_manifest(
        result.study_dir / "execution_manifest.csv"
    )
    assert len(execution_rows) == 2
    assert result.execution_plan["mode"] == "cell_array"
    assert "--cpus-per-task=8" in calls[0]
    assert "--partition=all" in calls[0]
    assert "--qos=normal" in calls[0]
    assert "--nodes=1" in calls[0]
    assert "--ntasks=1" in calls[0]
    assert "--mem=8G" in calls[0]
    assert "--time=04:00:00" in calls[0]
    assert any(
        argument == f"--output={result.study_dir}/logs/slurm-%A_%a.out"
        for argument in calls[0]
    )
    assert any(
        argument == f"--error={result.study_dir}/logs/slurm-%A_%a.err"
        for argument in calls[0]
    )
    assert any(argument.startswith("--array=0-1%") for argument in calls[0])


def test_required_results_root_rejects_home_repository_destination(tmp_path):
    _standalone_config(tmp_path / "config.yaml")
    permitted = tmp_path / "work" / "results"
    (tmp_path / "study.yaml").write_text(
        "study: {name: rooted}\nconfigs: [config.yaml]\nexecution:\n"
        f"  results_root: {permitted}\n"
        f"  require_results_under: {permitted}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be stored under"):
        submit_study(
            tmp_path, tmp_path / "home-results", run=lambda *args, **kwargs: None
        )


def test_cell_shards_reconstruct_complete_scientific_cells(tmp_path):
    raw = yaml.safe_load(
        Path(
            "configs/runs/relational_reasoning/"
            "relational_imitation_round_feedback_controlled_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["storage"]["artifact_profile"] = "results_only"
    raw["storage"]["overwrite"] = False
    raw["grid"] = {"control.options.intervention_budget": [0, 1]}
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "grid.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    (config_dir / "study.yaml").write_text(
        "study: {name: shard-smoke}\nconfigs: [grid.yaml]\n"
        "execution: {mode: auto, target_rpm: 60, assumed_latency_seconds: 1}\n",
        encoding="utf-8",
    )
    analysis_recipe = config_dir / "analysis.yaml"
    analysis_recipe.write_text(
        "estimators: [round_sensing_mi]\n"
        "resampling: {bootstrap_resamples: 1, null_permutations: 1, confidence: 0.95, seed: 7}\n",
        encoding="utf-8",
    )
    spec = discover_study(config_dir)
    study_dir = tmp_path / "study"
    submissions = build_submission_entries(spec, study_dir, git_commit="test")
    write_submission_manifest(study_dir / "submission_manifest.csv", submissions)
    shards = build_cell_execution_entries(spec, submissions)
    execution_manifest = write_execution_manifest(
        study_dir / "execution_manifest.csv", shards
    )
    (study_dir / "study_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study_id": spec.name,
                "analysis_recipe": str(analysis_recipe),
                "expected_config_count": 1,
                "expected_cell_count": 2,
                "expected_episode_count": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert cell_worker_main([str(execution_manifest), "0"]) == 0
    assert cell_worker_main([str(execution_manifest), "1"]) == 0
    summary = aggregate_study(study_dir)
    assert summary["complete"] is True
    cells = pd.read_csv(study_dir / "analysis" / "tables" / "cells.csv")
    assert len(cells) == 2
    assert set(cells["source_cell_id"]) == {"cell-0000", "cell-0001"}
    analysis = study_dir / "analysis"
    information = pd.read_csv(analysis / "tables" / "information_estimates.csv")
    assert set(information["null_permutations"]) == {1}
    assert set(information["bootstrap_resamples"]) == {1}
    assert information["null_type"].notna().all()
    assert information["null_mean"].notna().all()
    assert information["null_std"].notna().all()
    assert information["p_value"].notna().all()
    assert not (analysis / "tables" / "information_nulls.parquet").exists()
    assert not (analysis / "cache").exists()
    assert not (analysis / "cell_cache").exists()
    assert not list(analysis.rglob("*.pickle"))
    assert not list((analysis / "tables").glob("*.parquet"))
    with zipfile.ZipFile(summary["archive"]) as archive:
        names = set(archive.namelist())
    assert {
        "analysis_manifest.json",
        "validation.json",
        "validation.md",
        "tables/cells.csv",
        "tables/episodes.csv",
        "tables/rounds.csv",
        "tables/micro_slots.csv",
        "tables/primary_estimates.csv",
        "tables/information_estimates.csv",
        "tables/support_diagnostics.csv",
        "tables/derived_observables.csv",
        "reports/summary.md",
        "reports/methods.md",
        "provenance/study_manifest.json",
        "provenance/submission_manifest.csv",
    } <= names
    assert not any("cache/" in name or name.endswith(".pickle") for name in names)
    assert not any(name.endswith("information_nulls.parquet") for name in names)
    assert not any(
        name.startswith("tables/") and name.endswith(".parquet") for name in names
    )

    manifest = json.loads((analysis / "analysis_manifest.json").read_text())
    assert manifest["resampling"] == {
        "bootstrap_resamples": 1,
        "confidence": 0.95,
        "null_permutations": 1,
        "seed": 7,
    }
    assert manifest["retention_contract"]["persistent_analysis_cache"] is False
    assert manifest["retention_contract"]["individual_null_draws"] is False
    assert manifest["retention_contract"]["individual_bootstrap_draws"] is False
    assert manifest["retention_contract"]["canonical_table_format"] == "csv"
    assert manifest["retention_contract"]["csv_tables"] is True
    assert manifest["retention_contract"]["parquet_tables"] is False

    before = information.sort_values(["cell_id", "metric"]).reset_index(drop=True)
    for entry in submissions:
        shutil.rmtree(entry.output_dir)
    aggregate_study(study_dir)
    after = pd.read_csv(analysis / "tables" / "information_estimates.csv")
    after = after.sort_values(["cell_id", "metric"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


def test_aggregate_writes_compact_canonical_package(tmp_path):
    config_path = _standalone_config(tmp_path / "config.yaml")
    study_dir = tmp_path / "study"
    entries = build_submission_entries(
        StudySpec("compact-study", tmp_path, (config_path,)),
        study_dir,
        git_commit="test",
    )
    run_experiment_command(config_path, entries[0].output_dir, show_progress=False)
    write_submission_manifest(study_dir / "submission_manifest.csv", entries)
    (study_dir / "study_manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "study_id": "compact-study", "analysis_recipe": None}
        )
        + "\n",
        encoding="utf-8",
    )

    first = aggregate_study(study_dir)
    second = aggregate_study(study_dir)
    assert first["complete"] is True
    expected = {
        "cells.csv",
        "episodes.csv",
        "rounds.csv",
        "micro_slots.csv",
        "primary_estimates.csv",
        "information_estimates.csv",
        "support_diagnostics.csv",
        "derived_observables.csv",
    }
    tables = study_dir / "analysis" / "tables"
    assert expected <= {path.name for path in tables.glob("*.csv")}
    assert not list(tables.glob("*.parquet"))
    cell_table = pd.read_csv(tables / "cells.csv")
    assert len(cell_table) == 1
    assert cell_table.iloc[0]["cell_id"] == "config-0000/run"
    assert len(pd.read_csv(tables / "episodes.csv")) == 1
    assert set(pd.read_csv(tables / "rounds.csv")["cell_id"]) == {"config-0000/run"}
    assert Path(first["archive"]).is_file()
    assert Path(second["archive"]).is_file()
    assert not (study_dir / "analysis" / "cache").exists()
    assert not list((study_dir / "analysis").rglob("*.pickle"))

    write_submission_manifest(
        study_dir / "submission_manifest.csv",
        (replace(entries[0], expected_episode_count=2),),
    )
    with pytest.raises(ValueError, match="validation failed"):
        aggregate_study(study_dir)
    partial = aggregate_study(study_dir, allow_incomplete=True)
    assert partial["complete"] is False
    validation = json.loads((study_dir / "analysis" / "validation.json").read_text())
    assert validation["allow_incomplete"] is True
    assert validation["valid"] is False


def test_scientific_table_csv_round_trip_and_legacy_parquet_read(tmp_path):
    frame = pd.DataFrame(
        {
            "cell_id": ["cell-1", "cell-2"],
            "target_count_before": [1, 2],
            "estimate": [0.125, math.nan],
            "supported": [True, False],
            "nested": [{"rho": 0.7, "states": [1, 2]}, None],
        }
    )
    csv_path = write_scientific_table(tmp_path, "fidelity", frame)
    restored = pd.read_csv(csv_path)
    assert list(restored.columns) == list(frame.columns)
    assert restored.loc[0, "estimate"] == pytest.approx(0.125)
    assert math.isnan(restored.loc[1, "estimate"])
    assert json.loads(restored.loc[0, "nested"]) == {"rho": 0.7, "states": [1, 2]}

    legacy = tmp_path / "legacy.parquet"
    frame.to_parquet(legacy, index=False, engine="pyarrow")
    pd.testing.assert_frame_equal(read_scientific_table(legacy), frame)


def test_canonical_record_selection_excludes_incomplete_and_retry_prefixes():
    from mas_cc.studies.canonical import _completed_unique_records

    episodes = [
        {"episode_id": "complete", "status": "completed"},
        {"episode_id": "failed", "status": "failed"},
    ]
    rows = [
        {"episode_id": "complete", "round_index": 0, "value": "old"},
        {"episode_id": "complete", "round_index": 1, "value": "old"},
        {"episode_id": "failed", "round_index": 0, "value": "partial"},
        {"episode_id": "complete", "round_index": 0, "value": "retry"},
        {"episode_id": "complete", "round_index": 1, "value": "retry"},
        {"episode_id": "complete", "round_index": 2, "value": "retry"},
    ]

    selected, diagnostics = _completed_unique_records(
        rows, episodes, coordinate_columns=("round_index",)
    )

    assert [(row["round_index"], row["value"]) for row in selected] == [
        (0, "retry"),
        (1, "retry"),
        (2, "retry"),
    ]
    assert diagnostics == {
        "input_records": 6,
        "excluded_incomplete_records": 1,
        "superseded_retry_records": 2,
        "retained_records": 3,
    }


def test_effective_affinity_reuses_transition_rate_definition():
    rows = []
    for episode in ("e0", "e1"):
        rows.extend(
            [
                {
                    "cell_id": "cell",
                    "episode_id": episode,
                    "round_controller_action": "ADVOCATE_Z",
                    "controlled_slot": True,
                    "analysis_target": "Z",
                    "focal_opinion_before": "A",
                    "focal_opinion_after": "Z",
                },
                {
                    "cell_id": "cell",
                    "episode_id": episode,
                    "round_controller_action": "ADVOCATE_Z",
                    "controlled_slot": True,
                    "analysis_target": "Z",
                    "focal_opinion_before": "A",
                    "focal_opinion_after": "A",
                },
                {
                    "cell_id": "cell",
                    "episode_id": episode,
                    "round_controller_action": "ADVOCATE_Z",
                    "controlled_slot": True,
                    "analysis_target": "Z",
                    "focal_opinion_before": "Z",
                    "focal_opinion_after": "A",
                },
            ]
        )
    cell = next(
        row
        for row in effective_affinity_analysis(rows, bootstrap_resamples=4, seed=3)
        if row["cell_id"] == "cell"
    )
    assert cell["p_plus"] == pytest.approx(0.5)
    assert cell["p_minus"] == pytest.approx(1.0)
    assert cell["effective_affinity"] == pytest.approx(math.log(0.5))
    assert cell["kinetic_compliance"] == pytest.approx(1.5)


def test_memory_phi_alias_is_target_history_plus_phi_conditioning():
    statistics = _requested_statistics(
        {"estimators": ["round_target_actuation_cmi_memory_phi"]}
    )
    assert statistics == ("round_phi_target_actuation_cmi",)

    from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
        ROUND_CONDITIONING_STATE,
    )

    requested_keys: list[str] = []
    event = SimpleNamespace(
        target_before=7,
        augmented_state=lambda key: requested_keys.append(key) or (2,),
    )
    state = ROUND_CONDITIONING_STATE[statistics[0]](event)
    assert state == (7, (2,))
    assert requested_keys == ["conditioning_phi_bin"]
    assert json.loads(_conditioning_json(statistics[0])) == {
        "state": ["target_count_before", "conditioning_phi_bin"]
    }


def _double_round_event(cell_id, episode_id, round_index, action, before, after):
    """A minimal stand-in with the attributes the derived builders read."""

    event = {
        "N": 4,
        "episode_id": episode_id,
        "target_count_before": before,
        "target_count_after": after,
        "delta_p_ctrl": (after - before) / 4,
        "sensor_sample_size": 2,
        "controller_action": action,
    }
    return SimpleNamespace(
        cell_id=cell_id,
        episode_id=episode_id,
        round_index=round_index,
        U_k=action,
        target_before=before,
        target_after=after,
        N_k=(before, 4 - before),
        event=event,
    )


def test_state_local_eta_ir_joins_on_identical_grouping_and_conditioning():
    """The state-local eta_IR(n) uses the target-FRACTION response.

    `round_target_signed_actuation` is the same difference in aligned
    magnetization and is larger by K/(K-1); squaring it inside the Pinsker
    numerator would inflate the result, so it is not an eligible dependency.
    """

    grouping = json.dumps({"cell_id": "cell"}, sort_keys=True)
    target_state = _conditioning_json("round_target_actuation_cmi")
    marginal_state = _conditioning_json("round_target_signed_response_share")

    def estimate(metric, value, conditioning):
        return {
            "study_id": "study",
            "source_run_id": "run",
            "cell_id": "cell",
            "metric": metric,
            "grouping_json": grouping,
            "conditioning_json": conditioning,
            "estimate": value,
            "support_status": "adequate",
        }

    events = [
        _double_round_event("cell", "episode", 0, "ADVOCATE_TARGET", 2, 3),
        _double_round_event("cell", "episode", 1, "NO_OP", 2, 2),
    ]
    mismatched = pd.DataFrame(
        [
            estimate("round_target_actuation_cmi", 0.5, target_state),
            estimate("round_target_susceptibility", 0.25, marginal_state),
            estimate("round_target_signed_actuation", 0.375, target_state),
        ]
    )
    state_local, _ = _derived("study", ["eta_ir"], mismatched, events, "hash")
    assert state_local[state_local["metric"] == "eta_ir_state_local"].empty

    matched = mismatched.copy()
    matched.loc[
        matched["metric"] == "round_target_susceptibility", "conditioning_json"
    ] = target_state
    result, comparison = _derived("study", ["eta_ir"], matched, events, "hash")
    # the family runs on these doubles, but a study double records no
    # controlled micro-slots, so the theory column honestly refuses
    assert bool(comparison["available"].iloc[0]) is False
    result = result[result["metric"] == "eta_ir_state_local"]
    assert len(result) == 1
    expected = 2 * 0.5 * 0.5 * 0.25**2 / (math.log(2) * 0.5)
    assert result.iloc[0]["estimate"] == pytest.approx(expected)
    assert result.iloc[0]["grouping_json"] == grouping
    assert result.iloc[0]["conditioning_json"] == target_state
    dependencies = json.loads(result.iloc[0]["dependencies_json"])
    assert dependencies["metrics"][1] == "round_target_susceptibility"
    assert dependencies["response_coordinate"] == "target_fraction"


def test_single_affinity_derived_family_is_written_by_offline_aggregation(tmp_path):
    """The acceptance run: one offline analysis, no provider calls beyond the mock.

    Checks the things a reader of the output table has to be able to trust -
    that `eta_ir` is the occupancy-weighted one and not the legacy
    magnetization-scaled value, that `eta_th` exists as a real column with its
    two terms beside it, that `cell_current` and the full-vector sensing MI are
    still present but are NOT what the thermodynamic quantities were built
    from, and that every derived row says in machine-readable form what was
    done to produce it.
    """

    raw = yaml.safe_load(
        Path(
            "configs/runs/relational_reasoning/"
            "relational_imitation_round_feedback_controlled_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["storage"]["overwrite"] = False
    raw["grid"] = {"control.options.intervention_budget": [2, 4]}
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "grid.yaml").write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )
    (config_dir / "study.yaml").write_text(
        "study: {name: single-affinity-smoke}\nconfigs: [grid.yaml]\n"
        "execution: {mode: auto, target_rpm: 60, assumed_latency_seconds: 1}\n",
        encoding="utf-8",
    )
    analysis_recipe = config_dir / "analysis.yaml"
    analysis_recipe.write_text(
        yaml.safe_dump(
            {
                "estimators": [
                    "round_sensing_mi",
                    "round_target_sensing_mi",
                    "round_target_actuation_cmi",
                    "round_target_susceptibility",
                    "round_target_signed_actuation",
                    "cell_current",
                    "effective_affinity",
                    "kinetic_compliance",
                ],
                "resampling": {
                    "bootstrap_resamples": 4,
                    "null_permutations": 1,
                    "confidence": 0.95,
                    "seed": 7,
                },
                "derived": [
                    "round_target_susceptibility",
                    "eta_ir",
                    "target_sensing_information_nats",
                    "controlled_current",
                    "affinity_weighted_current_nats",
                    "thermodynamic_control_expenditure_nats",
                    "eta_th",
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    spec = discover_study(config_dir)
    study_dir = tmp_path / "study"
    submissions = build_submission_entries(spec, study_dir, git_commit="test")
    write_submission_manifest(study_dir / "submission_manifest.csv", submissions)
    shards = build_cell_execution_entries(spec, submissions)
    execution_manifest = write_execution_manifest(
        study_dir / "execution_manifest.csv", shards
    )
    (study_dir / "study_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study_id": spec.name,
                "analysis_recipe": str(analysis_recipe),
                "expected_config_count": 1,
                "expected_cell_count": 2,
                "expected_episode_count": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert cell_worker_main([str(execution_manifest), "0"]) == 0
    assert cell_worker_main([str(execution_manifest), "1"]) == 0
    aggregate_study(study_dir)

    tables = study_dir / "analysis" / "tables"
    primary = pd.read_csv(tables / "primary_estimates.csv")
    derived = pd.read_csv(tables / "derived_observables.csv")
    diagnostics = pd.read_csv(tables / "thermodynamic_efficiency_diagnostics.csv")
    assert len(diagnostics) == 2
    assert diagnostics["cell_id"].nunique() == 2
    assert {
        "h",
        "controlled_current",
        "eta_th_signed",
        "eta_th_bounded",
        "eta_th_numeric_defined",
        "eta_th_undefined_reason",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
    } <= set(diagnostics.columns)

    # --- response ---------------------------------------------------------
    chi = primary[primary["metric"] == "round_target_susceptibility"]
    assert not chi.empty
    assert json.loads(chi.iloc[0]["conditioning_json"]) == {
        "state": ["target_count_before"]
    }
    # the legacy magnetization response is retained, not replaced
    assert not primary[primary["metric"] == "round_target_signed_actuation"].empty
    # so is the full-vector sensing MI and the terminal behavioural current
    assert not primary[primary["metric"] == "round_sensing_mi"].empty
    assert not primary[primary["metric"] == "cell_current"].empty

    # --- the derived family ----------------------------------------------
    expected = {
        "susceptibility_occupancy_weighted",
        "eta_ir",
        "eta_ir_pinsker_numerator_bits",
        "eta_ir_denominator_T_bits",
        "target_sensing_information_nats",
        "target_sensing_information_horizon_nats",
        "controlled_current",
        "controlled_current_horizon",
        "affinity_weighted_current_nats",
        "thermodynamic_control_expenditure_nats",
        "eta_th",
        "eta_th_signed",
        "eta_th_bounded",
    }
    assert expected <= set(derived["metric"])

    units = dict(zip(derived["metric"], derived["units"], strict=True))
    assert units["eta_ir_pinsker_numerator_bits"] == "bits"
    assert units["target_sensing_information_nats"] == "nats_per_cycle"
    assert units["controlled_current"] == "target_count_per_cycle"
    assert units["eta_th"] == "dimensionless"

    family = derived[derived["metric"].isin(expected)]
    assert set(family["theory_semantics_version"]) == {"single_affinity_v1"}
    assert set(family["response_coordinate"]) == {"target_fraction"}
    assert set(family["sensing_coordinate"]) == {"target_count"}
    assert set(family["affinity_log_base"]) == {"e"}
    assert set(family["bootstrap_unit"]) == {"episode"}
    assert set(family["eta_ir_aggregation"]) == {"occupancy_ratio_of_sums"}
    # support and validity travel with every row of the family
    for column in (
        "chi_dual_action_state_fraction",
        "chi_identified_occupancy_mass",
        "eta_ir_valid",
        "eta_th_valid",
        "eta_th_target_directed",
    ):
        assert column in family.columns
    assert {"ci_low", "ci_high", "confidence"} <= set(family.columns)

    # eta_ir is the occupancy-weighted headline; the state-resolved surface is
    # a separately named metric so the two can never be confused in a join.
    headline = derived[derived["metric"] == "eta_ir"]
    assert len(headline) == len(set(headline["cell_id"]))
    dependencies = json.loads(headline.iloc[0]["dependencies_json"])
    assert "round_target_susceptibility" in dependencies["metrics"]
    assert "round_target_signed_actuation" not in dependencies["metrics"]


def test_study06_results_only_retains_requested_round_and_micro_fields(tmp_path):
    study_root = Path("configs/runs/relational_reasoning/population_study_06")
    study = discover_study(study_root)
    assert [path.name for path in study.configs] == [
        "study06_main_b_theta.yaml",
        "study06_beta_ablation.yaml",
    ]

    environment = {
        "POTSDAM_API_KEY": "unused-smoke-value",
        "BASE_POTSDAM_LLM_URL": "https://unused.invalid",
    }
    specs = [
        load_run_config_or_grid(path, environment=environment) for path in study.configs
    ]
    assert all(isinstance(spec, GridSpec) for spec in specs)
    assert all(spec.base.storage.artifact_profile == "results_only" for spec in specs)

    main = specs[0].base
    mock = replace(
        main,
        llm_provider=replace(
            main.llm_provider,
            type="mock",
            model="deterministic-v1",
            credentials_env=None,
            base_url_env=None,
            request_concurrency=1,
            max_output_tokens=128,
            options={
                "response": '{"vote":"C","reason":"smoke","shared_fact_id":"none"}'
            },
        ),
        execution=replace(main.execution, repetitions=1, parallelism=1, fail_fast=True),
        logging=replace(main.logging, console=False, audit=False, comet=False),
        pricing=replace(
            main.pricing,
            mode="offline",
            require_fresh_at_launch=False,
            explicit_unknown_price_override=True,
        ),
        budget=replace(main.budget, accounting_unit="USD"),
    )
    run_experiment_sync(
        mock, tmp_path / "study06-retention", resume=False, show_progress=False
    )

    round_paths = sorted(
        (tmp_path / "study06-retention").rglob("round_trajectory.jsonl")
    )
    micro_paths = sorted(
        (tmp_path / "study06-retention").rglob("micro_slot_trajectory.jsonl")
    )
    assert round_paths and micro_paths
    round_row = json.loads(round_paths[0].read_text(encoding="utf-8").splitlines()[0])
    micro_row = json.loads(micro_paths[0].read_text(encoding="utf-8").splitlines()[0])

    assert {
        "episode_id",
        "round_index",
        "possible_answers",
        "correct_answer",
        "analysis_target",
        "occupation_counts_before",
        "occupation_counts_after",
        "controller_action",
        "controller_advocate_probability",
        "sensor_count_vector",
        "knowledge_stratum_counts_before",
        "mean_supporting_fact_coverage_before",
        "full_proof_agent_share_before",
        "delta_m_ctrl",
        "delta_m_truth",
        "delta_m_order",
    } <= round_row.keys()
    assert {
        "episode_id",
        "round_index",
        "within_round_index",
        "round_controller_action",
        "round_controller_target",
        "controlled_slot",
        "intervention_budget",
        "focal_opinion_before",
        "focal_opinion_after",
        "occupation_counts_before",
        "occupation_counts_after",
        "possible_answers",
    } <= micro_row.keys()
