from __future__ import annotations

import csv
import json
import math
import subprocess
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
from mas_cc.studies.submission import (
    array_task_command,
    build_submission_entries,
    resolve_array_entry,
    submit_study,
    write_submission_manifest,
)


def _standalone_config(path: Path, *, name: str = "study-smoke") -> Path:
    config = load_run_config("configs/runs/old/toy_game_smoke_test.yaml", environment={})
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
        "--config", str(second.resolve()), "--output-dir", selected.output_dir, "--no-progress"
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

    monkeypatch.setattr("mas_cc.cli.experiment.run_experiment_preflight", fake_preflight)
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


def test_aggregate_writes_canonical_package_and_reuses_cache(tmp_path):
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
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    expected = {
        "cells.parquet", "episodes.parquet", "rounds.parquet", "micro_slots.parquet",
        "primary_estimates.parquet", "information_estimates.parquet",
        "information_nulls.parquet", "support_diagnostics.parquet",
        "derived_observables.parquet",
    }
    tables = study_dir / "analysis" / "tables"
    assert expected <= {path.name for path in tables.glob("*.parquet")}
    cell_table = pd.read_parquet(tables / "cells.parquet")
    assert len(cell_table) == 1
    assert cell_table.iloc[0]["cell_id"] == "config-0000/run"
    assert len(pd.read_parquet(tables / "episodes.parquet")) == 1
    assert set(pd.read_parquet(tables / "rounds.parquet")["cell_id"]) == {
        "config-0000/run"
    }
    assert Path(first["archive"]).is_file()

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


def test_eta_ir_requires_identical_grouping_and_conditioning_resolution():
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
        SimpleNamespace(cell_id="cell", U_k="ADVOCATE_TARGET"),
        SimpleNamespace(cell_id="cell", U_k="NO_OP"),
    ]
    mismatched = pd.DataFrame(
        [
            estimate("round_target_actuation_cmi", 0.5, target_state),
            estimate("round_target_signed_actuation", 0.25, marginal_state),
            estimate("round_target_signed_response_share", 99.0, target_state),
        ]
    )
    assert _derived("study", ["eta_ir"], mismatched, events, "hash").empty

    matched = mismatched.copy()
    matched.loc[
        matched["metric"] == "round_target_signed_actuation", "conditioning_json"
    ] = target_state
    result = _derived("study", ["eta_ir"], matched, events, "hash")
    assert len(result) == 1
    expected = 2 * 0.5 * 0.5 * 0.25**2 / (math.log(2) * 0.5)
    assert result.iloc[0]["estimate"] == pytest.approx(expected)
    assert result.iloc[0]["grouping_json"] == grouping
    assert result.iloc[0]["conditioning_json"] == target_state
    dependencies = json.loads(result.iloc[0]["dependencies_json"])
    assert dependencies["metrics"][1] == "round_target_signed_actuation"


def test_study06_results_only_retains_requested_round_and_micro_fields(tmp_path):
    study_root = Path(
        "configs/runs/relational_reasoning/population_study_06"
    )
    study = discover_study(study_root)
    assert [path.name for path in study.configs] == [
        "study06_main_b_theta.yaml",
        "study06_beta_ablation.yaml",
    ]

    environment = {
        "POTSDAM_API_KEY": "unused-smoke-value",
        "BASE_POTSDAM_LLM_URL": "https://unused.invalid",
    }
    specs = [load_run_config_or_grid(path, environment=environment) for path in study.configs]
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
        execution=replace(
            main.execution, repetitions=1, parallelism=1, fail_fast=True
        ),
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

    round_paths = sorted((tmp_path / "study06-retention").rglob("round_trajectory.jsonl"))
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
