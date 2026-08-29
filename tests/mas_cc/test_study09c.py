from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.control import create_control
from mas_cc.games.relational_reasoning.data import load_relational_task
from mas_cc.studies.episode_endpoints import relational_persistence_tables
from mas_cc.studies.manifest import discover_study
from mas_cc.studies.preflight import validate_study_preflight_contract
from mas_cc.studies.submission import build_submission_entries


ROOT = Path("configs/runs/relational_reasoning/population_study_09c")
DATASET = Path(
    "src/mas_cc/relational_task_generator/relational_task_generator/datasets/"
    "n12_L3_r03_k3"
)


def test_study09c_is_exactly_the_twelve_episode_persistence_design():
    spec = discover_study(ROOT)
    report = validate_study_preflight_contract(spec)
    entries = build_submission_entries(spec, "/tmp/test-study09c", git_commit="test")

    assert report["status"] == "permitted"
    assert report["population_size"] == [12]
    assert report["rounds"] == [30]
    assert report["q_values"] == [2]
    assert report["sensor_size"] == [6]
    assert report["L_values"] == [3]
    assert report["support_redundancy"] == [3]
    assert report["rho_values"] == [0.6, 0.8, 0.9]
    assert report["b_values"] == [3, 6, 9, 12]
    assert report["receiver_dispositions"] == ["naive"]
    assert report["evidence_strategies"] == ["strategic"]
    assert report["message_modes"] == ["recommendation_plus_fact"]
    assert report["schedule"] == ["soft"]
    assert report["beta"] == [4.0]
    assert report["theta"] == [0.75]
    assert report["repetitions"] == [1]
    assert report["total_cells"] == report["total_episodes"] == 12
    assert report["matched_revised_theory_applicable"] is False
    assert len(entries) == 1
    assert entries[0].expected_cell_count == 12
    assert entries[0].expected_episode_count == 12


def test_study09c_uses_the_exact_frozen_task_and_false_controller_semantics():
    source = load_run_config_or_grid(ROOT / "study09c_task0002_false_persistence.yaml")
    assert isinstance(source, GridSpec)
    assert [(axis.path, list(axis.values)) for axis in source.axes] == [
        ("game.options.epistemic_persistence", [0.6, 0.8, 0.9]),
        ("control.options.intervention_budget", [3, 6, 9, 12]),
    ]

    task = load_relational_task(DATASET, "task_0002", population_size=12)
    assert task.correct_relation == "NORTH"
    for cell in source.cells:
        config = cell.config
        control = create_control(config.control)
        assert config.game.options["task_id"] == "task_0002"
        assert config.game.options["rounds"] == 30
        assert config.game.options["social_group_size"] == 2
        assert config.execution.repetitions == 1
        assert (
            control.resolved_target_for_task(task, config.execution.seed) == "NORTHWEST"
        )
        assert control.resolve_fact_id(task, config.execution.seed) == "f1"
        assert config.storage.artifact_profile == "results_only"


def test_study09c_analysis_is_empirical_and_requests_late_time_outputs():
    import yaml

    recipe = yaml.safe_load((ROOT / "analysis.yaml").read_text(encoding="utf-8"))
    assert recipe["theoretical_reference"] == "none"
    assert recipe["episode_endpoints"]["classifier"] == (
        "relational_persistence_exploratory_v1"
    )
    assert {
        "round_target_susceptibility",
        "round_target_information_fraction",
        "effective_affinity",
        "kinetic_compliance",
    } <= set(recipe["estimators"])
    assert {"eta_ir", "eta_th", "controlled_current"} <= set(recipe["derived"])
    assert {
        "false_share_trajectory",
        "active_phi_trajectory",
        "truth_share_trajectory",
        "late_time_false_share_by_budget",
        "late_time_active_phi_by_budget",
        "late_time_false_share_heatmap",
        "late_time_active_phi_heatmap",
    } <= set(recipe["plots"])


def test_study09c_contract_rejects_an_extra_rho(tmp_path):
    copied = tmp_path / "population_study_09c"
    shutil.copytree(ROOT, copied)
    path = copied / "study09c_task0002_false_persistence.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "[0.6, 0.8, 0.9]", "[0.6, 0.8, 0.9, 1.0]"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="grid axes must be rho"):
        validate_study_preflight_contract(discover_study(copied))


def _round(index: int, *, false_count: int, truth_count: int, active_phi: float):
    other = 12 - false_count - truth_count
    return SimpleNamespace(
        cell_id="config-0000/cell-0000",
        episode_id="episode-0000",
        round_index=index,
        event={
            "study_id": "study09c",
            "task_id": "task_0002",
            "possible_answers": ["NORTHWEST", "NORTH", "EAST"],
            "correct_answer": "NORTH",
            "controller_target": "NORTHWEST",
            "analysis_target": "NORTHWEST",
            "epistemic_persistence": 0.8,
            "intervention_budget": 6,
            "occupation_counts_before": [false_count, truth_count, other],
            "occupation_counts_after": [false_count, truth_count, other],
            "controller_target_share": false_count / 12,
            "truth_vote_share": truth_count / 12,
            "active_full_proof_agent_share_after": active_phi,
            "active_mean_supporting_fact_coverage_after": 0.5 + index / 100,
            "historical_full_proof_agent_share_after": min(1.0, active_phi + 0.25),
            "historical_mean_supporting_fact_coverage_after": 0.75 + index / 200,
            "controller_action": "ADVOCATE_Z" if index % 2 else "NO_OP",
            "controlled_position_count": 6 if index % 2 else 0,
            "new_peer_facts": 1,
            "new_controller_facts": 2,
            "reactivated_peer_fact_count": 3,
            "reactivated_controller_fact_count": 4,
            "persistence_deactivated_fact_count": 5,
            "knowledge_stratum_counts": [3, 3, 3, 3],
            "truth_counts_by_stratum": [1, 1, 2, 2],
        },
    )


def test_persistence_endpoint_uses_rounds_21_to_30_and_truth_proof_strata():
    rounds = [
        _round(
            index,
            false_count=2 + index % 5,
            truth_count=7 - index % 4,
            active_phi=(index % 6) / 12,
        )
        for index in range(30)
    ]
    cells = pd.DataFrame(
        [
            {
                "cell_id": "config-0000/cell-0000",
                "epistemic_persistence": 0.8,
                "intervention_budget": 6,
            }
        ]
    )

    episodes, summary = relational_persistence_tables(rounds, cells)
    row = episodes.iloc[0]

    assert row["late_time_round_start_one_based"] == 21
    assert row["late_time_round_end_one_based"] == 30
    assert row["late_time_label"] == "late-time"
    assert row["actuation_fraction"] == 0.5
    assert row["advocate_rounds"] == 15
    assert row["controlled_microscopic_updates"] == 90
    assert row["fact_acquisitions"] == 90
    assert row["fact_reactivations"] == 210
    assert row["fact_deactivations"] == 150
    assert row["final_truth_share_given_active_full_proof"] == pytest.approx(2 / 3)
    assert row["final_truth_share_given_not_active_full_proof"] == pytest.approx(4 / 9)
    assert bool(row["false_target_conditional_on_active_proof_available"]) is False
    assert len(summary) == 1
