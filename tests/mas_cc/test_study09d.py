from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.control import create_control
from mas_cc.games.relational_reasoning.data import load_relational_task
from mas_cc.studies.episode_endpoints import (
    relational_persistence_refinement_tables,
    relational_persistence_truth_tables,
)
from mas_cc.studies.manifest import discover_study
from mas_cc.studies.preflight import validate_study_preflight_contract
from mas_cc.studies.submission import build_submission_entries

ROOT = Path("configs/runs/relational_reasoning/population_study_09d")
DATASET = Path(
    "src/mas_cc/relational_task_generator/relational_task_generator/datasets/"
    "n12_L3_r03_k3"
)


def test_study09d_is_exactly_the_20_cell_200_episode_refinement():
    spec = discover_study(ROOT)
    report = validate_study_preflight_contract(spec)
    entries = build_submission_entries(spec, "/tmp/test-study09d", git_commit="test")

    assert report["status"] == "permitted"
    assert report["rho_values"] == [0.7, 0.75, 0.8, 0.85, 0.9]
    assert report["b_values"] == [3, 6, 9, 12]
    assert report["population_size"] == [12]
    assert report["rounds"] == [30]
    assert report["q_values"] == [2]
    assert report["sensor_size"] == [6]
    assert report["L_values"] == [3]
    assert report["support_redundancy"] == [3]
    assert report["repetitions"] == [10]
    assert report["total_cells"] == 20
    assert report["total_episodes"] == 200
    assert len(entries) == 1
    assert entries[0].expected_cell_count == 20
    assert entries[0].expected_episode_count == 200


def test_study09d_preserves_study09c_scientific_semantics():
    source = load_run_config_or_grid(
        ROOT / "study09d_task0002_persistence_refinement.yaml"
    )
    assert isinstance(source, GridSpec)
    assert [(axis.path, list(axis.values)) for axis in source.axes] == [
        ("game.options.epistemic_persistence", [0.7, 0.75, 0.8, 0.85, 0.9]),
        ("control.options.intervention_budget", [3, 6, 9, 12]),
    ]
    task = load_relational_task(DATASET, "task_0002", population_size=12)
    assert task.correct_relation == "NORTH"
    for cell in source.cells:
        config = cell.config
        control = create_control(config.control)
        assert config.execution.repetitions == 10
        assert config.game.options["social_group_size"] == 2
        assert config.control.options["sensor_sample_size"] == 6
        assert config.control.options["message_mode"] == "recommendation_plus_fact"
        assert config.control.options["controller_evidence_strategy"] == "strategic"
        assert (
            control.resolved_target_for_task(task, config.execution.seed) == "NORTHWEST"
        )
        assert control.resolve_fact_id(task, config.execution.seed) == "f1"


def test_study09d_analysis_is_empirical_and_has_required_plots():
    recipe = yaml.safe_load((ROOT / "analysis.yaml").read_text(encoding="utf-8"))
    assert recipe["theoretical_reference"] == "none"
    assert recipe["episode_endpoints"]["classifier"] == (
        "relational_persistence_refinement_v1"
    )
    assert recipe["resampling"] == {
        "bootstrap_resamples": 1000,
        "null_permutations": 1000,
        "confidence": 0.95,
        "seed": 20260830,
    }
    assert {
        "chi_state_x_b_by_rho",
        "eta_ir_state_x_b_by_rho",
        "chi_heatmap",
        "eta_ir_cell_heatmap",
        "chi_persistence_slices",
        "eta_ir_persistence_slices",
    } <= set(recipe["plots"])


def test_study09d_contract_rejects_wrong_repetitions(tmp_path):
    copied = tmp_path / "population_study_09d"
    shutil.copytree(ROOT, copied)
    path = copied / "study09d_task0002_persistence_refinement.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("repetitions: 10", "repetitions: 9"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="repetitions must be \\[10\\]"):
        validate_study_preflight_contract(discover_study(copied))


def _round(episode: str, index: int, false_count: int) -> SimpleNamespace:
    truth_count = 12 - false_count
    return SimpleNamespace(
        cell_id="config-0000/cell-0000",
        episode_id=episode,
        round_index=index,
        event={
            "task_id": "task_0002",
            "possible_answers": ["NORTHWEST", "NORTH", "SOUTHEAST"],
            "correct_answer": "NORTH",
            "controller_target": "NORTHWEST",
            "analysis_target": "NORTHWEST",
            "epistemic_persistence": 0.8,
            "intervention_budget": 6,
            "occupation_counts_before": [false_count, truth_count, 0],
            "occupation_counts_after": [false_count, truth_count, 0],
            "controller_target_share": false_count / 12,
            "truth_vote_share": truth_count / 12,
            "active_full_proof_agent_share_after": 0.5,
            "active_mean_supporting_fact_coverage_after": 0.6,
            "historical_full_proof_agent_share_after": 1.0,
            "historical_mean_supporting_fact_coverage_after": 1.0,
            "controller_action": "ADVOCATE_Z" if index % 2 else "NO_OP",
            "controlled_position_count": 6,
            "new_peer_facts": 0,
            "new_controller_facts": 0,
            "reactivated_peer_fact_count": 0,
            "reactivated_controller_fact_count": 0,
            "persistence_deactivated_fact_count": 0,
            "knowledge_stratum_counts": [3, 3, 3, 3],
            "truth_counts_by_stratum": [3, 3, 3, 3],
        },
    )


def test_refinement_endpoints_keep_raw_counts_fractions_and_variability():
    rounds = []
    for episode, false_count in (("takeover", 8), ("truth", 2)):
        rounds.extend(_round(episode, index, false_count) for index in range(30))
    cells = pd.DataFrame(
        [
            {
                "cell_id": "config-0000/cell-0000",
                "epistemic_persistence": 0.8,
                "intervention_budget": 6,
            }
        ]
    )

    episodes, summary = relational_persistence_refinement_tables(rounds, cells)

    assert set(episodes["outcome_classification"]) == {
        "FALSE_FINAL_TAKEOVER",
        "NO_FALSE_MAJORITY",
    }
    assert set(episodes["classification_version"]) == {
        "relational_persistence_refinement_v1"
    }
    row = summary.iloc[0]
    assert row["episodes"] == 2
    assert row["false_final_takeover_count"] == 1
    assert row["false_final_takeover_fraction"] == 0.5
    assert row["any_false_majority_count"] == 1
    assert row["false_target_share_mean"] == pytest.approx(5 / 12)
    assert row["false_target_share_minimum"] == pytest.approx(2 / 12)
    assert row["false_target_share_maximum"] == pytest.approx(8 / 12)


def test_truth_aligned_endpoints_use_truth_names_and_keep_late_time_summaries():
    rounds = [_round("truth", index, 2) for index in range(30)]
    for row in rounds:
        row.event["controller_target"] = "NORTH"
        row.event["analysis_target"] = "NORTH"
    cells = pd.DataFrame(
        [
            {
                "cell_id": "config-0000/cell-0000",
                "epistemic_persistence": 0.8,
                "intervention_budget": 6,
            }
        ]
    )

    episodes, summary = relational_persistence_truth_tables(rounds, cells)

    assert "late_time_mean_controller_target_share" in episodes
    assert "late_time_mean_false_target_share" not in episodes
    assert episodes.iloc[0]["classification_version"] == (
        "relational_persistence_truth_aligned_v1"
    )
    assert summary.iloc[0]["truth_final_takeover_count"] == 1
