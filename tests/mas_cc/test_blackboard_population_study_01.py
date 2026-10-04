from pathlib import Path

import pandas as pd
import yaml

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.control import create_control
from mas_cc.games.relational_reasoning.data import load_musr_team_allocation_task
from mas_cc.studies.aggregation import _blackboard_diagnostic_table
from mas_cc.studies.initialization import build_initialization_plan
from mas_cc.studies.manifest import discover_study
from mas_cc.studies.preflight import (
    run_study_preflight,
    validate_study_preflight_contract,
)
from mas_cc.studies.submission import build_submission_entries


ROOT = Path("configs/runs/relational_reasoning/blackboard_game/blackboard_1")
ASSIGNMENT = Path(
    "configs/runs/relational_reasoning/blackboard_game/artifacts/task_001_F9_N24.json"
)
DATASET = Path("results/studies/musr_symbolic_ambiguity_calibration_01/accepted_tasks")
SMOKE_ROOT = Path("configs/runs/smoke/blackboard_deepinfra")


def _grid(name: str) -> GridSpec:
    source = load_run_config_or_grid(ROOT / name)
    assert isinstance(source, GridSpec)
    return source


def test_blackboard_1_contract_is_exact_and_complete():
    spec = discover_study(ROOT)
    report = validate_study_preflight_contract(spec)
    entries = build_submission_entries(
        spec, "/tmp/musr_blackboard_population_01", git_commit="test"
    )
    assert report["status"] == "permitted"
    assert report["arm_cells"] == {
        "no_control": 3,
        "truth_control": 12,
        "false_control": 12,
    }
    assert report["rho_values"] == [0.74, 0.85, 1.0]
    assert report["b_values"] == [3, 6, 12, 24]
    assert report["total_cells"] == 27
    assert report["total_episodes"] == 270
    assert sum(entry.expected_cell_count for entry in entries) == 27
    assert sum(entry.expected_episode_count for entry in entries) == 270


def test_targets_are_semantic_and_no_control_has_no_budget_axis():
    task = load_musr_team_allocation_task(
        DATASET,
        "task_001",
        population_size=24,
        initial_information_path=ASSIGNMENT,
        initial_information_sha256="a0bd717bcca2f67f73e4aa981f292f9974f541949bf4fd2843cec47e281de45f",
    )
    assert task.semantic_answers == ("ALLOCATION_0", "ALLOCATION_1", "ALLOCATION_2")
    assert task.correct_relation == "ALLOCATION_0"
    truth = _grid("blackboard_1_truth_control.yaml")
    false = _grid("blackboard_1_false_control.yaml")
    baseline = _grid("blackboard_1_no_control.yaml")
    assert (
        create_control(truth.base.control).resolved_target_for_task(
            task, truth.base.execution.seed
        )
        == "ALLOCATION_0"
    )
    assert (
        create_control(false.base.control).resolved_target_for_task(
            task, false.base.execution.seed
        )
        == "ALLOCATION_1"
    )
    assert [(axis.path, list(axis.values)) for axis in baseline.axes] == [
        ("game.options.epistemic_persistence", [0.74, 0.85, 1.0])
    ]
    assert baseline.base.control.mechanism == "none"


def test_all_arms_share_ten_paired_initial_states(tmp_path):
    spec = discover_study(ROOT)
    plan = build_initialization_plan(spec.configs, tmp_path)
    assert len(plan) == 10
    assert len({entry.episode_seed for entry in plan}) == 10
    assert all(
        cell.config.execution.repetitions == 10
        for name in (
            "blackboard_1_no_control.yaml",
            "blackboard_1_truth_control.yaml",
            "blackboard_1_false_control.yaml",
        )
        for cell in _grid(name).cells
    )


def test_response_correction_retries_remain_enabled():
    for name in (
        "blackboard_1_no_control.yaml",
        "blackboard_1_truth_control.yaml",
        "blackboard_1_false_control.yaml",
    ):
        config = _grid(name).base
        assert config.game.options["invalid_response_retries"] == 3
        assert config.llm_provider.max_retries == 2


def test_smoke_study_preflight_without_a_design_contract_renders(tmp_path):
    result = run_study_preflight(SMOKE_ROOT, tmp_path)

    assert result.design["status"] == "not_requested"
    assert result.design["total_cells"] == 1
    assert result.design["total_episodes"] == 1
    assert result.design["provider_calls"] == {
        "lower": 24,
        "expected": 26,
        "conservative": 96,
    }
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Status: **NOT_REQUESTED**" in report
    assert "Total cells: 1" in report
    assert "Total episodes: 1" in report
    assert "Expected provider calls: 26" in report


def test_analysis_recipe_preserves_core_metrics_and_blackboard_views():
    recipe = yaml.safe_load((ROOT / "analysis.yaml").read_text(encoding="utf-8"))
    assert recipe["theoretical_reference"] == "none"
    assert recipe["blackboard_population_outputs"] is True
    assert {
        "round_sensing_mi",
        "round_target_actuation_cmi",
        "round_target_susceptibility",
    } <= set(recipe["estimators"])
    assert {"eta_ir", "controlled_current", "eta_th"} <= set(recipe["derived"])


def test_blackboard_diagnostics_use_weighted_eligible_fraction():
    rounds = pd.DataFrame(
        [
            {
                "cell_id": "c",
                "episode_id": "e1",
                "dawn_directive_count": 3,
                "controller_message_exposures": 2,
                "directive_exposed_focal_updates": 2,
                "controller_unique_readers": 2,
                "eligible_message_opportunities": 10,
                "eligible_directive_opportunities": 5,
                "request_count": 1,
                "report_count": 2,
                "new_evidence_acquisitions": 1,
                "reactivated_peer_fact_count": 1,
                "reactivated_controller_fact_count": 0,
                "directive_report_reply_count": 1,
                "directive_attributed_acquisitions": 1,
                "directive_attributed_refreshes": 0,
                "active_mean_fact_count_after": 1.5,
                "historical_mean_supporting_fact_coverage_after": 0.8,
                "active_mean_supporting_fact_coverage_after": 0.6,
                "realized_directive_exposure_fraction": 2 / 24,
            },
            {
                "cell_id": "c",
                "episode_id": "e1",
                "dawn_directive_count": 0,
                "controller_message_exposures": 0,
                "directive_exposed_focal_updates": 0,
                "controller_unique_readers": 0,
                "eligible_message_opportunities": 30,
                "eligible_directive_opportunities": 5,
                "request_count": 0,
                "report_count": 1,
                "new_evidence_acquisitions": 0,
                "reactivated_peer_fact_count": 0,
                "reactivated_controller_fact_count": 1,
                "directive_report_reply_count": 0,
                "directive_attributed_acquisitions": 0,
                "directive_attributed_refreshes": 1,
                "active_mean_fact_count_after": 1.0,
                "historical_mean_supporting_fact_coverage_after": 0.9,
                "active_mean_supporting_fact_coverage_after": 0.5,
                "realized_directive_exposure_fraction": 0.0,
            },
        ]
    )
    result = _blackboard_diagnostic_table(
        rounds, pd.DataFrame({"cell_id": ["c"]})
    ).iloc[0]
    assert result["eligible_directive_fraction"] == 0.25
    assert result["directives_posted"] == 3
    assert result["refresh_events"] == 2
