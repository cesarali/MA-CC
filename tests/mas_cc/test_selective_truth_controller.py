"""Focused validation for matched truth control on selective MuSR tasks."""

from mas_cc.games.relational_reasoning.data import load_musr_team_allocation_task
from mas_cc.games.relational_reasoning.imitation_round_feedback.controller import (
    RelationalRoundBudgetedControl,
)


def test_selective_false_pool_can_be_reused_for_exact_truth_counterpart():
    task = load_musr_team_allocation_task(
        "results/studies/musr_truthful_selective_task_calibration_01/tasks",
        "task_003",
        population_size=24,
    )
    control = RelationalRoundBudgetedControl.from_options(
        {
            "target": "correct",
            "sensor_sample_size": 12,
            "policy": "soft_target",
            "threshold": 0.5,
            "beta": 4.0,
            "intervention_budget": 24,
            "advocacy_schedule": "soft",
            "message_mode": "recommendation_only",
            "controller_actuation_mode": "adaptive_communication",
            "controller_timing": "dawn_only",
            "allow_controller_requests": True,
            "allow_controller_directives": True,
            "controller_communication_policy": "llm_structured_v1",
            "controller_communication_policy_version": 1,
            "controller_communication_fallback_policy": "contextual_weighted_v1",
            "controller_communication_max_retries": 2,
            "controller_report_cooldown_rounds": 1,
            "controller_report_max_posts_per_fact": 3,
            "controller_report_selection_strategy": "target_preserving_v1",
        }
    )

    assert task.controller_target == "ALLOCATION_2"
    assert task.correct_relation == "ALLOCATION_0"
    control.validate_truthful_report_task(task, episode_seed=0)
