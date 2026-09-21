from __future__ import annotations

from pathlib import Path

import pytest

from mas_cc.config import GameConfig, load_run_config_or_grid
from mas_cc.games.relational_reasoning.imitation_round_feedback.controller import (
    CONTROLLER_AUTHORING_DETERMINISTIC,
    CONTROLLER_AUTHORING_LLM,
    RelationalRoundBudgetedControl,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.state import (
    COMMUNICATION_PROFILE_FULL,
    COMMUNICATION_PROFILE_REPORT_ONLY,
    RelationalRules,
)


STUDY = Path(
    "configs/runs/relational_reasoning/blackboard_game/iclr_experiments/"
    "report_only_authored_q3_q12_chatoss_v5_test_10x10_b6_b18"
)


def _game(profile: str) -> GameConfig:
    return GameConfig(
        type="relational_imitation_round_feedback",
        population_size=4,
        horizon=1,
        options={
            "rounds": 1,
            "social_group_size": 1,
            "social_mode": "board",
            "prompt_version": 5,
            "board": {
                "communication_profile": profile,
                "allow_no_post": True,
            },
        },
    )


@pytest.mark.parametrize(
    ("profile", "participant_requests"),
    [
        (COMMUNICATION_PROFILE_REPORT_ONLY, False),
        (COMMUNICATION_PROFILE_FULL, True),
    ],
)
def test_communication_profile_is_the_single_participant_vocabulary_handle(
    profile, participant_requests
):
    rules = RelationalRules.from_config(_game(profile))
    assert rules.communication_profile == profile
    assert rules.allow_participant_requests is participant_requests


@pytest.mark.parametrize(
    "authoring", [CONTROLLER_AUTHORING_DETERMINISTIC, CONTROLLER_AUTHORING_LLM]
)
def test_controller_authoring_handle_is_validated(authoring):
    control = RelationalRoundBudgetedControl.from_options(
        {
            "target": "correct",
            "sensor_sample_size": 1,
            "threshold": 0.5,
            "beta": 4.0,
            "intervention_budget": 1,
            "controller_actuation_mode": "adaptive_communication",
            "controller_timing": "dawn_only",
            "controller_authoring": authoring,
        }
    )
    assert control.controller_authoring == authoring


@pytest.mark.parametrize(
    "legacy_field",
    [
        "controller_communication_policy",
        "controller_communication_policy_version",
        "controller_communication_fallback_policy",
        "allow_controller_requests",
        "allow_controller_directives",
    ],
)
def test_controller_authoring_rejects_legacy_communication_handles(legacy_field):
    options = {
        "target": "correct",
        "sensor_sample_size": 1,
        "threshold": 0.5,
        "beta": 4.0,
        "intervention_budget": 1,
        "controller_actuation_mode": "adaptive_communication",
        "controller_timing": "dawn_only",
        "controller_authoring": CONTROLLER_AUTHORING_LLM,
    }
    options[legacy_field] = (
        True if legacy_field.startswith("allow_") else 1
        if legacy_field.endswith("version")
        else "algorithmic_context_v1"
    )
    with pytest.raises(ValueError, match=legacy_field):
        RelationalRoundBudgetedControl.from_options(options)


def test_report_only_authored_study_uses_only_the_two_public_handles():
    for path in sorted(STUDY.glob("*control_q*.yaml")):
        source = load_run_config_or_grid(path, environment={})
        for cell in source.cells:
            board = cell.config.game.options["board"]
            assert board["communication_profile"] == COMMUNICATION_PROFILE_REPORT_ONLY
            assert "allow_participant_requests" not in board
            if cell.config.control.mechanism == "none":
                continue
            options = cell.config.control.options
            assert options["controller_authoring"] == CONTROLLER_AUTHORING_LLM
            assert "controller_communication_policy" not in options
            assert "controller_communication_fallback_policy" not in options
            assert "allow_controller_requests" not in options
            assert "allow_controller_directives" not in options
