from __future__ import annotations

from dataclasses import replace

import pytest

from mas_cc.config import load_run_config
from mas_cc.games.relational_reasoning.data import load_relational_task
from mas_cc.games.relational_reasoning.imitation_round_feedback.controller import (
    EVIDENCE_NEUTRAL,
    EVIDENCE_STRATEGIC,
    RECOMMENDATION_ONLY,
    RECOMMENDATION_PLUS_FACT,
    RelationalRoundBudgetedControl,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.prompts import (
    SOCIAL_ENVIRONMENT_VIGILANT,
    epistemic_framing,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.state import RelationalRules
from mas_cc.llm_runtime.exceptions import ConfigurationError


BASE = "configs/runs/relational_reasoning/relational_imitation_round_feedback_controlled_smoke.yaml"
DATASET = "src/mas_cc/relational_task_generator/relational_task_generator/datasets/pop24_L2_r06"


def test_receiver_axis_has_exactly_two_values_and_keeps_historical_vigilant_text():
    config = load_run_config(BASE)
    for value in ("naive", "vigilant"):
        options = {**config.game.options, "receiver_epistemic_disposition": value}
        options.pop("epistemic_prompt_class", None)
        options.pop("social_distrust", None)
        assert RelationalRules.from_config(replace(config.game, options=options)).receiver_epistemic_disposition == value
    assert epistemic_framing("vigilant") == SOCIAL_ENVIRONMENT_VIGILANT
    assert "objectives that differ" not in epistemic_framing("naive")
    with pytest.raises(ValueError, match="receiver_epistemic_disposition"):
        epistemic_framing("distributed_information")


def _control(strategy: str, *, target: str | int = "correct"):
    return RelationalRoundBudgetedControl.from_options({
        "target": target,
        "sensor_sample_size": 1,
        "intervention_budget": 1,
        "message_mode": RECOMMENDATION_PLUS_FACT,
        "controller_evidence_strategy": strategy,
    })


def test_evidence_strategy_requires_fact_bearing_message_mode():
    with pytest.raises(ConfigurationError, match="transmits no fact"):
        RelationalRoundBudgetedControl.from_options({
            "message_mode": RECOMMENDATION_ONLY,
            "controller_evidence_strategy": EVIDENCE_NEUTRAL,
        })


@pytest.mark.parametrize("task_id", ["task_0001", "task_0002", "task_0003", "task_0004"])
def test_neutral_is_real_deterministic_and_target_independent(task_id):
    task = load_relational_task(DATASET, task_id, population_size=24)
    truth = _control(EVIDENCE_NEUTRAL, target="correct").resolve_fact_id(task, 17)
    false = _control(EVIDENCE_NEUTRAL, target=2).resolve_fact_id(task, 17)
    assert truth == false == task.fact_order[0]
    assert truth in task.facts


@pytest.mark.parametrize("task_id", ["task_0001", "task_0002", "task_0003", "task_0004"])
def test_study08_tasks_are_strategically_admissible_for_truth_and_false(task_id):
    task = load_relational_task(DATASET, task_id, population_size=24)
    for target in ("correct", 2):
        selected = _control(EVIDENCE_STRATEGIC, target=target).resolve_fact_id(task, 17)
        assert selected in task.facts
        assert _control(EVIDENCE_STRATEGIC, target=target).resolve_fact_id(task, 17) == selected
