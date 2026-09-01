"""MuSR Team Allocation adapter tests for the q-message game."""

from __future__ import annotations

from dataclasses import replace

from mas_cc.config import load_run_config
from mas_cc.games import create_game
from mas_cc.games.relational_reasoning.data import load_musr_team_allocation_task
from mas_cc.games.relational_reasoning.imitation_round_feedback.metrics import (
    supporting_fact_coverage,
)
from mas_cc.llm_runtime.prompts import RegexTokenCounter

CONFIG = "configs/runs/relational_reasoning/relational_blackboard_no_control_smoke.yaml"
DATASET = "results/studies/musr_team_allocation_validation_01/tasks"


def _musr_config():
    config = load_run_config(CONFIG, environment={})
    options = {
        **dict(config.game.options),
        "task_family": "musr_team_allocation",
        "task_dataset_dir": DATASET,
        "task_id": "task_001",
        "n_agents": 12,
    }
    return replace(
        config,
        game=replace(config.game, population_size=12, options=options),
    )


def test_validated_musr_task_and_n12_distribution_load_exactly():
    task = load_musr_team_allocation_task(DATASET, "task_001", population_size=12)

    assert task.task_family == "musr_team_allocation"
    assert task.correct_relation == "ALLOCATION_2"
    assert task.semantic_answers == ("ALLOCATION_0", "ALLOCATION_1", "ALLOCATION_2")
    assert len(task.agent_ids) == 12
    assert len(task.fact_order) == 27
    assert len(task.supporting_fact_groups or {}) == 9
    assert "A project lead must allocate" in task.question
    assert task.known_facts("agent_001")[0] == "e_skill_p0_t0_b00"
    assert "e_coop_p1_p2_b01" in task.known_facts("agent_001")
    assert (
        supporting_fact_coverage(
            task.known_facts("agent_001"),
            task.supporting_fact_ids,
            task.supporting_fact_groups,
        )
        == 7 / 9
    )


def test_musr_board_prompt_renders_scenario_allocations_and_no_hidden_latent_data():
    config = _musr_config()
    game = create_game(config.game)
    state = game.initialize(config.game, config.execution.seed)
    request = game.ballot_request(
        state,
        state.agents[0].agent_id,
        (),
        config.game,
    )
    prompt = "\n\n".join(
        message.content
        for message in request.prompt.compile(RegexTokenCounter()).messages
    )

    assert "A project lead must allocate" in prompt
    assert "build the data pipeline: Farah" in prompt
    assert "ALLOCATION_2" not in prompt
    assert "skill_matrix" not in prompt
    assert "cooperation_matrix" not in prompt
    assert "hidden_claim" not in prompt
