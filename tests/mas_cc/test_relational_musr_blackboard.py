"""MuSR Team Allocation adapter tests for the q-message game."""

from __future__ import annotations

from dataclasses import replace
import asyncio
import hashlib
import json
from pathlib import Path

from mas_cc.config import load_run_config
from mas_cc.games import create_game
from mas_cc.games.relational_reasoning.data import load_musr_team_allocation_task
from mas_cc.games.relational_reasoning.imitation_round_feedback.metrics import (
    supporting_fact_coverage,
)
from mas_cc.llm_runtime.prompts import RegexTokenCounter
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider
from mas_cc.games.relational_reasoning.imitation_round_feedback.runtime import (
    run_relational_imitation_round_feedback_game,
)

CONFIG = (
    "configs/runs/relational_reasoning/misselaneous/"
    "relational_blackboard_no_control_smoke.yaml"
)
DATASET = "results/studies/musr_team_allocation_validation_01/tasks"
PILOT_CONFIG = (
    "configs/runs/relational_reasoning/blackboard_game/"
    "musr_blackboard_task001_5round_simplified_messages.yaml"
)


def _musr_config():
    config = load_run_config(CONFIG, environment={})
    options = {
        **dict(config.game.options),
        "task_family": "musr_team_allocation",
        "task_dataset_dir": DATASET,
        "task_id": "task_001",
        "n_agents": 12,
        "prompt_version": 2,
    }
    return replace(
        config,
        game=replace(config.game, population_size=12, options=options),
        prompt=replace(config.prompt, prompt_version=2),
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


def test_initialization_only_asks_each_agent_once_and_runs_no_social_updates():
    config = _musr_config()
    options = {
        **dict(config.game.options),
        "initialization_only": True,
        "initialization": {"mode": "local_vote"},
    }
    config = replace(config, game=replace(config.game, options=options))
    provider = MockLLMProvider(
        config.llm_provider,
        response_factory=lambda _request: json.dumps(
            {"vote": "A", "reason": "private", "shared_fact_id": "none"}
        ),
    )
    result = asyncio.run(
        run_relational_imitation_round_feedback_game(
            create_game(config.game), config, provider
        )
    )

    assert len(result.initial_decisions) == 12
    assert result.logical_decisions == 12
    assert result.interactions == ()
    assert result.rounds == ()
    assert len(result.initial_state.initial_votes) == 12


def test_pilot_assignment_is_hash_pinned_f9_and_exactly_one_card_per_agent():
    config = load_run_config(PILOT_CONFIG, environment={})
    game = create_game(config.game)
    task = game.load_task(config.game)
    artifact_path = Path(config.game.options["initial_information"]["artifact_path"])

    assert (
        hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        == (config.game.options["initial_information"]["expected_file_sha256"])
    )
    assert len(task.agent_ids) == 24
    assert len(task.fact_order) == 9
    assert len(task.supporting_fact_groups or {}) == 9
    assert all(len(task.known_facts(agent)) == 1 for agent in task.agent_ids)
    assert {
        fact for agent in task.agent_ids for fact in task.known_facts(agent)
    } == set(task.fact_order)
    counts = [
        sum(fact in task.known_facts(agent) for agent in task.agent_ids)
        for fact in task.fact_order
    ]
    assert sorted(counts) == [2, 2, 2, 3, 3, 3, 3, 3, 3]


def test_pilot_prompt_exposes_only_simplified_actions_and_no_hidden_values():
    config = load_run_config(PILOT_CONFIG, environment={})
    game = create_game(config.game)
    state = game.initialize(config.game, config.execution.seed)
    prompt = "\n\n".join(
        message.content
        for message in game.ballot_request(
            state, state.agents[0].agent_id, (), config.game
        )
        .prompt.compile(RegexTokenCounter())
        .messages
    )

    assert "REQUEST | REPORT | NONE" in prompt
    for legacy in ("CLAIM", "RESULT", "REPLY", "CORRECTION"):
        assert legacy not in prompt
    assert "skill_matrix" not in prompt
    assert "cooperation_matrix" not in prompt
    assert "hidden_claim" not in prompt
