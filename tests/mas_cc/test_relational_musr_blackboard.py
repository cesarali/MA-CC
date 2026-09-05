"""MuSR Team Allocation adapter tests for the q-message game."""

from __future__ import annotations

from dataclasses import replace
import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from mas_cc.config import load_run_config
from mas_cc.control import RoundControlSignal
from mas_cc.games import create_game
from mas_cc.games.relational_reasoning.data import load_musr_team_allocation_task
from mas_cc.games.relational_reasoning.imitation_round_feedback.controller import (
    ADAPTIVE_COMMUNICATION,
    RECOMMENDATION_ONLY,
    SCHEDULE_ALWAYS,
    SCHEDULE_NEVER,
    TIMING_DAWN_ONLY,
    TRUTHFUL_STRATEGIC_REPORT,
    RelationalRoundBudgetedControl,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.adaptive_communication import (
    CommunicationChoice,
    CommunicationMode,
)
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
TRUTHFUL_DESIGN = (
    "configs/runs/relational_reasoning/blackboard_game/artifacts/"
    "task_002_truthful_controller.json"
)
TRUTHFUL_DESIGN_SHA256 = (
    "3c730e0dcc9c88129d4ec842a7110471d1fee831b5f6b9d1106a54149df9b89c"
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


def _truthful_config(*, rounds=1, budget=4, schedule=SCHEDULE_ALWAYS):
    config = _musr_config()
    options = {
        **dict(config.game.options),
        "task_id": "task_002",
        "truthful_controller_design": {
            "artifact_path": TRUTHFUL_DESIGN,
            "expected_file_sha256": TRUTHFUL_DESIGN_SHA256,
        },
        "rounds": rounds,
        "social_mode": "board",
        "social_group_size": 4,
        "board": {
            "sampling": "uniform",
            "message_lifetime_rounds": 1,
            "exclude_self_authored": True,
            "allow_no_post": True,
        },
        "initialization": {
            "mode": "explicit",
            "initial_votes": [
                "ALLOCATION_0",
                "ALLOCATION_1",
                "ALLOCATION_2",
            ]
            * 4,
        },
    }
    control = replace(
        config.control,
        options={
            **dict(config.control.options),
            "target": "ALLOCATION_2",
            "sensor_sample_size": 6,
            "intervention_budget": budget,
            "advocacy_schedule": schedule,
            "message_mode": RECOMMENDATION_ONLY,
            "controller_actuation_mode": TRUTHFUL_STRATEGIC_REPORT,
            "controller_timing": TIMING_DAWN_ONLY,
            "controller_report_cooldown_rounds": 1,
        },
    )
    return replace(
        config,
        game=replace(config.game, horizon=rounds, options=options),
        prompt=replace(
            config.prompt,
            prompt_family="relational_blackboard_ballot",
            prompt_version=2,
        ),
        control=control,
    )


def _truthful_provider(config, prompts):
    def factory(request):
        prompt = "\n\n".join(message.content for message in request.messages)
        prompts.append(prompt)
        return json.dumps(
            {
                "vote": "A",
                "private_reason": "private",
                "public_message": {
                    "type": "NONE",
                    "text": None,
                    "shared_fact_id": None,
                    "reply_to": None,
                },
            }
        )

    return MockLLMProvider(config.llm_provider, response_factory=factory)


def _adaptive_config(*, rounds=1, schedule=SCHEDULE_ALWAYS):
    config = _truthful_config(rounds=rounds, budget=3, schedule=schedule)
    return replace(
        config,
        game=replace(
            config.game,
            options={
                **dict(config.game.options),
                "prompt_version": 4,
                "board": {
                    **dict(config.game.options["board"]),
                    "allow_participant_requests": True,
                },
            },
        ),
        prompt=replace(config.prompt, prompt_version=4),
        control=replace(
            config.control,
            options={
                **dict(config.control.options),
                "controller_actuation_mode": ADAPTIVE_COMMUNICATION,
                "allow_controller_requests": True,
                "allow_controller_directives": True,
            },
        ),
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


def test_truthful_controller_design_is_symbolically_valid_and_has_24_reports():
    task = load_musr_team_allocation_task(
        DATASET,
        "task_002",
        population_size=12,
        truthful_controller_design_path=TRUTHFUL_DESIGN,
        truthful_controller_design_sha256=TRUTHFUL_DESIGN_SHA256,
    )

    assert task.controller_target == "ALLOCATION_2"
    assert task.controller_target != task.correct_relation
    assert len(task.controller_reportable_fact_ids) == 24
    assert len(set(task.controller_reportable_fact_ids)) == 24
    assert set(task.controller_reportable_fact_ids).issubset(task.facts)
    assert task.decisive_fact_ids


def test_truthful_report_selection_rotates_deterministically_with_cooldown():
    task = load_musr_team_allocation_task(
        DATASET,
        "task_002",
        population_size=12,
        truthful_controller_design_path=TRUTHFUL_DESIGN,
        truthful_controller_design_sha256=TRUTHFUL_DESIGN_SHA256,
    )
    config = _truthful_config(budget=4)
    control = RelationalRoundBudgetedControl.from_options(config.control.options)

    first = control.select_truthful_reports(
        task,
        episode_seed=11,
        round_index=0,
        live_fact_counts={},
        selected_rounds={},
    )
    repeated = control.select_truthful_reports(
        task,
        episode_seed=11,
        round_index=0,
        live_fact_counts={},
        selected_rounds={},
    )
    history = {row.fact_id: [0] for row in first}
    second = control.select_truthful_reports(
        task,
        episode_seed=11,
        round_index=1,
        live_fact_counts={row.fact_id: 1 for row in first},
        selected_rounds=history,
    )

    assert first == repeated
    assert len({row.fact_id for row in first}) == 4
    assert {row.fact_id for row in first}.isdisjoint({row.fact_id for row in second})
    assert all(row.cooldown_eligible for row in second)


def test_truthful_report_preflight_rejects_budget_above_distinct_pool():
    task = load_musr_team_allocation_task(
        DATASET,
        "task_002",
        population_size=24,
        truthful_controller_design_path=TRUTHFUL_DESIGN,
        truthful_controller_design_sha256=TRUTHFUL_DESIGN_SHA256,
    )
    control = RelationalRoundBudgetedControl.from_options(
        _truthful_config(budget=25).control.options
    )

    with pytest.raises(ValueError, match="distinct controller-reportable pool"):
        control.validate_truthful_report_task(task, 0)


def test_truthful_report_dawn_posts_exact_canonical_reports_and_transfers_facts():
    config = _truthful_config(budget=4)
    control = RelationalRoundBudgetedControl.from_options(config.control.options)
    prompts = []
    game = create_game(config.game)
    result = asyncio.run(
        run_relational_imitation_round_feedback_game(
            game,
            config,
            _truthful_provider(config, prompts),
            control=control,
        )
    )
    task = game.load_task(config.game)
    reports = [
        message
        for message in result.final_state.blackboard.messages
        if message.author_kind == "controller"
    ]
    event = result.rounds[0].event

    assert len(reports) == 4
    assert len({message.shared_fact_id for message in reports}) == 4
    assert all(message.message_type == "REPORT" for message in reports)
    assert all(
        message.text == task.fact_text(str(message.shared_fact_id))
        for message in reports
    )
    assert event["controller_posts"] == 4
    assert event["controller_reports_requested"] == 4
    assert event["controller_reports_admitted"] == 4
    assert event["controller_report_fact_acquisitions"] >= 1
    rendered = "\n".join(prompts)
    assert "Type: REPORT" in rendered
    assert "Type: DIRECTIVE" not in rendered
    for forbidden in ("controller", "authority", "policy"):
        assert forbidden not in rendered.casefold()


def test_truthful_report_no_op_posts_nothing():
    config = _truthful_config(budget=4, schedule=SCHEDULE_NEVER)
    control = RelationalRoundBudgetedControl.from_options(config.control.options)
    result = asyncio.run(
        run_relational_imitation_round_feedback_game(
            create_game(config.game),
            config,
            _truthful_provider(config, []),
            control=control,
        )
    )

    assert result.rounds[0].event["controller_posts"] == 0
    assert result.rounds[0].event["controller_reports_requested"] == 0
    assert all(
        message.author_kind != "controller"
        for message in result.final_state.blackboard.messages
    )


@pytest.mark.parametrize(
    ("mode", "expected_posts"),
    [
        (CommunicationMode.REPORT, 3),
        (CommunicationMode.REQUEST, 1),
        (CommunicationMode.DIRECTIVE, 1),
    ],
)
def test_adaptive_act_dispatches_exactly_one_allowed_strategy(
    monkeypatch, mode, expected_posts
):
    config = _adaptive_config()
    control = RelationalRoundBudgetedControl.from_options(config.control.options)
    seen_allowed = []

    def choose(_context, allowed, _rng):
        seen_allowed.append(tuple(allowed))
        assert mode in allowed
        return CommunicationChoice(mode=mode, reason="test")

    monkeypatch.setattr(
        "mas_cc.games.relational_reasoning.imitation_round_feedback.runtime."
        "choose_communication_mode",
        choose,
    )
    result = asyncio.run(
        run_relational_imitation_round_feedback_game(
            create_game(config.game),
            config,
            _truthful_provider(config, []),
            control=control,
        )
    )
    event = result.rounds[0].event
    controller_messages = [
        message
        for message in result.final_state.blackboard.messages
        if message.author_kind == "controller"
    ]

    assert event["U_k"] == 1
    assert event["chosen_message_mode"] == mode.value
    assert event["actual_controller_posts"] == expected_posts
    assert len(controller_messages) == expected_posts
    assert {message.message_type for message in controller_messages} == {mode.value}
    assert seen_allowed == [
        (
            CommunicationMode.REPORT,
            CommunicationMode.REQUEST,
            CommunicationMode.DIRECTIVE,
        )
    ]
    if mode != CommunicationMode.REPORT:
        assert all(message.shared_fact_id is None for message in controller_messages)


def test_adaptive_no_op_is_silent_and_does_not_invoke_chooser(monkeypatch):
    config = _adaptive_config(schedule=SCHEDULE_NEVER)
    control = RelationalRoundBudgetedControl.from_options(config.control.options)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("adaptive chooser ran for U=0")

    monkeypatch.setattr(
        "mas_cc.games.relational_reasoning.imitation_round_feedback.runtime."
        "choose_communication_mode",
        forbidden,
    )
    result = asyncio.run(
        run_relational_imitation_round_feedback_game(
            create_game(config.game),
            config,
            _truthful_provider(config, []),
            control=control,
        )
    )
    event = result.rounds[0].event

    assert event["U_k"] == 0
    assert event["chosen_message_mode"] is None
    assert event["actual_controller_posts"] == 0
    assert event["allowed_message_modes"] == ["REPORT", "REQUEST", "DIRECTIVE"]


def test_adaptive_disabled_directive_never_reaches_chooser(monkeypatch):
    config = _adaptive_config()
    config = replace(
        config,
        control=replace(
            config.control,
            options={
                **dict(config.control.options),
                "allow_controller_directives": False,
            },
        ),
    )
    control = RelationalRoundBudgetedControl.from_options(config.control.options)

    def choose(_context, allowed, _rng):
        assert CommunicationMode.DIRECTIVE not in allowed
        return CommunicationChoice(mode=CommunicationMode.REQUEST, reason="test")

    monkeypatch.setattr(
        "mas_cc.games.relational_reasoning.imitation_round_feedback.runtime."
        "choose_communication_mode",
        choose,
    )
    result = asyncio.run(
        run_relational_imitation_round_feedback_game(
            create_game(config.game),
            config,
            _truthful_provider(config, []),
            control=control,
        )
    )

    assert result.rounds[0].event["allowed_message_modes"] == ["REPORT", "REQUEST"]
    assert result.rounds[0].event["chosen_message_mode"] == "REQUEST"


def test_fake_provider_four_round_adaptive_sequence(monkeypatch):
    config = _adaptive_config(rounds=4)
    control = RelationalRoundBudgetedControl.from_options(config.control.options)
    modes = iter(
        (
            CommunicationMode.REPORT,
            CommunicationMode.REQUEST,
            CommunicationMode.DIRECTIVE,
        )
    )

    def scripted_round_signal(self, *, round_index, state, rng):
        action = "NO_OP" if round_index == 0 else "ADVOCATE_Z"
        target = self.resolved_target_for_task(
            create_game(config.game).load_task(config.game), config.execution.seed
        )
        counts = {option: 0 for option in state.data["task"]["possible_answers"]}
        return RoundControlSignal(
            action=action,
            target=target,
            observation={
                "sampled_agent_ids": [],
                "sampled_opinions": [],
                "sampled_opinion_counts": counts,
                "sample_size": 0,
            },
            metadata={
                "policy": "scripted_fake_smoke",
                "advocacy_probability": 0.0 if round_index == 0 else 1.0,
                "threshold": self.threshold,
                "beta": self.beta,
            },
        )

    def scripted_choice(_context, allowed, _rng):
        mode = next(modes)
        assert mode in allowed
        return CommunicationChoice(mode=mode, reason="scripted_fake_smoke")

    monkeypatch.setattr(
        RelationalRoundBudgetedControl, "round_signal", scripted_round_signal
    )
    monkeypatch.setattr(
        "mas_cc.games.relational_reasoning.imitation_round_feedback.runtime."
        "choose_communication_mode",
        scripted_choice,
    )
    result = asyncio.run(
        run_relational_imitation_round_feedback_game(
            create_game(config.game),
            config,
            _truthful_provider(config, []),
            control=control,
        )
    )

    assert [row.event["U_k"] for row in result.rounds] == [0, 1, 1, 1]
    assert [row.event["chosen_message_mode"] for row in result.rounds] == [
        None,
        "REPORT",
        "REQUEST",
        "DIRECTIVE",
    ]
    assert [row.event["actual_controller_posts"] for row in result.rounds] == [
        0,
        3,
        1,
        1,
    ]
    assert all(
        len(row.event["population_state_before"]) == 12
        and len(row.event["population_state_after"]) == 12
        for row in result.rounds
    )
