import asyncio
import json
import re
from dataclasses import replace

import pytest

from mas_cc.config import load_run_config
from mas_cc.games import create_game
from mas_cc.games.relational_reasoning.imitation_round_feedback.adaptive_communication import (
    CommunicationMode,
    ControllerCommunicationContext,
    ControllerVisibleFact,
    ControllerVisibleMessage,
    LLM_AUTHORED_REPORT_ONLY_POLICY,
    LLM_AUTHORED_FIXED_REPORT_ONLY_POLICY,
    LLM_AUTHORED_FULL_COMMUNICATION_POLICY,
    LLM_AUTHORED_VARIABLE_FULL_COMMUNICATION_POLICY,
    parse_llm_communication_choice,
    render_llm_controller_prompt,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.controller import (
    RelationalRoundBudgetedControl,
)
from mas_cc.games.hidden_bench.imitation.controller import (
    ADVOCATE_TARGET, NO_OP,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.controller_memory import (
    build_public_observation_ledger,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.initialization import (
    initialization_compatibility_key,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.runtime import (
    run_relational_imitation_round_feedback_game,
)
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider
from mas_cc.llm_runtime.exceptions import ConfigurationError


MEMORY_CONFIG = "configs/runs/smoke/task004_public_ledger_smoke/false_control.yaml"


def test_all_nondecisive_pool_preserves_task_and_paired_initialization():
    base = load_run_config(
        "configs/runs/smoke/task004_markovian_b90_10ep/false_control.yaml"
    )
    variants = (
        load_run_config(
            "configs/runs/smoke/task004_allfacts_b90_parallel/false_allfacts_rho1.yaml"
        ),
        load_run_config(
            "configs/runs/smoke/task004_allfacts_b90_parallel/truth_allfacts_rho075.yaml"
        ),
    )
    game = create_game(base.game)
    task = game.load_task(base.game)
    base_key = initialization_compatibility_key(game, base, base.execution.seed)
    assert len(task.controller_reportable_fact_ids) == 24
    assert len(task.decisive_fact_ids) == 6

    for config in variants:
        control = RelationalRoundBudgetedControl.from_options(config.control.options)
        pool = control.reportable_fact_ids(task)
        ranked = control.select_truthful_reports(
            task,
            episode_seed=config.execution.seed,
            round_index=1,
            live_fact_counts={},
            selected_rounds={},
        )
        assert len(pool) == len(ranked) == 43
        assert set(pool) == set(task.facts) - set(task.decisive_fact_ids)
        assert {row.fact_id for row in ranked} == set(pool)
        assert all(row.strategy_class == "all-nondecisive" for row in ranked)
        assert initialization_compatibility_key(
            create_game(config.game), config, config.execution.seed
        ) == base_key


def test_full_board_episode_prompt_describes_day_and_optional_spending():
    context = ControllerCommunicationContext(
        round_index=6,
        target="ALLOCATION_2",
        sampled_opinion_counts={"ALLOCATION_0": 2},
        live_message_type_counts={"REPORT": 2},
        previous_board_messages=(
            ControllerVisibleMessage(
                message_id="m1",
                author="Agent 1",
                message_type="REPORT",
                text="A verified observation.",
                vote="ALLOCATION_0",
                shared_fact_id="fact-1",
                reply_to=None,
                round_created=5,
            ),
            ControllerVisibleMessage(
                message_id="m2",
                author="Agent 2",
                message_type="REPORT",
                text="Another observation.",
                vote="ALLOCATION_0",
                shared_fact_id="fact-2",
                reply_to=None,
                round_created=5,
            ),
        ),
        eligible_facts=(
            ControllerVisibleFact("fact-1", "A verified observation.", 1, 4),
            ControllerVisibleFact("fact-2", "Another observation.", 1, 4),
        ),
        budget=2,
        budget_scope="episode",
        budget_total=30,
        budget_spent=2,
        horizon=30,
        rounds_remaining=23,
        population=15,
        sensing_mode="board",
        board_view_complete=True,
        scenario_question="SCENARIO\nAllocate three people.\n\nQUESTION\nWhich is best?",
        answer_display_texts={"ALLOCATION_2": "One candidate allocation"},
    )
    prompt = render_llm_controller_prompt(
        context,
        (CommunicationMode.REPORT, CommunicationMode.HOLD),
        LLM_AUTHORED_REPORT_ONLY_POLICY,
    )

    assert "The episode lasts 30 days" in prompt
    assert "every message posted to the public board on the previous day" in prompt
    assert "This is day 7 of 30; 23 days remain after this one." in prompt
    assert "28 messages remaining for the whole episode" in prompt
    assert "the limit is a ceiling, not a target" in prompt
    assert '"mode":"HOLD"' in prompt
    assert "Votes attached to yesterday's observed board messages" in prompt
    assert "These counts are per message, not per participant" in prompt
    assert "sample of 2 of the 15 participants" not in prompt
    assert "day 5" in prompt


def test_public_ledger_counts_distinct_observed_authors_and_excludes_own_posts():
    def post(message_id, *, day, step, author, vote, fact=None, own=False):
        return {
            "message_id": message_id,
            "round_created": day - 1,
            "micro_step_created": step,
            "author_id": author,
            "author_kind": "controller" if own else "agent",
            "message_type": "REPORT",
            "vote": vote,
            "shared_fact_id": fact,
        }

    messages = {
        row["message_id"]: row
        for row in (
            post("m1", day=1, step=1, author="agent_001", vote="ALLOCATION_0", fact="f1"),
            post("m2", day=1, step=2, author="agent_001", vote="ALLOCATION_2", fact="f2"),
            post("m3", day=1, step=3, author="agent_002", vote="ALLOCATION_0", fact="f1"),
            post("c1", day=1, step=0, author="controller", vote="ALLOCATION_2", fact="f3", own=True),
            post("unseen", day=1, step=4, author="agent_003", vote="ALLOCATION_1", fact="secret"),
            post("m4", day=2, step=5, author="agent_002", vote="ALLOCATION_2", fact="f2"),
        )
    }
    memory = build_public_observation_ledger(
        (
            {"day": 1, "message_ids": ["m2", "c1", "m1", "m3"], "view_complete": False},
            {"day": 2, "message_ids": ["m4"], "view_complete": True},
        ),
        messages,
        population_size=15,
        options=("ALLOCATION_0", "ALLOCATION_1", "ALLOCATION_2"),
    )

    first = memory["days"][0]
    assert first["messages_seen"] == 4
    assert first["distinct_participants_seen"] == 2
    assert first["last_posted_vote_counts"] == {
        "ALLOCATION_0": 1, "ALLOCATION_1": 0, "ALLOCATION_2": 1
    }
    assert first["controller_fact_ids"] == ["f3"]
    assert "secret" not in json.dumps(memory)
    assert memory["participants"][0]["last_posted_vote"] == "ALLOCATION_2"
    assert memory["facts"][0]["distinct_participant_authors"] == 2
    context = ControllerCommunicationContext(
        round_index=2,
        target="ALLOCATION_2",
        sampled_opinion_counts={"ALLOCATION_2": 1},
        live_message_type_counts={"REPORT": 1},
        public_memory=memory,
        budget=3,
        budget_scope="episode",
        budget_total=30,
        population=15,
        sensing_mode="board",
    )
    prompt = render_llm_controller_prompt(
        context,
        (CommunicationMode.REPORT, CommunicationMode.HOLD),
        LLM_AUTHORED_REPORT_ONLY_POLICY,
    )
    assert "2/15" in prompt
    assert "A last posted vote is not a current population vote" in prompt
    assert "it is still not a population census" in prompt
    assert "secret" not in prompt


def _mock_memory_episode(
    config, *, first_state=None, first_runtime=None, start=None, length=None,
    controller_answers=None,
):
    prompts = []

    def answer(request):
        content = "\n".join(message.content for message in request.messages)
        if request.metadata.get("decision_stage") == "controller_communication":
            prompts.append(content)
            if controller_answers is not None:
                return json.dumps(controller_answers[len(prompts) - 1])
            return json.dumps({
                "mode": "HOLD", "fact_ids": [], "text": None,
                "report_texts": [], "reason": "Save the remaining budget.",
            })
        if request.metadata.get("decision_stage") == "initial_vote":
            return json.dumps({
                "vote": "A", "reason": "My initial evidence.",
                "shared_fact_id": "none",
            })
        match = re.search(r"^- (cf_[A-Za-z0-9_]+):", content, re.MULTILINE)
        fact_id = match.group(1) if match else None
        return json.dumps({
            "vote": "A",
            "private_reason": "I used the evidence available to me.",
            "public_message": {
                "type": "REPORT" if fact_id else "NONE",
                "text": "I report a verified observation." if fact_id else None,
                "shared_fact_id": fact_id,
                "reply_to": None,
            },
        })

    result = asyncio.run(run_relational_imitation_round_feedback_game(
        create_game(config.game),
        config,
        MockLLMProvider(config.llm_provider, response_factory=answer),
        control=RelationalRoundBudgetedControl.from_options(config.control.options),
        initial_state=first_state,
        restored_runtime_state=first_runtime,
        start_round=start,
        continuation_length=length,
    ))
    return result, prompts


@pytest.mark.parametrize("profile", ("report_only", "full_communication"))
def test_round_zero_or_exact_budget_can_hold_then_post_three(profile):
    config = load_run_config(
        f"configs/runs/smoke/task004_allfacts_b3_round_{profile}_suite/"
        "truth_allfacts_rho075.yaml",
        environment={},
    )
    config = replace(
        config,
        control=replace(
            config.control,
            options={
                **dict(config.control.options),
                "advocacy_schedule": "always",
                "controller_round_budget_mode": "zero_or_exact",
            },
        ),
        game=replace(
            config.game,
            horizon=3,
            options={**dict(config.game.options), "rounds": 3},
        ),
    )
    task = create_game(config.game).load_task(config.game)
    control = RelationalRoundBudgetedControl.from_options(config.control.options)
    facts = tuple(control.reportable_fact_ids(task)[:3])
    hold = {
        "mode": "HOLD", "fact_ids": [], "text": None,
        "report_texts": [], "reason": "Observe another day.",
    }
    if profile == "report_only":
        post = {
            "mode": "REPORT", "fact_ids": list(facts), "text": None,
            "report_texts": [task.fact_text(fact) for fact in facts],
            "reason": "These three facts help the target.",
        }
        expected_policy = LLM_AUTHORED_FIXED_REPORT_ONLY_POLICY
    else:
        hold["message_texts"] = []
        post = {
            "mode": "REQUEST", "fact_ids": [], "text": None,
            "report_texts": [],
            "message_texts": [
                "Which observation supports this allocation?",
                "Can anyone compare the alternatives?",
                "What evidence is still missing?",
            ],
            "reason": "Ask for three useful comparisons.",
        }
        expected_policy = LLM_AUTHORED_FULL_COMMUNICATION_POLICY

    result, prompts = _mock_memory_episode(
        config, controller_answers=(hold, post)
    )
    assert len(prompts) == 2
    assert "The episode lasts 3 days" in prompts[0]
    assert "Unused slots expire tonight" in prompts[0]
    assert "HOLD" in prompts[0]
    assert "exactly 3" in prompts[0]
    assert "remaining for the whole episode" not in prompts[0]
    if profile == "full_communication":
        assert "request relevant evidence" in prompts[0]
        assert "direct attention to a comparison" in prompts[0]
    day_two, day_three = (row.event for row in result.rounds[1:])
    assert [day_two["actual_controller_posts"],
            day_three["actual_controller_posts"]] == [0, 3]
    assert day_two["chosen_message_mode"] == "HOLD"
    assert day_three["controller_communication_choice"]["policy"] == expected_policy
    assert day_three["controller_visible_input"]["budget"] == 3
    assert day_three["controller_visible_input"]["round_budget_mode"] == "zero_or_exact"
    assert "budget_remaining_for_episode" not in day_three["controller_visible_input"]


def test_round_zero_or_exact_budget_rejects_partial_posts_and_episode_scope():
    context = ControllerCommunicationContext(
        round_index=1,
        target="ALLOCATION_2",
        sampled_opinion_counts={},
        live_message_type_counts={},
        eligible_facts=(ControllerVisibleFact("fact-1", "One verified fact.", 0, None),),
        budget=3,
        budget_scope="per_round",
        round_budget_mode="zero_or_exact",
    )
    report = {
        "mode": "REPORT", "fact_ids": ["fact-1"], "text": None,
        "report_texts": ["One verified fact."], "reason": "One fact.",
    }
    with pytest.raises(ValueError, match="exactly budget fact IDs"):
        parse_llm_communication_choice(
            json.dumps(report), context=context,
            allowed_modes=(CommunicationMode.REPORT, CommunicationMode.HOLD),
            policy=LLM_AUTHORED_FIXED_REPORT_ONLY_POLICY,
        )
    request = {
        "mode": "REQUEST", "fact_ids": [], "text": None,
        "report_texts": [], "message_texts": ["What evidence?"],
        "reason": "Ask once.",
    }
    with pytest.raises(ValueError, match="exactly budget message_texts"):
        parse_llm_communication_choice(
            json.dumps(request), context=context,
            allowed_modes=(CommunicationMode.REQUEST, CommunicationMode.HOLD),
            policy=LLM_AUTHORED_FULL_COMMUNICATION_POLICY,
        )
    options = dict(load_run_config(
        "configs/runs/smoke/task004_allfacts_b3_round_report_only_suite/"
        "truth_allfacts_rho075.yaml", environment={},
    ).control.options)
    options["controller_round_budget_mode"] = "zero_or_exact"
    options["controller_budget_scope"] = "episode"
    with pytest.raises(ConfigurationError, match="zero_or_exact requires"):
        RelationalRoundBudgetedControl.from_options(options)


@pytest.mark.parametrize("profile", ("report_only", "full_communication"))
def test_b3_suite_uses_sigmoid_gate_and_exact_posts(profile):
    class Draw:
        def __init__(self, value):
            self.value = value

        def random(self):
            return self.value

    for arm in ("false_allfacts_rho075", "truth_allfacts_rho075",
                "false_allfacts_rho1", "truth_allfacts_rho1"):
        config = load_run_config(
            f"configs/runs/smoke/task004_allfacts_b3_round_{profile}_suite/"
            f"{arm}.yaml", environment={},
        )
        control = RelationalRoundBudgetedControl.from_options(config.control.options)
        assert control.advocacy_schedule == "soft"
        assert control.controller_budget_scope == "per_round"
        assert control.controller_round_budget_mode == "exact"
        assert control.intervention_budget == 3
        assert control.select_action(0.5, Draw(0.4)) == (ADVOCATE_TARGET, 0.5)
        assert control.select_action(0.5, Draw(0.6)) == (NO_OP, 0.5)


def test_full_communication_episode_budget_chooses_a_partial_post_then_holds():
    config = load_run_config(
        "configs/runs/smoke/task004_allfacts_b90_full_communication_suite/"
        "false_allfacts_rho1.yaml",
        environment={},
    )
    config = replace(
        config,
        game=replace(
            config.game,
            horizon=3,
            options={**dict(config.game.options), "rounds": 3},
        ),
    )
    answers = (
        {
            "mode": "REQUEST", "fact_ids": [], "text": None,
            "report_texts": [], "message_texts": ["Which evidence distinguishes the allocations?"],
            "reason": "Ask for the missing comparison.",
        },
        {
            "mode": "HOLD", "fact_ids": [], "text": None,
            "report_texts": [], "message_texts": [],
            "reason": "Wait for the discussion to respond.",
        },
    )
    result, prompts = _mock_memory_episode(config, controller_answers=answers)
    assert len(prompts) == 2
    assert "the limit is a ceiling, not a target" in prompts[0].lower()
    assert '"mode":"REPORT|REQUEST|DIRECTIVE|HOLD"' in prompts[0]
    day_two, day_three = (row.event for row in result.rounds[1:])
    assert day_two["controller_communication_choice"]["policy"] == (
        LLM_AUTHORED_VARIABLE_FULL_COMMUNICATION_POLICY
    )
    assert day_two["actual_controller_posts"] == 1
    posts = [message for message in result.final_state.blackboard.messages
             if message.author_kind == "controller"]
    assert len(posts) == 1
    assert posts[0].message_type == "REQUEST"
    assert posts[0].text == "Which evidence distinguishes the allocations?"
    assert day_three["actual_controller_posts"] == 0
    assert day_three["chosen_message_mode"] == "HOLD"
    assert day_three["controller_visible_input"]["budget_remaining_for_episode"] == 89


def test_full_communication_episode_budget_validates_variable_report_count():
    context = ControllerCommunicationContext(
        round_index=1,
        target="ALLOCATION_2",
        sampled_opinion_counts={},
        live_message_type_counts={},
        eligible_facts=(ControllerVisibleFact("fact-1", "One verified fact.", 0, None),),
        budget=2,
        budget_scope="episode",
        budget_total=90,
    )
    report = {
        "mode": "REPORT", "fact_ids": ["fact-1"], "text": None,
        "report_texts": ["One verified fact."], "message_texts": [],
        "reason": "This fact helps the target.",
    }
    allowed = (
        CommunicationMode.REPORT, CommunicationMode.REQUEST,
        CommunicationMode.DIRECTIVE, CommunicationMode.HOLD,
    )
    choice = parse_llm_communication_choice(
        json.dumps(report), context=context, allowed_modes=allowed,
        policy=LLM_AUTHORED_VARIABLE_FULL_COMMUNICATION_POLICY,
    )
    assert choice.fact_ids == ("fact-1",)
    report["fact_ids"].append("fact-2")
    report["report_texts"].append("Invented.")
    with pytest.raises(ValueError, match="outside the eligible pool"):
        parse_llm_communication_choice(
            json.dumps(report), context=context, allowed_modes=allowed,
            policy=LLM_AUTHORED_VARIABLE_FULL_COMMUNICATION_POLICY,
        )


def test_full_communication_episode_budget_posts_one_grounded_report():
    config = load_run_config(
        "configs/runs/smoke/task004_allfacts_b90_full_communication_suite/"
        "truth_allfacts_rho075.yaml",
        environment={},
    )
    config = replace(
        config,
        game=replace(
            config.game,
            horizon=2,
            options={**dict(config.game.options), "rounds": 2},
        ),
    )
    task = create_game(config.game).load_task(config.game)
    fact_id = task.controller_reportable_fact_ids[0]
    answer = {
        "mode": "REPORT", "fact_ids": [fact_id], "text": None,
        "report_texts": [task.fact_text(fact_id)], "message_texts": [],
        "reason": "This verified fact supports the target.",
    }
    result, _ = _mock_memory_episode(config, controller_answers=(answer,))
    event = result.rounds[1].event
    assert event["actual_controller_posts"] == 1
    assert event["controller_report_fact_ids"] == [fact_id]
    posts = [message for message in result.final_state.blackboard.messages
             if message.author_kind == "controller"]
    assert len(posts) == 1
    assert posts[0].message_type == "REPORT"
    assert posts[0].shared_fact_id == fact_id


def test_all_nondecisive_facts_reach_live_controller_prompt():
    config = load_run_config(
        "configs/runs/smoke/task004_allfacts_b90_parallel/false_allfacts_rho1.yaml",
        environment={},
    )
    config = replace(
        config,
        game=replace(
            config.game,
            horizon=2,
            options={**dict(config.game.options), "rounds": 2},
        ),
    )
    task = create_game(config.game).load_task(config.game)
    _, prompts = _mock_memory_episode(config)
    assert len(prompts) == 1
    assert all(f"| {fact_id} |" in prompts[0] for fact_id in task.facts
               if fact_id not in task.decisive_fact_ids)
    assert all(f"| {fact_id} |" not in prompts[0]
               for fact_id in task.decisive_fact_ids)


def test_public_ledger_is_opt_in_and_survives_round_boundary_resume():
    config = load_run_config(MEMORY_CONFIG, environment={})
    game_options = {**dict(config.game.options), "rounds": 3}
    config = replace(
        config,
        game=replace(config.game, horizon=3, options=game_options),
    )
    full, prompts = _mock_memory_episode(config)
    preparation, _ = _mock_memory_episode(config, length=2)
    continued, resumed_prompts = _mock_memory_episode(
        config,
        first_state=preparation.final_state,
        first_runtime=preparation.runtime_state,
        start=2,
        length=1,
    )
    assert len(prompts) == 2
    assert "Public observation memory" in prompts[0]
    assert "distinct participants seen" in prompts[0]
    assert "Public observation memory" in resumed_prompts[0]
    assert full.rounds[1].event["controller_visible_input"]["public_memory"]["days"][0][
        "distinct_participants_seen"
    ] > 0
    assert len(full.runtime_state["controller_observed_board_days"]) == 2
    assert (
        continued.rounds[0].event["controller_visible_input"]["public_memory"]
        == full.rounds[2].event["controller_visible_input"]["public_memory"]
    )

    control_options = dict(config.control.options)
    control_options.pop("controller_memory_mode")
    without_memory = replace(
        config,
        control=replace(config.control, options=control_options),
    )
    old, old_prompts = _mock_memory_episode(without_memory)
    assert "Public observation memory" not in old_prompts[0]
    assert "public_memory" not in old.rounds[1].event["controller_visible_input"]
    assert "controller_observed_board_days" not in old.runtime_state
    assert [row.event["occupation_counts_after"] for row in old.rounds] == [
        row.event["occupation_counts_after"] for row in full.rounds
    ]

    never_options = {**dict(config.control.options), "advocacy_schedule": "never"}
    never = replace(config, control=replace(config.control, options=never_options))
    silent, silent_prompts = _mock_memory_episode(never, length=2)
    assert silent_prompts == []
    assert len(silent.runtime_state["controller_observed_board_days"]) == 1


def test_public_ledger_rejects_non_board_sensing_and_long_lived_posts():
    config = load_run_config(MEMORY_CONFIG, environment={})
    options = {**dict(config.control.options), "sensing_mode": "votes"}
    with pytest.raises(ConfigurationError, match="controller_memory_mode"):
        RelationalRoundBudgetedControl.from_options(options)

    board = {**dict(config.game.options["board"]), "message_lifetime_rounds": 2}
    game_options = {**dict(config.game.options), "board": board}
    config = replace(config, game=replace(config.game, options=game_options))
    with pytest.raises(ValueError, match="one-day board messages"):
        _mock_memory_episode(config, length=1)
