"""Finite-memory q-message board behavior for the relational game."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from mas_cc.config import load_run_config
from mas_cc.core import AgentId
from mas_cc.experiments.configured_analysis import validate_configured_analysis
from mas_cc.games import create_game
from mas_cc.games.relational_reasoning.data import DEFAULT_TASK_DATASET_DIR
from mas_cc.games.relational_reasoning.imitation_round_feedback.controller import (
    COORDINATION_REQUEST,
    DIRECT_RECOMMENDATION,
    RECOMMENDATION_ONLY,
    SCHEDULE_ALWAYS,
    SCHEDULE_NEVER,
    SCHEDULE_SOFT,
    TIMING_DAWN_ONLY,
    RelationalRoundBudgetedControl,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.analysis import (
    adapt_relational_round_record,
)
from mas_cc.games.hidden_bench.imitation.controller import advocacy_probability
from mas_cc.games.relational_reasoning.imitation_round_feedback.prompts import (
    BlackboardBallotContract,
    build_relational_blackboard_prompt,
    relational_blackboard_ballot_prompt,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.pilot_artifacts import (
    REQUIRED_OUTPUTS,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.runtime import (
    run_relational_imitation_round_feedback_game,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.state import (
    BLACKBOARD_MESSAGE_SCHEMA_VERSION,
    LEGACY_BLACKBOARD_MESSAGE_SCHEMA_VERSION,
    ORDINARY_ACTION_TYPES,
    BlackboardMessage,
    BlackboardState,
)
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider
from mas_cc.llm_runtime.prompts import RegexTokenCounter
from mas_cc.experiments import run_experiment_sync

pytestmark = pytest.mark.skipif(
    not (DEFAULT_TASK_DATASET_DIR / "task_0001.json").exists(),
    reason="the relational example dataset is not present",
)

CONFIG = (
    "configs/runs/relational_reasoning/"
    "misselaneous/relational_imitation_round_feedback_no_control_smoke.yaml"
)


def _config(*, rounds=1, q=1, lifetime=1, prompt_version=2):
    config = load_run_config(CONFIG, environment={})
    options = {
        **dict(config.game.options),
        "rounds": rounds,
        "social_group_size": q,
        "social_mode": "board",
        "prompt_version": prompt_version,
        "board": {
            "sampling": "uniform",
            "message_lifetime_rounds": lifetime,
            "exclude_self_authored": True,
            "allow_no_post": True,
        },
    }
    prompt = replace(
        config.prompt,
        prompt_family="relational_blackboard_ballot",
        prompt_version=prompt_version,
    )
    return replace(
        config,
        game=replace(config.game, horizon=rounds, options=options),
        prompt=prompt,
    )


class _BoardBallots:
    def __init__(self, *, post=True):
        self.prompts = []
        self.post = post

    def provider(self, config):
        def factory(request):
            prompt = "\n\n".join(message.content for message in request.messages)
            self.prompts.append(prompt)
            return json.dumps(
                {
                    "vote": "A",
                    "private_reason": "private reasoning that must not be shared",
                    "public_message": {
                        "type": "REPORT" if self.post else "NONE",
                        "text": "I compared my available evidence."
                        if self.post
                        else None,
                        "shared_fact_id": None,
                        "reply_to": None,
                    },
                }
            )

        return MockLLMProvider(config, response_factory=factory)


def _run(config, *, control=None, ballots=None):
    ballots = ballots or _BoardBallots()
    result = asyncio.run(
        run_relational_imitation_round_feedback_game(
            create_game(config.game),
            config,
            ballots.provider(config.llm_provider),
            control=control,
        )
    )
    return result, ballots


def _control(config, mode):
    return RelationalRoundBudgetedControl.from_options(
        {
            **dict(config.control.options),
            "sensor_sample_size": 6,
            "intervention_budget": 4,
            "advocacy_schedule": SCHEDULE_ALWAYS,
            "message_mode": RECOMMENDATION_ONLY,
            "controller_actuation_mode": mode,
        }
    )


def _dawn_control(config, *, schedule=SCHEDULE_ALWAYS, budget=4):
    return RelationalRoundBudgetedControl.from_options(
        {
            **dict(config.control.options),
            "sensor_sample_size": 6,
            "intervention_budget": budget,
            "advocacy_schedule": schedule,
            "message_mode": RECOMMENDATION_ONLY,
            "controller_actuation_mode": COORDINATION_REQUEST,
            "controller_timing": TIMING_DAWN_ONLY,
        }
    )


def test_board_state_lifetime_and_serialization():
    message = BlackboardMessage(
        message_id="m1",
        author_id="agent_001",
        message_type="REQUEST",
        text="What evidence separates A and B?",
        vote="NORTH",
        shared_fact_id=None,
        reply_to=None,
        round_created=2,
        micro_step_created=7,
        expires_after_round=3,
    )
    board = BlackboardState().append(message)

    assert board.live_messages(2) == (message,)
    assert board.live_messages(3) == (message,)
    assert board.live_messages(4) == ()
    assert BlackboardState.from_sequence(board.to_list()).find("m1") == message
    assert board.expire(3)[1] == ("m1",)


def test_board_sampling_excludes_self_and_never_duplicates_messages():
    messages = tuple(
        BlackboardMessage(
            message_id=f"m{index}",
            author_id="agent_001" if index == 1 else "agent_002",
            message_type="REPORT",
            text=f"message {index}",
            vote="NORTH",
            shared_fact_id=None,
            reply_to=None,
            round_created=0,
            micro_step_created=index,
            expires_after_round=0,
        )
        for index in range(1, 4)
    )
    board = BlackboardState(messages)
    sampled = board.sample_live(
        0, 3, __import__("random").Random(1), exclude_author_id="agent_001"
    )

    assert len(sampled) == 2
    assert len({message.message_id for message in sampled}) == 2
    assert {message.author_id for message in sampled} == {"agent_002"}


def test_all_ordinary_actions_validate_and_replies_need_visible_target():
    for kind in ORDINARY_ACTION_TYPES:
        contract = BlackboardBallotContract(
            allowed_values=("A", "B"),
            options={"fact_ids": (), "relations": (), "visible_message_ids": ("m1",)},
        )
        response = json.dumps(
            {
                "vote": "A",
                "private_reason": "private",
                "public_message": {
                    "type": kind,
                    "text": None if kind == "NONE" else "public",
                    "shared_fact_id": None,
                    "reply_to": None,
                },
            }
        )
        assert contract.validate(response).valid

    invalid = json.dumps(
        {
            "vote": "A",
            "private_reason": "private",
            "public_message": {
                "type": "REPORT",
                "text": "public",
                "shared_fact_id": None,
                "reply_to": "missing",
            },
        }
    )
    assert not contract.validate(invalid).valid


def test_request_cannot_attach_evidence_and_report_can():
    contract = BlackboardBallotContract(
        allowed_values=("A", "B"),
        options={"fact_ids": ("f1",), "relations": (), "visible_message_ids": ()},
    )
    request = {
        "vote": "A",
        "private_reason": "private",
        "public_message": {
            "type": "REQUEST",
            "text": "Who has evidence?",
            "shared_fact_id": "f1",
            "reply_to": None,
        },
    }
    assert not contract.validate(json.dumps(request)).valid
    request["public_message"]["type"] = "REPORT"
    assert contract.validate(json.dumps(request)).valid


def _rendered_blackboard_prompt(*, version, text="So I stick with C.", shared=None):
    return build_relational_blackboard_prompt(
        identity="Agent 2",
        question="Which allocation is best?",
        option_letters={
            "A": "ALLOCATION_1",
            "B": "ALLOCATION_0",
            "C": "ALLOCATION_2",
        },
        known_facts=("f1: Farah is highly skilled at pipeline work.",),
        fact_ids=("f1",),
        current_vote="Bruno builds the pipeline; Alice and Chandra interview.",
        board_messages=(
            {
                "message_id": "m1",
                "label": "Agent 1",
                "message_type": "REPORT",
                "vote": "ALLOCATION_1",
                "text": text,
                "shared_fact_text": shared,
                "reply_to": None,
            },
        ),
        social_context=True,
        answer_display_texts={
            "ALLOCATION_0": "Alice builds; Bruno and Chandra interview.",
            "ALLOCATION_1": "Bruno builds; Alice and Chandra interview.",
            "ALLOCATION_2": "Chandra builds; Alice and Bruno interview.",
        },
        version=version,
    )


def test_blackboard_v3_localizes_semantic_vote_without_rewriting_public_text():
    prompt = _rendered_blackboard_prompt(version=3)
    rendered = "\n\n".join(
        message.content for message in prompt.compile(RegexTokenCounter()).messages
    )

    assert "Current vote: A (Bruno builds; Alice and Chandra interview.)" in rendered
    assert "Current vote: C" not in rendered
    assert "So I stick with C." in rendered


def test_blackboard_v3_runtime_stores_model_letters_as_semantic_votes():
    config = _config(prompt_version=3)
    result, _ = _run(config)

    for interaction in result.interactions:
        mapping = interaction.decisions[0].action.metadata["option_letters"]
        message = result.final_state.blackboard.find(
            interaction.transition.event["new_message_id"]
        )
        assert message is not None
        assert message.vote == mapping["A"]
        assert message.vote != "A"


def test_blackboard_v3_prompt_guidance_and_json_contract_are_surgical():
    prompt = _rendered_blackboard_prompt(version=3)
    rendered = "\n\n".join(
        message.content for message in prompt.compile(RegexTokenCounter()).messages
    )

    assert "YOUR VERIFIED EVIDENCE" in rendered
    assert "VERIFIED SHARED FACT" in rendered
    assert "REPORT text is their interpretation" in rendered
    assert "YOUR PREVIOUS VOTE" in rendered
    assert "You may keep or revise this vote" in rendered
    assert "Some participants may have objectives" not in rendered
    assert "rather than by repetition, confidence" in rendered
    assert "REQUEST asks for specific missing evidence or information" in rendered
    assert "missing or ambiguous" in rendered
    assert "Prefer REQUEST over NONE" in rendered
    assert "Ask for something specific rather than a generic explanation" in rendered
    assert "rather than using option letters\n  A/B/C" in rendered
    assert '"vote": "<A | B | C>"' in rendered
    assert '"type": "<REQUEST | REPORT | NONE>"' in rendered


def test_blackboard_v3_renders_identical_shared_fact_once_and_interpretation_twice():
    fact = "Farah is highly skilled at pipeline work."
    duplicate = "\n\n".join(
        message.content
        for message in _rendered_blackboard_prompt(
            version=3, text=f"  {fact.upper()}  ", shared=fact
        )
        .compile(RegexTokenCounter())
        .messages
    )
    interpreted = "\n\n".join(
        message.content
        for message in _rendered_blackboard_prompt(
            version=3,
            text="This makes Bruno's allocation less plausible.",
            shared=fact,
        )
        .compile(RegexTokenCounter())
        .messages
    )
    duplicate_social = duplicate.split("CURRENT SOCIAL INFORMATION", 1)[1].split(
        "\n\nDECISION", 1
    )[0]

    assert duplicate_social.count(fact) == 1
    assert "Public message:" not in duplicate_social
    assert f"Verified shared fact:\n{fact}" in duplicate_social
    assert (
        "Public message:\nThis makes Bruno's allocation less plausible." in interpreted
    )
    assert f"Verified shared fact:\n{fact}" in interpreted


def test_blackboard_v2_replays_old_wording_and_v3_has_new_fingerprint():
    old = _rendered_blackboard_prompt(version=2)
    new = _rendered_blackboard_prompt(version=3)
    old_text = "\n\n".join(
        message.content for message in old.compile(RegexTokenCounter()).messages
    )

    assert old.version == 2
    assert "YOUR CURRENT KNOWLEDGE" in old_text
    assert "YOUR CURRENT POSITION" in old_text
    assert "Evidence they are sharing:" not in old_text
    assert "Some participants may have objectives" in old_text
    assert old.definition_hash != new.definition_hash


def test_blackboard_registry_keeps_v2_and_registers_v3():
    from mas_cc.games.registry import (
        create_default_prompt_registry,
        register_game_prompt_factories,
    )

    registry = register_game_prompt_factories(create_default_prompt_registry())

    assert registry.get("relational_blackboard_ballot", 2).version == 2
    assert registry.get("relational_blackboard_ballot", 3).version == 3


def test_blackboard_v3_rejects_a_letter_as_authoritative_message_vote():
    with pytest.raises(ValueError, match="must be a semantic answer"):
        build_relational_blackboard_prompt(
            identity="Agent 2",
            question="Which allocation is best?",
            option_letters={"A": "ALLOCATION_1", "B": "ALLOCATION_0"},
            known_facts=(),
            fact_ids=(),
            current_vote=None,
            board_messages=(
                {
                    "message_id": "m1",
                    "label": "Agent 1",
                    "message_type": "REPORT",
                    "vote": "A",
                    "text": "The first allocation seems best.",
                    "shared_fact_text": None,
                    "reply_to": None,
                },
            ),
            version=3,
        )


def test_blackboard_prompt_factory_supports_only_historical_and_current_versions():
    assert relational_blackboard_ballot_prompt(version=2).version == 2
    assert relational_blackboard_ballot_prompt(version=3).version == 3
    with pytest.raises(ValueError, match="must be one of"):
        relational_blackboard_ballot_prompt(version=4)


def test_invalid_blackboard_fact_repair_requires_null_without_coercion():
    contract = BlackboardBallotContract(
        allowed_values=("A", "B"),
        options={"fact_ids": ("f1",), "relations": (), "visible_message_ids": ()},
    )
    response = json.dumps(
        {
            "vote": "A",
            "private_reason": "private",
            "public_message": {
                "type": "REPORT",
                "text": "public",
                "shared_fact_id": "f99",
                "reply_to": None,
            },
        }
    )

    result = contract.validate(response)
    assert not result.valid
    guidance = contract.repair_guidance(result.issues)
    assert "shared_fact_id to null" in guidance
    assert "Do not repeat" in guidance
    assert "f1" not in guidance


def test_new_messages_are_role_aware_and_legacy_records_remain_readable():
    with pytest.raises(ValueError, match="vote must be non-empty"):
        BlackboardMessage(
            message_id="m0",
            author_id="agent_001",
            message_type="REQUEST",
            text="invalid missing vote",
            vote="",
            shared_fact_id=None,
            reply_to=None,
            round_created=0,
            micro_step_created=0,
            expires_after_round=0,
        )
    with pytest.raises(ValueError, match="requires shared_fact_id"):
        BlackboardMessage(
            message_id="m1",
            author_id="control-source",
            author_kind="controller",
            message_type="REPORT",
            text="invalid role",
            vote="NORTH",
            shared_fact_id=None,
            reply_to=None,
            round_created=0,
            micro_step_created=0,
            expires_after_round=0,
        )
    report = BlackboardMessage(
        message_id="m2",
        author_id="control-source",
        author_kind="controller",
        message_type="REPORT",
        text="canonical fact text",
        vote="NORTH",
        shared_fact_id="f1",
        reply_to=None,
        round_created=0,
        micro_step_created=0,
        expires_after_round=0,
    )
    assert report.shared_fact_id == "f1"
    legacy = BlackboardMessage.from_mapping(
        {
            "message_id": "old",
            "author_id": "agent_001",
            "message_type": "CORRECTION",
            "text": "historical record",
            "vote": "NORTH",
            "shared_fact_id": None,
            "reply_to": "older",
            "round_created": 0,
            "micro_step_created": 1,
            "expires_after_round": 0,
        }
    )
    assert legacy.schema_version == LEGACY_BLACKBOARD_MESSAGE_SCHEMA_VERSION
    assert BLACKBOARD_MESSAGE_SCHEMA_VERSION == 2


def test_empty_board_has_no_peer_fallback_and_later_posts_are_visible():
    result, ballots = _run(_config(q=2))
    events = [item.transition.event for item in result.interactions]

    assert events[0]["sampled_peer_ids"] == []
    assert events[0]["q_effective"] == 0
    assert events[0]["sampled_message_ids"] == []
    assert any(event["q_effective"] > 0 for event in events[1:])
    assert "private reasoning that must not be shared" not in "\n".join(
        ballots.prompts[1:]
    )


def test_no_post_appends_no_message():
    result, _ = _run(_config(), ballots=_BoardBallots(post=False))

    assert result.final_state.blackboard.messages == ()
    assert all(
        not event.transition.event["focal_posted_message"]
        for event in result.interactions
    )


def test_lifetime_one_expires_before_next_round_and_two_survives():
    one, _ = _run(_config(rounds=2, lifetime=1))
    two, _ = _run(_config(rounds=2, lifetime=2))

    first_round_ids = {
        message.message_id
        for message in one.final_state.blackboard.messages
        if message.round_created == 0
    }
    second_round_samples = {
        message_id
        for event in one.interactions[one.final_state.data["rules"]["n_agents"] :]
        for message_id in event.transition.event["sampled_message_ids"]
    }
    assert first_round_ids.isdisjoint(second_round_samples)
    assert any(
        message_id
        in {
            m.message_id
            for m in two.final_state.blackboard.messages
            if m.round_created == 0
        }
        for event in two.interactions[two.final_state.data["rules"]["n_agents"] :]
        for message_id in event.transition.event["sampled_message_ids"]
    )


def test_direct_recommendation_is_transient_and_exact_budget():
    config = _config(q=1)
    result, _ = _run(config, control=_control(config, DIRECT_RECOMMENDATION))
    events = [item.transition.event for item in result.interactions]
    controlled = [event for event in events if event["controlled_slot"]]

    assert len(controlled) == 4
    assert all(event["controller_message_directly_exposed"] for event in controlled)
    assert all(not event["controller_message_posted"] for event in controlled)
    assert all(event["q_effective"] == 1 for event in controlled)
    assert all(
        message.author_kind != "controller"
        for message in result.final_state.blackboard.messages
    )


def test_coordination_request_posts_exact_budget_before_sampling():
    config = _config(q=1)
    result, _ = _run(config, control=_control(config, COORDINATION_REQUEST))
    round_record = result.rounds[0].event
    controller_messages = [
        message
        for message in result.final_state.blackboard.messages
        if message.author_kind == "controller"
    ]

    assert len(controller_messages) == 4
    assert len({message.message_id for message in controller_messages}) == 4
    assert all(message.message_type == "DIRECTIVE" for message in controller_messages)
    assert all(message.shared_fact_id is None for message in controller_messages)
    assert round_record["controller_posts"] == 4
    assert round_record["controller_message_exposures"] <= 4
    assert round_record["theory_status"] == "reference_only"


@pytest.mark.parametrize(
    ("schedule", "expected_u", "expected_probability", "expected_directives"),
    [
        (SCHEDULE_NEVER, 0, 0.0, 0),
        (SCHEDULE_ALWAYS, 1, 1.0, 4),
    ],
)
def test_controller_round_record_logs_y_u_and_directive_injection_times(
    schedule, expected_u, expected_probability, expected_directives
):
    config = _config(q=1)
    control = RelationalRoundBudgetedControl.from_options(
        {
            **dict(config.control.options),
            "sensor_sample_size": 6,
            "intervention_budget": 4,
            "advocacy_schedule": schedule,
            "message_mode": RECOMMENDATION_ONLY,
            "controller_actuation_mode": COORDINATION_REQUEST,
        }
    )
    result, _ = _run(config, control=control)
    event = result.rounds[0].event

    assert event["controller_sensor_Y"]["sample_size"] == 6
    assert len(event["controller_sensor_Y"]["sampled_agent_ids"]) == 6
    assert len(event["controller_sensor_Y"]["sampled_votes"]) == 6
    assert event["controller_sampled_U"] == expected_u
    assert event["controller_probability_U1_given_Y"] == expected_probability
    assert (
        len(event["controller_injection_within_round_indices"]) == expected_directives
    )
    assert (
        len(event["controller_injection_global_update_indices"]) == expected_directives
    )
    assert event["directive_count"] == expected_directives
    assert event["controller_posts"] == expected_directives


def test_pilot_uses_existing_stochastic_soft_policy_and_exact_controller_parameters():
    config = load_run_config(
        "configs/runs/relational_reasoning/blackboard_game/"
        "musr_blackboard_task001_5round_simplified_messages.yaml",
        environment={},
    )
    control = RelationalRoundBudgetedControl.from_options(config.control.options)

    assert control.sensor_sample_size == 12
    assert control.intervention_budget == 6
    assert control.advocacy_schedule == SCHEDULE_SOFT
    assert control.beta == 4.0
    assert control.threshold == 0.5
    assert control.target == "correct"
    assert control.controller_actuation_mode == COORDINATION_REQUEST
    assert control.controller_timing == TIMING_DAWN_ONLY


def test_dawn_blackboard_seeds_exact_budget_before_day_and_never_posts_during_day():
    config = _config(q=1)
    result, ballots = _run(config, control=_dawn_control(config))
    event = result.rounds[0].event
    micro_events = [item.transition.event for item in result.interactions]
    directives = [
        message
        for message in result.final_state.blackboard.messages
        if message.author_kind == "controller"
    ]

    assert event["controller_sampled_U"] == 1
    assert event["b"] == 4
    assert event["controlled_positions"] == []
    assert event["controlled_positions_seed"] is None
    assert event["controlled_positions_hash_or_id"] is None
    assert event["dawn_directive_count"] == 4
    assert len(directives) == 4
    assert all(message.micro_step_created == 0 for message in directives)
    assert micro_events[0]["board_size_before"] == 4
    assert all(not item["controlled_slot"] for item in micro_events)
    assert all(not item["controller_message_posted"] for item in micro_events)
    assert sum(
        message.author_kind == "agent"
        for message in result.final_state.blackboard.messages
    ) == len(result.interactions)
    prompt = next(text for text in ballots.prompts if "Type: DIRECTIVE" in text)
    assert f"Agent {event['N'] + 1}" in prompt
    assert "controller" not in prompt.lower()
    assert "analysis/dashboard" not in prompt.lower()


def test_dawn_no_op_has_no_coordinator_message_and_population_contract_is_unchanged():
    config = _config(q=1)
    result, _ = _run(config, control=_dawn_control(config, schedule=SCHEDULE_NEVER))
    event = result.rounds[0].event
    adapted = adapt_relational_round_record(event)

    assert event["U_k"] == 0
    assert event["dawn_directive_count"] == 0
    assert event["directive_message_ids"] == []
    assert all(
        message.author_kind != "controller"
        for message in result.final_state.blackboard.messages
    )
    assert len(event["sensor_agent_ids"]) == 6
    assert set(event["sensor_agent_ids"]).issubset(set(event["agent_ids"]))
    assert "control-source" not in event["sensor_agent_ids"]
    assert sum(event["occupation_counts_before"]) == event["N"]
    assert sum(event["occupation_counts_after"]) == event["N"]
    assert adapted.event["target_count_before"] == event["n_k"]
    assert adapted.event["target_count_after"] == event["n_k_plus_1"]
    assert adapted.event["b"] == event["b"]


def test_dawn_mode_never_calls_the_microscopic_position_scheduler(monkeypatch):
    config = _config(q=1)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("microscopic controller scheduler was called")

    monkeypatch.setattr(
        "mas_cc.games.relational_reasoning.imitation_round_feedback.runtime."
        "sample_controlled_positions",
        forbidden,
    )
    result, _ = _run(config, control=_dawn_control(config))

    assert result.rounds[0].event["dawn_directive_count"] == 4


def test_dawn_persistence_runs_before_the_day_and_never_deletes_history():
    config = _config(q=1)
    options = {
        **dict(config.game.options),
        "epistemic_persistence": 0.0,
    }
    config = replace(config, game=replace(config.game, options=options))
    result, _ = _run(config, control=_dawn_control(config, schedule=SCHEDULE_NEVER))
    event = result.rounds[0].event

    assert event["persistence_deactivated_fact_count"] > 0
    assert event["active_mean_fact_count_before"] == 0.0
    assert event["historical_mean_supporting_fact_coverage_before"] > 0.0
    assert all(
        set(agent.active_fact_ids).issubset(set(agent.known_fact_ids))
        for agent in result.final_state.agents
    )
    assert all(
        set(initial.known_fact_ids).issubset(set(final.known_fact_ids))
        for initial, final in zip(
            result.initial_state.agents, result.final_state.agents, strict=True
        )
    )


def test_directive_report_reply_lineage_reaches_later_exact_acquisition():
    config = _config(q=1)

    class _LineageBallots(_BoardBallots):
        def provider(self, provider_config):
            def factory(request):
                prompt = "\n\n".join(message.content for message in request.messages)
                self.prompts.append(prompt)
                known = [
                    line[2:].split(":", 1)[0]
                    for line in prompt.splitlines()
                    if line.startswith("- f")
                ]
                visible_id = next(
                    (
                        line.split(":", 1)[1].strip()
                        for line in prompt.splitlines()
                        if line.startswith("Message ID:")
                    ),
                    None,
                )
                directive = "Type: DIRECTIVE" in prompt
                return json.dumps(
                    {
                        "vote": "A",
                        "private_reason": "private",
                        "public_message": {
                            "type": "REPORT" if directive and known else "NONE",
                            "text": "Here is exact evidence relevant to the request."
                            if directive and known
                            else None,
                            "shared_fact_id": known[0] if directive and known else None,
                            "reply_to": visible_id if directive and known else None,
                        },
                    }
                )

            return MockLLMProvider(provider_config, response_factory=factory)

    result, _ = _run(
        config,
        control=_dawn_control(config, budget=1),
        ballots=_LineageBallots(),
    )
    lineage = result.rounds[0].event["directive_lineage_events"]

    assert lineage
    assert all(row["origin_directive_id"].startswith("m") for row in lineage)
    assert all(row["reply_message_id"].startswith("m") for row in lineage)
    assert any(row["event_type"] == "acquisition" for row in lineage)
    assert result.rounds[0].event["directive_attributed_acquisitions"] >= 1


@pytest.mark.parametrize("target_count", (0, 1, 3, 6))
def test_soft_policy_monte_carlo_matches_the_configured_probability(target_count):
    control = RelationalRoundBudgetedControl.from_options(
        {
            "target": "correct",
            "sensor_sample_size": 6,
            "threshold": 0.5,
            "beta": 4.0,
            "intervention_budget": 4,
            "advocacy_schedule": SCHEDULE_SOFT,
            "message_mode": RECOMMENDATION_ONLY,
            "controller_actuation_mode": COORDINATION_REQUEST,
            "controller_timing": TIMING_DAWN_ONLY,
        }
    )
    rng = __import__("random").Random(20260902 + target_count)
    trials = 40_000
    acted = sum(
        control.select_action(target_count / 6, rng)[0] == "ADVOCATE_Z"
        for _ in range(trials)
    )
    expected = advocacy_probability(
        target_count / 6, threshold=control.threshold, beta=control.beta
    )
    standard_error = (expected * (1.0 - expected) / trials) ** 0.5

    assert abs(acted / trials - expected) <= 5 * standard_error + 0.001


def test_board_shared_evidence_is_acquired_with_message_provenance():
    config = _config(q=1)

    class _EvidenceBallots(_BoardBallots):
        def provider(self, provider_config):
            def factory(request):
                prompt = "\n\n".join(message.content for message in request.messages)
                self.prompts.append(prompt)
                known = [
                    line[2:].split(":", 1)[0]
                    for line in prompt.splitlines()
                    if line.startswith("- f")
                ]
                shared = known[0] if known else "none"
                return json.dumps(
                    {
                        "vote": "A",
                        "private_reason": "private",
                        "public_message": {
                            "type": "REPORT",
                            "text": "I found a useful exact evidence item.",
                            "shared_fact_id": None if shared == "none" else shared,
                            "reply_to": None,
                        },
                    }
                )

            return MockLLMProvider(provider_config, response_factory=factory)

    result, _ = _run(config, ballots=_EvidenceBallots())
    acquisitions = [
        (event.transition.event, fact_id)
        for event in result.interactions
        for fact_id in event.transition.event["new_peer_fact_ids"]
    ]
    assert acquisitions
    event, fact_id = acquisitions[0]
    focal = result.final_state.relational_agent(AgentId(event["focal_agent_id"]))
    assert focal.fact_provenance[fact_id]["message_id"] in event["sampled_message_ids"]


def test_board_mode_rejects_peer_only_configured_theory():
    config = _config()
    analysis = replace(
        config.analysis,
        enabled=True,
        estimators=("round_sensing_mi",),
        options={
            **dict(config.analysis.options),
            "theoretical_reference": "single_affinity_revised",
        },
    )
    config = replace(config, analysis=analysis)

    with pytest.raises(ValueError, match="social_mode 'board'"):
        validate_configured_analysis(config)


def test_peer_mode_is_the_default():
    config = load_run_config(CONFIG, environment={})
    rules = create_game(config.game).rules(config.game)

    assert rules.social_mode == "peer"
    assert (
        RelationalRoundBudgetedControl.from_options(
            config.control.options
        ).controller_actuation_mode
        == DIRECT_RECOMMENDATION
    )


def test_pilot_artifact_builder_writes_complete_inspection_bundle(tmp_path):
    config = load_run_config(
        "configs/runs/relational_reasoning/blackboard_game/"
        "musr_blackboard_task001_5round_simplified_messages.yaml",
        environment={},
    )
    options = {
        **dict(config.game.options),
        "rounds": 1,
        "initialization": {
            "mode": "explicit",
            "initial_votes": ["ALLOCATION_0"] * 24,
        },
    }
    response = json.dumps(
        {
            "vote": "A",
            "private_reason": "private",
            "public_message": {
                "type": "REPORT",
                "text": "Public report without exact evidence.",
                "shared_fact_id": None,
                "reply_to": None,
            },
        }
    )
    config = replace(
        config,
        game=replace(config.game, horizon=1, options=options),
        llm_provider=replace(
            config.llm_provider,
            type="mock",
            model="deterministic-smoke",
            temperature=0.0,
            max_output_tokens=256,
            options={"response": response},
        ),
        execution=replace(config.execution, repetitions=1, parallelism=1),
        control=replace(
            config.control,
            options={
                **dict(config.control.options),
                "advocacy_schedule": SCHEDULE_ALWAYS,
            },
        ),
        pricing=replace(
            config.pricing,
            mode="offline",
            require_fresh_at_launch=False,
            explicit_unknown_price_override=True,
        ),
        budget=replace(
            config.budget,
            system_max_cost_per_run=None,
            max_cost_per_run=None,
            allow_unbounded_paid_requests=True,
        ),
        storage=replace(config.storage, output_dir=str(tmp_path), overwrite=True),
    )
    result = run_experiment_sync(config, tmp_path, resume=False, show_progress=False)
    summary = json.loads(
        (result.output_dir / "analysis" / "artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["counts"]["reports"] == 24
    assert summary["counts"]["directives"] == 6
    assert summary["counts"]["prompt_attempts"] == 24
    for relative in REQUIRED_OUTPUTS:
        assert (result.output_dir / relative).is_file(), relative
    prompt_paths = list((result.output_dir / "analysis" / "prompts").glob("*.md"))
    assert len(prompt_paths) == 24
    directive_prompt = next(
        path.read_text(encoding="utf-8")
        for path in prompt_paths
        if "Type: DIRECTIVE" in path.read_text(encoding="utf-8")
    )
    assert "Agent 25" in directive_prompt
    assert "controller" not in directive_prompt.lower()

    control_rows = [
        json.loads(line)
        for line in (result.output_dir / "round_control_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(control_rows) == 1
    assert {
        "n_k",
        "Y_k",
        "P_U1_given_Y",
        "U",
        "b",
        "n_k_plus_1",
        "directive_message_ids",
    }.issubset(control_rows[0])
    message_rows = [
        json.loads(line)
        for line in (result.output_dir / "messages.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    public_schema = {
        "message_id",
        "author_public_id",
        "vote",
        "type",
        "text",
        "shared_fact_id",
        "reply_to",
        "created_round",
        "expires_round",
    }
    assert all(public_schema.issubset(row) for row in message_rows)
    assert all(row["vote"] for row in message_rows)
    assert {
        row["author_public_id"] for row in message_rows if row["type"] == "DIRECTIVE"
    } == {"Agent 25"}
    dashboard = (result.output_dir / "analysis/dashboard/index.html").read_text(
        encoding="utf-8"
    )
    assert all(phase in dashboard for phase in ("NIGHT", "DAWN", "DAY", "END OF DAY"))
    assert (
        result.output_dir
        / "relational_imitation_round_feedback_analysis"
        / "round_information_estimates.csv"
    ).is_file()
