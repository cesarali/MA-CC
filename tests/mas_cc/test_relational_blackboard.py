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
    RelationalRoundBudgetedControl,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.prompts import (
    BlackboardBallotContract,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.runtime import (
    run_relational_imitation_round_feedback_game,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.state import (
    BOARD_MESSAGE_TYPES,
    BlackboardMessage,
    BlackboardState,
)
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider

pytestmark = pytest.mark.skipif(
    not (DEFAULT_TASK_DATASET_DIR / "task_0001.json").exists(),
    reason="the relational example dataset is not present",
)

CONFIG = (
    "configs/runs/relational_reasoning/"
    "relational_imitation_round_feedback_no_control_smoke.yaml"
)


def _config(*, rounds=1, q=1, lifetime=1):
    config = load_run_config(CONFIG, environment={})
    options = {
        **dict(config.game.options),
        "rounds": rounds,
        "social_group_size": q,
        "social_mode": "board",
        "board": {
            "sampling": "uniform",
            "message_lifetime_rounds": lifetime,
            "exclude_self_authored": True,
            "allow_no_post": True,
        },
    }
    prompt = replace(config.prompt, prompt_family="relational_blackboard_ballot")
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
                    "reason": "private reasoning that must not be shared",
                    "shared_fact_id": "none",
                    "public_message": (
                        {
                            "type": "RESULT",
                            "text": "I compared my available evidence.",
                            "reply_to": None,
                        }
                        if self.post
                        else None
                    ),
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


def test_board_state_lifetime_and_serialization():
    message = BlackboardMessage(
        message_id="m1",
        author_id="agent_001",
        message_type="QUESTION",
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
            message_type="CLAIM",
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


def test_all_message_types_validate_and_replies_need_visible_target():
    for kind in BOARD_MESSAGE_TYPES:
        reply_to = "m1" if kind in {"REPLY", "CORRECTION"} else None
        contract = BlackboardBallotContract(
            allowed_values=("A", "B"),
            options={"fact_ids": (), "relations": (), "visible_message_ids": ("m1",)},
        )
        response = json.dumps(
            {
                "vote": "A",
                "reason": "private",
                "shared_fact_id": "none",
                "public_message": {
                    "type": kind,
                    "text": "public",
                    "reply_to": reply_to,
                },
            }
        )
        assert contract.validate(response).valid

    invalid = response.replace('"reply_to": "m1"', '"reply_to": "missing"')
    assert not contract.validate(invalid).valid


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
    assert all(message.message_type == "REQUEST" for message in controller_messages)
    assert all(message.shared_fact_id is None for message in controller_messages)
    assert round_record["controller_posts"] == 4
    assert round_record["controller_message_exposures"] <= 4
    assert round_record["theory_status"] == "reference_only"


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
                        "reason": "private",
                        "shared_fact_id": shared,
                        "public_message": {
                            "type": "RESULT",
                            "text": "I found a useful exact evidence item.",
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
