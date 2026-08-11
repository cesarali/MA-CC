"""Acceptance tests for the v2 prompt and controller revision.

Covers Parts 1-4 of `docs/tdd/misselaneous/11082026_prompt_modifications_v2.md`.
Two properties are load-bearing across the whole file:

- **v1 is frozen.** Every controlled run recorded before 2026-08-11 was produced
  under it, so `test_v1_prompt_text_is_frozen` pins the exact characters. If you
  "improve" v1 wording, that test fails, which is the intended behaviour.
- **v2 is an intervention, not a fix.** It is opt-in for that reason, and the
  tests assert it stays opt-in.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import replace

import pytest

from mas_cc.config import load_run_config
from mas_cc.control import NoneControl, create_control
from mas_cc.games import create_game
from mas_cc.games.hidden_bench.data import DEFAULT_CORPUS_ROOT, load_task_set, normalized_text
from mas_cc.games.hidden_bench.imitation import run_hidden_bench_imitation_game
from mas_cc.games.hidden_bench.imitation.controller import (
    FORBIDDEN_MESSAGE_TERMS,
    ThresholdTargetControl,
    advocacy_message,
    check_frames,
    check_peer_style,
    peer_advocacy_message,
)
from mas_cc.games.hidden_bench.imitation.game import HiddenBenchImitationGame
from mas_cc.games.hidden_bench.imitation.prompts import (
    ASYMMETRY_NOTICE,
    RESPONSE_STYLE_V2,
    PromptStyle,
    PrivateHistoryBlock,
    bind_initial_prompt,
    bind_message_prompt,
    bind_update_prompt,
    message_interaction_text,
    scenario_for_variant,
)
from mas_cc.llm_runtime.providers import create_llm_provider
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider

pytestmark = pytest.mark.skipif(
    not (DEFAULT_CORPUS_ROOT / "canonical" / "tasks.json").exists(),
    reason="HiddenBench corpus is not present",
)

CLASSICAL_CONFIG = "configs/runs/hidden_bench/hidden_bench_imitation_classical.yaml"
REASONING_CONFIG = "configs/runs/hidden_bench/hidden_bench_imitation_reasoning_mock.yaml"

V1 = PromptStyle()
V2 = PromptStyle(version=2)

HISTORY = (
    {
        "event": 15,
        "received_message": "the roads are open",
        "own_message": "I still favour East Town",
        "own_vote_after": "East Town",
    },
)


def _config(path: str, *, horizon: int | None = None, **options):
    config = load_run_config(path, environment={})
    return replace(
        config,
        game=replace(
            config.game,
            horizon=config.game.horizon if horizon is None else horizon,
            options={**dict(config.game.options), **options},
        ),
    )


def _run(config, *, provider=None, control=None):
    game = create_game(config.game)
    provider = provider or create_llm_provider(config.llm_provider)
    return asyncio.run(
        run_hidden_bench_imitation_game(game, config, provider, control=control)
    )


def _controlled(config, **control_options):
    return replace(
        config,
        control=replace(
            config.control,
            mechanism="threshold_target",
            options={
                "target": "correct",
                "sensor_sample_size": 1,
                "threshold": 0.5,
                **control_options,
            },
        ),
    )


# --------------------------------------------------------------------------
# Part 1 - making information actually surface
# --------------------------------------------------------------------------


def test_v1_prompt_text_is_frozen():
    """The exact characters every pre-2026-08-11 controlled run was produced under."""

    assert PrivateHistoryBlock(version=1).bind(HISTORY).render() == (
        "Earlier private interactions you participated in:\n"
        "- Event 15: partner/controller said the roads are open; you committed East Town."
    )
    assert message_interaction_text(style=V1, allow_relay=True, dialogue=()) == (
        "This is a private exchange with one participant. "
        "You may relay information learned in earlier interactions. Speak now."
    )
    rendered = bind_update_prompt(
        scenario="s",
        information=("f",),
        history=(),
        possible_answers=("A", "B", "C"),
        current_vote="A",
        dialogue=(),
        controller_message="advocacy",
        style=V1,
    ).compile()
    assert "External controller message: advocacy" in "\n".join(
        block.content for block in rendered.blocks
    )


def test_v2_response_style_drops_the_length_cap_and_asks_for_something_new():
    """§1.1. The two-sentence cap is the single biggest suppressor of disclosure."""

    assert "one or two sentences" in V1.message_response_style
    assert "two to four sentences" in V2.message_response_style
    assert "not already mentioned in an earlier message" in RESPONSE_STYLE_V2
    assert "option you are voting for" in RESPONSE_STYLE_V2


def test_v2_response_style_conditions_speaking_turns_only():
    """A vote turn is not a speaking turn.

    Regression test for run `...-control-10-v2-20260840`, where every episode
    died on `must contain a JSON object with vote and rationale` because the
    speaking instruction inflated the `rationale` field past
    `max_output_tokens` and the JSON was truncated with no closing brace. It
    also keeps the measurement instrument unconditioned, which is the same rule
    `bind_vote_prompt` follows in the vanilla game.
    """

    assert V2.vote_response_style == V1.vote_response_style == V1.message_response_style

    def styles(prompt):
        return next(
            block.content for block in prompt.compile().blocks if block.name == "response_style"
        )

    speaking = bind_message_prompt(
        scenario="s", information=("f",), history=(), dialogue=(), allow_relay=True, style=V2
    )
    voting = bind_update_prompt(
        scenario="s",
        information=("f",),
        history=(),
        possible_answers=("A", "B", "C"),
        current_vote="A",
        dialogue=({"is_self": False, "message": "peer"},),
        style=V2,
    )
    initial = bind_initial_prompt(
        scenario="s", information=("f",), possible_answers=("A", "B", "C"), style=V2
    )
    assert styles(speaking) == RESPONSE_STYLE_V2
    assert styles(voting) == V1.message_response_style
    assert styles(initial) == V1.message_response_style


def test_v2_history_shows_the_agents_own_message_and_hides_the_source():
    """§1.3 plus leak 2 of §2.1, which are the same rendered line."""

    rendered = PrivateHistoryBlock(version=2).bind(HISTORY).render()
    assert "you replied I still favour East Town" in rendered
    assert "the other participant said the roads are open" in rendered
    assert "partner/controller" not in rendered
    assert "controller" not in rendered.lower()


def test_v2_history_omits_the_reply_clause_when_the_agent_did_not_speak():
    """A control event has no peer exchange, so there is no own message to show."""

    silent = ({**HISTORY[0], "own_message": None},)
    rendered = PrivateHistoryBlock(version=2).bind(silent).render()
    assert "you replied" not in rendered
    assert rendered.endswith("the other participant said the roads are open; you committed East Town.")


def test_v2_interaction_stops_nudging_toward_recycling():
    """§1.4. "Relay information learned" points at facts already circulating."""

    text = message_interaction_text(style=V2, allow_relay=True, dialogue=())
    assert "relay information learned" not in text.lower()
    assert "anything you have not yet told anyone" in text
    assert "say how you are voting" in text


def test_v2_no_relay_variant_still_asks_for_undisclosed_information():
    text = message_interaction_text(style=V2, allow_relay=False, dialogue=())
    assert "only state information that was given to you directly" in text
    assert "anything you have not yet told anyone" in text


def test_asymmetry_notice_is_opt_in_and_never_marks_which_facts_are_private():
    """§1.2. Telling agents *which* facts are unique destroys the paradigm."""

    style = PromptStyle(version=2, inform_asymmetry=True)
    facts = ("shared fact", "private fact")
    blocks = {
        block.name: block.content
        for block in bind_message_prompt(
            scenario="s",
            information=facts,
            history=(),
            dialogue=(),
            allow_relay=True,
            style=style,
        ).compile().blocks
    }
    information = blocks["information"]
    assert ASYMMETRY_NOTICE in information
    # Every fact renders through the identical "- " bullet: nothing in the block
    # lets an agent recover which of its own facts nobody else holds.
    assert "- shared fact\n- private fact" in information
    assert "private" not in information.replace("private fact", "")

    without = bind_message_prompt(
        scenario="s",
        information=facts,
        history=(),
        dialogue=(),
        allow_relay=True,
        style=V2,
    ).compile()
    assert ASYMMETRY_NOTICE not in "\n".join(block.content for block in without.blocks)


def test_asymmetry_notice_cannot_be_bolted_onto_frozen_v1():
    with pytest.raises(ValueError, match="requires prompt_version 2"):
        PromptStyle(version=1, inform_asymmetry=True)


def test_scenario_variant_2_removes_only_the_coordination_bonus():
    """§1.5. The payoff clause is the imitation coupling strength, in English."""

    task = load_task_set("vanilla").by_name("evacuation_north_hill")
    original = task.description_for(4)
    kept, changed_v1 = scenario_for_variant(original, 1)
    stripped, changed_v2 = scenario_for_variant(original, 2)

    assert kept == original and changed_v1 is False
    assert changed_v2 is True
    assert "coordinating with others is critical" not in stripped
    assert "you will earn an additional $1" not in stripped
    # The individual-accuracy payoff and the options survive: only the
    # instruction to conform is removed.
    assert "If you choose the correct location, you will earn $1." in stripped
    for option in task.possible_answers:
        assert option in stripped


def test_scenario_variant_2_reports_a_no_op_on_tasks_without_the_clause():
    """58 of 65 corpus tasks never had the clause; that is not a failure."""

    task = load_task_set("vanilla").by_name("scientists_animal_base_decision")
    text, changed = scenario_for_variant(task.description_for(4), 2)
    assert changed is False
    assert text == task.description_for(4)

    state = HiddenBenchImitationGame().initialize(
        _config(CLASSICAL_CONFIG, scenario_variant=2).game, 11
    )
    assert state.data["task"]["scenario_variant"] == 2
    assert state.data["task"]["coordination_bonus_removed"] is True


def test_prompt_version_defaults_to_v1_and_is_recorded_on_every_episode():
    """v2 shifts the reasoning arm relative to the classical one, so it is opt-in."""

    game = HiddenBenchImitationGame()
    assert game.rules(_config(CLASSICAL_CONFIG).game).prompt_version == 1
    rules = game.initialize(_config(CLASSICAL_CONFIG, prompt_version=2).game, 7).data["rules"]
    assert rules["prompt_version"] == 2
    assert rules["inform_asymmetry"] is False
    assert rules["scenario_variant"] == 1


def test_a_prompt_version_disagreement_between_config_sections_is_refused():
    config = _config(REASONING_CONFIG, horizon=1, prompt_version=2)
    with pytest.raises(ValueError, match="they must match"):
        _run(config)


# --------------------------------------------------------------------------
# Part 2 - making the controller invisible
# --------------------------------------------------------------------------


def test_v1_controller_announces_itself_and_v2_does_not():
    """§2.1 leak 1. v1 is kept only so its numbers stay reproducible."""

    assert "external controller" in advocacy_message("North Hill").lower()

    task = load_task_set("vanilla").by_name("evacuation_north_hill")
    rng = random.Random(0)
    for _ in range(40):
        message = peer_advocacy_message("North Hill", task.shared_information, rng).text
        lowered = message.lower()
        for term in FORBIDDEN_MESSAGE_TERMS:
            assert term not in lowered, message


def test_v2_controller_frames_and_messages_obey_the_style_rules():
    """§2.4, as assertions rather than prose."""

    assert check_frames() == ()
    task = load_task_set("vanilla").by_name("evacuation_east_town")
    rng = random.Random(1)
    for _ in range(40):
        advocacy = peer_advocacy_message("East Town", task.shared_information, rng)
        assert check_peer_style(advocacy.text, target="East Town") == ()
        assert advocacy.text.endswith("I'm voting **East Town**.")
        assert "**East Town**" in advocacy.text


def test_v2_controller_never_invents_evidence():
    """§2.2. A controller that manufactures facts turns "the population moved
    toward Z" into "an agent believed a new fact", and contaminates truth too."""

    task = load_task_set("vanilla").by_name("evacuation_north_hill")
    shared = [normalized_text(fact) for fact in task.shared_information]
    rng = random.Random(2)
    for _ in range(60):
        advocacy = peer_advocacy_message("North Hill", task.shared_information, rng)
        surface = normalized_text(advocacy.text)
        for hidden in task.hidden_information:
            assert normalized_text(hidden) not in surface
        assert advocacy.fact_index is not None
        assert shared[advocacy.fact_index] in surface


def test_v2_controller_falls_back_to_a_factless_frame_without_shared_information():
    advocacy = peer_advocacy_message("North Hill", (), random.Random(3))
    assert advocacy.fact_index is None
    assert check_peer_style(advocacy.text, target="North Hill") == ()


def test_v2_uses_a_paraphrase_bank_and_logs_which_variant_fired():
    """§2.5. Agent-000 sees five past events at once; one repeated sentence reads
    as a bot even when the wording is natural."""

    config = _controlled(
        _config(CLASSICAL_CONFIG, horizon=60), template_version=2, threshold=1.0
    )
    result = _run(config, control=create_control(config.control))
    controlled = [
        item.transition.event
        for item in result.interactions
        if item.transition.event["controller_action"] == "ADVOCATE_Z"
    ]
    assert controlled, "threshold 1.0 must produce advocacy events"
    assert len({event["controller_template_id"] for event in controlled}) >= 3
    assert all(event["controller_template_version"] == 2 for event in controlled)
    assert all(event["controller_fact_index"] is not None for event in controlled)


def test_template_version_1_stays_the_default_and_is_recorded():
    """A config that does not ask for v2 keeps the exact v1 intervention text."""

    control = ThresholdTargetControl.from_options({"target": "correct"})
    assert control.template_version == 1

    state = HiddenBenchImitationGame().initialize(_config(CLASSICAL_CONFIG).game, 5)
    # Sampling the whole population makes the action deterministic: the shipped
    # initial state puts one of four agents on the correct answer, so support is
    # 0.25 and the threshold policy always advocates.
    signal = replace(control, sensor_sample_size=len(state.agents)).interaction_signal(
        agent_id=state.agents[0].agent_id,
        interaction_index=1,
        state=state,
        rng=random.Random(0),
    )
    assert signal.action == "ADVOCATE_Z"
    assert signal.message == advocacy_message(state.correct_answer)
    assert signal.metadata["message_template_version"] == 1
    assert signal.metadata["message_template_id"] == "labelled-advocacy-v1"
    assert signal.metadata["message_fact_index"] is None


def test_v2_update_prompt_renders_the_controller_turn_as_an_ordinary_exchange():
    """Leak 1 is only closed if the *prompt* stops labelling the turn as well."""

    compiled = bind_update_prompt(
        scenario="s",
        information=("f",),
        history=(),
        possible_answers=("A", "B", "C"),
        current_vote="A",
        dialogue=(),
        controller_message="I'm voting **B**.",
        style=V2,
    ).compile()
    text = "\n".join(block.content for block in compiled.blocks)
    assert "External controller message" not in text
    assert "controller" not in text.lower()
    assert "Private exchange:\nThe other participant: I'm voting **B**." in text


# --------------------------------------------------------------------------
# Part 3 - event scheduling
# --------------------------------------------------------------------------


def _reasoning_provider(config):
    def respond(request):
        joined = "\n".join(message.content for message in request.messages)
        if "following JSON format" in joined:
            # A wrong-answer population, so a target-seeking controller has
            # something to correct and actually emits advocacy events.
            return '{"vote": "East Town", "rationale": "mock"}'
        return "The bridge to West City is still passable."

    return MockLLMProvider(config.llm_provider, response_factory=respond)


def _four_cells(horizon: int = 8):
    reasoning = _config(REASONING_CONFIG, horizon=horizon, messages_per_agent=1)
    classical = _config(CLASSICAL_CONFIG, horizon=horizon)
    return {
        "A_reasoning_off": (reasoning, NoneControl()),
        "B_reasoning_on": (
            _controlled(reasoning, template_version=2, threshold=1.0),
            create_control(_controlled(reasoning, template_version=2, threshold=1.0).control),
        ),
        "C_classical_off": (classical, NoneControl()),
        "D_classical_on": (
            _controlled(classical, template_version=2, threshold=1.0),
            create_control(_controlled(classical, template_version=2, threshold=1.0).control),
        ),
    }


def test_total_events_are_identical_across_all_four_cells():
    """§3.4 invariant 1. Matched initial conditions do not buy a matched
    comparison if the arms run different numbers of events."""

    counts = {}
    for name, (config, control) in _four_cells().items():
        provider = _reasoning_provider(config) if "reasoning" in name else None
        counts[name] = len(_run(config, provider=provider, control=control).interactions)
    assert set(counts.values()) == {8}, counts


def test_control_replaces_a_peer_interaction_in_both_dynamics_modes():
    """§3.1 and §3.4 invariant 2. Replacement, never addition - and the same way
    in both arms, otherwise B - D is confounded by event counts."""

    for name, (config, control) in _four_cells().items():
        if name.endswith("_off"):
            continue
        provider = _reasoning_provider(config) if "reasoning" in name else None
        events = [
            item.transition.event
            for item in _run(config, provider=provider, control=control).interactions
        ]
        advocacy = sum(event["controller_action"] == "ADVOCATE_Z" for event in events)
        peers = sum(event["peer_interaction"] for event in events)
        assert advocacy > 0, name
        assert peers == len(events) - advocacy, name
        # A control event consumed the peer slot; it did not get an extra one.
        for event in events:
            if event["controller_action"] == "ADVOCATE_Z":
                assert event["peer_agent_id"] is None, name
                assert event["sampled_peer_agent_id"] is not None, name


def test_the_event_schedule_replays_across_control_on_and_off():
    """§3.4 invariant 3, which goes beyond replaying X_0: the same focal agent
    meets the same partner in the same order whether or not control is on."""

    for mode, path in (("reasoning", REASONING_CONFIG), ("classical", CLASSICAL_CONFIG)):
        base = _config(path, horizon=8, **({"messages_per_agent": 1} if mode == "reasoning" else {}))
        controlled = _controlled(base, template_version=2)
        schedules = []
        for config, control in ((base, NoneControl()), (controlled, create_control(controlled.control))):
            provider = _reasoning_provider(config) if mode == "reasoning" else None
            schedules.append(
                [
                    (
                        item.transition.event["focal_agent_id"],
                        item.transition.event["sampled_peer_agent_id"],
                    )
                    for item in _run(config, provider=provider, control=control).interactions
                ]
            )
        assert schedules[0] == schedules[1], mode


# --------------------------------------------------------------------------
# Part 4 - what to log
# --------------------------------------------------------------------------


def test_disclosure_events_record_which_fact_entered_circulation_from_whom_and_when():
    """The cheapest real result available: shared-vs-unshared diffusion curves."""

    config = _config(REASONING_CONFIG, horizon=4, messages_per_agent=1, prompt_version=2)
    config = replace(config, prompt=replace(config.prompt, prompt_version=2))
    game = create_game(config.game)
    hidden = game.initialize(config.game, config.execution.seed).hidden_information
    quoted = hidden[0]

    def respond(request):
        joined = "\n".join(message.content for message in request.messages)
        if "following JSON format" in joined:
            return '{"vote": "West City", "rationale": "mock"}'
        return quoted

    result = _run(config, provider=MockLLMProvider(config.llm_provider, response_factory=respond))
    events = [item.transition.event for item in result.interactions]
    first = events[0]
    disclosures = first["disclosure_events"]
    assert disclosures, "a message quoting a hidden fact must register a disclosure"
    assert {item["fact_index"] for item in disclosures} == {0}
    assert all(item["interaction_index"] == 1 for item in disclosures)
    assert any(item["first_disclosure"] for item in disclosures)
    speakers = {item["speaker_agent_id"] for item in disclosures}
    assert speakers <= {first["focal_agent_id"], first["peer_agent_id"]}
    # The fact only enters circulation once, however often it is repeated.
    assert not any(
        item["first_disclosure"] for event in events[1:] for item in event["disclosure_events"]
    )
    assert first["disclosure_reach"][0] >= 1


def test_every_event_records_the_message_text_needed_for_offline_fact_detection():
    """Detection runs as a separate pass: asking the agent to list the facts it
    used would itself increase disclosure."""

    config = _config(REASONING_CONFIG, horizon=3, messages_per_agent=1)
    result = _run(config, provider=_reasoning_provider(config))
    for item in result.interactions:
        event = item.transition.event
        assert event["focal_message"]
        assert event["peer_message"]
        assert event["H_vote"] is not None
        assert event["prompt_version"] == 1

    focal_id = result.interactions[0].transition.event["focal_agent_id"]
    memory = result.final_state.hidden_bench_agent(
        next(agent.agent_id for agent in result.final_state.agents if str(agent.agent_id) == focal_id)
    ).memory
    assert memory[0]["own_message"]
