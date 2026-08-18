"""Scientific invariants of the atomic public-ballot reasoning kernel.

One focal update is one provider call returning one `{vote, reason}` object.
The vote is the agent's actual new committed action, the reason is the public
text later focal agents read, and control changes exactly one of the `q` social
inputs rather than adding a message, a call, or a second vote field.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import replace

import pytest

from mas_cc.config import load_run_config
from mas_cc.control import Control, RoundControlSignal
from mas_cc.games import create_game
from mas_cc.games.hidden_bench.data import DEFAULT_CORPUS_ROOT
from mas_cc.games.hidden_bench.imitation.controller import ADVOCATE_TARGET
from mas_cc.games.hidden_bench.imitation_round_feedback.controller import (
    RoundSoftTargetBudgetedControl,
)
from mas_cc.games.hidden_bench.imitation_round_feedback.prompts import (
    CONTROL_EVIDENCE_PREFIX,
    DECISION_BASIS_SOCIAL,
    MAX_REASON_CHARACTERS,
    NO_PREVIOUS_REASON,
    SOCIAL_ENVIRONMENT,
    PublicBallotContract,
    agent_label,
    parse_public_ballot_update,
    render_control_reason,
    render_social_source,
    sanitize_scenario,
)
from mas_cc.games.hidden_bench.imitation_round_feedback.runtime import (
    CONTROL_SOURCE_ID,
    run_hidden_bench_imitation_round_feedback_game,
)
from mas_cc.games.hidden_bench.imitation_round_feedback.state import get_public_reason
from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider

pytestmark = pytest.mark.skipif(
    not (DEFAULT_CORPUS_ROOT / "canonical" / "tasks.json").exists(),
    reason="HiddenBench corpus is not present",
)

REASONING = (
    "configs/runs/hidden_bench/"
    "hidden_bench_imitation_round_feedback_reasoning_smoke.yaml"
)
OPTIONS = ("West City", "East Town", "North Hill")
FORBIDDEN = ("controller", "external", "intervention", "experiment", "simulation")


class _BudgetedControl(Control):
    """A fixed round decision: always advocate, always the same budget."""

    sensor_sample_size = 2

    def __init__(
        self,
        *,
        target: str = "West City",
        budget: int = 2,
        evidence_mode: str = "none",
    ) -> None:
        self.intervention_budget = budget
        self.target = target
        self.evidence_mode = evidence_mode

    def override(self, **_):
        return None

    def round_signal(self, *, round_index, state, rng):
        sampled = rng.sample(list(state.agents), self.sensor_sample_size)
        opinions = [str(agent.attributes["committed_action"]) for agent in sampled]
        return RoundControlSignal(
            action=ADVOCATE_TARGET,
            target=self.target,
            # Deliberately a message the reasoning path must ignore: the
            # controller's visible contribution is its rendered public reason.
            message="unused legacy controller message",
            observation={
                "sampled_agent_ids": [str(agent.agent_id) for agent in sampled],
                "sampled_opinions": opinions,
                "sampled_opinion_counts": dict(Counter(opinions)),
                "sample_size": self.sensor_sample_size,
            },
            metadata={"policy": "test", "advocacy_probability": 1.0},
        )


class _Ballots:
    """A provider that records every prompt and votes a scripted cycle."""

    def __init__(self, *, votes=OPTIONS) -> None:
        self.prompts: list[str] = []
        self.votes = tuple(votes)

    def provider(self, config):
        def factory(request):
            index = len(self.prompts)
            self.prompts.append(
                "\n\n".join(message.content for message in request.messages)
            )
            return json.dumps(
                {
                    "vote": self.votes[index % len(self.votes)],
                    "reason": f"public reason {index}",
                }
            )

        return MockLLMProvider(config, response_factory=factory)


def _config(*, rounds=1, q=2, initialization=None, n_agents=None):
    config = load_run_config(REASONING, environment={})
    options = dict(config.game.options)
    options["rounds"] = rounds
    options["social_group_size"] = q
    if initialization is not None:
        options["initialization"] = initialization
    population = config.game.population_size if n_agents is None else n_agents
    if n_agents is not None:
        options["n_agents"] = n_agents
    return replace(
        config,
        game=replace(
            config.game, horizon=rounds, population_size=population, options=options
        ),
    )


def _run(config, *, control=None, ballots=None):
    ballots = ballots or _Ballots()
    result = asyncio.run(
        run_hidden_bench_imitation_round_feedback_game(
            create_game(config.game),
            config,
            ballots.provider(config.llm_provider),
            control=control,
        )
    )
    return result, ballots


def _social_block(prompt: str) -> list[str]:
    """The rendered sources of one prompt, one entry per social slot."""

    if "CURRENT SOCIAL INFORMATION" not in prompt:
        return []
    body = prompt.split("CURRENT SOCIAL INFORMATION\n\n", 1)[1]
    body = body.split("\n\nDECISION\n", 1)[0]
    return body.split("\n\n")


# ---- provider demand ---------------------------------------------------


@pytest.mark.parametrize(
    ("initialization", "extra_calls"),
    [
        ({"mode": "uniform_random"}, 0),
        ({"mode": "local_vote"}, 4),
    ],
    ids=["uniform_random", "local_vote"],
)
def test_one_provider_call_per_focal_update_and_no_peer_dialogue(
    initialization, extra_calls
):
    config = _config(rounds=2, initialization=initialization)
    result, ballots = _run(config, control=_BudgetedControl())

    updates = config.game.options["rounds"] * config.game.population_size
    assert len(ballots.prompts) == updates + extra_calls
    assert result.logical_decisions == updates + extra_calls
    assert all(len(item.decisions) == 1 for item in result.interactions)
    assert all("Private exchange" not in prompt for prompt in ballots.prompts)
    assert all("turn to speak" not in prompt for prompt in ballots.prompts)


def test_a_controlled_update_costs_the_same_single_call_as_an_uncontrolled_one():
    config = _config(rounds=1)
    result, ballots = _run(config, control=_BudgetedControl(budget=2))

    controlled = [
        item for item in result.interactions if item.transition.event["controlled_slot"]
    ]
    assert len(controlled) == 2
    assert {len(item.decisions) for item in result.interactions} == {1}
    assert len(ballots.prompts) == config.game.population_size


# ---- the atomic ballot -------------------------------------------------


def test_the_returned_vote_is_the_committed_action_and_the_only_vote_field():
    result, _ = _run(_config(rounds=2), control=_BudgetedControl())

    for item in result.interactions:
        event = item.transition.event
        decision = item.decisions[0]
        parsed = json.loads(decision.attempts[-1].response.content)
        focal = event["focal_agent_id"]
        committed = next(
            agent.committed_action
            for agent in item.transition.next_state.agents
            if str(agent.agent_id) == focal
        )
        assert decision.action.value == parsed["vote"] == committed
        assert event["focal_vote_after"] == parsed["vote"]
        # No second opinion variable anywhere in the persisted row.
        assert not {key for key in event if "spoken" in key or "stated" in key}


def test_the_returned_reason_becomes_the_agents_persistent_public_reason():
    result, _ = _run(_config(rounds=3), control=_BudgetedControl())

    published: dict[str, str] = {}
    for item in result.interactions:
        event = item.transition.event
        focal = event["focal_agent_id"]
        # Every ordinary source shows exactly the reason its agent last
        # published, and nothing at all before its first update.
        for source in event["social_sources"]:
            if source["source_type"] != "ordinary":
                continue
            assert source["reason"] == published.get(source["source_id"])
        assert event["focal_reason_before"] == published.get(focal)
        published[focal] = event["focal_reason_after"]

    for agent in result.final_state.agents:
        assert get_public_reason(agent) == published.get(str(agent.agent_id))


def test_update_indices_and_public_agent_labels_are_deterministic():
    result, _ = _run(_config(rounds=3), control=_BudgetedControl())

    labels: dict[str, str] = {}
    for expected, item in enumerate(result.interactions):
        event = item.transition.event
        assert event["global_update_index"] == expected
        assert expected == (
            event["round_index"] * event["N"] + event["within_round_index"]
        )
        focal = event["focal_agent_id"]
        labels.setdefault(focal, agent_label(focal))
        assert labels[focal] == agent_label(focal)

    assert agent_label("agent-004") == "Agent 5"


def test_a_published_reason_is_shown_verbatim_to_the_next_focal_agent():
    result, ballots = _run(_config(rounds=3), control=_BudgetedControl())

    seen = 0
    for prompt, item in zip(ballots.prompts, result.interactions, strict=True):
        for source, rendered in zip(
            item.transition.event["social_sources"],
            _social_block(prompt),
            strict=True,
        ):
            assert rendered == render_social_source(source)
            if source["source_type"] == "ordinary" and source["reason"]:
                assert source["reason"] in prompt
                seen += 1
    assert seen > 0


def test_only_the_focal_agent_changes_at_one_microscopic_position():
    result, _ = _run(_config(rounds=2), control=_BudgetedControl())

    for item in result.interactions:
        event = item.transition.event
        changed = [
            index
            for index, pair in enumerate(
                zip(event["population_state_before"], event["population_state_after"])
            )
            if pair[0] != pair[1]
        ]
        assert len(changed) <= 1


# ---- social slots ------------------------------------------------------


@pytest.mark.parametrize("q", [1, 2])
def test_control_replaces_one_social_slot_and_never_adds_a_source(q):
    result, ballots = _run(_config(rounds=2, q=q), control=_BudgetedControl())

    controlled = 0
    for prompt, item in zip(ballots.prompts, result.interactions, strict=True):
        event = item.transition.event
        sources = event["social_sources"]
        control = [item for item in sources if item["source_type"] == "control"]
        ordinary = [item for item in sources if item["source_type"] == "ordinary"]
        expected = int(event["controlled_slot"])
        assert len(sources) == len(_social_block(prompt)) == q
        assert len(control) == expected
        assert len(ordinary) == q - expected
        assert len({str(item["source_id"]) for item in ordinary}) == len(ordinary)
        controlled += expected
    assert controlled > 0


def test_q2_controlled_shows_exactly_one_peer_and_one_controller_like_source():
    result, ballots = _run(_config(rounds=2, q=2), control=_BudgetedControl(budget=4))

    controlled = [
        (prompt, item.transition.event)
        for prompt, item in zip(ballots.prompts, result.interactions, strict=True)
        if item.transition.event["controlled_slot"]
    ]
    assert controlled
    for prompt, event in controlled:
        kinds = [source["source_type"] for source in event["social_sources"]]
        assert sorted(kinds) == ["control", "ordinary"]
        assert len(_social_block(prompt)) == 2


# ---- the controller ----------------------------------------------------


def test_the_controller_reason_is_a_deterministic_fact_free_function_of_the_target():
    config = _config(rounds=2)
    task = create_game(config.game).initialize(config.game, config.execution.seed).task
    result, _ = _run(config, control=_BudgetedControl(target="West City"))

    sources = [
        source
        for item in result.interactions
        for source in item.transition.event["social_sources"]
        if source["source_type"] == "control"
    ]
    assert sources
    for source in sources:
        assert source["vote"] == "West City"
        assert source["reason"] == render_control_reason("West City")
        assert source["source_id"] == CONTROL_SOURCE_ID
        # It carries no HiddenBench evidence, shared or hidden.
        assert not any(
            fact in source["reason"]
            for fact in (*task["hidden_information"], task["description"])
        )
    assert len({source["reason"] for source in sources}) == 1


def test_the_controller_appears_as_one_persistent_ordinary_participant():
    config = _config(rounds=2)
    result, ballots = _run(config, control=_BudgetedControl())

    labels = set()
    for prompt, item in zip(ballots.prompts, result.interactions, strict=True):
        for source in item.transition.event["social_sources"]:
            if source["source_type"] == "control":
                labels.add(source["label"])
                assert f"{source['label']}\nVote: {source['vote']}\n" in prompt
    assert len(labels) == 1
    label = labels.pop()
    assert label.startswith("Agent ")
    assert label not in {
        agent_label(agent.agent_id) for agent in result.final_state.agents
    }


def test_no_prompt_ever_identifies_a_source_as_control_or_as_an_experiment():
    result, ballots = _run(_config(rounds=2), control=_BudgetedControl())

    for prompt in ballots.prompts:
        lowered = prompt.lower()
        for word in FORBIDDEN:
            assert word not in lowered, f"{word!r} leaked into a focal prompt"
        assert "latent evidence types" not in lowered
        assert "you will discuss" not in lowered
        assert "after the discussion" not in lowered
        assert "chat will" not in lowered
        assert "participating in a study" not in lowered
        assert CONTROL_SOURCE_ID not in prompt
        assert "unused legacy controller message" not in prompt

    # Whatever the corpus says, it says identically at a controlled and an
    # uncontrolled update: nothing outside the social block reacts to control.
    scenarios = {
        prompt.split("\nTASK\n", 1)[1].split("\nYOUR PRIVATE INFORMATION\n", 1)[0]
        for prompt in ballots.prompts
    }
    assert len(scenarios) == 1
    assert any(item.transition.event["controlled_slot"] for item in result.interactions)


# ---- controller evidence (`control.options.evidence_mode`) -------------


def _control_sources(result):
    return [
        source
        for item in result.interactions
        for source in item.transition.event["social_sources"]
        if source["source_type"] == "control"
    ]


def _shared_facts(config):
    state = create_game(config.game).initialize(config.game, config.execution.seed)
    return tuple(state.agents[0].attributes["shared_information"])


def _round_control_reasons(result, round_index):
    return {
        source["reason"]
        for item in result.interactions
        if item.transition.event["round_index"] == round_index
        for source in item.transition.event["social_sources"]
        if source["source_type"] == "control"
    }


def test_evidence_mode_defaults_to_none_and_is_validated():
    base = {
        "target": "correct",
        "sensor_sample_size": 2,
        "intervention_budget": 2,
        "template_version": 3,
    }
    assert RoundSoftTargetBudgetedControl.from_options(base).evidence_mode == "none"
    assert (
        RoundSoftTargetBudgetedControl.from_options(
            {**base, "evidence_mode": "shared_fact"}
        ).evidence_mode
        == "shared_fact"
    )
    with pytest.raises(ConfigurationError):
        RoundSoftTargetBudgetedControl.from_options({**base, "evidence_mode": "hidden_fact"})


def test_evidence_mode_none_renders_the_historical_fact_free_advocacy():
    config = _config(rounds=2)
    bare, _ = _run(config, control=_BudgetedControl(evidence_mode="none"))
    absent, _ = _run(config, control=_BudgetedControl())

    reasons = {source["reason"] for source in _control_sources(bare)}
    assert reasons == {render_control_reason("West City")}
    assert reasons == {source["reason"] for source in _control_sources(absent)}
    assert all(
        record.event["controller_evidence_mode"] == "none" for record in bare.rounds
    )
    assert all(record.event["controller_evidence_fact"] is None for record in bare.rounds)


def test_shared_fact_mode_appends_one_true_shared_fact_that_names_the_target():
    config = _config(rounds=2)
    shared = _shared_facts(config)
    result, ballots = _run(
        config, control=_BudgetedControl(evidence_mode="shared_fact")
    )

    sources = _control_sources(result)
    assert sources
    for source in sources:
        assert source["reason"].startswith(render_control_reason("West City"))
        assert CONTROL_EVIDENCE_PREFIX in source["reason"]
        # Verbatim, from `Is`, and pointing at the target - never invented.
        cited = source["reason"].split(CONTROL_EVIDENCE_PREFIX, 1)[1].strip()
        assert cited in shared
        assert "West City" in cited

    advocating = [
        record for record in result.rounds if record.event["controlled_position_count"]
    ]
    assert advocating
    for record in advocating:
        assert record.event["controller_evidence_mode"] == "shared_fact"
        assert record.event["controller_evidence_fact"] in shared
        assert (
            shared[record.event["controller_evidence_fact_index"]]
            == record.event["controller_evidence_fact"]
        )
    # The fact reaches the population, not just the log.
    assert any(
        record.event["controller_evidence_fact"] in prompt
        for record in advocating
        for prompt in ballots.prompts
    )


def test_shared_fact_mode_never_cites_unshared_information():
    config = _config(rounds=2)
    task = create_game(config.game).initialize(config.game, config.execution.seed).task
    result, ballots = _run(
        config, control=_BudgetedControl(evidence_mode="shared_fact")
    )

    for source in _control_sources(result):
        for fact in task["hidden_information"]:
            assert fact not in source["reason"]
    for prompt in ballots.prompts:
        lowered = prompt.lower()
        for word in FORBIDDEN:
            assert word not in lowered, f"{word!r} leaked into a focal prompt"


def test_the_cited_fact_is_one_per_round_and_replays_from_the_seed():
    config = _config(rounds=4)
    # `North Hill` has two on-target shared facts, so the draw is a real draw.
    control = _BudgetedControl(target="North Hill", evidence_mode="shared_fact")
    first, _ = _run(config, control=control)
    second, _ = _run(config, control=control)

    cited = [record.event["controller_evidence_fact"] for record in first.rounds]
    assert cited == [record.event["controller_evidence_fact"] for record in second.rounds]
    assert any(fact is not None for fact in cited)
    for record, fact in zip(first.rounds, cited, strict=True):
        # One citation per controller decision: every actuated slot of a round
        # shows the population the same message.
        reasons = _round_control_reasons(first, record.round_index)
        assert len(reasons) <= 1
        if reasons and fact is not None:
            assert fact in reasons.pop()


def test_scenario_sanitizer_removes_only_the_legacy_protocol_layer():
    raw = (
        "You are participating in a study, acting as a community leader. Heavy rain "
        "has blocked roads.\nYour Task:\nyou will discuss with other participants, "
        "who are also acting as community leaders, to decide where to evacuate. "
        "You have three options:\n- West City\n- East Town\n- North Hill\n"
        "There is only one correct evacuation location. After the discussion:\n"
        "- If you choose the correct location, you will earn $1.\n"
        "The chat will at most take 15 minutes. However, the exact time when the "
        "chat will end is unknown.\n\n"
        "There are exactly 6 agents in this experiment. Private information derives "
        "from 4 latent evidence types. Several agents may carry different realizations "
        "or components of the same type. Ignore any historical participant count left "
        "in the scenario wording."
    )

    assert sanitize_scenario(raw) == (
        "You are a community leader. Heavy rain has blocked roads.\n"
        "Your task is to decide where to evacuate together with the other community "
        "leaders. The available options are:\n- West City\n- East Town\n- North Hill\n"
        "There is only one correct evacuation location."
    )


# ---- the fixed social environment --------------------------------------


def test_the_strategic_uncertainty_block_is_byte_identical_in_every_prompt():
    runs = [
        _run(_config(rounds=1), control=None),
        _run(_config(rounds=1), control=_BudgetedControl(budget=0)),
        _run(_config(rounds=1), control=_BudgetedControl(target="North Hill")),
        _run(_config(rounds=1), control=_BudgetedControl(target="West City")),
        _run(_config(rounds=1, q=1), control=_BudgetedControl()),
    ]
    prompts = [prompt for _, ballots in runs for prompt in ballots.prompts]

    assert prompts
    for prompt in prompts:
        assert prompt.count(SOCIAL_ENVIRONMENT) == 1
    # The block is bound as a fixed prompt block, so it is also pinned by the
    # definition hash rather than only by the rendered text.
    definitions = {
        decision.prompt_definition_hash
        for result, _ in runs
        for item in result.interactions
        for decision in item.decisions
    }
    assert len(definitions) == 1


# ---- prompt snapshots --------------------------------------------------


@pytest.mark.parametrize("q", [1, 2])
@pytest.mark.parametrize("controlled", [False, True])
def test_prompt_snapshot(q, controlled):
    budget = 4 if controlled else 0
    result, ballots = _run(_config(rounds=1, q=q), control=_BudgetedControl(budget=budget))

    chosen = next(
        (prompt, item.transition.event)
        for prompt, item in zip(ballots.prompts, result.interactions, strict=True)
        if bool(item.transition.event["controlled_slot"]) is controlled
    )
    prompt, event = chosen
    sources = event["social_sources"]
    focal_label = agent_label(event["focal_agent_id"])

    assert prompt.startswith(
        f"You are {focal_label} participating in a group decision.\n"
        "\n"
        "Your goal is to identify the correct option.\n"
        "\n"
        f"{SOCIAL_ENVIRONMENT}\n"
        "\n"
        f"{DECISION_BASIS_SOCIAL}\n"
        "\n"
        "TASK\n"
    )
    assert prompt.count("The available options are:") == 1
    assert all(f"- {option}" in prompt for option in OPTIONS)
    assert "\nYOUR PRIVATE INFORMATION\n" in prompt
    assert (
        f"YOUR CURRENT PUBLIC POSITION\n\nVote: {event['focal_vote_before']}\n"
        f"Reason: {event['focal_reason_before'] or NO_PREVIOUS_REASON}"
    ) in prompt
    assert _social_block(prompt) == [render_social_source(item) for item in sources]
    assert [item["source_type"] for item in sources].count("control") == int(controlled)
    assert len(sources) == q
    assert prompt.rstrip().endswith(
        'Return only valid JSON:\n'
        "\n"
        "{\n"
        '  "vote": "<West City | East Town | North Hill>",\n'
        '  "reason": "<brief public reason>"\n'
        "}"
    )


# ---- contract and renderer units ---------------------------------------


@pytest.mark.parametrize(
    ("response", "field"),
    [
        ("not json at all", "response"),
        ('{"vote": "Atlantis", "reason": "ok"}', "response.vote"),
        ('{"vote": "West City"}', "response.reason"),
        ('{"vote": "West City", "reason": "   "}', "response.reason"),
        (
            json.dumps({"vote": "West City", "reason": "x" * (MAX_REASON_CHARACTERS + 1)}),
            "response.reason",
        ),
    ],
)
def test_the_contract_rejects_anything_that_is_not_one_vote_and_one_reason(
    response, field
):
    result = PublicBallotContract(allowed_values=OPTIONS).validate(response)
    assert not result.is_valid
    assert result.issues[0].field == field


def test_the_contract_accepts_a_fenced_two_field_ballot():
    response = '```json\n{"vote": "north hill", "reason": "Tunnel is blocked."}\n```'
    assert PublicBallotContract(allowed_values=OPTIONS).validate(response).is_valid

    ballot = parse_public_ballot_update(response, OPTIONS)
    assert ballot.vote == "North Hill"
    assert ballot.reason == "Tunnel is blocked."


def test_the_hidden_mode_renderer_drops_the_vote_and_keeps_everything_else():
    source = {"label": "Agent 2", "vote": "East Town", "reason": "Tunnel is blocked."}

    assert render_social_source(source, vote_visibility="public") == (
        "Agent 2\nVote: East Town\nReason: Tunnel is blocked."
    )
    assert render_social_source(source, vote_visibility="hidden") == (
        "Agent 2\nReason: Tunnel is blocked."
    )
    assert render_social_source({**source, "reason": None}) == (
        f"Agent 2\nVote: East Town\nReason: {NO_PREVIOUS_REASON}"
    )
    with pytest.raises(ValueError, match="vote_visibility"):
        render_social_source(source, vote_visibility="whispered")


def test_hidden_vote_visibility_is_reserved_rather_than_silently_accepted():
    config = _config(rounds=1)
    options = {**dict(config.game.options), "vote_visibility": "hidden"}
    game = create_game(config.game)

    with pytest.raises(ValueError, match="reserved"):
        game.rules(replace(config.game, options=options))


def test_the_reasoning_game_refuses_the_parent_games_second_prompt_version():
    config = _config(rounds=1)
    options = {**dict(config.game.options), "prompt_version": 2}
    game = create_game(config.game)

    with pytest.raises(ValueError, match="prompt_version must be 1"):
        game.rules(replace(config.game, options=options))
