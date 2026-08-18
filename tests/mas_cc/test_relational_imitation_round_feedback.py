"""Scientific invariants of the relational reasoning round-feedback game.

Two state variables move here, and the tests are organised around keeping them
honest: the vote ``X_i``, which any ballot may change, and the knowledge set
``K_i``, which may grow only when some participant actually exposed a fact to
*this* agent at an interaction it took part in.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import replace

import pytest

from mas_cc.config import load_run_config
from mas_cc.control import Control, RoundControlSignal, create_control
from mas_cc.games import create_game
from mas_cc.games.hidden_bench.imitation.controller import ADVOCATE_TARGET, NO_OP
from mas_cc.games.relational_reasoning.data import DEFAULT_TASK_DATASET_DIR
from mas_cc.games.relational_reasoning.imitation_round_feedback.controller import (
    SCHEDULE_ALWAYS,
    SCHEDULE_SOFT,
    RECOMMENDATION_ONLY,
    RECOMMENDATION_PLUS_FACT,
    RelationalRoundBudgetedControl,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.prompts import (
    EVIDENCE_HEADER,
    MAX_REASON_CHARACTERS,
    NO_KNOWN_FACTS,
    SOCIAL_ENVIRONMENT,
    SOCIAL_ENVIRONMENT_DISTRUST,
    SOCIAL_ENVIRONMENT_NEUTRAL,
    RelationalBallotContract,
    agent_label,
    localize_sources,
    parse_relational_ballot,
    relational_public_ballot_prompt,
    render_control_reason,
    render_social_source,
    social_environment,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.runtime import (
    CONTROL_SOURCE_ID,
    RelationalDecisionFailed,
    run_relational_imitation_round_feedback_game,
    sample_controlled_positions,
)
from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider

pytestmark = pytest.mark.skipif(
    not (DEFAULT_TASK_DATASET_DIR / "task_0001.json").exists(),
    reason="the relational example dataset is not present",
)

NO_CONTROL = (
    "configs/runs/relational_reasoning/"
    "relational_imitation_round_feedback_no_control_smoke.yaml"
)
CONTROLLED = (
    "configs/runs/relational_reasoning/"
    "relational_imitation_round_feedback_controlled_smoke.yaml"
)
# The population state lives in the SEMANTIC alphabet; A/B/C are per-call
# presentation letters that never leave a single prompt.
OPTIONS = ("NORTHEAST", "SOUTHWEST", "NORTH")
CORRECT = "NORTH"
LETTERS = ("A", "B", "C")
SUPPORTING = ("f1", "f2")
FORBIDDEN = ("controller", "external", "intervention", "experiment", "simulation")


# ---- harness -----------------------------------------------------------


class _AlwaysAdvocate(Control):
    """A fixed round decision: always advocate, always the same budget.

    Used instead of the real soft policy wherever a test needs actuation to
    happen deterministically rather than with the policy's own probability.
    """

    sensor_sample_size = 6

    def __init__(self, *, target: str = CORRECT, budget: int = 4) -> None:
        self.intervention_budget = budget
        self.target = target

    def override(self, **_):
        return None

    def round_signal(self, *, round_index, state, rng):
        sampled = rng.sample(list(state.agents), self.sensor_sample_size)
        opinions = [str(agent.attributes["committed_action"]) for agent in sampled]
        return RoundControlSignal(
            action=ADVOCATE_TARGET,
            target=self.target,
            message="unused legacy controller message",
            observation={
                "sampled_agent_ids": [str(agent.agent_id) for agent in sampled],
                "sampled_opinions": opinions,
                "sampled_opinion_counts": dict(Counter(opinions)),
                "sample_size": self.sensor_sample_size,
            },
            metadata={"policy": "test", "advocacy_probability": 1.0},
        )


class _NeverAdvocate(_AlwaysAdvocate):
    def round_signal(self, **kwargs):
        signal = super().round_signal(**kwargs)
        return replace(signal, action=NO_OP)


class _AlwaysAdvocateRelational(RelationalRoundBudgetedControl):
    """The real configured controller, with its stochastic policy pinned on.

    Everything else - the sensor, the budget, the message mode, the fact
    resolution - is the shipped implementation, so a test that needs actuation
    to happen every round still exercises the production path.
    """

    def round_signal(self, *, round_index, state, rng):
        signal = super().round_signal(round_index=round_index, state=state, rng=rng)
        return replace(signal, action=ADVOCATE_TARGET)


def _forced(config, **overrides):
    """The config's own controller, forced to advocate every round."""

    return _AlwaysAdvocateRelational.from_options(
        {**dict(config.control.options), **overrides}
    )


class _Ballots:
    """A provider that records every prompt and answers a scripted ballot.

    ``share`` decides ``shared_fact_id``: ``"first_known"`` echoes back the
    first fact the prompt says this agent knows, which is how the propagation
    tests get real evidence moving without a model.
    """

    def __init__(self, *, votes=LETTERS, share="none") -> None:
        self.prompts: list[str] = []
        self.votes = tuple(votes)
        self.share = share

    def _known_ids(self, prompt: str) -> list[str]:
        if NO_KNOWN_FACTS in prompt:
            return []
        block = prompt.split("YOUR CURRENT KNOWLEDGE", 1)[1]
        block = block.split("YOUR CURRENT POSITION")[0]
        block = block.split("CURRENT SOCIAL INFORMATION")[0].split("\nDECISION\n")[0]
        return [
            line[2:].split(":", 1)[0]
            for line in block.splitlines()
            if line.startswith("- f")
        ]

    def provider(self, config):
        def factory(request):
            index = len(self.prompts)
            prompt = "\n\n".join(message.content for message in request.messages)
            self.prompts.append(prompt)
            known = self._known_ids(prompt)
            if self.share == "first_known":
                shared = known[0] if known else "none"
            elif self.share == "hallucinate":
                shared = "f6" if "f6" not in known else "f5"
            else:
                shared = self.share
            return json.dumps(
                {
                    "vote": self.votes[index % len(self.votes)],
                    "reason": self.reason_for(index),
                    "shared_fact_id": shared,
                }
            )

        return MockLLMProvider(config, response_factory=factory)

    def reason_for(self, index: int) -> str:
        return f"private reason {index}"


def _config(path=NO_CONTROL, *, rounds=1, q=1, initialization=None, population=None):
    config = load_run_config(path, environment={})
    options = dict(config.game.options)
    options["rounds"] = rounds
    options["social_group_size"] = q
    if initialization is not None:
        options["initialization"] = initialization
    size = config.game.population_size if population is None else population
    if population is not None:
        options["n_agents"] = population
    return replace(
        config,
        game=replace(config.game, horizon=rounds, population_size=size, options=options),
    )


def _run(config, *, control=None, ballots=None):
    ballots = ballots or _Ballots()
    result = asyncio.run(
        run_relational_imitation_round_feedback_game(
            create_game(config.game),
            config,
            ballots.provider(config.llm_provider),
            control=control,
        )
    )
    return result, ballots


def _social_block(prompt: str) -> list[str]:
    if "CURRENT SOCIAL INFORMATION" not in prompt:
        return []
    body = prompt.split("CURRENT SOCIAL INFORMATION\n\n", 1)[1]
    body = body.split("\n\nDECISION\n", 1)[0]
    return body.split("\n\n")


def _events(result):
    return [item.transition.event for item in result.interactions]


def _letters(item):
    """The `letter -> relation` map the focal of this update was shown."""

    return dict(item.decisions[0].request.observation.visible_state["option_letters"])


def _rendered_sources(item):
    """Its social sources as the prompt actually showed them."""

    return [
        render_social_source(source)
        for source in localize_sources(item.transition.event["social_sources"], _letters(item))
    ]


# ---- initial information (§18) -----------------------------------------


def test_initial_knowledge_matches_the_frozen_assignment_exactly():
    config = _config()
    game = create_game(config.game)
    task = game.load_task(config.game)
    state = game.initialize(config.game, config.execution.seed)

    assert len(state.agents) == task.population_size == 24
    for agent in state.agents:
        assert agent.known_fact_ids == task.known_facts(str(agent.agent_id))
        assert agent.initial_fact_ids == agent.known_fact_ids
        # No agent starts holding a fact it was never assigned.
        assert set(agent.known_fact_ids) <= set(task.fact_order)


def test_the_population_union_holds_every_supporting_fact_at_t0():
    config = _config()
    state = create_game(config.game).initialize(config.game, config.execution.seed)

    pooled = {fact for agent in state.agents for fact in agent.known_fact_ids}
    assert set(state.supporting_fact_ids) <= pooled
    assert state.supporting_fact_ids == SUPPORTING


def test_a_population_size_that_does_not_match_the_task_is_refused():
    config = _config(population=6)

    with pytest.raises(ValueError, match="population_size"):
        create_game(config.game).initialize(config.game, config.execution.seed)


def test_local_vote_initialization_asks_each_agent_once_from_its_own_facts():
    config = _config(rounds=1, initialization={"mode": "local_vote"})
    result, ballots = _run(config)

    n = config.game.population_size
    assert len(result.initial_decisions) == n
    assert len(ballots.prompts) == n + n  # initialization + one round of updates
    initial = ballots.prompts[:n]
    assert all("CURRENT SOCIAL INFORMATION" not in prompt for prompt in initial)
    assert list(result.initial_state.initial_votes) == [
        decision.action.value for decision in result.initial_decisions
    ]


def test_uniform_random_initialization_needs_no_provider_call():
    config = _config(rounds=1, initialization={"mode": "uniform_random"})
    result, ballots = _run(config)

    assert result.initial_decisions == ()
    assert len(ballots.prompts) == config.game.population_size
    assert set(result.initial_state.initial_votes) <= set(OPTIONS)


# ---- scheduling and sampling -------------------------------------------


def test_each_round_contains_exactly_n_updates_with_one_focal_each():
    config = _config(rounds=3)
    result, _ = _run(config)

    n = config.game.population_size
    assert len(result.interactions) == 3 * n
    assert len(result.rounds) == 3
    for expected, event in enumerate(_events(result)):
        assert event["global_update_index"] == expected
        assert expected == event["round_index"] * n + event["within_round_index"]
        changed = [
            index
            for index, pair in enumerate(
                zip(event["population_state_before"], event["population_state_after"])
            )
            if pair[0] != pair[1]
        ]
        assert len(changed) <= 1


@pytest.mark.parametrize("q", [1, 2])
def test_focal_and_peer_sampling_draws_distinct_agents(q):
    result, _ = _run(_config(rounds=2, q=q))

    for event in _events(result):
        peers = event["sampled_peer_ids"]
        assert len(peers) == q
        assert len(set(peers)) == q
        assert event["focal_agent_id"] not in peers


def test_a_fixed_seed_replays_the_whole_trajectory():
    config = _config(rounds=2)
    first, _ = _run(config, control=_AlwaysAdvocate())
    second, _ = _run(config, control=_AlwaysAdvocate())

    assert [event["focal_agent_id"] for event in _events(first)] == [
        event["focal_agent_id"] for event in _events(second)
    ]
    assert [record.event["controlled_positions"] for record in first.rounds] == [
        record.event["controlled_positions"] for record in second.rounds
    ]
    assert [agent.known_fact_ids for agent in first.final_state.agents] == [
        agent.known_fact_ids for agent in second.final_state.agents
    ]


def test_the_controlled_schedule_is_an_exact_budget_drawn_without_replacement():
    class _Rng:
        def sample(self, population, k):
            return list(population)[-k:]

    assert sample_controlled_positions(6, 3, _Rng()) == (3, 4, 5)
    with pytest.raises(ValueError, match="intervention_budget"):
        sample_controlled_positions(6, 7, _Rng())


def test_the_controlled_schedule_is_deterministic_and_matches_the_budget():
    config = _config(rounds=3)
    first, _ = _run(config, control=_AlwaysAdvocate(budget=5))
    second, _ = _run(config, control=_AlwaysAdvocate(budget=5))

    for a, b in zip(first.rounds, second.rounds, strict=True):
        assert a.event["controlled_positions"] == b.event["controlled_positions"]
        assert a.event["controlled_position_count"] == 5
        assert len(set(a.event["controlled_positions"])) == 5
    for event in _events(first):
        expected = event["within_round_index"] in set(
            first.rounds[event["round_index"]].event["controlled_positions"]
        )
        assert event["controlled_slot"] is expected


# ---- ballots and vote parsing ------------------------------------------


def test_the_returned_vote_is_the_committed_action_and_the_only_vote_field():
    result, _ = _run(_config(rounds=2))

    for item in result.interactions:
        event = item.transition.event
        parsed = json.loads(item.decisions[0].attempts[-1].response.content)
        focal = event["focal_agent_id"]
        committed = next(
            agent.committed_action
            for agent in item.transition.next_state.agents
            if str(agent.agent_id) == focal
        )
        # The model answered with a letter; the action, the committed state and
        # the logged vote are all the relation that letter named on this call.
        semantic = _letters(item)[parsed["vote"]]
        assert item.decisions[0].action.value == semantic == committed
        assert event["vote_after"] == event["focal_vote_after"] == semantic
        assert item.decisions[0].action.metadata["presented_letter"] == parsed["vote"]


def test_a_vote_may_be_given_as_a_letter_or_as_the_relation_it_names():
    letters = {"A": "NORTHEAST", "B": "SOUTHWEST", "C": "NORTH"}

    assert parse_relational_ballot(
        '{"vote":"c","reason":"x","shared_fact_id":"none"}', LETTERS, letters
    ).vote == "C"
    assert parse_relational_ballot(
        '{"vote":"Option B","reason":"x","shared_fact_id":"none"}', LETTERS, letters
    ).vote == "B"
    # Spelled out, the relation resolves to whichever letter carried it here.
    assert parse_relational_ballot(
        '{"vote":"northeast","reason":"x","shared_fact_id":"none"}', LETTERS, letters
    ).vote == "A"
    # No substring guessing: a sentence is not a vote.
    assert parse_relational_ballot(
        '{"vote":"a bit north of B","reason":"x","shared_fact_id":"none"}',
        LETTERS,
        letters,
    ).vote is None


@pytest.mark.parametrize(
    ("response", "field"),
    [
        ("not json at all", "response"),
        ('{"vote":"Z","reason":"ok","shared_fact_id":"none"}', "response.vote"),
        ('{"vote":"A","shared_fact_id":"none"}', "response.reason"),
        ('{"vote":"A","reason":"  ","shared_fact_id":"none"}', "response.reason"),
        (
            json.dumps(
                {
                    "vote": "A",
                    "reason": "x" * (MAX_REASON_CHARACTERS + 1),
                    "shared_fact_id": "none",
                }
            ),
            "response.reason",
        ),
        ('{"vote":"A","reason":"ok"}', "response.shared_fact_id"),
        (
            '{"vote":"A","reason":"ok","shared_fact_id":"f99"}',
            "response.shared_fact_id",
        ),
        (
            '{"vote":"A","reason":"ok","shared_fact_id":"the northwest one"}',
            "response.shared_fact_id",
        ),
    ],
)
def test_the_contract_rejects_anything_that_is_not_one_complete_ballot(response, field):
    contract = RelationalBallotContract(
        allowed_values=LETTERS,
        options={"fact_ids": ("f1", "f2", "f3"), "relations": ("EAST", "NORTH", "WEST")},
    )
    result = contract.validate(response)

    assert not result.is_valid
    assert result.issues[0].field == field


def test_the_contract_accepts_a_fenced_ballot_with_or_without_evidence():
    contract = RelationalBallotContract(
        allowed_values=LETTERS,
        options={"fact_ids": ("f1", "f2"), "relations": ("NORTH",)},
    )

    fenced = '```json\n{"vote":"a","reason":"Chain points north.","shared_fact_id":"f1"}\n```'
    assert contract.validate(fenced).is_valid
    ballot = parse_relational_ballot(fenced, LETTERS)
    assert (ballot.vote, ballot.shared_fact_id) == ("A", "f1")
    assert contract.validate(
        '{"vote":"A","reason":"ok","shared_fact_id":"none"}'
    ).is_valid
    assert parse_relational_ballot(
        '{"vote":"A","reason":"ok","shared_fact_id":"NONE"}', LETTERS
    ).shared_fact_id is None


# ---- evidence honesty (§18) --------------------------------------------


def test_an_agent_may_expose_only_a_fact_it_currently_knows():
    config = _config(rounds=2)
    result, _ = _run(config, ballots=_Ballots(share="first_known"))

    for event in _events(result):
        shared = event["focal_shared_fact_id"]
        if shared is None:
            continue
        assert shared in event["focal_known_fact_ids_before"]


def test_a_hallucinated_citation_is_rejected_and_retried_not_silently_dropped():
    config = _config(rounds=1)
    options = {**dict(config.game.options), "invalid_response_retries": 0}
    config = replace(config, game=replace(config.game, options=options))

    with pytest.raises(RelationalDecisionFailed, match="focal_update"):
        _run(config, ballots=_Ballots(share="hallucinate"))


def test_the_validator_names_the_knowledge_set_when_it_rejects_a_citation():
    config = _config(rounds=1)
    game = create_game(config.game)
    state = game.initialize(config.game, config.execution.seed)
    focal = next(agent for agent in state.agents if agent.known_fact_ids)
    request = game.ballot_request(state, focal.agent_id, (), config.game)
    unknown = next(
        fact for fact in state.fact_ids if fact not in focal.known_fact_ids
    )

    good = game.parse_action(
        request,
        json.dumps(
            {
                "vote": "A",
                "reason": "ok",
                "shared_fact_id": focal.known_fact_ids[0],
            }
        ),
    )
    bad = game.parse_action(
        request,
        json.dumps({"vote": "A", "reason": "ok", "shared_fact_id": unknown}),
    )

    assert game.validate_action(state, request, good, config.game).is_valid
    issues = game.validate_action(state, request, bad, config.game).issues
    assert [issue.field for issue in issues] == ["action.shared_fact_id"]


# ---- knowledge propagation (§7, §18) -----------------------------------


def test_a_peer_exposed_fact_reaches_exactly_the_focal_agent_that_saw_it():
    config = _config(rounds=3, q=1)
    result, _ = _run(config, ballots=_Ballots(share="first_known"))

    seen = 0
    for item in result.interactions:
        event = item.transition.event
        focal = event["focal_agent_id"]
        exposed = set(event["peer_exposed_fact_ids"])
        after = {
            str(agent.agent_id): set(agent.known_fact_ids)
            for agent in item.transition.next_state.agents
        }
        before = {
            str(agent.agent_id): set(agent.known_fact_ids)
            for agent in create_game(config.game).initialize(
                config.game, config.execution.seed
            ).agents
        }
        assert exposed <= after[focal]
        # Nobody else's knowledge moved at this event.
        for agent_id, facts in after.items():
            if agent_id == focal:
                continue
            assert facts >= before[agent_id]
        seen += len(event["new_peer_fact_ids"])
    assert seen > 0


def test_knowledge_only_ever_grows_and_only_through_a_recorded_exposure():
    config = _config(rounds=3, q=2)
    result, _ = _run(config, control=_AlwaysAdvocate(), ballots=_Ballots(share="first_known"))

    initial = {
        str(agent.agent_id): set(agent.known_fact_ids)
        for agent in create_game(config.game)
        .initialize(config.game, config.execution.seed)
        .agents
    }
    known = {key: set(value) for key, value in initial.items()}
    for event in _events(result):
        focal = event["focal_agent_id"]
        assert set(event["focal_known_fact_ids_before"]) == known[focal]
        acquired = set(event["focal_known_fact_ids_after"]) - known[focal]
        # Every acquisition is accounted for by an exposure at this event.
        assert acquired == set(event["new_peer_fact_ids"]) | set(
            event["new_controller_fact_ids"]
        )
        assert acquired <= set(event["peer_exposed_fact_ids"]) | {
            event["controller_fact_id"]
        }
        known[focal] |= acquired

    for agent in result.final_state.agents:
        assert set(agent.known_fact_ids) == known[str(agent.agent_id)]
        assert initial[str(agent.agent_id)] <= set(agent.known_fact_ids)


def test_an_already_known_fact_is_an_exposure_but_not_an_acquisition():
    config = _config(rounds=3, q=1)
    result, _ = _run(config, ballots=_Ballots(share="first_known"))

    repeated = 0
    for event in _events(result):
        for fact in event["peer_exposed_fact_ids"]:
            if fact in event["focal_known_fact_ids_before"]:
                assert fact not in event["new_peer_fact_ids"]
                repeated += 1
    assert repeated > 0


def test_the_provenance_of_every_acquired_fact_is_recorded():
    config = _config(rounds=2, q=1)
    result, _ = _run(
        config, control=_AlwaysAdvocate(), ballots=_Ballots(share="first_known")
    )

    for agent in result.final_state.agents:
        provenance = agent.fact_provenance
        assert set(provenance) == set(agent.known_fact_ids)
        for fact in agent.initial_fact_ids:
            assert provenance[fact]["source"] == "initial"
        for fact in set(agent.known_fact_ids) - set(agent.initial_fact_ids):
            assert provenance[fact]["source"] in {"peer", "controller"}
            assert provenance[fact]["round_index"] is not None


# ---- the controller (§9-§12) -------------------------------------------


@pytest.mark.parametrize("q", [1, 2])
def test_control_replaces_one_peer_slot_and_never_adds_a_source(q):
    result, ballots = _run(_config(rounds=2, q=q), control=_AlwaysAdvocate())

    controlled = 0
    for prompt, item in zip(ballots.prompts, result.interactions, strict=True):
        event = item.transition.event
        sources = event["social_sources"]
        control = [source for source in sources if source["source_type"] == "control"]
        ordinary = [source for source in sources if source["source_type"] == "ordinary"]
        expected = int(event["controlled_slot"])
        assert len(sources) == len(_social_block(prompt)) == q
        assert len(control) == expected
        assert len(ordinary) == q - expected
        assert len(event["effective_peer_ids"]) == q - expected
        if expected:
            assert event["replaced_peer_id"] in event["sampled_peer_ids"]
            assert event["replaced_peer_id"] not in event["effective_peer_ids"]
        controlled += expected
    assert controlled > 0


def test_at_q1_a_controlled_focal_sees_the_controller_instead_of_its_peer():
    result, _ = _run(_config(rounds=2, q=1), control=_AlwaysAdvocate())

    controlled = [
        event for event in _events(result) if event["controlled_slot"]
    ]
    assert controlled
    for event in controlled:
        assert event["effective_peer_ids"] == []
        assert event["replaced_peer_id"] == event["sampled_peer_ids"][0]
        assert [source["source_type"] for source in event["social_sources"]] == ["control"]


def test_the_controller_senses_votes_only_and_never_a_knowledge_set():
    seen: list[dict] = []

    class _Recording(_AlwaysAdvocate):
        def round_signal(self, *, round_index, state, rng):
            seen.extend(dict(agent.attributes) for agent in state.agents)
            assert "task" in state.data and set(state.data["task"]) == {
                "task_id",
                "possible_answers",
                "correct_answer",
            }
            return super().round_signal(round_index=round_index, state=state, rng=rng)

    _run(_config(rounds=2), control=_Recording())

    assert seen
    for attributes in seen:
        assert set(attributes) == {"committed_action"}


def test_recommendation_only_transmits_no_fact_at_all():
    config = _config(CONTROLLED, rounds=2)
    options = {
        **dict(config.control.options),
        "message_mode": RECOMMENDATION_ONLY,
        "controller_fact_id": None,
    }
    config = replace(config, control=replace(config.control, options=options))
    control = create_control(config.control)
    result, ballots = _run(config, control=_AlwaysAdvocate(target=CORRECT))

    assert control.message_mode == RECOMMENDATION_ONLY
    assert control.resolve_fact_id(create_game(config.game).load_task(config.game)) is None
    for event in _events(result):
        assert event["controller_fact_id"] is None
        assert event["new_controller_fact_ids"] == []
    for prompt in ballots.prompts:
        assert EVIDENCE_HEADER not in prompt.split("CURRENT SOCIAL INFORMATION")[-1] or (
            "CURRENT SOCIAL INFORMATION" not in prompt
        )


def test_recommendation_plus_fact_injects_one_exact_existing_fact():
    config = _config(CONTROLLED, rounds=2, q=1)
    game = create_game(config.game)
    task = game.load_task(config.game)
    control = _forced(config)
    result, ballots = _run(config, control=control)

    assert control.message_mode == RECOMMENDATION_PLUS_FACT
    assert control.resolve_fact_id(task) == "f2"

    controlled = [event for event in _events(result) if event["controlled_slot"]]
    assert controlled
    for event in controlled:
        assert event["controller_fact_id"] == "f2"
        assert event["controller_fact_exposures"] == 1
        source = next(
            item for item in event["social_sources"] if item["source_type"] == "control"
        )
        assert source["vote"] == CORRECT
        assert source["reason"] == render_control_reason(CORRECT)
        assert source["shared_fact_id"] == "f2"
        # Verbatim from the generator's own rendering - never paraphrased.
        assert source["shared_fact_text"] == task.fact_text("f2")
        assert "f2" in event["focal_known_fact_ids_after"]
    for record in result.rounds:
        assert record.event["controller_fact_id"] == "f2"
        assert record.event["controller_fact_text"] == task.fact_text("f2")
    assert any(task.fact_text("f2") in prompt for prompt in ballots.prompts)


def test_a_controller_fact_reaches_only_the_focals_at_controlled_positions():
    config = _config(CONTROLLED, rounds=2, q=1)
    result, _ = _run(config, control=_forced(config, intervention_budget=3))

    exposed_to = set()
    for event in _events(result):
        if event["controlled_slot"]:
            exposed_to.add(event["focal_agent_id"])
            assert event["controller_fact_id"] == "f2"
        else:
            assert event["controller_fact_id"] is None
            assert event["new_controller_fact_ids"] == []

    initial = {
        str(agent.agent_id): set(agent.known_fact_ids)
        for agent in create_game(config.game)
        .initialize(config.game, config.execution.seed)
        .agents
    }
    for agent in result.final_state.agents:
        gained_from_controller = {
            fact
            for fact, entry in agent.fact_provenance.items()
            if entry["source"] == "controller"
        }
        if gained_from_controller:
            assert str(agent.agent_id) in exposed_to
            assert gained_from_controller == {"f2"} - initial[str(agent.agent_id)]


def test_a_no_op_round_transmits_no_controller_evidence():
    config = _config(CONTROLLED, rounds=2, q=1)
    result, _ = _run(config, control=_NeverAdvocate(target=CORRECT))

    for record in result.rounds:
        assert record.event["controller_action"] == NO_OP
        assert record.event["controlled_position_count"] == 0
        assert record.event["controller_fact_id"] is None
        assert record.event["controller_fact_exposures"] == 0
    for event in _events(result):
        assert not event["controlled_slot"]
        assert event["controller_fact_id"] is None


def test_with_no_control_no_controller_evidence_ever_enters_any_knowledge_set():
    config = _config(NO_CONTROL, rounds=3, q=1)
    result, ballots = _run(
        config, control=create_control(config.control), ballots=_Ballots(share="first_known")
    )

    assert config.control.mechanism == "none"
    for event in _events(result):
        assert event["controller_enabled"] is False
        assert event["controlled_slot"] is False
        assert event["controller_fact_id"] is None
        assert event["new_controller_fact_ids"] == []
        assert all(
            source["source_type"] == "ordinary" for source in event["social_sources"]
        )
    for agent in result.final_state.agents:
        assert all(
            entry["source"] != "controller" for entry in agent.fact_provenance.values()
        )
    for record in result.rounds:
        assert record.event["controller_fact_exposures"] == 0
        assert record.event["new_controller_facts"] == 0
    for prompt in ballots.prompts:
        assert CONTROL_SOURCE_ID not in prompt


def test_the_controller_appears_as_one_persistent_ordinary_participant():
    result, ballots = _run(_config(rounds=2), control=_AlwaysAdvocate())

    labels = set()
    for prompt, item in zip(ballots.prompts, result.interactions, strict=True):
        for source in item.transition.event["social_sources"]:
            if source["source_type"] == "control":
                labels.add(source["label"])
                letter = _letters(item)
                inverse = {relation: key for key, relation in letter.items()}
                assert f"{source['label']}\nVote: {inverse[source['vote']]}" in prompt
    assert labels == {"Agent 25"}
    assert "Agent 25" not in {
        agent_label(agent.agent_id) for agent in result.final_state.agents
    }


def test_no_prompt_ever_identifies_a_source_as_control_or_as_an_experiment():
    result, ballots = _run(
        _config(CONTROLLED, rounds=2),
        control=_AlwaysAdvocate(target=CORRECT),
        ballots=_Ballots(share="first_known"),
    )

    assert any(event["controlled_slot"] for event in _events(result))
    for prompt in ballots.prompts:
        lowered = prompt.lower()
        for word in FORBIDDEN:
            assert word not in lowered, f"{word!r} leaked into a focal prompt"
        assert CONTROL_SOURCE_ID not in prompt
        assert "unused legacy controller message" not in prompt


# ---- controller configuration ------------------------------------------


def _control_options(**overrides):
    return {
        "target": "correct",
        "sensor_sample_size": 6,
        "policy": "soft_target",
        "threshold": 0.5,
        "beta": 4.0,
        "intervention_budget": 4,
        **overrides,
    }


def test_the_message_mode_defaults_to_recommendation_only():
    control = RelationalRoundBudgetedControl.from_options(_control_options())

    assert control.message_mode == RECOMMENDATION_ONLY
    assert control.intervention_budget == 4
    assert not control.transmits_fact


def test_a_fact_selector_resolves_deterministically_to_a_supporting_fact():
    config = _config(CONTROLLED)
    task = create_game(config.game).load_task(config.game)
    control = RelationalRoundBudgetedControl.from_options(
        _control_options(
            message_mode=RECOMMENDATION_PLUS_FACT, controller_fact_selector="supporting"
        )
    )

    assert control.resolve_fact_id(task) == task.supporting_fact_ids[0] == "f1"


@pytest.mark.parametrize(
    "overrides",
    [
        {"message_mode": "shout"},
        {"message_mode": RECOMMENDATION_PLUS_FACT},
        {"message_mode": RECOMMENDATION_ONLY, "controller_fact_id": "f2"},
        {
            "message_mode": RECOMMENDATION_PLUS_FACT,
            "controller_fact_id": "f2",
            "controller_fact_selector": "supporting",
        },
        {"message_mode": RECOMMENDATION_PLUS_FACT, "controller_fact_selector": "random"},
        {"intervention_budget": -1},
        {"evidence_mode": "shared_fact"},
    ],
)
def test_an_incoherent_controller_configuration_is_refused(overrides):
    with pytest.raises(ConfigurationError):
        RelationalRoundBudgetedControl.from_options(_control_options(**overrides))


def test_a_controller_fact_outside_the_task_is_refused_at_resolution():
    config = _config(CONTROLLED)
    task = create_game(config.game).load_task(config.game)
    control = RelationalRoundBudgetedControl.from_options(
        _control_options(
            message_mode=RECOMMENDATION_PLUS_FACT, controller_fact_id="f99"
        )
    )

    with pytest.raises(ValueError, match="not a fact of task"):
        control.resolve_fact_id(task)


# ---- open-loop advocacy schedule ----------------------------------------


def test_the_advocacy_schedule_defaults_to_the_closed_soft_loop():
    control = RelationalRoundBudgetedControl.from_options(_control_options())

    assert control.advocacy_schedule == SCHEDULE_SOFT
    # A population already fully on target is left alone by the soft policy.
    action, probability = control.select_action(1.0, __import__("random").Random(0))
    assert action == NO_OP and probability < 0.5


def test_always_advocates_every_round_whatever_the_sensor_saw():
    control = RelationalRoundBudgetedControl.from_options(
        _control_options(advocacy_schedule=SCHEDULE_ALWAYS)
    )

    # No rng is consulted at all, so the schedule cannot depend on the stream.
    for share in (0.0, 0.5, 1.0):
        assert control.select_action(share, None) == (ADVOCATE_TARGET, 1.0)


def test_an_always_schedule_actuates_in_every_round_of_an_episode():
    config = _config(CONTROLLED, rounds=4, q=1)
    options = {**dict(config.control.options), "advocacy_schedule": SCHEDULE_ALWAYS}
    config = replace(config, control=replace(config.control, options=options))
    result, _ = _run(config, control=create_control(config.control))

    assert len(result.rounds) == 4
    for record in result.rounds:
        assert record.event["controller_action"] == ADVOCATE_TARGET
        assert record.event["controller_advocacy_probability"] == 1.0
        assert record.event["controlled_position_count"] == (
            config.control.options["intervention_budget"]
        )
        # Sensing still happened and is still logged; only the decision ignores it.
        assert record.event["sensor_sample_size"] == options["sensor_sample_size"]
        assert len(record.event["sensor_agent_ids"]) == options["sensor_sample_size"]


def test_an_unknown_advocacy_schedule_is_refused():
    with pytest.raises(ConfigurationError):
        RelationalRoundBudgetedControl.from_options(
            _control_options(advocacy_schedule="sometimes")
        )


# ---- rendering and prompts ---------------------------------------------


def test_a_peer_ballot_is_shown_with_its_rendered_fact_and_no_symbolic_tuple():
    config = _config(rounds=3, q=1)
    task = create_game(config.game).load_task(config.game)
    result, ballots = _run(config, ballots=_Ballots(share="first_known"))

    shown = 0
    for prompt, item in zip(ballots.prompts, result.interactions, strict=True):
        for source, rendered, expected in zip(
            item.transition.event["social_sources"],
            _social_block(prompt),
            _rendered_sources(item),
            strict=True,
        ):
            assert rendered == expected
            if source["shared_fact_id"]:
                assert source["shared_fact_text"] == task.fact_text(
                    source["shared_fact_id"]
                )
                assert source["shared_fact_text"] in prompt
                # A peer's fact identifier stays in the log, not in the prompt.
                assert f"{source['shared_fact_id']}: " not in rendered
                shown += 1
    assert shown > 0


def test_an_agent_sees_its_own_facts_with_the_identifiers_it_may_cite():
    config = _config(rounds=1)
    game = create_game(config.game)
    state = game.initialize(config.game, config.execution.seed)
    with_facts = next(agent for agent in state.agents if agent.known_fact_ids)
    without = next(agent for agent in state.agents if not agent.known_fact_ids)

    rich = "\n\n".join(
        message.content
        for message in game.ballot_request(state, with_facts.agent_id, (), config.game)
        .prompt.compile()
        .messages
    )
    bare = "\n\n".join(
        message.content
        for message in game.ballot_request(state, without.agent_id, (), config.game)
        .prompt.compile()
        .messages
    )

    for fact in with_facts.known_fact_ids:
        assert f"- {fact}: {state.fact_text(fact)}" in rich
    assert NO_KNOWN_FACTS in bare
    # An agent never sees another agent's facts through this block.
    for fact in set(state.fact_ids) - set(with_facts.known_fact_ids):
        assert f"- {fact}: " not in rich


def test_the_social_environment_block_is_byte_identical_in_every_prompt():
    runs = [
        _run(_config(rounds=1), control=None),
        _run(_config(rounds=1), control=_AlwaysAdvocate(budget=0)),
        _run(_config(rounds=1, q=2), control=_AlwaysAdvocate()),
        _run(_config(CONTROLLED, rounds=1), control=_AlwaysAdvocate(target=CORRECT)),
    ]
    prompts = [prompt for _, ballots in runs for prompt in ballots.prompts]

    assert prompts
    for prompt in prompts:
        assert prompt.count(SOCIAL_ENVIRONMENT) == 1
    # The definition hash is *not* constant across agents any more: the contract
    # advertises each agent's own citable fact ids, and the contract is part of
    # the definition. What must stay constant is that two agents with the same
    # knowledge get the same definition, so nothing else drifts per prompt.
    by_knowledge: dict[tuple[str, ...], set[str]] = {}
    for result, _ in runs:
        for item in result.interactions:
            key = tuple(item.transition.event["focal_known_fact_ids_before"])
            for decision in item.decisions:
                by_knowledge.setdefault(key, set()).add(decision.prompt_definition_hash)
    assert by_knowledge
    for key, hashes in by_knowledge.items():
        assert len(hashes) == 1, key
    # ...and different knowledge really does give a different definition.
    assert len({next(iter(v)) for v in by_knowledge.values()}) == len(by_knowledge)


def test_prompt_snapshot():
    config = _config(rounds=1, q=1)
    result, ballots = _run(config, control=_AlwaysAdvocate(budget=24))
    prompt = ballots.prompts[0]
    event = _events(result)[0]
    known = tuple(event["focal_known_fact_ids_before"])

    assert prompt.startswith(
        f"You are {agent_label(event['focal_agent_id'])}, one participant in a "
        "group reasoning problem.\n"
        "\n"
        "Your goal is to identify the correct answer.\n"
        "\n"
        f"{SOCIAL_ENVIRONMENT}\n"
    )
    assert "QUESTION\n\nWhere is Bavi relative to Ralo?" in prompt
    # The option list is this call's shuffle, not a fixed global ordering.
    letters = _letters(result.interactions[0])
    assert set(letters.values()) == set(OPTIONS)
    assert "\n".join(f"- {k}) {v}" for k, v in sorted(letters.items())) in prompt
    assert "Vote by its letter." in prompt
    assert "\nYOUR CURRENT KNOWLEDGE\n" in prompt
    # The standing position is the vote and nothing else.
    assert f"YOUR CURRENT POSITION\n\nVote: {event['focal_vote_before']}\n" in prompt
    assert "YOUR CURRENT PUBLIC POSITION" not in prompt
    citable = " | ".join((*known, "none"))
    assert prompt.rstrip().endswith(
        "Return only valid JSON:\n"
        "\n"
        "{\n"
        '  "vote": "<A | B | C>",\n'
        '  "reason": "<brief private reason>",\n'
        f'  "shared_fact_id": "<{citable}>"\n'
        "}"
    )


def test_a_rendered_source_is_identity_vote_and_evidence_and_never_its_reason():
    source = {
        "label": "Agent 7",
        "vote": "B",
        "reason": "The second relation turns east, and f2 says Kavi is east of Tero.",
        "shared_fact_text": "Kavi is east of Tero.",
    }

    assert render_social_source(source) == (
        f"Agent 7\nVote: B\n{EVIDENCE_HEADER}\nKavi is east of Tero."
    )
    assert render_social_source(source, vote_visibility="hidden") == (
        f"Agent 7\n{EVIDENCE_HEADER}\nKavi is east of Tero."
    )
    assert render_social_source({**source, "shared_fact_text": None}) == (
        "Agent 7\nVote: B"
    )
    # The reason is carried on the record and dropped by the renderer, in every
    # mode - it is the speaker's own record, not a channel.
    for visibility in ("public", "hidden"):
        rendered = render_social_source(source, vote_visibility=visibility)
        assert "Reason" not in rendered
        assert source["reason"] not in rendered
    with pytest.raises(ValueError, match="vote_visibility"):
        render_social_source(source, vote_visibility="whispered")


def test_agent_labels_follow_the_tasks_own_one_based_numbering():
    assert agent_label("agent_001") == "Agent 1"
    assert agent_label("agent_024") == "Agent 24"
    assert agent_label("controller") == "controller"


# ---- round boundaries and logged observables (§15, §17) ----------------


def test_round_boundaries_agree_with_the_microscopic_trajectory():
    config = _config(rounds=3, q=1)
    result, _ = _run(
        config, control=_AlwaysAdvocate(), ballots=_Ballots(share="first_known")
    )
    events = _events(result)
    n = config.game.population_size

    for record in result.rounds:
        window = [
            event for event in events if event["round_index"] == record.round_index
        ]
        assert len(window) == n
        assert record.event["population_state_before"] == window[0][
            "population_state_before"
        ]
        assert record.event["population_state_after"] == window[-1][
            "population_state_after"
        ]
        assert record.event["peer_fact_exposures"] == sum(
            event["peer_fact_exposures"] for event in window
        )
        assert record.event["controller_fact_exposures"] == sum(
            event["controller_fact_exposures"] for event in window
        )
        assert record.event["new_peer_facts"] == sum(
            event["new_peer_facts"] for event in window
        )
        assert record.event["new_controller_facts"] == sum(
            event["new_controller_facts"] for event in window
        )
        assert record.event["mean_supporting_fact_coverage"] == window[-1][
            "mean_supporting_fact_coverage"
        ]


def test_the_round_record_carries_every_declared_observable():
    result, _ = _run(
        _config(CONTROLLED, rounds=2, q=1),
        control=_AlwaysAdvocate(target=CORRECT),
        ballots=_Ballots(share="first_known"),
    )

    required = {
        "occupation_counts_before",
        "occupation_counts_after",
        "m_truth_after",
        "m_ctrl_after",
        "m_order_after",
        "H_vote_after",
        "truth_vote_share",
        "controller_target_share",
        "mean_supporting_fact_coverage",
        "full_proof_agent_share",
        "controller_fact_id",
        "controller_fact_exposures",
        "peer_fact_exposures",
        "controlled_positions",
        "controlled_positions_seed",
        "controlled_positions_hash_or_id",
        "supporting_fact_reach",
    }
    for record in result.rounds:
        assert required <= set(record.event)
        assert 0.0 <= record.event["mean_supporting_fact_coverage"] <= 1.0
        assert 0.0 <= record.event["full_proof_agent_share"] <= 1.0
        assert len(record.event["supporting_fact_reach"]) == len(SUPPORTING)


def test_the_microscopic_record_carries_every_declared_field():
    result, _ = _run(
        _config(CONTROLLED, rounds=1, q=2),
        control=_AlwaysAdvocate(target=CORRECT),
        ballots=_Ballots(share="first_known"),
    )

    required = {
        "focal_known_fact_ids_before",
        "focal_known_fact_ids_after",
        "peer_exposed_fact_ids",
        "controller_fact_id",
        "new_peer_fact_ids",
        "new_controller_fact_ids",
        "round_index",
        "within_round_index",
        "focal_agent_id",
        "sampled_peer_ids",
        "effective_peer_ids",
        "replaced_peer_id",
        "controlled_slot",
        "controller_action",
        "controller_target",
        "intervention_budget",
        "vote_before",
        "vote_after",
    }
    for event in _events(result):
        assert required <= set(event)


def test_knowledge_coverage_rises_as_evidence_circulates():
    config = _config(rounds=4, q=2)
    result, _ = _run(config, ballots=_Ballots(share="first_known"))

    coverage = [record.event["mean_supporting_fact_coverage"] for record in result.rounds]
    assert coverage == sorted(coverage)
    assert coverage[-1] > result.rounds[0].event[
        "mean_supporting_fact_coverage_before"
    ]
    # The hidden-profile task starts with nobody holding the whole proof.
    assert result.rounds[0].event["full_proof_agent_share_before"] == 0.0


def test_no_fact_moves_when_nobody_shares_anything():
    config = _config(rounds=3, q=2)
    result, _ = _run(config, ballots=_Ballots(share="none"))

    initial = {
        str(agent.agent_id): agent.known_fact_ids
        for agent in create_game(config.game)
        .initialize(config.game, config.execution.seed)
        .agents
    }
    for agent in result.final_state.agents:
        assert agent.known_fact_ids == initial[str(agent.agent_id)]
    for record in result.rounds:
        assert record.event["peer_fact_exposures"] == 0
        assert record.event["new_peer_facts"] == 0
        assert (
            record.event["mean_supporting_fact_coverage"]
            == record.event["mean_supporting_fact_coverage_before"]
        )


# ---- configuration guards ----------------------------------------------


def test_classical_dynamics_is_refused_with_a_clear_message():
    config = _config()
    options = {**dict(config.game.options), "dynamics_mode": "classical"}

    with pytest.raises(ValueError, match="not implemented"):
        create_game(config.game).rules(replace(config.game, options=options))


def test_hidden_vote_visibility_is_reserved_rather_than_silently_accepted():
    config = _config()
    options = {**dict(config.game.options), "vote_visibility": "hidden"}

    with pytest.raises(ValueError, match="reserved"):
        create_game(config.game).rules(replace(config.game, options=options))


def test_a_second_prompt_version_is_refused():
    config = _config()
    options = {**dict(config.game.options), "prompt_version": 2}

    with pytest.raises(ValueError, match="prompt_version must be 1"):
        create_game(config.game).rules(replace(config.game, options=options))


def test_the_wrong_prompt_family_is_refused_by_the_runtime():
    config = _config()
    config = replace(config, prompt=replace(config.prompt, prompt_family="hidden_bench_public_ballot"))

    with pytest.raises(ValueError, match="prompt_family"):
        _run(config)


def test_the_horizon_is_rounds_times_population():
    config = _config(rounds=5)
    rules = create_game(config.game).rules(config.game)

    assert rules.rounds == 5
    assert rules.horizon == 5 * config.game.population_size


def test_the_shipped_configs_declare_the_registered_game_and_control():
    no_control = load_run_config(NO_CONTROL, environment={})
    controlled = load_run_config(CONTROLLED, environment={})

    assert no_control.game.type == "relational_imitation_round_feedback"
    assert no_control.control.mechanism == "none"
    assert controlled.control.mechanism == "relational_round_budgeted"
    assert create_game(controlled.game).spec.game_type == controlled.game.type
    assert isinstance(create_control(controlled.control), RelationalRoundBudgetedControl)


def test_the_call_plan_prices_one_call_per_focal_update():
    config = _config(rounds=2, initialization={"mode": "local_vote"})
    plan = create_game(config.game).call_plan(config.game)
    stages = {stage.name: stage for stage in plan.decision_stages}

    n = config.game.population_size
    assert stages["local_initialization"].requests_per_interaction == n
    assert stages["relational_ballot_update"].requests_per_interaction == 2 * n


# ---- semantic votes / option-position bias ------------------------------


def test_the_population_alphabet_is_semantic_not_the_frozen_letters():
    config = _config()
    game = create_game(config.game)
    task = game.load_task(config.game)
    state = game.initialize(config.game, config.execution.seed)

    assert state.possible_answers == task.semantic_answers == OPTIONS
    assert state.correct_answer == task.correct_relation == CORRECT
    # The frozen A/B/C labels survive only as provenance.
    assert set(state.possible_answers) & set(task.option_labels) == set()
    assert tuple(state.task["option_labels"]) == task.option_labels
    assert state.task["correct_option"] == task.correct_option


def test_every_call_gets_its_own_shuffled_letter_map():
    config = _config(rounds=3, q=1)
    result, _ = _run(config, ballots=_Ballots(share="first_known"))

    maps = [_letters(item) for item in result.interactions]
    assert maps
    for letters in maps:
        # A permutation: every letter used once, every relation covered once.
        assert sorted(letters) == list(LETTERS)
        assert sorted(letters.values()) == sorted(OPTIONS)
    # Not one global ordering: the correct answer is not always the same letter.
    correct_letters = {
        next(k for k, v in letters.items() if v == CORRECT) for letters in maps
    }
    assert len(correct_letters) > 1


def test_the_letter_map_replays_and_is_stable_across_retries():
    config = _config(rounds=2, q=1)
    first, _ = _run(config)
    second, _ = _run(config)

    assert [_letters(i) for i in first.interactions] == [
        _letters(i) for i in second.interactions
    ]

    # Stability across retries of one decision: the map is derived from the
    # state and the agent, never from the attempt counter.
    game = create_game(config.game)
    state = game.initialize(config.game, config.execution.seed)
    focal = state.agents[0].agent_id
    assert game.option_letters(state, focal) == game.option_letters(state, focal)
    # ...and different agents at the same turn get different presentations.
    presentations = {
        tuple(sorted(game.option_letters(state, agent.agent_id).items()))
        for agent in state.agents
    }
    assert len(presentations) > 1


def test_a_letter_never_reaches_the_persistent_or_socially_visible_state():
    config = _config(rounds=3, q=2)
    result, _ = _run(config, control=_AlwaysAdvocate(), ballots=_Ballots(share="first_known"))

    for item in result.interactions:
        event = item.transition.event
        for key in ("vote_before", "vote_after", "focal_vote_after"):
            assert event[key] is None or event[key] in OPTIONS
        for value in event["population_state_after"]:
            assert value in OPTIONS
        for source in event["social_sources"]:
            assert source["vote"] in OPTIONS
        assert set(event["occupation_counts_after"]) <= set(OPTIONS)
    for agent in result.final_state.agents:
        assert agent.committed_action in OPTIONS
    for record in result.rounds:
        assert record.event["correct_answer"] == CORRECT
        assert set(record.event["possible_answers"]) == set(OPTIONS)


def test_the_same_letter_means_different_things_to_two_agents_in_one_round():
    """The point of the shuffle: 'vote B' cannot be a population attractor."""

    config = _config(rounds=1, q=1)
    result, ballots = _run(config, ballots=_Ballots(votes=("B",)))

    # Every agent answered the literal letter B...
    assert all(
        json.loads(item.decisions[0].attempts[-1].response.content)["vote"] == "B"
        for item in result.interactions
    )
    # ...and the population did not converge, because B named different
    # relations for different agents.
    votes = {item.transition.event["focal_vote_after"] for item in result.interactions}
    assert len(votes) > 1
    assert votes <= set(OPTIONS)


def test_a_peers_vote_is_shown_in_the_reading_agents_own_letters():
    config = _config(rounds=3, q=1)
    result, ballots = _run(config, ballots=_Ballots(share="first_known"))

    checked = 0
    for prompt, item in zip(ballots.prompts, result.interactions, strict=True):
        inverse = {relation: letter for letter, relation in _letters(item).items()}
        for source, rendered in zip(
            item.transition.event["social_sources"], _social_block(prompt), strict=True
        ):
            if not source["vote"]:
                continue
            assert rendered.splitlines()[1] == f"Vote: {inverse[source['vote']]}"
            checked += 1
    assert checked > 0


def test_scoring_and_the_controller_target_are_semantic():
    config = _config(CONTROLLED, rounds=2, q=1)
    game = create_game(config.game)
    task = game.load_task(config.game)
    result, _ = _run(config, control=_forced(config))

    for record in result.rounds:
        # `target: correct` resolves to the relation, not to a letter.
        assert record.event["controller_target"] == task.correct_relation == CORRECT
        assert record.event["correct_answer"] == CORRECT
        # truth share is computed against the semantic answer.
        share = sum(
            1 for vote in record.event["population_state_after"] if vote == CORRECT
        ) / record.event["N"]
        assert record.event["truth_vote_share"] == pytest.approx(share)


def test_a_random_incorrect_target_is_a_relation_and_is_recorded():
    config = _config(CONTROLLED, rounds=2, q=1)
    options = {**dict(config.control.options), "target": "random_incorrect"}
    config = replace(config, control=replace(config.control, options=options))
    result, _ = _run(config, control=_forced(config, target="random_incorrect"))

    targets = {record.event["controller_target"] for record in result.rounds}
    assert len(targets) == 1
    target = targets.pop()
    assert target in OPTIONS and target != CORRECT


# ---- shared_fact_id is the only inter-agent channel --------------------


class _ChattyBallots(_Ballots):
    """A provider whose prose deliberately smuggles task content.

    Every reason names fact ids and quotes facts verbatim. If prose ever reached
    a peer, this string would show up in someone else's prompt - which is
    exactly what the tests below assert never happens.
    """

    LEAK = (
        "I know f1 and f2: Bavi is northeast of Zora, and Zora is northwest of "
        "Ralo, so the answer must be C."
    )

    def reason_for(self, index: int) -> str:
        return self.LEAK


def test_a_peers_free_form_reason_never_reaches_another_agents_prompt():
    config = _config(rounds=3, q=2)
    ballots = _ChattyBallots(share="none")
    result, _ = _run(config, ballots=ballots)

    # The reasons were produced and stored...
    published = {
        event["focal_agent_id"]: event["focal_reason_after"] for event in _events(result)
    }
    assert published and set(published.values()) == {_ChattyBallots.LEAK}

    # ...and none of them was ever rendered into anybody's prompt.
    for prompt in ballots.prompts:
        social = prompt.split("CURRENT SOCIAL INFORMATION", 1)[-1].split("\nDECISION\n")[0]
        assert _ChattyBallots.LEAK not in social
        assert "Reason:" not in social


def test_a_social_block_shows_only_identity_vote_and_the_shared_fact():
    config = _config(rounds=3, q=2)
    task = create_game(config.game).load_task(config.game)
    result, ballots = _run(config, ballots=_ChattyBallots(share="first_known"))

    checked = 0
    spoke = 0
    for prompt, item in zip(ballots.prompts, result.interactions, strict=True):
        letters = _letters(item)
        inverse = {relation: letter for letter, relation in letters.items()}
        for source, rendered in zip(
            item.transition.event["social_sources"], _social_block(prompt), strict=True
        ):
            lines = rendered.splitlines()
            assert lines[0] == source["label"]
            # The record holds the semantic vote; the prompt shows this call's
            # letter for it.
            assert lines[1] == f"Vote: {inverse[source['vote']]}"
            if source["shared_fact_id"]:
                assert lines[2] == EVIDENCE_HEADER
                assert lines[3] == task.fact_text(source["shared_fact_id"])
                assert len(lines) == 4
            else:
                assert len(lines) == 2
            # Whatever this speaker wrote in prose stayed on the record. A
            # source that has not spoken yet simply has no reason to withhold.
            if source["reason"]:
                assert source["reason"] not in rendered
                spoke += 1
            checked += 1
    assert checked > 0
    assert spoke > 0


def test_the_controller_gets_no_prose_channel_a_peer_does_not_have():
    config = _config(CONTROLLED, rounds=2, q=1)
    task = create_game(config.game).load_task(config.game)
    result, ballots = _run(config, control=_forced(config), ballots=_ChattyBallots())

    controlled = [item for item in result.interactions if item.transition.event["controlled_slot"]]
    assert controlled
    for item in controlled:
        event = item.transition.event
        source = next(
            entry for entry in event["social_sources"] if entry["source_type"] == "control"
        )
        # Its recommendation is recorded, and its rendering is vote + evidence.
        assert source["reason"] == render_control_reason(CORRECT)
        letter = {relation: key for key, relation in _letters(item).items()}[CORRECT]
        assert _rendered_sources(item)[source["slot"]] == (
            f"Agent 25\nVote: {letter}\n{EVIDENCE_HEADER}\n{task.fact_text('f2')}"
        )
    for prompt in ballots.prompts:
        assert render_control_reason(CORRECT) not in prompt
        assert "I recommend" not in prompt


def test_recommendation_only_renders_a_bare_vote_with_no_evidence_line():
    config = _config(CONTROLLED, rounds=2, q=1)
    options = {
        **dict(config.control.options),
        "message_mode": RECOMMENDATION_ONLY,
        "controller_fact_id": None,
    }
    config = replace(config, control=replace(config.control, options=options))
    result, ballots = _run(config, control=_forced(config))

    controlled = [item for item in result.interactions if item.transition.event["controlled_slot"]]
    assert controlled
    for item in controlled:
        event = item.transition.event
        source = next(
            entry for entry in event["social_sources"] if entry["source_type"] == "control"
        )
        assert source["shared_fact_id"] is None
        letter = {relation: key for key, relation in _letters(item).items()}[CORRECT]
        assert _rendered_sources(item)[source["slot"]] == f"Agent 25\nVote: {letter}"
    for prompt in ballots.prompts:
        assert EVIDENCE_HEADER not in prompt
        assert "I recommend" not in prompt


def test_the_reason_is_still_written_to_the_trajectory_for_analysis():
    config = _config(rounds=2, q=1)
    ballots = _ChattyBallots(share="first_known")
    result, _ = _run(config, ballots=ballots)

    for item in result.interactions:
        event = item.transition.event
        parsed = json.loads(item.decisions[0].attempts[-1].response.content)
        # Parsed off the response, applied to the agent, and persisted twice:
        # on the event row and on the agent's own public ballot.
        assert item.decisions[0].action.metadata["reason"] == parsed["reason"]
        assert event["focal_reason_after"] == parsed["reason"]
        focal = event["focal_agent_id"]
        assert next(
            agent.public_reason
            for agent in item.transition.next_state.agents
            if str(agent.agent_id) == focal
        ) == parsed["reason"]

    # And it survives into the serialized episode record. Not every agent is
    # drawn as focal in a short run, so the assertion is over those that spoke.
    payload = result.to_dict()
    spoken = {
        agent["public_reason"]
        for agent in payload["final_state"]["agents"]
        if agent["public_reason"] is not None
    }
    assert spoken == {_ChattyBallots.LEAK}
    assert all(
        item["event"]["focal_reason_after"] == _ChattyBallots.LEAK
        for item in payload["interactions"]
    )


def test_an_agent_never_sees_its_own_previous_reason_either():
    """Prose is not an internal memory channel any more than a social one.

    An agent that got its own last reason back could carry conclusions forward
    in text that nothing in the state records, and `K_i` would stop being the
    only thing that explains what it knows.
    """

    config = _config(rounds=3, q=1)
    ballots = _ChattyBallots()
    result, _ = _run(config, ballots=ballots)

    with_history = [
        (prompt, item.transition.event)
        for prompt, item in zip(ballots.prompts, result.interactions, strict=True)
        if item.transition.event["focal_reason_before"]
    ]
    assert with_history, "no agent was drawn twice; the test proves nothing"
    for prompt, event in with_history:
        # The reason exists on the record...
        assert event["focal_reason_before"] == _ChattyBallots.LEAK
        # ...and appears nowhere in the prompt that agent is handed.
        assert _ChattyBallots.LEAK not in prompt
        assert "Reason:" not in prompt


def test_the_standing_position_block_carries_the_vote_and_only_the_vote():
    config = _config(rounds=3, q=1)
    ballots = _Ballots(share="first_known")
    result, _ = _run(config, ballots=ballots)

    seen = 0
    for prompt, item in zip(ballots.prompts, result.interactions, strict=True):
        event = item.transition.event
        if event["focal_vote_before"] is None:
            continue
        block = prompt.split("YOUR CURRENT POSITION\n\n", 1)[1]
        block = block.split("\n\n", 1)[0]
        assert block == f"Vote: {event['focal_vote_before']}"
        # Not the previous reason, and not the fact it previously exposed - the
        # latter is already in YOUR CURRENT KNOWLEDGE.
        assert "Evidence you are sharing" not in prompt
        seen += 1
    assert seen > 0


# ---- social_distrust ----------------------------------------------------


def _distrust(config, value):
    options = {**dict(config.game.options), "social_distrust": value}
    return replace(config, game=replace(config.game, options=options))


def test_social_distrust_defaults_to_true_and_preserves_the_historical_text():
    config = _config(rounds=1)

    assert create_game(config.game).rules(config.game).social_distrust is True
    _, ballots = _run(config)
    assert ballots.prompts
    for prompt in ballots.prompts:
        assert SOCIAL_ENVIRONMENT_DISTRUST in prompt
        assert "objectives that differ from yours" in prompt


def test_social_distrust_false_swaps_in_the_neutral_cooperative_baseline():
    config = _distrust(_config(rounds=1), False)

    assert create_game(config.game).rules(config.game).social_distrust is False
    _, ballots = _run(config)
    assert ballots.prompts
    for prompt in ballots.prompts:
        assert SOCIAL_ENVIRONMENT_NEUTRAL in prompt
        assert SOCIAL_ENVIRONMENT_DISTRUST not in prompt
        assert "objectives that differ from yours" not in prompt
        # The distributed-information statement survives; only the strategic
        # warning is replaced.
        assert "Different participants know different facts" in prompt
        assert "trying to identify the correct answer" in prompt


def test_the_two_environments_are_fixed_blocks_with_different_definitions():
    true_prompt = relational_public_ballot_prompt(fact_ids=("f1",), social_distrust=True)
    false_prompt = relational_public_ballot_prompt(fact_ids=("f1",), social_distrust=False)

    assert social_environment(True) == SOCIAL_ENVIRONMENT_DISTRUST
    assert social_environment(False) == SOCIAL_ENVIRONMENT_NEUTRAL
    for prompt in (true_prompt, false_prompt):
        block = prompt.block("social_environment")
        assert block.binding == "fixed"
        with pytest.raises(ValueError, match="fixed block cannot be rebound"):
            block.bind("something else")
    # The condition is pinned by the definition, not by convention.
    assert (
        true_prompt.block("social_environment").value
        != false_prompt.block("social_environment").value
    )


def test_social_distrust_changes_nothing_but_the_environment_block():
    trusting, _ = _run(_distrust(_config(rounds=2, q=1), False),
                       ballots=_Ballots(share="first_known"))
    wary, _ = _run(_distrust(_config(rounds=2, q=1), True),
                   ballots=_Ballots(share="first_known"))

    # Same schedule, same citations, same knowledge trajectory.
    assert [event["focal_agent_id"] for event in _events(trusting)] == [
        event["focal_agent_id"] for event in _events(wary)
    ]
    assert [agent.known_fact_ids for agent in trusting.final_state.agents] == [
        agent.known_fact_ids for agent in wary.final_state.agents
    ]


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_a_non_boolean_social_distrust_is_refused(value):
    config = _distrust(_config(rounds=1), value)

    with pytest.raises(ValueError, match="social_distrust must be a boolean"):
        create_game(config.game).rules(config.game)


# ---- the advertised citable fact ids ------------------------------------


def test_the_instruction_advertises_only_the_facts_this_agent_may_share():
    config = _config(rounds=3, q=1)
    result, ballots = _run(config, ballots=_Ballots(share="first_known"))

    empty = rich = 0
    for prompt, item in zip(ballots.prompts, result.interactions, strict=True):
        known = tuple(item.transition.event["focal_known_fact_ids_before"])
        expected = f'"shared_fact_id": "<{" | ".join((*known, "none"))}>"'
        assert expected in prompt
        if known:
            rich += 1
            # No fact this agent does not hold is offered to it.
            for fact_id in set(item.transition.next_state.fact_ids) - set(known):
                assert f"<{fact_id} " not in prompt and f"| {fact_id} " not in prompt
        else:
            empty += 1
            assert '"shared_fact_id": "<none>"' in prompt
    assert empty > 0 and rich > 0


def test_the_contract_rejects_a_fact_the_agent_does_not_hold():
    contract = RelationalBallotContract(
        allowed_values=LETTERS,
        options={"fact_ids": ("f2",), "relations": ("NORTH",)},
    )

    assert contract.validate('{"vote":"A","reason":"ok","shared_fact_id":"f2"}').is_valid
    assert contract.validate('{"vote":"A","reason":"ok","shared_fact_id":"none"}').is_valid
    # f1 exists in the task but not in this agent's knowledge set.
    rejected = contract.validate('{"vote":"A","reason":"ok","shared_fact_id":"f1"}')
    assert not rejected.is_valid
    assert rejected.issues[0].field == "response.shared_fact_id"


def test_an_agent_that_knows_nothing_may_cite_nothing():
    contract = RelationalBallotContract(
        allowed_values=LETTERS, options={"fact_ids": (), "relations": ("NORTH",)}
    )

    assert '"shared_fact_id": "<none>"' in contract.instruction()
    assert contract.validate('{"vote":"A","reason":"ok","shared_fact_id":"none"}').is_valid
    assert not contract.validate(
        '{"vote":"A","reason":"ok","shared_fact_id":"f1"}'
    ).is_valid


def test_runtime_evidence_honesty_still_backstops_the_contract():
    """Both safeguards stay: the contract is first, validate_action is second."""

    config = _config(rounds=1)
    game = create_game(config.game)
    state = game.initialize(config.game, config.execution.seed)
    focal = next(agent for agent in state.agents if agent.known_fact_ids)
    request = game.ballot_request(state, focal.agent_id, (), config.game)
    unknown = next(f for f in state.fact_ids if f not in focal.known_fact_ids)

    # The contract would already have caught this...
    assert not request.prompt.response_contract.validate(
        json.dumps({"vote": "A", "reason": "ok", "shared_fact_id": unknown})
    ).is_valid
    # ...and validate_action independently rejects it from the state itself.
    action = game.parse_action(
        request, json.dumps({"vote": "A", "reason": "ok", "shared_fact_id": unknown})
    )
    issues = game.validate_action(state, request, action, config.game).issues
    assert [issue.field for issue in issues] == ["action.shared_fact_id"]


def test_knowledge_propagation_is_unchanged_when_prose_is_withheld():
    config = _config(rounds=3, q=2)
    quiet, _ = _run(config, ballots=_Ballots(share="first_known"))
    chatty, _ = _run(config, ballots=_ChattyBallots(share="first_known"))

    # Same votes, same citations, therefore the same knowledge trajectory: the
    # prose is inert by construction, not merely unread.
    assert [agent.known_fact_ids for agent in quiet.final_state.agents] == [
        agent.known_fact_ids for agent in chatty.final_state.agents
    ]
    assert [event["new_peer_fact_ids"] for event in _events(quiet)] == [
        event["new_peer_fact_ids"] for event in _events(chatty)
    ]
    assert [
        record.event["mean_supporting_fact_coverage"] for record in quiet.rounds
    ] == [record.event["mean_supporting_fact_coverage"] for record in chatty.rounds]


def test_the_prompt_tells_the_agent_its_reason_is_not_shown_to_others():
    """The instructions must not invite a channel the runtime does not provide."""

    _, ballots = _run(_config(rounds=1, q=1))

    assert ballots.prompts
    for prompt in ballots.prompts:
        assert "Your reason is your own record: it is\nnot shown to anyone." in prompt or (
            "Your reason is your own record: it is not shown\nto anyone." in prompt
        )
        assert (
            "Sharing a fact is the only way to pass information to other participants."
            in prompt
        )


# ---- backward compatibility (§23) --------------------------------------


def test_the_new_game_is_registered_without_disturbing_the_existing_ones():
    """Adding this game must be purely additive to the three registries."""

    from mas_cc.control import create_default_control_registry
    from mas_cc.games import create_default_game_registry
    from mas_cc.games.registry import (
        create_default_prompt_registry,
        register_game_prompt_factories,
    )

    games = create_default_game_registry()
    controls = create_default_control_registry()
    prompts = register_game_prompt_factories(create_default_prompt_registry())

    assert "relational_imitation_round_feedback" in games.names()
    assert "relational_round_budgeted" in controls.names()
    # Every pre-existing entry still resolves to exactly what it did before.
    assert {
        "hidden_bench_imitation",
        "hidden_bench_imitation_round_feedback",
        "hidden_bench_naming",
        "hidden_bench_vanilla",
        "naming_convention",
        "synthetic_bernoulli",
        "synthetic_controlled_markov",
        "synthetic_markov",
        "toy_coordination",
    } <= set(games.names())
    assert {
        "none",
        "forced_action",
        "threshold_target",
        "soft_target",
        "round_soft_target_budgeted",
    } <= set(controls.names())
    assert prompts.get("hidden_bench_public_ballot", 1).family == (
        "hidden_bench_public_ballot"
    )
    assert prompts.get("relational_public_ballot", 1).family == "relational_public_ballot"


def test_the_relational_prompt_family_is_distinct_from_the_hidden_bench_one():
    from mas_cc.games.hidden_bench.imitation_round_feedback.prompts import (
        PROMPT_FAMILY as HIDDEN_BENCH_FAMILY,
        hidden_bench_public_ballot_prompt,
    )
    from mas_cc.games.relational_reasoning.imitation_round_feedback.prompts import (
        PROMPT_FAMILY as RELATIONAL_FAMILY,
        relational_public_ballot_prompt,
    )

    assert HIDDEN_BENCH_FAMILY != RELATIONAL_FAMILY
    relational = relational_public_ballot_prompt()
    hidden_bench = hidden_bench_public_ballot_prompt()
    blocks = {block.name for block in relational.blocks}

    # The extra ballot field is what makes it a family of its own: HiddenBench
    # has no notion of an agent's own citable fact set.
    assert "known_facts" in blocks
    assert "private_information" not in blocks
    assert {block.name for block in hidden_bench.blocks} != blocks
    assert relational.response_contract.type == "relational_public_ballot"
    assert "shared_fact_id" in relational.response_contract.instruction()
    assert "shared_fact_id" not in hidden_bench.response_contract.instruction()
