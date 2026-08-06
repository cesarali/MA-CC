"""No agent may see another agent's private information (brief §9.3).

This is the single most important correctness property of the whole benchmark.
If it leaks, every number the benchmark produces is meaningless - a Hidden
Profile task where the profile is not actually hidden is just an easy task.

The check is deliberately paranoid: it walks a full episode of each game and,
for every observation handed to every agent, asserts that no *other* agent's
private fact appears anywhere in the rendered prompt - not in a block value, not
in a compiled message, not in the observation dict that gets written to the
audit log. The only permitted route is that agent having said it out loud, so
the test also proves the leak-detector works by checking that a disclosure
through a message *is* found where it should be.
"""

from __future__ import annotations

import random

import pytest

from mas_cc.config.models import GameConfig
from mas_cc.games.hidden_bench.data import DEFAULT_CORPUS_ROOT, normalized_text
from mas_cc.games.hidden_bench.naming.game import HiddenBenchNamingGame
from mas_cc.games.hidden_bench.records import DISCUSS, EXCHANGE
from mas_cc.games.hidden_bench.vanilla.game import HiddenBenchVanillaGame

pytestmark = pytest.mark.skipif(
    not (DEFAULT_CORPUS_ROOT / "canonical" / "tasks.json").exists(),
    reason="HiddenBench corpus not present; see docs/hidden_bench/data_provenance.md",
)

TASK = "evacuation_west_city"


def _vanilla(**options) -> tuple[HiddenBenchVanillaGame, GameConfig]:
    base = {
        "task_id": TASK,
        "profile": "hidden",
        "assignment_scheme": "exact_replication",
        "rounds": 2,
    }
    return HiddenBenchVanillaGame(), GameConfig(
        type="hidden_bench_vanilla", population_size=5, horizon=1, options={**base, **options}
    )


def _naming(**options) -> tuple[HiddenBenchNamingGame, GameConfig]:
    base = {
        "task_id": TASK,
        "profile": "hidden",
        "assignment_scheme": "exact_replication",
        "rounds": 4,
        "messages_per_turn": 1,
        "stop_on_consensus": False,
    }
    return HiddenBenchNamingGame(), GameConfig(
        type="hidden_bench_naming", population_size=5, horizon=1, options={**base, **options}
    )


def _surfaces(observation, prompt) -> str:
    """Everything an agent or an audit log could possibly read, as one blob."""

    compiled = prompt.compile()
    return "\n".join(
        [
            str(observation.to_dict()),
            *(block.content for block in compiled.blocks),
            *(message.content for message in compiled.messages),
        ]
    )


def _walk(game, config, *, responses, seed=17):
    """Run an episode, yielding (state, observation, prompt) for every decision."""

    state = game.initialize(config, seed)
    rng = random.Random(seed)
    while game.detect_termination(state, config) is None:
        participants = game.select_participants(state, config, rng)
        observations = game.construct_observations(state, participants, config)
        requests = game.build_decision_requests(state, observations, config)
        for observation, request in zip(observations, requests, strict=True):
            yield state, observation, request.prompt
        actions = tuple(
            game.parse_action(request, responses(request)) for request in requests
        )
        for request, action in zip(requests, actions, strict=True):
            result = game.validate_action(state, request, action, config)
            assert result.is_valid, result.issues
        state = game.apply_transition(state, participants, actions, config).next_state


def _silent(request) -> str:
    """Responses that disclose nothing, so any leak found must be structural."""

    if request.stage in {DISCUSS, EXCHANGE}:
        return "I think we should weigh the options carefully before deciding."
    return '{"vote": "West City", "rationale": "the shared information favours it"}'


@pytest.mark.parametrize("build", [_vanilla, _naming], ids=["vanilla", "naming"])
def test_no_observation_contains_a_hidden_fact_the_focal_agent_does_not_hold(build):
    """The privacy property, stated the only way that is actually correct.

    Not "no other agent's fact": under `exact_replication` with N > C two agents
    legitimately hold the *same* fact, so a shared fact appearing in both their
    observations is the design working, not a leak. The real property is that a
    hidden fact reaches an agent only if that agent holds it - and, since these
    responses disclose nothing, only if it holds it.
    """

    game, config = build()
    leaks = []
    for state, observation, prompt in _walk(game, config, responses=_silent):
        surface = normalized_text(_surfaces(observation, prompt))
        held = {normalized_text(fact) for fact in observation.visible_state["presented_information"]}
        for fact in state.hidden_information:
            if normalized_text(fact) in surface and normalized_text(fact) not in held:
                leaks.append((str(observation.agent_id), fact))
    assert not leaks, f"hidden information leaked into {len(leaks)} observation(s): {leaks[:3]}"


@pytest.mark.parametrize("build", [_vanilla, _naming], ids=["vanilla", "naming"])
def test_every_agent_does_see_its_own_private_facts(build):
    """The complement: the split must not be so tight that nobody sees anything."""

    game, config = build()
    seen = set()
    for _, observation, prompt in _walk(game, config, responses=_silent):
        surface = normalized_text(_surfaces(observation, prompt))
        presented = observation.visible_state["presented_information"]
        assert presented, "an agent was shown no information at all"
        for fact in presented:
            assert normalized_text(fact) in surface
            seen.add(fact)
    assert len(seen) > 1


def test_the_leak_detector_actually_detects_a_leak():
    """A negative test is worthless if the detector cannot see a real leak.

    Here an agent *does* disclose its private fact in a discussion turn, which
    is the one permitted route. The fact must then appear in other agents'
    observations - proving the assertion above is measuring something.
    """

    game, config = _vanilla(rounds=1)
    state = game.initialize(config, 17)
    secret = state.agents[0].private_information[0]

    def leaky(request):
        if request.stage == DISCUSS:
            return secret
        return '{"vote": "West City", "rationale": "r"}'

    found = False
    for _, observation, prompt in _walk(game, config, responses=leaky):
        if observation.agent_id == state.agents[0].agent_id:
            continue
        if normalized_text(secret) in normalized_text(_surfaces(observation, prompt)):
            found = True
    assert found, "a fact spoken aloud never reached another agent - the detector is broken"


@pytest.mark.parametrize("build", [_vanilla, _naming], ids=["vanilla", "naming"])
def test_full_profile_deliberately_shows_everyone_everything(build):
    """The one condition where sharing is correct, so the test must expect it."""

    game, config = build(profile="full")
    state = game.initialize(config, 3)
    everything = set(state.hidden_information)
    for agent in state.agents:
        assert set(agent.private_information) == everything


def test_naming_agents_only_ever_see_words_from_partners_they_actually_met():
    """The dyadic game's own boundary: no global transcript reaches an agent.

    An agent legitimately remembers what *past* partners said to it - that is
    what private per-partner memory is - so the property is not "only the
    current partner". It is that a message from an agent this one has never been
    paired with must never appear, by any route. That is what makes the
    transcript a local object and information diffusion a graph process.

    Also asserts the episode actually mixed: if every agent met every other
    agent, the test would pass vacuously.
    """

    game, config = _naming(rounds=6)
    marker = "SIDE-CHANNEL"
    met: dict[str, set[str]] = {}
    violations = []

    def tagged(request):
        if request.stage == EXCHANGE:
            return f"{marker}-{request.agent_id}"
        return '{"vote": "West City", "rationale": "r"}'

    for state, observation, prompt in _walk(game, config, responses=tagged):
        focal = str(observation.agent_id)
        partners = met.setdefault(focal, set())
        surface = _surfaces(observation, prompt)
        for agent in state.agents:
            other = str(agent.agent_id)
            if other == focal or other in partners:
                continue
            if f"{marker}-{other}" in surface:
                violations.append((focal, other))
        partners.update(str(item) for item in observation.participants if str(item) != focal)

    assert not violations, f"words reached an agent from a stranger: {violations[:3]}"
    unmet = [focal for focal, partners in met.items() if len(partners) < len(met) - 1]
    assert unmet, "every agent met every other agent; the test proved nothing"


def test_naming_memory_carries_only_what_a_partner_said_to_this_agent():
    """Private memory is per-partner, not a shared log."""

    game, config = _naming(rounds=6)

    def tagged(request):
        if request.stage == EXCHANGE:
            return f"MSG-FROM-{request.agent_id}"
        return '{"vote": "West City", "rationale": "r"}'

    state = game.initialize(config, 17)
    rng = random.Random(17)
    while game.detect_termination(state, config) is None:
        participants = game.select_participants(state, config, rng)
        observations = game.construct_observations(state, participants, config)
        requests = game.build_decision_requests(state, observations, config)
        actions = tuple(game.parse_action(request, tagged(request)) for request in requests)
        state = game.apply_transition(state, participants, actions, config).next_state

    for agent in state.agents:
        for entry in agent.memory:
            for message in entry["partner_said"]:
                assert f"MSG-FROM-{agent.agent_id}" != message, (
                    "an agent's memory recorded its own message as its partner's"
                )
            assert entry["partner_choice"] is not None
