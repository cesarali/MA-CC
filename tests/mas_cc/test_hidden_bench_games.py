"""Both HiddenBench games end to end against the mock provider (brief §9.1).

Deterministic under a fixed seed, and `call_plan` matches the observed provider
call count exactly - the second is the one that keeps budget preflight honest,
since a game that under-reports its own demand will be priced for a run it never
makes.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from mas_cc.config import load_run_config
from mas_cc.config.models import GameConfig
from mas_cc.games import create_game
from mas_cc.games.hidden_bench import run_hidden_bench_game
from mas_cc.games.hidden_bench.data import DEFAULT_CORPUS_ROOT
from mas_cc.games.hidden_bench.naming.game import HiddenBenchNamingGame
from mas_cc.games.hidden_bench.records import COMMIT, DISCUSS, EXCHANGE, POST_VOTE, PRE_VOTE
from mas_cc.games.hidden_bench.rules import PayoffRules
from mas_cc.games.hidden_bench.schemas import HiddenBenchDataError
from mas_cc.games.hidden_bench.vanilla.game import HiddenBenchVanillaGame
from mas_cc.games.registry import create_default_game_registry, game_metrics
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider

pytestmark = pytest.mark.skipif(
    not (DEFAULT_CORPUS_ROOT / "canonical" / "tasks.json").exists(),
    reason="HiddenBench corpus not present; see docs/hidden_bench/data_provenance.md",
)

ENVIRONMENT = {"POTSDAM_API_KEY": "test-key", "BASE_POTSDAM_LLM_URL": "http://localhost"}


def _responder(request):
    """A cooperative agent: votes West City, and says something when asked."""

    text = "\n".join(message.content for message in request.messages)
    if "following JSON format" in text:
        return '{"vote": "West City", "rationale": "the bridge is still passable"}'
    return "The river level is still below the bridge to West City."


def _run(config_path: str, **option_overrides):
    config = load_run_config(config_path, environment=ENVIRONMENT)
    if option_overrides:
        options = {**dict(config.game.options), **option_overrides}
        config = replace(config, game=replace(config.game, options=options))
    game = create_game(config.game)
    provider = MockLLMProvider(
        replace(config.llm_provider, type="mock"), response_factory=_responder
    )
    result = asyncio.run(run_hidden_bench_game(game, config, provider))
    return game, config, result


@pytest.mark.parametrize(
    "config_path",
    ["configs/runs/hidden_bench_vanilla.yaml", "configs/runs/hidden_bench_naming.yaml"],
    ids=["vanilla", "naming"],
)
def test_shipped_configs_run_end_to_end(config_path):
    _, config, result = _run(config_path, rounds=2)
    assert result.termination_reason
    assert result.interactions
    assert result.final_state.terminated


@pytest.mark.parametrize(
    "config_path",
    ["configs/runs/hidden_bench_vanilla.yaml", "configs/runs/hidden_bench_naming.yaml"],
    ids=["vanilla", "naming"],
)
def test_call_plan_matches_the_observed_call_count_exactly(config_path):
    game, config, result = _run(config_path, rounds=2)
    planned = game.call_plan(config.game).provider_requests.lower
    assert result.logical_decisions == planned


@pytest.mark.parametrize(
    "config_path",
    ["configs/runs/hidden_bench_vanilla.yaml", "configs/runs/hidden_bench_naming.yaml"],
    ids=["vanilla", "naming"],
)
def test_runs_are_deterministic_under_a_fixed_seed(config_path):
    _, _, first = _run(config_path, rounds=2)
    _, _, second = _run(config_path, rounds=2)
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )


def test_vanilla_call_count_is_n_plus_n_times_t_plus_n():
    """§5's arithmetic, checked at the numbers the paper actually used."""

    game = HiddenBenchVanillaGame()
    for n, rounds in [(4, 15), (5, 10), (7, 3)]:
        config = GameConfig(
            type="hidden_bench_vanilla",
            population_size=n,
            horizon=1,
            options={"task_id": "evacuation_west_city", "assignment_scheme": "exact_replication", "rounds": rounds},
        )
        assert game.call_plan(config).provider_requests.lower == n + n * rounds + n


def test_naming_call_count_is_rounds_times_two_times_m_plus_one():
    """§6's arithmetic."""

    game = HiddenBenchNamingGame()
    for rounds, messages in [(10, 1), (5, 2), (3, 0)]:
        config = GameConfig(
            type="hidden_bench_naming",
            population_size=5,
            horizon=1,
            options={
                "task_id": "evacuation_west_city",
                "assignment_scheme": "exact_replication",
                "rounds": rounds,
                "messages_per_turn": messages,
                "stop_on_consensus": False,
            },
        )
        assert game.call_plan(config).provider_requests.lower == rounds * 2 * (messages + 1)


# --------------------------------------------------------------------------
# Phase structure
# --------------------------------------------------------------------------


def test_vanilla_phases_run_in_order_and_terminate_after_the_post_vote():
    _, _, result = _run("configs/runs/hidden_bench_vanilla.yaml", rounds=2)
    phases = [item.phase for item in result.interactions]
    assert phases[0] == PRE_VOTE
    assert phases[-1] == POST_VOTE
    assert set(phases[1:-1]) == {DISCUSS}
    assert phases.count(PRE_VOTE) == 1
    assert phases.count(POST_VOTE) == 1
    assert result.termination_reason == "post_vote_recorded"


def test_full_profile_with_zero_rounds_is_the_y_full_ceiling_condition():
    """§1.2: no discussion, everyone holds all of Iu, one vote each side.

    This is why Full Profile is a config flag rather than a second game: Y_full
    and Y_post come out of the same code path, so they are comparable.
    """

    game, config, result = _run(
        "configs/runs/hidden_bench_vanilla.yaml", profile="full", rounds=0
    )
    phases = [item.phase for item in result.interactions]
    assert phases == [PRE_VOTE, POST_VOTE]
    assert result.logical_decisions == 2 * config.game.population_size
    everything = set(result.final_state.hidden_information)
    for agent in result.final_state.agents:
        assert set(agent.private_information) == everything


def test_rounds_unit_flag_switches_between_passes_and_speaking_turns():
    """The two `T` conventions differ by a factor of N; both are reachable."""

    game = HiddenBenchVanillaGame()
    common = {"task_id": "evacuation_west_city", "assignment_scheme": "exact_replication", "rounds": 15}
    passes = GameConfig(
        type="hidden_bench_vanilla", population_size=4, horizon=1, options=common
    )
    turns = GameConfig(
        type="hidden_bench_vanilla",
        population_size=4,
        horizon=1,
        options={**common, "rounds_are_speaking_turns": True},
    )
    assert game.call_plan(passes).metadata["discussion_turns"] == 60
    assert game.call_plan(turns).metadata["discussion_turns"] == 15


def test_naming_alternates_exchange_and_commit():
    _, _, result = _run("configs/runs/hidden_bench_naming.yaml", rounds=3, messages_per_turn=1)
    phases = [item.phase for item in result.interactions]
    assert phases == [EXCHANGE, COMMIT] * 3
    for item in result.interactions:
        assert len(item.participants) == 2


def test_naming_stops_early_on_consensus():
    """Everyone commits to the same option, so the run must not use all rounds."""

    _, _, result = _run(
        "configs/runs/hidden_bench_naming.yaml", rounds=40, stop_on_consensus=True
    )
    assert result.termination_reason == "consensus_reached"
    assert len(result.interactions) < 40 * 2


# --------------------------------------------------------------------------
# Options validation
# --------------------------------------------------------------------------


def test_n_agents_disagreeing_with_population_size_is_rejected():
    """The footgun the grid config's comment warns about, enforced."""

    game = HiddenBenchVanillaGame()
    config = GameConfig(
        type="hidden_bench_vanilla", population_size=4, horizon=1, options={"n_agents": 6}
    )
    with pytest.raises(HiddenBenchDataError, match="disagrees with"):
        game.initialize(config, 0)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"profile": "partial"}, "profile"),
        ({"assignment_scheme": "redundant"}, "assignment_scheme"),
        ({"aggregation": "plurality"}, "aggregation"),
        ({"dissenter_fraction": 0.5}, "dissenter_extra_prompt"),
    ],
)
def test_bad_options_fail_loudly_naming_the_field(options, message):
    game = HiddenBenchVanillaGame()
    config = GameConfig(
        type="hidden_bench_vanilla", population_size=4, horizon=1, options=options
    )
    with pytest.raises(HiddenBenchDataError, match=message):
        game.initialize(config, 0)


# --------------------------------------------------------------------------
# Payoff modes (§6)
# --------------------------------------------------------------------------


def test_coordination_mode_does_not_pay_for_being_right():
    """Convention formation and truth-finding stay separable observables."""

    rules = PayoffRules(mode="coordination", match_reward=1.0, mismatch_penalty=-1.0)
    assert rules.payoff(matched=True, correct=True) == 1.0
    assert rules.payoff(matched=True, correct=False) == 1.0
    assert rules.payoff(matched=False, correct=True) == -1.0


def test_correctness_mode_pays_only_for_being_right():
    rules = PayoffRules(mode="correctness", match_reward=1.0, mismatch_penalty=-1.0)
    assert rules.payoff(matched=False, correct=True) == 1.0
    assert rules.payoff(matched=True, correct=False) == -1.0


def test_combined_mode_adds_the_bonus_only_when_both_hold():
    rules = PayoffRules(
        mode="coordination_plus_correctness", match_reward=1.0, mismatch_penalty=-1.0,
        correctness_bonus=0.5,
    )
    assert rules.payoff(matched=True, correct=True) == 1.5
    assert rules.payoff(matched=True, correct=False) == 1.0
    assert rules.payoff(matched=False, correct=True) == -1.0


def test_unknown_payoff_mode_is_rejected():
    with pytest.raises(HiddenBenchDataError, match="payoff.mode"):
        PayoffRules(mode="altruism")


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


@pytest.mark.parametrize("game_type", ["hidden_bench_vanilla", "hidden_bench_naming"])
def test_games_are_registered_and_declare_compatible_metrics(game_type):
    """`game_metrics` raises if a metric's required family mismatches the game."""

    registry = create_default_game_registry()
    assert game_type in registry.names()
    game = registry.create(
        GameConfig(type=game_type, population_size=4, horizon=1, options={"rounds": 1})
    )
    metrics, adapter = game_metrics(game)
    assert metrics and adapter is not None
    assert game.spec.game_family == "choice"


def test_runtime_rejects_a_prompt_family_that_is_not_this_games():
    config = load_run_config("configs/runs/hidden_bench_vanilla.yaml", environment=ENVIRONMENT)
    config = replace(
        config, prompt=replace(config.prompt, prompt_family="naming_convention_decision")
    )
    game = create_game(config.game)
    provider = MockLLMProvider(replace(config.llm_provider, type="mock"), response_factory=_responder)
    with pytest.raises(ValueError, match="prompt.prompt_family"):
        asyncio.run(run_hidden_bench_game(game, config, provider))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_path,expected",
    [
        ("configs/runs/hidden_bench_vanilla.yaml", {"y_pre", "y_post", "improvement"}),
        (
            "configs/runs/hidden_bench_naming.yaml",
            {"accuracy_first_commitment", "accuracy_final", "improvement"},
        ),
    ],
    ids=["vanilla", "naming"],
)
def test_final_metrics_are_computable_from_a_finished_episode(config_path, expected):
    game, _, result = _run(config_path, rounds=3)
    metrics, to_round_view = game_metrics(game)
    views = tuple(
        to_round_view(item.transition.next_state) for item in result.interactions
    )
    computed = {
        metric.name: metric.compute_final(views)
        for metric in metrics
        if hasattr(metric, "compute_final")
    }
    assert expected <= set(computed)
    # Every agent voted West City, which is correct for this task.
    for name in expected - {"improvement"}:
        assert computed[name] == 1.0
    assert computed["improvement"] == 0.0


def test_streaming_metrics_produce_a_value_every_round():
    game, _, result = _run("configs/runs/hidden_bench_vanilla.yaml", rounds=2)
    metrics, to_round_view = game_metrics(game)
    streaming = [metric for metric in metrics if hasattr(metric, "compute_round")]
    for item in result.interactions:
        view = to_round_view(item.transition.next_state)
        for metric in streaming:
            assert metric.compute_round(view), f"{metric.name} produced no value"


def test_disclosure_rate_rises_when_agents_actually_disclose():
    """The paper's central diagnostic must move when the thing it measures does."""

    config = load_run_config("configs/runs/hidden_bench_vanilla.yaml", environment=ENVIRONMENT)
    config = replace(config, game=replace(config.game, options={**dict(config.game.options), "rounds": 2}))
    game = create_game(config.game)
    state = game.initialize(config.game, config.execution.seed)
    secrets = list(state.hidden_information)

    counter = {"index": 0}

    def disclosing(request):
        text = "\n".join(message.content for message in request.messages)
        if "following JSON format" in text:
            return '{"vote": "West City", "rationale": "r"}'
        fact = secrets[counter["index"] % len(secrets)]
        counter["index"] += 1
        return fact

    provider = MockLLMProvider(replace(config.llm_provider, type="mock"), response_factory=disclosing)
    result = asyncio.run(run_hidden_bench_game(game, config, provider))
    _, to_round_view = game_metrics(game)
    final = to_round_view(result.final_state)
    rate = final.recent_history[-1]["unshared_disclosure_rate"]
    assert rate == 1.0, "every hidden fact was stated verbatim but was not detected"

    silent = _run("configs/runs/hidden_bench_vanilla.yaml", rounds=2)[2]
    silent_view = to_round_view(silent.final_state)
    assert silent_view.recent_history[-1]["unshared_disclosure_rate"] == 0.0
