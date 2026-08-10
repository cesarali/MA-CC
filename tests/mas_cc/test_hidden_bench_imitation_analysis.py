"""Focused fixtures for the first HiddenBench imitation diagnostics."""

from __future__ import annotations

import itertools
import asyncio
from dataclasses import replace

import pytest

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.control import create_control
from mas_cc.games import create_game
from mas_cc.games.hidden_bench.data import DEFAULT_CORPUS_ROOT
from mas_cc.games.hidden_bench.imitation import run_hidden_bench_imitation_game
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider
from mas_cc.games.hidden_bench.imitation.analysis import (
    adapt_event,
    binary_action_entropy_bits,
    bootstrap_episode_ids,
    information_analysis,
)
from mas_cc.games.hidden_bench.imitation.metrics import (
    behavioral_transition_metrics,
    population_observables,
)


def _event(
    before,
    after,
    *,
    options=("A", "B", "C"),
    target="C",
    action="ADVOCATE_Z",
    sensor=None,
    episode="episode-0",
    index=1,
):
    sensor = sensor or {option: before.count(option) for option in options}
    return {
        "episode_id": episode,
        "interaction_index": index,
        "N": len(before),
        "K": len(options),
        "dynamics_mode": "classical",
        "possible_answers": list(options),
        "correct_answer": target,
        "analysis_target": target,
        "population_state_before": list(before),
        "occupation_counts_before": {option: before.count(option) for option in reversed(options)},
        "population_state_after": list(after),
        "occupation_counts_after": {option: after.count(option) for option in reversed(options)},
        "population_shares_before": {option: before.count(option) / len(before) for option in options},
        "population_shares_after": {option: after.count(option) / len(after) for option in options},
        "focal_opinion_before": before[0],
        "focal_opinion_after": after[0],
        "controller_enabled": action is not None,
        "controller_target": target if action is not None else None,
        "controller_action": action,
        "sensor_sample_size": sum(sensor.values()) if action is not None else None,
        "sensor_count_vector": sensor if action is not None else {option: 0 for option in options},
    }


def test_exact_deltas_and_target_adoption_on_hand_created_states():
    before = population_observables(["A", "A", "B"], ["A", "B", "C"], "B", "C")
    after = population_observables(["C", "A", "B"], ["A", "B", "C"], "B", "C")
    result = behavioral_transition_metrics(
        before,
        after,
        focal_opinion_before="A",
        focal_opinion_after="C",
        target="C",
        controller_action="ADVOCATE_Z",
        sensor_count_vector={"A": 1, "B": 0, "C": 1},
        sensor_sample_size=2,
    )
    assert result["delta_m_ctrl"] == pytest.approx(0.5)
    assert result["delta_m_truth"] == pytest.approx(0.0)
    assert result["focal_changed"] == 1
    assert result["focal_adopted_target"] == 1
    assert result["focal_left_target"] == 0
    assert result["sensor_target_share"] == pytest.approx(0.5)
    assert result["population_target_share"] == pytest.approx(0.0)
    assert result["sensor_target_error"] == pytest.approx(0.5)


def test_no_control_does_not_fabricate_controller_or_sensor_metrics():
    state = population_observables(["A", "B", "C"], ["A", "B", "C"], "C", "C")
    result = behavioral_transition_metrics(
        state,
        state,
        focal_opinion_before="A",
        focal_opinion_after="A",
        target="C",
        controller_action=None,
    )
    for field in (
        "u_advocate",
        "sensor_target_share",
        "population_target_share",
        "sensor_target_error",
        "sensor_target_abs_error",
    ):
        assert result[field] is None


def test_binary_action_entropy_has_exact_constant_and_balanced_values():
    assert binary_action_entropy_bits(["ADVOCATE_Z"] * 8) == pytest.approx(0.0)
    assert binary_action_entropy_bits(["ADVOCATE_Z", "NO_OP"] * 4) == pytest.approx(1.0)
    assert binary_action_entropy_bits([]) is None


@pytest.mark.parametrize(
    ("options", "target", "before", "expected"),
    [
        (("A", "B", "C"), "C", ("A", "C", "C"), 2),
        (("W", "X", "Y", "Z"), "Z", ("Z", "X", "Z", "Y"), 2),
    ],
)
def test_target_count_and_canonical_encoding_are_stable(options, target, before, expected):
    adapted = adapt_event(
        _event(before, before, options=options, target=target), episode_id="fixture"
    )
    assert adapted.N_t == tuple(before.count(option) for option in options)
    assert adapted.Y_t == tuple(before.count(option) for option in options)
    assert adapted.Z_t == expected
    assert adapted.Z_t1 == expected


def test_independent_sensing_fixture_is_zero_and_deterministic_measurement_is_positive():
    independent = []
    deterministic = []
    states = [("A", "A", "B", "C"), ("A", "B", "B", "C")]
    sensors = [{"A": 2, "B": 0, "C": 0}, {"A": 0, "B": 2, "C": 0}]
    index = 0
    for repetition in range(8):
        for state, sensor in itertools.product(states, sensors):
            index += 1
            independent.append(
                adapt_event(
                    _event(
                        state,
                        state,
                        action="NO_OP",
                        sensor=sensor,
                        episode=f"e-{repetition}",
                        index=index,
                    )
                )
            )
        for state in states:
            index += 1
            deterministic.append(
                adapt_event(
                    _event(
                        state,
                        state,
                        action="NO_OP",
                        sensor={option: state.count(option) for option in ("A", "B", "C")},
                        episode=f"d-{repetition}",
                        index=index,
                    )
                )
            )
    independent_rows, _ = information_analysis(
        independent, bootstrap_resamples=0, null_permutations=0
    )
    deterministic_rows, _ = information_analysis(
        deterministic, bootstrap_resamples=0, null_permutations=0
    )
    independent_sensing = next(row for row in independent_rows if row["statistic"] == "sensing_mi")
    deterministic_sensing = next(row for row in deterministic_rows if row["statistic"] == "sensing_mi")
    assert independent_sensing["unsmoothed"] == pytest.approx(0.0, abs=1e-12)
    assert deterministic_sensing["unsmoothed"] > 0.5


def test_controller_action_permutation_reduces_actuation_cmi():
    events = []
    before = ("A", "A", "B", "C")
    for episode_index in range(12):
        for interaction_index in range(12):
            action = "ADVOCATE_Z" if interaction_index % 2 == 0 else "NO_OP"
            after = ("C", "A", "B", "C") if action == "ADVOCATE_Z" else before
            events.append(
                adapt_event(
                    _event(
                        before,
                        after,
                        action=action,
                        episode=f"episode-{episode_index}",
                        index=interaction_index + 1,
                    )
                )
            )
    estimates, _ = information_analysis(
        events, bootstrap_resamples=20, null_permutations=100, seed=7
    )
    population = next(
        row for row in estimates if row["statistic"] == "population_actuation_cmi"
    )
    assert population["unsmoothed"] == pytest.approx(1.0)
    assert population["null_mean"] < 0.2
    assert population["controller_degenerate"] is False


def test_episode_bootstrap_draws_unique_episode_units_not_rows():
    draws = bootstrap_episode_ids(
        ["episode-a"] * 20 + ["episode-b"] * 2, resamples=10, seed=3
    )
    assert len(draws) == 10
    assert all(len(draw) == 2 for draw in draws)
    assert all(set(draw) <= {"episode-a", "episode-b"} for draw in draws)


def test_first_control_grid_resolves_exactly_four_matched_cells_and_provider():
    config = load_run_config_or_grid(
        "configs/runs/hidden_bench/hidden_bench_imitation_first_control_grid.yaml",
        environment={},
    )
    assert isinstance(config, GridSpec)
    assert len(config.cells) == 4
    assert {
        (
            cell.config.game.options["dynamics_mode"],
            cell.config.control.mechanism,
        )
        for cell in config.cells
    } == {
        ("reasoning", "none"),
        ("reasoning", "threshold_target"),
        ("classical", "none"),
        ("classical", "threshold_target"),
    }
    assert all(
        tuple(cell.config.game.options["initialization"]["initial_votes"])
        == ("East Town", "East Town", "North Hill", "West City")
        for cell in config.cells
    )
    provider = config.base.llm_provider
    assert provider.type == "university"
    assert provider.model == "gwdg/qwen3-30b-a3b-instruct-2507"
    assert provider.credentials_env == "POTSDAM_API_KEY"
    assert provider.base_url_env == "BASE_POTSDAM_LLM_URL"
    assert provider.timeout_seconds == 60
    assert provider.max_retries == 2
    assert provider.request_concurrency == 10
    assert provider.temperature == 0.0
    assert provider.max_output_tokens == 128
    assert provider.options["estimated_latency_seconds"] == 3.0
    assert config.base.execution.repetitions == 12
    assert config.base.game.options["task_id"] == "evacuation_north_hill"
    assert config.base.game.options["interactions"] == 20
    assert config.base.control.options["sensor_sample_size"] == 2


@pytest.mark.skipif(
    not (DEFAULT_CORPUS_ROOT / "canonical" / "tasks.json").exists(),
    reason="HiddenBench corpus is not present",
)
def test_classical_cells_from_first_grid_make_zero_provider_calls():
    spec = load_run_config_or_grid(
        "configs/runs/hidden_bench/hidden_bench_imitation_first_control_grid.yaml",
        environment={},
    )
    assert isinstance(spec, GridSpec)
    calls = 0

    def forbidden(_request):
        nonlocal calls
        calls += 1
        raise AssertionError("classical grid cell called the provider")

    for cell in spec.cells:
        if cell.config.game.options["dynamics_mode"] != "classical":
            continue
        config = replace(
            cell.config,
            game=replace(
                cell.config.game,
                horizon=3,
                options={**dict(cell.config.game.options), "interactions": 3},
            ),
            llm_provider=replace(cell.config.llm_provider, type="mock", model="fixture"),
        )
        provider = MockLLMProvider(config.llm_provider, response_factory=forbidden)
        result = asyncio.run(
            run_hidden_bench_imitation_game(
                create_game(config.game),
                config,
                provider,
                control=create_control(config.control),
            )
        )
        assert result.logical_decisions == 0
    assert calls == 0
