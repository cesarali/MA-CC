"""Scaled-population, q-slot, q_c-sensor, and truth-current acceptance tests."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import replace

import pytest

from mas_cc.config import load_run_config, load_run_config_or_grid
from mas_cc.control import NoneControl, create_control
from mas_cc.games import create_game
from mas_cc.games.hidden_bench.data import _scaled_task, assign
from mas_cc.games.hidden_bench.imitation import run_hidden_bench_imitation_game
from mas_cc.games.hidden_bench.imitation.analysis import (
    adapt_event,
    episode_summary,
    truth_current_analysis,
)
from mas_cc.games.hidden_bench.imitation.game import HiddenBenchImitationGame
from mas_cc.llm_runtime.providers import create_llm_provider
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider


CLASSICAL = "configs/runs/hidden_bench/hidden_bench_imitation_classical_control_10.yaml"
REASONING = "configs/runs/hidden_bench/hidden_bench_imitation_reasoning_control_10.yaml"


def _run(config, *, control=None, provider=None):
    return asyncio.run(
        run_hidden_bench_imitation_game(
            create_game(config.game),
            config,
            provider or create_llm_provider(config.llm_provider),
            control=control,
        )
    )


def _classical(*, q: int, q_c: int, horizon: int = 4, threshold: float = 1.0):
    config = load_run_config(CLASSICAL, environment={})
    n = config.game.population_size
    options = {
        **dict(config.game.options),
        "social_group_size": q,
        "initialization": {
            "mode": "explicit",
            "initial_votes": ["West City", *(["East Town"] * (n - 1))],
        },
    }
    return replace(
        config,
        game=replace(config.game, horizon=horizon, options=options),
        control=replace(
            config.control,
            mechanism="threshold_target",
            options={
                "target": "correct",
                "sensor_sample_size": q_c,
                "threshold": threshold,
                "template_version": 2,
            },
        ),
    )


def test_imitation_n_grid_is_exactly_the_requested_three_by_three_phase_diagram():
    spec = load_run_config_or_grid(
        "configs/runs/hidden_bench/hidden_bench_imitation_N_q_qc_phase_grid.yaml",
        environment={},
    )
    assert spec.base.experiment.name == "imitation_N"
    assert spec.base.game.options["task_id"] == "evacuation_west_city"
    assert "allowed_values" not in spec.base.prompt.response_contract
    assert len(spec.cells) == 9
    assert {
        (
            cell.config.game.options["social_group_size"],
            cell.config.control.options["sensor_sample_size"],
        )
        for cell in spec.cells
    } == {(q, q_c) for q in (1, 2, 4) for q_c in (2, 8, 32)}
    assert all(cell.config.game.population_size == 32 for cell in spec.cells)
    assert all(cell.config.game.horizon == 10 for cell in spec.cells)
    assert all(
        HiddenBenchImitationGame().rules(cell.config.game).horizon == 320
        for cell in spec.cells
    )
    assert all(
        cell.config.game.options["assignment_scheme"] == "paraphrased_replication"
        for cell in spec.cells
    )
    assert spec.base.storage.artifact_profile == "results_only"
    assert spec.base.analysis.options["per_cell_reports"] is True
    assert spec.base.analysis.estimators[-2:] == (
        "truth_current",
        "truth_current_fano",
    )


def test_random_incorrect_grid_matches_the_phase_diagram_without_a_hard_coded_target():
    spec = load_run_config_or_grid(
        "configs/runs/hidden_bench/"
        "hidden_bench_imitation_N_q_qc_phase_grid_random_incorrect.yaml",
        environment={},
    )
    assert spec.base.experiment.name == "imitation_N_random_incorrect"
    assert spec.base.control.options["target"] == "random_incorrect"
    assert len(spec.cells) == 9
    assert {
        (
            cell.config.game.options["social_group_size"],
            cell.config.control.options["sensor_sample_size"],
        )
        for cell in spec.cells
    } == {(q, q_c) for q in (1, 2, 4) for q_c in (2, 8, 24)}
    assert all(cell.config.game.population_size == 24 for cell in spec.cells)
    assert all(cell.config.game.horizon == 10 for cell in spec.cells)
    assert all(
        HiddenBenchImitationGame().rules(cell.config.game).horizon == 240
        for cell in spec.cells
    )


@pytest.mark.parametrize("value", [0, 4, True, 1.5])
def test_social_group_size_is_validated_against_population(value):
    config = _classical(q=1, q_c=2)
    config = replace(
        config,
        game=replace(
            config.game,
            options={**dict(config.game.options), "social_group_size": value},
        ),
    )
    with pytest.raises(ValueError, match="social_group_size"):
        HiddenBenchImitationGame().rules(config.game)


def test_advocacy_replaces_exactly_one_of_q_distinct_social_slots():
    config = _classical(q=3, q_c=4, horizon=5, threshold=1.0)
    result = _run(config, control=create_control(config.control))
    assert len(result.initial_state.agents) == 4
    assert len(result.interactions) == 5 * 4
    advocacy_events = 0
    for interaction in result.interactions:
        event = interaction.transition.event
        assert len(event["social_peer_ids"]) == 3
        assert len(set(event["social_peer_ids"])) == 3
        assert event["focal_agent_id"] not in event["social_peer_ids"]
        assert len(event["influence_slots"]) == 3
        if event["controller_action"] == "ADVOCATE_Z":
            advocacy_events += 1
            assert [slot["kind"] for slot in event["influence_slots"]].count("controller") == 1
            assert [slot["kind"] for slot in event["influence_slots"]].count("peer") == 2
            assert event["replaced_peer_id"] in event["social_peer_ids"]
            assert event["social_peer_ids"][event["replaced_peer_slot"]] == event["replaced_peer_id"]
        assert len(event["controller_sensor_ids"]) == 4
        assert len(set(event["controller_sensor_ids"])) == 4
        assert event["controller_sensor_includes_focal"] is True
        assert set(event["controller_sensor_social_overlap_ids"]) == set(event["social_peer_ids"])
        before, after = event["population_state_before"], event["population_state_after"]
        assert sum(left != right for left, right in zip(before, after)) == 1
    assert advocacy_events > 0


def test_noop_keeps_all_q_ordinary_peer_slots():
    config = _classical(q=3, q_c=2, horizon=3, threshold=0.0)
    result = _run(config, control=create_control(config.control))
    for interaction in result.interactions:
        event = interaction.transition.event
        assert event["controller_action"] == "NO_OP"
        assert event["replaced_peer_id"] is None
        assert event["replaced_peer_slot"] is None
        assert [slot["kind"] for slot in event["influence_slots"]] == ["peer"] * 3
        assert len(interaction.participants) == 4


def test_explicit_q1_replays_the_default_legacy_schedule_and_transitions():
    base = load_run_config(CLASSICAL, environment={})
    base = replace(base, game=replace(base.game, horizon=12))
    explicit = replace(
        base,
        game=replace(
            base.game,
            options={**dict(base.game.options), "social_group_size": 1},
        ),
    )
    first = _run(base, control=create_control(base.control))
    second = _run(explicit, control=create_control(explicit.control))
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )
    for item in second.interactions:
        event = item.transition.event
        assert len(event["social_peer_ids"]) == 1
        assert len(event["controller_sensor_ids"]) == 2
        if event["controller_action"] == "ADVOCATE_Z":
            assert event["replaced_peer_slot"] == 0
            assert event["peer_agent_id"] is None


def test_reasoning_q3_presents_three_ordered_inputs_and_updates_only_focal():
    config = load_run_config(REASONING, environment={})
    config = replace(
        config,
        game=replace(
            config.game,
            horizon=1,
            options={**dict(config.game.options), "social_group_size": 3},
        ),
        control=replace(
            config.control,
            mechanism="threshold_target",
            options={
                "target": "correct",
                "sensor_sample_size": 4,
                "threshold": 1.0,
                "template_version": 2,
            },
        ),
    )
    rendered_requests: list[str] = []

    def respond(request):
        rendered = "\n".join(message.content for message in request.messages)
        rendered_requests.append(rendered)
        if "following JSON format" in rendered:
            return '{"vote": "East Town", "rationale": "mock"}'
        return "A concise peer message."

    result = _run(
        config,
        control=create_control(config.control),
        provider=MockLLMProvider(config.llm_provider, response_factory=respond),
    )
    event = result.interactions[0].transition.event
    assert len(event["influence_slots"]) == 3
    assert [slot["kind"] for slot in event["influence_slots"]].count("controller") == 1
    assert all(slot["message"] for slot in event["influence_slots"])
    update_prompt = rendered_requests[-1]
    assert all(f"slot {index}" in update_prompt for index in (1, 2, 3))
    assert len(result.interactions[0].decisions) == 5  # two dyads x two messages + one update
    before, after = event["population_state_before"], event["population_state_after"]
    assert sum(left != right for left, right in zip(before, after)) <= 1
    assert len(result.final_state.agents) == 4  # the controller is never population agent N+1


def test_paraphrased_population_retains_frozen_variant_provenance():
    hidden = [f"source evidence {index}" for index in range(4)]
    agents = [
        {
            "agent_id": index,
            "evidence_type": index % 4,
            "variant_id": f"variant-{index:02d}",
            "private_information": [f"validated paraphrase {index}"],
            "source_hidden_indices": [index % 4],
            "source_text": hidden[index % 4],
            "transformation": "validated_paraphrase",
        }
        for index in range(32)
    ]
    task, allocation = _scaled_task(
        {
            "task_id": 2,
            "name": "fixture",
            "source_description": "fixture",
            "scenario_description": "fixture",
            "shared_information": ["shared fixture"],
            "source_hidden_information": hidden,
            "possible_answers": ["A", "B", "C"],
            "correct_answer": "C",
            "population_instruction": "There are 32 agents.",
            "population": {
                "num_agents": 32,
                "source_base_agent_count": 4,
                "method": "paraphrased_replication",
                "diagnostics": {"variant_reuse_allowed": False},
            },
            "agents": agents,
        },
        "paraphrased_replication",
        32,
    )
    assignment = assign(
        task,
        32,
        "paraphrased_replication",
        __import__("random").Random(0),
        prebuilt=allocation,
    )
    counts = {}
    variants = set()
    for info in assignment.values():
        evidence_type = info.evidence_types[0]
        counts[evidence_type] = counts.get(evidence_type, 0) + 1
        assert info.transformation == "validated_paraphrase"
        assert info.provenance["source_hidden_indices"] == [evidence_type]
        assert info.provenance["source_text"] == task.hidden_information[evidence_type]
        assert all(answer not in info.private for answer in task.possible_answers)
        variants.add(info.provenance["variant_id"])
    assert len(variants) == 32
    assert max(counts.values()) - min(counts.values()) <= 1


def _trajectory_event(
    *, episode: str, index: int, truth_before: int, toward: bool
):
    options = ["truth", "wrong"]
    before = ["truth"] * truth_before + ["wrong"] * (4 - truth_before)
    truth_after = truth_before + int(toward)
    after = ["truth"] * truth_after + ["wrong"] * (4 - truth_after)
    return adapt_event(
        {
            "episode_id": episode,
            "interaction_index": index,
            "N": 4,
            "possible_answers": options,
            "correct_answer": "truth",
            "population_state_before": before,
            "population_state_after": after,
            "occupation_counts_before": {
                "truth": truth_before,
                "wrong": 4 - truth_before,
            },
            "occupation_counts_after": {
                "truth": truth_after,
                "wrong": 4 - truth_after,
            },
            "focal_opinion_before": "wrong",
            "focal_opinion_after": "truth" if toward else "wrong",
            "controller_action": None,
            "dynamics_mode": "reasoning",
        },
        episode_id=episode,
    )


def _current_episodes(currents):
    events = []
    for episode_index, current in enumerate(currents):
        truth_count = 0
        for interaction_index in range(1, 3):
            toward = interaction_index <= current
            events.append(
                _trajectory_event(
                    episode=f"episode-{episode_index}",
                    index=interaction_index,
                    truth_before=truth_count,
                    toward=toward,
                )
            )
            truth_count += int(toward)
    return events


@pytest.mark.parametrize(
    ("before_focal", "after_focal", "expected"),
    [
        ("wrong-a", "truth", 1),
        ("truth", "wrong-a", -1),
        ("wrong-a", "wrong-b", 0),
        ("truth", "truth", 0),
    ],
)
def test_truth_current_step_cases(before_focal, after_focal, expected):
    options = ["truth", "wrong-a", "wrong-b"]
    before = [before_focal, "wrong-a"]
    after = [after_focal, "wrong-a"]
    event = adapt_event(
        {
            "episode_id": "toy",
            "interaction_index": 1,
            "N": 2,
            "possible_answers": options,
            "correct_answer": "truth",
            "population_state_before": before,
            "population_state_after": after,
            "occupation_counts_before": dict(__import__("collections").Counter(before)),
            "occupation_counts_after": dict(__import__("collections").Counter(after)),
            "focal_opinion_before": before_focal,
            "focal_opinion_after": after_focal,
            "controller_action": None,
        }
    )
    assert episode_summary([event])["truth_current"] == expected


def test_truth_current_telescopes_and_fano_uses_sample_variance():
    events = _current_episodes([1, 2, 1, 0])
    first_episode = episode_summary(events[:2])
    assert first_episode["truth_current"] == 1
    assert first_episode["truth_switches_toward"] == 1
    assert first_episode["truth_switches_away"] == 0
    assert first_episode["truth_current"] == events[1].Mtruth_t1 - events[0].Mtruth_t

    row = truth_current_analysis(events, bootstrap_resamples=20, seed=7)
    assert row["truth_current_mean"] == pytest.approx(1.0)
    assert row["truth_current_variance"] == pytest.approx(2 / 3)
    assert row["truth_current_fano"] == pytest.approx(1.5)
    assert row["episodes"] == 4
    assert row["fixed_horizon"] is True
    assert row["null_model"] is None


def test_controller_exposure_diagnostics_distinguish_decisions_and_advocacy():
    config = _classical(q=2, q_c=2, horizon=6, threshold=1.0)
    result = _run(config, control=create_control(config.control))
    events = [adapt_event(item.transition.event) for item in result.interactions]
    row = episode_summary(events)
    assert row["controller_decision_count"] == 6 * 4
    assert row["controller_advocacy_count"] > 0
    assert row["controller_advocacy_count"] + row["controller_noop_count"] == 6 * 4
    assert row["controller_decisions_per_agent"] == pytest.approx(6.0)
    assert row["controller_advocacies_per_agent"] == pytest.approx(
        row["controller_advocacy_count"] / 4
    )
    assert 1 <= row["unique_focal_agents_exposed_to_controller"] <= 4
    assert row["fraction_population_ever_exposed_to_controller"] == pytest.approx(
        row["unique_focal_agents_exposed_to_controller"] / 4
    )


def test_truth_current_zero_dispersion_edges_are_not_clipped_to_zero():
    positive = truth_current_analysis(_current_episodes([1, 1]), bootstrap_resamples=0)
    zero = truth_current_analysis(_current_episodes([0, 0]), bootstrap_resamples=0)
    assert math.isinf(positive["truth_current_fano"])
    assert positive["zero_dispersion"] is True
    assert math.isnan(zero["truth_current_fano"])
    assert zero["truth_current_fano_undefined_reason"] == "zero_mean_and_zero_dispersion"
