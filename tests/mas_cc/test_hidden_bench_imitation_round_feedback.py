"""Scientific invariants for the slow-clock budgeted feedback game."""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from mas_cc.config import GridSpec, load_run_config, load_run_config_or_grid
from mas_cc.control import Control, RoundControlSignal, create_control
from mas_cc.control.registry import create_default_control_registry
from mas_cc.games import create_game
from mas_cc.games.hidden_bench.data import DEFAULT_CORPUS_ROOT
from mas_cc.games.hidden_bench.imitation.controller import ADVOCATE_TARGET, NO_OP
from mas_cc.games.hidden_bench.imitation_round_feedback import (
    HiddenBenchImitationRoundFeedbackGame,
    run_hidden_bench_imitation_round_feedback_game,
)
from mas_cc.games.hidden_bench.imitation_round_feedback.runtime import (
    sample_controlled_positions,
)
from mas_cc.games.registry import create_default_game_registry
from mas_cc.llm_runtime.providers import create_llm_provider
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider

pytestmark = pytest.mark.skipif(
    not (DEFAULT_CORPUS_ROOT / "canonical" / "tasks.json").exists(),
    reason="HiddenBench corpus is not present",
)

CLASSICAL = (
    "configs/runs/hidden_bench/"
    "hidden_bench_imitation_round_feedback_classical_smoke.yaml"
)
REASONING = (
    "configs/runs/hidden_bench/"
    "hidden_bench_imitation_round_feedback_reasoning_smoke.yaml"
)
QWEN_GRID_A = (
    "configs/runs/imitation_round_feedback/"
    "hidden_bench_imitation_round_feedback_qwen_first_llm_pilot_grid_A.yaml"
)
QWEN_PILOT_GRIDS = (
    QWEN_GRID_A,
    *(
        "configs/runs/imitation_round_feedback/"
        f"hidden_bench_imitation_round_feedback_qwen_first_llm_pilot_grid_{name}.yaml"
        for name in ("B", "C", "D")
    ),
)
ROUND_FEEDBACK_CONFIG_DIR = Path("configs/runs/imitation_round_feedback")
ROUND_FEEDBACK_RUN_CONFIGS = tuple(sorted(ROUND_FEEDBACK_CONFIG_DIR.glob("*.yaml")))


class _CountingControl(Control):
    sensor_sample_size = 2
    intervention_budget = 2

    def __init__(self, action: str = ADVOCATE_TARGET) -> None:
        self.action = action
        self.calls = 0
        self.views = []

    def override(self, **_):
        return None

    def round_signal(self, *, round_index, state, rng):
        self.calls += 1
        self.views.append(state)
        sampled = rng.sample(list(state.agents), self.sensor_sample_size)
        target = str(state.data["task"]["correct_answer"])
        opinions = [str(agent.attributes["committed_action"]) for agent in sampled]
        return RoundControlSignal(
            action=self.action,
            target=target,
            message=(f"I currently favor {target}." if self.action == ADVOCATE_TARGET else None),
            observation={
                "sampled_agent_ids": [str(agent.agent_id) for agent in sampled],
                "sampled_opinions": opinions,
                "sampled_opinion_counts": dict(Counter(opinions)),
                "sample_size": self.sensor_sample_size,
            },
            metadata={
                "policy": "test",
                "threshold": 0.5,
                "beta": 1.0,
                "advocacy_probability": 1.0 if self.action == ADVOCATE_TARGET else 0.0,
            },
        )


class _Observer:
    def __init__(self) -> None:
        self.micro = []
        self.rounds = []

    def record_trajectory(self, *, record):
        self.micro.append(record)

    def record_round_trajectory(self, *, record):
        self.rounds.append(record)


def _config(path: str, *, rounds: int | None = None, q: int | None = None):
    config = load_run_config(path, environment={})
    options = dict(config.game.options)
    if rounds is not None:
        options["rounds"] = rounds
    if q is not None:
        options["social_group_size"] = q
    return replace(
        config,
        game=replace(
            config.game,
            horizon=options["rounds"],
            options=options,
        ),
    )


def _run(config, *, control=None, provider=None, observer=None):
    return asyncio.run(
        run_hidden_bench_imitation_round_feedback_game(
            create_game(config.game),
            config,
            provider or create_llm_provider(config.llm_provider),
            control=control,
            observer=observer,
        )
    )


def test_game_control_and_shipped_configs_are_registered():
    assert "hidden_bench_imitation_round_feedback" in create_default_game_registry().names()
    assert "round_soft_target_budgeted" in create_default_control_registry().names()
    for path in (CLASSICAL, REASONING):
        config = _config(path)
        assert isinstance(create_game(config.game), HiddenBenchImitationRoundFeedbackGame)
        assert create_control(config.control) is not None


def test_qwen_first_pilot_grid_a_resolves_the_requested_four_cells():
    grid = load_run_config_or_grid(QWEN_GRID_A, environment={})

    assert isinstance(grid, GridSpec)
    assert len(grid.cells) == 4
    assert grid.base.game.population_size == 24
    assert grid.base.game.horizon == 10
    assert grid.base.execution.repetitions == 30
    assert grid.base.logging.comet is True
    assert grid.base.analysis.comet_export is True
    assert grid.base.analysis.options["per_cell_reports"] is True
    assert grid.base.observability.comet.cell_reporting == "master"
    assert grid.base.observability.comet.sweep_experiment is True
    assert "allowed_values" not in grid.base.prompt.response_contract
    assert grid.base.game.options["initialization"] == {"mode": "uniform_random"}
    assert grid.base.game.options["population_preparation"] == {
        "auto_build_missing": True,
        "paraphrase_annotations": "data/hidden_bench/annotations/paraphrases.json",
    }
    rules = create_game(grid.base.game).rules(grid.base.game)
    assert rules.auto_prepare_paraphrases is True
    assert rules.paraphrase_annotations == (
        "data/hidden_bench/annotations/paraphrases.json"
    )
    assert [
        (
            cell.config.game.options["social_group_size"],
            cell.config.control.options["intervention_budget"],
        )
        for cell in grid.cells
    ] == [
        (1, 0),
        (1, 6),
        (2, 0),
        (2, 6),
    ]


def test_qwen_first_pilot_suite_resolves_exactly_grids_a_through_d():
    expected = (
        {(q, 2, b, "correct") for q in (1, 2) for b in (0, 6)},
        {(q, 2, b, "correct") for q in (1, 2) for b in (12, 18)},
        {(2, qc, b, "correct") for qc in (2, 16, 24) for b in (6, 12)},
        {
            (q, 12, b, target)
            for q in (1, 2)
            for target in ("correct", "West City")
            for b in (6, 12)
        },
    )
    grids = [load_run_config_or_grid(path, environment={}) for path in QWEN_PILOT_GRIDS]

    assert all(isinstance(grid, GridSpec) for grid in grids)
    assert [len(grid.cells) for grid in grids] == [4, 4, 6, 8]
    assert sum(len(grid.cells) for grid in grids) == 22
    assert sum(len(grid.cells) * grid.base.execution.repetitions for grid in grids) == 660
    for name, grid, requested in zip("ABCD", grids, expected, strict=True):
        assert grid.base.experiment.metadata["pilot_grid"] == name
        assert grid.base.game.population_size == 24
        assert grid.base.game.options["rounds"] == 10
        assert grid.base.game.options["task_id"] == "evacuation_north_hill"
        assert grid.base.execution.seed == 20260813
        assert grid.base.execution.repetitions == 30
        assert grid.base.logging.comet is True
        assert grid.base.analysis.comet_export is True
        assert grid.base.analysis.options["per_cell_reports"] is True
        assert grid.base.observability.comet.writer == "master_only"
        assert grid.base.observability.comet.sweep_experiment is True
        assert grid.base.observability.comet.cell_reporting == "master"
        assert grid.base.observability.comet.metric_plots is True
        assert "round_population_actuation_cmi" not in grid.base.analysis.estimators
        assert {
            "round_target_actuation_cmi",
            "round_truth_actuation_cmi",
        } <= set(grid.base.analysis.estimators)
        assert {
            (
                cell.config.game.options["social_group_size"],
                cell.config.control.options["sensor_sample_size"],
                cell.config.control.options["intervention_budget"],
                cell.config.control.options["target"],
            )
            for cell in grid.cells
        } == requested


@pytest.mark.parametrize(
    "path", ROUND_FEEDBACK_RUN_CONFIGS, ids=lambda path: path.stem
)
def test_every_round_feedback_run_has_live_console_and_completed_cell_reporting(path):
    grid = load_run_config_or_grid(path, environment={})

    assert isinstance(grid, GridSpec)
    assert grid.base.logging.console is True
    assert grid.base.logging.comet is True
    assert grid.base.analysis.enabled is True
    assert grid.base.analysis.comet_export is True
    assert grid.base.analysis.options["per_cell_reports"] is True
    assert grid.base.observability.comet.writer == "master_only"
    assert grid.base.observability.comet.sweep_experiment is True
    assert grid.base.observability.comet.cell_reporting == "master"
    assert grid.base.observability.comet.metric_plots is True

    instructions = path.read_text(encoding="utf-8")
    assert f"--config {path.as_posix()}" in instructions
    assert "conda run -n MA-CC --no-capture-output" in instructions
    assert "2>&1 | tee logs/" in instructions
    assert "configured analysis ready" in instructions


def test_qwen_grid_uniform_initialization_is_planned_as_provider_free():
    grid = load_run_config_or_grid(QWEN_GRID_A, environment={})
    assert isinstance(grid, GridSpec)
    reasoning = grid.cells[0].config

    plan = create_game(reasoning.game).call_plan(reasoning.game)
    stages = {stage.name: stage for stage in plan.decision_stages}

    assert stages["local_initialization"].requests_per_interaction == 0
    # Ten rounds x 24 focal updates: 480 private messages + 240 focal votes.
    assert plan.provider_requests.lower == 720


def test_controller_is_called_once_per_round_and_micro_rows_hold_one_action():
    config = _config(CLASSICAL, rounds=3)
    control = _CountingControl()
    observer = _Observer()
    result = _run(config, control=control, observer=observer)

    assert control.calls == 3
    assert len(result.rounds) == len(observer.rounds) == 3
    assert len(result.interactions) == len(observer.micro) == 3 * config.game.population_size
    for round_index in range(3):
        round_row = result.rounds[round_index].to_dict()
        assert len(round_row["sensor_agent_ids"]) == 2
        assert len(set(round_row["sensor_agent_ids"])) == 2
        events = [
            item.transition.event
            for item in result.interactions
            if item.transition.event["round_index"] == round_index
        ]
        assert len(events) == config.game.population_size
        assert {event["round_controller_action"] for event in events} == {
            ADVOCATE_TARGET
        }


def test_controller_view_contains_only_opinions_and_public_target_metadata():
    control = _CountingControl()
    _run(_config(CLASSICAL, rounds=1), control=control)
    view = control.views[0]
    assert set(view.data) == {"seed", "task"}
    assert set(view.data["task"]) == {"task_id", "possible_answers", "correct_answer"}
    assert all(not agent.memory for agent in view.agents)
    assert all(set(agent.attributes) == {"committed_action"} for agent in view.agents)


def test_no_control_does_not_fabricate_a_round_action_or_sensor():
    config = _config(CLASSICAL, rounds=2)
    config = replace(
        config,
        control=replace(config.control, mechanism="none", options={}),
    )
    result = _run(config, control=create_control(config.control))
    for record in result.rounds:
        row = record.to_dict()
        assert row["controller_enabled"] is False
        assert row["controller_action"] is None
        assert row["sensor_count_vector"] is None
        assert row["controlled_positions"] == []
    assert all(
        interaction.transition.event["round_controller_action"] is None
        for interaction in result.interactions
    )


@pytest.mark.parametrize(
    ("action", "expected"),
    [(NO_OP, 0), (ADVOCATE_TARGET, 2)],
)
def test_budget_schedule_has_exact_unique_preallocated_positions(action, expected):
    result = _run(_config(CLASSICAL, rounds=2), control=_CountingControl(action))
    for round_record in result.rounds:
        row = round_record.to_dict()
        positions = row["controlled_positions"]
        assert len(positions) == len(set(positions)) == expected
        events = [
            item.transition.event
            for item in result.interactions
            if item.transition.event["round_index"] == row["round_index"]
        ]
        observed = [
            event["within_round_index"] for event in events if event["controlled_slot"]
        ]
        assert observed == positions
        assert {event["controlled_positions_hash_or_id"] for event in events} == {
            row["controlled_positions_hash_or_id"]
        }


def test_schedule_sampling_is_seeded_and_uses_uniform_sample_without_replacement():
    first = sample_controlled_positions(8, 3, random.Random(42))
    second = sample_controlled_positions(8, 3, random.Random(42))
    assert first == second
    assert len(first) == len(set(first)) == 3

    class _RecordingRng:
        def __init__(self):
            self.population = None
            self.k = None

        def sample(self, population, k):
            self.population = tuple(population)
            self.k = k
            return [5, 1, 3]

    rng = _RecordingRng()
    assert sample_controlled_positions(8, 3, rng) == (1, 3, 5)
    assert rng.population == tuple(range(8))
    assert rng.k == 3


@pytest.mark.parametrize("q", [1, 2])
def test_social_slots_replace_exactly_one_peer_and_only_focal_vote_may_change(q):
    result = _run(_config(CLASSICAL, rounds=1, q=q), control=_CountingControl())
    for interaction in result.interactions:
        event = interaction.transition.event
        assert len(event["sampled_peer_agent_ids"]) == q
        assert len(event["influence_slots"]) == q
        ordinary = [slot for slot in event["influence_slots"] if slot["kind"] == "peer"]
        controller = [
            slot for slot in event["influence_slots"] if slot["kind"] == "controller"
        ]
        expected_controller = int(event["controlled_slot"])
        assert len(controller) == expected_controller
        assert len(ordinary) == q - expected_controller
        before = event["population_state_before"]
        after = event["population_state_after"]
        changed = [index for index, pair in enumerate(zip(before, after)) if pair[0] != pair[1]]
        assert len(changed) <= 1
        if changed:
            focal_id = event["focal_agent_id"]
            assert changed == [
                next(
                    index
                    for index, agent in enumerate(interaction.transition.next_state.agents)
                    if str(agent.agent_id) == focal_id
                )
            ]


def test_round_boundaries_are_contiguous():
    result = _run(_config(CLASSICAL, rounds=3), control=_CountingControl())
    rows = [record.to_dict() for record in result.rounds]
    for index, row in enumerate(rows):
        events = [
            item.transition.event
            for item in result.interactions
            if item.transition.event["round_index"] == index
        ]
        assert row["occupation_counts_before"] == [
            events[0]["occupation_counts_before"][option]
            for option in events[0]["possible_answers"]
        ]
        assert row["occupation_counts_after"] == [
            events[-1]["occupation_counts_after"][option]
            for option in events[-1]["possible_answers"]
        ]
        if index:
            assert row["population_state_before"] == rows[index - 1]["population_state_after"]


def test_classical_mode_is_provider_free_and_consumes_q_peer_context():
    calls = 0

    def forbidden(_):
        nonlocal calls
        calls += 1
        raise AssertionError("classical mode called provider")

    config = _config(CLASSICAL, rounds=2, q=2)
    provider = MockLLMProvider(config.llm_provider, response_factory=forbidden)
    result = _run(config, control=_CountingControl(), provider=provider)
    assert calls == result.logical_decisions == 0
    assert all(
        item.transition.event["classical_social_context_consumed"] is True
        for item in result.interactions
    )
    assert {
        item.transition.event["classical_kernel_rule"]
        for item in result.interactions
    } == {"strict_unanimity_q_voter"}
    assert {record.to_dict()["classical_kernel_rule"] for record in result.rounds} == {
        "strict_unanimity_q_voter"
    }


def test_reasoning_mode_uses_the_same_exact_round_schedule_semantics():
    result = _run(_config(REASONING, rounds=1), control=_CountingControl())
    row = result.rounds[0].to_dict()
    events = [item.transition.event for item in result.interactions]
    assert len(events) == row["N"]
    assert sum(bool(event["controlled_slot"]) for event in events) == 2
    assert all(len(event["influence_slots"]) == 2 for event in events)
