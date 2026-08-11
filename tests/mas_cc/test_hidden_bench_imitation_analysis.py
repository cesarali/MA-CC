"""Focused fixtures for the first HiddenBench imitation diagnostics."""

from __future__ import annotations

import itertools
import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.control import create_control
from mas_cc.games import create_game
from mas_cc.games.hidden_bench.data import DEFAULT_CORPUS_ROOT
from mas_cc.games.hidden_bench.imitation import run_hidden_bench_imitation_game
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider
from mas_cc.games.hidden_bench.imitation.analysis import (
    INFORMATION_STATISTICS,
    _export_information_estimates_to_comet,
    _write_information_estimates_markdown,
    adapt_event,
    binary_action_entropy_bits,
    bootstrap_episode_ids,
    information_analysis,
    information_comet_metrics,
    plot_information_estimates,
)
from mas_cc.games.hidden_bench.imitation.metrics import (
    METRICS,
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
    assert config.base.game.horizon == 20
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


def test_standalone_behavior_configs_export_their_selected_monitoring_metrics():
    event_metrics = {
        "delta_m_ctrl",
        "delta_m_truth",
        "delta_m_order",
        "delta_H_vote",
        "focal_changed",
        "focal_adopted_target",
        "focal_left_target",
        "u_advocate",
        "sensor_target_share",
        "population_target_share",
        "sensor_target_error",
        "sensor_target_abs_error",
    }
    paths = (
        "configs/runs/hidden_bench/hidden_bench_imitation_reasoning_control_10.yaml",
        "configs/runs/hidden_bench/hidden_bench_imitation_classical_control_10.yaml",
    )
    reasoning, classical = [load_run_config_or_grid(path, environment={}) for path in paths]
    assert reasoning.game.options["dynamics_mode"] == "reasoning"
    assert classical.game.options["dynamics_mode"] == "classical"
    assert reasoning.game.options["initialization"] == classical.game.options["initialization"]
    assert reasoning.control == classical.control
    assert reasoning.llm_provider == classical.llm_provider
    state_metrics = {"m_ctrl", "m_truth", "m_order"}
    assert event_metrics | state_metrics <= {metric.name for metric in METRICS}
    assert reasoning.metrics.comet_export_names() == ()
    assert set(reasoning.aggregation.cell_metrics) == state_metrics | {
        "population_action_share_per_option",
        "dominant_action_share",
    }
    # A single-cell run publishes everything onto one experiment: no sibling
    # `<run>/run` or `<run>/analysis` to hunt through for the aggregate plots.
    assert reasoning.observability.comet.cell_experiments is False
    assert reasoning.observability.comet.master_aggregates is True
    assert reasoning.observability.comet.metric_plots is True
    assert set(classical.metrics.comet_export_names()) == event_metrics
    assert reasoning.analysis.enabled is True
    assert classical.analysis.enabled is True
    assert reasoning.analysis.estimators == INFORMATION_STATISTICS
    # The classical comparator still runs only the count-vector channels; the
    # reasoning cell additionally projects them onto the order parameters.
    assert classical.analysis.estimators == (
        "sensing_mi",
        "population_actuation_cmi",
        "target_actuation_cmi",
        "focal_actuation_cmi",
    )
    assert reasoning.analysis.options["bootstrap_resamples"] == 1000
    assert reasoning.analysis.options["null_permutations"] == 1000
    assert (
        create_game(reasoning.game).call_plan(reasoning.game).provider_requests.maximum
        == 3 * reasoning.game.horizon
    )
    assert create_game(classical.game).call_plan(classical.game).provider_requests.maximum == 0
    assert reasoning.analysis.comet_export is True
    assert classical.analysis.comet_export is False


def _information_rows_fixture():
    events = []
    before = ("A", "A", "B", "C")
    for episode_index in range(4):
        for interaction_index in range(4):
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
    estimates, _ = information_analysis(events, bootstrap_resamples=5, null_permutations=5, seed=1)
    return [
        {
            "cell_id": "run",
            "scientific_cell": None,
            "dynamics_mode": "classical",
            "control_mechanism": "threshold_target",
            **row,
        }
        for row in estimates
    ]


def test_information_estimates_markdown_reports_every_statistic_and_its_meaning(tmp_path):
    destination = tmp_path / "information_estimates.md"

    _write_information_estimates_markdown(_information_rows_fixture(), destination)

    text = destination.read_text(encoding="utf-8")
    for name in INFORMATION_STATISTICS:
        assert f"`{name}`" in text
    assert "Cell `run`" in text
    assert "Bootstrap CI" in text
    assert "Null mean" in text
    assert "|" in text  # rendered as markdown tables, not a raw CSV dump


def test_information_estimates_markdown_handles_missing_values(tmp_path):
    destination = tmp_path / "information_estimates.md"
    rows = [
        {
            "cell_id": "cell-0000",
            "scientific_cell": "A",
            "dynamics_mode": "reasoning",
            "control_mechanism": "none",
            "statistic": "sensing_mi",
            "estimate": float("nan"),
            "unsmoothed": float("nan"),
            "jeffreys": float("nan"),
            "miller_madow": float("nan"),
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
            "null_mean": float("nan"),
            "null_ci_low": float("nan"),
            "null_ci_high": float("nan"),
            "scientifically_interpretable": False,
            "n_episodes": 0,
            "n_events": 0,
            "occupied_conditioning_states": 0,
            "min_events_per_conditioning_state": None,
            "median_events_per_conditioning_state": None,
            "max_events_per_conditioning_state": None,
            "fraction_events_singleton_conditioning_states": None,
            "sparse_conditioning_table": False,
            "controller_degenerate": None,
        }
    ]

    _write_information_estimates_markdown(rows, destination)

    text = destination.read_text(encoding="utf-8")
    assert "—" in text
    assert "nan" not in text.lower()


class _FakeSink:
    """Records the calls an export makes, in order."""

    def __init__(self, calls, enabled, *, project_name, run_name):
        self.calls = calls
        self.status = "active"
        self.url = "https://comet.invalid/analysis"
        calls.append(("init", enabled, project_name, run_name))

    def add_tags(self, tags):
        self.calls.append(("add_tags", tuple(tags)))

    def log_metrics(self, metrics, step):
        self.calls.append(("log_metrics", dict(metrics), step))

    def log_image(self, path, *, name, step):
        self.calls.append(("log_image", Path(path).name, name, step))

    def log_asset(self, path, *, name):
        self.calls.append(("log_asset", name))

    def close(self):
        self.calls.append(("close",))
        return {"status": self.status}


def _patch_sink(monkeypatch, calls):
    monkeypatch.setattr(
        "mas_cc.observability.recorder.CometMetricSink",
        lambda enabled, **kwargs: _FakeSink(calls, enabled, **kwargs),
    )


def test_information_estimates_are_not_exported_to_comet_by_default(tmp_path, monkeypatch):
    calls = []
    _patch_sink(monkeypatch, calls)

    summary = _export_information_estimates_to_comet(
        _information_rows_fixture(), (), (),
        enabled=False, project_name="mas-cc", run_name="run/analysis",
    )

    assert calls == []
    assert summary["status"] == "disabled"


def test_the_export_sends_the_numbers_not_only_the_report(tmp_path, monkeypatch):
    """The report was once the whole export.

    That made every quantity this analysis exists to produce downloadable but
    never plottable, never comparable across runs, and invisible on the
    dashboard - which reads exactly like "the analysis did not run".
    """

    calls = []
    _patch_sink(monkeypatch, calls)
    report = tmp_path / "information_estimates.md"
    report.write_text("# report\n", encoding="utf-8")
    figure = tmp_path / "information_estimates_run.png"
    figure.write_bytes(b"\x89PNG")
    rows = _information_rows_fixture()

    summary = _export_information_estimates_to_comet(
        rows, (report,), (figure,),
        enabled=True, project_name="mas-cc", run_name="run/analysis",
    )

    kinds = [call[0] for call in calls]
    assert kinds[0] == "init" and kinds[-1] == "close"
    assert "log_metrics" in kinds
    logged = next(call[1] for call in calls if call[0] == "log_metrics")
    assert logged, "the estimates must reach Comet as metrics"
    assert all(name.startswith("information/run/") for name in logged)
    assert ("log_image", figure.name, figure.stem, 0) in calls
    assert ("log_asset", "information_estimates.md") in calls
    assert summary["metrics"] == len(logged)
    assert summary["images"] == 1
    assert summary["url"] == "https://comet.invalid/analysis"


def test_a_missing_asset_or_image_is_skipped_rather_than_failing_the_export(
    tmp_path, monkeypatch
):
    calls = []
    _patch_sink(monkeypatch, calls)

    summary = _export_information_estimates_to_comet(
        _information_rows_fixture(),
        (tmp_path / "missing.md",),
        (tmp_path / "missing.png",),
        enabled=True, project_name="mas-cc", run_name="run/analysis",
    )

    assert summary["assets"] == 0
    assert summary["images"] == 0
    assert summary["metrics"] > 0
    assert not any(call[0] in {"log_asset", "log_image"} for call in calls)


def test_comet_metrics_derive_the_excess_over_the_permutation_null():
    """An MI value on its own is not a result; the gap over the null is.

    A permutation null is rarely zero, so a reader comparing bare estimates
    across statistics is comparing quantities with different floors.
    """

    rows = [
        {
            "cell_id": "run",
            "statistic": "sensing_mi",
            "estimate": 1.25,
            "null_mean": 0.5,
            "bootstrap_ci_low": 1.0,
            "bootstrap_ci_high": 1.4,
            "scientifically_interpretable": True,
        }
    ]

    metrics = information_comet_metrics(rows)

    assert metrics["information/run/sensing_mi/estimate"] == pytest.approx(1.25)
    assert metrics["information/run/sensing_mi/excess_over_null"] == pytest.approx(0.75)
    assert metrics["information/run/sensing_mi/scientifically_interpretable"] == 1.0


def test_a_statistic_with_no_estimate_contributes_no_metric():
    metrics = information_comet_metrics(
        [{"cell_id": "run", "statistic": "sensing_mi", "estimate": None, "null_mean": None}]
    )
    assert metrics == {}


def test_one_information_figure_is_written_per_cell(tmp_path):
    rows = [
        {**row, "cell_id": cell}
        for cell in ("A", "B")
        for row in _information_rows_fixture()
    ]

    written = plot_information_estimates(rows, tmp_path)

    assert sorted(path.name for path in written) == [
        "information_estimates_A.png",
        "information_estimates_B.png",
    ]
    assert all(path.stat().st_size > 0 for path in written)


def test_no_estimates_means_no_figure(tmp_path):
    assert plot_information_estimates((), tmp_path) == []
