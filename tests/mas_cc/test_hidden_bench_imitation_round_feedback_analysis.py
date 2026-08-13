"""Finite-channel fixtures for the round-feedback estimator."""

from __future__ import annotations

import json
import math

import pytest

from mas_cc.games.hidden_bench.imitation.controller import ADVOCATE_TARGET, NO_OP
from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
    adapt_round_record,
    analyze_hidden_bench_imitation_round_feedback,
    bootstrap_episode_rows,
    policy_resampling_null,
    round_analysis_comet_metrics,
    round_information_analysis,
)


def _row(
    *,
    episode: str,
    index: int,
    action: str | None,
    before=(2, 2, 0),
    after=(2, 2, 0),
    sensor=(1, 1, 0),
    probability=0.5,
):
    population_size = sum(before)
    option_count = len(before)

    def aligned(count):
        return (option_count * count / population_size - 1) / (option_count - 1)

    return adapt_round_record(
        {
            "record_type": "imitation_round_feedback",
            "episode_id": episode,
            "round_index": index,
            "occupation_counts_before": list(before),
            "occupation_counts_after": list(after),
            "target_count_before": before[0],
            "target_count_after": after[0],
            "truth_count_before": before[0],
            "truth_count_after": after[0],
            "sensor_count_vector": None if action is None else list(sensor),
            "sensor_target_share": None if action is None else sensor[0] / sum(sensor),
            "controller_action": action,
            "controller_advocate_probability": None if action is None else probability,
            "controller_enabled": action is not None,
            "delta_m_ctrl": (after[0] - before[0]) / 4,
            "delta_m_truth": (after[0] - before[0]) / 4,
            "delta_m_order": (max(after) - max(before)) / 4,
            "delta_H_vote": 0.0,
            "m_ctrl_before": aligned(before[0]),
            "m_ctrl_after": aligned(after[0]),
            "m_truth_before": aligned(before[0]),
            "m_truth_after": aligned(after[0]),
            "m_order_before": aligned(max(before)),
            "m_order_after": aligned(max(after)),
        }
    )


def _estimate(rows, statistic):
    estimates, _ = round_information_analysis(
        rows,
        statistics=[statistic],
        bootstrap_resamples=0,
        null_permutations=0,
    )
    return estimates[0]["estimate"] if estimates else None


def test_constant_and_balanced_binary_actions_have_zero_and_one_bit_entropy():
    constant = [_row(episode="e", index=i, action=NO_OP) for i in range(20)]
    balanced = [
        _row(
            episode="e",
            index=i,
            action=ADVOCATE_TARGET if i % 2 else NO_OP,
        )
        for i in range(20)
    ]
    assert _estimate(constant, "round_controller_action_entropy") == pytest.approx(0.0)
    assert _estimate(balanced, "round_controller_action_entropy") == pytest.approx(1.0)


def test_conditionally_independent_fixture_has_zero_actuation_cmi():
    rows = []
    index = 0
    for before in ((2, 2, 0), (1, 2, 1)):
        outcomes = (before, (before[0] + 1, before[1] - 1, before[2]))
        for action in (NO_OP, ADVOCATE_TARGET):
            for after in outcomes:
                rows.extend(
                    _row(
                        episode=f"e-{index}",
                        index=repetition,
                        action=action,
                        before=before,
                        after=after,
                    )
                    for repetition in range(10)
                )
                index += 1
    assert _estimate(rows, "round_population_actuation_cmi") == pytest.approx(0.0)


def test_deterministic_action_dependent_next_state_has_positive_cmi():
    rows = []
    for index in range(100):
        action = ADVOCATE_TARGET if index % 2 else NO_OP
        after = (3, 1, 0) if action == ADVOCATE_TARGET else (1, 3, 0)
        rows.append(_row(episode=f"e-{index // 10}", index=index, action=action, after=after))
    assert _estimate(rows, "round_population_actuation_cmi") == pytest.approx(1.0)


def test_policy_resampling_null_collapses_synthetic_action_outcome_signal():
    rows = []
    for index in range(400):
        action = ADVOCATE_TARGET if index % 2 else NO_OP
        after = (3, 1, 0) if action == ADVOCATE_TARGET else (1, 3, 0)
        rows.append(_row(episode=f"e-{index // 20}", index=index, action=action, after=after))
    null = policy_resampling_null(
        "round_population_actuation_cmi", rows, permutations=50, seed=20260813
    )
    assert sum(null) / len(null) < 0.02


def test_bootstrap_draws_whole_episode_ids():
    rows = [
        _row(episode=episode, index=index, action=NO_OP)
        for episode in ("a", "b", "c")
        for index in range(4)
    ]
    draws = bootstrap_episode_rows(rows, resamples=10, seed=7)
    assert all(len(draw) == 12 for draw in draws)
    for draw in draws:
        counts = {episode: sum(row.episode_id == episode for row in draw) for episode in ("a", "b", "c")}
        assert all(count % 4 == 0 for count in counts.values())


@pytest.mark.parametrize("counts", [(2, 1, 1), (1, 1, 1, 1)])
def test_target_count_encoding_is_stable_for_three_and_four_options(counts):
    row = _row(
        episode="e",
        index=0,
        action=NO_OP,
        before=counts,
        after=counts,
        sensor=tuple(1 if index < 2 else 0 for index in range(len(counts))),
    )
    assert row.N_k == counts
    assert row.target_before == counts[0]


def test_no_control_rows_do_not_emit_fabricated_controller_information():
    rows = [_row(episode="e", index=i, action=None) for i in range(10)]
    estimates, nulls = round_information_analysis(
        rows,
        statistics=[
            "round_population_actuation_cmi",
            "round_controller_action_entropy",
        ],
        bootstrap_resamples=0,
        null_permutations=10,
    )
    assert estimates == []
    assert nulls == []


def test_analysis_is_rerunnable_from_persisted_round_records(tmp_path):
    episode = tmp_path / "episode-0001"
    episode.mkdir()
    records = [
        _row(
            episode="episode-0001",
            index=index,
            action=ADVOCATE_TARGET if index % 2 else NO_OP,
            after=(3, 1, 0) if index % 2 else (1, 3, 0),
        ).event
        for index in range(10)
    ]
    (episode / "round_trajectory.jsonl").write_text(
        "\n".join(
            json.dumps({"record_type": "imitation_round_feedback", **dict(record)})
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "analysis"
    summary = analyze_hidden_bench_imitation_round_feedback(
        tmp_path,
        output,
        bootstrap_resamples=5,
        null_permutations=5,
        statistics=[
            "round_population_actuation_cmi",
            "round_controller_action_entropy",
        ],
    )
    assert summary["n_rounds"] == 10
    for name in (
        "round_information_estimates.csv",
        "round_information_estimates.md",
        "round_information_nulls.csv",
        "round_support_diagnostics.csv",
        "round_behavioral_summary.csv",
        "micro_slot_diagnostics.csv",
        "episode_currents.csv",
        "cell_summaries.csv",
    ):
        assert (output / name).is_file()


class _FakeCometSink:
    def __init__(self):
        self.status = "active"
        self.url = "https://comet.invalid/master"
        self.calls = []
        self.closed = False

    def add_tags(self, tags):
        self.calls.append(("add_tags", tuple(tags)))

    def log_metrics(self, metrics, step):
        self.calls.append(("log_metrics", dict(metrics), step))

    def log_asset(self, path, *, name):
        self.calls.append(("log_asset", name))

    def close(self):
        self.closed = True


def test_completed_cell_analysis_publishes_metrics_and_assets_to_borrowed_master(tmp_path):
    episode = tmp_path / "episode-0001"
    episode.mkdir()
    records = [
        _row(
            episode="episode-0001",
            index=index,
            action=ADVOCATE_TARGET if index % 2 else NO_OP,
            after=(3, 1, 0) if index % 2 else (1, 3, 0),
        ).event
        for index in range(10)
    ]
    (episode / "round_trajectory.jsonl").write_text(
        "\n".join(
            json.dumps({"record_type": "imitation_round_feedback", **dict(record)})
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )
    sink = _FakeCometSink()

    summary = analyze_hidden_bench_imitation_round_feedback(
        tmp_path,
        tmp_path / "analysis",
        bootstrap_resamples=5,
        null_permutations=5,
        statistics=[
            "round_population_actuation_cmi",
            "round_controller_action_entropy",
        ],
        comet_export=True,
        comet_sink=sink,
        comet_name_suffix="cell-0000",
    )

    logged = next(call[1] for call in sink.calls if call[0] == "log_metrics")
    assert logged["run/round_population_actuation_cmi/estimate"] == pytest.approx(1.0)
    assert "run/round_population_actuation_cmi/excess_over_null" in logged
    assert summary["comet"]["published_to"] == "master"
    assert summary["comet"]["metrics"] == len(logged)
    assert summary["comet"]["assets"] == 5
    assert any(
        call == ("log_asset", "round_information_estimates__cell-0000.csv")
        for call in sink.calls
    )
    assert any(
        call == ("log_asset", "analysis_summary__cell-0000.json")
        for call in sink.calls
    )
    assert sink.closed is False


def test_round_comet_metrics_skip_non_finite_values_and_encode_boolean_bounds():
    metrics = round_analysis_comet_metrics(
        [
            {
                "cell_id": "cell/one",
                "statistic": "round_population_actuation_cmi",
                "estimate": 0.25,
                "null_mean": 0.1,
                "bootstrap_ci_low": math.nan,
                "entropy_bound_satisfied": True,
            }
        ]
    )

    assert metrics == {
        "cell_one/round_population_actuation_cmi/estimate": 0.25,
        "cell_one/round_population_actuation_cmi/null_mean": 0.1,
        "cell_one/round_population_actuation_cmi/excess_over_null": pytest.approx(0.15),
        "cell_one/round_population_actuation_cmi/entropy_bound_satisfied": 1.0,
    }
