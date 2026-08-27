from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from mas_cc.games.hidden_bench.imitation.controller import ADVOCATE_TARGET
from mas_cc.games.relational_reasoning.imitation_round_feedback.analysis import (
    adapt_relational_round_record,
    analyze_relational_imitation_round_feedback,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.current import (
    CURRENT_SUMMARY_REQUIRED_FIELDS,
    current_analysis_comet_metrics,
    current_cell_summary,
    empirical_current_statistics,
    episode_current_rows,
    write_current_analysis,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.matched_qvoter import (
    TheoryParameters,
    classical_reference,
    q1_current_closed_forms,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.theory_revised import (
    finite_horizon_current_moments,
)


def _event(
    *,
    episode: str,
    task: str = "task-a",
    round_index: int = 0,
    before: int = 1,
    after: int = 2,
    probability: float = 1.0,
):
    N = 4
    record = {
        "record_type": "relational_imitation_round_feedback",
        "episode_id": episode,
        "task_id": task,
        "round_index": round_index,
        "possible_answers": ["A", "B"],
        "correct_answer": "A",
        "analysis_target": "B",
        "occupation_counts_before": [N - before, before],
        "occupation_counts_after": [N - after, after],
        "controller_action": ADVOCATE_TARGET,
        "controller_advocate_probability": probability,
        "N": N,
        "social_group_size": 1,
        "sensor_sample_size": 2,
        "intervention_budget": 1,
        "controller_beta": 4.0,
        "controller_threshold": 0.5,
        "sensor_count_vector": [1, 1],
        "knowledge_stratum_counts_before": [4, 0],
        "mean_supporting_fact_coverage_before": 0.0,
        "full_proof_agent_share_before": 0.0,
        "delta_m_ctrl": (after - before) / N,
        "delta_m_truth": (before - after) / N,
        "delta_m_order": 0.0,
        "delta_H_vote": 0.0,
    }
    return adapt_relational_round_record(record, cell_id="cell-0")


def test_empirical_current_fixture_and_reciprocal_ratios():
    row = empirical_current_statistics([1.0, 3.0])
    assert row["current_mean_empirical"] == pytest.approx(2.0)
    assert row["current_variance_empirical"] == pytest.approx(2.0)
    assert row["current_fano_dispersion_empirical"] == pytest.approx(1.0)
    assert row["current_precision_irisarri_empirical"] == pytest.approx(1.0)
    assert row["current_snr2_empirical"] == pytest.approx(2.0)
    assert (
        row["current_fano_dispersion_empirical"]
        * row["current_precision_irisarri_empirical"]
    ) == pytest.approx(1.0)


def test_empirical_current_degenerate_cases_are_explicit():
    fixed = empirical_current_statistics([2.0, 2.0])
    assert fixed["current_fano_dispersion_empirical"] == 0.0
    assert math.isinf(fixed["current_precision_irisarri_empirical"])
    assert math.isinf(fixed["current_snr2_empirical"])
    assert fixed["current_snr2_zero_variance_nonzero_mean_empirical"] is True

    zero = empirical_current_statistics([0.0, 0.0])
    assert math.isnan(zero["current_fano_dispersion_empirical"])
    assert math.isnan(zero["current_precision_irisarri_empirical"])
    assert math.isnan(zero["current_snr2_empirical"])
    assert zero["current_snr2_degenerate_zero_current_empirical"] is True

    cancelling = empirical_current_statistics([-1.0, 1.0])
    assert math.isinf(cancelling["current_fano_dispersion_empirical"])
    assert cancelling["current_precision_irisarri_empirical"] == 0.0
    assert cancelling["current_snr2_empirical"] == 0.0
    assert cancelling["current_fano_zero_mean_empirical"] is True


def test_episode_current_is_terminal_difference_and_checks_micro_sum():
    rows = [
        _event(episode="e0", round_index=0, before=1, after=2),
        _event(episode="e0", round_index=1, before=2, after=3),
    ]
    micro = [
        {
            "cell_id": "cell-0",
            "episode_id": "e0",
            "focal_opinion_before": "A",
            "focal_opinion_after": "B",
        },
        {
            "cell_id": "cell-0",
            "episode_id": "e0",
            "focal_opinion_before": "A",
            "focal_opinion_after": "B",
        },
    ]
    result = episode_current_rows(rows, micro)[0]
    assert result["episode_current"] == 3 - 1
    assert result["microscopic_current"] == 2
    assert result["microscopic_current_matches_terminal"] is True


def test_finite_horizon_moments_match_direct_enumeration():
    kernel = np.array(
        [
            [0.7, 0.3, 0.0],
            [0.2, 0.5, 0.3],
            [0.0, 0.4, 0.6],
        ]
    )
    initial = np.array([0.25, 0.75, 0.0])
    transition = np.linalg.matrix_power(kernel, 2)
    outcomes = [
        (initial[n] * transition[n, m], m - n)
        for n in range(3)
        for m in range(3)
    ]
    mean = sum(probability * current for probability, current in outcomes)
    second = sum(probability * current**2 for probability, current in outcomes)
    result = finite_horizon_current_moments(kernel, initial, 2)
    assert result["mean"] == pytest.approx(mean)
    assert result["second_moment"] == pytest.approx(second)
    assert result["variance"] == pytest.approx(second - mean**2)


def test_q1_closed_form_matches_the_exact_one_round_kernel():
    parameters = TheoryParameters(N=4, q=1, q_c=2, b=1, beta=4.0, theta=0.5)
    reference = classical_reference(parameters)
    initial = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    closed = q1_current_closed_forms(initial, N=4, b=1)
    states = np.arange(5, dtype=float)
    advocate_mean = float(reference.R1[1] @ states) - 1.0
    noop_mean = float(reference.R0[1] @ states) - 1.0
    assert closed["q1_current_mean_noop_closed_form_theory"] == pytest.approx(
        noop_mean, abs=1e-12
    )
    assert closed["q1_current_mean_advocate_closed_form_theory"] == pytest.approx(
        advocate_mean, abs=1e-12
    )


def test_cell_summary_uses_stochastic_kernel_when_realized_actions_do_not_vary():
    # Seeing only ADVOCATE in two finite episodes does not make the controller
    # open loop: the logged action probability remains strictly between 0 and 1.
    rows = [
        _event(episode="e0", before=1, after=2, probability=0.5),
        _event(episode="e1", before=1, after=4, probability=0.5),
    ]
    summary, episodes = current_cell_summary(
        rows, bootstrap_resamples=5, confidence=0.95, seed=7
    )
    assert [row["episode_current"] for row in episodes] == [1, 3]
    assert summary["theory_mode"] == "single_affinity_revised"
    assert summary["current_mean_empirical"] == pytest.approx(2.0)
    assert summary["current_variance_empirical"] == pytest.approx(2.0)
    assert summary["current_precision_support"] == "descriptive_only"
    assert summary["current_bootstrap_unit"] == "episode"
    assert summary["theory_applicable"] is False
    assert "do not identify a finite h" in summary["theory_skip_reason"]


def test_tasks_are_not_pooled_and_both_sides_share_each_report(tmp_path):
    rows = [
        _event(episode="a0", task="task-a", before=1, after=2),
        _event(episode="a1", task="task-a", before=1, after=4),
        _event(episode="b0", task="task-b", before=1, after=1),
        _event(episode="b1", task="task-b", before=1, after=1),
    ]
    summaries, episodes, reports = write_current_analysis(
        rows,
        tmp_path,
        bootstrap_resamples=2,
        confidence=0.95,
        seed=1,
    )
    assert len(summaries) == len(reports) == 2
    assert len(episodes) == 4
    by_task = {row["task_id"]: row for row in summaries}
    assert by_task["task-a"]["current_variance_empirical"] == pytest.approx(2.0)
    assert by_task["task-b"]["current_variance_empirical"] == pytest.approx(0.0)
    for path in reports:
        text = path.read_text(encoding="utf-8")
        assert "Empirical repeated-episode current" in text
        assert "Revised single-affinity finite-horizon current" in text
        assert "Direct comparison" in text
    frame = pd.read_csv(tmp_path / "currents" / "cell_current_summary.csv")
    assert set(CURRENT_SUMMARY_REQUIRED_FIELDS) <= set(frame.columns)


def test_comet_current_keys_do_not_emit_generic_or_unavailable_theory_values():
    summary, _ = current_cell_summary(
        [_event(episode="e0"), _event(episode="e1", after=3)],
        bootstrap_resamples=0,
        confidence=0.95,
        seed=1,
    )
    metrics = current_analysis_comet_metrics([summary])
    assert {
        "current/mean_empirical",
        "current/variance_empirical",
        "current/fano_dispersion_empirical",
        "current/precision_irisarri_empirical",
        "current/snr2_empirical",
        "current/n_repetitions",
    } <= set(metrics)
    assert not any(key.endswith("_theory") for key in metrics)


class _FakeCometSink:
    status = "active"
    url = "https://comet.invalid/master"

    def __init__(self):
        self.logged = []
        self.assets = []
        self.tags = []

    def add_tags(self, tags):
        self.tags.extend(tags)

    def log_metrics(self, metrics, step):
        self.logged.append((dict(metrics), step))

    def log_asset(self, path, *, name):
        self.assets.append((path, name))


def test_comet_off_keeps_local_results_and_master_sink_gets_only_aggregates(tmp_path):
    episode_dir = tmp_path / "cells" / "cell-0" / "data" / "episodes" / "e0"
    episode_dir.mkdir(parents=True)
    (tmp_path / "cells" / "cell-0" / "overrides.json").write_text(
        "{}\n", encoding="utf-8"
    )
    records = []
    for episode, after in (("e0", 2), ("e1", 3)):
        row = _event(episode=episode, before=1, after=after)
        records.append(json.dumps(dict(row.event)))
    (episode_dir / "round_trajectory.jsonl").write_text(
        "\n".join(records) + "\n", encoding="utf-8"
    )

    local = tmp_path / "local-analysis"
    summary = analyze_relational_imitation_round_feedback(
        tmp_path,
        local,
        statistics=["round_target_actuation_cmi"],
        bootstrap_resamples=0,
        null_permutations=0,
    )
    assert summary["comet"]["status"] == "disabled"
    assert (local / "currents" / "cell_current_summary.csv").is_file()
    assert (local / "currents" / "episode_currents.csv").is_file()
    assert (local / "currents" / "current_analysis.md").is_file()

    sink = _FakeCometSink()
    remote = tmp_path / "remote-analysis"
    summary = analyze_relational_imitation_round_feedback(
        tmp_path,
        remote,
        statistics=["round_target_actuation_cmi"],
        bootstrap_resamples=0,
        null_permutations=0,
        comet_export=True,
        comet_sink=sink,
    )
    assert summary["comet"]["published_to"] == "master"
    assert len(sink.logged) == 1
    logged, step = sink.logged[0]
    assert step == 0
    assert "current/mean_empirical" in logged
    assert "current/mean_theory" not in logged
    assert not any(key.endswith("_theory") for key in logged)
    assert all("prompt" not in key and "provider" not in key for key in logged)
