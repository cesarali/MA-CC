"""The cell/sweep metric tiers and the aggregation rules they must obey.

The rules in the spec's section 4 are correctness requirements, so each one is
tested by the bias it removes, not by its implementation: a forward-fill test
that only checked "the arrays are the same length" would still pass with the
biased version.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from mas_cc.experiments.aggregation import GridAggregator, aggregate_grid_directory
from mas_cc.config import AggregationConfig
from mas_cc.metrics import (
    AggregateResult,
    AggregationPolicy,
    CellProgress,
    LaggedConditionalMutualInformation,
    MacrostateCounts,
    MutualInformationGroundTruthGap,
    MutualInformationNullBand,
    TerminalMutualInformation,
    aggregate_cell,
    align,
    band_curve,
    create_cell_metrics,
    grid_progress_figure,
    read_cell_episodes,
    read_episode_frame,
    rolling_within,
    winner_ranking,
)

SHARE = "population_action_share_per_option"


def write_episode(directory: Path, shares: dict[str, list[float]], consensus=None) -> Path:
    """One episode directory in exactly the shape a worker leaves behind."""

    metrics_dir = directory / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rounds = len(next(iter(shares.values())))
    with (metrics_dir / "streaming.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["round_index", "episode_id", "agent_id", "series", "metric_name", "value"])
        for index in range(rounds):
            for option, values in shares.items():
                writer.writerow([index + 1, directory.name, "", option, SHARE, values[index]])
    with (metrics_dir / "final.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["episode_id", "metric_name", "value"])
        writer.writerow([directory.name, "first_consensus_time_by_action_share", consensus])
    return directory


def write_cell(cell_dir: Path, episodes: list[dict[str, list[float]]], consensus=None) -> Path:
    for index, shares in enumerate(episodes):
        write_episode(
            cell_dir / "data" / "episodes" / f"ep-{index:03d}",
            shares,
            consensus=None if consensus is None else consensus[index],
        )
    return cell_dir


# --- reading back what a worker wrote ----------------------------------------


def test_episode_frame_reads_option_curves_and_drops_agent_rows(tmp_path: Path):
    directory = write_episode(tmp_path / "ep", {"Q": [0.5, 0.7, 1.0], "M": [0.5, 0.3, 0.0]}, consensus=3)
    with (directory / "metrics" / "streaming.csv").open("a", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow([1, "ep", "agent-0", "", "agent_current_action", "Q"])

    frame = read_episode_frame(directory)

    assert frame is not None
    assert frame.rounds == (1, 2, 3)
    assert frame.option_series(SHARE) == {"Q": (0.5, 0.7, 1.0), "M": (0.5, 0.3, 0.0)}
    assert frame.final["first_consensus_time_by_action_share"] == 3.0


def test_episode_without_streaming_metrics_is_skipped_rather_than_counted(tmp_path: Path):
    """A game with no observer-aware runtime records nothing, which is not an error.

    Returning an empty frame instead would silently drag every percentile in
    the cell toward zero.
    """

    (tmp_path / "ep" / "metrics").mkdir(parents=True)
    assert read_episode_frame(tmp_path / "ep") is None


# --- 4.1 forward-fill absorbed episodes --------------------------------------


def test_forward_fill_pads_with_the_terminal_value_and_truncate_cuts_to_the_shortest():
    series = [[0.5, 1.0], [0.5, 0.6, 0.7, 1.0]]

    assert align(series, "absorbing") == ((0.5, 1.0, 1.0, 1.0), (0.5, 0.6, 0.7, 1.0))
    assert align(series, "truncate") == ((0.5, 1.0), (0.5, 0.6))
    assert align(series, "none")[0][2:] == (None, None)


def test_absorbing_fill_removes_the_mid_curve_dip_that_conditioning_on_survivors_creates():
    """Spec acceptance check 4.

    Episodes that converge early are exactly the *fast* ones. Averaging over
    only the runs still going at round t therefore conditions on "hasn't
    converged yet" and drags the middle of the curve back down. With absorbing
    fill the median must be monotone, and `active_fraction` must decay
    alongside it to show why the band tightens.
    """

    fast = [0.5, 0.8, 1.0]
    slow = [0.5, 0.55, 0.6, 0.65, 0.9, 1.0]
    episodes = [fast, fast, slow]

    filled = dict(band_curve(episodes, AggregationPolicy(rolling_window=1)).level("p50"))
    unfilled = dict(
        band_curve(episodes, AggregationPolicy(rolling_window=1, forward_fill="none")).level("p50")
    )

    assert list(filled.values()) == sorted(filled.values())
    # The biased variant collapses at round 4 onto the one surviving episode -
    # the slow one - so the median falls from 1.0 back to 0.65.
    assert unfilled[3] == pytest.approx(1.0)
    assert unfilled[4] == pytest.approx(0.65)
    assert filled[4] == pytest.approx(1.0)


def test_active_fraction_decays_alongside_a_forward_filled_band(tmp_path: Path):
    cell = write_cell(
        tmp_path / "cell",
        [
            {"Q": [0.5, 1.0], "M": [0.5, 0.0]},
            {"Q": [0.5, 0.7, 1.0], "M": [0.5, 0.3, 0.0]},
            {"Q": [0.5, 0.6, 0.8, 1.0], "M": [0.5, 0.4, 0.2, 0.0]},
        ],
    )
    result = aggregate_cell(
        read_cell_episodes(cell),
        create_cell_metrics(("dominant_action_share", "active_fraction")),
        AggregationPolicy(rolling_window=1),
    )

    active = dict(result.curves["active_fraction"].level("value"))
    assert active == {1: 1.0, 2: 1.0, 3: pytest.approx(2 / 3), 4: pytest.approx(1 / 3)}
    dominant = [value for _, value in result.curves["dominant_action_share"].level("p50")]
    assert dominant == sorted(dominant)


# --- 4.2 relabel by winner ---------------------------------------------------


def test_winner_ranking_orders_by_the_terminal_share_and_breaks_ties_on_the_label():
    assert winner_ranking({"Q": [0.5, 0.2], "M": [0.5, 0.8]}) == ("M", "Q")
    assert winner_ranking({"M": [0.5, 0.5], "Q": [0.5, 0.5]}) == ("M", "Q")


def test_relabelled_curves_stay_symmetric_when_no_option_systematically_wins(tmp_path: Path):
    """Spec acceptance check 3.

    Half the episodes end on Q and half on M, and neither option is favoured
    on the way there. The winner-aligned curves must therefore be mirror images
    away from the terminal round; if they are not, the relabel is wrong.
    """

    to_q = {"Q": [0.5, 0.7, 1.0], "M": [0.5, 0.3, 0.0]}
    to_m = {"Q": [0.5, 0.3, 0.0], "M": [0.5, 0.7, 1.0]}
    cell = write_cell(tmp_path / "cell", [to_q, to_m, to_q, to_m])

    result = aggregate_cell(
        read_cell_episodes(cell),
        create_cell_metrics(("action_share_relabelled",)),
        AggregationPolicy(rolling_window=1),
    )

    winner = dict(result.curves["action_share_relabelled_option_1"].level("p50"))
    loser = dict(result.curves["action_share_relabelled_option_2"].level("p50"))
    for round_index in winner:
        assert winner[round_index] + loser[round_index] == pytest.approx(1.0)
    assert winner[1] == pytest.approx(0.5)  # symmetric start, not a spurious lead
    assert winner[3] == pytest.approx(1.0)


def test_without_relabelling_the_same_cell_washes_out_to_a_flat_half(tmp_path: Path):
    """The bias 4.2 exists to remove, demonstrated on the same data."""

    to_q = {"Q": [0.5, 0.7, 1.0], "M": [0.5, 0.3, 0.0]}
    to_m = {"Q": [0.5, 0.3, 0.0], "M": [0.5, 0.7, 1.0]}
    cell = write_cell(tmp_path / "cell", [to_q, to_m, to_q, to_m])

    result = aggregate_cell(
        read_cell_episodes(cell),
        create_cell_metrics(("action_share_relabelled",)),
        AggregationPolicy(rolling_window=1, relabel_by_winner=False),
    )

    terminal = dict(result.curves["action_share_relabelled_Q"].level("p50"))[3]
    assert terminal == pytest.approx(0.5)  # "nothing happened", which is false


# --- 4.3 percentile bands ----------------------------------------------------


def test_percentile_bands_stay_inside_the_unit_interval_on_a_bimodal_cell(tmp_path: Path):
    cell = write_cell(
        tmp_path / "cell",
        [{"Q": [0.5, 1.0], "M": [0.5, 0.0]}] * 3 + [{"Q": [0.5, 0.0], "M": [0.5, 1.0]}] * 3,
    )
    result = aggregate_cell(
        read_cell_episodes(cell),
        create_cell_metrics(("action_share_relabelled",)),
        AggregationPolicy(rolling_window=1, relabel_by_winner=False),
    )

    for curve in result.curves.values():
        for values in curve.points.values():
            assert all(0.0 <= value <= 1.0 for value in values)


# --- 4.4 roll within, then aggregate -----------------------------------------


def test_rolling_is_applied_within_each_episode_not_after_aggregating():
    """The two orders give different answers, and only one of them is asked for."""

    episodes = [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]
    policy = AggregationPolicy(rolling_window=2, percentiles=(50,), forward_fill="truncate")

    within = [value for _, value in band_curve(episodes, policy).level("p50")]
    after = rolling_within([0.5, 0.5, 0.5, 0.5], 2)  # median first, then roll

    assert within != after
    assert rolling_within(episodes[0], 2) == (0.0, 0.0, 0.5, 1.0)


# --- consensus and convergence ----------------------------------------------


def test_consensus_scalars_exclude_never_converged_episodes_and_count_them_separately(tmp_path: Path):
    cell = write_cell(
        tmp_path / "cell",
        [{"Q": [0.5, 1.0], "M": [0.5, 0.0]}] * 3,
        consensus=[2, 4, None],
    )
    result = aggregate_cell(
        read_cell_episodes(cell),
        create_cell_metrics(("consensus_round", "converged_fraction")),
        AggregationPolicy(),
    )

    assert result.scalars["median_consensus_round"] == pytest.approx(3.0)
    assert result.scalars["converged_fraction"] == pytest.approx(2 / 3)


# --- the sweep tier ----------------------------------------------------------


def _cell_with_terminal(counts: dict[str, float]) -> AggregateResult:
    return AggregateResult(counts={"terminal_outcome": counts}, episodes=int(sum(counts.values())))


def test_a_perfectly_informative_grid_gives_one_bit_and_a_null_band_far_below_it():
    cells = {"cell-0000": _cell_with_terminal({"Q": 50.0}), "cell-0001": _cell_with_terminal({"M": 50.0})}

    estimate = TerminalMutualInformation().compute(cells).scalars
    null = MutualInformationNullBand(permutations=100, seed=7).compute(cells).scalars

    assert estimate["terminal_mi_unsmoothed"] == pytest.approx(1.0)
    assert estimate["terminal_mi_episodes"] == 100
    assert null["terminal_mi_null_p95"] < 0.2


def test_an_uninformative_grid_lands_inside_its_own_null_band():
    cells = {
        "cell-0000": _cell_with_terminal({"Q": 25.0, "M": 25.0}),
        "cell-0001": _cell_with_terminal({"Q": 25.0, "M": 25.0}),
    }

    estimate = TerminalMutualInformation().compute(cells).scalars["terminal_mi_estimate"]
    null = MutualInformationNullBand(permutations=200, seed=3).compute(cells).scalars

    assert estimate <= null["terminal_mi_null_p95"]


def test_a_single_completed_cell_reports_nan_rather_than_zero_bits():
    """One condition level carries no information *by construction*.

    Reporting 0.0 would be indistinguishable from "swept, and it did nothing".
    """

    result = TerminalMutualInformation().compute({"cell-0000": _cell_with_terminal({"Q": 10.0})})
    assert math.isnan(result.scalars["terminal_mi_estimate"])


def test_the_ground_truth_gap_is_absent_without_an_answer_key():
    cells = {"cell-0000": _cell_with_terminal({"Q": 50.0}), "cell-0001": _cell_with_terminal({"M": 50.0})}

    assert MutualInformationGroundTruthGap(None).compute(cells).scalars == {}

    gap = MutualInformationGroundTruthGap(1.0).compute(cells).scalars

    assert gap["terminal_mi_ground_truth"] == 1.0
    # A perfectly informative grid *is* 1 bit, so the gap is only the estimator's
    # own smoothing bias — small, negative, and not zero.
    assert -0.2 < gap["terminal_mi_gap"] <= 0.0


def test_macrostate_counts_never_pair_rounds_across_an_episode_boundary(tmp_path: Path):
    cell = write_cell(
        tmp_path / "cell",
        [{"Q": [1.0, 1.0], "M": [0.0, 0.0]}, {"Q": [0.0, 0.0], "M": [1.0, 1.0]}],
    )
    result = MacrostateCounts(horizons=(1,)).compute(read_cell_episodes(cell), AggregationPolicy())

    assert result.counts["terminal_outcome"] == {"Q": 1.0, "M": 1.0}
    # Two episodes of two rounds give one pair each, never a Q->M crossing.
    assert result.counts["macrostate_transition_h1"] == {"Q>Q": 1.0, "M>M": 1.0}


# --- persistence and recomputation ------------------------------------------


def test_aggregating_a_cell_writes_its_result_before_anything_is_published(tmp_path: Path):
    """Spec acceptance check 1: a cell finalized at its own completion survives.

    The file must exist as soon as the cell is aggregated, with no dependence
    on the rest of the sweep finishing or on Comet being reachable.
    """

    write_cell(tmp_path / "cells" / "cell-0000", [{"Q": [0.5, 1.0], "M": [0.5, 0.0]}] * 2)
    aggregator = GridAggregator(tmp_path, AggregationConfig())

    aggregator.aggregate("cell-0000")

    assert (tmp_path / "cells" / "cell-0000" / "aggregate.json").exists()


def test_reaggregating_from_disk_reproduces_identical_curves(tmp_path: Path):
    """Spec acceptance check 2: aggregates are derived, so they are recomputable."""

    write_cell(tmp_path / "cells" / "cell-0000", [{"Q": [0.5, 0.7, 1.0], "M": [0.5, 0.3, 0.0]}] * 3)
    write_cell(tmp_path / "cells" / "cell-0001", [{"Q": [0.5, 0.3, 0.0], "M": [0.5, 0.7, 1.0]}] * 3)
    config = AggregationConfig(sweep_metrics=("terminal_mi",))

    first = GridAggregator(tmp_path, config)
    for cell_id in ("cell-0000", "cell-0001"):
        first.aggregate(cell_id)
    original = (tmp_path / "cells" / "cell-0000" / "aggregate.json").read_text(encoding="utf-8")

    summary = aggregate_grid_directory(tmp_path, config)

    assert summary["cells_aggregated"] == ["cell-0000", "cell-0001"]
    assert (tmp_path / "cells" / "cell-0000" / "aggregate.json").read_text(encoding="utf-8") == original
    assert summary["sweep_metrics"]["terminal_mi_unsmoothed"] == pytest.approx(1.0)


def test_a_grid_killed_partway_still_yields_correct_output_for_the_finished_cell(tmp_path: Path):
    """Spec acceptance check 1, from the other side: no dependence on the unfinished."""

    write_cell(tmp_path / "cells" / "cell-0000", [{"Q": [0.5, 1.0], "M": [0.5, 0.0]}] * 4)
    (tmp_path / "cells" / "cell-0001").mkdir(parents=True)  # never started

    summary = aggregate_grid_directory(tmp_path, AggregationConfig())

    assert summary["cells_aggregated"] == ["cell-0000"]
    assert not (tmp_path / "cells" / "cell-0001" / "aggregate.json").exists()


def test_a_sweep_metric_needing_count_tables_pulls_them_in_automatically():
    """Asking for `terminal_mi` without `macrostate_counts` would silently give NaN."""

    config = AggregationConfig(sweep_metrics=("terminal_mi",))
    assert "macrostate_counts" in config.resolved_cell_metrics()
    assert "macrostate_counts" not in AggregationConfig().resolved_cell_metrics()


# --- the grid image ----------------------------------------------------------


def test_the_grid_image_marks_a_cell_with_failures_as_bad_rather_than_full():
    """All-green must mean all-*healthy*, not merely all-finished."""

    figure = grid_progress_figure(
        [("game.options.epsilon", [0.1, 0.2])],
        [CellProgress((0.1,), done=10, total=10), CellProgress((0.2,), done=10, total=10, failed=1)],
    )
    values = figure.axes[0].images[0].get_array()

    assert values[0][0] == pytest.approx(1.0)
    assert values.mask[1][0]  # pushed out of the colour scale, so it draws red


def test_the_grid_image_needs_at_least_one_axis():
    with pytest.raises(ValueError, match="at least one swept axis"):
        grid_progress_figure([], [])


def test_configured_horizons_reach_the_cell_that_builds_the_transition_tables(tmp_path: Path):
    """A lagged CMI at h=3 needs the cell to have counted pairs at h=3.

    The cell tier and the sweep tier read the same `horizons:` list, and only
    the sweep tier used to. A cell built at the default h=1 against a sweep
    asking for h=3 reports NaN with nothing to say why, which is the worst
    possible failure for a number someone is about to put in a paper.
    """

    cell = write_cell(tmp_path / "cell", [{"Q": [1.0, 1.0, 0.0, 0.0], "M": [0.0, 0.0, 1.0, 1.0]}])
    metrics = create_cell_metrics(("macrostate_counts",), horizons=(1, 3))

    result = aggregate_cell(read_cell_episodes(cell), metrics, AggregationPolicy())

    assert set(result.counts) == {"terminal_outcome", "macrostate_transition_h1", "macrostate_transition_h3"}
    assert result.counts["macrostate_transition_h3"] == {"Q>M": 1.0}


def test_a_lagged_cmi_horizon_the_cells_never_counted_is_nan_not_zero():
    cells = {
        "cell-0000": AggregateResult(counts={"macrostate_transition_h1": {"Q>Q": 10.0, "Q>M": 2.0}}),
        "cell-0001": AggregateResult(counts={"macrostate_transition_h1": {"M>M": 10.0, "M>Q": 2.0}}),
    }

    scalars = LaggedConditionalMutualInformation(horizons=(1, 7)).compute(cells).scalars

    assert scalars["lagged_cmi_h1_estimate"] > 0
    assert math.isnan(scalars["lagged_cmi_h7_estimate"])
