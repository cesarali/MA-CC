"""The validation suite required by the Ashery metrics spec, section 10.

Test names map to the spec's own numbered requirements so a reader can check
the implementation against the document rather than against these tests.
"""

import pytest

from mas_cc.metrics.interactions import (
    InteractionOutcome,
    consensus_flip,
    non_overlapping_bins,
    production_counts,
    production_probabilities,
    production_probability_rows,
    success_rate,
    success_rate_rows,
)


def _records(pairs, committed=None) -> list[InteractionOutcome]:
    """Build a trajectory from (action_1, action_2) pairs, 1-based indices."""

    flags = committed or [(False, False)] * len(pairs)
    return [
        InteractionOutcome(interaction_index=index, actions=pair, committed=flag)
        for index, (pair, flag) in enumerate(zip(pairs, flags, strict=True), start=1)
    ]


# The spec's running example (sections 2.3, 3.2 and 10's joint sanity check).
_EXAMPLE = _records([("M", "M"), ("M", "Q"), ("Q", "Q"), ("M", "M")])


# --- section 10, success rate ------------------------------------------------


def test_all_pairs_match_gives_success_rate_one():
    assert success_rate(_records([("M", "M"), ("Q", "Q")])) == 1.0


def test_no_pairs_match_gives_success_rate_zero():
    assert success_rate(_records([("M", "Q"), ("Q", "M")])) == 0.0


def test_three_matches_in_four_interactions_gives_success_rate_three_quarters():
    assert success_rate(_EXAMPLE) == 0.75


def test_success_rate_on_empty_input_raises():
    with pytest.raises(ValueError, match="at least one interaction"):
        success_rate([])


# --- section 10, production probability --------------------------------------


def test_probabilities_sum_to_one_over_the_complete_action_space():
    probabilities = production_probabilities(_EXAMPLE, action_space=("M", "Q"))
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_a_bin_of_l_ordinary_interactions_has_two_l_eligible_outputs():
    _, eligible = production_counts(_EXAMPLE)
    assert eligible == 2 * len(_EXAMPLE) == 8


def test_unobserved_legal_actions_get_probability_zero():
    records = _records([("M", "M"), ("M", "M")])
    assert production_probabilities(records, action_space=("M", "Q", "Z")) == {
        "M": 1.0, "Q": 0.0, "Z": 0.0,
    }


def test_excluding_committed_outputs_changes_numerator_and_denominator():
    # Two interactions; the first output of each comes from a committed agent
    # hard-wired to M, so four outputs total and two of them are excluded.
    records = _records(
        [("M", "Q"), ("M", "Q")],
        committed=[(True, False), (True, False)],
    )
    assert production_counts(records) == ({"M": 2, "Q": 2}, 4)
    assert production_counts(records, exclude_committed_outputs=True) == ({"Q": 2}, 2)

    included = production_probabilities(records, action_space=("M", "Q"))
    ordinary = production_probabilities(
        records, action_space=("M", "Q"), exclude_committed_outputs=True
    )
    assert included == {"M": 0.5, "Q": 0.5}
    # The free agents never said M; without filtering the committed pair hides that.
    assert ordinary == {"M": 0.0, "Q": 1.0}


def test_excluding_every_output_raises_instead_of_returning_nan():
    records = _records([("M", "M")], committed=[(True, True)])
    with pytest.raises(ValueError, match="no eligible outputs"):
        production_probabilities(records, exclude_committed_outputs=True)


# --- section 10, joint sanity check ------------------------------------------


def test_joint_sanity_check_from_the_specification():
    assert success_rate(_EXAMPLE) == 0.75
    probabilities = production_probabilities(_EXAMPLE, action_space=("M", "Q"))
    assert probabilities["M"] == 0.625
    assert probabilities["Q"] == 0.375


def test_identical_production_probabilities_can_have_opposite_success_rates():
    """Section 4: production probability alone does not measure coordination."""

    coordinated = _records([("M", "M"), ("M", "M"), ("Q", "Q"), ("Q", "Q")])
    mismatched = _records([("M", "Q"), ("M", "Q"), ("Q", "M"), ("Q", "M")])
    assert production_probabilities(coordinated, action_space=("M", "Q")) == {"M": 0.5, "Q": 0.5}
    assert production_probabilities(mismatched, action_space=("M", "Q")) == {"M": 0.5, "Q": 0.5}
    assert success_rate(coordinated) == 1.0
    assert success_rate(mismatched) == 0.0


# --- section 5, binning ------------------------------------------------------


def test_bins_are_consecutive_and_non_overlapping():
    records = _records([("M", "M")] * 6)
    bins = non_overlapping_bins(records, bin_size=2)
    assert [[r.interaction_index for r in b] for b in bins] == [[1, 2], [3, 4], [5, 6]]


def test_partial_final_bin_drop_omits_it():
    records = _records([("M", "M")] * 5)
    assert [len(b) for b in non_overlapping_bins(records, bin_size=2)] == [2, 2]


def test_partial_final_bin_include_keeps_it():
    records = _records([("M", "M")] * 5)
    bins = non_overlapping_bins(records, bin_size=2, partial_final_bin="include")
    assert [len(b) for b in bins] == [2, 2, 1]


def test_partial_final_bin_error_rejects_an_indivisible_trajectory():
    records = _records([("M", "M")] * 5)
    with pytest.raises(ValueError, match="do not divide into bins"):
        non_overlapping_bins(records, bin_size=2, partial_final_bin="error")


def test_bin_size_must_be_positive():
    with pytest.raises(ValueError, match="bin_size must be positive"):
        non_overlapping_bins(_EXAMPLE, bin_size=0)


def test_unknown_partial_bin_policy_is_rejected():
    with pytest.raises(ValueError, match="partial_final_bin must be"):
        non_overlapping_bins(_EXAMPLE, bin_size=2, partial_final_bin="keep")


# --- section 9, output schema ------------------------------------------------


def test_success_rate_rows_carry_raw_counts_and_bounds():
    rows = success_rate_rows(_EXAMPLE, episode_id="ep-1", bin_size=2)
    assert rows == [
        {
            "episode_id": "ep-1", "bin_index": 0, "start_interaction": 1, "end_interaction": 2,
            "num_pair_interactions": 2, "success_count": 1, "success_rate": 0.5,
        },
        {
            "episode_id": "ep-1", "bin_index": 1, "start_interaction": 3, "end_interaction": 4,
            "num_pair_interactions": 2, "success_count": 2, "success_rate": 1.0,
        },
    ]


def test_production_probability_rows_carry_raw_counts_per_action():
    rows = production_probability_rows(
        _EXAMPLE, episode_id="ep-1", bin_size=4, action_space=("M", "Q")
    )
    assert [row["action"] for row in rows] == ["M", "Q"]
    assert [row["action_count"] for row in rows] == [5, 3]
    assert all(row["eligible_output_count"] == 8 for row in rows)
    assert [row["production_probability"] for row in rows] == [0.625, 0.375]
    assert all(row["excluded_committed_outputs"] is False for row in rows)


def test_row_counts_recompute_the_stored_probability():
    """Section 11: raw counts stored with normalized values must reconstruct them."""

    rows = production_probability_rows(
        _EXAMPLE, episode_id="ep-1", bin_size=2, action_space=("M", "Q")
    )
    for row in rows:
        assert row["action_count"] / row["eligible_output_count"] == row["production_probability"]


# --- section 7, consensus detection ------------------------------------------


def test_consensus_flip_reports_the_first_qualifying_window_and_the_winner():
    # Four consecutive successes on Q after one failure; window of 4 first
    # qualifies at interaction 5.
    records = _records([("M", "Q"), ("Q", "Q"), ("Q", "Q"), ("Q", "Q"), ("Q", "Q")])
    assert consensus_flip(records, window=4) == (5, "Q")


def test_consensus_flip_is_none_when_the_criterion_never_holds():
    records = _records([("M", "Q")] * 10)
    assert consensus_flip(records, window=4) is None


def test_consensus_flip_needs_a_full_window_before_it_can_fire():
    records = _records([("Q", "Q")] * 3)
    assert consensus_flip(records, window=4) is None
    assert consensus_flip(records, window=3) == (3, "Q")


def test_consensus_flip_names_the_winning_word_not_just_agreement():
    """Success rate alone says they agreed; it does not say on what."""

    records = _records([("M", "M")] * 4)
    assert consensus_flip(records, window=4) == (4, "M")


def test_consensus_flip_rejects_invalid_window_and_threshold():
    with pytest.raises(ValueError, match="window must be positive"):
        consensus_flip(_EXAMPLE, window=0)
    with pytest.raises(ValueError, match="threshold must be in"):
        consensus_flip(_EXAMPLE, window=2, threshold=1.5)


# --- record construction ------------------------------------------------------


def test_committed_defaults_to_all_ordinary():
    record = InteractionOutcome(interaction_index=1, actions=("M", "Q"))
    assert record.committed == (False, False)


def test_committed_flags_must_match_the_action_count():
    with pytest.raises(ValueError, match="one flag per action"):
        InteractionOutcome(interaction_index=1, actions=("M", "Q"), committed=(True,))


def test_from_evaluator_entry_reads_a_naming_convention_history_row():
    record = InteractionOutcome.from_evaluator_entry(
        {"interaction_index": 7, "actions": ["Q", "Q"], "success": True, "committed": [True, False]}
    )
    assert record.interaction_index == 7
    assert record.actions == ("Q", "Q")
    assert record.committed == (True, False)
    assert record.succeeded is True


def test_from_evaluator_entry_tolerates_history_without_committed_flags():
    record = InteractionOutcome.from_evaluator_entry(
        {"interaction_index": 1, "actions": ["Q", "M"], "success": False}
    )
    assert record.committed == (False, False)
