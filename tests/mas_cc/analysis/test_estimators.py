import math

import numpy as np
import pytest

from mas_cc.analysis.estimators import (
    complete_cells,
    conditional_mutual_information_from_counts,
    mutual_information_from_counts,
)


def test_perfectly_correlated_2x2_table_gives_log2():
    counts = np.array([[50.0, 0.0], [0.0, 50.0]])
    estimate = mutual_information_from_counts(counts)
    assert estimate.unsmoothed == pytest.approx(math.log2(2), abs=1e-9)
    assert estimate.observations == 100


def test_independent_table_gives_near_zero_mi():
    counts = np.array([[25.0, 25.0], [25.0, 25.0]])
    estimate = mutual_information_from_counts(counts)
    assert estimate.unsmoothed == pytest.approx(0.0, abs=1e-9)


def test_jeffreys_and_miller_madow_differ_from_unsmoothed_on_sparse_counts():
    counts = np.array([[1.0, 0.0], [0.0, 1.0]])
    estimate = mutual_information_from_counts(counts)
    assert estimate.jeffreys != estimate.unsmoothed
    assert estimate.miller_madow != estimate.unsmoothed


def test_empty_counts_return_nan():
    counts = np.zeros((2, 2))
    estimate = mutual_information_from_counts(counts)
    assert math.isnan(estimate.unsmoothed)
    assert estimate.observations == 0


def test_negative_or_wrong_shape_counts_are_rejected():
    with pytest.raises(ValueError):
        mutual_information_from_counts(np.array([[-1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError):
        mutual_information_from_counts(np.zeros((2, 2, 2)))


def test_conditional_mi_is_zero_when_y_is_independent_of_x_given_z():
    # For every z, x and y are independent -> I(X;Y|Z) ~ 0.
    counts = np.zeros((2, 2, 2))
    for z in range(2):
        counts[:, z, :] = 25.0
    estimate = conditional_mutual_information_from_counts(counts)
    assert estimate.unsmoothed == pytest.approx(0.0, abs=1e-9)


def test_conditional_mi_is_positive_when_x_determines_y_given_z():
    counts = np.zeros((2, 2, 2))
    for z in range(2):
        counts[0, z, 0] = 25.0
        counts[1, z, 1] = 25.0
    estimate = conditional_mutual_information_from_counts(counts)
    assert estimate.unsmoothed == pytest.approx(math.log2(2), abs=1e-9)


def test_complete_cells_is_the_cartesian_product():
    cells = complete_cells(("a", "b"), (1, 2))
    assert cells == (("a", 1), ("a", 2), ("b", 1), ("b", 2))
