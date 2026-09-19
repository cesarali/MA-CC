"""Horizon-0 checkpoint rows: one per parent/copy/target, no branch budget, order-independent."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mas_cc.studies.aggregation import _checkpoint_h0_rows


def _endpoints(order):
    rows = []
    for parent in ("p0", "p1"):
        for policy, budget in order:
            for horizon in (1, 2):
                for semantics, n0 in (("truth", 5), ("false_target", 2)):
                    rows.append({"parent_id": parent, "copy_id": 1, "q": 3, "rho": 0.7, "target_semantics": semantics,
                                 "checkpoint_id": f"{parent}-h0", "branch_policy": policy, "posting_budget": budget,
                                 "N": 10, "L": 2, "M": 2, "post_branch_horizon": horizon, "absolute_round": 2 + horizon,
                                 "target_answer": "T", "n_0": n0, "target_count": n0 + horizon, "target_fraction": (n0 + horizon) / 10})
    return pd.DataFrame(rows)


def test_h0_rows_are_one_per_group_with_no_branch_coordinates():
    h0 = _checkpoint_h0_rows(_endpoints([("none", np.nan), ("always_truth", 12), ("always_truth", 3)]))
    assert len(h0) == 4 and (h0["post_branch_horizon"] == 0).all()
    assert (h0["branch_policy"] == "checkpoint").all() and h0["posting_budget"].isna().all()
    assert (h0["target_count"] == h0["n_0"]).all() and np.allclose(h0["target_fraction"], h0["n_0"] / 10)


def test_h0_rows_do_not_depend_on_branch_row_order():
    a = _checkpoint_h0_rows(_endpoints([("none", np.nan), ("always_truth", 12), ("always_truth", 3)]))
    b = _checkpoint_h0_rows(_endpoints([("always_truth", 3), ("always_truth", 12), ("none", np.nan)]))
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True)
