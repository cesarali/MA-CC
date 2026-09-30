"""Count reconstruction, validation and triplet formation on a tiny fabricated record table."""

import json

import numpy as np
import pandas as pd
import pytest

from bound_estimators import data as D


def _labels(n0, n1, n2):
    return json.dumps(["ALLOCATION_0"] * n0 + ["ALLOCATION_1"] * n1 + ["ALLOCATION_2"] * n2)


def _records(parents=("A", "B"), break_continuity=False, drop_branch=None):
    rows = []
    for pid in parents:
        for policy in ("none", "always_truth", "always_false"):
            if drop_branch == (pid, policy):
                continue
            budgets = [np.nan] if policy == "none" else [3.0]
            for b in budgets:
                counts = [(20, 2, 2)]
                for t in range(1, 11):
                    prev = counts[-1]
                    counts.append((max(prev[0] - 1, 0), prev[1] + 1, prev[2]))
                for h in range(1, 11):
                    before, after = counts[h - 1], counts[h]
                    if break_continuity and pid == "B" and h == 5 and policy == "none":
                        before = (0, 0, 24)
                    rows.append({"parent_id": pid, "checkpoint_hash": f"hash{pid}", "cell_id": "c", "q": 3, "rho": 0.7,
                                 "branch_policy": policy, "posting_budget": b, "copy_id": 1.0, "post_branch_horizon": float(h),
                                 "absolute_round": h + 2, "population_state_before": _labels(*before),
                                 "population_state_after": _labels(*after),
                                 "controller_target_semantic_id": None if policy == "none" else f"ALLOCATION_{D.TARGET_OF_POLICY[policy]}",
                                 "correct_answer_semantic_id": "ALLOCATION_0", "false_target_semantic_id": "ALLOCATION_2",
                                 "branch_status": "complete"})
    return pd.DataFrame(rows)


def test_labels_to_counts_validates():
    assert D.labels_to_counts(_labels(20, 2, 2)).tolist() == [20, 2, 2]
    with pytest.raises(ValueError):
        D.labels_to_counts(_labels(20, 2, 1))
    with pytest.raises(ValueError):
        D.labels_to_counts(json.dumps(["ALLOCATION_9"] * 24))


def test_paths_and_triplets_from_clean_records():
    paths, excl = D.build_paths(_records())
    assert len(paths) == 6 and not excl
    comps, audit = D.build_comparisons(paths, excl)
    assert set(comps) == {"q3_rho0.70_always_b3"}
    c = comps["q3_rho0.70_always_b3"]
    assert c.m == 2 and c.silent.shape == (2, 11, 3)
    assert c.silent[0, 0].tolist() == [20, 2, 2] and c.silent[0, 10].tolist() == [10, 12, 2]
    assert audit.comparison_counts["q3_rho0.70_sensing_b3"] == 0  # no sensing branches fabricated


def test_continuity_violation_is_excluded_not_repaired():
    paths, excl = D.build_paths(_records(break_continuity=True))
    assert len(paths) == 5 and len(excl) == 1 and "continuity" in excl[0]["reason"]
    comps, audit = D.build_comparisons(paths, excl)
    assert comps["q3_rho0.70_always_b3"].m == 1
    assert any(e.get("comparison") == "q3_rho0.70_always_b3" and "silent" in e["reason"] for e in audit.exclusions)


def test_missing_branch_drops_only_that_parent():
    paths, excl = D.build_paths(_records(drop_branch=("A", "always_false")))
    comps, audit = D.build_comparisons(paths, excl)
    assert comps["q3_rho0.70_always_b3"].parent_ids == ["B"]
