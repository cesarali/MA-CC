"""The calibration input adapter gives the same frame per cell in a pool as serially."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from mas_cc.analysis import blackboard_calibration as bc


def _canonical(cells: int = 3, episodes: int = 2, rounds: int = 2, slots: int = 3, seed: int = 4):
    """Minimal canonical tables that satisfy every adapter check."""
    rng = np.random.default_rng(seed)
    cell_rows, episode_rows, round_rows, micro_rows = [], [], [], []
    options = ["ALLOCATION_0", "ALLOCATION_1", "ALLOCATION_2"]
    for c in range(cells):
        cell_id = f"cell-{c:04d}"
        cell_rows.append({"cell_id": cell_id, "study_id": "synthetic", "task_id": "task_003", "intervention_budget": 9 + 3 * c,
                          "population_size": 6, "focal_selection_rule": "uniform_with_replacement", "source_run_id": "run-0"})
        for e in range(episodes):
            episode_id = f"{cell_id}-{e:04d}"
            # one episode per cell is incomplete to exercise the skip path
            status = "completed" if not (c == 1 and e == 1) else "failed"
            episode_rows.append({"cell_id": cell_id, "episode_id": episode_id, "status": status})
            block = f"block-{e}"
            for r in range(rounds):
                before = [int(v) for v in rng.multinomial(6, [0.4, 0.3, 0.3])]
                after = [int(v) for v in rng.multinomial(6, [0.4, 0.3, 0.3])]
                round_rows.append({
                    "cell_id": cell_id, "episode_id": episode_id, "round_index": r, "social_mode": "board",
                    "analysis_target": "ALLOCATION_2", "U_k": int(rng.integers(0, 2)),
                    "physical_initial_state_hash": block, "initialization_source": "paired_artifact",
                    "occupation_counts_before": json.dumps(before), "occupation_counts_after": json.dumps(after),
                    "possible_answers": json.dumps(options), "board_sampling": "uniform",
                    "actual_update_count": slots, "P_U1_given_Y": 0.5, "actual_controller_posts": int(rng.integers(0, 3)),
                    "controller_target_share_before": before[2] / 6, "controller_target_share": after[2] / 6,
                })
                for s in range(slots):
                    exposed = bool(rng.random() < 0.5)
                    missing = c == 2 and s == 0  # a slot with unknown composition
                    micro_rows.append({
                        "cell_id": cell_id, "episode_id": episode_id, "round_index": r, "micro_slot_index": s,
                        "focal_agent_id": f"agent_{s:03d}", "focal_opinion_before": options[int(rng.integers(0, 3))],
                        "focal_opinion_after": options[int(rng.integers(0, 3))], "valid_update": True,
                        "sampled_controller_message_ids": json.dumps(["c1"] if exposed else []),
                        "sampled_message_ids": json.dumps(["c1", "p1", "p2"] if exposed else ["p1", "p2", "p3"]),
                        "controller_message_directly_exposed": False,
                        "eligible_peer_message_count": None if missing else 5,
                        "eligible_controller_message_count": None if missing else 1,
                        "board_sample_size": None if missing else 3,
                        "analysis_target": "ALLOCATION_2",
                    })
    return (pd.DataFrame(micro_rows), pd.DataFrame(round_rows), pd.DataFrame(episode_rows), pd.DataFrame(cell_rows))


def test_parallel_adapter_matches_serial_including_dtypes_and_order():
    micro, rounds, episodes, cells = _canonical()
    serial = bc.adapt_calibration_inputs(micro, rounds, episodes, cells, workers=1)
    pooled = bc.adapt_calibration_inputs(micro, rounds, episodes, cells, workers=2)
    assert not serial.empty and len(serial) == 3 * 2 * 2 * 3 - 2 * 3  # one failed episode skipped
    pd.testing.assert_frame_equal(serial, pooled, check_like=False)
    assert list(serial.dtypes) == list(pooled.dtypes)
    # rows come back in micro order across cells even though cells were adapted separately
    assert serial["cell_id"].tolist() == pooled["cell_id"].tolist()


def test_parallel_adapter_keeps_interleaved_cell_order():
    micro, rounds, episodes, cells = _canonical()
    shuffled = micro.sample(frac=1.0, random_state=9).reset_index(drop=True)
    serial = bc.adapt_calibration_inputs(shuffled, rounds, episodes, cells, workers=1)
    pooled = bc.adapt_calibration_inputs(shuffled, rounds, episodes, cells, workers=3)
    pd.testing.assert_frame_equal(serial, pooled)


def test_identity_checks_still_raise_in_pool_mode():
    micro, rounds, episodes, cells = _canonical()
    duplicated = pd.concat([micro, micro.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError):
        bc.adapt_calibration_inputs(duplicated, rounds, episodes, cells, workers=2)


@pytest.mark.parametrize("value", [None, float("nan"), np.nan, np.float64("nan"), 0.0, 1, np.int64(3), True, "", "x",
                                   pd.NA, pd.NaT, np.float32(2.0), [1, 2]])
def test_present_fast_path_matches_pandas(value):
    if isinstance(value, list):
        expected = True  # pd.isna on a list is elementwise; the helper treats containers as present
    else:
        missing = pd.isna(value)
        expected = not bool(missing)
    assert bc._present(value) is expected
