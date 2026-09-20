"""The calibration engine must produce identical tables serial and in a process pool."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mas_cc.analysis import blackboard_calibration as bc


def _synthetic_inputs(seed: int = 7, cells: int = 2, budgets=(3, 12), episodes: int = 6, rounds: int = 4) -> pd.DataFrame:
    """Rows in the adapter's output schema: enough structure for every estimator branch."""
    rng = np.random.default_rng(seed)
    records = []
    for cell in range(cells):
        cell_id = f"cell-{cell:04d}"
        for budget in budgets:
            for episode in range(episodes):
                episode_id = f"{cell_id}-b{budget}-{episode:04d}"
                block = f"block-{episode % 3}"
                for round_index in range(rounds):
                    m = 3
                    for slot in range(m):
                        action = int(rng.integers(0, 2))
                        exposure = int(rng.integers(0, 2)) if action == 1 else 0
                        before = int(rng.integers(0, 2))
                        after = before if rng.random() < 0.6 else 1 - before
                        records.append(
                            dict(
                                study_id="synthetic", source_run_id="run-0",
                                cell_id=cell_id, episode_id=episode_id, round_index=round_index,
                                micro_slot_index=slot, block_id=block, task_id="task_003",
                                focal_agent_id=f"agent_{slot:03d}", U=action, E=exposure,
                                z_before=before, z_after=after, valid_update=True,
                                b_budget=budget, b_posted=int(rng.integers(0, budget + 1)),
                                B=8, C=2, m=m, sampling_probability=0.25, sampling_status="exact",
                                N=24, x=float(rng.random()), x_after=float(rng.random()),
                                actual_update_count=m, propensity=0.5,
                                focal_selection_rule="uniform", adapter_version=bc.ADAPTER_VERSION,
                                source_selection="completed_canonical_episodes", M=m,
                            )
                        )
    return pd.DataFrame(records)


@pytest.fixture
def patched_inputs(monkeypatch):
    frame = _synthetic_inputs()
    monkeypatch.setattr(bc, "adapt_calibration_inputs", lambda *args, **kwargs: frame.copy())
    return frame


def _run(workers: int, progress=None):
    empty = pd.DataFrame()
    return bc.analyze_blackboard_calibration(
        empty, empty, empty, empty,
        settings={"enabled": True},
        bootstrap_resamples=25, confidence=0.9, seed=11, analysis_hash="test",
        provisional=False, progress=progress, workers=workers,
    )


def test_parallel_matches_serial(patched_inputs):
    serial = _run(1)
    parallel = _run(2)
    assert set(serial) == set(parallel)
    for name in serial:
        pd.testing.assert_frame_equal(serial[name], parallel[name], check_like=False)
    estimates = serial["blackboard_calibration_estimates"]
    assert not estimates.empty
    assert estimates["cell_id"].nunique() == 2


def test_progress_reports_group_completion_in_pool(patched_inputs):
    events = []
    _run(2, progress=events.append)
    completed = [e for e in events if e.get("stage") == "blackboard_calibration_bootstrap"]
    assert completed, "pool mode must still report progress per group"
    assert completed[-1]["groups_completed"] == completed[-1]["groups_total"] == 4


def test_workers_must_be_positive(patched_inputs):
    with pytest.raises(ValueError):
        _run(0)
