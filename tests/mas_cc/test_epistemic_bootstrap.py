"""Equivalence to literal shared-block resampling, including sparse draws."""

import numpy as np
import pandas as pd
import pytest

from mas_cc.analysis.epistemic_phase import _BlockBootstrap


def literal_draws(frame, resamples, seed):
    blocks = sorted(frame.initialization_block_id.astype(str).unique())
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        yield pd.concat(
            [
                frame[frame.initialization_block_id.astype(str) == block]
                for block in rng.choice(blocks, len(blocks), replace=True)
            ],
            ignore_index=True,
        )


def observations():
    rng = np.random.default_rng(100)
    frame = pd.DataFrame(
        {
            "initialization_block_id": np.repeat(["a", "b", "c", "d"], [1, 7, 3, 20]),
            "cell_id": ["rare"] + ["common"] * 30,
            "score": rng.normal(size=31),
            "other_score": rng.normal(size=31),
            "x": rng.random(31),
            "phi": rng.random(31),
            "round": rng.integers(1, 20, 31).astype(float),
        }
    )
    frame.loc[[2, 4, 8], "score"] = np.nan
    # Include a block shared across cells, with unequal contributions.
    frame.loc[7:9, "cell_id"] = "shared"
    return frame


@pytest.mark.parametrize("resamples", [0, 1, 137])
def test_means_match_literal_block_draws(resamples):
    frame = observations()
    frame["missing"] = np.nan
    columns = ["score", "other_score", "missing"]
    bootstrap = _BlockBootstrap(frame, resamples, 12)
    draws = list(literal_draws(frame, resamples, 12))
    for cell, group in frame.groupby("cell_id"):
        actual = bootstrap.means(group, group[columns].to_numpy())
        expected = np.array(
            [draw[draw.cell_id == cell][columns].mean().to_numpy() for draw in draws]
        ).reshape(resamples, len(columns))
        np.testing.assert_allclose(
            actual, expected, rtol=1e-12, atol=1e-14, equal_nan=True
        )


@pytest.mark.parametrize("design_kind", ["full_rank", "singular", "ill_conditioned"])
def test_regression_matches_literal_draws(design_kind):
    frame = observations()
    if design_kind == "singular":
        frame["phi"] = frame["x"]
    elif design_kind == "ill_conditioned":
        frame["phi"] = frame["x"] + frame["phi"] * 1e-10
    frame["intercept"] = 1.0
    columns = ["intercept", "x", "phi", "round"]
    bootstrap = _BlockBootstrap(frame, 137, 12)
    draws = list(literal_draws(frame, 137, 12))
    for cell, group in frame.groupby("cell_id"):
        actual = bootstrap.regression(
            group, group[columns].to_numpy(), group.other_score.to_numpy()
        )
        expected = []
        for draw in draws:
            subset = draw[draw.cell_id == cell]
            if subset.empty:
                continue
            x = subset[columns].to_numpy()
            if np.linalg.matrix_rank(x) == 4 and np.linalg.cond(x) <= 1e8:
                expected.append(
                    np.linalg.lstsq(x, subset.other_score, rcond=None)[0][2] * 0.1
                )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_batches_are_bounded_and_replay_global_draws():
    frame = observations()
    bootstrap = _BlockBootstrap(frame, 137, 12)
    batches = list(bootstrap.batches())
    assert max(len(batch) for batch in batches) <= 64
    counts = np.concatenate(batches)
    for weights, draw in zip(counts, literal_draws(frame, 137, 12), strict=True):
        original_sizes = frame.groupby("initialization_block_id").size()
        drawn_sizes = (
            draw.groupby("initialization_block_id")
            .size()
            .reindex(original_sizes.index, fill_value=0)
        )
        np.testing.assert_array_equal(weights, drawn_sizes / original_sizes)


@pytest.mark.parametrize("batch_size", [1, 7, 31])
def test_estimates_are_independent_of_draw_batch_size(monkeypatch, batch_size):
    frame = observations()
    frame["intercept"] = 1.0
    x = frame[["intercept", "x", "phi", "round"]].to_numpy()
    y = frame.other_score.to_numpy()
    plan = _BlockBootstrap(frame, 137, 12)
    expected_means = plan.means(frame, frame[["score", "other_score"]].to_numpy())
    expected_regression = plan.regression(frame, x, y)
    original = _BlockBootstrap.batches

    def smaller_batches(self):
        for batch in original(self):
            for start in range(0, len(batch), batch_size):
                yield batch[start:start + batch_size]

    monkeypatch.setattr(_BlockBootstrap, "batches", smaller_batches)
    np.testing.assert_allclose(plan.means(frame, frame[["score", "other_score"]].to_numpy()),
                               expected_means, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(plan.regression(frame, x, y), expected_regression,
                               rtol=1e-10, atol=1e-12)
