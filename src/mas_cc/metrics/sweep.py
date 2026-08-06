"""The grid-level metrics: what the sweep as a whole says, so far.

Every one of these reads only `AggregateResult.counts` — the contingency
tables each cell computed once, at its own completion.  Nothing here opens an
episode file.  Two consequences, both intended:

- A sweep metric is cheap enough to re-run on *every* cell completion, so a
  partial MI estimate appears and tightens as the grid fills in, instead of
  arriving only at the end of a day-long job.
- The estimate depends on finished cells only.  A grid killed at 80% yields the
  correct MI for the 80% that finished, rather than nothing.

These estimate the same quantities as `mas_cc.analysis.pipeline`, from the same
estimators, and must agree with it: the pipeline is the offline, bootstrap-CI
version run over a completed grid; this is the live one.  Where they could
diverge — the macrostate tie rule — the cell tier deliberately matches
`analysis/reader.py` (see `cell._macrostate_series`).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from mas_cc.metrics.aggregate import AggregateResult, CellId, SweepMetric, SweepResult

TERMINAL_TABLE = "terminal_outcome"


def _levels(cells: Mapping[CellId, AggregateResult], table: str) -> tuple[str, ...]:
    """Every key seen in one count table, in stable order.

    A cell that never saw an outcome still needs a zero column for it, or the
    alphabet would change as the grid fills and the MI would move for reasons
    that have nothing to do with the dynamics.
    """

    keys: set[str] = set()
    for result in cells.values():
        keys.update(result.counts.get(table, {}))
    return tuple(sorted(keys))


def _terminal_table(cells: Mapping[CellId, AggregateResult]) -> Any:
    """``(cell, outcome)`` count matrix, or ``None`` when there is nothing to estimate."""

    import numpy as np

    outcomes = _levels(cells, TERMINAL_TABLE)
    populated = [cid for cid in sorted(cells) if cells[cid].counts.get(TERMINAL_TABLE)]
    if len(populated) < 2 or len(outcomes) < 2:
        # One condition level carries no information by construction, and a
        # single outcome level has no entropy to share; both are "not yet",
        # not "zero", so they are reported as NaN rather than as 0 bits.
        return None
    matrix = np.zeros((len(populated), len(outcomes)), dtype=float)
    for row, cell_id in enumerate(populated):
        table = cells[cell_id].counts[TERMINAL_TABLE]
        for column, outcome in enumerate(outcomes):
            matrix[row, column] = float(table.get(outcome, 0.0))
    return matrix


class TerminalMutualInformation(SweepMetric):
    """``I(cell ; terminal outcome)`` over the cells finished so far.

    The grid *is* the channel input: which cells ran, and how many episodes
    each got, define ``p(c)``.  Reading the design off the actual per-cell
    episode counts rather than assuming a uniform grid is what keeps this
    honest on a resumed or partially failed sweep, where the cells are
    genuinely unbalanced.
    """

    def __init__(self, *, name: str = "terminal_mi") -> None:
        super().__init__(name)

    def compute(self, cells: Mapping[CellId, AggregateResult]) -> SweepResult:
        from mas_cc.analysis.estimators import mutual_information_from_counts

        matrix = _terminal_table(cells)
        if matrix is None:
            return SweepResult(scalars={"terminal_mi_estimate": math.nan})
        estimate = mutual_information_from_counts(matrix)
        return SweepResult(
            scalars={
                "terminal_mi_estimate": estimate.jeffreys,
                "terminal_mi_unsmoothed": estimate.unsmoothed,
                "terminal_mi_miller_madow": estimate.miller_madow,
                "terminal_mi_episodes": float(estimate.observations),
            }
        )


class LaggedConditionalMutualInformation(SweepMetric):
    """``I(cell ; S_{t+h} | S_t)`` at each horizon, from the cells' transition tables.

    Conditioning on the current macrostate is the point: without it the
    statistic mostly measures that the population is where it already was.
    Degrees of freedom grow as ``|S|(|C|-1)(|S|-1)``, so the plug-in bias here
    can exceed the effect at a fine macrostate — which is why the unsmoothed
    estimate is reported beside the smoothed one rather than hidden.
    """

    def __init__(self, horizons: Sequence[int] = (1,), *, name: str = "lagged_cmi") -> None:
        super().__init__(name)
        self.horizons = tuple(int(h) for h in horizons)

    def compute(self, cells: Mapping[CellId, AggregateResult]) -> SweepResult:
        import numpy as np

        from mas_cc.analysis.estimators import conditional_mutual_information_from_counts

        scalars: dict[str, float] = {}
        for horizon in self.horizons:
            table_name = f"macrostate_transition_h{horizon}"
            populated = [cid for cid in sorted(cells) if cells[cid].counts.get(table_name)]
            states = sorted(
                {
                    part
                    for key in _levels(cells, table_name)
                    for part in key.split(">", 1)
                }
            )
            if len(populated) < 2 or len(states) < 2:
                scalars[f"lagged_cmi_h{horizon}_estimate"] = math.nan
                continue
            index = {state: position for position, state in enumerate(states)}
            counts = np.zeros((len(populated), len(states), len(states)), dtype=float)
            for row, cell_id in enumerate(populated):
                for key, value in cells[cell_id].counts[table_name].items():
                    current, future = key.split(">", 1)
                    counts[row, index[current], index[future]] += float(value)
            estimate = conditional_mutual_information_from_counts(counts)
            scalars[f"lagged_cmi_h{horizon}_estimate"] = estimate.jeffreys
            scalars[f"lagged_cmi_h{horizon}_unsmoothed"] = estimate.unsmoothed
        return SweepResult(scalars=scalars)


class MutualInformationNullBand(SweepMetric):
    """The label-shuffle null for the terminal statistic: how much MI is noise.

    A plug-in MI is positive in expectation even when the condition does
    nothing — with two cells and a hundred episodes the floor is small but not
    zero, and it *grows* with the alphabet.  Reporting an estimate without this
    band invites reading finite-sample bias as an effect.

    The null is built from the same per-cell counts: pool every episode's
    terminal outcome, reshuffle it across cells of unchanged sizes, and
    re-estimate.  That destroys the condition/outcome association while holding
    both marginals fixed, which is exactly the hypothesis being tested.
    """

    def __init__(
        self, *, permutations: int = 200, seed: int = 1, name: str = "mi_null_band"
    ) -> None:
        super().__init__(name)
        self.permutations = int(permutations)
        self.seed = int(seed)

    def compute(self, cells: Mapping[CellId, AggregateResult]) -> SweepResult:
        import numpy as np

        from mas_cc.analysis.estimators import mutual_information_from_counts

        matrix = _terminal_table(cells)
        if matrix is None or self.permutations < 1:
            return SweepResult(scalars={"terminal_mi_null_p95": math.nan})
        sizes = matrix.sum(axis=1).astype(int)
        pooled = np.repeat(np.arange(matrix.shape[1]), matrix.sum(axis=0).astype(int))
        rng = np.random.default_rng(self.seed)
        draws = np.empty(self.permutations, dtype=float)
        for draw in range(self.permutations):
            shuffled = rng.permutation(pooled)
            table = np.zeros_like(matrix)
            start = 0
            for row, size in enumerate(sizes):
                for outcome in shuffled[start : start + size]:
                    table[row, outcome] += 1.0
                start += size
            draws[draw] = mutual_information_from_counts(table).jeffreys
        finite = draws[np.isfinite(draws)]
        if finite.size == 0:
            return SweepResult(scalars={"terminal_mi_null_p95": math.nan})
        return SweepResult(
            scalars={
                "terminal_mi_null_p95": float(np.quantile(finite, 0.95)),
                "terminal_mi_null_mean": float(finite.mean()),
            }
        )


class MutualInformationGroundTruthGap(SweepMetric):
    """Estimate minus the closed-form answer, where one exists.

    Only a synthetic game has a ground truth, and it is a property of the
    *sweep* — which cells ran and with how many episodes — not of any single
    config, so it is passed in by whoever resolved the grid rather than
    recomputed here.  Given ``None`` the metric emits nothing: a missing
    answer key must not be reported as a zero gap.
    """

    def __init__(self, ground_truth: float | None = None, *, name: str = "mi_ground_truth_gap") -> None:
        super().__init__(name)
        self.ground_truth = ground_truth

    def compute(self, cells: Mapping[CellId, AggregateResult]) -> SweepResult:
        if self.ground_truth is None:
            return SweepResult()
        estimate = TerminalMutualInformation().compute(cells).scalars["terminal_mi_estimate"]
        return SweepResult(
            scalars={
                "terminal_mi_ground_truth": float(self.ground_truth),
                "terminal_mi_gap": estimate - float(self.ground_truth),
            }
        )


SWEEP_METRICS = {
    "terminal_mi": TerminalMutualInformation,
    "lagged_cmi": LaggedConditionalMutualInformation,
    "mi_null_band": MutualInformationNullBand,
    "mi_ground_truth_gap": MutualInformationGroundTruthGap,
}
"""Config name -> class, for resolving ``aggregation.sweep_metrics``."""


def create_sweep_metrics(
    names: Sequence[str],
    *,
    horizons: Sequence[int] = (1,),
    permutations: int = 200,
    seed: int = 1,
    ground_truth: float | None = None,
) -> tuple[SweepMetric, ...]:
    """Instantiate the configured sweep metrics, rejecting unknown names up front."""

    unknown = [name for name in names if name not in SWEEP_METRICS]
    if unknown:
        raise ValueError(
            f"unknown aggregation.sweep_metrics: {', '.join(unknown)}; "
            f"available: {', '.join(sorted(SWEEP_METRICS))}"
        )
    arguments: dict[str, dict[str, Any]] = {
        "lagged_cmi": {"horizons": tuple(horizons)},
        "mi_null_band": {"permutations": permutations, "seed": seed},
        "mi_ground_truth_gap": {"ground_truth": ground_truth},
    }
    return tuple(SWEEP_METRICS[name](**arguments.get(name, {})) for name in names)


__all__ = [
    "SWEEP_METRICS",
    "LaggedConditionalMutualInformation",
    "MutualInformationGroundTruthGap",
    "MutualInformationNullBand",
    "TerminalMutualInformation",
    "create_sweep_metrics",
]
