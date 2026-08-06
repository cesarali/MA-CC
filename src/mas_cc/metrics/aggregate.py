"""The cell and grid tiers of the metric stack, and the rules they aggregate by.

`StreamingMetric` produces one value per round, `FinalMetric` one value per
episode.  Neither can answer "what did this grid cell do", because that
question spans episodes.  This module adds the two tiers that do:

| Tier              | Scope                          |
|-------------------|--------------------------------|
| `StreamingMetric` | one round                      |
| `FinalMetric`     | one episode                    |
| `AggregateMetric` | one **cell** (many episodes)   |
| `SweepMetric`     | the **grid** (many cells)      |

Both new tiers are *post-hoc*: they consume what a finished episode already
wrote to disk (`EpisodeFrame`), never live game state.  That is what makes
aggregates recomputable — percentiles, relabeling and smoothing can be changed
after the fact without re-running a single episode — and what makes a job
killed at 80% still yield complete, correct aggregates for every finished cell.

`AggregateResult` carries three things, and the third is the load-bearing one:
curves (for looking at), scalars (for the sweep dashboard), and **counts** —
the sufficient statistics a `SweepMetric` needs.  Grid-level mutual information
is computed from those counts, so a `SweepMetric` never re-reads an episode and
a cell's contribution to the grid survives the cell's own completion.

The aggregation rules in `AggregationPolicy` are correctness requirements, not
preferences; each one's docstring says which bias it removes.
"""

from __future__ import annotations

import csv
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

CellId = str

FORWARD_FILL_MODES = ("absorbing", "truncate", "none")
"""How to reconcile episodes that ended at different rounds; see `align`."""


# --- what a finished episode looks like on the way back in -------------------


@dataclass(frozen=True, slots=True)
class EpisodeFrame:
    """One completed episode's recorded series, read back from disk.

    `population` and `options` mirror the two non-agent metric scopes of
    `metrics/streaming.csv`; per-agent rows are dropped, because a cell-level
    aggregate over individual agents of *different episodes* compares agents
    that were never in the same population.

    Non-numeric values (a metric that reports the winning *word* rather than a
    share) are dropped on read: everything in this tier averages, and there is
    no meaningful percentile of a string.
    """

    episode_id: str
    rounds: tuple[int, ...] = ()
    population: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    options: Mapping[str, Mapping[str, tuple[float, ...]]] = field(default_factory=dict)
    final: Mapping[str, float | None] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.rounds)

    def option_series(self, metric_name: str) -> Mapping[str, tuple[float, ...]]:
        return self.options.get(metric_name, {})


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def read_episode_frame(episode_dir: str | Path) -> EpisodeFrame | None:
    """Read one episode directory, or ``None`` if it recorded no round metrics.

    An episode with no `metrics/streaming.csv` is not an error: a game without
    an observer-aware runtime runs fine and simply has nothing to aggregate.
    Returning ``None`` lets the caller skip it rather than fabricate an empty
    episode that would drag every percentile toward zero.
    """

    directory = Path(episode_dir)
    streaming_path = directory / "metrics" / "streaming.csv"
    if not streaming_path.exists():
        return None

    population: dict[str, dict[int, float]] = {}
    options: dict[str, dict[str, dict[int, float]]] = {}
    rounds: set[int] = set()
    with streaming_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("agent_id"):
                continue
            value = _number(row.get("value", ""))
            if value is None:
                continue
            round_index = int(row["round_index"])
            rounds.add(round_index)
            name, series = row["metric_name"], row.get("series") or ""
            if series:
                options.setdefault(name, {}).setdefault(series, {})[round_index] = value
            else:
                population.setdefault(name, {})[round_index] = value

    if not rounds:
        return None
    ordered = tuple(sorted(rounds))

    def densify(points: Mapping[int, float]) -> tuple[float, ...]:
        # A metric that skipped a round leaves a hole; carrying the previous
        # value across it keeps every series the same length as `ordered`, so
        # position i means round `ordered[i]` in all of them.
        values: list[float] = []
        previous = math.nan
        for round_index in ordered:
            previous = points.get(round_index, previous)
            values.append(previous)
        return tuple(values)

    final: dict[str, float | None] = {}
    final_path = directory / "metrics" / "final.csv"
    if final_path.exists():
        with final_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                final[row["metric_name"]] = _number(row.get("value", ""))

    return EpisodeFrame(
        episode_id=directory.name,
        rounds=ordered,
        population={name: densify(points) for name, points in population.items()},
        options={
            name: {series: densify(points) for series, points in by_series.items()}
            for name, by_series in options.items()
        },
        final=final,
    )


def read_cell_episodes(cell_dir: str | Path) -> tuple[EpisodeFrame, ...]:
    """Every completed episode under one grid cell, in stable episode-id order."""

    episodes_dir = Path(cell_dir) / "data" / "episodes"
    if not episodes_dir.is_dir():
        return ()
    frames = (
        read_episode_frame(path) for path in sorted(episodes_dir.iterdir()) if path.is_dir()
    )
    return tuple(frame for frame in frames if frame is not None)


# --- results -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Curve:
    """One aggregated series: ``round -> one value per level``.

    `levels` names what the tuple positions mean, so a percentile band
    (``("p10", "p50", "p90")``) and a plain single-valued curve
    (``("value",)``) are the same type.  Without it a reader cannot tell a
    three-level band from three unrelated curves that happen to share a name.
    """

    levels: tuple[str, ...]
    points: Mapping[int, tuple[float, ...]] = field(default_factory=dict)

    def level(self, name: str) -> list[tuple[int, float]]:
        index = self.levels.index(name)
        return [(round_index, values[index]) for round_index, values in sorted(self.points.items())]

    def to_dict(self) -> dict[str, Any]:
        return {
            "levels": list(self.levels),
            "points": {str(k): list(v) for k, v in sorted(self.points.items())},
        }


@dataclass(frozen=True, slots=True)
class AggregateResult:
    """What one grid cell amounts to: curves, scalars, and sufficient statistics.

    `counts` is what lets `SweepMetric` work without re-reading episodes: a
    contingency table the cell computed once (``table name -> key -> count``),
    which a grid-level estimator can sum across cells.  Keeping them here
    rather than recomputing from episodes is the difference between a sweep
    metric that is cheap enough to run on every cell completion and one that
    rescans the whole filesystem each time.
    """

    curves: Mapping[str, Curve] = field(default_factory=dict)
    scalars: Mapping[str, float] = field(default_factory=dict)
    counts: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    episodes: int = 0

    def merged_with(self, other: "AggregateResult") -> "AggregateResult":
        """Combine two metrics' contributions to the same cell.

        Later keys win on collision — metrics are applied in the configured
        order, so a deliberate override reads the way the config does.
        """

        return AggregateResult(
            curves={**self.curves, **other.curves},
            scalars={**self.scalars, **other.scalars},
            counts={**self.counts, **other.counts},
            episodes=max(self.episodes, other.episodes),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "episodes": self.episodes,
            "scalars": dict(sorted(self.scalars.items())),
            "counts": {k: dict(sorted(v.items())) for k, v in sorted(self.counts.items())},
            "curves": {name: curve.to_dict() for name, curve in sorted(self.curves.items())},
        }


@dataclass(frozen=True, slots=True)
class SweepResult:
    """What the grid amounts to so far: scalars only.

    Deliberately not curves.  A grid-level quantity is one number per statistic
    (an MI estimate, its null band, its gap to ground truth); it becomes a
    *series* only through the sweep experiment's step axis, which is the
    number of episodes done, and that belongs to the logger rather than here.
    """

    scalars: Mapping[str, float] = field(default_factory=dict)

    def merged_with(self, other: "SweepResult") -> "SweepResult":
        return SweepResult(scalars={**self.scalars, **other.scalars})

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "scalars": dict(sorted(self.scalars.items()))}


class AggregateMetric(ABC):
    """Cell-level.  Consumes every completed episode in one grid cell."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def compute(
        self, episodes: Sequence[EpisodeFrame], policy: "AggregationPolicy"
    ) -> AggregateResult:
        """Aggregate one cell.  Must tolerate an empty or ragged episode list."""


class SweepMetric(ABC):
    """Grid-level.  Consumes the `AggregateResult` of every completed cell."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def compute(self, cells: Mapping[CellId, AggregateResult]) -> SweepResult:
        """Estimate over the cells finished so far; partial grids are normal."""


# --- the aggregation rules (spec section 4) ----------------------------------


@dataclass(frozen=True, slots=True)
class AggregationPolicy:
    """How episodes within a cell are reconciled before being averaged.

    Every field removes a specific bias.  They are not tuning knobs:

    ``forward_fill``      absorbing end conditions bias the mid-run mean (4.1)
    ``relabel_by_winner`` symmetric outcomes cancel to a flat 0.5 (4.2)
    ``percentiles``       a bounded bimodal share is misdescribed by +/-std (4.3)
    ``rolling_window``    smoothing after aggregation is a different quantity (4.4)
    """

    forward_fill: str = "absorbing"
    relabel_by_winner: bool = True
    percentiles: tuple[int, ...] = (10, 50, 90)
    rolling_window: int = 20

    def __post_init__(self) -> None:
        if self.forward_fill not in FORWARD_FILL_MODES:
            raise ValueError(
                f"forward_fill must be one of {FORWARD_FILL_MODES}, got {self.forward_fill!r}"
            )
        object.__setattr__(self, "percentiles", tuple(int(p) for p in self.percentiles))
        if not self.percentiles:
            raise ValueError("percentiles must list at least one percentile")
        if any(not 0 <= p <= 100 for p in self.percentiles):
            raise ValueError("percentiles must lie in [0, 100]")
        if self.rolling_window < 1:
            raise ValueError("rolling_window must be a positive number of rounds")

    @property
    def levels(self) -> tuple[str, ...]:
        return tuple(f"p{p}" for p in self.percentiles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "forward_fill": self.forward_fill,
            "relabel_by_winner": self.relabel_by_winner,
            "percentiles": list(self.percentiles),
            "rolling_window": self.rolling_window,
        }


def rolling_within(values: Sequence[float], window: int) -> tuple[float, ...]:
    """Trailing-mean smoothing applied *inside* one episode (4.4).

    Rolling first and aggregating second is not the same operation as
    aggregating first and rolling second, and only the first order answers
    "what does a typical run look like".  The window is trailing with a partial
    head (``min_periods=1``) so the curve starts at round 1 rather than at
    round `window`, which would silently crop the most interesting stretch.
    """

    if window <= 1:
        return tuple(float(v) for v in values)
    output: list[float] = []
    total = 0.0
    for index, value in enumerate(values):
        total += float(value)
        if index >= window:
            total -= float(values[index - window])
        output.append(total / min(index + 1, window))
    return tuple(output)


def align(
    series: Sequence[Sequence[float]], mode: str = "absorbing"
) -> tuple[tuple[float | None, ...], ...]:
    """Put every episode's series on a common round axis (4.1).

    Episodes end at different rounds because consensus is absorbing.  Averaging
    at round *t* over only the episodes still running conditions on "hasn't
    converged yet" — a biased subset of slow, near-50/50 runs — which puts a
    spurious dip in the middle of every curve.

    ``absorbing``  pad each finished episode forward with its terminal value.
                   Legitimate precisely because the end condition is absorbing:
                   the value genuinely persists.
    ``truncate``   cut every episode to the shortest one.  Unbiased, but throws
                   away the tail that usually carries the finding.
    ``none``       pad with ``None``, which the percentile step drops.  This is
                   the biased option, kept only so the bias can be demonstrated.
    """

    if mode not in FORWARD_FILL_MODES:
        raise ValueError(f"forward_fill must be one of {FORWARD_FILL_MODES}, got {mode!r}")
    lengths = [len(item) for item in series]
    if not lengths:
        return ()
    if mode == "truncate":
        shortest = min(lengths)
        return tuple(tuple(float(v) for v in item[:shortest]) for item in series)
    longest = max(lengths)
    aligned: list[tuple[float | None, ...]] = []
    for item in series:
        values: tuple[float | None, ...] = tuple(float(v) for v in item)
        pad = values[-1] if (mode == "absorbing" and values) else None
        aligned.append(values + (pad,) * (longest - len(values)))
    return tuple(aligned)


def percentile(values: Sequence[float], percent: float) -> float:
    """Linear-interpolated percentile, matching NumPy's default method.

    Written out rather than imported so this tier stays standard-library only:
    the master reads it on every cell completion, and pulling NumPy into that
    path would make the liveness logger depend on the analysis stack.
    """

    ordered = sorted(float(v) for v in values)
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percent / 100.0)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def band_curve(
    series: Sequence[Sequence[float]], policy: AggregationPolicy
) -> Curve:
    """Roll within each episode, align, then take percentiles across episodes (4.3).

    Percentiles rather than mean +/- std because the quantity is bounded in
    [0,1] and bimodal for much of a run: a standard-deviation band leaves the
    unit interval and draws two humps as one wide one.  The band answers
    *dispersion across runs*, not uncertainty in the mean.
    """

    smoothed = [rolling_within(item, policy.rolling_window) for item in series]
    aligned = align(smoothed, policy.forward_fill)
    if not aligned:
        return Curve(levels=policy.levels)
    points: dict[int, tuple[float, ...]] = {}
    for index in range(len(aligned[0])):
        # `None` marks a padded round under `forward_fill: none`; NaN marks a
        # round a metric never recorded. Both mean "no measurement here", and
        # including either would silently poison the whole band rather than
        # narrowing it.
        column = [
            value
            for value in (row[index] for row in aligned)
            if value is not None and not math.isnan(value)
        ]
        points[index + 1] = tuple(percentile(column, p) for p in policy.percentiles)
    return Curve(levels=policy.levels, points=points)


def active_fraction_curve(lengths: Sequence[int]) -> Curve:
    """``n_active(t) / n_total`` — always logged beside a forward-filled band.

    Without it a tight band late in a run is unreadable: it may mean the
    episodes agree, or it may mean they are all padding.
    """

    if not lengths:
        return Curve(levels=("value",))
    longest = max(lengths)
    total = float(len(lengths))
    return Curve(
        levels=("value",),
        points={
            round_index: (sum(1 for item in lengths if item >= round_index) / total,)
            for round_index in range(1, longest + 1)
        },
    )


def winner_ranking(shares: Mapping[str, Sequence[float]]) -> tuple[str, ...]:
    """Option labels ordered by their terminal share, winner first (4.2).

    Averaging a per-option share across episodes washes out: roughly half the
    episodes converge to Q and half to M, so the mean for Q converges to 0.5
    with growing variance and looks like nothing happened.  Relabeling each
    episode so its eventual winner is option 1 fixes that while *keeping the
    loser's trajectory*, so the asymmetry of the approach stays visible —
    which aggregating the dominant share alone would discard.

    Ties break on the option label, so a game with no winner (synthetic Game 1)
    relabels deterministically and its relabelled curves stay symmetric away
    from the terminal round.  If they don't, the relabel is wrong.
    """

    return tuple(
        sorted(shares, key=lambda option: (-(shares[option][-1] if shares[option] else 0.0), option))
    )


def aggregate_cell(
    episodes: Sequence[EpisodeFrame],
    metrics: Sequence[AggregateMetric],
    policy: AggregationPolicy,
) -> AggregateResult:
    """Run every configured cell metric over one cell's episodes and merge them."""

    result = AggregateResult(episodes=len(episodes))
    for metric in metrics:
        result = result.merged_with(metric.compute(episodes, policy))
    return AggregateResult(
        curves=result.curves, scalars=result.scalars, counts=result.counts,
        episodes=len(episodes),
    )


def sweep_over_cells(
    cells: Mapping[CellId, AggregateResult], metrics: Sequence[SweepMetric]
) -> SweepResult:
    """Run every configured sweep metric over the cells finished so far."""

    result = SweepResult()
    for metric in metrics:
        result = result.merged_with(metric.compute(cells))
    return result


__all__ = [
    "FORWARD_FILL_MODES",
    "AggregateMetric",
    "AggregateResult",
    "AggregationPolicy",
    "CellId",
    "Curve",
    "EpisodeFrame",
    "SweepMetric",
    "SweepResult",
    "active_fraction_curve",
    "aggregate_cell",
    "align",
    "band_curve",
    "percentile",
    "read_cell_episodes",
    "read_episode_frame",
    "rolling_within",
    "sweep_over_cells",
    "winner_ranking",
]
