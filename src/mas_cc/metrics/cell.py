"""The cell-level metrics: what one grid cell's ~100 episodes amount to.

Each one is an `AggregateMetric`, computed once when the cell completes, from
the episode files the workers already wrote.  Nothing here talks to a provider,
a game object, or Comet; a cell can be re-aggregated from disk at any time and
must produce the identical result, which is what makes the curves recomputable
after a change of percentile, window, or fill rule.

All of them read the same two recorded quantities — the per-option standing
share (`population_action_share_per_option`) and, where present, the episode's
recorded consensus round — so a new game becomes aggregatable by recording
those, not by writing new aggregate metrics.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from mas_cc.metrics.aggregate import (
    AggregateMetric,
    AggregateResult,
    AggregationPolicy,
    EpisodeFrame,
    active_fraction_curve,
    band_curve,
    percentile,
    winner_ranking,
)

SHARE_METRIC = "population_action_share_per_option"
"""The option-scope metric every choice-family game records; see `metrics/generic.py`."""

CONSENSUS_FINAL_METRIC = "first_consensus_time_by_action_share"
"""The `FinalMetric` that already answers "when did this episode converge"."""


def _shares(frame: EpisodeFrame, metric: str = SHARE_METRIC) -> Mapping[str, tuple[float, ...]]:
    return frame.option_series(metric)


def _dominant_series(frame: EpisodeFrame, metric: str = SHARE_METRIC) -> tuple[float, ...]:
    """Per-round share of whichever option is largest *that round*.

    Derived from the per-option shares rather than read from the recorded
    `dominant_action_share` population metric, so a game that records only the
    per-option family still aggregates.  The two agree by construction.
    """

    shares = _shares(frame, metric)
    if not shares:
        return frame.population.get("dominant_action_share", ())
    columns = list(shares.values())
    return tuple(max(column[index] for column in columns) for index in range(len(columns[0])))


def _macrostate_series(frame: EpisodeFrame, metric: str = SHARE_METRIC) -> tuple[str, ...]:
    """Per-round dominant *option label* — the macrostate `analysis/reader.py` uses.

    Ties resolve to the first declared option, matching `reader.py`'s `idxmax`
    over rows written in declared-option order.  Any other tie rule would make
    the grid-level MI computed here disagree with the offline pipeline's.
    """

    shares = _shares(frame, metric)
    if not shares:
        return ()
    labels = list(shares)
    length = len(shares[labels[0]])
    # `max` returns the first maximal element in iteration order, and `labels`
    # is in declared-option order — which is precisely `idxmax` over rows
    # written in that order, so ties break the same way in both places.
    return tuple(
        max(labels, key=lambda label: shares[label][index]) for index in range(length)
    )


class DominantActionShare(AggregateMetric):
    """Band of the leading option's share — the curve that converges to 1.0.

    The unrelabelled per-option shares are the wrong thing to average (see
    `ActionShareRelabelled`); this is the always-safe alternative, because
    "how far along is the leader" is symmetric under which option wins.
    """

    def __init__(self, *, name: str = "dominant_action_share") -> None:
        super().__init__(name)

    def compute(
        self, episodes: Sequence[EpisodeFrame], policy: AggregationPolicy
    ) -> AggregateResult:
        series = [values for values in (_dominant_series(f) for f in episodes) if values]
        if not series:
            return AggregateResult()
        return AggregateResult(curves={self.name: band_curve(series, policy)})


class ActionShareRelabelled(AggregateMetric):
    """Per-option share bands after aligning every episode on its own winner.

    Emits one curve per *rank*, not per option: `<name>_option_1` is the share
    of whichever option that episode eventually won with, `_option_2` the
    runner-up, and so on.  This keeps the loser's trajectory — the asymmetry
    between the winning and losing approach is usually the finding — while
    removing the cancellation that makes the raw per-option mean flatline at
    0.5 in a game with no systematic winner.

    With ``relabel_by_winner`` off, the curves are the raw per-option shares
    under their real labels, so the washed-out version can still be looked at
    deliberately.
    """

    def __init__(self, *, name: str = "action_share_relabelled") -> None:
        super().__init__(name)

    def compute(
        self, episodes: Sequence[EpisodeFrame], policy: AggregationPolicy
    ) -> AggregateResult:
        by_key: dict[str, list[tuple[float, ...]]] = {}
        for frame in episodes:
            shares = _shares(frame)
            if not shares:
                continue
            if policy.relabel_by_winner:
                for rank, option in enumerate(winner_ranking(shares), start=1):
                    by_key.setdefault(f"option_{rank}", []).append(shares[option])
            else:
                for option, values in shares.items():
                    by_key.setdefault(option, []).append(values)
        if not by_key:
            return AggregateResult()
        return AggregateResult(
            curves={
                f"{self.name}_{key}": band_curve(series, policy)
                for key, series in sorted(by_key.items())
            }
        )


class ActiveFraction(AggregateMetric):
    """Fraction of the cell's episodes still running at each round.

    Always emitted alongside any forward-filled band: it is the only thing that
    distinguishes "these runs agree" from "these runs are padding".
    """

    def __init__(self, *, name: str = "active_fraction") -> None:
        super().__init__(name)

    def compute(
        self, episodes: Sequence[EpisodeFrame], policy: AggregationPolicy
    ) -> AggregateResult:
        lengths = [frame.length for frame in episodes if frame.length]
        if not lengths:
            return AggregateResult()
        return AggregateResult(
            curves={self.name: active_fraction_curve(lengths)},
            scalars={"episode_rounds_max": float(max(lengths)),
                     "episode_rounds_median": percentile(lengths, 50)},
        )


def _consensus_round(frame: EpisodeFrame, threshold: float) -> float | None:
    """This episode's consensus round, or ``None`` if it never converged.

    Prefers the value the episode itself recorded, because that used the game's
    own criterion; falls back to a threshold crossing of the dominant share for
    games that record no such `FinalMetric`.
    """

    recorded = frame.final.get(CONSENSUS_FINAL_METRIC)
    if recorded is not None:
        return float(recorded)
    if CONSENSUS_FINAL_METRIC in frame.final:
        return None
    dominant = _dominant_series(frame)
    for index, value in enumerate(dominant):
        if value >= threshold:
            return float(frame.rounds[index]) if index < len(frame.rounds) else float(index + 1)
    return None


class ConsensusRound(AggregateMetric):
    """Distribution of the round each episode converged on — median and IQR.

    Reported as order statistics rather than a mean because the distribution is
    heavy-tailed and, in cells where some episodes never converge, censored:
    the mean of the converged subset would move with the *censoring rate*
    rather than with the dynamics.  Non-converging episodes are excluded here
    and counted by `ConvergedFraction` instead, which is the honest split.
    """

    def __init__(self, threshold: float = 0.95, *, name: str = "consensus_round") -> None:
        super().__init__(name)
        self.threshold = threshold

    def compute(
        self, episodes: Sequence[EpisodeFrame], policy: AggregationPolicy
    ) -> AggregateResult:
        rounds = [
            value
            for value in (_consensus_round(frame, self.threshold) for frame in episodes)
            if value is not None
        ]
        if not rounds:
            return AggregateResult(scalars={"median_consensus_round": math.nan})
        low, high = percentile(rounds, 25), percentile(rounds, 75)
        return AggregateResult(
            scalars={
                "median_consensus_round": percentile(rounds, 50),
                "consensus_round_p25": low,
                "consensus_round_p75": high,
                "consensus_round_iqr": high - low,
            }
        )


class ConvergedFraction(AggregateMetric):
    """Fraction of the cell's episodes that reached consensus at all."""

    def __init__(self, threshold: float = 0.95, *, name: str = "converged_fraction") -> None:
        super().__init__(name)
        self.threshold = threshold

    def compute(
        self, episodes: Sequence[EpisodeFrame], policy: AggregationPolicy
    ) -> AggregateResult:
        if not episodes:
            return AggregateResult()
        converged = sum(
            1 for frame in episodes if _consensus_round(frame, self.threshold) is not None
        )
        return AggregateResult(scalars={self.name: converged / len(episodes)})


class MacrostateCounts(AggregateMetric):
    """The contingency tables a `SweepMetric` needs, summed over the cell.

    Not a curve and not for looking at: this is the cell's *sufficient
    statistic* for grid-level mutual information.  Carrying it in the
    `AggregateResult` is what lets `SweepMetric` run on every cell completion
    without rescanning episodes, and what makes a cell's contribution to the
    grid outlive the cell — a master killed later still has correct grid MI
    from the cells that finished.

    ``terminal_outcome``  one count per winning option, over episodes.
    ``macrostate_transition_h<h>``  ``"<current>><future>" -> count`` over all
    within-episode round pairs at lag h.  Pairs never cross an episode
    boundary, which is what makes them a transition rather than an artifact.
    """

    def __init__(self, horizons: Sequence[int] = (1,), *, name: str = "macrostate_counts") -> None:
        super().__init__(name)
        self.horizons = tuple(int(h) for h in horizons)
        if any(h < 1 for h in self.horizons):
            raise ValueError("horizons must be positive")

    def compute(
        self, episodes: Sequence[EpisodeFrame], policy: AggregationPolicy
    ) -> AggregateResult:
        terminal: dict[str, float] = {}
        transitions: dict[int, dict[str, float]] = {h: {} for h in self.horizons}
        for frame in episodes:
            macrostates = _macrostate_series(frame)
            if not macrostates:
                continue
            terminal[macrostates[-1]] = terminal.get(macrostates[-1], 0.0) + 1.0
            for horizon in self.horizons:
                table = transitions[horizon]
                for index in range(len(macrostates) - horizon):
                    key = f"{macrostates[index]}>{macrostates[index + horizon]}"
                    table[key] = table.get(key, 0.0) + 1.0
        if not terminal:
            return AggregateResult()
        total = sum(terminal.values())
        counts: dict[str, Mapping[str, float]] = {"terminal_outcome": terminal}
        for horizon, table in transitions.items():
            if table:
                counts[f"macrostate_transition_h{horizon}"] = table
        return AggregateResult(
            counts=counts,
            scalars={
                f"terminal_outcome_fraction_{option}": count / total
                for option, count in sorted(terminal.items())
            },
        )


CELL_METRICS = {
    "dominant_action_share": DominantActionShare,
    "action_share_relabelled": ActionShareRelabelled,
    "active_fraction": ActiveFraction,
    "consensus_round": ConsensusRound,
    "converged_fraction": ConvergedFraction,
    "macrostate_counts": MacrostateCounts,
}
"""Config name -> class, for resolving ``aggregation.cell_metrics``."""


def create_cell_metrics(
    names: Sequence[str], *, horizons: Sequence[int] = (1,)
) -> tuple[AggregateMetric, ...]:
    """Instantiate the configured cell metrics, rejecting unknown names up front.

    ``horizons`` has to be threaded in from the same config the sweep metrics
    read. `MacrostateCounts` builds one transition table per horizon, and
    `LaggedConditionalMutualInformation` can only estimate at a horizon whose
    table exists — so a cell built at the default h=1 against a sweep asking
    for h=5 would report NaN with nothing to indicate why.
    """

    unknown = [name for name in names if name not in CELL_METRICS]
    if unknown:
        raise ValueError(
            f"unknown aggregation.cell_metrics: {', '.join(unknown)}; "
            f"available: {', '.join(sorted(CELL_METRICS))}"
        )
    arguments: dict[str, dict[str, Any]] = {"macrostate_counts": {"horizons": tuple(horizons)}}
    return tuple(CELL_METRICS[name](**arguments.get(name, {})) for name in names)


__all__ = [
    "CELL_METRICS",
    "CONSENSUS_FINAL_METRIC",
    "SHARE_METRIC",
    "ActionShareRelabelled",
    "ActiveFraction",
    "ConsensusRound",
    "ConvergedFraction",
    "DominantActionShare",
    "MacrostateCounts",
    "create_cell_metrics",
]
