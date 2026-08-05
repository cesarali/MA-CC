"""Where the estimate meets the answer key.

Because these games are a rehearsal of the workflow rather than a test suite,
the analytic answer cannot live in an assertion - it has to be visible at the
moment we look at results. Everything here is built around that: each table
carries `truth` and `gap` as columns next to the estimate, so "did it work" is
a look rather than an investigation.

Three views, decided in advance:

`null_distribution`   epsilon = 0.5, many seeds, where the true MI is exactly
                      zero. Plug-in MI is biased upward at finite sample; this
                      says by how much, and therefore what magnitude of MI
                      means anything at all in a real run.
`calibration_curve`   MI swept across epsilon against 1 - H(q). Estimator bias
                      becomes a visible offset from the diagonal instead of a
                      number someone has to judge.
`parity`              the same seeds through both modes. Bit-exact trajectory
                      agreement is the strong form; the estimate comparison is
                      reported alongside it because a run that fails the strong
                      check still wants to show how far apart the two landed.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from mas_cc.analysis.estimators import Estimate, mutual_information_from_counts
from mas_cc.config import GameConfig

from .metrics_common import ACTION_METRIC_NAME, DEFAULT_WINDOW
from .protocols import GroundTruth, SyntheticGame

ESTIMATORS = ("unsmoothed", "jeffreys", "miller_madow")
"""The three estimators reported for every pair.

`unsmoothed` is the plug-in estimator and the one the null distribution is
about - it is the one whose finite-sample bias we are trying to measure, so
smoothing it away by default would hide the very quantity in question.
"""


# -- reading a finished episode ------------------------------------------


def read_action_series(run_dir: str | Path) -> dict[str, list[str]]:
    """Per-agent action series from a completed run's `metrics/streaming.csv`.

    Deliberately reads the *artifact*, not an in-memory result: this is the
    path a run pulled back off the cluster takes, so exercising it is part of
    what is being rehearsed.
    """

    path = Path(run_dir) / "metrics" / "streaming.csv"
    if not path.is_file():
        raise FileNotFoundError(f"no streaming metrics at {path}")
    by_agent: dict[str, list[tuple[int, str]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["metric_name"] != ACTION_METRIC_NAME or not row["agent_id"]:
                continue
            by_agent.setdefault(row["agent_id"], []).append(
                (int(row["round_index"]), row["value"])
            )
    return {
        agent_id: [value for _, value in sorted(entries)]
        for agent_id, entries in sorted(by_agent.items())
    }


def read_streaming_metrics(run_dir: str | Path) -> pd.DataFrame:
    """The whole `metrics/streaming.csv` as a frame, or empty if absent."""

    path = Path(run_dir) / "metrics" / "streaming.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["round_index", "agent_id", "series", "metric_name", "value"])
    return pd.read_csv(path)


def read_final_metrics(run_dir: str | Path) -> dict[str, Any]:
    """`metrics/final.csv` as ``{metric_name: value}``, with blanks read as None."""

    path = Path(run_dir) / "metrics" / "final.csv"
    if not path.is_file():
        return {}
    values: dict[str, Any] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            raw = row["value"]
            values[row["metric_name"]] = None if raw in ("", "None") else raw
    return values


def _tolerance(expected: float | None, samples: int) -> float | None:
    """How far a time-averaged proportion may sit from its closed form.

    Derived rather than picked: these metrics are proportions averaged over
    ``samples`` rounds, so their sampling error is ``sqrt(p(1-p)/n)`` and four
    of those is a band a correct implementation essentially always sits inside.
    A fixed tolerance would be wrong in both directions - far too tight for a
    share whose per-round value swings between 0 and 1 over a short episode,
    and far too loose for a rate that concentrates hard.

    The floor keeps a degenerate quantity (``p`` of 0 or 1, where the sampling
    error is exactly zero) from demanding bit-perfect equality of a float mean.

    This band judges whether the *recorder wrote what the game did*. It is not
    a test of estimator bias - the mutual-information table and the parity
    check are what answer that, at a precision this deliberately does not.
    """

    if expected is None:
        return None
    variance = max(expected * (1.0 - expected), 0.0)
    return max(0.02, 4.0 * math.sqrt(variance / max(samples, 1)))


def metric_check(
    run_dir: str | Path, truth: GroundTruth, *, warmup_rounds: int = DEFAULT_WINDOW
) -> pd.DataFrame:
    """The ordinary choice metrics, next to what they analytically had to be.

    The mutual-information table answers "is the estimator right". This answers
    the more mundane and equally necessary question: did the *recorder* write
    the numbers the game actually produced. Every row is a metric the game
    declared; `expected` is filled wherever `ground_truth()` has a closed form
    for it and left blank where it genuinely has none, rather than inventing a
    target to compare against.

    Streaming metrics are reduced to a time-average, which is what converges to
    the closed form. The rolling ones skip the first ``warmup_rounds``, because
    a trailing window is still filling up over exactly that many rounds and its
    values there are not estimates of anything.

    Skipping only the warm-up, rather than a fixed fraction of the episode,
    matters more than it sounds. For a rare event - unanimity at 3% - throwing
    away half the rounds turns a good estimate into a noisy one, and an episode
    whose few unanimous rounds happened to fall early then reads as a flat zero
    against a correct closed form. That is a measurement artifact masquerading
    as a discrepancy, in exactly the place we are trying to detect real ones.
    """

    frame = read_streaming_metrics(run_dir)
    final = read_final_metrics(run_dir)
    rows: list[dict[str, Any]] = []

    def _expected(name: str, subject: Sequence[str] = ()) -> float | None:
        try:
            return truth.value(name, subject)
        except KeyError:
            return None

    if not frame.empty:
        warm = frame[frame["round_index"] > warmup_rounds]
        for name, group in frame.groupby("metric_name", sort=True):
            if name == ACTION_METRIC_NAME:
                continue  # a categorical passthrough; it has no meaningful mean
            rolling = name.startswith("rolling_")
            source = (warm if rolling else frame)
            source = source[source["metric_name"] == name]
            for series, part in source.groupby(source["series"].fillna(""), sort=True):
                observed = pd.to_numeric(part["value"], errors="coerce").mean()
                subject = (str(series),) if series else ()
                expected = (
                    _expected("expected_action_share", subject)
                    if name.endswith("action_share_per_option")
                    else _expected("unanimity_probability")
                    if name == "rolling_coordination_rate"
                    else _expected("expected_dominant_action_share")
                    if name == "dominant_action_share"
                    else None
                )
                rows.append(
                    {
                        "metric": name,
                        "series": str(series),
                        "kind": "streaming (time-mean)",
                        "observed": None if pd.isna(observed) else float(observed),
                        "expected": expected,
                        "tolerance": _tolerance(expected, len(part)),
                        # Carried as its own flag rather than inferred from
                        # `expected` later: these rows become a DataFrame, and a
                        # None in a float column silently becomes NaN - which
                        # would turn "no closed form for this metric" into a
                        # failed comparison against nothing.
                        "has_expected": expected is not None,
                        "judged": True,
                        "note": f"after {warmup_rounds}-round window warm-up" if rolling else "",
                    }
                )

    # Final metrics are matched by exact name, never by substring. The two
    # consensus metrics measure genuinely different things and have different
    # answer keys, and matching on "consensus" would silently hold one of them
    # to the other's target - which is exactly the mistake that makes a metric
    # look broken when it is behaving correctly.
    must_converge = _expected("expected_consensus_by_success_rate")
    chance_round = _expected("expected_first_chance_unanimity_round")
    for name, value in sorted(final.items()):
        expected: Any = None
        judged = True
        note = ""
        if name in ("first_consensus_time_by_success_rate", "consensus_action_by_success_rate"):
            if must_converge is not None:
                expected = "fires" if must_converge == 1.0 else "None"
            note = "rolling window at threshold; needs sustained agreement"
        elif name == "first_consensus_time_by_action_share":
            # A standing-share criterion with no persistence requirement, so a
            # single lucky round satisfies it. Reported against its own closed
            # form but left unjudged: the waiting time is geometric, and one
            # episode is a single draw whose spread equals its own mean.
            expected = chance_round
            judged = False
            note = "no persistence requirement; geometric waiting time, one draw"
        rows.append(
            {
                "metric": name,
                "series": "",
                "kind": "final",
                "observed": value,
                "expected": expected,
                "tolerance": None,
                "has_expected": expected is not None,
                "judged": judged,
                "note": note,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    def _agrees(row: Mapping[str, Any]) -> Any:
        # Blank rather than True wherever there is no closed form, or where the
        # quantity is too dispersed for one episode to judge - "no target" and
        # "too noisy to call" must never read as "checked and fine".
        expected, observed = row["expected"], row["observed"]
        if not row["has_expected"] or not row["judged"]:
            return None
        if isinstance(expected, str):
            fired = observed not in (None, "None", "")
            return (expected == "fires") == fired
        if observed is None:
            return False
        return abs(float(observed) - float(expected)) <= float(row["tolerance"])

    result["agrees"] = result.apply(_agrees, axis=1)
    return result


def series_to_indices(
    series: Mapping[str, Sequence[str]], labels: Sequence[str]
) -> np.ndarray:
    """A recorded action series as a ``(rounds, agents)`` index array."""

    index = {label: position for position, label in enumerate(labels)}
    agents = sorted(series)
    if not agents:
        return np.zeros((0, 0), dtype=np.int8)
    lengths = {len(series[agent]) for agent in agents}
    if len(lengths) != 1:
        raise ValueError(f"agents have unequal round counts: {sorted(lengths)}")
    unknown = {
        value for agent in agents for value in series[agent] if value not in index
    }
    if unknown:
        raise ValueError(f"recorded actions outside the declared alphabet: {sorted(unknown)}")
    return np.array(
        [[index[value] for value in series[agent]] for agent in agents], dtype=np.int8
    ).T


# -- estimation -----------------------------------------------------------


def pairwise_counts(actions: np.ndarray, left: int, right: int, alphabet: int) -> np.ndarray:
    """The full contingency table for one agent pair over an episode.

    Every cell of the alphabet-by-alphabet table is present even when unseen,
    so a level that never occurred is a zero rather than a missing row - which
    is what keeps the estimate comparable across seeds.
    """

    table = np.zeros((alphabet, alphabet), dtype=float)
    np.add.at(table, (actions[:, left], actions[:, right]), 1.0)
    return table


def pairwise_estimates(
    actions: np.ndarray, agent_ids: Sequence[str], alphabet: int
) -> list[dict[str, Any]]:
    """MI for every unordered pair of agents in one episode's action array."""

    rows: list[dict[str, Any]] = []
    for left in range(actions.shape[1]):
        for right in range(left + 1, actions.shape[1]):
            estimate = mutual_information_from_counts(
                pairwise_counts(actions, left, right, alphabet)
            )
            rows.append(
                {
                    "agent_a": agent_ids[left],
                    "agent_b": agent_ids[right],
                    "observations": estimate.observations,
                    **{name: getattr(estimate, name) for name in ESTIMATORS},
                }
            )
    return rows


def with_truth(rows: Sequence[Mapping[str, Any]], truth: GroundTruth) -> pd.DataFrame:
    """Join each pair's estimate onto its own closed-form value.

    The join is on the pair, not on a single global number, because the
    truth genuinely differs per pair whenever the agents have different noise
    levels - and a table that silently compared every pair to one value would
    look fine on the symmetric configs and lie on the asymmetric ones.
    """

    joined: list[dict[str, Any]] = []
    for row in rows:
        value = truth.value("mutual_information", (row["agent_a"], row["agent_b"]))
        entry = {**dict(row), "truth": value}
        for name in ESTIMATORS:
            entry[f"gap_{name}"] = entry[name] - value
        joined.append(entry)
    return pd.DataFrame(joined)


def episode_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """One episode reduced to the numbers worth printing."""

    if frame.empty:
        return {"pairs": 0}
    return {
        "pairs": int(len(frame)),
        "truth_mean": float(frame["truth"].mean()),
        **{f"{name}_mean": float(frame[name].mean()) for name in ESTIMATORS},
        **{f"gap_{name}_mean": float(frame[f"gap_{name}"].mean()) for name in ESTIMATORS},
        "gap_unsmoothed_max_abs": float(frame["gap_unsmoothed"].abs().max()),
    }


# -- speed-mode sweeps ----------------------------------------------------


def _config_with_epsilon(config: GameConfig, epsilon: float) -> GameConfig:
    """The same config with one knob moved, so truth and sample stay in step.

    Both the sampler and `ground_truth()` are then driven by this one object -
    which is the whole defence against the phantom bug where epsilon changes in
    the YAML and an expected value somewhere else does not.
    """

    options = {key: value for key, value in config.options.items() if key != "epsilons"}
    options["epsilon"] = float(epsilon)
    return replace(config, options=options)


def simulate_estimates(
    game: SyntheticGame, config: GameConfig, seeds: Sequence[int]
) -> pd.DataFrame:
    """Per-seed, per-pair MI from speed mode, with truth already attached."""

    episodes = game.simulate(config, seeds)
    truth = game.ground_truth(config)
    agent_ids = [f"agent-{index:03d}" for index in range(episodes.agents)]
    alphabet = len(episodes.action_labels)
    frames: list[pd.DataFrame] = []
    for position, seed in enumerate(episodes.seeds):
        rows = pairwise_estimates(episodes.actions[position], agent_ids, alphabet)
        frame = with_truth(rows, truth)
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def null_distribution(
    game: SyntheticGame, config: GameConfig, seeds: Sequence[int]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The eps = 0.5 case, where true MI is exactly zero, across many seeds.

    Returns the per-seed-per-pair estimates and a summary whose most useful
    entry is the 95th percentile: below that, an MI reported by a real run at
    this population and round count is indistinguishable from nothing.
    """

    null_config = _config_with_epsilon(config, 0.5)
    frame = simulate_estimates(game, null_config, seeds)
    summary: dict[str, Any] = {
        "epsilon": 0.5,
        "true_mutual_information_bits": 0.0,
        "seeds": len(tuple(seeds)),
        "rounds": null_config.horizon,
        "population_size": null_config.population_size,
        "samples": int(len(frame)),
    }
    for name in ESTIMATORS:
        if frame.empty:
            continue
        values = frame[name].to_numpy(dtype=float)
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary[f"{name}_p95"] = float(np.quantile(values, 0.95))
        summary[f"{name}_max"] = float(np.max(values))
    summary["significance_floor_bits"] = summary.get("unsmoothed_p95", math.nan)
    return frame, summary


def calibration_curve(
    game: SyntheticGame,
    config: GameConfig,
    epsilon_grid: Sequence[float],
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Estimated MI against 1 - H(q) across the epsilon grid, one row per point."""

    rows: list[dict[str, Any]] = []
    for epsilon in epsilon_grid:
        point_config = _config_with_epsilon(config, epsilon)
        frame = simulate_estimates(game, point_config, seeds)
        truth = float(frame["truth"].iloc[0]) if not frame.empty else math.nan
        row: dict[str, Any] = {
            "epsilon": float(epsilon),
            "truth": truth,
            "seeds": len(tuple(seeds)),
            "samples": int(len(frame)),
        }
        for name in ESTIMATORS:
            values = frame[name].to_numpy(dtype=float)
            row[f"{name}_mean"] = float(np.mean(values))
            row[f"{name}_p05"] = float(np.quantile(values, 0.05))
            row[f"{name}_p95"] = float(np.quantile(values, 0.95))
            row[f"gap_{name}_mean"] = row[f"{name}_mean"] - truth
        rows.append(row)
    return pd.DataFrame(rows)


# -- the fidelity / speed agreement check ---------------------------------


@dataclass(frozen=True, slots=True)
class ParityResult:
    """What the two modes said about the same seed.

    `identical` is the strong claim and the one worth acting on: the full
    pipeline and the bare sampler produced the same action for every agent in
    every round. `estimate_gap` is reported whether or not that holds, because
    a run that fails the strong check still wants to say how far apart the two
    landed - a single mismatched round and a systematic off-by-one look very
    different in that number.
    """

    seed: int
    rounds: int
    agents: int
    identical: bool
    mismatched_cells: int
    first_mismatch: tuple[int, int] | None
    fidelity_estimate: float
    speed_estimate: float
    truth: float

    @property
    def estimate_gap(self) -> float:
        return self.fidelity_estimate - self.speed_estimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "rounds": self.rounds,
            "agents": self.agents,
            "identical": self.identical,
            "mismatched_cells": self.mismatched_cells,
            "first_mismatch_round": None if self.first_mismatch is None else self.first_mismatch[0],
            "first_mismatch_agent": None if self.first_mismatch is None else self.first_mismatch[1],
            "fidelity_mean_mutual_information": self.fidelity_estimate,
            "speed_mean_mutual_information": self.speed_estimate,
            "truth_mean_mutual_information": self.truth,
            "estimate_gap": self.estimate_gap,
        }


def compare_modes(
    game: SyntheticGame,
    config: GameConfig,
    seed: int,
    recorded: Mapping[str, Sequence[str]],
) -> ParityResult:
    """Check one recorded episode against speed mode on the same seed."""

    episodes = game.simulate(config, (seed,))
    labels = episodes.action_labels
    expected = episodes.actions[0]
    observed = series_to_indices(recorded, labels)
    if observed.shape != expected.shape:
        raise ValueError(
            f"recorded episode is {observed.shape} but speed mode produced {expected.shape}; "
            "the two modes disagree about the shape of an episode, not just its content"
        )
    difference = observed != expected
    mismatched = int(difference.sum())
    first: tuple[int, int] | None = None
    if mismatched:
        round_index, agent_index = (int(value) for value in np.argwhere(difference)[0])
        first = (round_index + 1, agent_index)

    truth = game.ground_truth(config)
    agent_ids = sorted(recorded)
    alphabet = len(labels)
    fidelity = with_truth(pairwise_estimates(observed, agent_ids, alphabet), truth)
    speed = with_truth(pairwise_estimates(expected, agent_ids, alphabet), truth)
    return ParityResult(
        seed=seed,
        rounds=int(expected.shape[0]),
        agents=int(expected.shape[1]),
        identical=mismatched == 0,
        mismatched_cells=mismatched,
        first_mismatch=first,
        fidelity_estimate=float(fidelity["unsmoothed"].mean()),
        speed_estimate=float(speed["unsmoothed"].mean()),
        truth=float(speed["truth"].mean()),
    )


# -- the canonical view ---------------------------------------------------


_ESTIMATOR_STYLE = {
    "unsmoothed": ("#1f77b4", "o"),
    "jeffreys": ("#d62728", "s"),
    "miller_madow": ("#2ca02c", "^"),
}


def plot_calibration(frame: pd.DataFrame, destination: str | Path) -> Path:
    """The canonical Bernoulli view: calibration against the diagonal, and the residual.

    Two panels, because one is not enough. The top panel answers "is the
    estimator roughly right" - a well-behaved one lies on the diagonal, and it
    is the honest first look. But an estimator that lies on the diagonal *to
    the eye* can still carry a systematic bias two orders of magnitude smaller
    than the axis, which at these scales is invisible and is exactly the
    quantity we came to measure.

    So the bottom panel plots estimate minus truth, where a constant offset
    across the sweep - rather than scatter around zero - is the signature of
    finite-sample bias rather than noise. The shaded band is the seed-to-seed
    spread, which says whether that offset is something you could see in one
    run or only in the average of many.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(6.4, 7.6), height_ratios=(3, 2), constrained_layout=True
    )
    limit = max(1.0, float(frame["truth"].max()) if not frame.empty else 1.0)

    upper.plot(
        [0, limit], [0, limit], linestyle="--", linewidth=1.2, color="0.25",
        label="truth (1 - H(q))", zorder=1,
    )
    for name in ESTIMATORS:
        colour, marker = _ESTIMATOR_STYLE[name]
        upper.fill_between(
            frame["truth"], frame[f"{name}_p05"], frame[f"{name}_p95"],
            alpha=0.12, color=colour, linewidth=0, zorder=0,
        )
        upper.plot(
            frame["truth"], frame[f"{name}_mean"], marker=marker, markersize=4,
            linewidth=1.2, color=colour, label=name, alpha=0.85, zorder=2,
        )
    upper.set_ylabel("estimated mutual information  [bits]")
    upper.set_title("Bernoulli calibration: estimate vs. closed form")
    upper.legend(loc="upper left", frameon=False, fontsize=9)

    lower.axhline(0.0, linestyle="--", linewidth=1.2, color="0.25", zorder=1)
    for name in ESTIMATORS:
        colour, marker = _ESTIMATOR_STYLE[name]
        lower.fill_between(
            frame["truth"],
            frame[f"{name}_p05"] - frame["truth"],
            frame[f"{name}_p95"] - frame["truth"],
            alpha=0.12, color=colour, linewidth=0, zorder=0,
        )
        lower.plot(
            frame["truth"], frame[f"{name}_mean"] - frame["truth"], marker=marker,
            markersize=4, linewidth=1.2, color=colour, label=name, alpha=0.85, zorder=2,
        )
    lower.set_xlabel("true I(A_i; A_j) = 1 - H(q)  [bits]")
    lower.set_ylabel("estimate - truth  [bits]")
    lower.set_title("Residual: a constant offset is bias, scatter is noise", fontsize=10)
    lower.legend(loc="upper right", frameon=False, fontsize=9)

    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


__all__ = [
    "ESTIMATORS",
    "Estimate",
    "ParityResult",
    "calibration_curve",
    "compare_modes",
    "episode_summary",
    "metric_check",
    "null_distribution",
    "read_final_metrics",
    "read_streaming_metrics",
    "pairwise_counts",
    "pairwise_estimates",
    "plot_calibration",
    "read_action_series",
    "series_to_indices",
    "simulate_estimates",
    "with_truth",
]
