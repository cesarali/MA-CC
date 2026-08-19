"""Round-level controller-information analysis for the relational game.

This module contains **no estimator**.  Everything scientific here - the
direct-counting MI/CMI, the whole-episode bootstrap, the policy-conditional
null, the action-support and overlap diagnostics - is the HiddenBench
round-feedback pipeline
(`mas_cc.games.hidden_bench.imitation_round_feedback.analysis`), reached
through its public `round_information_analysis`.

What is actually new is one adapter.  The relational round record already
carries every quantity the pipeline needs, but under its own names and one
level of indirection: occupation counts come as a vector aligned to
`possible_answers` rather than as the scalar target/truth counts the pipeline
reads, and the epistemic state is a knowledge histogram the pipeline has never
had a reason to look at.  `adapt_relational_round_record` resolves those into
the `RoundEvent` shape and adds three keys the pipeline knows how to consume
generically:

``conditioning_memory_state``
    `E_k = (n_k^(0), ..., n_k^(L))`, the number of agents holding exactly `j`
    of the `L` supporting facts at the **start** of the round.  This is what
    turns `I(U_k ; n_Z,k+1 | n_Z,k)` into
    `I(U_k ; n_Z,k+1 | n_Z,k, E_k)` - same estimator, wider `z`.

``conditioning_epistemic_state``
    a deliberately coarse `(kappa_k, phi_k)` bin pair, reported under its own
    statistic name.  It exists because the full `E_k` conditioning can be too
    sparse to estimate on a small pilot, and a sparse exact answer must stay
    visible instead of being quietly replaced by a dense approximate one.

``delta_p_ctrl``
    the target share's per-round change, so the signed response can be read in
    share units as well as in aligned-magnetization units.

The controller semantics are not touched anywhere in this file: it reads
finished trajectories.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# `_support` and `_write_markdown` are imported deliberately rather than
# reimplemented: they ARE the support-diagnostic and report contract this
# analysis is supposed to match, and a second copy would drift.
from ...hidden_bench.imitation_round_feedback.analysis import (
    MAIN_ESTIMATOR_VARIANT,
    ROUND_ANALYSIS_STATISTICS,
    ROUND_MEMORY_STATISTICS,
    RoundEvent,
    _support,
    _write_markdown,
    round_information_analysis,
)
from .state import ROUND_RECORD_TYPE

DEFAULT_EPISTEMIC_BINS = 4
"""Bins per axis for the coarse `(kappa, phi)` diagnostic conditioning.

Four is small on purpose.  This state exists to stay estimable when `E_k` does
not, so making it finer would defeat the only reason it is there."""

REQUIRED_FIELDS = (
    "round_index",
    "occupation_counts_before",
    "occupation_counts_after",
    "possible_answers",
    "correct_answer",
)


def _bin(value: float | None, bins: int) -> int | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return min(bins - 1, max(0, int(float(value) * bins)))


def adapt_relational_round_record(
    record: Mapping[str, Any],
    *,
    cell_id: str = "run",
    episode_id: str | None = None,
    epistemic_bins: int = DEFAULT_EPISTEMIC_BINS,
) -> RoundEvent:
    """One relational round record as a `RoundEvent` the shared pipeline reads."""

    if record.get("record_type") not in {None, ROUND_RECORD_TYPE}:
        raise ValueError(f"record is not a {ROUND_RECORD_TYPE} row")
    missing = sorted(set(REQUIRED_FIELDS) - set(record))
    if missing:
        raise ValueError("relational round record is missing: " + ", ".join(missing))

    options = [str(item) for item in record["possible_answers"]]
    before = [int(value) for value in record["occupation_counts_before"]]
    after = [int(value) for value in record["occupation_counts_after"]]
    if not (len(before) == len(after) == len(options)) or len(options) < 2:
        raise ValueError("occupation vectors must align with possible_answers, K >= 2")
    if sum(before) != sum(after):
        raise ValueError("round occupation vectors must conserve population size")

    # `analysis_target` is the controller's target when there is one and the
    # correct answer otherwise, which is exactly the fallback `m_ctrl` already
    # uses - so target and truth channels coincide on an uncontrolled cell
    # instead of the target channel silently vanishing.
    target = str(record.get("analysis_target") or record["correct_answer"])
    correct = str(record["correct_answer"])
    for label in (target, correct):
        if label not in options:
            raise ValueError(f"{label!r} is outside the task option alphabet")
    target_index = options.index(target)
    truth_index = options.index(correct)

    kappa = record.get("mean_supporting_fact_coverage_before")
    phi = record.get("full_proof_agent_share_before")
    kappa_bin, phi_bin = _bin(kappa, epistemic_bins), _bin(phi, epistemic_bins)
    memory = record.get("knowledge_stratum_counts_before")
    total = sum(before)

    event = {
        **dict(record),
        "target_count_before": before[target_index],
        "target_count_after": after[target_index],
        "truth_count_before": before[truth_index],
        "truth_count_after": after[truth_index],
        "delta_p_ctrl": (after[target_index] - before[target_index]) / total,
        "delta_p_truth": (after[truth_index] - before[truth_index]) / total,
        "conditioning_memory_state": (
            None if memory is None else [int(value) for value in memory]
        ),
        "conditioning_epistemic_state": (
            None if kappa_bin is None or phi_bin is None else [kappa_bin, phi_bin]
        ),
    }
    return RoundEvent(
        cell_id=cell_id,
        episode_id=str(episode_id or record.get("episode_id") or "episode"),
        round_index=int(record["round_index"]),
        event=event,
    )


def _cell_id_for(path: Path) -> str:
    for parent in path.parents:
        if (parent / "overrides.json").is_file():
            return parent.name
    return "run"


def read_relational_round_records(
    root: str | Path, *, epistemic_bins: int = DEFAULT_EPISTEMIC_BINS
) -> list[RoundEvent]:
    source = Path(root)
    paths = [source] if source.is_file() else sorted(source.rglob("round_trajectory.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no round_trajectory.jsonl files under {source}")
    rows: list[RoundEvent] = []
    for path in paths:
        cell_id = _cell_id_for(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("record_type") != ROUND_RECORD_TYPE:
                continue
            rows.append(
                adapt_relational_round_record(
                    payload, cell_id=cell_id, epistemic_bins=epistemic_bins
                )
            )
    if not rows:
        raise ValueError(f"round trajectory files under {source} contain no round records")
    return rows


def _grouped(
    rows: Sequence[RoundEvent], key: Any
) -> dict[Hashable, list[RoundEvent]]:
    result: dict[Hashable, list[RoundEvent]] = defaultdict(list)
    for row in rows:
        result[key(row)].append(row)
    return dict(result)


def _mean(values: Sequence[Any]) -> float:
    numbers = [float(value) for value in values if value is not None]
    return math.nan if not numbers else float(np.mean(numbers))


def controller_action_summary(rows: Sequence[RoundEvent]) -> dict[str, Any]:
    """Raw action bookkeeping: what the controller did, and how often."""

    actions = [str(row.U_k) for row in rows if row.U_k is not None]
    advocate = sum(1 for action in actions if action != "NO_OP")
    return {
        "rounds": len(rows),
        "controlled_rounds": len(actions),
        "advocate_rounds": advocate,
        "no_op_rounds": len(actions) - advocate,
        "advocate_frequency": math.nan if not actions else advocate / len(actions),
        "mean_advocate_probability": _mean(
            [row.p_k for row in rows if row.p_k is not None]
        ),
        "mean_controlled_positions": _mean(
            [row.event.get("controlled_position_count") for row in rows]
        ),
        "mean_controlled_positions_on_advocate": _mean(
            [
                row.event.get("controlled_position_count")
                for row in rows
                if row.U_k not in (None, "NO_OP")
            ]
        ),
        "max_controlled_positions_on_no_op": max(
            [
                int(row.event.get("controlled_position_count") or 0)
                for row in rows
                if row.U_k == "NO_OP"
            ]
            or [0]
        ),
        "mean_sensor_sample_size": _mean(
            [row.event.get("sensor_sample_size") for row in rows]
        ),
    }


def epistemic_regime_summary(rows: Sequence[RoundEvent]) -> dict[str, Any]:
    """kappa_k and phi_k, kept next to the information numbers on purpose.

    An actuation CMI says how much the controller moved the opinion channel; it
    says nothing about whether the population was epistemically able to move.
    Reading the two apart is how a null result gets misattributed.
    """

    def series(field: str) -> list[float]:
        return [
            float(row.event[field])
            for row in rows
            if row.event.get(field) is not None
        ]

    result: dict[str, Any] = {}
    for label, field in (
        ("kappa", "mean_supporting_fact_coverage"),
        ("phi", "full_proof_agent_share"),
    ):
        start = series(f"{field}_before")
        end = series(field)
        result[f"{label}_mean"] = math.nan if not end else float(np.mean(end))
        result[f"{label}_initial"] = math.nan if not start else float(start[0])
        result[f"{label}_final"] = math.nan if not end else float(end[-1])
        result[f"{label}_max"] = math.nan if not end else float(max(end))
    result["mean_peer_fact_exposures"] = _mean(
        [row.event.get("peer_fact_exposures") for row in rows]
    )
    result["mean_new_peer_facts"] = _mean(
        [row.event.get("new_peer_facts") for row in rows]
    )
    result["mean_controller_fact_exposures"] = _mean(
        [row.event.get("controller_fact_exposures") for row in rows]
    )
    return result


def analyze_relational_imitation_round_feedback(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_resamples: int = 1000,
    null_permutations: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
    statistics: Sequence[str] | None = None,
    epistemic_bins: int = DEFAULT_EPISTEMIC_BINS,
) -> dict[str, Any]:
    """Run the shared round-feedback pipeline over a relational grid."""

    rounds = read_relational_round_records(run_dir, epistemic_bins=epistemic_bins)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    estimates: list[dict[str, Any]] = []
    nulls: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    regimes: list[dict[str, Any]] = []
    by_cell = _grouped(rounds, key=lambda row: row.cell_id)
    for cell_id, group in by_cell.items():
        cell_estimates, cell_nulls = round_information_analysis(
            group,
            statistics=statistics,
            bootstrap_resamples=bootstrap_resamples,
            null_permutations=null_permutations,
            confidence=confidence,
            seed=seed,
        )
        estimates.extend({"cell_id": cell_id, **row} for row in cell_estimates)
        nulls.extend({"cell_id": cell_id, **row} for row in cell_nulls)
        support.append({"cell_id": cell_id, **_support(group)})
        actions.append({"cell_id": cell_id, **controller_action_summary(group)})
        for episode_id, episode in _grouped(
            group, key=lambda row: row.episode_id
        ).items():
            ordered = sorted(episode, key=lambda row: row.round_index)
            regimes.append(
                {
                    "cell_id": cell_id,
                    "episode_id": episode_id,
                    "task_id": ordered[0].event.get("task_id"),
                    "rounds": len(ordered),
                    **epistemic_regime_summary(ordered),
                    **controller_action_summary(ordered),
                }
            )

    # Pooled across cells as well: with 10 rounds x 10 repetitions per cell the
    # per-cell tables are thin, and the pooled slice is the one with a real
    # chance of populating the memory-aware conditioning.
    pooled_estimates, pooled_nulls = round_information_analysis(
        rounds,
        statistics=statistics,
        bootstrap_resamples=bootstrap_resamples,
        null_permutations=null_permutations,
        confidence=confidence,
        seed=seed,
    )
    estimates.extend({"cell_id": "pooled", **row} for row in pooled_estimates)
    nulls.extend({"cell_id": "pooled", **row} for row in pooled_nulls)
    support.append({"cell_id": "pooled", **_support(rounds)})
    actions.append({"cell_id": "pooled", **controller_action_summary(rounds)})

    pd.DataFrame(estimates).to_csv(
        destination / "round_information_estimates.csv", index=False
    )
    _write_markdown(estimates, destination / "round_information_estimates.md")
    pd.DataFrame(nulls).to_csv(destination / "round_information_nulls.csv", index=False)
    pd.DataFrame(support).to_csv(
        destination / "round_support_diagnostics.csv", index=False
    )
    pd.DataFrame(actions).to_csv(
        destination / "controller_action_summary.csv", index=False
    )
    pd.DataFrame(regimes).to_csv(
        destination / "episode_epistemic_regime.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "cell_id": row.cell_id,
                "episode_id": row.episode_id,
                "task_id": row.event.get("task_id"),
                "round_index": row.round_index,
                "controller_action": row.U_k,
                "advocate_probability": row.p_k,
                "sensor_target_share": row.event.get("sensor_target_share"),
                "controlled_position_count": row.event.get("controlled_position_count"),
                "n_target_before": row.target_before,
                "n_target_after": row.target_after,
                "n_truth_before": row.truth_before,
                "n_truth_after": row.truth_after,
                "delta_p_ctrl": row.event.get("delta_p_ctrl"),
                "delta_p_truth": row.event.get("delta_p_truth"),
                "kappa_before": row.event.get("mean_supporting_fact_coverage_before"),
                "kappa_after": row.event.get("mean_supporting_fact_coverage"),
                "phi_before": row.event.get("full_proof_agent_share_before"),
                "phi_after": row.event.get("full_proof_agent_share"),
                "E_k": row.memory_state,
                "epistemic_bin": row.epistemic_state,
            }
            for row in sorted(
                rounds, key=lambda row: (row.cell_id, row.episode_id, row.round_index)
            )
        ]
    ).to_csv(destination / "round_epistemic_trajectory.csv", index=False)

    memory_rows = [
        row for row in estimates if row["statistic"] in ROUND_MEMORY_STATISTICS
    ]
    summary = {
        "n_cells": len(by_cell),
        "n_episodes": len({(row.cell_id, row.episode_id) for row in rounds}),
        "n_rounds": len(rounds),
        "statistics": list(
            ROUND_ANALYSIS_STATISTICS if statistics is None else statistics
        ),
        "bootstrap_unit": "episode",
        "bootstrap_resamples": bootstrap_resamples,
        "null_permutations": null_permutations,
        "main_estimator_variant": MAIN_ESTIMATOR_VARIANT,
        "epistemic_bins": epistemic_bins,
        # Surfaced in the summary rather than left in a CSV column: on a pilot
        # this size the memory-aware CMI is expected to be support-limited, and
        # that has to be impossible to miss when reading the result.
        "memory_conditioning_support": [
            {
                "cell_id": row["cell_id"],
                "statistic": row["statistic"],
                "estimate": row["estimate"],
                "conditioning_states": row["round_conditioning_state_count"],
                "rounds": row["n_rounds"],
                "singleton_fraction": row["round_singleton_fraction"],
                "dual_action_state_fraction": row["round_dual_action_state_fraction"],
                "dual_action_event_fraction": row["round_dual_action_event_fraction"],
                "support_limited": bool(
                    row["round_dual_action_state_fraction"] == 0
                    or not math.isfinite(float(row["estimate"]))
                ),
            }
            for row in memory_rows
        ],
    }
    (destination / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "DEFAULT_EPISTEMIC_BINS",
    "adapt_relational_round_record",
    "analyze_relational_imitation_round_feedback",
    "controller_action_summary",
    "epistemic_regime_summary",
    "read_relational_round_records",
]
