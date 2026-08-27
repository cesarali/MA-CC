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
    a deliberately coarse joint `(kappa_k, phi_k)` bin pair, reported under its
    own statistic name.  It exists because the full `E_k` conditioning can be
    too sparse to estimate on a small pilot, and a sparse exact answer must
    stay visible instead of being quietly replaced by a dense approximate one.

``conditioning_phi_bin`` / ``conditioning_susceptible_bin`` /
``conditioning_kappa_bin``
    three *scalar* coarse-grained epistemic variables, each on three
    interpretable bins (``low`` / ``medium`` / ``high``) and each conditioned on
    **separately** - never jointly, which would rebuild the sparsity they exist
    to avoid.  `phi_k = n_k^(L)/N` is the full-proof fraction, `s_k = 1 - phi_k`
    the socially susceptible fraction, `kappa_k` the mean supporting-fact
    coverage.  See `epistemic_conditioning_values`.

``delta_p_ctrl``
    the target share's per-round change, so the signed response can be read in
    share units as well as in aligned-magnetization units.

The second half of this module is the **matched classical reference**.  Every
completed run is compared against the exact finite-`N` controlled q-voter at
its own `(N, q, q_c, b, beta, theta)`, so no information number is ever
reported without a classical number at the same sensing and actuation
resources beside it.  The two systems are treated as two realizations of one
control protocol - sense, decide, actuate under budget - differing only in the
population-response kernel: explicit unanimity classically, implicit LLM with
persistent knowledge here.  The exact quantities live in `theory`; what this
file adds is the alignment, which means

- reading `(N, q, q_c, b, beta, theta)` off the round records and refusing to
  pool cells that disagree on them;
- weighting the exact local `T_qv(n)` by the occupancy the run actually
  visited, which is the primary comparator beside the empirical CMI;
- the empirical policy `a_hat_n` and response `Delta_mu_emp(n)` the classical
  curves are read against;
- a bootstrap of the *difference*, resampling episodes through the pipeline's
  own `bootstrap_episode_rows` while leaving the exact theory curve alone.

The point is not that the LLM population is a q-voter.  It is that a departure
should be attributable: to a controller that is not following its own policy,
to a population answering differently from a q-voter, or to the two processes
travelling through different regions of state space.  Those are separated in
`_interpretation` and named in the report.

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
from ...hidden_bench.imitation.controller import ADVOCATE_TARGET, NO_OP
from ...hidden_bench.imitation_round_feedback.analysis import (
    MAIN_ESTIMATOR_VARIANT,
    ROUND_ANALYSIS_STATISTICS,
    ROUND_MEMORY_STATISTICS,
    RoundEvent,
    _estimate_for,
    _signed_response,
    _support,
    _write_markdown,
    bootstrap_episode_rows,
    round_information_analysis,
)
from .state import ROUND_RECORD_TYPE
from .current import (
    current_analysis_comet_metrics,
    read_relational_micro_events,
    write_current_analysis,
)
from .matched_qvoter import (
    ClassicalReference,
    TheoryParameters,
    binary_entropy_bits,
    classical_reference,
    mean_field_transfer_entropy,
    q1_mean_response,
    theory_parameters_from_record,
)

DEFAULT_EPISTEMIC_BINS = 4
"""Bins per axis for the joint `(kappa, phi)` diagnostic conditioning.

Four is small on purpose.  This state exists to stay estimable when `E_k` does
not, so making it finer would defeat the only reason it is there.  Left at four
rather than harmonised with `COARSE_BINS` below so the statistic keeps emitting
exactly what it emitted before the scalar conditionings were added."""

COARSE_BINS = 3
COARSE_BIN_EDGES = (0.0, 1 / 3, 2 / 3, 1.0)
COARSE_BIN_LABELS = ("low", "medium", "high")
"""The three interpretable bins the scalar epistemic conditionings use.

    low     [0, 1/3)
    medium  [1/3, 2/3)
    high    [2/3, 1]

Half-open below, closed at the top, so `1.0` lands in `high` rather than off
the end.  Three rather than four because these variables exist to be estimable
at pilot sample sizes: each extra bin multiplies the number of conditioning
slices the CMI has to fill, and it is the *slices*, not the rounds, that run
out first.  No repository-wide share-binning utility exists to reuse -
`metrics.interactions.non_overlapping_bins` bins interaction indices and
`synthetic.empowerment.binning_matrix` bins occupation counts; neither is a
[0, 1] share binner."""

EPISTEMIC_CONDITIONING_VARIABLES = ("phi", "susceptible", "kappa")
"""The scalar epistemic memory variables, each conditioned on SEPARATELY.

Never jointly: the whole point of these is to be low-dimensional, and
conditioning on all three at once would rebuild exactly the sparsity that
`E_k` already suffers from."""

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


def _bin_label(index: int | None) -> str | None:
    """The readable name of a coarse bin, for the human-facing tables."""

    return None if index is None else COARSE_BIN_LABELS[int(index)]


def coarse_bin(value: float | None) -> int | None:
    """A share in `[0, 1]` as one of `COARSE_BIN_LABELS`, by index.

    Compared against `COARSE_BIN_EDGES` directly rather than by the
    multiply-and-truncate trick `_bin` uses, so the boundaries are exactly the
    documented half-open intervals instead of whichever way `value * 3` happens
    to round.  `2/3` is the case that decides it: it must be `high`.

    `None` in, `None` out, so a record that never recorded the quantity drops
    out of that statistic instead of being binned as zero.
    """

    if value is None or not math.isfinite(float(value)):
        return None
    number = float(value)
    for index, upper in enumerate(COARSE_BIN_EDGES[1:-1]):
        if number < upper:
            return index
    return len(COARSE_BIN_LABELS) - 1


def epistemic_conditioning_values(
    record: Mapping[str, Any],
) -> dict[str, float | None]:
    """`phi_k`, `s_k` and `kappa_k` at the START of the round.

    `phi_k = n_k^(L) / N` comes off the recorded epistemic memory histogram
    `E_k` directly - the last stratum is by definition the agents holding all
    `L` supporting facts - and falls back to the separately recorded
    `full_proof_agent_share_before` on a record written before `E_k` existed.
    The two agree by construction; `knowledge_observables` computes the share
    as "coverage >= 1.0", which is the same set of agents.

    `s_k = 1 - phi_k` is carried as its own variable rather than inlined. The
    arithmetic is a reflection, but the object is different: `s_k` is the
    fraction still *socially susceptible* - agents who do not yet hold the
    whole proof and can therefore still be moved by what they are told.

    `kappa_k` is the already-recorded `mean_supporting_fact_coverage_before`,
    not recomputed here.
    """

    memory = record.get("knowledge_stratum_counts_before")
    phi: float | None
    if memory:
        counts = [int(value) for value in memory]
        total = sum(counts)
        phi = counts[-1] / total if total else None
    else:
        phi = record.get("full_proof_agent_share_before")
        phi = None if phi is None else float(phi)
    kappa = record.get("mean_supporting_fact_coverage_before")
    return {
        "phi": phi,
        "susceptible": None if phi is None else 1.0 - phi,
        "kappa": None if kappa is None else float(kappa),
    }


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
    epistemic = epistemic_conditioning_values(record)
    total = sum(before)

    # The controller senses a *scalar*: how many of its q_c sampled agents
    # voted for the target.  The runtime records the whole sampled count
    # vector, so the scalar coordinate the single-affinity theory needs is
    # projected out here once rather than re-derived at every call site.
    sensor_vector = record.get("sensor_count_vector")
    sensor_target_count = (
        None
        if sensor_vector is None or target_index >= len(sensor_vector)
        else int(sensor_vector[target_index])
    )

    event = {
        **dict(record),
        "target_count_before": before[target_index],
        "target_count_after": after[target_index],
        "truth_count_before": before[truth_index],
        "truth_count_after": after[truth_index],
        "target_fraction_before": before[target_index] / total,
        "target_fraction_after": after[target_index] / total,
        "delta_p_ctrl": (after[target_index] - before[target_index]) / total,
        "delta_p_truth": (after[truth_index] - before[truth_index]) / total,
        "sensor_target_count": sensor_target_count,
        # Protocol parameters under the names the theory uses, so the single
        # affinity estimators never have to know the runtime's field spelling.
        "logged_advocacy_probability": record.get(
            "controller_advocate_probability",
            record.get("controller_advocacy_probability"),
        ),
        "beta": record.get("controller_beta"),
        "theta": record.get("controller_threshold"),
        "b": record.get("intervention_budget"),
        "conditioning_memory_state": (
            None if memory is None else [int(value) for value in memory]
        ),
        "conditioning_epistemic_state": (
            None if kappa_bin is None or phi_bin is None else [kappa_bin, phi_bin]
        ),
        # One key per scalar variable, each read by exactly one statistic in
        # `ROUND_MEMORY_CONDITIONING_KEYS`. Kept as separate keys, not one
        # vector, because conditioning on them jointly is the thing this is
        # meant to avoid.
        **{
            f"conditioning_{name}_bin": coarse_bin(value)
            for name, value in epistemic.items()
        },
        **{f"{name}_before": value for name, value in epistemic.items()},
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


def _partition_identical(rows: Sequence[RoundEvent]) -> bool:
    """Do the phi bins and the susceptible bins group the rounds identically?

    A one-to-one map between the two labellings means the two CMIs are the same
    estimate under different names.
    """

    pairs = {
        (row.event.get("conditioning_phi_bin"), row.event.get("conditioning_susceptible_bin"))
        for row in rows
    }
    forward = {phi for phi, _ in pairs}
    backward = {susceptible for _, susceptible in pairs}
    return len(pairs) == len(forward) == len(backward)


# --------------------------------------------------------------------------
# Matched classical q-voter reference
#
# Everything below compares what this run measured against the exact finite-N
# controlled q-voter at the SAME controller parameters. It adds no estimator:
# the empirical side is `_estimate_for` and `_signed_response` from the shared
# pipeline and the resampling is the same `bootstrap_episode_rows`; the
# classical side is the deterministic `theory` module. What is new is only the
# alignment between them.
# --------------------------------------------------------------------------

TE_RATIO_FLOOR = 1e-6
"""Bits below which the classical TE is treated as a zero denominator.

`rho_T = T_emp / T_qv` is reported as undefined rather than as a large number
whenever the classical channel is numerically closed, because a ratio against
nothing is not a measurement of anything."""

POLICY_CALIBRATION_TOLERANCE = 0.15
COMPARISON_RELATIVE_TOLERANCE = 0.5
"""Thresholds for the DESCRIPTIVE labels in `theory_interpretation` only.

They decide which sentence the report prints, never what the numbers are, and
nothing downstream reads them. `0.15` is in probability units on the
occupancy-weighted policy MAE; `0.5` is a relative tolerance, so "comparable"
means the two agree within a factor of two either way. Both are round numbers
chosen to be legible rather than calibrated, which is exactly why they gate a
label and not an inference."""

THEORY_BOOTSTRAP_SEED_OFFSET = 7_000_000
"""Keeps the empirical-vs-theory resampling stream clear of the pipeline's.

`round_information_analysis` seeds statistic `i`'s bootstrap from `seed + i`
and its null from `seed + 100_000 * (i + 1)`. This offset sits above every
stream those can reach, so the residual CI is not silently correlated with a
CI it is meant to be compared against."""


def _controlled(rows: Sequence[RoundEvent]) -> list[RoundEvent]:
    """The rows the target CMI is actually estimated from.

    The same eligibility the shared pipeline applies, so the occupancy the
    theory is weighted over is the occupancy of the states the empirical
    estimate saw - not of every round in the file.
    """

    return [row for row in rows if row.U_k in {ADVOCATE_TARGET, NO_OP}]


def matched_qvoter_parameters_for(
    rows: Sequence[RoundEvent],
) -> tuple[TheoryParameters | None, str | None]:
    """The one matched protocol these rows share, or `(None, reason)`.

    A single tuple is required rather than assumed. Cells that differ in
    `(N, q, q_c, b, beta, theta)` have different classical references, and
    averaging their theory into one number would compare the run against a
    process that was never run - so a mixed group refuses instead, and the
    caller reports the refusal.
    """

    found: dict[tuple[int, int, int, int, float, float], TheoryParameters] = {}
    for row in rows:
        parameters = theory_parameters_from_record(row.event)
        if parameters is not None:
            found[parameters.key] = parameters
    if not found:
        return None, "the run records no controller sensing/actuation parameters"
    if len(found) > 1:
        return None, (
            f"rows span {len(found)} distinct (N, q, q_c, b, beta, theta) tuples; "
            "a single classical reference does not apply"
        )
    return next(iter(found.values())), None


def empirical_round_occupancy(rows: Sequence[RoundEvent], N: int) -> np.ndarray:
    """`P_k(n)` for each round index present, as a `(K, N+1)` array.

    One distribution per round index, estimated across the episodes in the
    group - not one pooled histogram. Episodes can stop early, so a pooled
    count would silently weight the rounds that more episodes reached; this
    keeps each round's occupancy its own object, which is also what the
    finite-horizon average in `finite_horizon_occupancy` needs.
    """

    by_round: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        by_round[row.round_index].append(row.target_before)
    occupancy = np.zeros((len(by_round), N + 1), dtype=float)
    for index, round_index in enumerate(sorted(by_round)):
        counts = by_round[round_index]
        for value in counts:
            occupancy[index, int(value)] += 1.0
        occupancy[index] /= len(counts)
    return occupancy


def finite_horizon_occupancy(rows: Sequence[RoundEvent], N: int) -> np.ndarray:
    """`(1/K) sum_k P_k(n)` - the weighting of the primary theory scalar.

    Equal weight per round index rather than per observation, so a horizon
    whose late rounds only a few episodes reached does not have those rounds
    quietly down-weighted out of the comparison.
    """

    occupancy = empirical_round_occupancy(rows, N)
    return occupancy.mean(axis=0) if len(occupancy) else np.zeros(N + 1, dtype=float)


def empirical_policy_curve(
    rows: Sequence[RoundEvent], N: int
) -> tuple[np.ndarray, np.ndarray]:
    """`(advocate_counts, no_op_counts)` indexed by the target count `n`."""

    advocate = np.zeros(N + 1, dtype=float)
    no_op = np.zeros(N + 1, dtype=float)
    for row in _controlled(rows):
        bucket = advocate if row.U_k == ADVOCATE_TARGET else no_op
        bucket[row.target_before] += 1.0
    return advocate, no_op


def empirical_response_curve(
    rows: Sequence[RoundEvent], N: int
) -> dict[int, dict[str, Any]]:
    """`E[dx | U, n]` per state and action, from the recorded `delta_p_ctrl`.

    At fixed `n` the current share is fixed too, so the difference of the two
    conditional means IS the action-induced mean separation `Delta_mu_emp(n)` -
    no separate before/after bookkeeping is needed.
    """

    buckets: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {ADVOCATE_TARGET: [], NO_OP: []}
    )
    for row in _controlled(rows):
        delta = row.event.get("delta_p_ctrl")
        if delta is not None:
            buckets[row.target_before][str(row.U_k)].append(float(delta))
    result: dict[int, dict[str, Any]] = {}
    for state, actions in buckets.items():
        advocated, no_op = actions[ADVOCATE_TARGET], actions[NO_OP]
        result[state] = {
            "advocate_mean": float(np.mean(advocated)) if advocated else math.nan,
            "no_op_mean": float(np.mean(no_op)) if no_op else math.nan,
            "advocate_count": len(advocated),
            "no_op_count": len(no_op),
            "dual_action": bool(advocated and no_op),
            "delta_mu_emp": (
                float(np.mean(advocated)) - float(np.mean(no_op))
                if advocated and no_op
                else math.nan
            ),
        }
    return result


def matched_qvoter_state_curves(
    rows: Sequence[RoundEvent], reference: ClassicalReference, *, cell_id: str
) -> list[dict[str, Any]]:
    """The state-resolved table behind every plot in the report bundle.

    One row per `n = 0..N`, carrying the exact curves, the empirical curves
    where support exists, the residuals, and the per-state action counts that
    say whether a residual may be read at all. Deliberately no per-state
    empirical MI: at pilot sample sizes those would be noise dressed as a
    curve, and the honest substitute - the occupancy histogram beside the exact
    local TE - is here instead.
    """

    parameters = reference.parameters
    N, q, b = parameters.N, parameters.q, parameters.b
    advocate, no_op = empirical_policy_curve(rows, N)
    response = empirical_response_curve(rows, N)
    occupancy = finite_horizon_occupancy(_controlled(rows), N)
    ceiling = reference.entropy_ceiling()
    curves = []
    for n in range(N + 1):
        x = n / N
        observed = advocate[n] + no_op[n]
        empirical_a = advocate[n] / observed if observed else math.nan
        theoretical_a = float(reference.advocacy[n])
        state_response = response.get(n, {})
        mean_field = mean_field_transfer_entropy(
            x, q=q, c=parameters.actuation_fraction, N=N, advocacy=theoretical_a
        )
        delta_mu_emp = float(state_response.get("delta_mu_emp", math.nan))
        delta_mu_theory = float(reference.mean_response[n])
        curves.append(
            {
                "cell_id": cell_id,
                "n_target": n,
                "x": x,
                # --- controller policy (section 4) ------------------------
                "a_n_theory": theoretical_a,
                "a_n_empirical": empirical_a,
                "a_n_residual": empirical_a - theoretical_a,
                "advocate_observations": int(advocate[n]),
                "no_op_observations": int(no_op[n]),
                "observations": int(observed),
                # --- occupancy (section 15.4) -----------------------------
                "empirical_occupancy": float(occupancy[n]),
                # --- local transfer entropy (section 7) -------------------
                "local_te_theory_bits": float(reference.local_te[n]),
                "local_te_entropy_ceiling_bits": float(ceiling[n]),
                "mean_field_te_bits": mean_field,
                # The weak-separation expansion assumes `c` is small. `h2(a_n)`
                # is an exact bound on the true local TE, so an approximation
                # above it has provably left its validity range - a cheap,
                # principled reliability check that needs no extra threshold.
                "mean_field_te_within_ceiling": bool(
                    math.isfinite(mean_field) and mean_field <= float(ceiling[n]) + 1e-9
                ),
                # --- response (sections 10 and 15.2) ----------------------
                "delta_mu_theory": delta_mu_theory,
                # The closed form exists only at q = 1. Where it exists it must
                # agree with the kernel column above, which is why both are
                # printed rather than one being derived from the other.
                "delta_mu_theory_q1_closed_form": (
                    q1_mean_response(x, N=N, b=b) if q == 1 else math.nan
                ),
                "delta_mu_empirical": delta_mu_emp,
                "delta_mu_residual": delta_mu_emp - delta_mu_theory,
                "response_advocate_mean": float(
                    state_response.get("advocate_mean", math.nan)
                ),
                "response_no_op_mean": float(
                    state_response.get("no_op_mean", math.nan)
                ),
                "dual_action_state": bool(state_response.get("dual_action", False)),
            }
        )
    return curves


def _weighted_response(
    curves: Sequence[Mapping[str, Any]], field: str
) -> tuple[float, float]:
    """One response scalar and its weight, over dual-action states only.

    The weights are `n_ADV + n_NO_OP` per state, which is exactly the weighting
    `_signed_response` uses - so the empirical aggregate computed here equals
    the pipeline's own signed response, and the matched classical aggregate is
    guaranteed to be read over the same states with the same emphasis. A state
    with only one observed action carries no separation to average and is
    therefore absent from both.
    """

    total = 0.0
    weight = 0.0
    for row in curves:
        if not row["dual_action_state"]:
            continue
        value = float(row[field])
        if not math.isfinite(value):
            continue
        size = float(row["advocate_observations"] + row["no_op_observations"])
        total += size * value
        weight += size
    return (math.nan if weight == 0 else total / weight), weight


def _policy_calibration(
    curves: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Occupancy-weighted `|a_hat_n - a_n|` and its RMS counterpart.

    Weighted by how often each state was actually visited, so a wild residual
    at a state seen twice cannot dominate a calibration statement about a run
    that spent its time elsewhere.
    """

    absolute = 0.0
    square = 0.0
    weight = 0.0
    for row in curves:
        residual = float(row["a_n_residual"])
        if not math.isfinite(residual):
            continue
        size = float(row["observations"])
        absolute += size * abs(residual)
        square += size * residual * residual
        weight += size
    if weight == 0:
        return {"policy_mae": math.nan, "policy_rmse": math.nan}
    return {
        "policy_mae": absolute / weight,
        "policy_rmse": math.sqrt(square / weight),
    }


def _theory_residual_bootstrap(
    rows: Sequence[RoundEvent],
    reference: ClassicalReference,
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    """A CI for `T_emp - T_qv^emp-occ`, treating the local theory as exact.

    Section 16 of the plan, literally: each replicate resamples whole episodes
    with the pipeline's own `bootstrap_episode_rows`, then recomputes BOTH the
    empirical CMI and the occupancy weighting on that same replicate. The
    `T_qv(n)` curve itself is never resampled - it has no sampling uncertainty
    - so what moves between replicates is only which states the resampled
    episodes visited and what the estimator read there. That is the whole point:
    the residual has a confidence interval even though half of it is exact.
    """

    eligible = _controlled(rows)
    alpha = (1.0 - confidence) / 2.0
    differences: list[float] = []
    ratios: list[float] = []
    for draw in bootstrap_episode_rows(
        eligible, resamples=resamples, seed=seed + THEORY_BOOTSTRAP_SEED_OFFSET
    ):
        empirical = float(
            getattr(
                _estimate_for("round_target_actuation_cmi", draw),
                MAIN_ESTIMATOR_VARIANT,
            )
        )
        theoretical = reference.occupancy_weighted_te(
            finite_horizon_occupancy(draw, reference.parameters.N)
        )
        if not (math.isfinite(empirical) and math.isfinite(theoretical)):
            continue
        differences.append(empirical - theoretical)
        if theoretical > TE_RATIO_FLOOR:
            ratios.append(empirical / theoretical)
    def interval(values: list[float], prefix: str) -> dict[str, float]:
        if not values:
            return {f"{prefix}_ci_low": math.nan, f"{prefix}_ci_high": math.nan}
        return {
            f"{prefix}_ci_low": float(np.quantile(values, alpha)),
            f"{prefix}_ci_high": float(np.quantile(values, 1.0 - alpha)),
        }
    return {
        **interval(differences, "delta_te"),
        **interval(ratios, "te_ratio"),
        "delta_te_bootstrap_replicates": len(differences),
    }


def _interpretation(
    *,
    identifiable: bool,
    policy_mae: float,
    empirical_te: float,
    theory_te: float,
    self_te: float,
    delta_mu_emp: float,
    delta_mu_theory: float,
) -> str:
    """The deterministic descriptive label of section 22.

    Descriptive, not inferential: it names which of three scientifically
    different things the numbers look like, and nothing reads it back. The
    three are worth keeping apart because they fail in different places - a
    controller that is not doing what the policy says, a population that
    answers the controller differently from a q-voter, and two processes that
    simply travel through different regions of state space.
    """

    if not identifiable:
        return "degenerate_no_action_variation"
    if not math.isfinite(policy_mae) or policy_mae > POLICY_CALIBRATION_TOLERANCE:
        return "controller_policy_mismatch"

    def comparable(empirical: float, theoretical: float) -> bool:
        if not (math.isfinite(empirical) and math.isfinite(theoretical)):
            return False
        scale = max(abs(theoretical), TE_RATIO_FLOOR)
        return abs(empirical - theoretical) <= COMPARISON_RELATIVE_TOLERANCE * scale

    response_close = comparable(delta_mu_emp, delta_mu_theory)
    te_close = comparable(empirical_te, theory_te)
    if te_close and response_close:
        return "channel_comparable_to_classical"
    # Order matters here, and it is the order of section 22. The response is
    # the LOCAL comparison: if it departs, the population is answering the
    # controller differently from a q-voter, and that is the finding no matter
    # what the finite-horizon scalar does. Only once the local response does
    # match is a TE gap allowed to be blamed on the two processes having
    # visited different regions - which is checkable, by asking whether the
    # classical process's own occupancy would have produced a different
    # scalar from the same exact local curve.
    if not response_close:
        return "population_kernel_departure"
    if math.isfinite(self_te) and not comparable(self_te, theory_te):
        return "occupancy_departure"
    return "population_kernel_departure"


def _epistemic_interpretation(
    estimates: Sequence[Mapping[str, Any]]
) -> tuple[str, dict[str, float]]:
    """Cases D and E: does the channel survive conditioning on what agents know?

    The classical reference is deliberately memoryless, so it has nothing to
    say here. This reads the epistemic-conditioned CMIs the pipeline already
    produced and asks only whether any of them still stands above its own
    policy-conditional null. Support-limited rows are excluded rather than
    counted as a disappearance - an estimate that could not be made is not
    evidence that the channel is gone.
    """

    values: dict[str, float] = {}
    survives = False
    informative = False
    for row in estimates:
        name = str(row.get("statistic"))
        if name not in ROUND_MEMORY_STATISTICS:
            continue
        estimate = float(row.get("estimate", math.nan))
        null_mean = float(row.get("null_mean", math.nan))
        values[f"epistemic_{name}"] = estimate
        supported = float(row.get("round_dual_action_state_fraction", 0.0)) > 0.0
        if supported and math.isfinite(estimate) and math.isfinite(null_mean):
            informative = True
            if estimate > null_mean:
                survives = True
    if not informative:
        return "epistemic_conditioning_unsupported", values
    return (
        "channel_survives_epistemic_conditioning"
        if survives
        else "channel_explained_by_epistemic_state"
    ), values


def matched_qvoter_comparison(
    rows: Sequence[RoundEvent],
    estimates: Sequence[Mapping[str, Any]],
    *,
    cell_id: str,
    bootstrap_resamples: int,
    confidence: float,
    seed: int,
    self_occupancy: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One cell's `(comparison row, state-resolved curves)`.

    The empirical half is taken from the estimates the shared pipeline already
    produced - the CMI, its CI, its null and its support all come off that row
    rather than being re-estimated here, so the number printed beside the
    theory is provably the number in the MI table.
    """

    parameters, skip = matched_qvoter_parameters_for(rows)
    if parameters is None:
        return (
            {
                "cell_id": cell_id,
                "theory_applicable": False,
                "theory_skip_reason": skip,
            },
            [],
        )
    reference = classical_reference(parameters)
    eligible = _controlled(rows)
    curves = matched_qvoter_state_curves(rows, reference, cell_id=cell_id)
    N = parameters.N

    cmi_row: Mapping[str, Any] = next(
        (
            row
            for row in estimates
            if row.get("statistic") == "round_target_actuation_cmi"
        ),
        {},
    )
    empirical_te = float(cmi_row.get("estimate", math.nan))

    occupancy = finite_horizon_occupancy(eligible, N)
    theory_te = reference.occupancy_weighted_te(occupancy)

    # Section 18: with only one action ever taken the CMI is zero by
    # construction, not by measurement. The response reference below stays
    # meaningful, so the run is still compared - only the TE line is marked.
    actions = {row.U_k for row in eligible}
    identifiable = len(actions) > 1

    delta_te = empirical_te - theory_te
    ratio = (
        empirical_te / theory_te if theory_te > TE_RATIO_FLOOR else math.nan
    )

    self_te = math.nan
    if self_occupancy and len(eligible):
        rounds_seen = len({row.round_index for row in eligible})
        initial = empirical_round_occupancy(eligible, N)[0]
        trajectory = reference.self_occupancy(initial, rounds_seen)
        self_te = float(
            np.mean([reference.occupancy_weighted_te(step) for step in trajectory])
        )

    delta_mu_emp, response_weight = _weighted_response(curves, "delta_mu_empirical")
    delta_mu_theory, _ = _weighted_response(curves, "delta_mu_theory")
    calibration = _policy_calibration(curves)

    advocate_rounds = sum(1 for row in eligible if row.U_k == ADVOCATE_TARGET)
    advocate_fraction = (
        advocate_rounds / len(eligible) if eligible else math.nan
    )
    theory_advocacy = float(occupancy @ reference.advocacy)
    mean_field_te = float(
        np.nansum(
            [row["empirical_occupancy"] * row["mean_field_te_bits"] for row in curves]
        )
    )
    epistemic_label, epistemic_values = _epistemic_interpretation(estimates)

    row = {
        "cell_id": cell_id,
        "theory_applicable": True,
        "theory_skip_reason": None,
        # The classical comparison is a binary projection of a three-option
        # task; recorded on every row so it cannot be read as anything else.
        "theory_state_coarse_graining": "target_vs_not_target",
        **parameters.as_fields(),
        # --- A. empirical information channel, quoted from the MI table ----
        "empirical_target_cmi_bits": empirical_te,
        "empirical_target_cmi_ci_low": float(
            cmi_row.get("bootstrap_ci_low", math.nan)
        ),
        "empirical_target_cmi_ci_high": float(
            cmi_row.get("bootstrap_ci_high", math.nan)
        ),
        "empirical_target_cmi_null_mean": float(cmi_row.get("null_mean", math.nan)),
        "empirical_target_cmi_null_type": cmi_row.get("null_type"),
        "empirical_target_cmi_excess_over_null": (
            empirical_te - float(cmi_row.get("null_mean", math.nan))
        ),
        "conditional_action_entropy_bits": float(
            cmi_row.get("conditional_action_entropy_bits", math.nan)
        ),
        # --- B. matched classical reference --------------------------------
        "theory_te_emp_occ_bits": theory_te,
        "theory_te_self_occ_bits": self_te,
        "theory_occupancy_weighted_advocacy": theory_advocacy,
        "theory_occupancy_weighted_entropy_ceiling_bits": float(
            occupancy @ np.array([binary_entropy_bits(a) for a in reference.advocacy])
        ),
        "mean_field_te_bits": mean_field_te,
        "mean_field_te_within_ceiling": bool(
            mean_field_te
            <= float(
                occupancy
                @ np.array([binary_entropy_bits(a) for a in reference.advocacy])
            )
            + 1e-9
        ),
        # --- C. empirical vs theory ----------------------------------------
        "delta_te_bits": delta_te,
        "te_ratio": ratio,
        "te_ratio_defined": bool(theory_te > TE_RATIO_FLOOR),
        "delta_mu_empirical": delta_mu_emp,
        "delta_mu_theory": delta_mu_theory,
        "delta_mu_residual": delta_mu_emp - delta_mu_theory,
        "response_comparison_observations": response_weight,
        **calibration,
        "empirical_advocate_fraction": advocate_fraction,
        # --- section 12 resource descriptors (NOT an efficiency) -----------
        "realized_actuation_slots_per_round": (
            parameters.b * advocate_fraction
            if math.isfinite(advocate_fraction)
            else math.nan
        ),
        "sensing_observations_per_round": parameters.q_c,
        # --- support, quoted from the same MI row --------------------------
        "n_rounds": int(cmi_row.get("n_rounds", len(eligible))),
        "n_episodes": int(
            cmi_row.get("n_episodes", len({row.episode_id for row in eligible}))
        ),
        "round_conditioning_state_count": cmi_row.get(
            "round_conditioning_state_count"
        ),
        "round_singleton_fraction": cmi_row.get("round_singleton_fraction"),
        "round_dual_action_state_fraction": cmi_row.get(
            "round_dual_action_state_fraction"
        ),
        "round_dual_action_event_fraction": cmi_row.get(
            "round_dual_action_event_fraction"
        ),
        "te_comparison_identifiable": identifiable,
        "number_of_actions_observed": len(actions),
        **_theory_residual_bootstrap(
            rows,
            reference,
            resamples=bootstrap_resamples,
            confidence=confidence,
            seed=seed,
        ),
        **epistemic_values,
        "epistemic_interpretation": epistemic_label,
    }
    row["theory_interpretation"] = _interpretation(
        identifiable=identifiable,
        policy_mae=calibration["policy_mae"],
        empirical_te=empirical_te,
        theory_te=theory_te,
        self_te=self_te,
        delta_mu_emp=delta_mu_emp,
        delta_mu_theory=delta_mu_theory,
    )
    return row, curves


THEORY_INTERPRETATION_PROSE = {
    "degenerate_no_action_variation": (
        "The controller never varied its action, so I(U;n'|n) = 0 by "
        "construction and the TE comparison is not identifiable. This is a "
        "response/susceptibility run, not a stochastic-feedback TE run; zero "
        "TE here does not mean zero behavioral control."
    ),
    "controller_policy_mismatch": (
        "The empirical action frequencies do not track the exact a_n. Treat "
        "this as a controller implementation/calibration finding first - the "
        "response and TE comparisons below are conditioned on a controller "
        "that is not doing what the matched theory assumes."
    ),
    "channel_comparable_to_classical": (
        "The LLM collective exhibits a control channel similar in scale and "
        "shape to the matched classical imitation reference over the visited "
        "state range."
    ),
    "population_kernel_departure": (
        "The controller is calibrated, but the LLM population response kernel "
        "departs from the q-voter reference. This is where reasoning, semantic "
        "and epistemic effects would live."
    ),
    "occupancy_departure": (
        "Much of the discrepancy is attributable to the two processes visiting "
        "different parts of state space rather than to a purely local "
        "channel-strength difference."
    ),
}

EPISTEMIC_INTERPRETATION_PROSE = {
    "channel_explained_by_epistemic_state": (
        "The apparent controller-to-population information channel is largely "
        "explained by the population's epistemic state at the chosen coarse "
        "graining."
    ),
    "channel_survives_epistemic_conditioning": (
        "A directed controller-to-population information channel remains after "
        "accounting for the selected epistemic macrostate."
    ),
    "epistemic_conditioning_unsupported": (
        "The epistemic-conditioned CMIs have no dual-action support at this "
        "sample size, so nothing can be concluded from them either way."
    ),
}


def _number(value: Any, digits: str = ".6g") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return "undefined" if not math.isfinite(float(value)) else format(value, digits)
    return str(value)


def append_theory_report(
    path: Path,
    comparisons: Sequence[Mapping[str, Any]],
    *,
    curves_filename: str,
    table_filename: str,
) -> None:
    """Append the matched-classical section to the EXISTING MI report.

    Appended to `round_information_estimates.md` rather than written as its own
    document on purpose. The whole point of the exercise is that no completed
    run has an information number without a classical number beside it, and a
    second file is a second thing to remember to open.
    """

    lines = [
        "",
        "## Matched classical q-voter reference",
        "",
        "Every empirical row above is compared here against the exact "
        "finite-N controlled q-voter at the *same* controller parameters "
        "`(N, q, q_c, b, beta, theta)`. The classical kernels, the "
        "state-conditioned advocacy probability `a_n` and the state-local "
        "transfer entropy `T_qv(n)` are deterministic - they carry no sampling "
        "uncertainty and are never bootstrapped. What is resampled is the "
        "empirical occupancy the theory is read over, and the empirical CMI.",
        "",
        "The classical reference is binary, the relational task has three "
        "answer options, so the comparison is a **target-count coarse "
        "graining**: the state is the number of agents voting for the "
        "controller's target, against everything else.",
        "",
        f"State-resolved curves (policy, response, local TE, occupancy): "
        f"`{curves_filename}`. Full per-cell fields: `{table_filename}`.",
        "",
    ]
    for row in comparisons:
        lines.append(f"### `{row['cell_id']}`")
        lines.append("")
        if not row.get("theory_applicable"):
            lines.extend(
                [
                    "```text",
                    "MATCHED CLASSICAL REFERENCE",
                    "---------------------------",
                    f"skipped: {row.get('theory_skip_reason')}",
                    "```",
                    "",
                ]
            )
            continue
        ratio = (
            _number(row["te_ratio"])
            if row.get("te_ratio_defined")
            else "undefined (classical TE is numerically zero)"
        )
        lines.extend(
            [
                "```text",
                "MATCHED CLASSICAL REFERENCE",
                "---------------------------",
                f"N={row['theory_N']}",
                f"q={row['theory_q']}",
                f"q_c={row['theory_qc']}",
                f"b={row['theory_b']}",
                f"c=b/N={_number(row['theory_c'])}",
                f"beta={_number(row['theory_beta'])}",
                f"theta={_number(row['theory_theta'])}",
                f"r_sense=q_c/N={_number(row['theory_sensing_fraction'])}",
                "",
                "Empirical target CMI:",
                f"    I(U_k ; n_Z,k+1 | n_Z,k) = "
                f"{_number(row['empirical_target_cmi_bits'])} bits",
                f"    bootstrap 95% CI = [{_number(row['empirical_target_cmi_ci_low'])}, "
                f"{_number(row['empirical_target_cmi_ci_high'])}]",
                f"    null = {_number(row['empirical_target_cmi_null_mean'])}"
                f" ({row['empirical_target_cmi_null_type']})",
                f"    excess over null = "
                f"{_number(row['empirical_target_cmi_excess_over_null'])}",
                f"    entropy ceiling H(U|n) = "
                f"{_number(row['conditional_action_entropy_bits'])}",
                "",
                "Classical exact TE, empirical-occupancy weighted:",
                f"    T_qv_emp_occ = {_number(row['theory_te_emp_occ_bits'])} bits",
                f"    [secondary] classical self-occupancy T_qv_self = "
                f"{_number(row['theory_te_self_occ_bits'])} bits",
                f"    [diagnostic] mean-field approximation = "
                f"{_number(row['mean_field_te_bits'])} bits"
                + (
                    ""
                    if row["mean_field_te_within_ceiling"]
                    else "  <- OUTSIDE VALIDITY RANGE: exceeds the exact h2(a_n) "
                    "ceiling, so the weak-control expansion does not apply at "
                    "this c; read the exact T_qv above instead"
                ),
                "",
                "Difference:",
                f"    empirical - classical = {_number(row['delta_te_bits'])} bits",
                f"    bootstrap 95% CI = [{_number(row['delta_te_ci_low'])}, "
                f"{_number(row['delta_te_ci_high'])}]",
                "",
                "Diagnostic ratio:",
                f"    empirical / classical = {ratio}",
                "    [NOT an efficiency]",
                "",
                "Empirical signed response:",
                f"    Delta_mu_emp = {_number(row['delta_mu_empirical'])}",
                "",
                "Matched classical response:",
                f"    Delta_mu_qv = {_number(row['delta_mu_theory'])}",
                f"    residual = {_number(row['delta_mu_residual'])}",
                "",
                "Controller-policy calibration:",
                f"    empirical ADV fraction = "
                f"{_number(row['empirical_advocate_fraction'])}",
                f"    theory occupancy-weighted ADV probability = "
                f"{_number(row['theory_occupancy_weighted_advocacy'])}",
                f"    statewise policy MAE = {_number(row['policy_mae'])}",
                f"    statewise policy RMSE = {_number(row['policy_rmse'])}",
                "",
                "Resources (descriptors only, no efficiency is defined):",
                f"    realized actuation slots per round = "
                f"{_number(row['realized_actuation_slots_per_round'])}",
                f"    sensing observations per round = "
                f"{row['sensing_observations_per_round']}",
                "",
                "Support:",
                f"    dual-action-state fraction = "
                f"{_number(row['round_dual_action_state_fraction'])}",
                f"    singleton fraction = "
                f"{_number(row['round_singleton_fraction'])}",
                f"    overlap observation fraction = "
                f"{_number(row['round_dual_action_event_fraction'])}",
                f"    actions observed = {row['number_of_actions_observed']}",
                f"    TE comparison identifiable = "
                f"{row['te_comparison_identifiable']}",
                "```",
                "",
                f"**{row['theory_interpretation']}** - "
                f"{THEORY_INTERPRETATION_PROSE[row['theory_interpretation']]}",
                "",
                f"**{row['epistemic_interpretation']}** - "
                f"{EPISTEMIC_INTERPRETATION_PROSE[row['epistemic_interpretation']]}",
                "",
            ]
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _export_current_analysis_to_comet(
    rows: Sequence[Mapping[str, Any]],
    assets: Sequence[Path],
    *,
    enabled: bool,
    project_name: str,
    run_name: str,
    sink: Any | None = None,
    name_suffix: str | None = None,
) -> dict[str, Any]:
    """Publish aggregate current values from the master/post-hoc layer only."""

    if not enabled:
        return {
            "status": "disabled",
            "metrics": 0,
            "assets": 0,
            "url": None,
            "published_to": None,
        }
    borrowed = sink is not None
    if sink is None:
        from mas_cc.observability.recorder import CometMetricSink

        sink = CometMetricSink(True, project_name=project_name, run_name=run_name)
    metrics = current_analysis_comet_metrics(rows)
    uploaded = 0
    try:
        sink.add_tags(("analysis", "relational", "current"))
        if metrics:
            sink.log_metrics(metrics, 0)
        for asset in assets:
            if not asset.is_file():
                continue
            asset_name = asset.name
            if name_suffix:
                asset_name = f"{asset.stem}__{name_suffix}{asset.suffix}"
            sink.log_asset(asset, name=asset_name)
            uploaded += 1
        return {
            "status": sink.status,
            "metrics": len(metrics),
            "assets": uploaded,
            "url": sink.url,
            "published_to": "master" if borrowed else "analysis_experiment",
        }
    finally:
        if not borrowed:
            sink.close()


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
    theoretical_reference: str = "single_affinity_revised",
    theory_comparison_enabled: bool | None = None,
    comet_export: bool = False,
    comet_project: str = "mas-cc",
    comet_run_name: str | None = None,
    comet_sink: Any | None = None,
    comet_name_suffix: str | None = None,
) -> dict[str, Any]:
    """Run the shared round-feedback pipeline over a relational grid.

    Revised single-affinity theory is the canonical default.  ``none`` keeps
    all empirical analysis and suppresses theory; ``matched_qvoter_null`` is
    an explicitly separate classical diagnostic.  The boolean argument is a
    one-release compatibility shim and never restores the old generic output.
    """

    allowed_references = {
        "single_affinity_revised",
        "none",
        "matched_qvoter_null",
    }
    if theory_comparison_enabled is not None:
        theoretical_reference = (
            "single_affinity_revised" if theory_comparison_enabled else "none"
        )
    if theoretical_reference not in allowed_references:
        raise ValueError(
            "theoretical_reference must be one of: "
            + ", ".join(sorted(allowed_references))
        )

    rounds = read_relational_round_records(run_dir, epistemic_bins=epistemic_bins)
    micro = read_relational_micro_events(run_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    estimates: list[dict[str, Any]] = []
    nulls: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    regimes: list[dict[str, Any]] = []
    theory_rows: list[dict[str, Any]] = []
    classical_null_rows: list[dict[str, Any]] = []
    classical_null_curves: list[dict[str, Any]] = []
    from mas_cc.analysis.single_affinity import (
        PROVENANCE as SINGLE_AFFINITY_PROVENANCE,
        theory_comparison as single_affinity_theory_comparison,
    )
    from mas_cc.storage import canonical_hash

    micro_by_cell: dict[str, list[dict[str, Any]]] = {}
    for item in micro:
        micro_by_cell.setdefault(str(item.get("cell_id", "run")), []).append(item)
    comparison_hash = canonical_hash(
        {
            "rounds": len(rounds),
            "micro_slots": len(micro),
            "theory": dict(SINGLE_AFFINITY_PROVENANCE),
        }
    )
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
        if theoretical_reference == "single_affinity_revised":
            comparison = single_affinity_theory_comparison(
                group, micro_by_cell.get(cell_id, ())
            )
            common = {
                "study_id": None,
                "source_run_id": Path(run_dir).name,
                "cell_id": cell_id,
                **SINGLE_AFFINITY_PROVENANCE,
                "analysis_hash": comparison_hash,
            }
            if comparison["available"]:
                theory_rows.extend(
                    {**common, **item, "available": True, "reason": None}
                    for item in comparison["rows"]
                )
            else:
                theory_rows.append(
                    {
                        **common,
                        "quantity": None,
                        "units": None,
                        "empirical": math.nan,
                        "single_affinity_theory": math.nan,
                        "residual": math.nan,
                        "reference": "single_affinity_revised",
                        "available": False,
                        "reason": comparison["reason"],
                    }
                )
        elif theoretical_reference == "matched_qvoter_null":
            comparison, curves = matched_qvoter_comparison(
                group,
                cell_estimates,
                cell_id=cell_id,
                bootstrap_resamples=bootstrap_resamples,
                confidence=confidence,
                seed=seed,
            )
            classical_null_rows.append(
                {**comparison, "reference": "matched_qvoter_classical_null"}
            )
            classical_null_curves.extend(curves)
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
    if theoretical_reference == "matched_qvoter_null":
        # The optional diagnostic retains the legacy pooled refusal semantics;
        # revised theory is never constructed across physical cells.
        pooled_comparison, pooled_curves = matched_qvoter_comparison(
            rounds,
            pooled_estimates,
            cell_id="pooled",
            bootstrap_resamples=bootstrap_resamples,
            confidence=confidence,
            seed=seed,
        )
        classical_null_rows.append(
            {**pooled_comparison, "reference": "matched_qvoter_classical_null"}
        )
        classical_null_curves.extend(pooled_curves)

    pd.DataFrame(estimates).to_csv(
        destination / "round_information_estimates.csv", index=False
    )
    report = destination / "round_information_estimates.md"
    _write_markdown(estimates, report)
    if theory_rows:
        pd.DataFrame(theory_rows).to_csv(
            destination / "single_affinity_theory_comparison.csv", index=False
        )
    if classical_null_rows:
        pd.DataFrame(classical_null_rows).to_csv(
            destination / "matched_qvoter_null.csv", index=False
        )
        pd.DataFrame(classical_null_curves).to_csv(
            destination / "matched_qvoter_null_state_curves.csv", index=False
        )
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

    current_rows, current_episodes, current_reports = write_current_analysis(
        rounds,
        destination,
        bootstrap_resamples=bootstrap_resamples,
        confidence=confidence,
        seed=seed,
        theoretical_reference=(
            theoretical_reference
            if theoretical_reference in {"single_affinity_revised", "none"}
            else "none"
        ),
        micro=micro,
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
                "kappa_before": row.event.get("kappa_before"),
                "kappa_after": row.event.get("mean_supporting_fact_coverage"),
                "phi_before": row.event.get("phi_before"),
                "phi_after": row.event.get("full_proof_agent_share"),
                "susceptible_before": row.event.get("susceptible_before"),
                "E_k": row.memory_state,
                "epistemic_bin": row.epistemic_state,
                **{
                    f"{name}_bin": _bin_label(row.event.get(f"conditioning_{name}_bin"))
                    for name in EPISTEMIC_CONDITIONING_VARIABLES
                },
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
        "coarse_bins": {
            "edges": list(COARSE_BIN_EDGES),
            "labels": list(COARSE_BIN_LABELS),
            "variables": list(EPISTEMIC_CONDITIONING_VARIABLES),
            "conditioned_separately": True,
        },
        # `s = 1 - phi`, so the two binnings usually induce the SAME partition
        # of the rounds, and CMI is invariant under relabelling the
        # conditioning variable - the two estimates then coincide exactly.
        # That is expected, not a bug; it is reported so nobody has to
        # rediscover it by staring at two identical numbers.
        "phi_susceptible_partition_identical": _partition_identical(rounds),
        # Surfaced in the summary rather than left in a CSV column: on a pilot
        # this size the E_k-conditioned CMI is expected to be support-limited,
        # and that has to be impossible to miss when reading the result. The
        # coarse scalar conditionings appear in the same block so the sparsity
        # they were introduced to fix is visible in the same glance.
        # The classical comparison, in the same summary the CLI already
        # prints, so "what did this run measure" and "what would the matched
        # classical controller have measured" arrive together.
        "theoretical_reference": theoretical_reference,
        "single_affinity_theory_comparison": theory_rows,
        "matched_qvoter_classical_null": classical_null_rows,
        "theory_state_coarse_graining": "target_vs_not_target",
        "current_analysis": current_rows,
        "current_episode_rows": len(current_episodes),
        "current_reports": [str(path) for path in current_reports],
        "n_micro_events_checked_for_current": len(micro),
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
    summary_path = destination / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    summary["comet"] = _export_current_analysis_to_comet(
        current_rows,
        (
            destination / "currents" / "cell_current_summary.csv",
            destination / "currents" / "episode_currents.csv",
            *current_reports,
            summary_path,
        ),
        enabled=comet_export,
        project_name=comet_project,
        run_name=comet_run_name or f"{Path(run_dir).name}/analysis",
        sink=comet_sink,
        name_suffix=comet_name_suffix,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "COARSE_BINS",
    "TE_RATIO_FLOOR",
    "append_theory_report",
    "empirical_policy_curve",
    "empirical_response_curve",
    "empirical_round_occupancy",
    "finite_horizon_occupancy",
    "matched_qvoter_comparison",
    "matched_qvoter_parameters_for",
    "matched_qvoter_state_curves",
    "COARSE_BIN_EDGES",
    "COARSE_BIN_LABELS",
    "DEFAULT_EPISTEMIC_BINS",
    "EPISTEMIC_CONDITIONING_VARIABLES",
    "coarse_bin",
    "epistemic_conditioning_values",
    "adapt_relational_round_record",
    "analyze_relational_imitation_round_feedback",
    "controller_action_summary",
    "current_analysis_comet_metrics",
    "epistemic_regime_summary",
    "read_relational_round_records",
]
