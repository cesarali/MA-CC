"""Offline direct-counting analysis at the round-feedback clock."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mas_cc.analysis.estimators import (
    Estimate,
    conditional_mutual_information,
    mutual_information,
)

from ..imitation.controller import ADVOCATE_TARGET, NO_OP

MAIN_ESTIMATOR_VARIANT = "unsmoothed"

ROUND_INFORMATION_STATISTICS = (
    "round_sensing_mi",
    "round_population_actuation_cmi",
    "round_target_actuation_cmi",
    "round_truth_actuation_cmi",
    "round_order_actuation_cmi",
)
ROUND_MEMORY_CONDITIONING_KEYS: Mapping[str, str] = {
    "round_memory_target_actuation_cmi": "conditioning_memory_state",
    "round_epistemic_target_actuation_cmi": "conditioning_epistemic_state",
    "round_phi_target_actuation_cmi": "conditioning_phi_bin",
    "round_susceptible_target_actuation_cmi": "conditioning_susceptible_bin",
    "round_kappa_target_actuation_cmi": "conditioning_kappa_bin",
}
"""Statistic name -> the round-record key holding its extra conditioning state.

Every one of these is `I(U_k ; n_Z,k+1 | n_Z,k, X_k)` for a different `X_k`,
and every one goes through the same `conditional_mutual_information` as the
plain `round_target_actuation_cmi`; only the `z` argument is wider.  A game
opts in by writing the key on its round record - a run whose records lack it
produces no eligible rows and the statistic is skipped, so all of this stays
inert for HiddenBench.

Adding a conditioning variable is one line here plus one extractor in the
game's adapter.  Two families live in this table on purpose:

`memory` / `epistemic`
    the high-dimensional reference.  `memory` is the game's exact internal
    state (for the relational game, the epistemic memory histogram `E_k`);
    `epistemic` is a joint bin pair.  Both are kept even when sparse.

`phi` / `susceptible` / `kappa`
    deliberately coarse, *scalar* conditionings that stay estimable at pilot
    sample sizes.  They are separate statistics rather than one joint state
    precisely so the conditioning stays small - conditioning on all three at
    once would reproduce the sparsity they exist to avoid.

`susceptible` is `1 - phi` and is carried under its own name because the
population it describes is a different scientific object (who is still movable
by talk) even though the arithmetic is a reflection.  Since CMI is invariant
under relabelling of `z`, the two estimates coincide wherever the two binnings
induce the same partition; that is expected, and the relational adapter reports
whether it happened rather than hiding it."""
ROUND_MEMORY_STATISTICS = tuple(ROUND_MEMORY_CONDITIONING_KEYS)
ROUND_MEMORY_SIGNED_RESPONSE_STATISTICS = tuple(
    name.replace("_actuation_cmi", "_signed_response")
    for name in ROUND_MEMORY_STATISTICS
)
"""`E[dp_Z | ADVOCATE] - E[dp_Z | NO_OP]`, matched on the *same* conditioning
state as the CMI of the same stem.  Same `_signed_response` as every other
signed diagnostic here; only the stratification changes."""
_SIGNED_RESPONSE_SOURCE: Mapping[str, str] = dict(
    zip(ROUND_MEMORY_SIGNED_RESPONSE_STATISTICS, ROUND_MEMORY_STATISTICS, strict=True)
)
_SHARE_RESPONSE_STATISTICS = frozenset(
    (
        "round_target_signed_response_share",
        "round_target_susceptibility",
        *ROUND_MEMORY_SIGNED_RESPONSE_STATISTICS,
    )
)
"""Everything read off `delta_p_ctrl`, which not every game records."""
ROUND_DIAGNOSTIC_STATISTICS = (
    "round_controller_action_entropy",
    "round_controller_action_entropy_given_population",
    "round_population_information_fraction",
    "round_target_information_fraction",
    "round_dual_action_state_fraction",
    "round_dual_action_event_fraction",
    "round_single_action_slice_fraction",
    "round_conditioning_state_count",
    "round_singleton_fraction",
    "round_target_signed_actuation",
    "round_truth_signed_actuation",
    "round_order_signed_actuation",
    "round_sensor_mae",
    "round_sensor_mse",
    "round_target_signed_response_share",
)
"""Diagnostics on a FIXED conditioning. The augmented-conditioning family
carries its own diagnostics next to its own CMIs, in
`ROUND_MEMORY_SIGNED_RESPONSE_STATISTICS`."""
ROUND_SINGLE_AFFINITY_STATISTICS = (
    "round_target_susceptibility",
    "round_target_sensing_mi",
)
"""The two statistics whose definition is fixed by the single-affinity theory.

`round_target_susceptibility` is the canonical `chi(n)` of that theory: the
state-matched difference of mean target-*fraction* changes,
`E[dx | ADVOCATE, n] - E[dx | NO_OP, n]` with `dx = (n_{k+1}-n_k)/N`.  It is
NOT `round_target_signed_actuation`, which measures the same difference in
aligned-magnetization units and is therefore larger by `K/(K-1)`.  The
theory's Pinsker bound is stated in target-fraction units, so only this one
may enter `eta_ir`.

`round_target_sensing_mi` is `I(n_Z,k ; Y_Z,k)`, the *scalar* target sensing
channel - how much the controller's finite sample tells it about the target
count alone.  `round_sensing_mi` is the full K-option vector channel
`I(N_k ; Y_k)` and is a different, larger quantity; the single-affinity
`I_sens` is the scalar one.  Reported in bits like every other direct-counting
estimate here; the nats-valued thermodynamic `I_sens` is a derived quantity
built from the exact sensor kernel, not from this count."""
ROUND_SENSOR_POLICY_STATISTICS = ("round_sensor_action_mi",)
"""The empirical policy channel ``I(Y_k; U_k)`` in bits.

This uses the established direct-counting estimator. It is distinct from the
sensor-fidelity channel ``I(N_k;Y_k)`` because it measures how much realized
sensor variation survives the stochastic policy map.
"""
# Appended, not interleaved: `round_information_analysis` seeds each statistic's
# bootstrap from `seed + name_index`, so inserting a name anywhere but the end
# would silently move every later statistic's resampling stream.
ROUND_ANALYSIS_STATISTICS = (
    *ROUND_INFORMATION_STATISTICS,
    *ROUND_DIAGNOSTIC_STATISTICS,
    *ROUND_MEMORY_STATISTICS,
    *ROUND_MEMORY_SIGNED_RESPONSE_STATISTICS,
    *ROUND_SINGLE_AFFINITY_STATISTICS,
    *ROUND_SENSOR_POLICY_STATISTICS,
)
ROUND_ACTUATION_STATISTICS = (
    *ROUND_INFORMATION_STATISTICS[1:],
    *ROUND_MEMORY_STATISTICS,
)
"""Everything that conditions on a current state and admits the policy null."""

_BITS_STATISTICS = (
    frozenset(ROUND_INFORMATION_STATISTICS)
    | frozenset(ROUND_MEMORY_STATISTICS)
    | frozenset({"round_target_sensing_mi", "round_sensor_action_mi"})
)
_RESPONSE_UNITS: Mapping[str, str] = {
    # Read off `delta_p_ctrl`: a change in the TARGET FRACTION `n_Z/N`.
    "round_target_susceptibility": "target_fraction_per_cycle",
    "round_target_signed_response_share": "target_fraction_per_cycle",
    **{
        name: "target_fraction_per_cycle"
        for name in ROUND_MEMORY_SIGNED_RESPONSE_STATISTICS
    },
    # Read off `delta_m_*`: the same motion in ALIGNED MAGNETIZATION,
    # `m = (K p - 1)/(K - 1)`, and therefore larger by `K/(K-1)`.
    "round_target_signed_actuation": "aligned_magnetization_per_cycle",
    "round_truth_signed_actuation": "aligned_magnetization_per_cycle",
    "round_order_signed_actuation": "aligned_magnetization_per_cycle",
}
"""The coordinate each signed response is measured in, spelled out.

These all used to be labelled `dimensionless`, which is true but useless: it
made a target-fraction response and a magnetization response indistinguishable
on a plot axis or in a joined table, and those two differ by `K/(K-1)`.  Naming
the coordinate is what stops the wrong one being fed to `eta_ir`."""

MICRO_SLOT_STATISTICS = (
    "micro_slot_focal_actuation_cmi",
    "micro_slot_target_signed_response",
)


@dataclass(frozen=True, slots=True)
class RoundEvent:
    cell_id: str
    episode_id: str
    round_index: int
    event: Mapping[str, Any]

    @property
    def N_k(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.event["occupation_counts_before"])

    @property
    def N_k1(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.event["occupation_counts_after"])

    @property
    def Y_k(self) -> tuple[int, ...] | None:
        value = self.event.get("sensor_count_vector")
        return None if value is None else tuple(int(item) for item in value)

    @property
    def U_k(self) -> str | None:
        value = self.event.get("controller_action")
        return None if value is None else str(value)

    @property
    def p_k(self) -> float | None:
        value = self.event.get(
            "controller_advocate_probability",
            self.event.get("controller_advocacy_probability"),
        )
        return None if value is None else float(value)

    @property
    def target_before(self) -> int:
        return int(self.event["target_count_before"])

    @property
    def target_after(self) -> int:
        return int(self.event["target_count_after"])

    @property
    def truth_before(self) -> int:
        return int(self.event["truth_count_before"])

    @property
    def truth_after(self) -> int:
        return int(self.event["truth_count_after"])

    def augmented_state(self, key: str) -> tuple[int, ...] | None:
        """One published conditioning state, or `None` when the game omits it.

        Opaque here on purpose - this module only ever hashes it as a
        conditioning label.  Scalars are normalised to a one-tuple so every
        entry of `ROUND_MEMORY_CONDITIONING_KEYS` hashes the same way whether
        the game published a histogram or a single bin index.
        """

        value = self.event.get(key)
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (int(value),)
        return tuple(int(item) for item in value)

    @property
    def memory_state(self) -> tuple[int, ...] | None:
        """The game's exact internal state at the start of the round.

        For the relational round-feedback game, the epistemic memory histogram
        `E_k = (n_k^(0), ..., n_k^(L))`.
        """

        return self.augmented_state("conditioning_memory_state")

    @property
    def epistemic_state(self) -> tuple[int, ...] | None:
        """A coarser, lower-dimensional companion to `memory_state`."""

        return self.augmented_state("conditioning_epistemic_state")

    @property
    def target_index(self) -> int | None:
        """Where the controller target sits in the task's option list.

        Needed to read the *scalar* target coordinate out of the vector-valued
        sensor record.  Games that do not publish `possible_answers` return
        `None` and the scalar sensing statistic is skipped for them.
        """

        options = self.event.get("possible_answers")
        if options is None:
            return None
        labels = [str(item) for item in options]
        for key in ("analysis_target", "controller_target", "correct_answer"):
            label = self.event.get(key)
            if label is not None and str(label) in labels:
                return labels.index(str(label))
        return None

    @property
    def sensor_target_count(self) -> int | None:
        """`Y_Z,k` - sampled agents voting for the target, or `None`.

        The theory's sensing channel is `n_Z -> Y_Z`, a scalar-to-scalar
        channel.  `Y_k` is the whole sampled count vector; this is its target
        component.  Adapters that already publish `sensor_target_count` win,
        so a game can record it directly instead of being reconstructed here.
        """

        value = self.event.get("sensor_target_count")
        if value is not None:
            return int(value)
        vector, index = self.Y_k, self.target_index
        if vector is None or index is None or index >= len(vector):
            return None
        return int(vector[index])

    @property
    def order_before(self) -> int:
        return max(self.N_k)

    @property
    def order_after(self) -> int:
        return max(self.N_k1)


def adapt_round_record(
    record: Mapping[str, Any], *, cell_id: str = "run", episode_id: str | None = None
) -> RoundEvent:
    if record.get("record_type") not in {None, "imitation_round_feedback"}:
        raise ValueError("record is not an imitation_round_feedback row")
    required = {
        "round_index",
        "occupation_counts_before",
        "occupation_counts_after",
        "target_count_before",
        "target_count_after",
        "truth_count_before",
        "truth_count_after",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError("round record is missing: " + ", ".join(missing))
    before = tuple(int(value) for value in record["occupation_counts_before"])
    after = tuple(int(value) for value in record["occupation_counts_after"])
    if len(before) != len(after) or len(before) < 2:
        raise ValueError("round occupation vectors must have the same K >= 2")
    if sum(before) != sum(after):
        raise ValueError("round occupation vectors must conserve population size")
    return RoundEvent(
        cell_id=str(cell_id),
        episode_id=str(episode_id or record.get("episode_id", "episode")),
        round_index=int(record["round_index"]),
        event=dict(record),
    )


def _cell_id_for(path: Path) -> str:
    for parent in path.parents:
        if (parent / "overrides.json").is_file():
            return parent.name
    return "run"


def read_round_records(root: str | Path) -> list[RoundEvent]:
    source = Path(root)
    paths = (
        [source] if source.is_file() else sorted(source.rglob("round_trajectory.jsonl"))
    )
    if not paths:
        raise FileNotFoundError(f"no round_trajectory.jsonl files under {source}")
    rows: list[RoundEvent] = []
    for path in paths:
        cell_id = _cell_id_for(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("record_type") != "imitation_round_feedback":
                continue
            rows.append(adapt_round_record(payload, cell_id=cell_id))
    if not rows:
        raise ValueError(
            f"round trajectory files under {source} contain no round records"
        )
    return rows


def _augmented_conditioning(key: str) -> Callable[[RoundEvent], Hashable]:
    """`(n_Z,k, X_k)` for the published state under `key`."""

    return lambda row: (row.target_before, row.augmented_state(key))


ROUND_CONDITIONING_STATE: Mapping[str, Callable[[RoundEvent], Hashable]] = {
    "round_population_actuation_cmi": lambda row: row.N_k,
    "round_target_actuation_cmi": lambda row: row.target_before,
    "round_truth_actuation_cmi": lambda row: row.truth_before,
    "round_order_actuation_cmi": lambda row: row.order_before,
    **{
        name: _augmented_conditioning(key)
        for name, key in ROUND_MEMORY_CONDITIONING_KEYS.items()
    },
}
"""`Z` in `I(U_k ; . | Z)`, per statistic - the single place the conditioning
state of an actuation estimate is defined.  Also what the entropy ceiling
`H(U_k | Z)`, the support/overlap diagnostics and the matched signed responses
are computed against."""

_ROUND_OUTCOME: Mapping[str, Callable[[RoundEvent], Hashable]] = {
    "round_population_actuation_cmi": lambda row: row.N_k1,
    "round_target_actuation_cmi": lambda row: row.target_after,
    "round_truth_actuation_cmi": lambda row: row.truth_after,
    "round_order_actuation_cmi": lambda row: row.order_after,
    # Every augmented conditioning measures the same opinion channel; only the
    # conditioning differs, which is what makes the family comparable.
    **{name: (lambda row: row.target_after) for name in ROUND_MEMORY_STATISTICS},
}


def _estimate_for(name: str, rows: Sequence[RoundEvent]) -> Estimate:
    if name == "round_sensing_mi":
        return mutual_information([row.N_k for row in rows], [row.Y_k for row in rows])
    if name == "round_target_sensing_mi":
        # The single-affinity sensing channel: scalar count in, scalar count
        # out.  Deliberately NOT the full occupation/sensor vectors.
        return mutual_information(
            [row.target_before for row in rows],
            [row.sensor_target_count for row in rows],
        )
    if name == "round_sensor_action_mi":
        return mutual_information(
            [row.Y_k for row in rows],
            [str(row.U_k) for row in rows],
        )
    outcome = _ROUND_OUTCOME.get(name)
    if outcome is None:
        raise ValueError(f"unknown round information statistic {name!r}")
    state = ROUND_CONDITIONING_STATE[name]
    return conditional_mutual_information(
        [str(row.U_k) for row in rows],
        [outcome(row) for row in rows],
        [state(row) for row in rows],
    )


def _entropy_bits(values: Sequence[Hashable]) -> float:
    if not values:
        return math.nan
    counts = Counter(values)
    total = len(values)
    return -sum(
        (count / total) * math.log2(count / total) for count in counts.values() if count
    )


def conditional_action_entropy_bits(
    actions: Sequence[str], states: Sequence[Hashable]
) -> float:
    if not actions:
        return math.nan
    grouped: dict[Hashable, list[str]] = defaultdict(list)
    for action, state in zip(actions, states, strict=True):
        grouped[state].append(action)
    return sum(
        len(group) / len(actions) * _entropy_bits(group) for group in grouped.values()
    )


def round_overlap_diagnostics(
    rows: Sequence[RoundEvent],
    *,
    state: Callable[[RoundEvent], Hashable] = lambda row: row.N_k,
) -> dict[str, Any]:
    """How much action overlap the conditioning slices actually carry.

    `state` defaults to the population occupation vector, which is what the
    historical columns mean.  A caller estimating a CMI under a *wider*
    conditioning state passes that state instead, so the sparsity reported next
    to an estimate is the sparsity that estimate actually faced.
    """

    controlled = [row for row in rows if row.U_k in {ADVOCATE_TARGET, NO_OP}]
    grouped: dict[Hashable, Counter[str]] = defaultdict(Counter)
    for row in controlled:
        grouped[state(row)][str(row.U_k)] += 1
    dual = [
        counter
        for counter in grouped.values()
        if counter[ADVOCATE_TARGET] and counter[NO_OP]
    ]
    events_in_dual = sum(sum(counter.values()) for counter in dual)
    singleton_events = sum(
        sum(counter.values())
        for counter in grouped.values()
        if sum(counter.values()) == 1
    )
    return {
        "round_dual_action_state_fraction": (
            math.nan if not grouped else len(dual) / len(grouped)
        ),
        "round_dual_action_event_fraction": (
            math.nan if not controlled else events_in_dual / len(controlled)
        ),
        "round_single_action_slice_fraction": (
            math.nan if not grouped else 1.0 - len(dual) / len(grouped)
        ),
        "round_conditioning_state_count": len(grouped),
        "round_singleton_fraction": (
            math.nan if not controlled else singleton_events / len(controlled)
        ),
    }


def _signed_response(
    rows: Sequence[RoundEvent],
    *,
    state: Callable[[RoundEvent], Hashable],
    delta: Callable[[RoundEvent], float],
) -> float:
    grouped: dict[Hashable, dict[str, list[float]]] = defaultdict(
        lambda: {ADVOCATE_TARGET: [], NO_OP: []}
    )
    for row in rows:
        if row.U_k in {ADVOCATE_TARGET, NO_OP}:
            grouped[state(row)][str(row.U_k)].append(delta(row))
    weighted = 0.0
    weight = 0
    for buckets in grouped.values():
        advocated = buckets[ADVOCATE_TARGET]
        no_op = buckets[NO_OP]
        if not advocated or not no_op:
            continue
        size = len(advocated) + len(no_op)
        weighted += size * (sum(advocated) / len(advocated) - sum(no_op) / len(no_op))
        weight += size
    return math.nan if weight == 0 else weighted / weight


def _diagnostic_for(name: str, rows: Sequence[RoundEvent]) -> float:
    controlled = [row for row in rows if row.U_k in {ADVOCATE_TARGET, NO_OP}]
    actions = [str(row.U_k) for row in controlled]
    if name == "round_controller_action_entropy":
        return _entropy_bits(actions)
    if name == "round_controller_action_entropy_given_population":
        return conditional_action_entropy_bits(actions, [row.N_k for row in controlled])
    if name == "round_population_information_fraction":
        denominator = conditional_action_entropy_bits(
            actions, [row.N_k for row in controlled]
        )
        numerator = getattr(
            _estimate_for("round_population_actuation_cmi", controlled),
            MAIN_ESTIMATOR_VARIANT,
        )
        return (
            math.nan
            if not math.isfinite(denominator) or denominator <= 1e-12
            else numerator / denominator
        )
    if name == "round_target_information_fraction":
        denominator = conditional_action_entropy_bits(
            actions, [row.target_before for row in controlled]
        )
        numerator = getattr(
            _estimate_for("round_target_actuation_cmi", controlled),
            MAIN_ESTIMATOR_VARIANT,
        )
        return (
            math.nan
            if not math.isfinite(denominator) or denominator <= 1e-12
            else numerator / denominator
        )
    if name in {
        "round_dual_action_state_fraction",
        "round_dual_action_event_fraction",
        "round_single_action_slice_fraction",
        "round_conditioning_state_count",
        "round_singleton_fraction",
    }:
        return float(round_overlap_diagnostics(controlled)[name])
    if name == "round_target_signed_actuation":
        return _signed_response(
            controlled,
            state=lambda row: row.target_before,
            delta=lambda row: float(row.event["delta_m_ctrl"]),
        )
    if name == "round_truth_signed_actuation":
        return _signed_response(
            controlled,
            state=lambda row: row.truth_before,
            delta=lambda row: float(row.event["delta_m_truth"]),
        )
    if name == "round_order_signed_actuation":
        return _signed_response(
            controlled,
            state=lambda row: row.order_before,
            delta=lambda row: float(row.event["delta_m_order"]),
        )
    if name == "round_target_susceptibility":
        # THE canonical chi of the revised single-affinity theory: matched on
        # `n_Z,k` exactly like `round_target_actuation_cmi`, and measured in
        # target-FRACTION units so it can be squared inside the Pinsker
        # numerator without a hidden K/(K-1) rescaling.
        return _signed_response(
            controlled,
            state=lambda row: row.target_before,
            delta=lambda row: float(row.event["delta_p_ctrl"]),
        )
    if name == "round_target_signed_response_share":
        # `E[dp_Z | ADVOCATE] - E[dp_Z | NO_OP]` in raw share units, unmatched:
        # a constant conditioning state collapses `_signed_response` to exactly
        # that marginal difference. The state-matched form of the same quantity
        # is `round_target_signed_actuation`, in aligned-magnetization units
        # (`dm = dp * K/(K-1)`), so the two are read together rather than
        # instead of each other.
        return _signed_response(
            controlled,
            state=lambda row: 0,
            delta=lambda row: float(row.event["delta_p_ctrl"]),
        )
    if name in _SIGNED_RESPONSE_SOURCE:
        # The same difference, stratified on the SAME conditioning state as the
        # CMI of the same stem - so "the controller moved the target" and "the
        # controller moved the target within this epistemic regime" are read
        # off one shared definition instead of two that can drift apart.
        return _signed_response(
            controlled,
            state=ROUND_CONDITIONING_STATE[_SIGNED_RESPONSE_SOURCE[name]],
            delta=lambda row: float(row.event["delta_p_ctrl"]),
        )
    errors = [
        float(row.event["sensor_target_share"]) - row.target_before / sum(row.N_k)
        for row in controlled
        if row.event.get("sensor_target_share") is not None
    ]
    if name == "round_sensor_mae":
        return (
            math.nan
            if not errors
            else sum(abs(value) for value in errors) / len(errors)
        )
    if name == "round_sensor_mse":
        return (
            math.nan
            if not errors
            else sum(value * value for value in errors) / len(errors)
        )
    raise ValueError(f"unknown round diagnostic {name!r}")


def _grouped(
    rows: Sequence[RoundEvent], key: Callable[[RoundEvent], Hashable]
) -> dict[Hashable, list[RoundEvent]]:
    result: dict[Hashable, list[RoundEvent]] = defaultdict(list)
    for row in rows:
        result[key(row)].append(row)
    return dict(result)


def bootstrap_episode_rows(
    rows: Sequence[RoundEvent], *, resamples: int, seed: int
) -> tuple[tuple[RoundEvent, ...], ...]:
    """Resample whole episode IDs, never individual round rows."""

    if resamples < 0:
        raise ValueError("resamples cannot be negative")
    by_episode = _grouped(rows, key=lambda row: row.episode_id)
    ids = tuple(by_episode)
    if not ids or resamples == 0:
        return ()
    rng = np.random.default_rng(seed)
    draws: list[tuple[RoundEvent, ...]] = []
    for _ in range(resamples):
        selected = rng.choice(ids, size=len(ids), replace=True)
        draws.append(
            tuple(row for episode_id in selected for row in by_episode[str(episode_id)])
        )
    return tuple(draws)


def _policy_resample(
    rows: Sequence[RoundEvent], rng: np.random.Generator
) -> list[RoundEvent]:
    result = []
    for row in rows:
        if row.p_k is None:
            result.append(row)
            continue
        event = dict(row.event)
        event["controller_action"] = (
            ADVOCATE_TARGET if rng.random() < row.p_k else NO_OP
        )
        result.append(replace(row, event=event))
    return result


def policy_resampling_null(
    name: str,
    rows: Sequence[RoundEvent],
    *,
    permutations: int,
    seed: int,
) -> tuple[float, ...]:
    if name not in ROUND_ACTUATION_STATISTICS:
        raise ValueError("policy resampling is defined for round actuation statistics")
    values = []
    for index in range(permutations):
        perturbed = _policy_resample(rows, np.random.default_rng(seed + index))
        values.append(
            float(getattr(_estimate_for(name, perturbed), MAIN_ESTIMATOR_VARIANT))
        )
    return tuple(values)


def _support(
    rows: Sequence[RoundEvent],
    *,
    state: Callable[[RoundEvent], Hashable] = lambda row: row.N_k,
) -> dict[str, Any]:
    overlap = round_overlap_diagnostics(rows, state=state)
    counts = Counter(state(row) for row in rows)
    return {
        "n_episodes": len({row.episode_id for row in rows}),
        "n_rounds": len(rows),
        "unique_population_states": len(counts),
        "unique_sensor_states": len({row.Y_k for row in rows if row.Y_k is not None}),
        "number_of_actions_observed": len(
            {row.U_k for row in rows if row.U_k is not None}
        ),
        "min_rounds_per_population_state": min(counts.values()) if counts else 0,
        "median_rounds_per_population_state": (
            math.nan if not counts else float(np.median(list(counts.values())))
        ),
        "max_rounds_per_population_state": max(counts.values()) if counts else 0,
        **overlap,
    }


def round_information_analysis(
    rows: Sequence[RoundEvent],
    *,
    statistics: Sequence[str] | None = None,
    bootstrap_resamples: int = 1000,
    null_permutations: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    names = tuple(ROUND_ANALYSIS_STATISTICS if statistics is None else statistics)
    unknown = sorted(set(names) - set(ROUND_ANALYSIS_STATISTICS))
    if unknown:
        raise ValueError("unknown round-feedback statistic(s): " + ", ".join(unknown))
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    alpha = (1.0 - confidence) / 2.0
    estimates: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []

    for name_index, name in enumerate(names):
        if name == "round_sensing_mi":
            eligible = [row for row in rows if row.Y_k is not None]
        elif name == "round_target_sensing_mi":
            eligible = [row for row in rows if row.sensor_target_count is not None]
        elif name == "round_sensor_action_mi":
            eligible = [
                row
                for row in rows
                if row.Y_k is not None and row.U_k in {ADVOCATE_TARGET, NO_OP}
            ]
        else:
            eligible = [row for row in rows if row.U_k in {ADVOCATE_TARGET, NO_OP}]
        # A statistic that needs a state or a delta the game does not record
        # drops out here rather than raising, which is what keeps the augmented
        # conditioning and the share-unit responses inert on runs without them.
        source = _SIGNED_RESPONSE_SOURCE.get(name, name)
        key = ROUND_MEMORY_CONDITIONING_KEYS.get(source)
        if key is not None:
            eligible = [row for row in eligible if row.augmented_state(key) is not None]
        if name in _SHARE_RESPONSE_STATISTICS:
            eligible = [
                row for row in eligible if row.event.get("delta_p_ctrl") is not None
            ]
        # No-control cells have no fabricated U=NO_OP rows and therefore emit
        # no controller information/diagnostic estimate at all.
        if not eligible:
            continue
        if name in _BITS_STATISTICS:
            estimate = _estimate_for(name, eligible)
            value = float(getattr(estimate, MAIN_ESTIMATOR_VARIANT))
            variants = {
                "jeffreys": estimate.jeffreys,
                "unsmoothed": estimate.unsmoothed,
                "miller_madow": estimate.miller_madow,
            }
        else:
            value = _diagnostic_for(name, eligible)
            variants = {
                "jeffreys": math.nan,
                "unsmoothed": value,
                "miller_madow": math.nan,
            }
        bootstrap_values = []
        for draw in bootstrap_episode_rows(
            eligible, resamples=bootstrap_resamples, seed=seed + name_index
        ):
            if name in _BITS_STATISTICS:
                boot = float(getattr(_estimate_for(name, draw), MAIN_ESTIMATOR_VARIANT))
            else:
                boot = _diagnostic_for(name, draw)
            if math.isfinite(boot):
                bootstrap_values.append(boot)
        interval = (
            (math.nan, math.nan)
            if not bootstrap_values
            else (
                float(np.quantile(bootstrap_values, alpha)),
                float(np.quantile(bootstrap_values, 1.0 - alpha)),
            )
        )
        null_values: tuple[float, ...] = ()
        null_type = None
        if name in ROUND_ACTUATION_STATISTICS:
            null_values = policy_resampling_null(
                name,
                eligible,
                permutations=null_permutations,
                seed=seed + 100_000 * (name_index + 1),
            )
            null_type = "policy_conditional_randomization"
        elif name in {
            "round_sensing_mi",
            "round_target_sensing_mi",
            "round_sensor_action_mi",
        }:
            permuted_values = []
            key = (
                "sensor_count_vector"
                if name in {"round_sensing_mi", "round_sensor_action_mi"}
                else "sensor_target_count"
            )
            for permutation in range(null_permutations):
                rng = np.random.default_rng(
                    seed + 100_000 * (name_index + 1) + permutation
                )
                sensors = [
                    row.Y_k
                    if name in {"round_sensing_mi", "round_sensor_action_mi"}
                    else row.sensor_target_count
                    for row in eligible
                ]
                order = rng.permutation(len(sensors))
                shuffled = [
                    replace(row, event={**dict(row.event), key: sensors[int(index)]})
                    for row, index in zip(eligible, order, strict=True)
                ]
                permuted_values.append(
                    float(
                        getattr(_estimate_for(name, shuffled), MAIN_ESTIMATOR_VARIANT)
                    )
                )
            null_values = tuple(permuted_values)
            null_type = "sensor_permutation"
        finite_null = [item for item in null_values if math.isfinite(item)]
        for permutation, null_value in enumerate(null_values):
            null_rows.append(
                {
                    "statistic": name,
                    "permutation": permutation,
                    "null_type": null_type,
                    "estimate": null_value,
                }
            )
        conditioning = ROUND_CONDITIONING_STATE.get(name)
        # Sparsity is reported against the statistic's OWN conditioning state,
        # so the memory-aware estimate carries its own slice counts rather than
        # the population-vector ones - and a matched signed response inherits
        # the sparsity of the CMI it mirrors, via `source`. Statistics that do
        # not condition keep the historical population-vector support.
        support = (
            _support(eligible, state=ROUND_CONDITIONING_STATE[source])
            if source in ROUND_CONDITIONING_STATE
            else _support(eligible)
        )
        entropy_ceiling = math.nan
        entropy_bound_satisfied: bool | None = None
        if conditioning is not None:
            entropy_ceiling = conditional_action_entropy_bits(
                [str(row.U_k) for row in eligible],
                [conditioning(row) for row in eligible],
            )
        if math.isfinite(entropy_ceiling):
            entropy_bound_satisfied = bool(value <= entropy_ceiling + 1e-9)
        estimates.append(
            {
                "statistic": name,
                "estimate": value,
                "main_estimator_variant": MAIN_ESTIMATOR_VARIANT,
                "estimator_variant": "direct_counting",
                **variants,
                "bootstrap_ci_low": interval[0],
                "bootstrap_ci_high": interval[1],
                "null_type": null_type,
                "null_mean": (
                    math.nan if not finite_null else float(np.mean(finite_null))
                ),
                "units": (
                    "bits"
                    if (name in _BITS_STATISTICS or "entropy" in name)
                    else _RESPONSE_UNITS.get(name, "dimensionless")
                ),
                "bootstrap_unit": "episode",
                "conditional_action_entropy_bits": entropy_ceiling,
                "entropy_bound_satisfied": entropy_bound_satisfied,
                **support,
            }
        )
    return estimates, null_rows


def _read_micro_events(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = [
        *root.rglob("trajectory.jsonl"),
        *root.rglob("micro_slot_trajectory.jsonl"),
    ]
    for path in sorted(paths):
        cell_id = _cell_id_for(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            event = payload.get("event", payload)
            if isinstance(event, Mapping) and "within_round_index" in event:
                rows.append({"cell_id": cell_id, **dict(event)})
    return rows


def _micro_signed(rows: Sequence[Mapping[str, Any]]) -> float:
    grouped: dict[Hashable, dict[bool, list[float]]] = defaultdict(
        lambda: {True: [], False: []}
    )
    for row in rows:
        state = tuple(
            int(row["occupation_counts_before"][option])
            for option in row["possible_answers"]
        )
        grouped[state][bool(row["controlled_slot"])].append(float(row["delta_m_ctrl"]))
    weighted = 0.0
    weight = 0
    for buckets in grouped.values():
        if not buckets[True] or not buckets[False]:
            continue
        size = len(buckets[True]) + len(buckets[False])
        weighted += size * (
            sum(buckets[True]) / len(buckets[True])
            - sum(buckets[False]) / len(buckets[False])
        )
        weight += size
    return math.nan if not weight else weighted / weight


def _micro_cmi(rows: Sequence[Mapping[str, Any]]) -> float:
    controls = [bool(row["controlled_slot"]) for row in rows]
    before = [
        (
            str(row["focal_opinion_before"]),
            tuple(
                int(row["occupation_counts_before"][option])
                for option in row["possible_answers"]
            ),
        )
        for row in rows
    ]
    after = [str(row["focal_opinion_after"]) for row in rows]
    return conditional_mutual_information(controls, after, before).unsmoothed


def _resample_micro_slots(
    rows: Sequence[Mapping[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["episode_id"]), int(row["round_index"]))].append(row)
    result: list[dict[str, Any]] = []
    for group in groups.values():
        budget = int(group[0]["intervention_budget"])
        selected = {
            int(index) for index in rng.choice(len(group), size=budget, replace=False)
        }
        result.extend(
            {**dict(row), "controlled_slot": index in selected}
            for index, row in enumerate(group)
        )
    return result


def micro_slot_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    null_permutations: int = 0,
    seed: int = 1,
) -> list[dict[str, Any]]:
    result = []
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("round_controller_action") == ADVOCATE_TARGET:
            grouped[str(row.get("cell_id", "run"))].append(row)
    for cell_id, group in grouped.items():
        null_cmi: list[float] = []
        null_signed: list[float] = []
        for permutation in range(null_permutations):
            perturbed = _resample_micro_slots(
                group, np.random.default_rng(seed + 10_000 * permutation)
            )
            null_cmi.append(_micro_cmi(perturbed))
            null_signed.append(_micro_signed(perturbed))
        result.extend(
            [
                {
                    "cell_id": cell_id,
                    "statistic": "micro_slot_focal_actuation_cmi",
                    "estimate": _micro_cmi(group),
                    "units": "bits",
                    "n_events": len(group),
                    "null_type": "exact_budget_within_round_slot_resampling",
                    "null_mean": (
                        math.nan if not null_cmi else float(np.nanmean(null_cmi))
                    ),
                },
                {
                    "cell_id": cell_id,
                    "statistic": "micro_slot_target_signed_response",
                    "estimate": _micro_signed(group),
                    "units": "order_parameter",
                    "n_events": len(group),
                    "null_type": "exact_budget_within_round_slot_resampling",
                    "null_mean": (
                        math.nan if not null_signed else float(np.nanmean(null_signed))
                    ),
                },
            ]
        )
    return result


def _episode_current_rows(
    rounds: Sequence[RoundEvent], micro: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    micro_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in micro:
        micro_groups[(str(row.get("cell_id", "run")), str(row["episode_id"]))].append(
            row
        )
    result = []
    for (cell_id, episode_id), group in _grouped(
        rounds, key=lambda row: (row.cell_id, row.episode_id)
    ).items():
        ordered = sorted(group, key=lambda row: row.round_index)
        micro_rows = micro_groups.get((str(cell_id), str(episode_id)), [])
        if micro_rows:
            increments = [
                int(row.get("truth_current_increment", 0)) for row in micro_rows
            ]
            truth_current = sum(increments)
            truth_activity: int | None = sum(abs(value) for value in increments)
        else:
            truth_current = ordered[-1].truth_after - ordered[0].truth_before
            truth_activity = None
        result.append(
            {
                "cell_id": cell_id,
                "episode_id": episode_id,
                "rounds": len(ordered),
                "truth_current": truth_current,
                "truth_activity": truth_activity,
                "initial_truth_count": ordered[0].truth_before,
                "final_truth_count": ordered[-1].truth_after,
            }
        )
    return result


def _cell_summaries(
    rounds: Sequence[RoundEvent],
    episodes: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    episode_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in episodes:
        episode_groups[str(row["cell_id"])].append(row)
    round_groups = _grouped(rounds, key=lambda row: row.cell_id)
    result = []
    for cell_id, group in round_groups.items():
        currents = np.asarray(
            [float(row["truth_current"]) for row in episode_groups[str(cell_id)]],
            dtype=float,
        )
        activities = [
            float(row["truth_activity"])
            for row in episode_groups[str(cell_id)]
            if row.get("truth_activity") is not None
        ]
        mean = float(currents.mean()) if len(currents) else math.nan
        variance = float(currents.var(ddof=1)) if len(currents) >= 2 else math.nan
        fano = (
            abs(mean) / variance
            if math.isfinite(variance) and variance > 0
            else math.nan
        )
        rng = np.random.default_rng(seed)
        bootstrap_means: list[float] = []
        bootstrap_fanos: list[float] = []
        if len(currents):
            for _ in range(bootstrap_resamples):
                drawn = rng.choice(currents, size=len(currents), replace=True)
                bootstrap_means.append(float(drawn.mean()))
                drawn_variance = (
                    float(drawn.var(ddof=1)) if len(drawn) >= 2 else math.nan
                )
                if math.isfinite(drawn_variance) and drawn_variance > 0:
                    bootstrap_fanos.append(abs(float(drawn.mean())) / drawn_variance)
        alpha = (1.0 - confidence) / 2.0
        result.append(
            {
                "cell_id": cell_id,
                "dynamics_mode": group[0].event.get("dynamics_mode"),
                "controller_enabled": bool(group[0].event.get("controller_enabled")),
                "episodes": len(currents),
                "rounds": len(group),
                "truth_current_mean": mean,
                "truth_current_variance": variance,
                "truth_current_fano": fano,
                "truth_current_mean_per_agent": mean / sum(group[0].N_k),
                "truth_current_mean_ci_low": (
                    math.nan
                    if not bootstrap_means
                    else float(np.quantile(bootstrap_means, alpha))
                ),
                "truth_current_mean_ci_high": (
                    math.nan
                    if not bootstrap_means
                    else float(np.quantile(bootstrap_means, 1.0 - alpha))
                ),
                "truth_current_fano_ci_low": (
                    math.nan
                    if not bootstrap_fanos
                    else float(np.quantile(bootstrap_fanos, alpha))
                ),
                "truth_current_fano_ci_high": (
                    math.nan
                    if not bootstrap_fanos
                    else float(np.quantile(bootstrap_fanos, 1.0 - alpha))
                ),
                "truth_activity_mean": (
                    math.nan if not activities else float(np.mean(activities))
                ),
                "truth_activity_variance": (
                    math.nan
                    if len(activities) < 2
                    else float(np.var(activities, ddof=1))
                ),
                "final_m_truth_mean": float(
                    np.mean(
                        [
                            row.event["m_truth_after"]
                            for row in group
                            if row.round_index
                            == max(
                                item.round_index
                                for item in group
                                if item.episode_id == row.episode_id
                            )
                        ]
                    )
                ),
            }
        )
    return result


def _write_markdown(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    lines = [
        "# Round feedback information estimates",
        "",
        "Direct-counting estimates use bits. Bootstrap resampling uses whole episodes; actuation nulls resample each action from its logged policy probability.",
        "",
        "| Cell | Statistic | Estimate | 95% bootstrap CI | Null mean |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['cell_id']}` | `{row['statistic']}` | {row['estimate']:.6g} | "
            f"[{row['bootstrap_ci_low']:.6g}, {row['bootstrap_ci_high']:.6g}] | "
            f"{row['null_mean']:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


ROUND_COMET_FIELDS = (
    "estimate",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "null_mean",
    "excess_over_null",
    "n_episodes",
    "n_rounds",
    "conditional_action_entropy_bits",
    "unique_population_states",
    "unique_sensor_states",
    "number_of_actions_observed",
    "round_dual_action_state_fraction",
    "round_dual_action_event_fraction",
    "round_single_action_slice_fraction",
    "round_conditioning_state_count",
    "round_singleton_fraction",
)
"""Per-cell round-feedback values that remain useful as Comet series."""


def _comet_metric_name(cell_id: Any, statistic: str, field: str) -> str:
    label = str(cell_id or "run").replace("/", "_").replace(" ", "_")
    return "/".join((label, statistic, field))


def round_analysis_comet_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Flatten round estimates and diagnostics into finite Comet scalars."""

    metrics: dict[str, float] = {}
    for row in rows:
        statistic = str(row.get("statistic") or "")
        if not statistic:
            continue
        derived = dict(row)
        estimate, null_mean = row.get("estimate"), row.get("null_mean")
        if (
            isinstance(estimate, (int, float))
            and not isinstance(estimate, bool)
            and isinstance(null_mean, (int, float))
            and not isinstance(null_mean, bool)
            and math.isfinite(float(estimate))
            and math.isfinite(float(null_mean))
        ):
            derived["excess_over_null"] = float(estimate) - float(null_mean)
        for field in ROUND_COMET_FIELDS:
            value = derived.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                continue
            metrics[_comet_metric_name(row.get("cell_id"), statistic, field)] = float(
                value
            )
        bound = row.get("entropy_bound_satisfied")
        if isinstance(bound, bool):
            metrics[
                _comet_metric_name(
                    row.get("cell_id"), statistic, "entropy_bound_satisfied"
                )
            ] = float(bound)
    return metrics


def _export_round_analysis_to_comet(
    rows: Sequence[Mapping[str, Any]],
    assets: Sequence[Path],
    *,
    enabled: bool,
    project_name: str,
    run_name: str,
    sink: Any | None = None,
    name_suffix: str | None = None,
) -> dict[str, Any]:
    """Publish one completed cell's metrics and reports to Comet."""

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
    metrics = round_analysis_comet_metrics(rows)
    uploaded_assets = 0
    try:
        sink.add_tags(("analysis", "information", "round-feedback"))
        if metrics:
            sink.log_metrics(metrics, 0)
        for asset in assets:
            path = Path(asset)
            if not path.is_file():
                continue
            asset_name = path.name
            if name_suffix:
                asset_name = f"{path.stem}__{name_suffix}{path.suffix}"
            sink.log_asset(path, name=asset_name)
            uploaded_assets += 1
        return {
            "status": sink.status,
            "metrics": len(metrics),
            "assets": uploaded_assets,
            "url": sink.url,
            "published_to": "master" if borrowed else "analysis_experiment",
        }
    finally:
        if not borrowed:
            sink.close()


def analyze_hidden_bench_imitation_round_feedback(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_resamples: int = 1000,
    null_permutations: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
    statistics: Sequence[str] | None = None,
    comet_export: bool = False,
    comet_project: str = "mas-cc",
    comet_run_name: str | None = None,
    comet_sink: Any | None = None,
    comet_name_suffix: str | None = None,
    artifact_profile: str = "full",
    resolved_config_hash: str | None = None,
) -> dict[str, Any]:
    rounds = read_round_records(run_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    estimates: list[dict[str, Any]] = []
    nulls: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    for cell_id, group in _grouped(rounds, key=lambda row: row.cell_id).items():
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

    behavioral = []
    for cell_id, group in _grouped(rounds, key=lambda row: row.cell_id).items():
        behavioral.append(
            {
                "cell_id": cell_id,
                "episodes": len({row.episode_id for row in group}),
                "rounds": len(group),
                "mean_delta_m_ctrl": float(
                    np.mean([row.event["delta_m_ctrl"] for row in group])
                ),
                "mean_delta_m_truth": float(
                    np.mean([row.event["delta_m_truth"] for row in group])
                ),
                "mean_delta_m_order": float(
                    np.mean([row.event["delta_m_order"] for row in group])
                ),
                "mean_delta_H_vote": float(
                    np.mean([row.event["delta_H_vote"] for row in group])
                ),
            }
        )

    micro = _read_micro_events(Path(run_dir)) if Path(run_dir).is_dir() else []
    micro_rows = micro_slot_analysis(
        micro, null_permutations=null_permutations, seed=seed
    )
    episode_rows = _episode_current_rows(rounds, micro)
    cell_rows = _cell_summaries(
        rounds,
        episode_rows,
        bootstrap_resamples=bootstrap_resamples,
        confidence=confidence,
        seed=seed,
    )

    pd.DataFrame(estimates).to_csv(
        destination / "round_information_estimates.csv", index=False
    )
    _write_markdown(estimates, destination / "round_information_estimates.md")
    pd.DataFrame(nulls).to_csv(destination / "round_information_nulls.csv", index=False)
    pd.DataFrame(support).to_csv(
        destination / "round_support_diagnostics.csv", index=False
    )
    pd.DataFrame(behavioral).to_csv(
        destination / "round_behavioral_summary.csv", index=False
    )
    pd.DataFrame(micro_rows).to_csv(
        destination / "micro_slot_diagnostics.csv", index=False
    )
    pd.DataFrame(episode_rows).to_csv(destination / "episode_currents.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(destination / "cell_summaries.csv", index=False)
    summary = {
        "n_cells": len({row.cell_id for row in rounds}),
        "n_episodes": len({(row.cell_id, row.episode_id) for row in rounds}),
        "n_rounds": len(rounds),
        "n_micro_events": len(micro),
        "statistics": list(
            ROUND_ANALYSIS_STATISTICS if statistics is None else statistics
        ),
        "bootstrap_unit": "episode",
        "bootstrap_resamples": bootstrap_resamples,
        "null_permutations": null_permutations,
        "main_estimator_variant": MAIN_ESTIMATOR_VARIANT,
        "artifact_profile": artifact_profile,
        "resolved_config_hash": resolved_config_hash,
    }
    summary_path = destination / "analysis_summary.json"
    # Write the scientific summary before export so it is one of the durable
    # per-cell Comet assets. It is rewritten below with the export status.
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["comet"] = _export_round_analysis_to_comet(
        estimates,
        (
            destination / "round_information_estimates.csv",
            destination / "round_information_estimates.md",
            destination / "round_support_diagnostics.csv",
            destination / "round_behavioral_summary.csv",
            summary_path,
        ),
        enabled=comet_export,
        project_name=comet_project,
        run_name=comet_run_name or f"{Path(run_dir).name}/analysis",
        sink=comet_sink,
        name_suffix=comet_name_suffix,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "MICRO_SLOT_STATISTICS",
    "ROUND_ACTUATION_STATISTICS",
    "ROUND_ANALYSIS_STATISTICS",
    "ROUND_CONDITIONING_STATE",
    "ROUND_MEMORY_CONDITIONING_KEYS",
    "ROUND_MEMORY_SIGNED_RESPONSE_STATISTICS",
    "ROUND_MEMORY_STATISTICS",
    "ROUND_DIAGNOSTIC_STATISTICS",
    "ROUND_INFORMATION_STATISTICS",
    "ROUND_SINGLE_AFFINITY_STATISTICS",
    "RoundEvent",
    "adapt_round_record",
    "analyze_hidden_bench_imitation_round_feedback",
    "bootstrap_episode_rows",
    "conditional_action_entropy_bits",
    "micro_slot_analysis",
    "policy_resampling_null",
    "read_round_records",
    "round_analysis_comet_metrics",
    "round_information_analysis",
    "round_overlap_diagnostics",
]
