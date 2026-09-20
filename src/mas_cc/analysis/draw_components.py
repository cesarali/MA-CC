"""Bootstrap draws as row-position arrays, and the estimators evaluated on them.

Shared by the derived study aggregates and the single-affinity analysis. A
draw is a sequence of rows (episodes, or initialization blocks repeated by
their drawn weight, in draw order); every quantity the row path computes on a
materialised row list is recomputed here from per-row codes with the row
path's arithmetic, so the values are bit-identical while the per-draw cost
drops from a Python pass over every row object to a few numpy gathers. The
row path stays the reference behind ``MA_CC_INFORMATION_ENGINE=rows``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
    _builtin_sum,
    _cmi_from_counts,
    _first_appearance,
    _grouped,
)

from .single_affinity import ADVOCATE_ACTIONS, CONTROLLER_ACTIONS

if False:  # typing only; PairedBootstrap lives in mas_cc.studies and would import back into analysis
    from mas_cc.studies.weighted_summaries import PairedBootstrap  # noqa: F401


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


class _DrawComponents:
    """``_components`` evaluated on row-position arrays of one cell's controlled rows.

    A bootstrap draw is a sequence of rows (episodes or initialization blocks
    repeated by their drawn weight, in draw order). Every quantity ``_components``
    computes on that sequence is recomputed here from per-row codes with the
    row path's arithmetic: the CMI from the same contingency table in the same
    first-appearance axis order, the conditional entropy from the same per-state
    terms summed with ``sum()``, ``chi`` and the Pinsker numerator with the same
    ``np.mean`` over the same values in the same order and the same running sums
    over states in ascending order. The row path stays the reference behind
    ``MA_CC_INFORMATION_ENGINE=rows``.
    """

    def __init__(self, rows: Sequence[Any], *, filter_controlled: bool = False):
        self.n = len(rows)
        # For callers that hand over every row of a cell (single-affinity), the
        # methods apply the same controlled-row filter the row path applies
        # inside each estimator; for callers that pre-filter (derived
        # aggregates) the mask is all-true and costs nothing.
        self.controlled = np.array([str(row.U_k) in CONTROLLER_ACTIONS for row in rows], dtype=bool)
        self.filter_controlled = filter_controlled
        self.round_index = np.array([int(row.round_index) for row in rows], dtype=np.int64)
        actions = [str(row.U_k) for row in rows]
        x_levels = tuple(dict.fromkeys(actions))
        self.x_index = {value: i for i, value in enumerate(x_levels)}
        self.x = np.fromiter((self.x_index[value] for value in actions), dtype=np.int64, count=self.n)
        after = [int(row.target_after) for row in rows]
        before = [int(row.target_before) for row in rows]
        y_levels, z_levels = tuple(dict.fromkeys(after)), tuple(dict.fromkeys(before))
        yi, zi = {v: i for i, v in enumerate(y_levels)}, {v: i for i, v in enumerate(z_levels)}
        self.y = np.fromiter((yi[v] for v in after), dtype=np.int64, count=self.n)
        self.z = np.fromiter((zi[v] for v in before), dtype=np.int64, count=self.n)
        self.shape = (len(x_levels), len(z_levels), len(y_levels))
        self.flat = self.x * (self.shape[1] * self.shape[2]) + self.z * self.shape[2] + self.y
        self.state = np.array(before, dtype=np.int64)
        self.advocated = np.array([str(row.U_k) in ADVOCATE_ACTIONS for row in rows], dtype=bool)
        deltas = [row.event.get("delta_p_ctrl") for row in rows]
        self.has_delta = np.array([value is not None for value in deltas], dtype=bool)
        self.delta = np.array([0.0 if value is None else float(value) for value in deltas], dtype=float)

    def cmi(self, order: np.ndarray) -> float:
        counts = np.bincount(self.flat[order], minlength=int(np.prod(self.shape))).reshape(self.shape).astype(float)
        table = counts[np.ix_(_first_appearance(self.x[order]), _first_appearance(self.z[order]),
                              _first_appearance(self.y[order]))]
        return float(_cmi_from_counts(table))

    def conditional_entropy(self, order: np.ndarray) -> float:
        """``conditional_action_entropy_bits(actions, target_before)`` on the draw."""
        x, z = self.x[order], self.z[order]
        n = order.size
        terms = []
        for state in _first_appearance(z):
            group = x[z == state]
            counts = np.bincount(group, minlength=self.shape[0])
            total = group.size
            entropy = -sum((int(counts[level]) / total) * math.log2(int(counts[level]) / total)
                           for level in _first_appearance(group) if counts[level])
            terms.append((total / n) * entropy)
        return _builtin_sum(np.array(terms, dtype=float))

    def _order(self, order: np.ndarray) -> np.ndarray:
        return order[self.controlled[order]] if self.filter_controlled else order

    def _states(self, order: np.ndarray) -> list[tuple[int, float, float, bool, float]]:
        """``state_response_table`` on the draw: (state, a, chi, identified, occupancy weight), states ascending.

        The occupancy weight is ``pooled_occupancy``'s ``count / total`` over every
        controlled row of the draw (not only rows with a delta).
        """
        state, advocated, has, delta = self.state[order], self.advocated[order], self.has_delta[order], self.delta[order]
        total = int(order.size)
        occupancy_states, occupancy_counts = np.unique(state, return_counts=True)
        occupancy = {int(s): int(c) / total for s, c in zip(occupancy_states, occupancy_counts)}
        table = []
        for value in np.unique(state[has]):
            in_state = has & (state == value)
            advocate, no_op = delta[in_state & advocated], delta[in_state & ~advocated]
            identified = bool(advocate.size and no_op.size)
            chi = float(np.mean(advocate)) - float(np.mean(no_op)) if identified else math.nan
            observed = advocate.size + no_op.size
            a = math.nan if not observed else advocate.size / observed
            table.append((int(value), a, chi, identified, occupancy.get(int(value), 0.0)))
        return table

    def round_occupancy(self, order: np.ndarray) -> list[tuple[int, dict[int, float]]]:
        """``round_occupancy`` on the draw: ``(k, {n: count / total})`` in ascending ``k``."""
        rounds, states = self.round_index[order], self.state[order]
        result = []
        for k in np.unique(rounds):
            in_round = states[rounds == k]
            values, counts = np.unique(in_round, return_counts=True)
            total = int(counts.sum())
            result.append((int(k), {int(n): int(c) / total for n, c in zip(values, counts)}))
        return result

    def single_affinity_scalars(self, order: np.ndarray, *, N: int, S: np.ndarray, micro_counts: Mapping[str, int]) -> dict[str, float]:
        """The ``_SCALARS`` of ``single_affinity.point_estimate`` on one draw, row path arithmetic."""
        from mas_cc.analysis.single_affinity import (
            _affinity_from_counts, eta_th_from_components, sensing_information_nats,
        )

        order = self._order(order)
        base = self.value(order)
        sensing_nats = sensing_horizon = math.nan
        current = current_horizon = math.nan
        if order.size:
            occupancy = self.round_occupancy(order)
            per_round: dict[int, float] = {}
            for k, weights in occupancy:
                vector = np.zeros(N + 1, dtype=float)
                for n, weight in weights.items():
                    if 0 <= n <= N:
                        vector[n] = weight
                total = float(vector.sum())
                if total <= 0.0:
                    continue
                value, _ = sensing_information_nats(vector / total, S)
                if math.isfinite(value):
                    per_round[k] = float(value)
            if per_round:
                sensing_horizon = float(sum(per_round.values()))
                sensing_nats = sensing_horizon / len(per_round)
            identified = [(state, a, chi) for state, a, chi, flag, _ in self._states(order) if flag]
            if identified:
                per_round_current: dict[int, float] = {}
                for k, weights in occupancy:
                    total = 0.0
                    for state, a, chi in identified:
                        weight = weights.get(state, 0.0)
                        total += weight * float(a) * float(chi)
                    per_round_current[k] = float(N) * total
                if per_round_current:
                    current_horizon = float(sum(per_round_current.values()))
                    current = current_horizon / len(per_round_current)
        affinity = _affinity_from_counts(micro_counts)
        thermodynamics = eta_th_from_components(
            h=affinity["effective_affinity"], current_horizon=current_horizon, sensing_horizon=sensing_horizon)
        return {
            "susceptibility_occupancy_weighted": base["chi"],
            "eta_ir": base["eta_ir"],
            "eta_ir_pinsker_numerator_bits": base["ir_numerator"],
            "eta_ir_denominator_T_bits": base["ir_denominator"],
            "target_sensing_information_nats": sensing_nats,
            "target_sensing_information_horizon_nats": sensing_horizon,
            "effective_affinity": affinity["effective_affinity"],
            "kinetic_compliance": affinity["kinetic_compliance"],
            "controlled_current": current,
            "controlled_current_horizon": current_horizon,
            "affinity_weighted_current_nats": thermodynamics["affinity_weighted_current_nats"],
            "thermodynamic_control_expenditure_nats": thermodynamics["thermodynamic_control_expenditure_nats"],
            "eta_th": thermodynamics["eta_th"],
            "eta_th_signed": thermodynamics["eta_th_signed"],
            "eta_th_bounded": thermodynamics["eta_th_bounded"],
        }

    def value(self, order: np.ndarray) -> dict[str, float]:
        order = self._order(order)
        if order.size == 0:
            return {name: math.nan for name in ("T", "H", "chi", "eta_if", "ir_numerator", "ir_denominator", "eta_ir")}
        transfer = self.cmi(order)
        entropy = self.conditional_entropy(order)
        # state_response_table + pooled_occupancy + susceptibility_summary + eta_ir, in draw order
        state, advocated, has, delta = self.state[order], self.advocated[order], self.has_delta[order], self.delta[order]
        total = int(order.size)
        occupancy_states, occupancy_counts = np.unique(state, return_counts=True)
        occupancy = {int(s): int(c) / total for s, c in zip(occupancy_states, occupancy_counts)}
        weighted = mass = numerator = 0.0
        for value in np.unique(state[has]):
            in_state = has & (state == value)
            advocate, no_op = delta[in_state & advocated], delta[in_state & ~advocated]
            if not (advocate.size and no_op.size):
                continue
            chi = float(np.mean(advocate)) - float(np.mean(no_op))
            a = advocate.size / (advocate.size + no_op.size)
            weight = occupancy.get(int(value), 0.0)
            weighted += weight * chi
            mass += weight
            numerator += weight * 2.0 * a * (1.0 - a) * chi * chi / math.log(2.0)
        chi_summary = math.nan if mass <= 0 else weighted / mass
        valid = bool(math.isfinite(transfer) and transfer > 0.0 and mass > 0.0)
        return {
            "T": transfer,
            "H": entropy,
            "chi": _finite(chi_summary),
            "eta_if": transfer / entropy if math.isfinite(entropy) and entropy > 1e-12 else math.nan,
            "ir_numerator": _finite(numerator),
            "ir_denominator": _finite(transfer),
            "eta_ir": _finite((numerator / transfer) if valid else math.nan),
        }


def _draw_orders(rows: Sequence[Any], *, bootstrap_plan: Any, resamples: int, seed: int):
    """Row-position arrays of each bootstrap draw, in the row path's draw order."""
    if bootstrap_plan is not None:
        by_block: dict[int, list[int]] = {}
        for position, event in enumerate(rows):
            by_block.setdefault(bootstrap_plan.episodes[(str(event.cell_id), str(event.episode_id))], []).append(position)
        blocks = sorted(by_block)
        members = {block: np.array(by_block[block], dtype=np.int64) for block in blocks}
        for weights in bootstrap_plan.weights:
            parts = [np.tile(members[block], int(weights[block])) for block in blocks if int(weights[block]) > 0]
            yield np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        return
    if resamples < 0:
        raise ValueError("resamples cannot be negative")
    by_episode = _grouped(rows, key=lambda row: row.episode_id)
    ids = tuple(by_episode)
    if not ids or resamples == 0:
        return
    position = {id(row): i for i, row in enumerate(rows)}
    members = {str(key): np.fromiter((position[id(row)] for row in group), dtype=np.int64, count=len(group))
               for key, group in by_episode.items()}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        selected = rng.choice(ids, size=len(ids), replace=True)
        yield np.concatenate([members[str(episode_id)] for episode_id in selected])


__all__ = ["_DrawComponents", "_draw_orders"]
