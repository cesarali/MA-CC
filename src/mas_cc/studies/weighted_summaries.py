"""Explicit estimands and paired uncertainty for cross-cell blackboard summaries.

Round scores are produced by the authoritative causal/epistemic engines. This
module combines sufficient statistics, never pooled heterogeneous CMI or cell
p-values. Bootstrap multiplicities are transient and shared across estimators.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Mapping

import numpy as np
import pandas as pd


class PairedBootstrap:
    """Resample initialization blocks within observed cell-membership strata.

Stratification preserves each cell's episode count even for incomplete paired
designs. Hashes establish pairing; local repetition numbers alone do not.
"""

    def __init__(self, rounds: pd.DataFrame, *, resamples: int, seed: int):
        if resamples < 0:
            raise ValueError("bootstrap resamples must be non-negative")
        self.resamples = resamples
        if "episode_complete" in rounds:
            rounds = rounds[rounds["episode_complete"].fillna(False)]
        identities: dict[tuple[str, str], str] = {}
        columns = [c for c in ("cell_id", "episode_id", "physical_initial_state_hash",
                               "initialization_artifact_hash") if c in rounds]
        for row in rounds[columns].drop_duplicates().to_dict("records"):
            key = (str(row["cell_id"]), str(row["episode_id"]))
            block = None
            for field in ("physical_initial_state_hash", "initialization_artifact_hash"):
                value = row.get(field)
                if value is not None and not pd.isna(value) and str(value):
                    block = str(value)
                    break
            block = block or json.dumps(key)
            if key in identities and identities[key] != block:
                raise ValueError("inconsistent initialization identity within episode")
            identities[key] = block
        self.blocks = sorted(set(identities.values()))
        self.index = {block: i for i, block in enumerate(self.blocks)}
        self.episodes = {key: self.index[block] for key, block in identities.items()}
        membership: dict[str, list[str]] = defaultdict(list)
        for (cell, _), block in identities.items():
            membership[block].append(cell)
        strata: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for block, members in membership.items():
            strata[tuple(sorted(members))].append(self.index[block])
        self.weights = np.zeros((resamples, len(self.blocks)), dtype=np.int32)
        self.strata = {members: tuple(sorted(indices)) for members, indices in strata.items()}
        for members, indices in sorted(strata.items()):
            indices.sort()
            digest = hashlib.sha256(repr((seed, members)).encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            if resamples:
                self.weights[:, indices] = rng.multinomial(
                    len(indices), np.full(len(indices), 1 / len(indices)), size=resamples
                )

    def event_draws(self, events):
        by_block: dict[int, list[Any]] = defaultdict(list)
        for event in events:
            by_block[self.episodes[(str(event.cell_id), str(event.episode_id))]].append(event)
        for weights in self.weights:
            yield tuple(event for block, rows in sorted(by_block.items())
                        for _ in range(int(weights[block])) for event in rows)

    def diagnostics(self, cell_ids):
        selected = set(map(str, cell_ids))
        sizes = [len(indices) for members, indices in self.strata.items() if selected.intersection(members)]
        return {"n_bootstrap_strata": len(sizes),
                "min_bootstrap_stratum_blocks": min(sizes, default=0),
                "singleton_bootstrap_strata": sum(size == 1 for size in sizes)}

    def totals(self, rows: pd.DataFrame, values: np.ndarray) -> np.ndarray:
        indices = [self.episodes[(str(c), str(e))]
                   for c, e in rows[["cell_id", "episode_id"]].itertuples(index=False, name=None)]
        sums = np.bincount(indices, weights=values, minlength=len(self.blocks))
        return self.weights @ sums


EPISTEMIC_MEANS = {
    "fraction_rounds_collectively_solvable": "collective_solvable",
    "mean_fragmentation_gap": "fragmentation_gap",
    "mean_symbolic_individual_solvability_share": "symbolic_individual_solvability_share",
    "mean_active_union_fact_fraction": "active_union_fact_fraction",
    "mean_active_fact_occurrence_count": "active_fact_occurrence_count",
    "mean_active_holder_redundancy": "active_mean_holder_redundancy",
    "mean_collective_gold_probability": "collective_gold_probability",
    "mean_collective_entropy": "collective_normalized_entropy",
    "mean_configured_robustness": "configured_robustness",
    "mean_reference_robustness": "reference_robustness",
}


def _ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan),
                     where=np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0))


def _cell_components(frame, numerator, denominator, plan, *, equal_episode=False):
    """Mean score/components within cell, optionally first averaging episodes."""
    source = frame[["cell_id", "episode_id"]].copy()
    source["num"] = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    source["den"] = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    source = source[np.isfinite(source["num"]) & np.isfinite(source["den"])]
    if equal_episode:
        source = source.groupby(["cell_id", "episode_id"], as_index=False)[["num", "den"]].mean()
    result = {}
    for cell, group in source.groupby("cell_id", sort=True):
        counts = plan.totals(group, np.ones(len(group)))
        result[str(cell)] = {
            "num": float(group["num"].mean()), "den": float(group["den"].mean()),
            "draw_num": _ratio(plan.totals(group, group["num"].to_numpy()), counts),
            "draw_den": _ratio(plan.totals(group, group["den"].to_numpy()), counts),
            "draw_count": counts, "count": len(group),
            "episodes": group["episode_id"].nunique(),
        }
    return result


def _summary_rows(components, cells, groupings, metric, plan, settings, analysis_hash,
                  *, extra=None, causal_support=None, weighting="balanced_cell",
                  within_cell="round_mean"):
    rows = []
    alpha = (1 - float(settings["confidence"])) / 2
    for spec in groupings:
        keys = list(spec["group_by"])
        grouped = cells.groupby(keys, sort=True, dropna=False) if keys else [((), cells)]
        for key, group in grouped:
            key = key if isinstance(key, tuple) else (key,)
            expected = sorted(group["cell_id"].astype(str))
            selected = [c for c in expected if c in components and
                        (causal_support is None or causal_support.get(c, False)) and
                        components[c]["den"] > 0]
            items = [components[c] for c in selected]
            weights = np.array([v["count"] if weighting == "n_observations" else 1.0 for v in items])
            if items:
                num = float(np.average([v["num"] for v in items], weights=weights))
                den = float(np.average([v["den"] for v in items], weights=weights))
                value = float(_ratio(num, den))
                draw_weights = np.stack([v["draw_count"] if weighting == "n_observations"
                                         else np.ones(plan.resamples) for v in items])
                # Missing components invalidate a draw; never renormalize a
                # different set of cells independently in a ratio's two parts.
                draw_num = np.stack([v["draw_num"] for v in items])
                draw_den = np.stack([v["draw_den"] for v in items])
                draw_num = np.where(draw_weights > 0, draw_num, 0.)
                draw_den = np.where(draw_weights > 0, draw_den, 0.)
                draws = _ratio((draw_weights * draw_num).sum(axis=0),
                               (draw_weights * draw_den).sum(axis=0))
                finite = draws[np.isfinite(draws)]
            else:
                num = den = value = float("nan")
                finite = np.array([])
            complete = len(selected) == len(expected)
            diagnostics = plan.diagnostics(selected)
            if {"completed_episodes", "expected_episodes"}.issubset(group.columns):
                complete = complete and bool((group.completed_episodes == group.expected_episodes).all())
            units = "target_share_change" if metric == "propensity_weighted_causal_response" else (
                "agent_fact_pairs" if metric == "mean_active_fact_occurrence_count" else
                "holders_per_fact" if metric == "mean_active_holder_redundancy" else "dimensionless")
            rows.append({
                **dict(zip(keys, key, strict=True)), **(extra or {}),
                "metric": metric, "estimate": value,
                "units": units,
                "component_numerator": num, "component_denominator": den,
                "ci_low": float(np.quantile(finite, alpha)) if len(finite) else np.nan,
                "ci_high": float(np.quantile(finite, 1-alpha)) if len(finite) else np.nan,
                "confidence": settings["confidence"],
                **diagnostics,
                "bootstrap_resamples": plan.resamples, "n_valid_bootstrap_draws": len(finite),
                "bootstrap_unit": "shared_initialization_block",
                "bootstrap_scope": "stratified_by_observed_cell_membership",
                "aggregation_name": spec["name"], "aggregation_weight": weighting,
                "within_cell_weighting": within_cell,
                "marginalized_dimensions": json.dumps(list(spec["marginalize"])),
                "n_expected_cells": len(expected), "n_contributing_cells": len(selected),
                "cell_coverage_fraction": len(selected)/len(expected) if expected else np.nan,
                "contributing_cell_ids": json.dumps(selected),
                "n_episodes": sum(v["episodes"] for v in items),
                "n_observations": sum(v["count"] for v in items),
                "support_status": "unsupported" if not items else "limited" if not complete or
                    any(v["episodes"] < 2 for v in items) or
                    diagnostics["singleton_bootstrap_strata"] > 0 or
                    (plan.resamples and len(finite) < plan.resamples) else "adequate",
                "descriptive_only": weighting == "n_observations" or causal_support is None,
                "analysis_hash": analysis_hash, "analysis_semantics_version": "paired-summary-v1",
            })
    return rows


def derive_blackboard_summaries(tables: Mapping[str, pd.DataFrame], cells: pd.DataFrame,
                               recipe, settings, analysis_hash, plan: PairedBootstrap):
    """Balanced causal effects, availability ratios, and equal-episode epistemics."""
    from .derived_aggregation import _validate_groupings, _validate_state_local_groupings

    config = recipe.get("derived_study_aggregates", {})
    if not config.get("enabled", False):
        return {}
    groupings = _validate_groupings(config, cells)
    result = {}
    causal = tables.get("causal_response_round_inputs", pd.DataFrame())
    if config.get("causal", False):
        if causal.empty:
            raise ValueError("causal study summaries require causal_response_round_inputs")
        causal = causal[causal["episode_complete"]].copy()
        rows = []
        for lag in (1, 2, 3):
            field = f"causal_response_h{lag}"
            frame = causal[causal[field].notna()]
            support = frame.groupby("cell_id")["U_t"].nunique().eq(2).to_dict()
            components = _cell_components(frame, frame[field], pd.Series(1., index=frame.index), plan)
            rows += _summary_rows(components, cells, groupings, "propensity_weighted_causal_response",
                                  plan, settings, analysis_hash, extra={"lag": lag}, causal_support=support)
        frame = causal[causal["causal_response_h1"].notna() & causal["x_t"].lt(1)].copy()
        support = frame.groupby("cell_id")["U_t"].nunique().eq(2).to_dict()
        components = _cell_components(frame, frame["causal_response_h1"], 1-frame["x_t"], plan)
        rows += _summary_rows(components, cells, groupings, "available_mass_weighted_causal_susceptibility",
                              plan, settings, analysis_hash, causal_support=support,
                              within_cell="ratio_of_round_means", extra={"lag": 1})
        result["causal_response_aggregated_metrics"] = pd.DataFrame(rows)
        local_config = config.get("state_local", {})
        if local_config.get("enabled", False):
            specs = _validate_state_local_groupings(local_config, cells)
            bins = int(local_config.get("x_bins", 8))
            rows = []
            for available in (False, True):
                local_source = frame if available else causal[causal["causal_response_h1"].notna()].copy()
                indices = np.minimum((bins*local_source["x_t"]).astype(int), bins-1)
                for index in range(bins):
                    local = local_source[indices == index]
                    support = local.groupby("cell_id")["U_t"].nunique().eq(2).to_dict()
                    score = local["causal_response_h1"]
                    if available:
                        score = score/(1-local["x_t"])
                    components = _cell_components(local, score, pd.Series(1., index=local.index), plan)
                    metric = "propensity_weighted_available_susceptibility" if available else "propensity_weighted_causal_response"
                    for spec in specs:
                        rows += _summary_rows(components, cells, [spec], metric,
                                              plan, settings, analysis_hash, causal_support=support,
                                              weighting=spec["weighting"], extra={"lag": 1,
                                              "target_fraction_bin_index": index,
                                              "target_fraction_bin_center": (index+.5)/bins})
            result["causal_state_local_aggregated_metrics"] = pd.DataFrame(rows)
    if config.get("epistemic", False):
        states = tables.get("epistemic_round_timeseries", pd.DataFrame())
        if states.empty or causal.empty:
            raise ValueError("epistemic study summaries require symbolic states and complete episode identities")
        complete = causal.loc[causal["episode_complete"], ["cell_id", "episode_id"]].drop_duplicates()
        states = states.merge(complete, on=["cell_id", "episode_id"], validate="many_to_one")
        rows = []
        for metric, field in EPISTEMIC_MEANS.items():
            components = _cell_components(states, states[field], pd.Series(1., index=states.index),
                                          plan, equal_episode=True)
            rows += _summary_rows(components, cells, groupings, metric, plan, settings, analysis_hash,
                                  within_cell="equal_episode_mean_of_round_means")
        result["epistemic_aggregated_metrics"] = pd.DataFrame(rows)
    from .derived_aggregation import PROTECTED_COORDINATES
    for output in result.values():
        for column in PROTECTED_COORDINATES:
            if column not in output and column in cells and cells[column].nunique(dropna=False) == 1:
                output[column] = cells[column].iloc[0]
    return result


def weighted_rho_aliases(outputs):
    """Keep established report sources on the component-weighted estimands.

Legacy descriptive tables used means of efficiency ratios. Once the new
derived suite is enabled, its rows are authoritative for matching rho views.
"""
    labels = {"round_target_actuation_cmi": "T_pi", "susceptibility_occupancy_weighted": "chi",
              "round_target_information_fraction": "eta_IF", "eta_ir": "eta_IR"}
    for source, target in (("study_aggregated_metrics", "rho_aggregated_descriptive_summary"),
                           ("state_local_aggregated_metrics", "rho_aggregated_state_local_maps")):
        new = outputs.get(source, pd.DataFrame())
        if new.empty:
            continue
        new = new[new["marginalized_dimensions"].map(json.loads).map(lambda v: v == ["epistemic_persistence"])].copy()
        if new.empty:
            continue
        new["source_metric"] = new["metric"]
        new["metric"] = new["metric"].map(labels)
        new = new[new["metric"].notna()]
        new["phase_status"] = new["support_status"].replace({"unsupported": "insufficient_estimator_support"})
        old = outputs.get(target, pd.DataFrame())
        # Preserve the expected empty state grid, but never retain an old
        # efficiency value or interval when its new estimate is unsupported.
        if not old.empty:
            replaced = old[old["metric"].isin(new["metric"])].copy()
            keys = [c for c in ("intervention_budget", "target_semantics", "target_fraction_bin_index", "metric")
                    if c in new and c in replaced]
            if new.duplicated(keys).any():
                raise ValueError("multiple derived rho views map to the same report coordinates")
            missing = replaced.merge(new[keys].assign(_present=True), on=keys, how="left")
            missing = missing[missing["_present"].isna()].drop(columns="_present")
            missing["estimate"] = np.nan
            missing["phase_status"] = "insufficient_estimator_support"
            new = pd.concat([new, missing, old[~old["metric"].isin(new["metric"])]], ignore_index=True)
        outputs[target] = new
