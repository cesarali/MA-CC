"""Versioned, offline blackboard calibration; no legacy affinity semantics change.

Inputs are completed canonical study tables. Exposure is descriptive, not a
randomized treatment. Closed-form predictions are explicitly assumed references.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

VERSION = "blackboard_calibration_v1"
ADAPTER_VERSION = "blackboard_retained_updates_v1"
VARIANTS = ("direct_active", "mixture_common_weight", "mixture_start_vote_weighted")
TABLES = ("blackboard_calibration_counts", "blackboard_calibration_estimates",
          "blackboard_calibration_diagnostics", "blackboard_model_predictions",
          "blackboard_model_validation", "blackboard_calibration_inputs",
          "blackboard_calibration_splits")
NAN = math.nan


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _id(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()[:24]


def _present(value: Any) -> bool:
    if value is None:
        return False
    missing = pd.isna(value)
    return not bool(missing) if isinstance(missing, (bool, np.bool_)) else True


def _first(row: Mapping, *keys: str, default=None):
    return next((row[k] for k in keys if k in row and _present(row[k])), default)


def _list(value: Any) -> list | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return list(value) if isinstance(value, (list, tuple, np.ndarray)) else None


def calibration_settings(raw: Any) -> dict | None:
    """Strict recipe parser: unsupported fits cannot be silently requested."""
    if raw is None or raw is False:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("blackboard_calibration_outputs must be a mapping")
    defaults = dict(enabled=True, estimator_version=VERSION,
                    exposure_definition="sampled_controller_authored_message",
                    calibration_scope="physical_cell", action_stratification=True,
                    shared_unexposed_baseline=False,
                    exposure_prediction="uniform_without_replacement",
                    active_parameter_variants=list(VARIANTS),
                    model_predictions={"enabled": False}, diagnostics={})
    unknown = set(raw) - set(defaults)
    if unknown:
        raise ValueError(f"unknown blackboard calibration settings: {sorted(unknown)}")
    result = {**defaults, **raw}
    for key in ("enabled", "action_stratification", "shared_unexposed_baseline"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("estimator_version", "exposure_definition", "calibration_scope"):
        if result[key] != defaults[key]:
            raise ValueError(f"unsupported blackboard {key}: {result[key]}")
    if not result["action_stratification"]:
        raise ValueError("blackboard v1 requires action_stratification")
    if result["exposure_prediction"] not in {"uniform_without_replacement", "none"}:
        raise ValueError("unsupported exposure_prediction")
    variants = result["active_parameter_variants"]
    if not isinstance(variants, list) or any(not isinstance(v, str) for v in variants) or len(set(variants)) != len(variants) or not set(variants) <= set(VARIANTS):
        raise ValueError("invalid active_parameter_variants")
    diagnostics = result["diagnostics"]
    if not isinstance(diagnostics, Mapping) or set(diagnostics) - {
        "starting_vote_exposure", "silent_exposure", "unexposed_branch_comparison"
    } or any(v is not True for v in diagnostics.values()):
        raise ValueError("v1 requires all calibration diagnostics")
    prediction = result["model_predictions"]
    if not isinstance(prediction, Mapping) or set(prediction) - {
        "enabled", "reference", "evaluation_fraction", "assume_homogeneous_channels"
    }:
        raise ValueError("invalid model_predictions settings")
    prediction = {"enabled": False, "reference": "frozen_board_homogeneous", **prediction}
    if type(prediction["enabled"]) is not bool:
        raise ValueError("model_predictions.enabled must be boolean")
    if prediction["reference"] != "frozen_board_homogeneous":
        raise ValueError("unsupported blackboard model reference")
    if prediction["enabled"]:
        fraction = prediction.get("evaluation_fraction")
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
            raise ValueError("predictions require explicit evaluation_fraction in (0,1)")
        if prediction.get("assume_homogeneous_channels") is not True:
            raise ValueError("predictions require assume_homogeneous_channels: true")
        if "mixture_common_weight" not in variants:
            raise ValueError("predictions require mixture_common_weight")
    result["model_predictions"] = prediction
    return result if result["enabled"] else None


def sampling_exposure_probability(B: int, C: int, m: int) -> float:
    """Stable hypergeometric probability of at least one controller message."""
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 0 for v in (B, C, m)) or m > B + C:
        raise ValueError("sampling counts must be nonnegative integers with m <= B+C")
    if m == 0 or C == 0:
        return 0.0
    if m > B:
        return 1.0
    return -math.expm1(sum(math.log1p(-C / (B + C - j)) for j in range(m)))


def channel_parameters(a: float, d: float) -> dict:
    """Descriptive rates and signed boundaries without clipping or pseudocounts."""
    if not math.isfinite(a) or not math.isfinite(d):
        return dict(p_plus=a, p_minus=d, gamma=NAN, p=NAN, h=NAN,
                    support_status="missing_opportunities", model_status="unsupported")
    if not 0 <= a <= 1 or not 0 <= d <= 1:
        raise ValueError("transition probabilities must lie in [0,1]")
    gamma = a + d
    p = a / gamma if gamma else NAN
    h = math.log(a / d) if a > 0 and d > 0 else (
        math.inf if a > 0 else -math.inf if d > 0 else NAN)
    return dict(p_plus=a, p_minus=d, gamma=gamma, p=p, h=h,
                support_status="boundary" if a == 0 or d == 0 else "supported",
                model_status="incompatible_gamma" if gamma > 1 else "compatible_rates")


def mixture_parameters(c0: Mapping, c1: Mapping, w_plus: float, w_minus: float | None = None) -> dict:
    def mix(key, w):
        if not math.isfinite(w):
            return NAN
        if not 0 <= w <= 1:
            raise ValueError("mixture weight must lie in [0,1]")
        if w == 0:
            return c0[key]
        if w == 1:
            return c1[key]
        return (1 - w) * c0[key] + w * c1[key]
    return channel_parameters(mix("p_plus", w_plus), mix("p_minus", w_plus if w_minus is None else w_minus))


def mean_map(a: float, d: float, N: int, M: int, x: float) -> dict:
    """Stable affine finite-round reference, including the identity kernel."""
    if not isinstance(N, (int, np.integer)) or N <= 0 or not isinstance(M, (int, np.integer)) or M < 0 or not 0 <= x <= 1:
        raise ValueError("mean map requires N>0, M>=0 integers and x in [0,1]")
    gamma = a + d
    if M == 0:
        return dict(r=1.0, intercept=0.0, mean=x)
    if not math.isfinite(gamma) or gamma > 1:
        return dict(r=NAN, intercept=NAN, mean=NAN)
    channel_parameters(a, d)
    if gamma == 0:
        return dict(r=1.0, intercept=0.0, mean=x)
    log_r = -math.inf if gamma == N else M * math.log1p(-gamma / N)
    r = math.exp(log_r)
    intercept = a / gamma * -math.expm1(log_r)
    return dict(r=r, intercept=intercept, mean=r * x + intercept)


def susceptibility(c0: Mapping, cb: Mapping, N: int, M: int, x: float) -> dict:
    f0 = mean_map(c0["p_plus"], c0["p_minus"], N, M, x)
    fb = mean_map(cb["p_plus"], cb["p_minus"], N, M, x)
    c, s = fb["intercept"] - f0["intercept"], fb["r"] - f0["r"]
    chi = c + s * x
    root = -c / s if math.isfinite(s) and s != 0 else NAN
    return dict(r0=f0["r"], rb=fb["r"], chi=chi,
                available=chi / (1-x) if x < 1 else NAN, intercept=c, slope=s,
                x_star=root, root_status=("in_range" if 0 <= root <= 1 else "out_of_range")
                if math.isfinite(root) else "all_states" if s == 0 and c == 0 else "no_root" if s == 0 else "unsupported")


def adapt_calibration_inputs(micro: pd.DataFrame, rounds: pd.DataFrame,
                             episodes: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Join only completed canonical identities; ambiguous duplicates are errors.

    Legacy compact records lack eligible composition. The transient-direct flag
    is deliberately never a fallback for actual sampled-board exposure.
    """
    if micro.empty:
        return pd.DataFrame()
    def unique(frame, keys, label):
        if not set(keys) <= set(frame) or frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"{label} requires unique, nonmissing canonical identities")
    unique(episodes, ["cell_id", "episode_id"], "episodes")
    unique(cells, ["cell_id"], "cells")
    unique(rounds, ["cell_id", "episode_id", "round_index"], "rounds")
    unique(micro, ["cell_id", "episode_id", "round_index", "micro_slot_index"], "micro slots")
    ep = {(str(r["cell_id"]), str(r["episode_id"])): r for r in episodes.to_dict("records")}
    rr = {(str(r["cell_id"]), str(r["episode_id"]), r["round_index"]): r for r in rounds.to_dict("records")}
    cc = {str(r["cell_id"]): r for r in cells.to_dict("records")}
    records = []
    for row in micro.to_dict("records"):
        key = (str(row["cell_id"]), str(row["episode_id"]))
        episode = ep.get(key)
        if episode is None:
            raise ValueError("micro slot references unknown canonical episode")
        if episode.get("status") not in {"completed", "skipped_resumed"}:
            continue
        rd = rr.get((*key, row["round_index"]))
        if rd is None or key[0] not in cc:
            raise ValueError("micro slot references unknown canonical round/cell")
        joined = {**cc[key[0]], **{k: v for k, v in rd.items() if _present(v)},
                  **{k: v for k, v in row.items() if _present(v)}}
        if joined.get("social_mode") != "board":
            continue
        target = _first(rd, "analysis_target", "controller_target")
        micro_target = _first(row, "analysis_target", "round_controller_target")
        if target is not None and micro_target is not None and str(target) != str(micro_target):
            raise ValueError("micro and round target orientations disagree")
        target = target if target is not None else micro_target
        # analysis_target in runtime includes the documented gold fallback.
        before, after = row.get("focal_opinion_before"), row.get("focal_opinion_after")
        valid = all(_present(v) for v in (target, before, after)) and row.get("valid_update", True) is not False
        action = _first(rd, "U_k", "controller_sampled_U")
        if action is None:
            label = _first(rd, "controller_action", default=row.get("round_controller_action"))
            action = 1 if label in {"ADVOCATE_TARGET", "ADVOCATE_Z"} else 0 if label in {"SILENCE", "SILENT", "NOOP", "NONE"} or rd.get("controller_enabled") is False else None
        if action is not None and action not in (0, 1):
            raise ValueError("round action must be binary")
        ids = _list(row.get("sampled_controller_message_ids"))
        sampled = _list(row.get("sampled_message_ids"))
        if ids is not None and sampled is not None and not set(ids) <= set(sampled):
            raise ValueError("controller sample ids are not a subset of sampled ids")
        transient = _first(row, "controller_message_directly_exposed", default=False)
        # Legacy ids include transient recommendations; those updates are a
        # different sampling law and cannot identify the requested board E.
        exposure = int(bool(ids)) if ids is not None and not transient else None
        sampling_status, wp = "missing_eligible_composition", NAN
        B, C, m = (_first(row, k) for k in ("eligible_peer_message_count", "eligible_controller_message_count", "board_sample_size"))
        if joined.get("board_sampling") not in {"uniform", "uniform_without_replacement"}:
            sampling_status = "inapplicable_sampler"
        elif transient:
            sampling_status = "inapplicable_transient_recommendation"
        elif all(v is not None for v in (B, C, m)):
            if any(float(v) != int(v) for v in (B, C, m)):
                raise ValueError("noninteger eligible/sample counts")
            B, C, m = int(B), int(C), int(m)
            wp = sampling_exposure_probability(B, C, m)
            if sampled is not None and len(sampled) != m:
                raise ValueError("sample size disagrees with retained messages")
            if ids is not None and (len(ids) > min(C, m) or m-len(ids) > B):
                raise ValueError("sampled controller ids disagree with eligible board composition")
            sampling_status = "available"
        block = _first(rd, "physical_initial_state_hash", "initialization_artifact_hash")
        if block is None and rd.get("initialization_source") == "paired_artifact":
            raise ValueError("paired initialization requires a retained block hash")
        block = str(block) if block is not None else "/".join(key)
        counts_before = _list(rd.get("occupation_counts_before"))
        counts_after = _list(rd.get("occupation_counts_after"))
        options = _list(rd.get("possible_answers"))
        N = _first(joined, "N", "population_size")
        x = _first(rd, "controller_target_share_before", "target_fraction_before", default=NAN)
        y = _first(rd, "controller_target_share", "target_fraction_after", default=NAN)
        if options and target is not None and str(target) in list(map(str, options)):
            idx = list(map(str, options)).index(str(target))
            for counts, direct in ((counts_before, x), (counts_after, y)):
                if counts and math.isfinite(float(direct)) and not math.isclose(float(direct), counts[idx]/sum(counts), abs_tol=1e-12):
                    raise ValueError("round target share disagrees with target counts")
            if counts_before and sum(counts_before):
                x = counts_before[idx] / sum(counts_before)
            if counts_after and sum(counts_after):
                y = counts_after[idx] / sum(counts_after)
        records.append(dict(
            study_id=joined.get("study_id"), source_run_id=joined.get("source_run_id"),
            cell_id=key[0], episode_id=key[1], round_index=row["round_index"],
            micro_slot_index=row["micro_slot_index"], block_id=block,
            task_id=joined.get("task_id"), focal_agent_id=row.get("focal_agent_id"),
            U=action, E=exposure,
            z_before=int(str(before)==str(target)) if all(_present(v) for v in (before, target)) else None,
            z_after=int(str(after)==str(target)) if all(_present(v) for v in (after, target)) else None,
            valid_update=valid,
            b_budget=_first(joined, "intervention_budget", "b"),
            b_posted=_first(rd, "actual_controller_posts", "controller_posts"),
            B=B, C=C, m=m, sampling_probability=wp, sampling_status=sampling_status,
            N=N, x=x, x_after=y,
            actual_update_count=rd.get("actual_update_count"),
            propensity=_first(rd, "P_U1_given_Y", "controller_probability_U1_given_Y", "controller_action_probability", default=NAN),
            focal_selection_rule=joined.get("focal_selection_rule", "unverified"),
            adapter_version=ADAPTER_VERSION, source_selection="completed_canonical_episodes",
        ))
    result = pd.DataFrame(records)
    if not result.empty:
        for _, cell in result.groupby("cell_id"):
            sizes = pd.to_numeric(cell["N"], errors="coerce").dropna()
            if sizes.nunique() > 1 or ((sizes <= 0) | (sizes % 1 != 0)).any():
                raise ValueError("physical-cell population size must be a fixed positive integer")
        result["M"] = result.groupby(["cell_id", "episode_id", "round_index"])["micro_slot_index"].transform("size")
        retained_count = pd.to_numeric(result["actual_update_count"], errors="coerce")
        if (retained_count.notna() & retained_count.ne(result["M"])).any():
            raise ValueError("retained micro slots disagree with actual round update count")
        for _, episode in result.groupby(["cell_id", "episode_id"]):
            if episode.block_id.nunique() != 1:
                raise ValueError("initialization block changes within episode")
    return result


def _counts(rows: Sequence[Mapping]) -> dict:
    valid = [r for r in rows if r["valid_update"]]
    plus = [r for r in valid if r["z_before"] == 0]
    minus = [r for r in valid if r["z_before"] == 1]
    known = [r for r in rows if _present(r["E"])]
    return dict(n_observations=len(rows), n_episodes=len({r["episode_id"] for r in rows}),
                n_blocks=len({r["block_id"] for r in rows}),
                n_plus_blocks=len({r["block_id"] for r in plus}),
                n_minus_blocks=len({r["block_id"] for r in minus}),
                n_plus_episodes=len({r["episode_id"] for r in plus}),
                n_minus_episodes=len({r["episode_id"] for r in minus}),
                D_plus=len(plus), A_plus=sum(r["z_after"] == 1 for r in plus),
                D_minus=len(minus), A_minus=sum(r["z_after"] == 0 for r in minus),
                exposure_known=len(known), exposure_events=sum(r["E"] == 1 for r in known),
                missing_exposure=len(rows)-len(known), missing_transition=len(rows)-len(valid),
                exposure_coverage=len(known)/len(rows) if rows else NAN)


def _rates(counts: Mapping) -> dict:
    return channel_parameters(counts["A_plus"] / counts["D_plus"] if counts["D_plus"] else NAN,
                              counts["A_minus"] / counts["D_minus"] if counts["D_minus"] else NAN)


def _weight(rows: Sequence[Mapping], start=None) -> float:
    known = [r for r in rows if _present(r["E"]) and (start is None or r["z_before"] == start)]
    return sum(r["E"] for r in known) / len(known) if known else NAN


def _summaries(rows: list[dict], settings: Mapping) -> dict:
    summaries = {}
    for action in (0, 1):
        branch = [r for r in rows if r["U"] == action]
        for channel in ("all", 0, 1):
            selected = branch if channel == "all" else [r for r in branch if r["E"] == channel]
            counts = _counts(selected)
            summaries[(action, str(channel), "direct_silent" if action == 0 and channel == "all" else "direct_active" if channel == "all" else "action_channel")] = {
                **counts, **_rates(counts), "w": _weight(selected),
                "w_nonZ": _weight(selected, 0), "w_Z": _weight(selected, 1),
                "branch_n_observations": len(branch),
                "excluded_unknown_exposure": sum(not _present(r["E"]) for r in branch) if channel != "all" else 0,
            }
    c0 = summaries[(1, "0", "action_channel")]
    if settings["shared_unexposed_baseline"]:
        pooled = _counts([r for r in rows if r["E"] == 0 and r["U"] in (0, 1)])
        c0 = {**pooled, **_rates(pooled)}
        summaries[("pooled", "0", "shared_unexposed")] = c0
    c1 = summaries[(1, "1", "action_channel")]
    active = [r for r in rows if r["U"] == 1]
    for variant in settings["active_parameter_variants"]:
        if variant == "direct_active":
            continue
        # Starting-vote decomposition always uses active-only channels.
        baseline = summaries[(1, "0", "action_channel")] if variant == "mixture_start_vote_weighted" else c0
        wp = _weight(active, 0) if variant == "mixture_start_vote_weighted" else _weight(active)
        wm = _weight(active, 1) if variant == "mixture_start_vote_weighted" else wp
        mixed = mixture_parameters(baseline, c1, wp, wm)
        if any(not _present(r["E"]) for r in active):
            mixed = {**mixed, "model_status": "partial_exposure_coverage"}
        summaries[(1, "mixture", variant)] = {**_counts(active), **mixed, "w": _weight(active), "w_nonZ": _weight(active, 0), "w_Z": _weight(active, 1)}
    return summaries


def _interval(point: float, draws: list[float], n_blocks: int, confidence: float,
              requested: int, boundary: bool = False) -> dict:
    valid = [float(v) for v in draws if not math.isnan(v)]
    infinite = sum(math.isinf(v) for v in valid)
    out = dict(ci_low=NAN, ci_high=NAN, bootstrap_resamples=requested,
               bootstrap_valid=len(valid), bootstrap_undefined=requested-len(valid),
               bootstrap_boundary=sum(not math.isfinite(v) or v in (0, 1) for v in valid),
               uncertainty_status="available", interval_policy="extended_real_inverted_cdf")
    if requested == 0:
        out["uncertainty_status"] = "not_requested"
    elif n_blocks < 2:
        out["uncertainty_status"] = "insufficient_independent_units"
    elif len(valid) < max(2, math.ceil(0.8*requested)):
        out["uncertainty_status"] = "insufficient_valid_replicates"
    elif math.isnan(point):
        out["uncertainty_status"] = "undefined_point"
    elif boundary and len(set(valid)) == 1:
        out["uncertainty_status"] = "degenerate_boundary"
    else:
        # Order statistics keep signed infinities; no interpolation inf-inf.
        alpha = (1-confidence)/2
        ordered = sorted(valid)
        out["ci_low"] = ordered[max(0, math.ceil(alpha*len(ordered))-1)]
        out["ci_high"] = ordered[max(0, math.ceil((1-alpha)*len(ordered))-1)]
        if infinite:
            out["uncertainty_status"] = "boundary_interval"
    return out


def _resample(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    units = defaultdict(list)
    for row in rows:
        units[row["block_id"]].append(row)
    keys = sorted(units)
    return [row for idx in rng.integers(0, len(keys), len(keys)) for row in units[keys[idx]]] if keys else []


def _exposure_stats(rows: list[dict], start=None, prediction=True) -> dict:
    selected = [r for r in rows if start is None or r["z_before"] == start]
    known = [r for r in selected if _present(r["E"])]
    matched = [r for r in known if math.isfinite(r["sampling_probability"])] if prediction else []
    return {**_counts(selected), "w": _weight(selected),
            "sampling_n": len(matched),
            "sampling": float(np.mean([r["sampling_probability"] for r in matched])) if matched else NAN,
            "sampling_observed": float(np.mean([r["E"] for r in matched])) if matched else NAN,
            "residual": float(np.mean([r["E"]-r["sampling_probability"] for r in matched])) if matched else NAN}


def _diagnostics(rows: list[dict], summary: Mapping) -> dict:
    by_round = defaultdict(list)
    for row in rows:
        by_round[(row["episode_id"], row["round_index"])].append(row)
    varying = 0
    for group in by_round.values():
        compositions = {(r["B"], r["C"], r["m"]) for r in group if all(_present(r[k]) for k in ("B", "C", "m"))}
        varying += len(compositions) > 1
    s0, a0 = summary[(0, "0", "action_channel")], summary[(1, "0", "action_channel")]
    active = summary[(1, "all", "direct_active")]
    silent = summary[(0, "all", "direct_silent")]
    common = summary.get((1, "mixture", "mixture_common_weight"), {})
    return dict(
        **_counts(rows), unknown_action=sum(not _present(r["U"]) for r in rows),
        eligible_composition_missing=sum(r["sampling_status"] == "missing_eligible_composition" for r in rows),
        sampler_inapplicable=sum(r["sampling_status"].startswith("inapplicable") for r in rows),
        rounds_with_varying_board=varying, observed_rounds=len(by_round),
        update_counts_json=_json(sorted({r["M"] for r in rows})),
        silent_exposure=silent["w"], starting_vote_exposure_difference=active["w_Z"]-active["w_nonZ"],
        unexposed_entry_branch_difference=a0["p_plus"]-s0["p_plus"],
        unexposed_exit_branch_difference=a0["p_minus"]-s0["p_minus"],
        direct_minus_common_entry=active["p_plus"]-common.get("p_plus", NAN),
        direct_minus_common_exit=active["p_minus"]-common.get("p_minus", NAN),
        exposure_interpretation="descriptive_post_treatment",
        weight_interpretation="update_weighted_observed_average",
        focal_selection_status="verified_uniform_with_replacement" if all(r["focal_selection_rule"] == "uniform_with_replacement" for r in rows) else "unverified",
        model_status="reference_assumption_required",
    )


def _prediction_status(rows: list[dict], summary: Mapping, diagnostics: Mapping) -> str:
    if not rows:
        return "unsupported_training"
    if any(not _present(r["E"]) or not r["valid_update"] for r in rows):
        return "incomplete_calibration_records"
    if any(s["model_status"] == "incompatible_gamma" for s in summary.values()):
        return "incompatible_gamma"
    silent = [r for r in rows if r["U"] == 0]
    if not silent:
        return "missing_silent_baseline"
    if any(r["E"] == 1 for r in silent):
        return "silent_exposure_incompatible"
    if diagnostics["rounds_with_varying_board"]:
        return "time_varying_board_reference_only"
    if diagnostics["eligible_composition_missing"] or diagnostics["sampler_inapplicable"]:
        return "unverified_board_reference_only"
    if diagnostics["focal_selection_status"] == "unverified":
        return "unverified_focal_selection_reference_only"
    return "homogeneous_reference_assumed"


def analyze_blackboard_calibration(
    micro: pd.DataFrame, rounds: pd.DataFrame, episodes: pd.DataFrame, cells: pd.DataFrame,
    *, settings: Mapping | None = None, bootstrap_resamples: int = 1000,
    confidence: float = 0.95, seed: int = 1, analysis_hash: str = "",
    provisional: bool = False, progress=None,
) -> dict[str, pd.DataFrame]:
    """Calibrate within physical cell/budget, with whole-block uncertainty.

    Prediction splitting is prespecified by fraction and seed, independent of
    outcomes. Fits use training blocks only; comparisons use evaluation blocks
    only. No per-slot bootstrap or cross-cell pooling occurs in v1.
    """
    settings = calibration_settings(settings if settings is not None else {})
    output = {name: [] for name in TABLES}
    if settings is None:
        return {name: pd.DataFrame() for name in TABLES}
    if bootstrap_resamples < 0 or not 0 < confidence < 1:
        raise ValueError("invalid calibration resampling settings")
    inputs = adapt_calibration_inputs(micro, rounds, episodes, cells)
    if inputs.empty:
        output["blackboard_calibration_diagnostics"].append(dict(
            support_status="no_completed_board_updates", model_status="unsupported",
            uncertainty_status="unsupported", source_selection="completed_canonical_episodes"))
    else:
        output["blackboard_calibration_inputs"] = inputs.to_dict("records")
    # Existing causal adapter owns propensity, orientation, lag and completion
    # eligibility. Do not construct purported causal strata from E or post count.
    causal_lookup = {}
    if settings["model_predictions"]["enabled"] and not inputs.empty:
        from .causal_response import build_causal_response_inputs
        complete_ids = set(zip(inputs.cell_id, inputs.episode_id))
        completed_rounds = rounds[[ (str(r.cell_id), str(r.episode_id)) in complete_ids for r in rounds.itertuples() ]]
        for cid, cell_rounds in completed_rounds.groupby("cell_id"):
            try:
                causal = build_causal_response_inputs(cell_rounds, cells, lags=(1,))
            except ValueError as exc:
                output["blackboard_calibration_diagnostics"].append(dict(
                    cell_id=cid, diagnostic_slice="causal_validation",
                    support_status="unavailable", model_status="not_applicable",
                    reason=str(exc), comparison_provenance="propensity_weighted_causal_response_v1:lag1"))
                continue
            for row in causal.to_dict("records"):
                if row["episode_complete"] and row["lag_1_available"]:
                    causal_lookup[(str(row["cell_id"]), str(row["episode_id"]), row["round_index"])] = row
    grouped = inputs.groupby(["cell_id", "b_budget"], dropna=False, sort=True) if not inputs.empty else []
    for group_index, ((cell_id, budget), frame) in enumerate(grouped):
        if progress:
            progress(dict(stage="blackboard_calibration", cell_id=cell_id, group_index=group_index))
        all_rows = frame.to_dict("records")
        units = sorted({r["block_id"] for r in all_rows})
        rng_seed = seed + int(_id([cell_id, str(budget)])[:8], 16)
        rng = np.random.default_rng(rng_seed)
        evaluation_units = set()
        if settings["model_predictions"]["enabled"]:
            if len(units) < 2:
                evaluation_units = set(units)
            else:
                number = min(len(units)-1, max(1, math.ceil(len(units)*settings["model_predictions"]["evaluation_fraction"])))
                evaluation_units = set(rng.choice(units, size=number, replace=False))
        train = [r for r in all_rows if r["block_id"] not in evaluation_units]
        evaluation = [r for r in all_rows if r["block_id"] in evaluation_units]
        training_id = _id([cell_id, str(budget), sorted(set(units)-evaluation_units), settings])
        identity = dict(study_id=all_rows[0]["study_id"], source_run_id=all_rows[0]["source_run_id"],
                        cell_id=cell_id, b_budget=budget, training_group_id=training_id,
                        grouping_json=_json({"cell_id": cell_id, "b_budget": budget}),
                        source_selection="completed_canonical_training_blocks" if evaluation_units else "completed_canonical_episodes")
        for block in units:
            block_rows = [r for r in all_rows if r["block_id"] == block]
            output["blackboard_calibration_splits"].append({**identity, "block_id": block,
                "split": "evaluation" if block in evaluation_units else "calibration",
                "episode_ids_json": _json(sorted({r["episode_id"] for r in block_rows})),
                "weight": len(block_rows), "weight_unit": "micro_update", "split_seed": rng_seed})
        summary = _summaries(train, settings)
        diagnostic = _diagnostics(train, summary)
        output["blackboard_calibration_diagnostics"].append({**identity, **diagnostic,
            "diagnostic_slice": "calibration_group", "conditioning_json": "{}"})
        # Prespecified heterogeneity diagnostics, never causal exposure strata.
        for field in ("round_index", "task_id", "focal_agent_id", "x", "b_posted"):
            slices = defaultdict(list)
            for r in train:
                slices[str(r[field])].append(r)
            for value, slice_rows in sorted(slices.items()):
                output["blackboard_calibration_diagnostics"].append({**identity,
                    **_diagnostics(slice_rows, _summaries(slice_rows, settings)),
                    "diagnostic_slice": field, "conditioning_json": _json({field: value})})
        exposure_draws, draw_summaries = [], []
        for replicate in range(bootstrap_resamples):
            sample = _resample(train, rng)
            draw_summaries.append(_summaries(sample, settings))
            exposure_draws.append({
                (action, start): _exposure_stats(
                    [r for r in sample if r["U"] == action], start,
                    settings["exposure_prediction"] != "none",
                ) for action in (0, 1) for start in (None, 0, 1)
            })
            if progress and replicate % 100 == 0:
                progress(dict(stage="blackboard_calibration_bootstrap", cell_id=cell_id,
                              replicate=replicate, requested=bootstrap_resamples))
        n_blocks = len(set(units)-evaluation_units)
        estimate_ids = {}

        def emit(metric, value, variant, action, channel, counts, draw_values, *, conditioning=None, dependencies=(), boundary=False):
            conditioning_json = _json(conditioning or {})
            eid = _id([training_id, metric, variant, action, channel, conditioning_json])
            support = ("unvisited" if counts["n_observations"] == 0 else
                       "missing_opportunities" if math.isnan(value) and counts.get("support_status") == "missing_opportunities" else
                       "unidentified_boundary" if math.isnan(value) and boundary else
                       "unsupported" if math.isnan(value) else "boundary" if boundary else "supported")
            result = {**identity, **counts, "estimate_id": eid, "metric": metric,
                      "estimate": value, "estimator_variant": variant, "branch": str(action),
                      "channel": str(channel), "conditioning_json": conditioning_json,
                      "dependencies_json": _json(list(dependencies)), "confidence": confidence,
                      "units": "nats" if metric.endswith("affinity") else "dimensionless",
                      "support_status": support,
                      "model_status": counts.get("model_status", "descriptive"),
                      **_interval(value, draw_values, min(n_blocks, counts.get("n_blocks", n_blocks)), confidence, bootstrap_resamples, boundary)}
            output["blackboard_calibration_estimates"].append(result)
            return eid

        for key, stat in summary.items():
            action, channel, variant = key
            count_id = _id([training_id, key, "counts"])
            output["blackboard_calibration_counts"].append({**identity, **stat, "count_id": count_id,
                "branch": str(action), "channel": channel, "estimator_variant": variant})
            if variant == "direct_active" and variant not in settings["active_parameter_variants"]:
                continue
            dep_ids = [count_id]
            if "mixture" in variant:
                dep_ids = [_id([training_id, (1, c, "action_channel"), "counts"]) for c in ("0", "1")]
                if settings["shared_unexposed_baseline"] and variant == "mixture_common_weight":
                    dep_ids[0] = _id([training_id, ("pooled", "0", "shared_unexposed"), "counts"])
                dep_ids.append(_id([training_id, (1, "all", "direct_active"), "counts"]))
            prefix = "blackboard_effective" if channel in {"all", "mixture"} else "blackboard_channel"
            metrics = {"p_plus": "blackboard_target_entry_probability", "p_minus": "blackboard_target_exit_probability",
                       "gamma": prefix+"_compliance", "p": prefix+"_preference", "h": prefix+"_affinity"}
            for field, metric in metrics.items():
                value = stat[field]
                support_blocks = stat["n_plus_blocks"] if field == "p_plus" else stat["n_minus_blocks"] if field == "p_minus" else min(stat["n_plus_blocks"], stat["n_minus_blocks"])
                estimate_ids[(*key, field)] = emit(metric, value, variant, action, channel, {**stat, "n_blocks": support_blocks},
                    [s[key][field] for s in draw_summaries], dependencies=dep_ids,
                    boundary=stat["support_status"] == "boundary")
                output["blackboard_calibration_estimates"][-1]["bootstrap_boundary"] = sum(
                    s[key]["support_status"] == "boundary" for s in draw_summaries
                )
        for action in (0, 1):
            branch = [r for r in train if r["U"] == action]
            for start in (None, 0, 1):
                stat = _exposure_stats(branch, start, settings["exposure_prediction"] != "none")
                ds = [sample[(action, start)] for sample in exposure_draws]
                count_id = _id([training_id, action, start, "exposure_counts"])
                output["blackboard_calibration_counts"].append({**identity, **stat, "count_id": count_id,
                    "branch": str(action), "channel": "all", "estimator_variant": "observed_average",
                    "conditioning_json": _json({"z_before": start})})
                for field, metric in (("w", "blackboard_exposure_probability"),
                                      ("sampling", "blackboard_sampling_exposure_probability"),
                                      ("residual", "blackboard_exposure_probability_residual")):
                    emit(metric, stat[field], "observed_average" if field == "w" else "matched_sampler_rows",
                         action, "all", stat, [d[field] for d in ds], conditioning={"z_before": start},
                         dependencies=[count_id], boundary=stat[field] in (0, 1))

        if not settings["model_predictions"]["enabled"]:
            continue
        base_key = ("pooled", "0", "shared_unexposed") if settings["shared_unexposed_baseline"] else (0, "0", "action_channel")
        active_key = (1, "mixture", "mixture_common_weight")
        base, active = summary[base_key], summary[active_key]
        status = _prediction_status(train, summary, diagnostic)
        evaluation_diagnostic = _diagnostics(evaluation, _summaries(evaluation, settings))
        output["blackboard_calibration_diagnostics"].append({**identity,
            **evaluation_diagnostic, "diagnostic_slice": "evaluation_group", "conditioning_json": "{}"})
        if status == "homogeneous_reference_assumed" and evaluation_diagnostic["rounds_with_varying_board"]:
            status = "evaluation_time_varying_board_reference_only"
        if any(r["U"] == 0 and r["E"] == 1 for r in evaluation):
            status = "silent_exposure_incompatible"
        dependencies = [estimate_ids[(*key, field)] for key in (base_key, active_key) for field in ("p_plus", "p_minus")]
        eval_rounds = {}
        for r in evaluation:
            eval_rounds[(r["episode_id"], r["round_index"])] = r
        validation_rows = []
        for key, r in sorted(eval_rounds.items()):
            supported = _present(r["N"]) and float(r["N"]).is_integer() and math.isfinite(r["x"])
            prediction = susceptibility(base, active, int(r["N"]), int(r["M"]), r["x"]) if supported else dict.fromkeys(("r0", "rb", "chi", "available", "intercept", "slope", "x_star"), NAN)
            # Never reinterpret exposed silence as K0. Suppress that model.
            if status in {"silent_exposure_incompatible", "missing_silent_baseline", "incomplete_calibration_records", "unsupported_training"}:
                prediction = {k: NAN for k in prediction}
            round_predictions = []
            for ds in draw_summaries:
                value = susceptibility(ds[base_key], ds[active_key], int(r["N"]), int(r["M"]), r["x"]) if supported else dict.fromkeys(prediction, NAN)
                round_predictions.append(value)
            for field, metric, branch in (
                ("r0", "blackboard_round_relaxation", "silent_unexposed"),
                ("rb", "blackboard_round_relaxation", "active"),
                ("chi", "blackboard_model_target_susceptibility", "contrast"),
                ("available", "blackboard_model_available_susceptibility", "contrast"),
                ("intercept", "blackboard_model_response_intercept", "contrast"),
                ("slope", "blackboard_model_response_slope", "contrast"),
                ("x_star", "blackboard_model_zero_response_share", "contrast"),
            ):
                output["blackboard_model_predictions"].append({**identity,
                    "episode_id": r["episode_id"], "round_index": r["round_index"], "block_id": r["block_id"],
                    "N": r["N"], "M": r["M"], "x": r["x"], "metric": metric, "branch": branch,
                    "estimate": prediction[field], "prediction_id": _id([training_id, key, field]),
                    "dependencies_json": _json(dependencies), "estimator_variant": "held_out_frozen_board_reference",
                    "conditioning_json": _json({"episode_id": r["episode_id"], "round_index": r["round_index"]}),
                    "units": "dimensionless", "confidence": confidence, "model_status": status,
                    "support_status": "supported" if math.isfinite(prediction[field]) else "undefined",
                    "root_status": prediction.get("root_status"), "n_blocks": n_blocks,
                    **_interval(prediction[field], [p[field] for p in round_predictions], n_blocks, confidence, bootstrap_resamples, prediction[field] in (0, 1)),
                })
            causal = causal_lookup.get((cell_id, r["episode_id"], r["round_index"]))
            empirical = causal["causal_response_h1"] if causal else NAN
            validation_rows.append({**identity, "episode_id": r["episode_id"], "round_index": r["round_index"],
                "block_id": r["block_id"], "x": r["x"], "M": r["M"], "N": r["N"], "U": r["U"],
                "predicted": prediction["chi"], "empirical": empirical,
                "residual": empirical-prediction["chi"], "model_status": status,
                "comparison_provenance": "propensity_weighted_causal_response_v1:lag1",
                "comparison_status": "held_out_round_contribution" if causal else "missing_eligible_causal_comparison",
                "weight": 1.0, "weight_unit": "round", "dependencies_json": _json(dependencies)})
        output["blackboard_model_validation"].extend(validation_rows)
        # Joint training and evaluation uncertainty, fixed split membership.
        matched = [r for r in validation_rows if math.isfinite(r["residual"])]
        from .causal_response import _support_status

        n_action = sum(r["U"] == 1 for r in matched)
        n_silence = sum(r["U"] == 0 for r in matched)
        comparison_support = _support_status(n_action, n_silence)
        residual_draws = []
        for ds in draw_summaries:
            sampled_eval = _resample(matched, rng)
            residuals = [r["empirical"] - susceptibility(ds[base_key], ds[active_key], int(r["N"]), int(r["M"]), r["x"])["chi"] for r in sampled_eval]
            residual_draws.append(float(np.mean(residuals)) if residuals and {r["U"] for r in sampled_eval} == {0, 1} else NAN)
        residual = float(np.mean([r["residual"] for r in matched])) if matched and comparison_support != "unsupported" else NAN
        output["blackboard_model_validation"].append({**identity,
            "comparison_status": "held_out_round_weighted_summary" if matched else "unavailable",
            "comparison_provenance": "propensity_weighted_causal_response_v1:lag1",
            "predicted": float(np.mean([r["predicted"] for r in matched])) if matched else NAN,
            "empirical": float(np.mean([r["empirical"] for r in matched])) if matched else NAN,
            "residual": residual, "n_observations": len(matched), "model_status": status,
            "support_status": comparison_support, "n_action": n_action, "n_silence": n_silence,
            "dependencies_json": _json(dependencies), "weight_unit": "round",
            **_interval(residual, residual_draws, min(n_blocks, len({r["block_id"] for r in matched})), confidence, bootstrap_resamples)})
    frames = {name: pd.DataFrame(rows) for name, rows in output.items()}
    for frame in frames.values():
        frame["estimator_version"] = VERSION
        frame["analysis_hash"] = analysis_hash
        frame["provisional"] = provisional
    return frames
