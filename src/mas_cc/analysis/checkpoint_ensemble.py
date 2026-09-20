"""Paired parent-checkpoint analysis using the repository's CMI engine."""

from __future__ import annotations

import json
import math
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from typing import Callable
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mas_cc.analysis.estimators import (
    conditional_mutual_information,
    conditional_mutual_information_from_counts,
)

VERSION = "blackboard_checkpoint_ensemble_v1"
CONTROLLED_POLICIES = (
    "always_truth",
    "always_false",
    "sensing_truth",
    "sensing_false",
)


def _qualified_parent_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Qualify game-local parent IDs by scientific cell identity."""

    result = frame.copy()
    if result.empty or "parent_id" not in result:
        return result
    scope = result.get("cell_key")
    if scope is None:
        scope = result.get("source_run_id", pd.Series("run", index=result.index))
    scope = scope.fillna(result.get("source_run_id", "run")).astype(str)
    result["source_parent_id"] = result["parent_id"].astype(str)
    result["parent_id"] = scope + "::" + result["source_parent_id"]
    return result


def _branch_key(row: Mapping[str, Any]) -> tuple[str, str, int, str, int]:
    budget = row.get("posting_budget")
    normalized_budget = "none" if budget is None or pd.isna(budget) else str(int(budget))
    return (
        str(row.get("parent_id")),
        str(row.get("branch_policy")),
        int(row.get("copy_id")),
        normalized_budget,
        int(row.get("post_branch_horizon")),
    )


def prepare_checkpoint_ensemble_inputs(
    rounds: pd.DataFrame,
    micro_slots: pd.DataFrame | None = None,
    *,
    available_round_prefixes: pd.DataFrame | None = None,
    available_micro_slot_prefixes: pd.DataFrame | None = None,
    continuation_rounds: int,
    expected_policies: Sequence[str] = (
        "none", "always_truth", "always_false", "sensing_truth", "sensing_false"
    ),
    posting_budgets: Sequence[int] = (3, 12),
    continuation_copies: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Recover complete checkpoint paths from completed and interrupted episodes.

    Generic episode completion may fail after a resumed parent bundle has already
    durably written its rich round and micro-slot trajectories. For an explicitly
    incomplete checkpoint analysis, a branch is scientifically usable when its
    final appended attempt contains every horizon ``1..M``. The last physical
    record at each coordinate supersedes an interrupted prefix. Branches with
    incomplete horizon coverage are excluded rather than filled with zeros.
    """

    round_frames = [
        frame
        for frame in (rounds, available_round_prefixes)
        if frame is not None and not frame.empty
    ]
    combined_rounds = (
        pd.concat(round_frames, ignore_index=True, sort=False)
        if round_frames
        else pd.DataFrame()
    )
    combined_rounds = _qualified_parent_ids(combined_rounds)
    combined_rounds["_checkpoint_input_order"] = np.arange(len(combined_rounds))
    policies = set(expected_policies)
    continuation = combined_rounds[
        combined_rounds.get(
            "branch_policy", pd.Series(index=combined_rounds.index, dtype=object)
        ).isin(policies)
    ].copy()
    coordinate_columns = [
        "parent_id", "branch_policy", "posting_budget", "copy_id",
        "post_branch_horizon",
    ]
    continuation = continuation.drop_duplicates(coordinate_columns, keep="last")
    expected_horizons = set(range(1, continuation_rounds + 1))
    complete_keys: set[tuple[str, str, int, str]] = set()
    for coordinates, group in continuation.groupby(
        ["parent_id", "branch_policy", "copy_id", "posting_budget"],
        dropna=False,
        sort=False,
    ):
        parent_id, policy, copy_id, budget = coordinates
        normalized_budget = "none" if pd.isna(budget) else str(int(budget))
        horizons = set(pd.to_numeric(group["post_branch_horizon"]).astype(int))
        if horizons == expected_horizons:
            complete_keys.add((str(parent_id), str(policy), int(copy_id), normalized_budget))

    def complete_branch(row: Mapping[str, Any]) -> bool:
        parent, policy, copy_id, budget, _ = _branch_key(row)
        return (parent, policy, copy_id, budget) in complete_keys

    continuation = continuation[
        continuation.apply(lambda row: complete_branch(row), axis=1)
    ]
    complete_parents = {key[0] for key in complete_keys}
    checkpoints = combined_rounds[
        combined_rounds.get(
            "branch_policy", pd.Series(index=combined_rounds.index, dtype=object)
        ).eq("checkpoint")
    ].copy()
    checkpoints = checkpoints[checkpoints["parent_id"].astype(str).isin(complete_parents)]
    checkpoints = checkpoints.drop_duplicates(
        ["parent_id", "copy_id", "post_branch_horizon"], keep="last"
    )
    recovered_rounds = pd.concat(
        [checkpoints, continuation], ignore_index=True, sort=False
    ).sort_values("_checkpoint_input_order")
    recovered_rounds = recovered_rounds.drop(columns=["_checkpoint_input_order"])

    micro_frames = [
        frame
        for frame in (micro_slots, available_micro_slot_prefixes)
        if frame is not None and not frame.empty
    ]
    combined_micro = (
        pd.concat(micro_frames, ignore_index=True, sort=False)
        if micro_frames
        else pd.DataFrame()
    )
    input_micro_records = len(combined_micro)
    combined_micro = _qualified_parent_ids(combined_micro)
    if not combined_micro.empty:
        combined_micro["_checkpoint_input_order"] = np.arange(len(combined_micro))
        combined_micro = combined_micro[
            combined_micro.get(
                "branch_policy", pd.Series(index=combined_micro.index, dtype=object)
            ).isin(policies)
        ].copy()
        micro_coordinates = [
            "parent_id", "branch_policy", "posting_budget", "copy_id",
            "post_branch_horizon", "round_index", "micro_slot_index",
        ]
        combined_micro = combined_micro.drop_duplicates(micro_coordinates, keep="last")
        combined_micro = combined_micro[
            combined_micro.apply(lambda row: complete_branch(row), axis=1)
        ].sort_values("_checkpoint_input_order")
        combined_micro = combined_micro.drop(columns=["_checkpoint_input_order"])

    expected_branch_keys = {
        (policy, "none" if policy == "none" else str(int(budget)), copy_id)
        for policy in expected_policies
        for budget in ((None,) if policy == "none" else posting_budgets)
        for copy_id in range(1, continuation_copies + 1)
    }
    paths_by_parent: dict[str, set[tuple[str, str, int]]] = {}
    for parent, policy, copy_id, budget in complete_keys:
        paths_by_parent.setdefault(parent, set()).add((policy, budget, copy_id))
    fully_covered = sum(paths == expected_branch_keys for paths in paths_by_parent.values())
    diagnostics = {
        "input_round_records": int(len(combined_rounds)),
        "retained_round_records": int(len(recovered_rounds)),
        "input_micro_slot_records": int(input_micro_records),
        "retained_micro_slot_records": int(len(combined_micro)),
        "parents_observed": int(len(paths_by_parent)),
        "parents_with_complete_branch_coverage": int(fully_covered),
        "complete_paths": int(len(complete_keys)),
        "expected_paths_for_observed_parents": int(
            len(paths_by_parent) * len(expected_branch_keys)
        ),
        "excluded_incomplete_paths": int(
            len(paths_by_parent) * len(expected_branch_keys) - len(complete_keys)
        ),
        "selection_rule": "last_record_per_coordinate_and_complete_horizons_1_to_M",
    }
    return (
        recovered_rounds.reset_index(drop=True),
        combined_micro.reset_index(drop=True),
        diagnostics,
    )

BRANCH_ROUND_STATISTICS = (
    "round_target_actuation_cmi",
    "round_target_information_fraction",
    "round_target_susceptibility",
    "round_sensor_mae",
    "round_sensor_mse",
    "round_controller_action_entropy",
)


def _object(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return None
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _count(row: Mapping[str, Any], answer: str) -> int:
    counts = _object(row.get("option_counts"))
    if isinstance(counts, Mapping):
        return int(counts.get(answer, 0))
    counts = _object(row.get("occupation_counts_after"))
    answers = _object(row.get("possible_answers"))
    if isinstance(counts, Mapping):
        return int(counts.get(answer, 0))
    if isinstance(counts, Sequence) and isinstance(answers, Sequence):
        return int(counts[list(answers).index(answer)])
    raise ValueError("checkpoint round lacks a semantic option-count vector")


def _count_before(row: Mapping[str, Any], answer: str) -> int:
    counts = _object(row.get("occupation_counts_before"))
    answers = _object(row.get("possible_answers"))
    if isinstance(counts, Mapping):
        return int(counts.get(answer, 0))
    if isinstance(counts, Sequence) and isinstance(answers, Sequence):
        return int(counts[list(answers).index(answer)])
    if int(row.get("post_branch_horizon", -1)) == 0:
        return _count(row, answer)
    raise ValueError("checkpoint round lacks a semantic pre-round option-count vector")


def validate_checkpoint_ensemble(
    rounds: pd.DataFrame,
    *,
    expected_policies: Sequence[str],
    posting_budgets: Sequence[int],
    continuation_copies: int,
    continuation_rounds: int,
    expected_parents: int | None = None,
    micro_slots: pd.DataFrame | None = None,
    episodes: pd.DataFrame | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Strict parent/branch trajectory validation independent of scheduler shards."""

    required = {
        "parent_id", "checkpoint_id", "checkpoint_hash", "branch_policy",
        "copy_id", "absolute_round", "post_branch_horizon", "q", "rho", "N",
        "L", "M", "correct_answer_semantic_id", "preparation_seed",
        "continuation_seed", "continuation_stream_identity", "branch_status",
        "possible_answers", "occupation_counts_before", "occupation_counts_after",
        "controller_sampled_U", "actual_controller_posts",
        "total_eligible_board_message_reads", "controller_unique_readers",
        "retry_attempts_this_round",
    }
    missing_columns = sorted(required - set(rounds.columns))
    errors: list[str] = []
    warnings: list[str] = []
    if missing_columns:
        errors.append("missing checkpoint columns: " + ", ".join(missing_columns))
        return {"complete": False, "errors": errors, "warnings": warnings}
    continuation_policies = set(expected_policies)
    h0 = rounds[rounds["branch_policy"] == "checkpoint"].copy()
    data = rounds[rounds["branch_policy"].isin(continuation_policies)].copy()
    if data.empty:
        errors.append("no continuation rows were found")
        return {"complete": False, "errors": errors, "warnings": warnings}
    data["copy_id"] = pd.to_numeric(data["copy_id"], errors="coerce")
    data["post_branch_horizon"] = pd.to_numeric(
        data["post_branch_horizon"], errors="coerce"
    )
    expected_controlled = {
        (policy, int(budget), copy_id)
        for policy in expected_policies if policy != "none"
        for budget in posting_budgets
        for copy_id in range(1, continuation_copies + 1)
    }
    expected_none = {("none", None, copy_id) for copy_id in range(1, continuation_copies + 1)}
    found_parents = int(data["parent_id"].nunique())
    if expected_parents is not None and found_parents != expected_parents:
        errors.append(
            f"found {found_parents} parents; expected {expected_parents}"
        )
    parent_rows = []
    for parent_id, parent in data.groupby("parent_id", sort=True):
        hashes = set(parent["checkpoint_hash"].dropna().astype(str))
        checkpoints = set(parent["checkpoint_id"].dropna().astype(str))
        if len(hashes) != 1 or len(checkpoints) != 1:
            errors.append(f"parent {parent_id} does not have one shared checkpoint")
        branch_keys: set[tuple[str, int | None, int]] = set()
        duplicate_trajectories = 0
        for (policy, budget, copy_id), branch in parent.groupby(
            ["branch_policy", "posting_budget", "copy_id"], dropna=False
        ):
            normalized_budget = None if str(policy) == "none" else int(budget)
            key = (str(policy), normalized_budget, int(copy_id))
            branch_keys.add(key)
            horizons = sorted(
                int(value) for value in branch["post_branch_horizon"].dropna().unique()
            )
            if horizons != list(range(1, continuation_rounds + 1)):
                errors.append(
                    f"parent {parent_id} branch {key} has horizons {horizons}; "
                    f"expected 1..{continuation_rounds}"
                )
            duplicate_trajectories += int(
                branch.duplicated(["post_branch_horizon"]).sum()
            )
        expected = expected_controlled | expected_none
        if branch_keys != expected:
            errors.append(
                f"parent {parent_id} branch coverage differs: missing="
                f"{sorted(expected - branch_keys, key=str)} extra="
                f"{sorted(branch_keys - expected, key=str)}"
            )
        if duplicate_trajectories:
            errors.append(f"parent {parent_id} has duplicate branch horizons")
        streams = parent.drop_duplicates(
            ["branch_policy", "posting_budget", "copy_id"]
        )["continuation_stream_identity"]
        if streams.map(lambda value: json.dumps(_object(value), sort_keys=True)).duplicated().any():
            errors.append(f"parent {parent_id} has duplicate continuation streams")
        none = parent[parent["branch_policy"] == "none"]
        for copy_id in range(1, continuation_copies + 1):
            if none[none["copy_id"] == copy_id]["posting_budget"].notna().any():
                errors.append(f"parent {parent_id} none branch has a posting budget")
        if not parent["branch_status"].isin({"complete", "in_progress"}).all():
            errors.append(f"parent {parent_id} contains failed branch status")
        truth = set(parent["correct_answer_semantic_id"].dropna().astype(str))
        if len(truth) != 1:
            errors.append(f"parent {parent_id} has inconsistent semantic truth targets")
        for policy, branch in parent.groupby("branch_policy"):
            targets = set(branch["controller_target_semantic_id"].dropna().astype(str))
            if policy == "none" and targets:
                errors.append(f"parent {parent_id} none branch has a controller target")
            elif policy != "none" and len(targets) != 1:
                errors.append(f"parent {parent_id} policy {policy} has inconsistent targets")
            elif policy.endswith("truth") and targets != truth:
                errors.append(f"parent {parent_id} policy {policy} does not target truth")
            elif policy.endswith("false") and targets & truth:
                errors.append(f"parent {parent_id} policy {policy} targets truth")
        parent_h0 = h0[h0["parent_id"] == parent_id]
        h0_copies = set(pd.to_numeric(parent_h0["copy_id"], errors="coerce").dropna().astype(int))
        if h0_copies != set(range(1, continuation_copies + 1)):
            errors.append(f"parent {parent_id} lacks one shared h=0 row per copy")
        if not parent_h0.empty and set(parent_h0["checkpoint_hash"].astype(str)) != hashes:
            errors.append(f"parent {parent_id} h=0 checkpoint hash differs from siblings")
        parent_rows.append({"parent_id": parent_id, "checkpoint_hash": next(iter(hashes), None)})
    if require_complete and errors:
        complete = False
    else:
        complete = not errors
        if errors:
            warnings.extend(errors)
    if micro_slots is not None:
        required_micro = {
            "parent_id", "checkpoint_hash", "branch_policy", "copy_id",
            "post_branch_horizon", "within_round_index", "sampled_message_ids",
            "controller_message_posted", "occupation_counts_before",
            "occupation_counts_after",
        }
        missing_micro = sorted(required_micro - set(micro_slots.columns))
        if missing_micro:
            errors.append("missing micro-slot fields: " + ", ".join(missing_micro))
    if episodes is not None:
        required_episode = {
            "status", "usage_requests", "usage_input_tokens", "usage_output_tokens",
            "started_at", "finished_at",
        }
        missing_episode = sorted(required_episode - set(episodes.columns))
        if missing_episode:
            errors.append("missing episode resource fields: " + ", ".join(missing_episode))
        failed = episodes[~episodes["status"].isin({"completed", "skipped_resumed"})]
        if not failed.empty:
            errors.append(f"found {len(failed)} failed or interrupted parent episodes")
    complete = not errors
    if not require_complete and errors:
        warnings.extend(errors)
        errors = []
        complete = False
    return {
        "version": VERSION,
        "complete": complete,
        "errors": errors if require_complete else [],
        "warnings": warnings,
        "counts": {
            "parents": int(data["parent_id"].nunique()),
            "checkpoints": int(data["checkpoint_hash"].nunique()),
            "branches": int(
                data[["parent_id", "branch_policy", "posting_budget", "copy_id"]]
                .drop_duplicates().shape[0]
            ),
            "round_rows": int(len(data)),
        },
    }


def resource_report(rounds: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Observed actuation/channel use plus parent-level provider resources."""

    data = rounds[rounds["branch_policy"].isin(CONTROLLED_POLICIES)].copy()
    group_keys = ["q", "rho", "branch_policy", "posting_budget"]
    metrics = {
        "controller_sampled_U": "activations",
        "actual_controller_posts": "controller_posts",
        "total_eligible_board_message_reads": "eligible_message_reads",
        "controller_unique_readers": "unique_controller_readers",
        "retry_attempts_this_round": "retry_attempts",
    }
    present = {source: target for source, target in metrics.items() if source in data}
    if data.empty:
        grouped = pd.DataFrame(columns=group_keys)
    else:
        grouped = data.groupby(group_keys, dropna=False, as_index=False).agg(
            **{target: (source, "sum") for source, target in present.items()},
            observed_rounds=("post_branch_horizon", "count"),
        )
    started = pd.to_datetime(episodes.get("started_at"), errors="coerce", utc=True)
    finished = pd.to_datetime(episodes.get("finished_at"), errors="coerce", utc=True)
    latency = (finished - started).dt.total_seconds()
    totals = {
        "parent_episodes": int(len(episodes)),
        "provider_requests": float(pd.to_numeric(episodes.get("usage_requests"), errors="coerce").sum()),
        "input_tokens": float(pd.to_numeric(episodes.get("usage_input_tokens"), errors="coerce").sum()),
        "output_tokens": float(pd.to_numeric(episodes.get("usage_output_tokens"), errors="coerce").sum()),
        "latency_seconds": float(latency.sum(min_count=1)),
        "currency_cost": math.nan,
        "currency_cost_status": "unavailable_without_authoritative_quote",
    }
    for key, value in totals.items():
        grouped[key] = value
    return grouped


def sensing_activation_response(
    rounds: pd.DataFrame,
    *,
    bootstrap_resamples: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    """Optional randomized-gate lag-one response for sensing branches only."""

    sensing = rounds[
        rounds["branch_policy"].isin(("sensing_truth", "sensing_false"))
    ].copy()
    keys = ["q", "rho", "branch_policy", "posting_budget"]
    rows: list[dict[str, Any]] = []
    for group_index, (coordinates, group) in enumerate(
        sensing.groupby(keys, dropna=False, sort=True)
    ):
        contributions: list[dict[str, Any]] = []
        for record in group.to_dict(orient="records"):
            probability = record.get(
                "controller_probability_U1_given_Y",
                record.get(
                    "controller_advocate_probability",
                    record.get("round_controller_advocate_probability"),
                ),
            )
            action = record.get("controller_sampled_U")
            target = record.get("controller_target_semantic_id")
            if (
                probability is None
                or action is None
                or target is None
                or pd.isna(probability)
                or pd.isna(action)
                or pd.isna(target)
            ):
                continue
            propensity = float(probability)
            if not 0.0 < propensity < 1.0:
                continue
            activated = int(action)
            if activated not in (0, 1):
                continue
            population = int(record["N"])
            change = (
                _count(record, str(target)) - _count_before(record, str(target))
            ) / population
            contribution = (
                activated / propensity
                - (1 - activated) / (1 - propensity)
            ) * change
            contributions.append(
                {
                    "parent_id": str(record["parent_id"]),
                    "contribution": contribution,
                }
            )
        frame = pd.DataFrame(contributions)
        estimate = (
            float(frame["contribution"].mean()) if not frame.empty else math.nan
        )
        ci_low = ci_high = math.nan
        parents = (
            frame["parent_id"].drop_duplicates().to_numpy()
            if not frame.empty
            else np.array([])
        )
        if len(parents) and bootstrap_resamples > 0:
            # Per-parent contribution arrays in frame order; a draw is the
            # concatenation over the sampled parents, exactly the list the
            # per-parent ``.loc`` lookups used to build, then ``np.mean``.
            contribution = frame["contribution"].to_numpy(dtype=float)
            parent_column = frame["parent_id"].to_numpy()
            per_parent = {
                parent: contribution[parent_column == parent] for parent in parents
            }
            rng = np.random.default_rng(seed + group_index)
            draws = []
            for _ in range(bootstrap_resamples):
                sampled = rng.choice(parents, size=len(parents), replace=True)
                values = np.concatenate([per_parent[parent] for parent in sampled])
                draws.append(float(np.mean(values)))
            alpha = (1.0 - confidence) / 2.0
            ci_low = float(np.quantile(draws, alpha))
            ci_high = float(np.quantile(draws, 1.0 - alpha))
        rows.append(
            {
                **dict(zip(keys, coordinates, strict=True)),
                "metric": "tau_hat_1",
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "units": "target_fraction_per_round",
                "eligible_rounds": int(len(frame)),
                "effective_K": int(len(parents)),
                "bootstrap_unit": "parent",
                "support_status": (
                    "supported" if len(frame) else "unsupported_no_eligible_rounds"
                ),
            }
        )
    return pd.DataFrame(rows)


def _branch_group_task(payload: tuple[Any, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One branch group of ``branch_round_metrics``: process-safe, order-independent."""
    (group_index, coordinates, records, keys, bootstrap_resamples, null_permutations,
     confidence, seed) = payload
    from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
        ADVOCATE_TARGET,
        NO_OP,
        round_information_analysis,
    )
    from mas_cc.games.relational_reasoning.imitation_round_feedback.analysis import (
        adapt_relational_round_record,
    )

    metric_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    if True:
        base = dict(zip(keys, coordinates, strict=True))
        branch_cell = "|".join(
            str(base[key]) for key in ("q", "rho", "branch_policy", "posting_budget")
        )
        events = [
            adapt_relational_round_record(
                {key: _object(value) for key, value in row.items()},
                cell_id=branch_cell,
                episode_id=str(row["parent_id"]),
            )
            for row in records
        ]
        n_parents = len({str(row["parent_id"]) for row in records})
        estimates, nulls = round_information_analysis(
            events,
            statistics=BRANCH_ROUND_STATISTICS,
            bootstrap_resamples=bootstrap_resamples,
            null_permutations=null_permutations,
            confidence=confidence,
            seed=seed + group_index,
        )
        by_name = {str(item["statistic"]): item for item in estimates}
        aliases = {
            "round_target_actuation_cmi": "T_pi",
            "round_target_information_fraction": "eta_IF",
            "round_target_susceptibility": "chi",
            "round_sensor_mae": "sensor_MAE",
            "round_sensor_mse": "sensor_MSE",
            "round_controller_action_entropy": "controller_action_entropy",
        }
        for name, alias in aliases.items():
            item = by_name.get(name)
            if item is None:
                metric_rows.append(
                    {
                        **base,
                        "metric": alias,
                        "source_metric": name,
                        "estimate": math.nan,
                        "support_status": "unavailable",
                        "n_observations": len(events),
                        "n_parents": n_parents,
                    }
                )
                continue
            metric_rows.append(
                {
                    **base,
                    "metric": alias,
                    "source_metric": name,
                    "estimate": item.get("estimate"),
                    "ci_low": item.get("ci_low"),
                    "ci_high": item.get("ci_high"),
                    "support_status": item.get("support_status"),
                    "estimator_variant": item.get("main_estimator_variant"),
                    "n_observations": item.get("n_rounds", len(events)),
                    "n_parents": n_parents,
                    "action_entropy_ceiling_bits": item.get(
                        "conditional_action_entropy_bits"
                    ),
                }
            )
        actions = [event.U_k for event in events if event.U_k in {ADVOCATE_TARGET, NO_OP}]
        activation = (
            sum(action == ADVOCATE_TARGET for action in actions) / len(actions)
            if actions
            else math.nan
        )
        transfer = by_name.get("round_target_actuation_cmi", {}).get("estimate")
        susceptibility = by_name.get("round_target_susceptibility", {}).get("estimate")
        eta_ir = math.nan
        if (
            transfer is not None
            and susceptibility is not None
            and math.isfinite(float(transfer))
            and float(transfer) > 0
            and math.isfinite(float(susceptibility))
        ):
            eta_ir = (
                2
                * activation
                * (1 - activation)
                * float(susceptibility) ** 2
                / (math.log(2) * float(transfer))
            )
        metric_rows.append(
            {
                **base,
                "metric": "eta_IR",
                "source_metric": "eta_ir",
                "estimate": eta_ir,
                "support_status": (
                    "supported"
                    if math.isfinite(eta_ir)
                    else "unsupported_constant_action_or_zero_information"
                ),
                "activation_frequency": activation,
                "n_observations": len(events),
                "n_parents": n_parents,
            }
        )
        null_rows.extend({**base, **item} for item in nulls)
    return metric_rows, null_rows


def branch_round_metrics(
    rounds: pd.DataFrame,
    *,
    bootstrap_resamples: int,
    null_permutations: int,
    confidence: float,
    seed: int,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Traditional round metrics estimated separately inside each branch.

    This deliberately synthesizes one estimator cell per policy/budget rather
    than allowing the generic cell-level analysis to pool mutually exclusive
    continuation policies. The authoritative round-information engine remains
    responsible for CMI, susceptibility, support, bootstrap, and nulls.

    Groups are independent (each seeds its own engine from ``seed + index``),
    so with ``workers > 1`` they run in an order-preserving spawn pool and the
    tables come back in the same order as the serial loop.
    """

    continuation = rounds[
        rounds["branch_policy"].isin(("none", *CONTROLLED_POLICIES))
    ].copy()
    keys = ["q", "rho", "branch_policy", "posting_budget"]
    tasks = [
        (group_index, coordinates, group.to_dict(orient="records"), keys,
         bootstrap_resamples, null_permutations, confidence, seed)
        for group_index, (coordinates, group) in enumerate(
            continuation.groupby(keys, dropna=False, sort=True)
        )
    ]
    if workers > 1 and len(tasks) > 1:
        from multiprocessing import get_context

        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), mp_context=get_context("spawn")
        ) as pool:
            results = list(pool.map(_branch_group_task, tasks))
    else:
        results = [_branch_group_task(task) for task in tasks]
    metric_rows = [row for metrics, _ in results for row in metrics]
    null_rows = [row for _, nulls in results for row in nulls]
    return pd.DataFrame(metric_rows), pd.DataFrame(null_rows)


def endpoint_table(rounds: pd.DataFrame) -> pd.DataFrame:
    """One semantic endpoint row per parent, branch, copy, target, and horizon."""

    rows: list[dict[str, Any]] = []
    continuation = rounds[
        rounds.get("branch_policy", pd.Series(index=rounds.index, dtype=object)).isin(
            ("none", *CONTROLLED_POLICIES)
        )
    ]
    for row in continuation.to_dict(orient="records"):
        correct = str(row.get("correct_answer_semantic_id") or row.get("correct_answer"))
        controller = row.get("controller_target_semantic_id")
        controller_present = controller is not None and not pd.isna(controller)
        targets = [("truth", correct)]
        if controller_present and str(controller) != correct:
            targets.append(("false_target", str(controller)))
        elif (
            row.get("false_target_semantic_id") is not None
            and not pd.isna(row.get("false_target_semantic_id"))
        ):
            targets.append(("false_target", str(row["false_target_semantic_id"])))
        for target_semantics, answer in targets:
            rows.append({
                **{key: row.get(key) for key in (
                    "parent_id", "checkpoint_id", "checkpoint_hash", "branch_policy",
                    "posting_budget", "copy_id", "q", "rho", "N", "L", "M",
                    "post_branch_horizon", "absolute_round",
                )},
                "target_semantics": target_semantics,
                "target_answer": answer,
                "n_0": _count_before(row, answer) if int(row.get("post_branch_horizon", 0)) == 1 else None,
                "target_count": _count(row, answer),
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["target_fraction"] = pd.to_numeric(result["target_count"]) / pd.to_numeric(result["N"])
    result["n_0"] = result.groupby(
        ["parent_id", "copy_id", "target_semantics"], dropna=False
    )["n_0"].transform(lambda values: values.dropna().iloc[0] if values.notna().any() else np.nan)
    return result


def _parent_bootstrap(
    values: pd.DataFrame, *, resamples: int, confidence: float, seed: int
) -> tuple[float, float]:
    """Whole-parent bootstrap of the mean paired difference.

    ``values`` holds one row per parent (the caller's groupby mean), so each
    drawn parent contributes its own value and a draw is ``np.mean`` over the
    sampled values in sampled order. The sampling call is the same
    ``rng.choice`` over the parent array as before (same RNG stream); the
    per-parent lookups are an index gather instead of a pandas mask per parent.
    """
    if values.empty or resamples <= 0:
        return math.nan, math.nan
    parents = values["parent_id"].drop_duplicates().to_numpy()
    position = {parent: index for index, parent in enumerate(parents)}
    per_parent = np.array([
        float(values.loc[values["parent_id"] == parent, "paired_difference"].mean())
        for parent in parents
    ])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        sampled = rng.choice(parents, size=len(parents), replace=True)
        draws.append(float(np.mean(per_parent[[position[parent] for parent in sampled]])))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha))


def paired_response(
    endpoints: pd.DataFrame,
    *,
    horizons: Sequence[int] = (1, 10),
    bootstrap_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Controlled-minus-shared-none effects with whole-parent resampling."""

    data = endpoints[endpoints["post_branch_horizon"].isin(horizons)].copy()
    baseline = data[data["branch_policy"] == "none"].rename(
        columns={"target_count": "baseline_count", "target_fraction": "baseline_fraction"}
    )
    controlled = data[data["branch_policy"].isin(CONTROLLED_POLICIES)].copy()
    keys = ["parent_id", "copy_id", "q", "rho", "target_semantics", "post_branch_horizon"]
    paired = controlled.merge(
        baseline[keys + ["baseline_count", "baseline_fraction"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    paired["paired_difference"] = paired["target_fraction"] - paired["baseline_fraction"]
    group_keys = [
        "q", "rho", "branch_policy", "posting_budget", "target_semantics",
        "post_branch_horizon",
    ]
    summaries = []
    for index, (coordinates, group) in enumerate(paired.groupby(group_keys, dropna=False, sort=True)):
        parent_values = group.groupby("parent_id", as_index=False).agg(
            paired_difference=("paired_difference", "mean")
        )
        low, high = _parent_bootstrap(
            parent_values,
            resamples=bootstrap_resamples,
            confidence=confidence,
            seed=seed + index,
        )
        estimate = float(parent_values["paired_difference"].mean())
        summaries.append({
            **dict(zip(group_keys, coordinates, strict=True)),
            "estimate_fraction": estimate,
            "estimate_percentage_points": 100.0 * estimate,
            "ci_low": low,
            "ci_high": high,
            "effective_K": int(parent_values["parent_id"].nunique()),
            "missing_pairs": int(group["baseline_count"].isna().sum()),
            "bootstrap_unit": "parent",
        })
    return paired, pd.DataFrame(summaries)


def assigned_policy_information(
    paired: pd.DataFrame, *, population_size: int, smoothing: Sequence[float] = (0, 1, 12.5)
) -> pd.DataFrame:
    """Balanced A-vs-none CMI adapted to the established direct-count engine."""

    rows = []
    group_keys = [
        "q", "rho", "branch_policy", "posting_budget", "target_semantics",
        "post_branch_horizon",
    ]
    for coordinates, group in paired.dropna(subset=["baseline_count"]).groupby(group_keys, dropna=False):
        labels = [0] * len(group) + [1] * len(group)
        outcomes = group["baseline_count"].astype(int).tolist() + group["target_count"].astype(int).tolist()
        # The checkpoint count is exactly the baseline branch's h=0 count,
        # retained in the endpoint input as n_0 by the canonical builder.
        starts = group["n_0"].astype(int).tolist() * 2 if "n_0" in group else [0] * (2 * len(group))
        starting_support = group["n_0"].astype(int).value_counts()
        support = {
            "n_parents": int(group["parent_id"].nunique()),
            "n_balanced_observations": int(2 * len(group)),
            "label_0_count": int(len(group)),
            "label_1_count": int(len(group)),
            "occupied_starting_states": int(len(starting_support)),
            "singleton_starting_states": int((starting_support == 1).sum()),
            "singleton_starting_state_fraction": float(
                (starting_support == 1).mean()
            ),
            "minimum_starting_state_support": int(starting_support.min()),
        }
        estimate = conditional_mutual_information(
            labels, outcomes, starts,
            x_levels=(0, 1), y_levels=range(population_size + 1),
        )
        base = dict(zip(group_keys, coordinates, strict=True))
        rows.extend([
            {**base, **support, "estimator": "frequency_unsmoothed", "estimate_bits": estimate.unsmoothed},
            {**base, **support, "estimator": "frequency_jeffreys_engine", "estimate_bits": estimate.jeffreys},
            {**base, **support, "estimator": "frequency_miller_madow", "estimate_bits": estimate.miller_madow},
        ])
        z_levels = sorted(set(starts))
        zi = {value: index for index, value in enumerate(z_levels)}
        counts = np.zeros((2, len(z_levels), population_size + 1), dtype=float)
        for a, n, m in zip(labels, starts, outcomes, strict=True):
            counts[a, zi[n], m] += 1
        for lam in smoothing:
            if float(lam) == 0:
                continue
            smoothed = counts + float(lam) / (population_size + 1)
            value = conditional_mutual_information_from_counts(smoothed).unsmoothed
            rows.append({
                **base,
                **support,
                "estimator": f"uniform_row_smoothing_lambda_{lam:g}",
                "estimate_bits": value,
            })
    return pd.DataFrame(rows)


def _features(n: np.ndarray, m: np.ndarray, population_size: int) -> np.ndarray:
    x = n / population_size
    delta = (m - n) / population_size
    return np.column_stack((x, delta, x * delta, delta**2))


def _folds(groups: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    assignment = {group: index % folds for index, group in enumerate(shuffled)}
    return [
        (np.flatnonzero([assignment[g] != fold for g in groups]),
         np.flatnonzero([assignment[g] == fold for g in groups]))
        for fold in range(folds)
    ]


def cross_fitted_classifier_score(
    paired_group: pd.DataFrame,
    *,
    population_size: int,
    seed: int = 1,
    penalties: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    outer_folds: int = 5,
) -> Mapping[str, Any]:
    """Grouped nested-CV predictive-information lower-bound score."""

    group = paired_group.dropna(subset=["baseline_count"]).copy()
    parents = group["parent_id"].astype(str).to_numpy()
    labels = np.tile(np.array([0, 1]), len(group))
    starts0 = group.get("n_0", pd.Series(0, index=group.index)).astype(float).to_numpy()
    starts = np.repeat(starts0, 2)
    outcomes = np.column_stack((group["baseline_count"], group["target_count"])).reshape(-1).astype(float)
    row_groups = np.repeat(parents, 2)
    X = _features(starts, outcomes, population_size)
    folds = min(outer_folds, len(np.unique(row_groups)))
    if folds < 2:
        return {"estimate_bits": math.nan, "held_out_log_loss": math.nan, "candidate": "constant_1_2"}
    probabilities = np.full(len(labels), 0.5)
    chosen: list[str] = []
    for fold_index, (train, test) in enumerate(_folds(row_groups, folds, seed)):
        candidates: list[tuple[float, str, float | None]] = [(math.log(2), "constant_1_2", None)]
        inner_groups = row_groups[train]
        inner_folds = min(3, len(np.unique(inner_groups)))
        if inner_folds >= 2:
            for penalty in penalties:
                losses = []
                for inner_train, inner_test in _folds(inner_groups, inner_folds, seed + fold_index + 101):
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=float(penalty), max_iter=2000, random_state=seed),
                    )
                    model.fit(X[train][inner_train], labels[train][inner_train])
                    losses.append(log_loss(labels[train][inner_test], model.predict_proba(X[train][inner_test])[:, 1], labels=[0, 1]))
                candidates.append((float(np.mean(losses)), f"logistic_C_{penalty:g}", float(penalty)))
        _, name, penalty = min(candidates, key=lambda item: item[0])
        chosen.append(name)
        if penalty is not None:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=penalty, max_iter=2000, random_state=seed + fold_index),
            )
            model.fit(X[train], labels[train])
            probabilities[test] = model.predict_proba(X[test])[:, 1]
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    true_probability = np.where(labels == 1, clipped, 1 - clipped)
    return {
        "estimate_bits": float(np.mean(np.log2(true_probability / 0.5))),
        "held_out_log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "held_out_brier": float(brier_score_loss(labels, clipped)),
        "outer_folds": folds,
        "feature_definition": "n0/N, delta/N, interaction, delta_squared",
        "penalty_grid": list(penalties),
        "chosen_candidates": chosen,
        "probability_clip": [1e-6, 1 - 1e-6],
        "calibration": "none",
        "parents": int(len(np.unique(row_groups))),
    }


def _classifier_permutation_refit(task: tuple[Any, ...]) -> Mapping[str, Any]:
    """One deterministic paired swap/refit, with no nested BLAS parallelism."""
    from threadpoolctl import threadpool_limits

    group, mask, population_size, seed = task
    swapped = group.copy()
    for index, row in swapped.iterrows():
        if mask[row["parent_id"]]:
            swapped.at[index, "baseline_count"], swapped.at[index, "target_count"] = (
                row["target_count"], row["baseline_count"]
            )
    with threadpool_limits(limits=1):
        return cross_fitted_classifier_score(
            swapped, population_size=population_size, seed=seed
        )


def classifier_analysis(
    paired: pd.DataFrame,
    *,
    population_size: int,
    seed: int,
    repeated_splits: int = 5,
    label_swap_permutations: int = 100,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if workers < 1:
        raise ValueError("classifier workers must be positive")
    keys = ["q", "rho", "branch_policy", "posting_budget", "target_semantics", "post_branch_horizon"]
    estimates, nulls = [], []
    rng = np.random.default_rng(seed)
    grouped = paired.groupby(keys, dropna=False)
    total_comparisons = len(grouped)
    total_refits = total_comparisons * (repeated_splits + label_swap_permutations)
    completed_refits = 0
    started = time.monotonic()

    def report(comparison: int, base: Mapping[str, Any], repeat: int, permutation: int, phase: str) -> None:
        if progress_callback is None:
            return
        elapsed = time.monotonic() - started
        progress_callback({
            "stage": "checkpoint_classifier",
            "classifier_phase": phase,
            "classifier_workers": min(workers, max(1, label_swap_permutations)),
            "comparison_index": comparison + 1,
            "total_comparisons": total_comparisons,
            "completed_comparisons": comparison + int(phase == "comparison_complete"),
            "active_comparison": dict(base),
            "completed_split_repeats": repeat,
            "total_split_repeats": repeated_splits,
            "completed_permutations": permutation,
            "total_permutations": label_swap_permutations,
            "completed_classifier_refits": completed_refits,
            "total_classifier_refits": total_refits,
            "classifier_elapsed_seconds": elapsed,
            "classifier_eta_seconds": (
                elapsed / completed_refits * (total_refits - completed_refits)
                if completed_refits else None
            ),
        })

    for group_index, (coordinates, group) in enumerate(grouped):
        base = dict(zip(keys, coordinates, strict=True))
        report(group_index, base, 0, 0, "split_sensitivity")
        for repeat in range(repeated_splits):
            result = cross_fitted_classifier_score(
                group, population_size=population_size, seed=seed + group_index * 100 + repeat
            )
            estimates.append({**base, "split_repeat": repeat, **result})
            completed_refits += 1
            report(group_index, base, repeat + 1, 0, "split_sensitivity")
        report(group_index, base, repeated_splits, 0, "paired_label_swap")
        tasks = []
        for permutation in range(label_swap_permutations):
            # Draw masks in the parent process in the original serial order.
            # Scheduling and worker count must never alter scientific randomness.
            mask = {parent: bool(rng.integers(0, 2)) for parent in group["parent_id"].unique()}
            tasks.append((group, mask, population_size, seed + permutation + 10000))
        pool_context = (
            ProcessPoolExecutor(
                max_workers=min(workers, label_swap_permutations),
                mp_context=multiprocessing.get_context("spawn"),
            )
            if workers > 1 and label_swap_permutations > 0 else nullcontext(None)
        )
        with pool_context as pool:
            results = (
                pool.map(_classifier_permutation_refit, tasks, chunksize=1)
                if pool is not None else map(_classifier_permutation_refit, tasks)
            )
            for permutation, result in enumerate(results):
                nulls.append({**base, "permutation": permutation, "estimate_bits": result["estimate_bits"]})
                completed_refits += 1
                report(group_index, base, repeated_splits, permutation + 1, "paired_label_swap")
        report(group_index, base, repeated_splits, label_swap_permutations, "comparison_complete")
    return pd.DataFrame(estimates), pd.DataFrame(nulls)


def render_checkpoint_plots(
    effects: pd.DataFrame,
    trajectories: pd.DataFrame,
    information: pd.DataFrame,
    output_dir: str | Path,
    resources: pd.DataFrame | None = None,
) -> list[Path]:
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame, x, y in (
        ("paired_effects", effects, "branch_policy", "estimate_percentage_points"),
        ("trajectories", trajectories, "post_branch_horizon", "target_fraction"),
        ("assigned_policy_information", information, "estimator", "estimate_bits"),
        (
            "observed_controller_resources",
            pd.DataFrame() if resources is None else resources,
            "branch_policy",
            "controller_posts",
        ),
    ):
        if frame.empty or x not in frame or y not in frame:
            continue
        figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for label, group in frame.groupby([column for column in ("q", "rho") if column in frame], dropna=False):
            summary = group.groupby(x, as_index=False)[y].mean()
            axis.plot(summary[x].astype(str), summary[y], marker="o", label=str(label))
        axis.set(xlabel=x, ylabel=y, title=name.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=30)
        if axis.lines:
            axis.legend()
        path = root / f"checkpoint_{name}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths


__all__ = [
    "VERSION", "assigned_policy_information", "classifier_analysis",
    "branch_round_metrics", "cross_fitted_classifier_score", "endpoint_table", "paired_response",
    "prepare_checkpoint_ensemble_inputs", "render_checkpoint_plots", "resource_report", "sensing_activation_response",
    "validate_checkpoint_ensemble",
]
