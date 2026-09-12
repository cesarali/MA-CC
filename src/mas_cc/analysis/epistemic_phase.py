"""Offline symbolic epistemic-state and finite-horizon regime analysis.

This module turns exact active-fact snapshots into symbolic solvability states.
It never calls a language model.  Causal grouping uses only the verified
pre-intervention boundary: after dawn forgetting and before communication.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mas_cc.analysis.causal_response import build_causal_response_inputs
from mas_cc.musr_team_allocation_generator.ambiguity import (
    PrivateViewMetrics,
    TeamAllocationCompletionIndex,
)
from mas_cc.musr_team_allocation_generator.symbolic_facts import CanonicalFact
from mas_cc.storage import canonical_hash


ANALYSIS_VERSION = "blackboard_epistemic_phase_v1"
PRE_BOUNDARY = "post_forgetting_pre_intervention_delivery"
SCOPE = "union_of_participant_active_inventories"
DEFAULT_X_BINS = 8
DEFAULT_PHI_BANDS = 3


@dataclass(slots=True)
class SymbolicTask:
    """Frozen task facts plus a cached exact finite-world solver."""

    task_id: str
    task_hash: str
    gold_answer: str
    gold_index: int
    facts: Mapping[str, CanonicalFact]
    index: TeamAllocationCompletionIndex
    _cache: dict[tuple[str, ...], PrivateViewMetrics]
    fact_world_masks: Mapping[str, int]
    winner_world_masks: tuple[int, int, int]
    all_world_mask: int

    def evaluate(self, fact_ids: Sequence[str]) -> PrivateViewMetrics:
        key = tuple(sorted(set(map(str, fact_ids))))
        unknown = set(key) - set(self.facts)
        if unknown:
            raise ValueError(
                f"task {self.task_id!r} inventory contains unknown facts: "
                f"{sorted(unknown)}"
            )
        if key not in self._cache:
            self._cache[key] = self.index.metrics_for_facts(
                [self.facts[fact_id] for fact_id in key]
            )
        return self._cache[key]

    def solvable(self, fact_ids: Sequence[str]) -> tuple[bool, PrivateViewMetrics]:
        metrics = self.evaluate(fact_ids)
        return (
            math.isclose(
                metrics.probabilities[self.gold_index], 1.0, rel_tol=0, abs_tol=1e-12
            ),
            metrics,
        )

    def solvable_from_masks(self, fact_ids: Sequence[str]) -> bool:
        """Test unique-gold solvability without caching a temporary draw."""

        compatible = self.all_world_mask
        for fact_id in fact_ids:
            try:
                compatible &= self.fact_world_masks[str(fact_id)]
            except KeyError as exc:
                raise ValueError(
                    f"task {self.task_id!r} inventory contains unknown fact {fact_id!r}"
                ) from exc
        if compatible == 0:
            raise ValueError("canonical fact set has no valid completions")
        return compatible & ~self.winner_world_masks[self.gold_index] == 0


def _decoded(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str) and value[:1] in {"[", "{"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    decoded = _decoded(value)
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return {str(key): item for key, item in decoded.items()}


def _sequence(value: Any, label: str) -> list[Any]:
    decoded = _decoded(value)
    if not isinstance(decoded, (list, tuple)):
        raise ValueError(f"{label} must be a sequence")
    return list(decoded)


def _repo_root(path: Path) -> Path:
    candidate = path if path.is_dir() else path.parent
    for parent in (candidate, *candidate.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def resolve_task_dataset(
    configured: str | Path, *, recipe_path: str | Path | None = None
) -> Path:
    path = Path(configured).expanduser()
    if path.is_absolute():
        return path.resolve()
    anchor = (
        _repo_root(Path(recipe_path).resolve())
        if recipe_path
        else _repo_root(Path.cwd())
    )
    return (anchor / path).resolve()


def load_symbolic_tasks(
    task_ids: Sequence[str], task_dataset_dir: str | Path
) -> dict[str, SymbolicTask]:
    """Load and validate frozen canonical facts for every requested task."""

    root = Path(task_dataset_dir)
    result: dict[str, SymbolicTask] = {}
    for task_id in sorted(set(map(str, task_ids))):
        task_root = root / task_id
        task_path = task_root / "task.json"
        facts_path = task_root / "facts" / "all_true_facts.json"
        if not task_path.is_file() or not facts_path.is_file():
            raise ValueError(
                f"symbolic task artifacts are missing for {task_id!r} under {root}"
            )
        task = json.loads(task_path.read_text(encoding="utf-8"))
        expected_hash = canonical_hash(
            {key: value for key, value in task.items() if key != "task_hash"}
        )
        if task.get("task_hash") != expected_hash:
            raise ValueError(f"symbolic task hash does not match for {task_id!r}")
        facts_raw = json.loads(facts_path.read_text(encoding="utf-8"))
        facts = [CanonicalFact.from_dict(value) for value in facts_raw]
        by_id = {fact.fact_id: fact for fact in facts}
        if len(by_id) != len(facts):
            raise ValueError(f"symbolic task {task_id!r} has duplicate fact IDs")
        gold = str(task["gold_target"])
        try:
            gold_index = int(gold.rsplit("_", 1)[-1])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"unsupported gold answer {gold!r}") from exc
        if gold_index not in (0, 1, 2):
            raise ValueError(f"unsupported gold answer {gold!r}")
        result[task_id] = SymbolicTask(
            task_id=task_id,
            task_hash=str(task["task_hash"]),
            gold_answer=gold,
            gold_index=gold_index,
            facts=by_id,
            index=TeamAllocationCompletionIndex(),
            _cache={},
            fact_world_masks={},
            winner_world_masks=(0, 0, 0),
            all_world_mask=0,
        )
        solver = result[task_id]
        worlds = solver.index.worlds
        solver.fact_world_masks = {
            fact_id: sum(
                1 << position
                for position, (vector, _, _) in enumerate(worlds)
                if fact.holds(vector)
            )
            for fact_id, fact in by_id.items()
        }
        solver.winner_world_masks = tuple(
            sum(
                1 << position
                for position, (_, winner, _) in enumerate(worlds)
                if winner == target
            )
            for target in range(3)
        )  # type: ignore[assignment]
        solver.all_world_mask = (1 << len(worlds)) - 1
    return result


def _initial_inventory(row: Mapping[str, Any]) -> dict[str, list[str]]:
    raw = _decoded(row.get("initial_active_fact_ids_by_agent"))
    agent_ids = _decoded(row.get("agent_ids"))
    if isinstance(raw, Mapping):
        return {
            str(agent): sorted(set(map(str, facts))) for agent, facts in raw.items()
        }
    if not isinstance(raw, (list, tuple)) or not isinstance(agent_ids, (list, tuple)):
        raise ValueError("initial active inventories and agent IDs are required")
    if len(raw) != len(agent_ids):
        raise ValueError("initial active inventories are not aligned with agent IDs")
    return {
        str(agent): sorted(set(map(str, facts)))
        for agent, facts in zip(agent_ids, raw, strict=True)
    }


def _after_inventory(row: Mapping[str, Any]) -> dict[str, list[str]]:
    raw = _mapping(
        row.get("active_fact_ids_by_agent_after"),
        "active_fact_ids_by_agent_after",
    )
    return {
        agent: sorted(set(map(str, _sequence(facts, f"active facts for {agent}"))))
        for agent, facts in raw.items()
    }


def _remove_deactivated(
    inventory: Mapping[str, Sequence[str]], row: Mapping[str, Any]
) -> dict[str, list[str]]:
    result = {agent: set(map(str, facts)) for agent, facts in inventory.items()}
    raw = _decoded(row.get("persistence_deactivated_pairs")) or []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("persistence_deactivated_pairs must be a sequence")
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("persistence deactivation entries must be mappings")
        agent, fact = str(item["agent_id"]), str(item["fact_id"])
        if agent not in result or fact not in result[agent]:
            raise ValueError(
                f"persistence removed absent occurrence ({agent!r}, {fact!r})"
            )
        result[agent].remove(fact)
    return {agent: sorted(facts) for agent, facts in result.items()}


def reconstruct_active_inventories(rounds: pd.DataFrame) -> pd.DataFrame:
    """Return exact pre-delivery and after-round active inventories.

    Dawn records subtract the current forgetting events from the preceding
    boundary.  Legacy records use the preceding after-state directly because
    their forgetting event is already included in that after-state.
    """

    required = {"cell_id", "episode_id", "round_index"}
    if rounds.empty or not required.issubset(rounds.columns):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    source = rounds.copy()
    source["round_index"] = pd.to_numeric(source["round_index"], errors="raise").astype(
        int
    )
    if source.duplicated(list(required), keep=False).any():
        raise ValueError("epistemic round identities must be unique")
    for (cell_id, episode_id), episode in source.groupby(
        ["cell_id", "episode_id"], sort=True
    ):
        ordered = episode.sort_values("round_index")
        expected = list(range(int(ordered["round_index"].max()) + 1))
        if ordered["round_index"].tolist() != expected:
            raise ValueError(
                f"epistemic inventory reconstruction requires contiguous rounds: "
                f"{cell_id}/{episode_id}"
            )
        previous_after: dict[str, list[str]] | None = None
        for raw in ordered.to_dict(orient="records"):
            base = _initial_inventory(raw) if previous_after is None else previous_after
            protocol = str(raw.get("protocol") or "legacy")
            pre = (
                _remove_deactivated(base, raw)
                if protocol == "night_dawn_autonomous_day_v1"
                else {agent: list(facts) for agent, facts in base.items()}
            )
            after = _after_inventory(raw)
            if set(pre) != set(after):
                raise ValueError("pre/after active inventories use different agent IDs")
            rows.append(
                {
                    **raw,
                    "epistemic_boundary": PRE_BOUNDARY,
                    "active_fact_ids_by_agent_before": pre,
                    "active_fact_ids_by_agent_after_verified": after,
                    "inventory_reconstruction_valid": True,
                }
            )
            previous_after = after
    return pd.DataFrame(rows)


def _state_metrics(
    task: SymbolicTask, inventory: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    if not inventory:
        raise ValueError("active inventory cannot contain zero agents")
    agent_results = [task.solvable(facts)[0] for facts in inventory.values()]
    union = sorted({fact for facts in inventory.values() for fact in facts})
    collective, collective_metrics = task.solvable(union)
    holder_counts = [
        sum(fact in set(facts) for facts in inventory.values()) for fact in union
    ]
    counts = [len(set(facts)) for facts in inventory.values()]
    phi = float(np.mean(agent_results))
    probabilities = collective_metrics.probabilities
    return {
        "collective_solvable": bool(collective),
        "symbolic_individual_solvability_share": phi,
        "fragmentation_gap": float(collective) - phi,
        "solvable_agent_count": int(sum(agent_results)),
        "unsolvable_agent_share": 1.0 - phi,
        "active_union_fact_count": len(union),
        "active_union_fact_fraction": len(union) / len(task.facts),
        "active_fact_occurrence_count": int(sum(counts)),
        "active_mean_fact_count": float(np.mean(counts)),
        "active_min_fact_count": int(min(counts)),
        "active_max_fact_count": int(max(counts)),
        "active_mean_holder_redundancy": (
            float(np.mean(holder_counts)) if holder_counts else 0.0
        ),
        "active_min_holder_redundancy": min(holder_counts) if holder_counts else 0,
        "collective_compatible_world_count": collective_metrics.valid_completion_count,
        "collective_gold_probability": probabilities[task.gold_index],
        "collective_normalized_entropy": collective_metrics.normalized_entropy,
        "solver_status": "valid",
        "evidence_scope": SCOPE,
    }


def _stable_seed(seed: int, *parts: Any) -> int:
    digest = hashlib.sha256(
        "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _robustness(
    task: SymbolicTask,
    inventory: Mapping[str, Sequence[str]],
    rho: float,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if not 0 <= rho <= 1:
        raise ValueError("robustness persistence must lie in [0, 1]")
    current, _ = task.solvable(
        sorted({fact for facts in inventory.values() for fact in facts})
    )
    if not current:
        return {
            "estimate": math.nan,
            "mc_se": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "solvable_draws": 0,
            "draws": draws,
        }
    if rho == 1:
        return {
            "estimate": 1.0,
            "mc_se": 0.0,
            "ci_low": 1.0,
            "ci_high": 1.0,
            "solvable_draws": draws,
            "draws": draws,
        }
    if draws <= 0:
        raise ValueError("robustness_draws must be positive")
    rng = np.random.default_rng(seed)
    successes = 0
    holder_counts = {
        fact: sum(fact in set(map(str, facts)) for facts in inventory.values())
        for fact in {fact for facts in inventory.values() for fact in map(str, facts)}
    }
    fact_ids = sorted(holder_counts)
    survival_probabilities = np.asarray(
        [1.0 - (1.0 - rho) ** holder_counts[fact] for fact in fact_ids],
        dtype=float,
    )
    retained = rng.random((draws, len(fact_ids))) < survival_probabilities
    for draw in retained:
        surviving = [fact for fact, keep in zip(fact_ids, draw, strict=True) if keep]
        successes += int(task.solvable_from_masks(surviving))
    estimate = successes / draws
    se = math.sqrt(estimate * (1.0 - estimate) / draws)
    return {
        "estimate": estimate,
        "mc_se": se,
        "ci_low": max(0.0, estimate - 1.96 * se),
        "ci_high": min(1.0, estimate + 1.96 * se),
        "solvable_draws": successes,
        "draws": draws,
    }


def build_epistemic_round_states(
    rounds: pd.DataFrame,
    tasks: Mapping[str, SymbolicTask],
    *,
    robustness_draws: int = 500,
    reference_persistence: float = 0.85,
    seed: int = 1,
) -> pd.DataFrame:
    """Build the main bookkeeping time series with more than ten metrics."""

    reconstructed = reconstruct_active_inventories(rounds)
    if reconstructed.empty:
        return reconstructed
    output: list[dict[str, Any]] = []
    for raw in reconstructed.to_dict(orient="records"):
        task_id = str(raw.get("task_id") or raw.get("initial_task_id"))
        if task_id not in tasks:
            raise ValueError(f"no symbolic solver loaded for task {task_id!r}")
        task = tasks[task_id]
        recorded_gold = str(raw.get("correct_answer") or task.gold_answer)
        if recorded_gold != task.gold_answer:
            raise ValueError(
                f"recorded gold {recorded_gold!r} disagrees with frozen task"
            )
        pre = _mapping(raw["active_fact_ids_by_agent_before"], "pre inventory")
        after = _mapping(
            raw["active_fact_ids_by_agent_after_verified"], "after inventory"
        )
        pre_metrics = _state_metrics(task, pre)
        after_metrics = _state_metrics(task, after)
        configured_rho = float(raw.get("epistemic_persistence", 1.0))
        identity = (raw["cell_id"], raw["episode_id"], raw["round_index"])
        configured = _robustness(
            task,
            pre,
            configured_rho,
            robustness_draws,
            _stable_seed(seed, *identity, "configured"),
        )
        reference = _robustness(
            task,
            pre,
            reference_persistence,
            robustness_draws,
            _stable_seed(seed, *identity, "reference"),
        )
        row = {
            key: value
            for key, value in raw.items()
            if key
            not in {
                "active_fact_ids_by_agent_before",
                "active_fact_ids_by_agent_after_verified",
            }
        }
        row.update(pre_metrics)
        row.update(
            {
                f"{key}_after": value
                for key, value in after_metrics.items()
                if key not in {"solver_status", "evidence_scope"}
            }
        )
        row.update(
            {
                "task_hash": task.task_hash,
                "symbolic_gold_answer": task.gold_answer,
                "symbolic_solver": "TeamAllocationCompletionIndex",
                "symbolic_full_proof_share_recorded": pd.to_numeric(
                    pd.Series([raw.get("active_full_proof_agent_share_before")]),
                    errors="coerce",
                ).iloc[0],
                "configured_robustness": configured["estimate"],
                "configured_robustness_mc_se": configured["mc_se"],
                "configured_robustness_ci_low": configured["ci_low"],
                "configured_robustness_ci_high": configured["ci_high"],
                "configured_robustness_solvable_draws": configured["solvable_draws"],
                "configured_robustness_draws": configured["draws"],
                "reference_persistence": reference_persistence,
                "reference_robustness": reference["estimate"],
                "reference_robustness_mc_se": reference["mc_se"],
                "reference_robustness_ci_low": reference["ci_low"],
                "reference_robustness_ci_high": reference["ci_high"],
                "reference_robustness_solvable_draws": reference["solvable_draws"],
                "reference_robustness_draws": reference["draws"],
                "delta_symbolic_individual_solvability_share": (
                    after_metrics["symbolic_individual_solvability_share"]
                    - pre_metrics["symbolic_individual_solvability_share"]
                ),
                "delta_collective_solvable": int(after_metrics["collective_solvable"])
                - int(pre_metrics["collective_solvable"]),
                "analysis_version": ANALYSIS_VERSION,
            }
        )
        output.append(row)
    result = pd.DataFrame(output)
    recorded = pd.to_numeric(
        result["symbolic_full_proof_share_recorded"], errors="coerce"
    )
    result["symbolic_minus_recorded_full_proof_share"] = (
        result["symbolic_individual_solvability_share"] - recorded
    )
    return result


def _bin(values: pd.Series, count: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").clip(0, 1)
    return np.minimum(np.floor(numeric * count), count - 1).astype("Int64")


def epistemic_occupancy(
    states: pd.DataFrame, *, x_bins: int, phi_bands: int
) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame()
    frame = states.copy()
    causal = build_causal_response_inputs(frame, lags=(1,))
    shares = causal[["cell_id", "episode_id", "round_index", "x_t"]]
    frame = frame.merge(shares, on=["cell_id", "episode_id", "round_index"], how="left")
    frame["x_bin"] = _bin(frame["x_t"], x_bins)
    frame["phi_star_band"] = _bin(
        frame["symbolic_individual_solvability_share"], phi_bands
    )
    group = frame.groupby(
        ["cell_id", "x_bin", "phi_star_band", "collective_solvable"],
        dropna=False,
        as_index=False,
    ).agg(
        occupancy_count=("episode_id", "size"),
        n_episodes=("episode_id", "nunique"),
        mean_fragmentation_gap=("fragmentation_gap", "mean"),
        mean_reference_robustness=("reference_robustness", "mean"),
    )
    group["x_bin_center"] = (group["x_bin"].astype(float) + 0.5) / x_bins
    group["phi_star_band_center"] = (
        group["phi_star_band"].astype(float) + 0.5
    ) / phi_bands
    return group


def epistemic_parameter_summary(states: pd.DataFrame) -> pd.DataFrame:
    """Equal-episode summaries for budget-versus-persistence maps."""

    if states.empty:
        return pd.DataFrame()
    per_episode = states.groupby(["cell_id", "episode_id"], as_index=False).agg(
        fraction_rounds_collectively_solvable=("collective_solvable", "mean"),
        mean_fragmentation_gap=("fragmentation_gap", "mean"),
        mean_symbolic_individual_solvability_share=(
            "symbolic_individual_solvability_share",
            "mean",
        ),
        mean_active_union_fact_fraction=("active_union_fact_fraction", "mean"),
        mean_active_fact_occurrence_count=("active_fact_occurrence_count", "mean"),
        mean_active_holder_redundancy=("active_mean_holder_redundancy", "mean"),
        mean_collective_gold_probability=("collective_gold_probability", "mean"),
        mean_collective_entropy=("collective_normalized_entropy", "mean"),
        mean_configured_robustness=("configured_robustness", "mean"),
        mean_reference_robustness=("reference_robustness", "mean"),
        observed_rounds=("round_index", "nunique"),
    )
    return per_episode.groupby("cell_id", as_index=False).agg(
        fraction_rounds_collectively_solvable=(
            "fraction_rounds_collectively_solvable",
            "mean",
        ),
        mean_fragmentation_gap=("mean_fragmentation_gap", "mean"),
        mean_symbolic_individual_solvability_share=(
            "mean_symbolic_individual_solvability_share",
            "mean",
        ),
        mean_active_union_fact_fraction=("mean_active_union_fact_fraction", "mean"),
        mean_active_fact_occurrence_count=(
            "mean_active_fact_occurrence_count",
            "mean",
        ),
        mean_active_holder_redundancy=("mean_active_holder_redundancy", "mean"),
        mean_collective_gold_probability=("mean_collective_gold_probability", "mean"),
        mean_collective_entropy=("mean_collective_entropy", "mean"),
        mean_configured_robustness=("mean_configured_robustness", "mean"),
        mean_reference_robustness=("mean_reference_robustness", "mean"),
        n_episodes=("episode_id", "nunique"),
        min_observed_rounds=("observed_rounds", "min"),
        max_observed_rounds=("observed_rounds", "max"),
    )


def _bootstrap_frames(
    frame: pd.DataFrame, resamples: int, seed: int
) -> list[pd.DataFrame]:
    if frame.empty or resamples <= 0:
        return []
    blocks = sorted(frame["initialization_block_id"].astype(str).unique())
    by_block = {
        block: frame[frame["initialization_block_id"].astype(str) == block]
        for block in blocks
    }
    rng = np.random.default_rng(seed)
    return [
        pd.concat(
            [
                by_block[block]
                for block in rng.choice(blocks, len(blocks), replace=True)
            ],
            ignore_index=True,
        )
        for _ in range(resamples)
    ]


def _interval(values: Sequence[float], confidence: float) -> tuple[float, float]:
    finite = np.asarray(
        [value for value in values if math.isfinite(value)], dtype=float
    )
    if not len(finite):
        return math.nan, math.nan
    alpha = (1 - confidence) / 2
    return float(np.quantile(finite, alpha)), float(np.quantile(finite, 1 - alpha))


def _drift_estimate(group: pd.DataFrame, branch: str, component: str) -> float:
    if group.empty:
        return math.nan
    if branch == "activation":
        score = group["U_t"] / group["e_t"] * group[component]
    elif branch == "silence":
        score = (1 - group["U_t"]) / (1 - group["e_t"]) * group[component]
    else:
        score = group["ipw_contrast_weight"] * group[component]
    return float(score.mean())


def estimate_joint_drift(
    states: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    x_bins: int,
    phi_bands: int,
    bootstrap_resamples: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame()
    frame = build_causal_response_inputs(states, cells, lags=(1,))
    frame = frame[frame["episode_complete"] & frame["lag_1_available"]].copy()
    frame["delta_phi_star"] = (
        frame["symbolic_individual_solvability_share_after"]
        - frame["symbolic_individual_solvability_share"]
    )
    frame["x_bin"] = _bin(frame["x_t"], x_bins)
    frame["phi_star_band"] = _bin(
        frame["symbolic_individual_solvability_share"], phi_bands
    )
    draws = _bootstrap_frames(frame, bootstrap_resamples, seed)
    rows: list[dict[str, Any]] = []
    keys = ["cell_id", "x_bin", "phi_star_band"]
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        cell_id, x_bin, phi_band = key
        action = int((group["U_t"] == 1).sum())
        silence = int((group["U_t"] == 0).sum())
        status = (
            "unsupported"
            if min(action, silence) == 0
            else "limited"
            if min(action, silence) < 2
            else "adequate"
        )
        for branch in ("silence", "activation", "contrast"):
            estimates_x = [
                _drift_estimate(
                    draw[
                        (draw["cell_id"] == cell_id)
                        & (draw["x_bin"] == x_bin)
                        & (draw["phi_star_band"] == phi_band)
                    ],
                    branch,
                    "delta_x_h1",
                )
                for draw in draws
            ]
            estimates_phi = [
                _drift_estimate(
                    draw[
                        (draw["cell_id"] == cell_id)
                        & (draw["x_bin"] == x_bin)
                        & (draw["phi_star_band"] == phi_band)
                    ],
                    branch,
                    "delta_phi_star",
                )
                for draw in draws
            ]
            x_low, x_high = _interval(estimates_x, confidence)
            p_low, p_high = _interval(estimates_phi, confidence)
            rows.append(
                {
                    "cell_id": cell_id,
                    "x_bin": int(x_bin),
                    "x_bin_center": (int(x_bin) + 0.5) / x_bins,
                    "phi_star_band": int(phi_band),
                    "phi_star_band_center": (int(phi_band) + 0.5) / phi_bands,
                    "branch": branch,
                    "delta_x": _drift_estimate(group, branch, "delta_x_h1"),
                    "delta_x_ci_low": x_low,
                    "delta_x_ci_high": x_high,
                    "delta_phi_star": _drift_estimate(group, branch, "delta_phi_star"),
                    "delta_phi_star_ci_low": p_low,
                    "delta_phi_star_ci_high": p_high,
                    "n_rounds": len(group),
                    "n_episodes": group["episode_id"].nunique(),
                    "n_initialization_blocks": group[
                        "initialization_block_id"
                    ].nunique(),
                    "action_count": action,
                    "silence_count": silence,
                    "support_status": status,
                    "bootstrap_unit": "shared_initialization_block",
                    "bootstrap_resamples": bootstrap_resamples,
                    "confidence": confidence,
                    "analysis_version": ANALYSIS_VERSION,
                }
            )
    return pd.DataFrame(rows)


def estimate_epistemic_modulation(
    states: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    x_bins: int,
    phi_bands: int,
    bootstrap_resamples: int,
    confidence: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if states.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = build_causal_response_inputs(states, cells, lags=(1,))
    frame = frame[frame["episode_complete"] & frame["lag_1_available"]].copy()
    frame["causal_score"] = frame["ipw_contrast_weight"] * frame["delta_x_h1"]
    frame["x_bin"] = _bin(frame["x_t"], x_bins)
    frame["phi_star_band"] = _bin(
        frame["symbolic_individual_solvability_share"], phi_bands
    )
    draws = _bootstrap_frames(frame, bootstrap_resamples, seed)
    surface_rows: list[dict[str, Any]] = []
    for (cell_id, x_bin, phi_band), group in frame.groupby(
        ["cell_id", "x_bin", "phi_star_band"], dropna=False, sort=True
    ):
        action, silence = int((group.U_t == 1).sum()), int((group.U_t == 0).sum())
        estimates = [
            float(
                draw[
                    (draw.cell_id == cell_id)
                    & (draw.x_bin == x_bin)
                    & (draw.phi_star_band == phi_band)
                ]["causal_score"].mean()
            )
            for draw in draws
        ]
        low, high = _interval(estimates, confidence)
        surface_rows.append(
            {
                "cell_id": cell_id,
                "x_bin": int(x_bin),
                "x_bin_center": (int(x_bin) + 0.5) / x_bins,
                "phi_star_band": int(phi_band),
                "phi_star_band_center": (int(phi_band) + 0.5) / phi_bands,
                "causal_susceptibility": float(group["causal_score"].mean()),
                "ci_low": low,
                "ci_high": high,
                "n_rounds": len(group),
                "n_episodes": group.episode_id.nunique(),
                "n_initialization_blocks": group.initialization_block_id.nunique(),
                "action_count": action,
                "silence_count": silence,
                "support_status": "unsupported"
                if min(action, silence) == 0
                else "limited"
                if min(action, silence) < 2
                else "adequate",
                "surface_kind": "raw_binned",
                "analysis_version": ANALYSIS_VERSION,
            }
        )

    regression_rows: list[dict[str, Any]] = []
    for cell_id, group in frame.groupby("cell_id", sort=True):
        x = pd.DataFrame(
            {
                "intercept": 1.0,
                "x_t": group["x_t"].astype(float),
                "phi_star": group["symbolic_individual_solvability_share"].astype(
                    float
                ),
                "round_scaled": group["round_index"].astype(float)
                / max(1.0, float(group["round_index"].max())),
            }
        ).to_numpy()
        y = group["causal_score"].to_numpy(dtype=float)
        rank = int(np.linalg.matrix_rank(x))
        condition = float(np.linalg.cond(x))
        identified = (
            rank == x.shape[1] and condition <= 1e8 and float(np.ptp(x[:, 2])) >= 0.1
        )
        coefficient = (
            float(np.linalg.lstsq(x, y, rcond=None)[0][2] * 0.1)
            if identified
            else math.nan
        )
        boot: list[float] = []
        for draw in draws:
            subset = draw[draw["cell_id"] == cell_id]
            if subset.empty:
                continue
            design = np.column_stack(
                [
                    np.ones(len(subset)),
                    subset["x_t"].astype(float),
                    subset["symbolic_individual_solvability_share"].astype(float),
                    subset["round_index"].astype(float)
                    / max(1.0, float(group["round_index"].max())),
                ]
            )
            if (
                np.linalg.matrix_rank(design) == design.shape[1]
                and np.linalg.cond(design) <= 1e8
            ):
                boot.append(
                    float(
                        np.linalg.lstsq(design, subset["causal_score"], rcond=None)[0][
                            2
                        ]
                        * 0.1
                    )
                )
        low, high = _interval(boot, confidence)
        low_group = group[
            group["symbolic_individual_solvability_share"]
            <= group["symbolic_individual_solvability_share"].quantile(1 / 3)
        ]
        high_group = group[
            group["symbolic_individual_solvability_share"]
            >= group["symbolic_individual_solvability_share"].quantile(2 / 3)
        ]
        low_support = set(zip(low_group.x_bin, low_group.round_index, strict=False))
        high_support = set(zip(high_group.x_bin, high_group.round_index, strict=False))
        overlap = len(low_support & high_support)
        regression_rows.append(
            {
                "cell_id": cell_id,
                "metric": "epistemic_modulation_per_0_1_phi_star",
                "estimate": coefficient,
                "ci_low": low,
                "ci_high": high,
                "identified": identified,
                "design_rank": rank,
                "design_columns": x.shape[1],
                "condition_number": condition,
                "phi_star_range": float(np.ptp(x[:, 2])),
                "low_high_x_time_overlap_cells": overlap,
                "low_high_comparison": (
                    float(
                        high_group.causal_score.mean() - low_group.causal_score.mean()
                    )
                    if overlap and len(low_group) and len(high_group)
                    else math.nan
                ),
                "n_rounds": len(group),
                "n_episodes": group.episode_id.nunique(),
                "n_initialization_blocks": group.initialization_block_id.nunique(),
                "action_count": int((group.U_t == 1).sum()),
                "silence_count": int((group.U_t == 0).sum()),
                "bootstrap_unit": "shared_initialization_block",
                "bootstrap_resamples": bootstrap_resamples,
                "confidence": confidence,
                "interpretation": "heterogeneity_of_randomized_activation_effect_not_effect_of_knowledge",
                "analysis_version": ANALYSIS_VERSION,
            }
        )
    return pd.DataFrame(surface_rows), pd.DataFrame(regression_rows)


def classify_capture_timing(
    states: pd.DataFrame, *, threshold: float, consecutive_rounds: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if states.empty:
        return pd.DataFrame(), pd.DataFrame()
    causal = build_causal_response_inputs(states, lags=(1,))
    shares = causal[["cell_id", "episode_id", "round_index", "x_t"]]
    frame = states.merge(shares, on=["cell_id", "episode_id", "round_index"])
    rows: list[dict[str, Any]] = []
    for cell_id, cell in frame.groupby("cell_id", sort=True):
        lengths = cell.groupby("episode_id")["round_index"].nunique()
        common_horizon = int(lengths.min())
        maximum_followup = int(lengths.max())
        for episode_id, episode in cell.groupby("episode_id", sort=True):
            ordered = (
                episode.sort_values("round_index")
                .iloc[:common_horizon]
                .reset_index(drop=True)
            )
            initial_g = bool(ordered.iloc[0]["collective_solvable"])
            losses = ordered.index[
                ~ordered["collective_solvable"].astype(bool)
            ].tolist()
            loss = int(ordered.iloc[losses[0]]["round_index"]) if losses else None
            capture = None
            capture_g = False
            for start in range(0, len(ordered) - consecutive_rounds + 1):
                window = ordered.iloc[start : start + consecutive_rounds]
                if (window["x_t"] >= threshold).all():
                    capture = int(window.iloc[0]["round_index"])
                    capture_g = bool(window["collective_solvable"].astype(bool).all())
                    break
            rows.append(
                {
                    "cell_id": cell_id,
                    "episode_id": episode_id,
                    "initial_collective_solvable": initial_g,
                    "initially_unsolvable": not initial_g,
                    "first_loss_round": loss,
                    "first_capture_start_round": capture,
                    "loss_observed": loss is not None,
                    "capture_observed": capture is not None,
                    "capture_before_first_loss": (
                        bool(capture < loss)
                        if initial_g and capture is not None and loss is not None
                        else (
                            True
                            if initial_g and capture is not None and loss is None
                            else False
                            if initial_g
                            else math.nan
                        )
                    ),
                    "capture_with_collective_solvability_throughout": (
                        capture_g if initial_g else math.nan
                    ),
                    "joint_false_capture_and_solvable_fraction": float(
                        (
                            (ordered["x_t"] >= threshold)
                            & ordered["collective_solvable"].astype(bool)
                        ).mean()
                    ),
                    "observed_rounds": int(lengths.loc[episode_id]),
                    "analyzed_rounds": common_horizon,
                    "common_horizon": common_horizon,
                    "maximum_followup": maximum_followup,
                    "followup_truncated_to_common_horizon": bool(
                        int(lengths.loc[episode_id]) > common_horizon
                    ),
                    "last_observed_round": int(ordered["round_index"].max()),
                    "capture_threshold": threshold,
                    "capture_consecutive_rounds": consecutive_rounds,
                    "censoring_status": (
                        "initially_unsolvable"
                        if not initial_g
                        else "both_observed"
                        if capture is not None and loss is not None
                        else "capture_only"
                        if capture is not None
                        else "loss_only"
                        if loss is not None
                        else "neither_observed"
                    ),
                    "analysis_version": ANALYSIS_VERSION,
                }
            )
    episode_table = pd.DataFrame(rows)
    summary = episode_table.groupby("cell_id", as_index=False).agg(
        capture_before_first_loss_probability=("capture_before_first_loss", "mean"),
        capture_with_solvable_window_probability=(
            "capture_with_collective_solvability_throughout",
            "mean",
        ),
        joint_false_capture_and_solvable_fraction=(
            "joint_false_capture_and_solvable_fraction",
            "mean",
        ),
        initially_unsolvable_fraction=("initially_unsolvable", "mean"),
        loss_observed_fraction=("loss_observed", "mean"),
        capture_observed_fraction=("capture_observed", "mean"),
        n_episodes=("episode_id", "nunique"),
        common_horizon=("common_horizon", "first"),
        maximum_followup=("maximum_followup", "first"),
        followup_truncated_episode_count=(
            "followup_truncated_to_common_horizon",
            "sum",
        ),
    )
    return episode_table, summary


def _attach(frame: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or cells.empty or "cell_id" not in frame:
        return frame
    excluded = {
        "study_id",
        "source_run_id",
        "source_run_path",
        "source_cell_id",
        "config_hash",
        "resolved_config_hash",
        "recorded_resolved_config_hash",
        "expected_episodes",
        "completed_episodes",
        "failed_episodes",
        "sealed",
    }
    coordinates = cells[
        [
            column
            for column in cells.columns
            if column == "cell_id" or column not in excluded
        ]
    ].drop_duplicates("cell_id")
    return frame.merge(
        coordinates, on="cell_id", how="left", suffixes=("", "_coordinate")
    )


def analyze_epistemic_phase_diagrams(
    rounds: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    task_dataset_dir: str | Path,
    robustness_draws: int = 500,
    reference_persistence: float = 0.85,
    x_bins: int = DEFAULT_X_BINS,
    phi_bands: int = DEFAULT_PHI_BANDS,
    capture_threshold: float = 0.75,
    capture_consecutive_rounds: int = 3,
    bootstrap_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
) -> dict[str, pd.DataFrame]:
    """Create all five analysis classes from retained canonical rounds."""

    if x_bins < 1 or phi_bands < 1:
        raise ValueError("x_bins and phi_bands must be positive")
    if robustness_draws < 1:
        raise ValueError("robustness_draws must be positive")
    if not 0 <= reference_persistence <= 1:
        raise ValueError("reference_persistence must lie in [0, 1]")
    if not 0 <= capture_threshold <= 1 or capture_consecutive_rounds < 1:
        raise ValueError("capture rule is invalid")
    task_series = rounds.get(
        "task_id", rounds.get("initial_task_id", pd.Series(dtype=str))
    )
    task_ids = [str(value) for value in task_series.dropna().unique()]
    tasks = load_symbolic_tasks(task_ids, task_dataset_dir)
    states = build_epistemic_round_states(
        rounds,
        tasks,
        robustness_draws=robustness_draws,
        reference_persistence=reference_persistence,
        seed=seed,
    )
    parameter = epistemic_parameter_summary(states)
    occupancy = epistemic_occupancy(states, x_bins=x_bins, phi_bands=phi_bands)
    drift = estimate_joint_drift(
        states,
        cells,
        x_bins=x_bins,
        phi_bands=phi_bands,
        bootstrap_resamples=bootstrap_resamples,
        confidence=confidence,
        seed=seed,
    )
    susceptibility, modulation = estimate_epistemic_modulation(
        states,
        cells,
        x_bins=x_bins,
        phi_bands=phi_bands,
        bootstrap_resamples=bootstrap_resamples,
        confidence=confidence,
        seed=seed + 1,
    )
    timing, timing_summary = classify_capture_timing(
        states,
        threshold=capture_threshold,
        consecutive_rounds=capture_consecutive_rounds,
    )
    return {
        "epistemic_round_timeseries": _attach(states, cells),
        "epistemic_parameter_summary": _attach(parameter, cells),
        "epistemic_state_occupancy": _attach(occupancy, cells),
        "epistemic_joint_drift": _attach(drift, cells),
        "epistemic_causal_susceptibility": _attach(susceptibility, cells),
        "epistemic_modulation": _attach(modulation, cells),
        "epistemic_capture_timing": _attach(timing, cells),
        "epistemic_capture_summary": _attach(timing_summary, cells),
    }


def render_epistemic_phase_plots(
    tables: Mapping[str, pd.DataFrame], destination: str | Path
) -> list[str]:
    """Render fixed parameter, state-motion, response, timing, and central views."""

    import matplotlib.pyplot as plt

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    parameter = tables.get("epistemic_parameter_summary", pd.DataFrame())
    drift = tables.get("epistemic_joint_drift", pd.DataFrame())
    susceptibility = tables.get("epistemic_causal_susceptibility", pd.DataFrame())
    timing = tables.get("epistemic_capture_summary", pd.DataFrame())

    def heatmap(axis: Any, frame: pd.DataFrame, value: str, title: str) -> None:
        pivot = frame.pivot_table(
            index="epistemic_persistence",
            columns="intervention_budget",
            values=value,
            aggfunc="mean",
        ).sort_index()
        image = axis.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto")
        axis.set_xticks(range(len(pivot.columns)), labels=map(str, pivot.columns))
        axis.set_yticks(range(len(pivot.index)), labels=map(str, pivot.index))
        axis.set(xlabel="assigned budget b", ylabel="persistence rho", title=title)
        axis.figure.colorbar(image, ax=axis, shrink=0.8)

    if not parameter.empty and {
        "intervention_budget",
        "epistemic_persistence",
    }.issubset(parameter):
        figure, axes = plt.subplots(1, 3, figsize=(14, 4))
        heatmap(
            axes[0],
            parameter,
            "fraction_rounds_collectively_solvable",
            "Collective solvability G",
        )
        heatmap(axes[1], parameter, "mean_fragmentation_gap", "Fragmentation gap")
        heatmap(
            axes[2],
            parameter,
            "mean_reference_robustness",
            "Robustness at reference rho",
        )
        figure.suptitle("Epistemic finite-horizon parameter maps")
        figure.tight_layout()
        path = destination / "epistemic_parameter_maps.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))

    supported_drift = (
        drift[
            (drift.get("branch") == "contrast")
            & (drift.get("support_status") != "unsupported")
        ]
        if not drift.empty
        else drift
    )
    if not supported_drift.empty:
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.quiver(
            supported_drift["x_bin_center"],
            supported_drift["phi_star_band_center"],
            supported_drift["delta_x"],
            supported_drift["delta_phi_star"],
            angles="xy",
            scale_units="xy",
            scale=1,
        )
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="current target share x",
            ylabel="symbolic proof share phi*",
            title="Causal displacement arrows",
        )
        figure.tight_layout()
        path = destination / "epistemic_causal_displacement.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))

    supported_chi = (
        susceptibility[susceptibility.get("support_status") != "unsupported"]
        if not susceptibility.empty
        else susceptibility
    )
    if not supported_chi.empty:
        pivot = supported_chi.pivot_table(
            index="phi_star_band_center",
            columns="x_bin_center",
            values="causal_susceptibility",
            aggfunc="mean",
        ).sort_index()
        figure, axis = plt.subplots(figsize=(7, 4.5))
        image = axis.imshow(
            pivot.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="coolwarm"
        )
        axis.set_xticks(
            range(len(pivot.columns)), labels=[f"{x:.2f}" for x in pivot.columns]
        )
        axis.set_yticks(
            range(len(pivot.index)), labels=[f"{x:.2f}" for x in pivot.index]
        )
        axis.set(
            xlabel="current target share x",
            ylabel="symbolic proof share phi*",
            title="Raw binned causal susceptibility",
        )
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        path = destination / "epistemic_causal_susceptibility.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))

    if not timing.empty and {"intervention_budget", "epistemic_persistence"}.issubset(
        timing
    ):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        heatmap(
            axes[0],
            timing,
            "capture_before_first_loss_probability",
            "Capture before first evidence loss",
        )
        heatmap(
            axes[1],
            timing,
            "capture_with_solvable_window_probability",
            "Capture while G=1",
        )
        figure.tight_layout()
        path = destination / "epistemic_capture_timing.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))

    if not parameter.empty:
        figure, axes = plt.subplots(1, 3, figsize=(14, 4))
        if {"intervention_budget", "epistemic_persistence"}.issubset(parameter):
            heatmap(
                axes[0],
                parameter,
                "fraction_rounds_collectively_solvable",
                "Evidence availability",
            )
        if not supported_drift.empty:
            axes[1].quiver(
                supported_drift["x_bin_center"],
                supported_drift["phi_star_band_center"],
                supported_drift["delta_x"],
                supported_drift["delta_phi_star"],
                angles="xy",
                scale_units="xy",
                scale=1,
            )
            axes[1].set(
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="x",
                ylabel="phi*",
                title="Causal state motion",
            )
        if not supported_chi.empty:
            pivot = supported_chi.pivot_table(
                index="phi_star_band_center",
                columns="x_bin_center",
                values="causal_susceptibility",
                aggfunc="mean",
            ).sort_index()
            image = axes[2].imshow(
                pivot.to_numpy(dtype=float),
                origin="lower",
                aspect="auto",
                cmap="coolwarm",
            )
            axes[2].set(title="Control response", xlabel="x bin", ylabel="phi* band")
            figure.colorbar(image, ax=axes[2], shrink=0.8)
        figure.suptitle("Evidence availability, state-space motion, and causal control")
        figure.tight_layout()
        path = destination / "epistemic_central_figure.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))
    return paths


__all__ = [
    "ANALYSIS_VERSION",
    "PRE_BOUNDARY",
    "SCOPE",
    "SymbolicTask",
    "analyze_epistemic_phase_diagrams",
    "build_epistemic_round_states",
    "classify_capture_timing",
    "epistemic_occupancy",
    "epistemic_parameter_summary",
    "estimate_epistemic_modulation",
    "estimate_joint_drift",
    "load_symbolic_tasks",
    "reconstruct_active_inventories",
    "render_epistemic_phase_plots",
    "resolve_task_dataset",
]
