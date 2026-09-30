"""Read-only adapter: archive parquet -> validated per-initialization arrays.

Nothing here writes to the source directory.  Every path is opened for reading.

The round records store the population as a JSON list of 24 allocation labels in
``population_state_before`` / ``population_state_after``.  We rebuild the count
vector Y_t = (n_0, n_1, n_2) from those labels and validate rather than repair:
a trajectory that fails a check is excluded with a recorded reason.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ARCHIVE_ROOT = Path("/Users/rsanchez/Projects/agents_control/new_rnd_init_experiment")
RECORDS = ARCHIVE_ROOT / "simulation_data/records/21-09-2026-full-vs-report-v1_analysis"

ALLOCATIONS = ("ALLOCATION_0", "ALLOCATION_1", "ALLOCATION_2")
POPULATION = 24
N_ROUNDS = 15                 # round_index 0..14
HORIZON = 15                  # Y_0 .. Y_15
TARGETS = (0, 2)
TARGET_WEIGHTS = {0: 0.5, 2: 0.5}
ARM_OF_SEMANTICS = {"correct": "t0", "ALLOCATION_2": "t2"}
TARGET_OF_ARM = {"t0": 0, "t2": 2}

# Columns actually used.  The round table has 464 columns; reading all of them for
# 20,730 rows is wasteful and makes provenance harder to state.
ROUND_COLUMNS = [
    "cell_id", "episode_id", "round_index", "physical_initial_state_hash",
    "population_state_before", "population_state_after",
    "U_k", "P_U1_given_Y", "actual_controller_posts", "controller_enabled",
    "sensor_count_vector", "sensor_target_share", "sensor_sample_size", "q_c_effective",
    "active_mean_supporting_fact_coverage_after", "active_full_proof_agent_share_after",
    "active_supporting_fact_reach_after", "active_mean_fact_count_after",
    "vote_entropy", "truth_vote_share", "controller_target_share",
    "controller_unique_readers", "controller_message_exposures",
    "new_evidence_acquisitions", "peer_fact_exposures",
]
CELL_COLUMNS = [
    "cell_id", "communication_profile", "epistemic_persistence", "intervention_budget",
    "controller_target_semantics", "sensor_sample_size", "population_size", "horizon",
    "threshold", "beta", "task_id",
]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def labels_to_counts(labels: Any) -> np.ndarray:
    """``'["ALLOCATION_0", ...]'`` (24 labels) -> integer counts ``(n_0, n_1, n_2)``."""
    if isinstance(labels, str):
        labels = json.loads(labels)
    if labels is None:
        raise ValueError("missing population state")
    counts = np.zeros(3, dtype=np.int64)
    for lab in labels:
        if lab not in ALLOCATIONS:
            raise ValueError(f"unknown allocation label {lab!r}")
        counts[ALLOCATIONS.index(lab)] += 1
    if counts.sum() != POPULATION:
        raise ValueError(f"population state has {counts.sum()} votes, expected {POPULATION}")
    return counts


@dataclass
class Trajectory:
    """One episode: a complete 15-round run from one initialization."""
    init: str
    cell_id: str
    episode_id: str
    setting: str
    profile: str
    rho: float
    arm: str
    budget: int | None
    counts: np.ndarray        # (HORIZON+1, 3) integer, row t = Y_t
    u: np.ndarray             # (N_ROUNDS,) activation, NaN where no gate (round 0, silent arm)
    e: np.ndarray             # (N_ROUNDS,) logged activation probability, NaN where absent
    posts: np.ndarray         # (N_ROUNDS,) controller posts in that round
    sensor: np.ndarray        # (N_ROUNDS, 3) sampled vote-count vector, NaN where absent
    kappa: np.ndarray         # (N_ROUNDS,) mean active supporting-fact coverage after the round
    phi: np.ndarray           # (N_ROUNDS,) full-proof ownership share after the round
    reach: np.ndarray         # (N_ROUNDS,) mean population share holding a supporting fact


@dataclass
class Comparison:
    """All initializations with a complete silent / t0 / t2 triple for one setting+budget."""
    setting: str
    profile: str
    rho: float
    budget: int
    inits: list[str]
    silent: np.ndarray                     # (m, HORIZON+1, 3)
    controlled: dict[int, np.ndarray]      # target -> (m, HORIZON+1, 3)
    u: dict[int, np.ndarray]               # target -> (m, N_ROUNDS)
    e: dict[int, np.ndarray]               # target -> (m, N_ROUNDS)
    posts: dict[int, np.ndarray]           # target -> (m, N_ROUNDS)
    sensor: dict[int, np.ndarray]          # target -> (m, N_ROUNDS, 3)
    kappa: dict[str, np.ndarray] = field(default_factory=dict)
    phi: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.profile}_rho{self.rho:.2f}_b{self.budget}"

    @property
    def m(self) -> int:
        return len(self.inits)


@dataclass
class Audit:
    n_round_rows: int
    n_episodes: int
    n_inits: int
    n_trajectories: int
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    arm_counts: dict[str, int] = field(default_factory=dict)
    comparison_counts: dict[str, int] = field(default_factory=dict)
    single_arm_counts: dict[str, int] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in
                ("n_round_rows", "n_episodes", "n_inits", "n_trajectories", "exclusions",
                 "arm_counts", "comparison_counts", "single_arm_counts", "provenance")}


def _vector(raw: Any) -> np.ndarray | None:
    """Parse a stored JSON array column; ``None`` when the cell is empty."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    if isinstance(raw, str):
        raw = json.loads(raw)
    return np.asarray(raw, dtype=float).ravel()


def _mean_reach(raw: Any) -> float:
    """``active_supporting_fact_reach_after`` stores one holder count per supporting
    fact (49 of them), not a scalar.  Summarize it as the mean share of the
    population holding a supporting fact."""
    v = _vector(raw)
    if v is None or v.size == 0:
        return float("nan")
    return float(v.mean() / POPULATION)


def _sensor_vector(raw: Any) -> np.ndarray:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.full(3, np.nan)
    v = _vector(raw)
    return v if v is not None and v.size == 3 else np.full(3, np.nan)


def load_trajectories(records: Path = RECORDS) -> tuple[list[Trajectory], Audit]:
    """Rebuild every complete trajectory and validate it."""
    cells = pd.read_parquet(records / "cells.parquet", columns=CELL_COLUMNS)
    rounds = pd.read_parquet(records / "rounds.parquet", columns=ROUND_COLUMNS)
    df = rounds.merge(cells, on="cell_id", suffixes=("", "_cell"))
    df["arm"] = np.where(df.intervention_budget.isna(), "silent",
                         df.controller_target_semantics.map(ARM_OF_SEMANTICS))
    df["setting"] = df.communication_profile + "/rho" + df.epistemic_persistence.map("{:.2f}".format)

    trajectories: list[Trajectory] = []
    exclusions: list[dict[str, Any]] = []
    for (cell_id, episode_id), g in df.groupby(["cell_id", "episode_id"], sort=True):
        note = {"cell_id": cell_id, "episode_id": episode_id,
                "setting": g.setting.iloc[0], "arm": g.arm.iloc[0]}
        try:
            g = g.sort_values("round_index")
            idx = g.round_index.to_numpy()
            if idx.tolist() != list(range(N_ROUNDS)):
                raise ValueError(f"incomplete or duplicated rounds: {idx.tolist()}")
            if g.physical_initial_state_hash.nunique() != 1:
                raise ValueError("initialization hash varies within an episode")
            before = np.stack([labels_to_counts(v) for v in g.population_state_before])
            after = np.stack([labels_to_counts(v) for v in g.population_state_after])
            # the before-vector of round t must equal the after-vector of round t-1
            if not np.array_equal(before[1:], after[:-1]):
                raise ValueError("before/previous-after continuity violated")
            counts = np.vstack([before[:1], after])           # Y_0 .. Y_15
            arm = g.arm.iloc[0]
            budget = None if arm == "silent" else int(g.intervention_budget.iloc[0])
            u = g.U_k.to_numpy(dtype=float)
            e = g.P_U1_given_Y.to_numpy(dtype=float)
            posts = g.actual_controller_posts.fillna(0).to_numpy(dtype=float)
            # the first round carries no controller decision by design
            if arm != "silent":
                if not np.isnan(u[0]):
                    raise ValueError("round 0 unexpectedly carries a controller decision")
                if np.isnan(u[1:]).any():
                    raise ValueError("missing controller decision after round 0")
                if not np.allclose(posts[1:], budget * u[1:]):
                    raise ValueError("posts do not equal budget times activation")
            trajectories.append(Trajectory(
                init=str(g.physical_initial_state_hash.iloc[0]), cell_id=cell_id,
                episode_id=str(episode_id), setting=g.setting.iloc[0],
                profile=g.communication_profile.iloc[0], rho=float(g.epistemic_persistence.iloc[0]),
                arm=arm, budget=budget, counts=counts, u=u, e=e, posts=posts,
                sensor=np.stack([_sensor_vector(v) for v in g.sensor_count_vector]),
                kappa=g.active_mean_supporting_fact_coverage_after.to_numpy(dtype=float),
                phi=g.active_full_proof_agent_share_after.to_numpy(dtype=float),
                reach=np.array([_mean_reach(v) for v in g.active_supporting_fact_reach_after])))
        except ValueError as err:
            note["reason"] = str(err)
            exclusions.append(note)

    audit = Audit(
        n_round_rows=len(rounds), n_episodes=int(df.groupby(["cell_id", "episode_id"]).ngroups),
        n_inits=int(df.physical_initial_state_hash.nunique()), n_trajectories=len(trajectories),
        exclusions=exclusions,
        arm_counts=pd.Series([t.arm for t in trajectories]).value_counts().to_dict())
    return trajectories, audit


def build_comparisons(trajectories: list[Trajectory], audit: Audit) -> dict[str, Comparison]:
    """Intersect initializations across the silent, t0 and t2 arms of each setting+budget.

    The silent arm has no budget, so the same silent trajectory is reused by every
    budget within a setting.  That reuse is real shared randomness: it is why the
    bootstrap resamples whole initializations rather than arms.
    """
    by_key: dict[tuple, dict[str, Trajectory]] = {}
    silent: dict[tuple[str, str], Trajectory] = {}
    for t in trajectories:
        if t.arm == "silent":
            silent[(t.setting, t.init)] = t
        else:
            by_key.setdefault((t.setting, t.budget), {}).setdefault(t.arm, {})[t.init] = t  # type: ignore[union-attr]

    comparisons: dict[str, Comparison] = {}
    single: dict[str, int] = {}
    for (setting, budget), arms in sorted(by_key.items()):
        t0, t2 = arms.get("t0", {}), arms.get("t2", {})
        sil = {i for (s, i) in silent if s == setting}
        common = sorted(set(t0) & set(t2) & sil)
        for arm, d in (("t0", t0), ("t2", t2)):
            single[f"{setting}_b{budget}_{arm}"] = len(set(d) & sil)
        if not common:
            continue
        any_t = next(iter(t0.values()))
        comp = Comparison(
            setting=setting, profile=any_t.profile, rho=any_t.rho, budget=int(budget), inits=common,
            silent=np.stack([silent[(setting, i)].counts for i in common]),
            controlled={0: np.stack([t0[i].counts for i in common]),
                        2: np.stack([t2[i].counts for i in common])},
            u={0: np.stack([t0[i].u for i in common]), 2: np.stack([t2[i].u for i in common])},
            e={0: np.stack([t0[i].e for i in common]), 2: np.stack([t2[i].e for i in common])},
            posts={0: np.stack([t0[i].posts for i in common]),
                   2: np.stack([t2[i].posts for i in common])},
            sensor={0: np.stack([t0[i].sensor for i in common]),
                    2: np.stack([t2[i].sensor for i in common])},
            kappa={"silent": np.stack([silent[(setting, i)].kappa for i in common]),
                   "t0": np.stack([t0[i].kappa for i in common]),
                   "t2": np.stack([t2[i].kappa for i in common])},
            phi={"silent": np.stack([silent[(setting, i)].phi for i in common]),
                 "t0": np.stack([t0[i].phi for i in common]),
                 "t2": np.stack([t2[i].phi for i in common])})
        comparisons[comp.key] = comp

    audit.comparison_counts = {k: c.m for k, c in comparisons.items()}
    audit.single_arm_counts = single
    return comparisons


def load(records: Path = RECORDS) -> tuple[dict[str, Comparison], list[Trajectory], Audit]:
    trajectories, audit = load_trajectories(records)
    comparisons = build_comparisons(trajectories, audit)
    audit.provenance = {
        "archive_root": str(ARCHIVE_ROOT),
        "rounds_parquet_sha256": sha256_of(records / "rounds.parquet"),
        "cells_parquet_sha256": sha256_of(records / "cells.parquet"),
        "columns_read": ROUND_COLUMNS,
        "read_only": True,
    }
    return comparisons, trajectories, audit
