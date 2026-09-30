"""Data adapter: immutable archive -> canonical vote-count paths -> parent triplets.

Terminology (see NOTATION.md in the study folder):

* parent      one independent 2-round preparation run whose end state is frozen as a
              checkpoint; all continuations of that checkpoint share ``parent_id``.
* branch      one 10-round continuation of a parent under a branch policy
              (``none`` = silent baseline, ``always_*`` / ``sensing_*`` = controlled)
              and a posting budget.
* comparison  a fixed (q, rho, schedule, budget); inside it every retained parent
              supplies one silent path, one target-0 path and one target-2 path.

Y_t is the three-count vote vector (N_0, N_1, N_2) after continuation round t, with
Y_0 the before-vector of continuation round 1.  Counts are rebuilt from the
``population_state_before`` / ``population_state_after`` JSON label lists; nothing
is repaired silently - every discrepancy is raised or recorded as an exclusion.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ALLOCATIONS = ("ALLOCATION_0", "ALLOCATION_1", "ALLOCATION_2")
POPULATION = 24
SILENT_POLICY = "none"
CHECKPOINT_POLICY = "checkpoint"
TARGET_OF_POLICY = {
    "always_truth": 0, "sensing_truth": 0,
    "always_false": 2, "sensing_false": 2,
}
SCHEDULE_OF_POLICY = {
    "always_truth": "always", "always_false": "always",
    "sensing_truth": "sensing", "sensing_false": "sensing",
}
TARGETS = (0, 2)
TARGET_WEIGHTS = {0: 0.5, 2: 0.5}

ROUND_RECORDS_MEMBER = "tables/checkpoint_complete_round_records.parquet"
ROUND_RECORD_COLUMNS = [
    "parent_id", "checkpoint_hash", "cell_id", "q", "rho", "branch_policy",
    "posting_budget", "copy_id", "post_branch_horizon", "absolute_round",
    "population_state_before", "population_state_after",
    "controller_target_semantic_id", "correct_answer_semantic_id",
    "false_target_semantic_id", "branch_status",
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


def verify_archive(path: Path, expected_sha256: str | None) -> str:
    """Return the archive's SHA256; raise if it does not match the declared value."""
    digest = sha256_of(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"archive hash mismatch for {path}: {digest} != {expected_sha256}")
    return digest


def extract_member(archive: Path, member: str, out_dir: Path) -> Path:
    """Extract one member read-only from the zip (the archive itself is never modified)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / member
    if not target.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extract(member, out_dir)
    return target


def load_round_records(parquet_path: Path) -> pd.DataFrame:
    return pd.read_parquet(parquet_path, columns=ROUND_RECORD_COLUMNS)


def labels_to_counts(labels: Any) -> np.ndarray:
    """``["ALLOCATION_0", ...]`` (24 labels) -> integer counts ``(N_0, N_1, N_2)``."""
    if isinstance(labels, str):
        labels = json.loads(labels)
    if labels is None or (isinstance(labels, float) and np.isnan(labels)):
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
class PathRecord:
    parent_id: str
    q: int
    rho: float
    branch_policy: str
    posting_budget: int | None  # None for the silent branch
    counts: np.ndarray          # shape (H+1, 3), row t = Y_t
    checkpoint_hash: str


@dataclass
class Comparison:
    """All complete parent triplets for one (q, rho, schedule, budget)."""
    q: int
    rho: float
    schedule: str
    budget: int
    parent_ids: list[str]
    silent: np.ndarray    # (m, H+1, 3)
    controlled: dict[int, np.ndarray]  # target -> (m, H+1, 3)
    horizon: int = 10

    @property
    def key(self) -> str:
        return f"q{self.q}_rho{self.rho:.2f}_{self.schedule}_b{self.budget}"

    @property
    def m(self) -> int:
        return len(self.parent_ids)


@dataclass
class DatasetAudit:
    n_round_rows: int
    n_parents_total: int
    complete_paths: int
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    comparison_counts: dict[str, int] = field(default_factory=dict)
    per_setting_parents: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_round_rows": self.n_round_rows,
            "n_parents_total": self.n_parents_total,
            "complete_paths": self.complete_paths,
            "exclusions": self.exclusions,
            "comparison_counts": self.comparison_counts,
            "per_setting_parents": self.per_setting_parents,
        }


def build_paths(records: pd.DataFrame, horizon: int = 10) -> tuple[list[PathRecord], list[dict[str, Any]]]:
    """Rebuild every continuation path; return (complete paths, exclusion notes).

    Validation performed per path: horizons 1..H present exactly once, integer counts
    summing to 24, before-vector of round t equals after-vector of round t-1, and the
    branch's controller target agrees with the policy name.  A failing path is excluded
    with a reason rather than repaired.
    """
    cont = records[records["branch_policy"] != CHECKPOINT_POLICY].copy()
    cont["posting_budget"] = cont["posting_budget"].astype("float")
    paths: list[PathRecord] = []
    exclusions: list[dict[str, Any]] = []
    group_cols = ["parent_id", "branch_policy", "posting_budget"]
    for (parent_id, policy, budget), g in cont.groupby(group_cols, dropna=False, sort=True):
        note = {"parent_id": parent_id, "branch_policy": policy, "posting_budget": budget}
        try:
            if g["copy_id"].nunique(dropna=False) > 1:
                raise ValueError("more than one continuation copy; deduplicate first")
            hs = g["post_branch_horizon"].astype(int).to_numpy()
            if sorted(hs.tolist()) != list(range(1, horizon + 1)):
                raise ValueError(f"incomplete or duplicated horizons: {sorted(hs.tolist())}")
            g = g.sort_values("post_branch_horizon")
            if g["checkpoint_hash"].nunique() != 1:
                raise ValueError("checkpoint hash varies within a branch")
            if policy != SILENT_POLICY:
                expected_target = TARGET_OF_POLICY[policy]
                tgt = g["controller_target_semantic_id"].dropna().unique().tolist()
                if tgt and tgt != [f"ALLOCATION_{expected_target}"]:
                    raise ValueError(f"controller target {tgt} disagrees with policy {policy}")
            before = np.stack([labels_to_counts(v) for v in g["population_state_before"]])
            after = np.stack([labels_to_counts(v) for v in g["population_state_after"]])
            if not np.array_equal(before[1:], after[:-1]):
                raise ValueError("before/previous-after continuity violated")
            counts = np.vstack([before[:1], after])  # Y_0, Y_1, ..., Y_H
            paths.append(PathRecord(
                parent_id=parent_id, q=int(g["q"].iloc[0]), rho=float(g["rho"].iloc[0]),
                branch_policy=policy, posting_budget=None if pd.isna(budget) else int(budget),
                counts=counts, checkpoint_hash=str(g["checkpoint_hash"].iloc[0])))
        except ValueError as err:
            note["reason"] = str(err)
            exclusions.append(note)
    return paths, exclusions


def build_comparisons(paths: list[PathRecord], exclusions: list[dict[str, Any]],
                      horizon: int = 10) -> tuple[dict[str, Comparison], DatasetAudit]:
    """Form complete (silent, target-0, target-2) triplets for every comparison."""
    by_key: dict[tuple, PathRecord] = {}
    for p in paths:
        by_key[(p.parent_id, p.branch_policy, p.posting_budget)] = p
    parents = sorted({p.parent_id for p in paths})
    settings = sorted({(p.q, p.rho) for p in paths})
    budgets = sorted({p.posting_budget for p in paths if p.posting_budget is not None})
    comparisons: dict[str, Comparison] = {}
    audit = DatasetAudit(n_round_rows=0, n_parents_total=len(parents),
                         complete_paths=len(paths), exclusions=list(exclusions))
    for q, rho in settings:
        setting_parents = sorted({p.parent_id for p in paths if (p.q, p.rho) == (q, rho)})
        audit.per_setting_parents[f"q{q}_rho{rho:.2f}"] = len(setting_parents)
        for schedule in ("always", "sensing"):
            for b in budgets:
                ids, silent, ctl = [], [], {0: [], 2: []}
                for pid in setting_parents:
                    s = by_key.get((pid, SILENT_POLICY, None))
                    t0 = by_key.get((pid, f"{schedule}_truth", b))
                    t2 = by_key.get((pid, f"{schedule}_false", b))
                    if s is None or t0 is None or t2 is None:
                        audit.exclusions.append({
                            "parent_id": pid, "comparison": f"q{q}_rho{rho:.2f}_{schedule}_b{b}",
                            "reason": "missing " + ",".join(
                                n for n, v in (("silent", s), ("target0", t0), ("target2", t2)) if v is None)})
                        continue
                    y0 = (s.counts[0], t0.counts[0], t2.counts[0])
                    if not (np.array_equal(y0[0], y0[1]) and np.array_equal(y0[0], y0[2])):
                        raise ValueError(f"Y_0 differs across branches of parent {pid}")
                    if len({s.checkpoint_hash, t0.checkpoint_hash, t2.checkpoint_hash}) != 1:
                        raise ValueError(f"checkpoint hash differs across branches of parent {pid}")
                    ids.append(pid); silent.append(s.counts)
                    ctl[0].append(t0.counts); ctl[2].append(t2.counts)
                if not ids:
                    audit.comparison_counts[f"q{q}_rho{rho:.2f}_{schedule}_b{b}"] = 0
                    continue
                comp = Comparison(q=q, rho=rho, schedule=schedule, budget=b, parent_ids=ids,
                                  silent=np.stack(silent), horizon=horizon,
                                  controlled={z: np.stack(v) for z, v in ctl.items()})
                comparisons[comp.key] = comp
                audit.comparison_counts[comp.key] = comp.m
    return comparisons, audit


def load_comparisons(archive: Path, work_dir: Path, expected_sha256: str | None = None,
                     horizon: int = 10) -> tuple[dict[str, Comparison], DatasetAudit, str]:
    digest = verify_archive(archive, expected_sha256)
    parquet = extract_member(archive, ROUND_RECORDS_MEMBER, work_dir)
    records = load_round_records(parquet)
    paths, exclusions = build_paths(records, horizon=horizon)
    comparisons, audit = build_comparisons(paths, exclusions, horizon=horizon)
    audit.n_round_rows = int(len(records))
    return comparisons, audit, digest


def save_canonical(comparisons: dict[str, Comparison], out: Path) -> None:
    """Write canonical arrays as one long-format parquet table (one row per round)."""
    rows = []
    for comp in comparisons.values():
        for i, pid in enumerate(comp.parent_ids):
            for label, arr in (("silent", comp.silent), ("target0", comp.controlled[0]), ("target2", comp.controlled[2])):
                for t in range(arr.shape[1]):
                    rows.append({"comparison": comp.key, "q": comp.q, "rho": comp.rho, "schedule": comp.schedule,
                                 "budget": comp.budget, "parent_id": pid, "branch": label, "t": t,
                                 "N0": int(arr[i, t, 0]), "N1": int(arr[i, t, 1]), "N2": int(arr[i, t, 2])})
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
