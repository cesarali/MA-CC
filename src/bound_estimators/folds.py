"""Parent-grouped cross-fitting folds.

The parent is the only independent unit.  Every branch, horizon, and bootstrap copy
of a parent stays in one fold, so a model is never scored on a parent it has seen.
Folds are derived from one master seed and saved before any fitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class FoldPlan:
    master_seed: int
    repeat: int
    n_outer: int
    n_inner: int
    parent_ids: list[str]
    outer: list[list[int]]              # outer[k] = test parent indices of fold k
    inner: list[list[list[int]]]        # inner[k][j] = validation indices (within outer-train) of inner fold j

    def outer_split(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        test = np.array(self.outer[k], dtype=int)
        train = np.array(sorted(set(range(len(self.parent_ids))) - set(test.tolist())), dtype=int)
        return train, test

    def inner_split(self, k: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        train, _ = self.outer_split(k)
        val = np.array(self.inner[k][j], dtype=int)
        fit = np.array(sorted(set(train.tolist()) - set(val.tolist())), dtype=int)
        return fit, val

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))


def derive_seed(master_seed: int, *parts: object) -> int:
    """Deterministic child seed from the master seed and a list of tags."""
    import hashlib
    text = f"{master_seed}|" + "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def make_fold_plan(parent_ids: list[str], master_seed: int, repeat: int = 0,
                   n_outer: int = 5, n_inner: int = 3, tag: str = "") -> FoldPlan:
    rng = np.random.default_rng(derive_seed(master_seed, "folds", tag, repeat))
    idx = np.arange(len(parent_ids))
    rng.shuffle(idx)
    outer = [sorted(chunk.tolist()) for chunk in np.array_split(idx, n_outer)]
    inner: list[list[list[int]]] = []
    for k in range(n_outer):
        train = np.array(sorted(set(idx.tolist()) - set(outer[k])))
        rng.shuffle(train)
        inner.append([sorted(chunk.tolist()) for chunk in np.array_split(train, n_inner)])
    return FoldPlan(master_seed=master_seed, repeat=repeat, n_outer=n_outer, n_inner=n_inner,
                    parent_ids=list(parent_ids), outer=outer, inner=inner)
