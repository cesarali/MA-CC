"""Episode cluster bootstrap over saved trajectories."""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .measurements import STATISTICS, statistics, susceptibility, transitions


def metrics(rounds: pd.DataFrame, bins: int, coordinate: str) -> dict[str, float]:
    if rounds.empty:
        return {}
    finals = rounds.loc[rounds.groupby("seed")["round"].idxmax()]
    d = transitions(rounds)
    chi, _ = susceptibility(d, bins, coordinate)
    return {"final_truth_share": float(finals.truth_share.mean()),
            "final_target_share": float(finals.target_share.mean()),
            "susceptibility": chi, **statistics(d, bins, coordinate)}


def bootstrap_cell(rounds: pd.DataFrame, bins: int, coordinate: str, n: int, confidence: float,
                   seed: int, progress: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = {seed: d.copy() for seed, d in rounds.groupby("seed")}
    keys = list(groups)
    observed = metrics(rounds, bins, coordinate)
    draws = {key: [] for key in observed}
    for _ in tqdm(range(n), desc="bootstrap episodes", leave=False, disable=not progress):
        sampled = rng.integers(0, len(keys), size=len(keys))
        parts = []
        for i, index in enumerate(sampled):
            part = groups[keys[int(index)]].copy()
            part["seed"] = i  # distinguish repeated sampled episodes
            parts.append(part)
        values = metrics(pd.concat(parts, ignore_index=True), bins, coordinate)
        for key, value in values.items():
            draws[key].append(value)
    alpha = (1 - confidence) / 2
    return pd.DataFrame([{"metric": key, "observed": value,
                          "ci_low": float(np.quantile(finite, alpha)) if len(finite := np.asarray(draws[key])[np.isfinite(draws[key])]) else float("nan"),
                          "ci_high": float(np.quantile(finite, 1 - alpha)) if len(finite) else float("nan"), "confidence_level": confidence,
                          "unit": "episode", "n_bootstrap": n, "n_episodes": len(keys)}
                         for key, value in observed.items()])


def analyze(rounds: pd.DataFrame, bins: int, coordinate: str, settings: dict) -> pd.DataFrame:
    frames = []
    for cell, d in tqdm(rounds.groupby("cell_id"), desc="bootstrap cells"):
        out = bootstrap_cell(d, bins, coordinate, int(settings.get("n_bootstrap", 1000)),
                             float(settings.get("confidence_level", 0.95)), int(settings.get("seed", 0)) + int(cell))
        out.insert(0, "cell_id", cell)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)
