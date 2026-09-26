"""Permutation nulls for MI and conditional MI."""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .measurements import STATISTICS, encoded, mutual_information_discrete, conditional_mutual_information_discrete, transitions


def _strata(z: np.ndarray) -> list[np.ndarray]:
    _, inverse = np.unique(np.atleast_2d(z).T if z.ndim == 1 else z, axis=0, return_inverse=True)
    return [np.flatnonzero(inverse == i) for i in range(inverse.max() + 1)]


def permutation_test(d: pd.DataFrame, bins: int, coordinate: str, n: int, seed: int,
                     preserve_conditioning_state: bool = True, progress: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    v = encoded(d, bins, coordinate)
    rng = np.random.default_rng(seed)
    definitions = [
        (STATISTICS[0], v["y"], v["x"], None),
        (STATISTICS[1], v["u"], v["xp"], v["x"]),
        (STATISTICS[2], v["u"], v["xp"], v["z"]),
    ]
    summaries, samples = [], []
    for name, action, outcome, condition in definitions:
        strata = _strata(condition) if condition is not None and preserve_conditioning_state else [np.arange(len(action))]
        supported = int(sum(len(ix) for ix in strata if len(np.unique(action[ix])) > 1))
        observed = (mutual_information_discrete(outcome, action) if condition is None else
                    conditional_mutual_information_discrete(action, outcome, condition))
        draws = []
        if condition is None or supported:
            for _ in tqdm(range(n), desc=f"permutations {name}", leave=False, disable=not progress):
                shuffled = action.copy()
                for ix in strata:
                    shuffled[ix] = rng.permutation(shuffled[ix])
                stat = (mutual_information_discrete(outcome, shuffled) if condition is None else
                        conditional_mutual_information_discrete(shuffled, outcome, condition))
                draws.append(stat)
        mean = float(np.mean(draws)) if draws else float("nan")
        std = float(np.std(draws, ddof=1)) if len(draws) > 1 else float("nan")
        summaries.append({"statistic": name, "observed": observed, "null_mean": mean, "null_std": std,
                          "null_q025": float(np.quantile(draws, .025)) if draws else float("nan"),
                          "null_q975": float(np.quantile(draws, .975)) if draws else float("nan"),
                          "p_value": (1 + sum(x >= observed for x in draws)) / (len(draws) + 1) if draws else float("nan"),
                          "observed_minus_null": observed - mean,
                          "z_score": (observed - mean) / std if std > 0 else float("nan"),
                          "n_samples": len(d), "n_supported": supported if condition is not None else len(d),
                          "n_permutations": len(draws), "preserve_conditioning_state": preserve_conditioning_state})
        samples.extend({"statistic": name, "draw": i, "value": value} for i, value in enumerate(draws))
    return pd.DataFrame(summaries), pd.DataFrame(samples)


def analyze(rounds: pd.DataFrame, bins: int, coordinate: str, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, samples = [], []
    for cell, d in tqdm(transitions(rounds).groupby("cell_id"), desc="null cells"):
        summary, draws = permutation_test(d, bins, coordinate, int(settings.get("n_permutations", 1000)),
                                          int(settings.get("seed", 0)) + int(cell),
                                          bool(settings.get("preserve_conditioning_state", True)))
        summary.insert(0, "cell_id", cell)
        draws.insert(0, "cell_id", cell)
        summaries.append(summary)
        samples.append(draws)
    return pd.concat(summaries, ignore_index=True), pd.concat(samples, ignore_index=True)
