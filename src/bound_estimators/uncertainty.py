"""Uncertainty: paired parent bootstrap of saved contributions, whole-parent label swaps.

The bootstrap here resamples *saved held-out per-parent contributions*.  It is the
cheaper "conditional on the fitted predictions" interval the specification allows as
a separate report; it omits training variability.  Numerator and denominator use the
same parent resamples so their covariance is retained.  A full-refit bootstrap is
available through ``run.py --full-bootstrap`` and is expensive.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .data import Comparison


def percentile_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    lo = (1 - confidence) / 2
    return float(np.quantile(values, lo)), float(np.quantile(values, 1 - lo))


def paired_bootstrap(per_parent: dict[str, np.ndarray], n_resamples: int, seed: int,
                     confidence: float = 0.95) -> dict[str, Any]:
    """Resample parents once per replicate; average every contribution array on the same draw.

    ``per_parent`` maps names to arrays of length m (one entry per parent).  The special
    pair (``info``, ``cost``) additionally produces the ratio ``eta = info / cost`` only
    on draws where the cost is positive; the fraction of non-positive cost draws is
    reported so a ratio can be suppressed as unstable.
    """
    names = list(per_parent)
    m = len(per_parent[names[0]])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_resamples, m))
    draws = {n: per_parent[n][idx].mean(axis=1) for n in names}
    out: dict[str, Any] = {}
    for n in names:
        lo, hi = percentile_interval(draws[n], confidence)
        out[n] = {"point": float(per_parent[n].mean()), "lo": lo, "hi": hi, "sd": float(draws[n].std(ddof=1))}
    if "info" in draws and "cost" in draws:
        pos = draws["cost"] > 0
        out["cost_nonpositive_fraction"] = float(1 - pos.mean())
        if pos.all():
            eta = draws["info"] / draws["cost"]
            lo, hi = percentile_interval(eta, confidence)
            out["eta"] = {"point": float(per_parent["info"].mean() / per_parent["cost"].mean()),
                          "lo": lo, "hi": hi, "sd": float(eta.std(ddof=1)), "supported": True}
        else:
            out["eta"] = {"supported": False}
    return out


def swap_targets(comp: Comparison, rng: np.random.Generator) -> Comparison:
    """Exchange both controlled target paths within a random half of the parents."""
    flip = rng.random(comp.m) < 0.5
    c0, c2 = comp.controlled[0].copy(), comp.controlled[2].copy()
    c0[flip], c2[flip] = comp.controlled[2][flip], comp.controlled[0][flip]
    return replace(comp, controlled={0: c0, 2: c2})


def swap_branch(comp: Comparison, target: int, rng: np.random.Generator) -> Comparison:
    """Exchange the controlled (target) and silent path within a random half of the parents."""
    flip = rng.random(comp.m) < 0.5
    ctl, sil = comp.controlled[target].copy(), comp.silent.copy()
    ctl[flip], sil[flip] = comp.silent[flip], comp.controlled[target][flip]
    ctl_all = dict(comp.controlled); ctl_all[target] = ctl
    return replace(comp, silent=sil, controlled=ctl_all)


def null_summary(null_scores: np.ndarray, observed: float) -> dict[str, float]:
    null_scores = np.asarray(null_scores, dtype=np.float64)
    return {
        "observed": float(observed),
        "null_mean": float(null_scores.mean()),
        "null_sd": float(null_scores.std(ddof=1)) if len(null_scores) > 1 else float("nan"),
        "null_q95": float(np.quantile(null_scores, 0.95)),
        "null_max": float(null_scores.max()),
        "p_value": float((1 + np.sum(null_scores >= observed)) / (1 + len(null_scores))),
        "n_swaps": int(len(null_scores)),
    }
