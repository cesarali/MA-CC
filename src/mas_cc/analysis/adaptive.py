"""Adaptive stopping for bootstrap draws (opt-in; off by default).

A bootstrap interval is a pair of order statistics of the draws made so far.
When the draws are generated one at a time from a fixed seed, the sequence is
prefix-stable: the first ``n`` draws are the same whatever the requested
total. :class:`IntervalStopper` watches that prefix and stops once both
interval endpoints have moved by less than ``tolerance`` times the current
width over the last ``check_every`` draws, never before ``min_resamples``.

Enabled from the analysis recipe::

    resampling:
      bootstrap_resamples: 1000
      adaptive: {enabled: true, min_resamples: 200, check_every: 100, tolerance: 0.02}

With ``enabled: false`` (or the key absent) nothing changes: the settings
mapping carries no ``adaptive`` entry, hashes and tables are byte-identical to
a build without this module. With it enabled the realised number of draws is
written to the tables' ``bootstrap_resamples`` column, so a reader can tell
which intervals stopped early. Null permutations are never adaptive: their
purpose is a tail count, and the stop rule here is written for intervals.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AdaptiveResampling:
    enabled: bool = False
    min_resamples: int = 200
    check_every: int = 100
    tolerance: float = 0.02

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "AdaptiveResampling":
        if not raw:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("resampling.adaptive must be a mapping")
        config = cls(
            enabled=bool(raw.get("enabled", False)),
            min_resamples=int(raw.get("min_resamples", 200)),
            check_every=int(raw.get("check_every", 100)),
            tolerance=float(raw.get("tolerance", 0.02)),
        )
        if config.min_resamples < 2 or config.check_every < 1 or not 0 < config.tolerance < 1:
            raise ValueError("resampling.adaptive needs min_resamples >= 2, check_every >= 1, 0 < tolerance < 1")
        return config

    def as_settings(self) -> dict[str, Any]:
        return asdict(self)


def coerce(value: "AdaptiveResampling | Mapping[str, Any] | None") -> AdaptiveResampling | None:
    """``None`` or a disabled config -> ``None`` (the byte-identical path); else the config."""
    config = value if isinstance(value, AdaptiveResampling) else AdaptiveResampling.from_mapping(value)
    return config if config.enabled else None


class IntervalStopper:
    """Decide, after each draw, whether the interval of the finite draws has settled."""

    def __init__(self, config: AdaptiveResampling, alpha: float) -> None:
        self.config = config
        self.alpha = alpha
        self.checks = 0
        self.previous: tuple[float, float] | None = None
        self.stopped_at: int | None = None

    def should_stop(self, draws_made: int, finite_values: Sequence[float]) -> bool:
        config = self.config
        if draws_made < config.min_resamples or (draws_made - config.min_resamples) % config.check_every:
            return False
        self.checks += 1
        if len(finite_values) < 2:
            return False
        low = float(np.quantile(finite_values, self.alpha))
        high = float(np.quantile(finite_values, 1.0 - self.alpha))
        current = (low, high)
        previous, self.previous = self.previous, current
        if previous is None:
            return False
        width = high - low
        if not math.isfinite(width):
            return False
        moved = (abs(low - previous[0]), abs(high - previous[1]))
        allowed = config.tolerance * width
        stable = (moved[0] <= allowed and moved[1] <= allowed) if width > 0 else moved == (0.0, 0.0)
        if stable:
            self.stopped_at = draws_made
        return stable


__all__ = ["AdaptiveResampling", "IntervalStopper", "coerce"]
