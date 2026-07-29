"""Direct-counting estimators for finite discrete channels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Hashable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Estimate:
    jeffreys: float
    unsmoothed: float
    miller_madow: float
    observations: int


def _entropy(counts: np.ndarray, *, miller_madow: bool = False) -> float:
    flat = counts.ravel().astype(float)
    positive = flat[flat > 0]
    total = positive.sum()
    if total <= 0:
        return math.nan
    probabilities = positive / total
    result = float(-(probabilities * np.log2(probabilities)).sum())
    if miller_madow:
        result += (len(positive) - 1) / (2 * total * math.log(2))
    return result


def _mi_from_counts(counts: np.ndarray, *, miller_madow: bool = False) -> float:
    return (
        _entropy(counts.sum(axis=1), miller_madow=miller_madow)
        + _entropy(counts.sum(axis=0), miller_madow=miller_madow)
        - _entropy(counts, miller_madow=miller_madow)
    )


def mutual_information(
    x: Sequence[Hashable],
    y: Sequence[Hashable],
    *,
    x_levels: Iterable[Hashable] | None = None,
    y_levels: Iterable[Hashable] | None = None,
) -> Estimate:
    if len(x) != len(y):
        raise ValueError("x and y must have equal length.")
    xs = tuple(dict.fromkeys(x_levels if x_levels is not None else x))
    ys = tuple(dict.fromkeys(y_levels if y_levels is not None else y))
    if not xs or not ys:
        return Estimate(math.nan, math.nan, math.nan, len(x))
    x_index, y_index = {value: i for i, value in enumerate(xs)}, {value: i for i, value in enumerate(ys)}
    counts = np.zeros((len(xs), len(ys)), dtype=float)
    for left, right in zip(x, y, strict=True):
        if left in x_index and right in y_index:
            counts[x_index[left], y_index[right]] += 1
    return mutual_information_from_counts(counts)


def mutual_information_from_counts(counts: np.ndarray) -> Estimate:
    """Estimate MI from a complete X-by-Y contingency table."""

    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or np.any(counts < 0):
        raise ValueError("MI counts must be a nonnegative two-dimensional table.")
    if counts.sum() == 0:
        return Estimate(math.nan, math.nan, math.nan, 0)
    return Estimate(
        jeffreys=_mi_from_counts(counts + 0.5),
        unsmoothed=_mi_from_counts(counts),
        miller_madow=_mi_from_counts(counts, miller_madow=True),
        observations=int(counts.sum()),
    )


def _cmi_from_counts(counts: np.ndarray, *, miller_madow: bool = False) -> float:
    # Axes are X, Z, Y. I(X;Y|Z)=H(X,Z)+H(Y,Z)-H(Z)-H(X,Y,Z).
    return (
        _entropy(counts.sum(axis=2), miller_madow=miller_madow)
        + _entropy(counts.sum(axis=0).T, miller_madow=miller_madow)
        - _entropy(counts.sum(axis=(0, 2)), miller_madow=miller_madow)
        - _entropy(counts, miller_madow=miller_madow)
    )


def conditional_mutual_information(
    x: Sequence[Hashable],
    y: Sequence[Hashable],
    z: Sequence[Hashable],
    *,
    x_levels: Iterable[Hashable] | None = None,
    y_levels: Iterable[Hashable] | None = None,
    z_levels: Iterable[Hashable] | None = None,
) -> Estimate:
    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y, and z must have equal length.")
    xs = tuple(dict.fromkeys(x_levels if x_levels is not None else x))
    ys = tuple(dict.fromkeys(y_levels if y_levels is not None else y))
    zs = tuple(dict.fromkeys(z_levels if z_levels is not None else z))
    if not xs or not ys or not zs:
        return Estimate(math.nan, math.nan, math.nan, len(x))
    xi, yi, zi = ({value: index for index, value in enumerate(levels)} for levels in (xs, ys, zs))
    counts = np.zeros((len(xs), len(zs), len(ys)), dtype=float)
    for left, future, current in zip(x, y, z, strict=True):
        if left in xi and future in yi and current in zi:
            counts[xi[left], zi[current], yi[future]] += 1
    return conditional_mutual_information_from_counts(counts)


def conditional_mutual_information_from_counts(counts: np.ndarray) -> Estimate:
    """Estimate conditional MI from a complete X-by-Z-by-Y table."""

    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 3 or np.any(counts < 0):
        raise ValueError("Conditional-MI counts must be a nonnegative three-dimensional table.")
    if counts.sum() == 0:
        return Estimate(math.nan, math.nan, math.nan, 0)
    return Estimate(
        jeffreys=_cmi_from_counts(counts + 0.5),
        unsmoothed=_cmi_from_counts(counts),
        miller_madow=_cmi_from_counts(counts, miller_madow=True),
        observations=int(counts.sum()),
    )


def complete_cells(*levels: Iterable[Hashable]) -> tuple[tuple[Hashable, ...], ...]:
    """Expose the Cartesian cell contract for validation and downstream reuse."""

    return tuple(product(*(tuple(level) for level in levels)))


__all__ = [
    "Estimate",
    "complete_cells",
    "conditional_mutual_information",
    "conditional_mutual_information_from_counts",
    "mutual_information",
    "mutual_information_from_counts",
]
