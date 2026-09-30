"""Parent-grouped nested cross-fitting with one-standard-error model selection.

A ``Dataset`` holds examples (endpoints or paths) with binary labels, whole-parent
weights and the parent index of every example.  ``Candidate`` describes one model
family with its regularization and width.  ``crossfit`` returns out-of-fold logits
for the selected model in each outer fold (and, for sensitivity, for every candidate)
so the caller can turn them into information scores or NWJ / DV cost scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from statistics import median
from typing import Any

import numpy as np

from . import models as M
from .folds import FoldPlan


@dataclass(frozen=True)
class Candidate:
    family: str                 # constant | linear | quadratic | mlp | flat | gru | gru_mean | gru_direct
    lam: float = 0.0            # L2 strength (lambda for logistic, weight decay for torch)
    width: int = 0              # hidden width / GRU hidden size
    history: int | None = None  # memory comparison: keep Y_0 + last `history` rounds
    cap: float | None = None    # critic cap for direct NWJ training

    @property
    def name(self) -> str:
        parts = [self.family]
        if self.width:
            parts.append(f"w{self.width}")
        if self.family not in ("constant",):
            parts.append(f"l{self.lam:g}")
        if self.history is not None:
            parts.append(f"h{self.history}")
        if self.cap is not None:
            parts.append(f"cap{self.cap:g}")
        return "-".join(parts)

    @property
    def is_torch(self) -> bool:
        return self.family in ("mlp", "flat", "gru", "gru_mean", "gru_direct")

    @property
    def objective(self) -> str:
        return "nwj" if self.family == "gru_direct" else "bce"

    @property
    def complexity(self) -> tuple:
        rank = {"constant": 0, "linear": 1, "quadratic": 2, "mlp": 3, "flat": 3,
                "gru": 5, "gru_mean": 5, "gru_direct": 7}[self.family]
        # simpler first: lower rank, smaller width, larger regularization
        return (rank, self.width, -self.lam)


@dataclass
class Dataset:
    kind: str                     # "endpoint" | "path"
    X: np.ndarray                 # endpoint: (n, 3) counts; path: (n, T, 3) counts
    y: np.ndarray                 # binary label, 1 = class P (target 0 / controlled)
    w: np.ndarray                 # whole-parent weights
    parent: np.ndarray            # parent index per example
    horizon: int

    def subset(self, parent_indices: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.isin(self.parent, parent_indices))


def features(ds: Dataset, cand: Candidate) -> np.ndarray:
    if cand.family == "constant":
        return np.zeros((len(ds.y), 1))
    if ds.kind == "endpoint":
        if cand.family in ("linear", "quadratic"):
            return M.endpoint_features(ds.X, cand.family)
        if cand.family == "mlp":
            return M.endpoint_features(ds.X, "linear")
        raise ValueError(f"{cand.family} is not an endpoint candidate")
    # path
    if cand.family in ("linear", "quadratic"):
        return M.path_summary_features(ds.X, cand.family)
    keep = None if cand.history is None else M.memory_indices(ds.horizon, cand.history)
    return M.path_sequence_input(ds.X, keep=keep)


def _factory(ds: Dataset, cand: Candidate, X: np.ndarray):
    if cand.family == "mlp":
        return lambda: M.EndpointMLP(n_in=X.shape[1], width=cand.width)
    if cand.family == "flat":
        return lambda: M.FlatPathMLP(n_steps=X.shape[1], n_feat=X.shape[2], width=cand.width)
    if cand.family in ("gru", "gru_direct"):
        return lambda: M.PathGRU(n_feat=X.shape[2], hidden=cand.width, pool="last",
                                 y0_readout=cand.history is not None)
    if cand.family == "gru_mean":
        return lambda: M.PathGRU(n_feat=X.shape[2], hidden=cand.width, pool="mean",
                                 y0_readout=cand.history is not None)
    raise ValueError(cand.family)


class _Standardized:
    """Wrap a torch ensemble with a training-data standardizer on the last axis."""

    def __init__(self, scaler: M.Standardizer, inner):
        self.scaler, self.inner = scaler, inner
        self.n_params = inner.n_params
        self.best_epochs = inner.best_epochs
        self.curves = inner.curves

    def _t(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.scaler.mean) / self.scaler.std

    def logit(self, X):
        return self.inner.logit(self._t(X))

    def prob(self, X):
        return self.inner.prob(self._t(X))


def fit_candidate(ds: Dataset, cand: Candidate, train: np.ndarray, cfg: M.TrainConfig,
                  val: np.ndarray | None = None, fixed_epochs: int | None = None):
    X = features(ds, cand)
    if cand.family == "constant":
        return M.ConstantModel()
    if not cand.is_torch:
        return M.LogisticModel(lam=cand.lam).fit(X[train], ds.y[train], ds.w[train])
    flat = X.reshape(X.shape[0], -1) if X.ndim == 3 else X
    scaler = M.Standardizer.fit(flat[train])
    Xs = ((flat - scaler.mean) / scaler.std).reshape(X.shape)
    tcfg = M.TrainConfig(**{**asdict(cfg), "l2": cand.lam, "critic_cap": cand.cap})
    ens = M.train_ensemble(_factory(ds, cand, X), Xs[train], ds.y[train], ds.w[train], cand.objective, tcfg,
                           X_val=None if val is None else Xs[val], y_val=None if val is None else ds.y[val],
                           w_val=None if val is None else ds.w[val], fixed_epochs=fixed_epochs)
    scaler3 = M.Standardizer(mean=scaler.mean.reshape(X.shape[1:]) if X.ndim == 3 else scaler.mean,
                             std=scaler.std.reshape(X.shape[1:]) if X.ndim == 3 else scaler.std)
    return _Standardized(scaler3, ens)


def selection_loss(model, ds: Dataset, idx: np.ndarray, cand: Candidate, objective: str) -> float:
    """Inner-validation objective to minimize: weighted BCE for classifiers, -NWJ for critics."""
    X = features(ds, cand)
    f = model.logit(X[idx])
    y, w = ds.y[idx], ds.w[idx]
    if objective == "nwj":
        wp, wr = w * y, w * (1 - y)
        return float(-((f * wp).sum() / wp.sum() - (np.exp(f) * wr).sum() / wr.sum() + 1.0))
    ce = np.logaddexp(0.0, f) - y * f
    return float(np.sum(w * ce) / np.sum(w))


@dataclass
class FoldResult:
    fold: int
    selected: str
    inner_mean: dict[str, float]
    inner_se: dict[str, float]
    refit_epochs: dict[str, int | None]
    n_params: dict[str, int]
    test_parents: list[int]
    curves: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossfitResult:
    candidates: list[str]
    oof_logit: dict[str, np.ndarray]      # candidate name -> out-of-fold logits (n examples)
    selected_logit: np.ndarray            # logits from the fold-selected candidate
    selected_by_fold: list[str]
    folds: list[FoldResult]

    def summary(self) -> dict[str, Any]:
        return {"candidates": self.candidates, "selected_by_fold": self.selected_by_fold,
                "folds": [{k: v for k, v in asdict(f).items() if k != "curves"} for f in self.folds]}


def one_se_select(inner_mean: dict[str, float], inner_se: dict[str, float],
                  cands: list[Candidate]) -> str:
    best = min(inner_mean, key=inner_mean.get)
    threshold = inner_mean[best] + inner_se[best]
    eligible = [c for c in cands if inner_mean[c.name] <= threshold]
    return min(eligible, key=lambda c: c.complexity).name


def crossfit(ds: Dataset, cands: list[Candidate], plan: FoldPlan, cfg: M.TrainConfig,
             selection_objective: str = "bce", keep_curves: bool = False) -> CrossfitResult:
    n = len(ds.y)
    oof = {c.name: np.full(n, np.nan) for c in cands}
    sel = np.full(n, np.nan)
    folds: list[FoldResult] = []
    for k in range(plan.n_outer):
        train_p, test_p = plan.outer_split(k)
        train_i, test_i = ds.subset(train_p), ds.subset(test_p)
        inner_losses: dict[str, list[float]] = {c.name: [] for c in cands}
        inner_epochs: dict[str, list[int]] = {c.name: [] for c in cands}
        for j in range(plan.n_inner):
            fit_p, val_p = plan.inner_split(k, j)
            fit_i, val_i = ds.subset(fit_p), ds.subset(val_p)
            for c in cands:
                obj = "nwj" if c.objective == "nwj" else selection_objective
                model = fit_candidate(ds, c, fit_i, cfg, val=val_i)
                inner_losses[c.name].append(selection_loss(model, ds, val_i, c, obj))
                if c.is_torch:
                    inner_epochs[c.name].extend(model.best_epochs)
        inner_mean = {k_: float(np.mean(v)) for k_, v in inner_losses.items()}
        inner_se = {k_: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
                    for k_, v in inner_losses.items()}
        chosen = one_se_select(inner_mean, inner_se, cands)
        refit_epochs: dict[str, int | None] = {}
        n_params: dict[str, int] = {}
        curves: dict[str, Any] = {}
        for c in cands:
            ep = int(median(inner_epochs[c.name])) if c.is_torch else None
            refit_epochs[c.name] = ep
            model = fit_candidate(ds, c, train_i, cfg, fixed_epochs=ep)
            n_params[c.name] = int(model.n_params)
            oof[c.name][test_i] = model.logit(features(ds, c)[test_i])
            if keep_curves and c.is_torch:
                curves[c.name] = model.curves
        sel[test_i] = oof[chosen][test_i]
        folds.append(FoldResult(fold=k, selected=chosen, inner_mean=inner_mean, inner_se=inner_se,
                                refit_epochs=refit_epochs, n_params=n_params,
                                test_parents=test_p.tolist(), curves=curves))
    return CrossfitResult(candidates=[c.name for c in cands], oof_logit=oof, selected_logit=sel,
                          selected_by_fold=[f.selected for f in folds], folds=folds)
