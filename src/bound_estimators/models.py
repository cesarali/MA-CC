"""Model families and training loops.

* ``LogisticModel``: regularized logistic regression with the explicit convention
  ``mean weighted cross-entropy + lambda * ||beta||^2 / 2`` (intercept unpenalized),
  fitted to numerical convergence with L-BFGS.  Standardization is fitted on the
  training data only.
* Torch models: endpoint MLP (2 -> tanh(width) -> 1), flattened-path MLP, and a
  one-layer unidirectional GRU with a scalar readout.  Trained with Adam, lr 1e-3,
  full batch, gradient-norm clipping at 1, L2 on weights only (biases free), early
  stopping on an inner-validation objective evaluated every 10 epochs.

Every model exposes ``logit(X) -> np.ndarray`` (float64).  For classifiers the logit is
log-odds of class 1; for direct variational critics it is the (capped) critic value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
from scipy.optimize import minimize

torch.set_num_threads(1)


# ------------------------------------------------------------------ feature builders
def endpoint_features(counts: np.ndarray, kind: str) -> np.ndarray:
    """counts (n, 3) -> features.  linear: [u, v]; quadratic: [u, v, u^2, uv, v^2]."""
    c = np.asarray(counts, dtype=np.float64)
    u = c[:, 0] / c.sum(axis=1)
    v = c[:, 2] / c.sum(axis=1)
    if kind == "linear":
        return np.column_stack([u, v])
    if kind == "quadratic":
        return np.column_stack([u, v, u * u, u * v, v * v])
    raise ValueError(kind)


def path_fractions(paths: np.ndarray) -> np.ndarray:
    """paths (n, T, 3) counts -> (n, T, 2) fractions of allocations 0 and 2."""
    p = np.asarray(paths, dtype=np.float64)
    tot = p.sum(axis=2, keepdims=True)
    return np.stack([p[:, :, 0] / tot[:, :, 0], p[:, :, 2] / tot[:, :, 0]], axis=2)


def path_summary_features(paths: np.ndarray, kind: str) -> np.ndarray:
    """Fixed summaries: initial, endpoint and mean-over-rounds-1..h fractions (6 numbers)."""
    fr = path_fractions(paths)
    base = np.column_stack([fr[:, 0, :], fr[:, -1, :], fr[:, 1:, :].mean(axis=1)])
    if kind == "linear":
        return base
    if kind == "quadratic":
        n, d = base.shape
        quad = [base[:, i] * base[:, j] for i in range(d) for j in range(i, d)]
        return np.column_stack([base] + quad)
    raise ValueError(kind)


def path_sequence_input(paths: np.ndarray, horizon_norm: float = 10.0,
                        keep: Sequence[int] | None = None) -> np.ndarray:
    """(n, T, 3) counts -> (n, T', 3) [u_t, v_t, t/10] restricted to the kept round indices."""
    fr = path_fractions(paths)
    T = fr.shape[1]
    idx = list(range(T)) if keep is None else sorted(set(int(k) for k in keep))
    t = np.asarray(idx, dtype=np.float64) / horizon_norm
    out = np.concatenate([fr[:, idx, :], np.broadcast_to(t[None, :, None], (fr.shape[0], len(idx), 1))], axis=2)
    return out


def memory_indices(horizon: int, history: int) -> list[int]:
    """Y_0 plus the last ``history`` observed vote vectors (duplicates removed)."""
    tail = list(range(max(1, horizon - history + 1), horizon + 1))
    return sorted(set([0] + tail))


# ------------------------------------------------------------------ logistic regression
@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        return cls(mean=mean, std=std)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.std


@dataclass
class LogisticModel:
    lam: float
    beta: np.ndarray | None = None
    intercept: float = 0.0
    scaler: Standardizer | None = None
    converged: bool = False

    @property
    def n_params(self) -> int:
        return 0 if self.beta is None else int(self.beta.size + 1)

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> "LogisticModel":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
        w = w / w.sum()
        self.scaler = Standardizer.fit(X)
        Xs = self.scaler(X)
        d = Xs.shape[1]

        def obj(theta):
            b0, b = theta[0], theta[1:]
            s = b0 + Xs @ b
            # mean weighted CE via stable log(1+exp)
            ce = np.sum(w * (np.logaddexp(0.0, s) - y * s))
            reg = 0.5 * self.lam * b @ b
            p = 1.0 / (1.0 + np.exp(-s))
            g = w * (p - y)
            grad = np.concatenate([[g.sum()], Xs.T @ g + self.lam * b])
            return ce + reg, grad

        res = minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                       options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-9})
        self.intercept, self.beta = float(res.x[0]), res.x[1:].copy()
        self.converged = bool(res.success)
        return self

    def logit(self, X: np.ndarray) -> np.ndarray:
        return self.intercept + self.scaler(X) @ self.beta

    def prob(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(X)))


class ConstantModel:
    """Logit zero everywhere: probability 1/2 for classifiers, zero critic for NWJ."""
    n_params = 0

    def fit(self, *args, **kwargs):
        return self

    def logit(self, X) -> np.ndarray:
        n = len(X)
        return np.zeros(n, dtype=np.float64)

    def prob(self, X) -> np.ndarray:
        return np.full(len(X), 0.5)


# ------------------------------------------------------------------ torch models
class EndpointMLP(torch.nn.Module):
    def __init__(self, n_in: int = 2, width: int = 4):
        super().__init__()
        self.hidden = torch.nn.Linear(n_in, width)
        self.out = torch.nn.Linear(width, 1)

    def forward(self, x):  # x (n, n_in)
        return self.out(torch.tanh(self.hidden(x))).squeeze(-1)


class FlatPathMLP(torch.nn.Module):
    def __init__(self, n_steps: int, n_feat: int = 3, width: int = 4):
        super().__init__()
        self.hidden = torch.nn.Linear(n_steps * n_feat, width)
        self.out = torch.nn.Linear(width, 1)

    def forward(self, x):  # x (n, T, F), order preserved by flattening
        return self.out(torch.tanh(self.hidden(x.reshape(x.shape[0], -1)))).squeeze(-1)


class PathGRU(torch.nn.Module):
    """One-layer unidirectional GRU, zero initial state, scalar linear readout.

    ``pool='last'`` reads the final hidden state; ``pool='mean'`` averages hidden states
    (the RNEEP-style comparison).  With ``y0_readout=True`` the first row of the input
    (Y_0) bypasses the GRU and feeds the readout directly, and the GRU sees only the
    remaining ordered tail (memory-length comparison).
    """

    def __init__(self, n_feat: int = 3, hidden: int = 4, pool: str = "last", y0_readout: bool = False):
        super().__init__()
        self.gru = torch.nn.GRU(n_feat, hidden, batch_first=True)
        self.pool = pool
        self.y0_readout = y0_readout
        self.out = torch.nn.Linear(hidden + (n_feat if y0_readout else 0), 1)

    def forward(self, x):  # x (n, T, F)
        if self.y0_readout:
            y0, seq = x[:, 0, :], x[:, 1:, :]
        else:
            y0, seq = None, x
        hs, h_last = self.gru(seq)
        h = hs.mean(dim=1) if self.pool == "mean" else h_last[0]
        if y0 is not None:
            h = torch.cat([h, y0], dim=1)
        return self.out(h).squeeze(-1)


def n_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# ------------------------------------------------------------------ objectives on tensors
def _weighted_mean(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (v * w).sum() / w.sum()


def bce_objective(logit: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted mean binary cross-entropy (to minimize)."""
    return _weighted_mean(torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none"), w)


def nwj_objective(critic: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Negative NWJ score (to minimize): -(E_P f - E_R e^f + 1), y=1 marks P (controlled)."""
    wp, wr = w * y, w * (1 - y)
    ep = (critic * wp).sum() / wp.sum()
    er = (torch.exp(critic) * wr).sum() / wr.sum()
    return -(ep - er + 1.0)


@dataclass
class TrainConfig:
    lr: float = 1e-3
    max_epochs: int = 1000
    l2: float = 0.01
    clip_norm: float = 1.0
    eval_every: int = 10
    patience: int = 10
    min_delta: float = 1e-4
    seeds: tuple[int, ...] = (0, 1, 2)
    critic_cap: float | None = None   # for objective="nwj": f = cap * tanh(s / cap)


@dataclass
class TorchEnsemble:
    """Several seeds of the same architecture; predictions are averaged."""
    members: list[torch.nn.Module]
    objective: str
    critic_cap: float | None
    best_epochs: list[int] = field(default_factory=list)
    curves: list[dict[str, list[float]]] = field(default_factory=list)

    @property
    def n_params(self) -> int:
        return n_parameters(self.members[0])

    def _raw(self, X: np.ndarray) -> np.ndarray:
        xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            outs = [m(xt).double().numpy() for m in self.members]
        return np.stack(outs)

    def logit(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw(X)
        if self.objective == "nwj":
            f = raw if self.critic_cap is None else self.critic_cap * np.tanh(raw / self.critic_cap)
            return f.mean(axis=0)               # average critic outputs
        p = 1.0 / (1.0 + np.exp(-raw))
        pm = p.mean(axis=0)                     # average probabilities, return matching logit
        pm = np.clip(pm, 1e-12, 1 - 1e-12)
        return np.log(pm) - np.log1p(-pm)

    def prob(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(X)))


def _l2_weights(model: torch.nn.Module) -> torch.Tensor:
    return sum((p ** 2).sum() for n, p in model.named_parameters() if not n.endswith("bias"))


def _objective_value(model, X, y, w, objective, cap):
    out = model(X)
    if objective == "nwj":
        f = out if cap is None else cap * torch.tanh(out / cap)
        return nwj_objective(f, y, w)
    return bce_objective(out, y, w)


def train_ensemble(factory: Callable[[], torch.nn.Module], X: np.ndarray, y: np.ndarray, w: np.ndarray,
                   objective: str, cfg: TrainConfig, X_val: np.ndarray | None = None,
                   y_val: np.ndarray | None = None, w_val: np.ndarray | None = None,
                   fixed_epochs: int | None = None) -> TorchEnsemble:
    """Train one member per seed.  With validation data: early stopping (evaluate every
    ``eval_every`` epochs, stop after ``patience`` checks without ``min_delta`` improvement)
    and the validation-optimal weights are kept.  With ``fixed_epochs``: train exactly that
    many epochs (used for the outer refit)."""
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    yt = torch.as_tensor(np.asarray(y, dtype=np.float32))
    wt = torch.as_tensor(np.asarray(w, dtype=np.float32))
    has_val = X_val is not None and fixed_epochs is None
    if has_val:
        Xv = torch.as_tensor(np.asarray(X_val, dtype=np.float32))
        yv = torch.as_tensor(np.asarray(y_val, dtype=np.float32))
        wv = torch.as_tensor(np.asarray(w_val, dtype=np.float32))
    members, best_epochs, curves = [], [], []
    n_epochs = fixed_epochs if fixed_epochs is not None else cfg.max_epochs
    for seed in cfg.seeds:
        torch.manual_seed(seed)
        model = factory()
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        best_val, best_state, best_epoch, stale = math.inf, None, n_epochs, 0
        curve = {"epoch": [], "train": [], "val": []}
        for epoch in range(1, n_epochs + 1):
            model.train()
            opt.zero_grad()
            loss = _objective_value(model, Xt, yt, wt, objective, cfg.critic_cap) + 0.5 * cfg.l2 * _l2_weights(model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            opt.step()
            if has_val and epoch % cfg.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    val = float(_objective_value(model, Xv, yv, wv, objective, cfg.critic_cap))
                    tr = float(_objective_value(model, Xt, yt, wt, objective, cfg.critic_cap))
                curve["epoch"].append(epoch); curve["train"].append(tr); curve["val"].append(val)
                if val < best_val - cfg.min_delta:
                    best_val, best_epoch, stale = val, epoch, 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= cfg.patience:
                        break
        if has_val and best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        members.append(model); best_epochs.append(best_epoch); curves.append(curve)
    return TorchEnsemble(members=members, objective=objective, critic_cap=cfg.critic_cap,
                         best_epochs=best_epochs, curves=curves)
