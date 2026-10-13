"""Success against actual cost: the operational frontier (2026-10-13_plan_frontier.md).

Per run: collective decision = plurality of the agents' last votes at time T (ties count 1/k);
target attainment = decision is the assigned target; task correctness = decision is A0. Per
configuration: Delta = P(success | controlled) - P(success | silent, same episodes), cost = facts
actually posted up to T (and controller reads). A frontier compares the controller configurations
of one group (simulation, setup, rho, q, target); metrics follow the handoff:
  eta_cost = C*(Delta) / C,   eta_performance = Delta / Delta*(C),   gain per fact = Delta / C,
with C*(d) = least cost of a tested configuration with Delta >= d, Delta*(B) = best Delta with cost <= B,
and the same with randomised mixtures (concave hull through (0, 0)).

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/efficiency_estimates/frontier.py [--date 2026-10-13]
"""
from __future__ import annotations
import argparse, datetime, hashlib, multiprocessing, pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent / "results"
SIM2 = RES / "simulation_2" / "2026-10-09_sim2_grid" / "cells"
SIM1 = RES / "simulation_1" / "2026-10-07_sim1_pm" / "cells"
TARGET = {"truth": 0, "false": 2}
THRESHOLDS = (0.05, 0.1, 0.2, 0.3)
B = 500
N_SUB = (50, 100, 300)
REPS = 200


# ---------------------------------------------------------------- pure functions (unit-tested)

def plurality_success(counts: np.ndarray, answer: int, rule: str = "plurality") -> np.ndarray:
    """counts: (..., 3) votes for A0, A1, A2. Returns the success of `answer`: plurality with ties
    counting 1/k (expected success under a random tie-break), or strict majority (> half)."""
    counts = np.asarray(counts, float)
    if rule == "majority":
        return (counts[..., answer] > counts.sum(-1) / 2).astype(float)
    top = counts.max(-1, keepdims=True)
    winners = (counts == top) & (top > 0)
    k = winners.sum(-1)
    return np.where(k > 0, winners[..., answer] / np.maximum(k, 1), 0.0)


def least_cost(delta: np.ndarray, cost: np.ndarray, d: float) -> float:
    """C*(d): least cost among configurations with Delta >= d (inf if none)."""
    ok = delta >= d
    return float(cost[ok].min()) if ok.any() else np.inf


def best_gain(delta: np.ndarray, cost: np.ndarray, b: float) -> float:
    """Delta*(B): best Delta among configurations with cost <= B (0 = do nothing, always allowed)."""
    ok = cost <= b
    return float(max(0.0, delta[ok].max())) if ok.any() else 0.0


def hull(delta: np.ndarray, cost: np.ndarray):
    """Upper concave envelope of {(0, 0)} and the configurations' (cost, Delta): what randomised
    mixtures (one configuration drawn per run) can reach. Returns its vertices sorted by cost."""
    pts = sorted({(0.0, 0.0)} | {(float(c), float(d)) for c, d in zip(cost, delta) if np.isfinite(c)})
    upper = []
    for p in pts:
        while len(upper) >= 2:
            (x1, y1), (x2, y2) = upper[-2], upper[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:   # middle point not above the chord
                upper.pop()
            else:
                break
        upper.append(p)
    xs, ys = np.array([p[0] for p in upper]), np.array([p[1] for p in upper])
    best = np.maximum.accumulate(ys)                     # unused budget is allowed: keep it nondecreasing
    return xs, best


def hull_least_cost(xs, ys, d: float) -> float:
    if d <= 0:
        return 0.0
    if ys.max() < d:
        return np.inf
    i = int(np.argmax(ys >= d))
    if i == 0:
        return float(xs[0])
    return float(xs[i - 1] + (d - ys[i - 1]) * (xs[i] - xs[i - 1]) / (ys[i] - ys[i - 1]))


def hull_best_gain(xs, ys, b: float) -> float:
    return float(np.interp(b, xs, ys, right=ys[-1]))


def efficiencies(delta: np.ndarray, cost: np.ndarray, resolved: np.ndarray | None = None) -> dict:
    """Per configuration: eta_cost, eta_performance (tested points and with mixtures), gain per fact.
    Not applicable (NaN) where Delta <= 0, cost <= 0, or the gain is not resolved."""
    n = len(delta)
    ok = (delta > 0) & (cost > 0) & (True if resolved is None else resolved)
    xs, ys = hull(delta, cost)
    out = {k: np.full(n, np.nan) for k in ("eta_cost", "eta_performance", "eta_cost_mix", "eta_performance_mix")}
    for i in np.where(ok)[0]:
        out["eta_cost"][i] = least_cost(delta, cost, delta[i]) / cost[i]
        out["eta_performance"][i] = delta[i] / best_gain(delta, cost, cost[i])
        out["eta_cost_mix"][i] = hull_least_cost(xs, ys, delta[i]) / cost[i]
        out["eta_performance_mix"][i] = delta[i] / hull_best_gain(xs, ys, cost[i])
    with np.errstate(divide="ignore", invalid="ignore"):
        out["gain_per_fact"] = np.where(cost > 0, delta / cost, np.nan)
    return out


# ---------------------------------------------------------------- data

def sim2_cell(setup, arm, q, qc, rho, lc, gate, stop):
    return SIM2 / f"{setup}__{arm}__q{q}__qc{qc}__rho{rho:g}__lc{lc:g}__gate{gate}__stop{stop}"


def load_sim2(folder, times):
    s = pd.read_parquet(folder / "snapshots.parquet", columns=["episode", "t", "share_A0", "share_A1", "share_A2",
                                                               "controller_messages", "controller_reads"],
                        filters=[("t", "in", list(times))])
    s = s.sort_values(["t", "episode"])
    E = s.episode.nunique()
    counts = np.rint(s[["share_A0", "share_A1", "share_A2"]].to_numpy() * 24).reshape(len(times), E, 3)
    return counts, s.controller_messages.to_numpy(float).reshape(len(times), E), \
        s.controller_reads.to_numpy(float).reshape(len(times), E)


def load_sim1(folder, days, controlled=True):
    d = pd.read_parquet(folder / "days.parquet", columns=["episode", "day", "share_A0", "share_A1", "share_A2",
                                                          "controller_posts_tonight"]).sort_values(["day", "episode"])
    E = d.episode.nunique()
    D = d.day.nunique()
    shares = d[["share_A0", "share_A1", "share_A2"]].to_numpy().reshape(D, E, 3)
    posted = d.controller_posts_tonight.to_numpy(float).reshape(D, E)
    reads = np.zeros((D, E))
    if controlled and (folder / "nights.parquet").exists():
        n = pd.read_parquet(folder / "nights.parquet", columns=["episode", "day", "n_posts_read"])
        np.add.at(reads, (n.day.to_numpy(), n.episode.to_numpy()), n.n_posts_read.to_numpy(float))
    idx = np.array([t - 1 for t in days])               # day T = index T-1 (end of that day)
    cum_posts = np.cumsum(np.vstack([np.zeros((1, E)), posted[:-1]]), axis=0)   # facts posted on nights before day T
    cum_reads = np.cumsum(np.vstack([np.zeros((1, E)), reads[:-1]]), axis=0)
    return np.rint(shares[idx] * 24), cum_posts[idx], cum_reads[idx]


def groups():
    out = []
    for setup in ("task003_symmetric", "task003_nosolution"):
        for rho in (0.75, 1.0):
            for q in (3, 6, 12):
                for arm in ("truth", "false"):
                    out.append(dict(sim="sim2", setup=setup, rho=rho, q=q, arm=arm))
                    out.append(dict(sim="sim1", setup=setup, rho=rho, q=q, arm=arm))
    return out


def configs(g):
    if g["sim"] == "sim2":
        stops = ["on", "off"] if (g["setup"] == "task003_symmetric" and g["arm"] == "truth") else ["on"]
        return [dict(qc=qc, x=lc, gate=gate, stop=stop) for qc in (3, 6, 12, 24) for lc in (0.5, 1.0, 2.0, 4.0, 8.0)
                for gate in ("on", "off") for stop in stops]
    return [dict(qc=qc, x=b, gate="on", stop="on") for qc in (3, 6, 12, 24) for b in (1, 2, 3, 6, 9)]


def load_group(g):
    """Returns times, success arrays (config, time, episode) for target and task, both rules, and costs."""
    z = TARGET[g["arm"]]
    cf = configs(g)
    if g["sim"] == "sim2":
        times = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        sil = SIM2 / f"{g['setup']}__silent__q{g['q']}__rho{g['rho']:g}"
        c0, _, _ = load_sim2(sil, times)
        loaded = [load_sim2(sim2_cell(g["setup"], g["arm"], g["q"], c["qc"], g["rho"], c["x"], c["gate"], c["stop"]), times)
                  for c in cf]
    else:
        times = [5, 10, 15, 20, 25, 30]
        r = "0.75" if g["rho"] == 0.75 else "1.0"
        sil = SIM1 / f"{g['setup']}__silent__q{g['q']}__rho{r}"
        c0, _, _ = load_sim1(sil, times, controlled=False)
        loaded = [load_sim1(SIM1 / f"{g['setup']}__{g['arm']}__q{g['q']}__qc{c['qc']}__b{c['x']}__rho{r}", times)
                  for c in cf]
    counts = np.stack([l[0] for l in loaded])            # (config, time, episode, 3)
    facts = np.stack([l[1] for l in loaded])
    reads = np.stack([l[2] for l in loaded])
    S = {}
    for rule in ("plurality", "majority"):
        S[("target", rule)] = plurality_success(counts, z, rule), plurality_success(c0, z, rule)
        S[("task", rule)] = plurality_success(counts, 0, rule), plurality_success(c0, 0, rule)
    return cf, times, S, facts, reads


def weighted_means(W, X):
    """W: (b, E) resample weights; X: (..., E) -> (b, ...)"""
    return np.tensordot(W, X, axes=(1, -1)).transpose((0,) + tuple(range(1, X.ndim))) / W.sum(1).reshape(-1, *([1] * (X.ndim - 1)))


def run_group(g):
    cf, times, S, facts, reads = load_group(g)
    E = facts.shape[-1]
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(repr(sorted(g.items())).encode()).digest()[:8], "big"))
    idx = rng.integers(0, E, size=(B, E))
    W = np.zeros((B, E))
    np.add.at(W, (np.arange(B)[:, None], idx), 1)
    half = rng.permutation(E)
    A_, B_ = half[: E // 2], half[E // 2:]
    pts, thr, ds = [], [], []
    for (outcome, rule), (Sc, S0) in S.items():
        for ti, T in enumerate(times):
            delta_e = Sc[:, ti, :] - S0[ti][None, :]          # (config, E): paired differences
            cost_e = facts[:, ti, :]
            delta, cost = delta_e.mean(1), cost_e.mean(1)
            bd = weighted_means(W, delta_e)                   # (B, config)
            bc = weighted_means(W, cost_e)
            lo, hi = np.percentile(bd, [2.5, 97.5], axis=0)
            resolved = lo > 0
            eff = efficiencies(delta, cost, resolved)
            beff = {k: np.full((B, len(cf)), np.nan) for k in ("eta_cost", "eta_performance")}
            for b in range(B):
                e = efficiencies(bd[b], bc[b])
                beff["eta_cost"][b], beff["eta_performance"][b] = e["eta_cost"], e["eta_performance"]
            # selection check: frontier chosen on half A, evaluated with half B's estimates
            dA, cA = delta_e[:, A_].mean(1), cost_e[:, A_].mean(1)
            dB, cB = delta_e[:, B_].mean(1), cost_e[:, B_].mean(1)
            eta_split = np.full(len(cf), np.nan)
            for i in np.where((dB > 0) & (cB > 0))[0]:
                okA = dA >= dB[i]
                if okA.any():
                    j = np.where(okA)[0][np.argmin(cA[okA])]   # cheapest configuration on half A
                    eta_split[i] = cB[j] / cB[i]                # > 1: the half-A choice was optimistic
            for i, c in enumerate(cf):
                row = dict(sim=g["sim"], setup=g["setup"], rho=g["rho"], q=g["q"], arm=g["arm"], outcome=outcome, rule=rule,
                           T=T, **c, p_controlled=float(Sc[i, ti].mean()), p_silent=float(S0[ti].mean()),
                           delta=float(delta[i]), delta_lo=float(lo[i]), delta_hi=float(hi[i]), facts=float(cost[i]),
                           reads=float(reads[i, ti].mean()), resolved=bool(resolved[i]),
                           eta_cost_split=float(eta_split[i]))
                for k, v in eff.items():
                    row[k] = float(v[i])
                for k in ("eta_cost", "eta_performance"):
                    bv = beff[k][:, i]
                    bv = bv[np.isfinite(bv)]
                    row[f"{k}_lo"] = float(np.percentile(bv, 2.5)) if len(bv) > 20 else np.nan
                    row[f"{k}_hi"] = float(np.percentile(bv, 97.5)) if len(bv) > 20 else np.nan
                pts.append(row)
            xs, ys = hull(delta, cost)
            for dlt in THRESHOLDS:
                bcs = np.array([least_cost(bd[b], bc[b], dlt) for b in range(B)])
                fin = bcs[np.isfinite(bcs)]
                thr.append(dict(sim=g["sim"], setup=g["setup"], rho=g["rho"], q=g["q"], arm=g["arm"], outcome=outcome,
                                rule=rule, T=T, threshold=dlt, least_cost=least_cost(delta, cost, dlt),
                                least_cost_mix=hull_least_cost(xs, ys, dlt), share_reached=float(len(fin) / B),
                                least_cost_lo=float(np.percentile(fin, 2.5)) if len(fin) > 20 else np.nan,
                                least_cost_hi=float(np.percentile(fin, 97.5)) if len(fin) > 20 else np.nan))
            if rule == "plurality" and T == (30.0 if g["sim"] == "sim2" else 30):
                for m in N_SUB:
                    sub = np.array([rng.choice(E, m, replace=False) for _ in range(REPS)])
                    Wm = np.zeros((REPS, E))
                    np.add.at(Wm, (np.arange(REPS)[:, None], sub), 1)
                    sd, sc = weighted_means(Wm, delta_e), weighted_means(Wm, cost_e)
                    ec = np.array([efficiencies(sd[r], sc[r])["eta_cost"] for r in range(REPS)])
                    for i, c in enumerate(cf):
                        f_e = eff["eta_cost"][i]
                        v = ec[:, i]
                        ds.append(dict(sim=g["sim"], setup=g["setup"], rho=g["rho"], q=g["q"], arm=g["arm"], outcome=outcome,
                                       **c, n_sub=m, delta_full=float(delta[i]),
                                       delta_rmse=float(np.sqrt(((sd[:, i] - delta[i]) ** 2).mean())),
                                       eta_cost_full=float(f_e),
                                       eta_cost_rmse=float(np.sqrt(np.nanmean((v - f_e) ** 2))) if np.isfinite(f_e) and np.isfinite(v).any() else np.nan,
                                       eta_cost_share_undefined=float(np.mean(~np.isfinite(v)))))
    return pts, thr, ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=f"{datetime.date.today():%Y-%m-%d}")
    a = ap.parse_args()
    d = RES / "efficiency_estimates" / a.date
    d.mkdir(parents=True, exist_ok=True)
    with multiprocessing.Pool(10) as pool:
        parts = pool.map(run_group, groups())
    for i, name in enumerate(("frontier_points", "frontier_thresholds", "frontier_downsample")):
        pd.DataFrame([r for p in parts for r in p[i]]).to_csv(d / f"{name}.csv", index=False)
    print(f"{sum(len(p[0]) for p in parts)} configuration rows -> {d}")


if __name__ == "__main__":
    main()
