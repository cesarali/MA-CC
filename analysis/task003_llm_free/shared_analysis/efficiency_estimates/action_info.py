"""Action information pilot (2026-10-13_plan_action_information.md).

U = intervention or not in a step (binary); V = an agent's own current vote; V' = its vote after
the step. Per controlled arm, compared with the silent arm over the same episodes and steps:
  I = I(U; V' | V),  R = sum_v w(v) KL(Kbar(.|v) || K0(.|v)),  C = I + R,  eta_U = I / C,
plus P(U=1), H(U|V), I/H(U|V) and the signed response chi (change in P(V' = target) when acting).

Each episode contributes a count tensor; totals, bootstrap resamples and downsampled subsets are
weighted sums of those tensors over episodes (agents of an episode always stay together).

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/shared_analysis/efficiency_estimates/action_info.py [--date 2026-10-13]
Outputs: results/efficiency_estimates/<date>/action_info.csv, action_downsample.csv
"""
from __future__ import annotations
import argparse, datetime, hashlib, multiprocessing, pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results"
SIM2_GRID = RES / "simulation_2" / "2026-10-09_sim2_grid" / "cells"
SIM2_RHO1 = RES / "simulation_2" / "2026-10-12_sim2_main_rho1_full" / "cells"
SIM1 = RES / "simulation_1" / "2026-10-07_sim1_pm" / "cells"
TARGET = {"truth": 0, "false": 2}
NV = 4                      # vote states: A0, A1, A2, none (Simulation 1 never uses "none")
NS = 3                      # coarse swarm state: target vote share in 3 equal bins
B = 500
N_SUB = (50, 100, 300)
REPS = 200
DELTAS = (0.25, 1.0)        # Simulation 2 windows


def seeded(name: str) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big"))


# ---------------------------------------------------------------- measures from counts

def kl_rows(p, q):
    """KL(p||q) in bits along the last axis for row-stochastic arrays; inf where q = 0 < p."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log2(p / np.where(q > 0, q, 1.0)), 0.0)
        bad = ((p > 0) & (q <= 0)).any(axis=-1)
    return np.where(bad, np.inf, t.sum(axis=-1))


def measures(N, N0, z):
    """N: (..., 2, V, V') controlled counts by action; N0: (..., V, V') silent counts. Leading axes
    are batch (bootstrap). Returns a dict of arrays over the batch."""
    N = np.asarray(N, float)
    N0 = np.asarray(N0, float)
    nuv = N.sum(-1)                                   # (..., 2, V)
    nv = nuv.sum(-2)                                  # (..., V)
    tot = nv.sum(-1)                                  # (...)
    with np.errstate(divide="ignore", invalid="ignore"):
        Pu = N / nuv[..., None]                       # P_u(v'|v)
        w = nuv / nv[..., None, :]                    # w(u|v)
        Kbar = N.sum(-3) / nv[..., None]              # Kbar(v'|v)
        K0 = N0 / N0.sum(-1, keepdims=True)
        om = nv / tot[..., None]                      # omega(v)
    Pu, w, Kbar, K0, om = (np.nan_to_num(a) for a in (Pu, w, Kbar, K0, om))
    kl_u_bar = kl_rows(Pu, Kbar[..., None, :, :])     # (..., 2, V)
    I = (om * (w * kl_u_bar).sum(-2)).sum(-1)
    kl_bar_0 = kl_rows(Kbar, K0)                      # (..., V)
    with np.errstate(invalid="ignore"):
        R = np.where(om > 0, om * kl_bar_0, 0.0).sum(-1)
    C = I + R
    with np.errstate(divide="ignore", invalid="ignore"):
        hU = -(np.where(w > 0, w * np.log2(np.where(w > 0, w, 1)), 0)).sum(-2)      # (..., V)
        HUV = (om * hU).sum(-1)
        eta = np.where(np.isfinite(C) & (C > 1e-6), I / C, np.nan)
        IH = np.where(HUV > 1e-6, I / HUV, np.nan)
        pU = nuv[..., 1, :].sum(-1) / tot
        chi = (om * (Pu[..., 1, :, z] - Pu[..., 0, :, z]) * (nuv.min(-2) > 0)).sum(-1)
    return {"I": I, "R": R, "C": C, "eta_U": eta, "P_U1": pU, "H_U_given_V": HUV, "I_over_H": IH, "chi": chi,
            "transitions": tot}


# ---------------------------------------------------------------- counts per episode

def counts_from(V, Vn, U, keep, S=None, nstate=NV):
    """Per-episode counts. V, Vn: (E, W, A) votes now/after; U: (E, W); keep: (E, W, A) bool.
    Returns controlled (E, 2, X, V') and the state index used, X = V or (S, V)."""
    E = V.shape[0]
    x = V if S is None else S[:, :, None] * nstate + V
    nx = nstate if S is None else NS * nstate
    out = np.zeros((E, 2, nx, nstate))
    e_idx = np.broadcast_to(np.arange(E)[:, None, None], V.shape)
    u_idx = np.broadcast_to(U[:, :, None], V.shape)
    np.add.at(out, (e_idx[keep], u_idx[keep], x[keep], Vn[keep]), 1)
    return out


def silent_counts(V, Vn, keep, S=None, nstate=NV):
    E = V.shape[0]
    x = V if S is None else S[:, :, None] * nstate + V
    nx = nstate if S is None else NS * nstate
    out = np.zeros((E, nx, nstate))
    e_idx = np.broadcast_to(np.arange(E)[:, None, None], V.shape)
    np.add.at(out, (e_idx[keep], x[keep], Vn[keep]), 1)
    return out


def swarm_state(V, z):
    share = (V == z).mean(axis=-1)                     # (E, W)
    return np.clip(np.floor(share * NS).astype(int), 0, NS - 1)


# ---------------------------------------------------------------- loaders

def sim2_votes(folder: pathlib.Path):
    a = pd.read_parquet(folder / "agent_snapshots.parquet", columns=["episode", "grid_index", "agent", "last_vote", "vote_age"])
    a = a.sort_values(["episode", "grid_index", "agent"])
    E, G = a.episode.nunique(), a.grid_index.nunique()
    v = a.last_vote.to_numpy().reshape(E, G, 24).astype(int)
    v = np.where(v < 0, 3, v)
    age = a.vote_age.to_numpy().reshape(E, G, 24)
    return v, age


def sim2_job(job):
    setup, rho, lc, gate, arm = job["setup"], job["rho"], job["lc"], job["gate"], job["arm"]
    if rho == 0.75:
        base, sfx = SIM2_GRID, f"__q6__qc12__rho0.75__lc{lc:g}__gate{gate}__stopon"
        silent = SIM2_GRID / f"{setup}__silent__q6__rho0.75"
    else:
        base, sfx = SIM2_RHO1, f"__lc{lc:g}__gate{gate}__stopon"
        silent = SIM2_RHO1 / f"{setup}__silent"
    folder = base / f"{setup}__{arm}{sfx}"
    z = TARGET[arm]
    V, AGE = sim2_votes(folder)
    V0, AGE0 = sim2_votes(silent)
    E = V.shape[0]
    p = pd.read_parquet(folder / "posts.parquet", columns=["episode", "t", "author"])
    p = p[p.author < 0]
    rows = []
    for delta in DELTAS:
        k = int(round(delta / 0.25))
        nwin = int(round(30.0 / delta))
        starts = np.arange(nwin) * k
        U = np.zeros((E, nwin), int)
        before = np.zeros((E, nwin), int)                # controller posts before the window
        if len(p):
            w = np.floor(p.t.to_numpy() / delta).astype(int)
            ok = w < nwin
            np.add.at(U, (p.episode.to_numpy()[ok], w[ok]), 1)
            cum = np.cumsum(U, axis=1)
            before = cum - U
        U = (U > 0).astype(int)
        budget = before < 30
        def pick(VV, AA):
            return VV[:, starts, :], VV[:, starts + k, :], AA[:, starts + k, :] < delta
        v, vn, acted = pick(V, AGE)
        v0, vn0, acted0 = pick(V0, AGE0)
        keep_all = np.broadcast_to(budget[:, :, None], v.shape)
        for variant, keep, keep0, S, S0 in (
                ("all agents", keep_all, keep_all, None, None),
                ("agents who voted", keep_all & acted, keep_all & acted0, None, None),
                ("plus swarm state", keep_all, keep_all, swarm_state(v, z), swarm_state(v0, z))):
            Nc = counts_from(v, vn, U, keep, S)
            N0 = silent_counts(v0, vn0, keep0, S0)
            rows.append(dict(sim="sim2", setup=setup, rho=rho, setting=f"lc{lc:g}_gate{gate}", arm=arm, delta=delta,
                             variant=variant, z=z, Nc=Nc, N0=N0))
    return rows


def sim1_job(job):
    setup, rho, b, arm = job["setup"], job["rho"], job["b"], job["arm"]
    r = "0.75" if rho == 0.75 else "1.0"
    folder = SIM1 / f"{setup}__{arm}__q6__qc12__b{b}__rho{r}"
    silent = SIM1 / f"{setup}__silent__q6__rho{r}"
    z = TARGET[arm]

    def votes(f):
        a = pd.read_parquet(f / "agents.parquet", columns=["episode", "day", "agent", "vote"]).sort_values(["episode", "day", "agent"])
        E, D = a.episode.nunique(), a.day.nunique()
        return a.vote.to_numpy().reshape(E, D, 24).astype(int)
    V, V0 = votes(folder), votes(silent)
    E, D = V.shape[:2]
    n = pd.read_parquet(folder / "nights.parquet", columns=["episode", "day", "decision", "n_posted"])
    U = np.zeros((E, D - 1), int)
    proved = np.zeros((E, D - 1), bool)
    U[n.episode.to_numpy(), n.day.to_numpy()] = (n.n_posted.to_numpy() > 0).astype(int)
    proved[n.episode.to_numpy(), n.day.to_numpy()] = (n.decision.to_numpy() == "proved")
    v, vn, v0, vn0 = V[:, :-1], V[:, 1:], V0[:, :-1], V0[:, 1:]
    keep = np.ones(v.shape, bool)
    rows = []
    variants = [("all agents", keep, None, None), ("plus swarm state", keep, swarm_state(v, z), swarm_state(v0, z))]
    if proved.any():
        variants.append(("without proof-stop nights", keep & ~proved[:, :, None], None, None))
    for variant, kp, S, S0 in variants:
        Nc = counts_from(v, vn, U, kp, S)
        N0 = silent_counts(v0, vn0, kp, S0)
        rows.append(dict(sim="sim1", setup=setup, rho=rho, setting=f"b{b}", arm=arm, delta=1.0, variant=variant, z=z,
                         Nc=Nc, N0=N0))
    return rows


def run(job):
    raw = sim2_job(job) if job["sim"] == "sim2" else sim1_job(job)
    out_rows, ds_rows = [], []
    for r in raw:
        Nc, N0, z = r.pop("Nc"), r.pop("N0"), r.pop("z")
        E = Nc.shape[0]
        point = measures(Nc.sum(0), N0.sum(0), z)
        rng = seeded("action_info")
        idx = rng.integers(0, E, size=(B, E))
        W = np.zeros((B, E))
        np.add.at(W, (np.arange(B)[:, None], idx), 1)
        bc = np.tensordot(W, Nc, axes=(1, 0))
        b0 = np.tensordot(W, N0, axes=(1, 0))
        boot = measures(bc, b0, z)
        row = dict(r)
        for k, v in point.items():
            row[k] = float(v)
            bv = boot[k][np.isfinite(boot[k])]
            row[f"{k}_lo"] = float(np.percentile(bv, 2.5)) if len(bv) else np.nan
            row[f"{k}_hi"] = float(np.percentile(bv, 97.5)) if len(bv) else np.nan
        out_rows.append(row)
        if r["variant"] == "all agents":
            srng = seeded("action_info_subsets")
            for m in N_SUB:
                sub = np.array([srng.choice(E, m, replace=False) for _ in range(REPS)])
                Wm = np.zeros((REPS, E))
                np.add.at(Wm, (np.arange(REPS)[:, None], sub), 1)
                s = measures(np.tensordot(Wm, Nc, axes=(1, 0)), np.tensordot(Wm, N0, axes=(1, 0)), z)
                for q in ("I", "R", "eta_U", "I_over_H", "chi"):
                    v = s[q]
                    ok = np.isfinite(v)
                    f = float(point[q])
                    ds_rows.append(dict(**{k: r[k] for k in ("sim", "setup", "rho", "setting", "arm", "delta")}, n_sub=m,
                                        quantity=q, full=f, rmse=float(np.sqrt(np.mean((v[ok] - f) ** 2))) if ok.any() and np.isfinite(f) else np.nan,
                                        share_undefined=float(1 - ok.mean())))
    return out_rows, ds_rows


def jobs():
    out = []
    for setup in ("task003_symmetric", "task003_nosolution"):
        for rho in (0.75, 1.0):
            for arm in ("truth", "false"):
                for lc in (0.5, 1.0, 2.0, 4.0, 8.0):
                    for gate in ("on", "off"):
                        out.append(dict(sim="sim2", setup=setup, rho=rho, lc=lc, gate=gate, arm=arm))
                for b in (1, 2, 3, 6, 9):
                    out.append(dict(sim="sim1", setup=setup, rho=rho, b=b, arm=arm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=f"{datetime.date.today():%Y-%m-%d}")
    a = ap.parse_args()
    d = RES / "efficiency_estimates" / a.date
    d.mkdir(parents=True, exist_ok=True)
    with multiprocessing.Pool(8) as pool:
        parts = pool.map(run, jobs())
    pd.DataFrame([r for p in parts for r in p[0]]).to_csv(d / "action_info.csv", index=False)
    pd.DataFrame([r for p in parts for r in p[1]]).to_csv(d / "action_downsample.csv", index=False)
    print(f"{sum(len(p[0]) for p in parts)} rows -> {d / 'action_info.csv'}")


if __name__ == "__main__":
    main()
