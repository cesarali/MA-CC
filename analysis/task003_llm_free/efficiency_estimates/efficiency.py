"""Directed information, total change and efficiency (PLAN.md §2), at full sample size.

Every resolution uses the same estimator: each episode e of arm a contributes a probability
vector y_a(e) over the values of X (agent level: its vote or belief shares; swarm level: a
one-hot vector marking the bin of s = m_A0 - m_A2). The distribution of X in arm a is the
average P_a = (1/n) sum_e y_a(e). Then, with Q = (P_+ + P_-)/2:
  I = (KL(P_+||Q) + KL(P_-||Q))/2   C = (KL(P_+||P_0) + KL(P_-||P_0))/2   R = KL(Q||P_0)
  eta = I/C (C = I + R exactly);  eta_JS = I/(I + JS(Q, P_0));  I3 = information about Z in {+,-,0}.
Bootstrap: 1,000 resamples of episode numbers, the same for all arms (paired).

Usage, from the repository root (after build_states.py):
    .venv/bin/python analysis/task003_llm_free/efficiency_estimates/efficiency.py [--date <date>]
Outputs in results/efficiency_estimates/<date>/: efficiency.csv, uncertainty.csv, null_I.csv
"""
from __future__ import annotations
import argparse, datetime, hashlib, pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent / "results" / "efficiency_estimates"
B = 1000
BINS = (2, 3, 4, 8)


def resolutions():
    """(name, level, source, function from a states frame to an (n, k) array of y vectors)."""
    out = [("agent_votes", "agent", "votes", lambda d: d[["x0", "x1", "x2", "xn"]].to_numpy()),
           ("agent_beliefs", "agent", "beliefs", lambda d: d[["m0", "m1", "m2"]].to_numpy())]
    for k in BINS:
        for source, (a, b) in (("beliefs", ("m0", "m2")), ("votes", ("x0", "x2"))):
            out.append((f"swarm_{source}_k{k}", "swarm", source,
                        lambda d, k=k, a=a, b=b: one_hot(d[a].to_numpy() - d[b].to_numpy(), k)))
    return out


def one_hot(s: np.ndarray, k: int) -> np.ndarray:
    """Equal bins on [-1, 1]; the top edge belongs to the last bin."""
    idx = np.clip(np.floor((s + 1) / 2 * k).astype(int), 0, k - 1)
    return np.eye(k)[idx]


def kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p||q) in bits along the last axis; inf where p > 0 and q = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log2(p / np.where(q > 0, q, 1.0)), 0.0)
        bad = ((p > 0) & (q <= 0)).any(axis=-1)
    return np.where(bad, np.inf, t.sum(axis=-1))


def kl_binary(a, b):
    """KL(Bern(a) || Bern(b)) in bits; inf where b is 0 or 1 and a differs from it."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(a > 0, a * np.log2(a / np.where(b > 0, b, 1.0)), 0.0)
        t0 = np.where(a < 1, (1 - a) * np.log2((1 - a) / np.where(b < 1, 1 - b, 1.0)), 0.0)
    bad = ((a > 0) & (b <= 0)) | ((a < 1) & (b >= 1))
    return np.where(bad, np.inf, t1 + t0)


def per_controller(Pz, P0, Cz, k) -> dict:
    """Measures of ONE controller against its own target k (agent level; k = 0 for A0, 2 for A2).
    S: minimum information any intervention needs to raise the target's probability from
       P0(k) to Pz(k) (the binary KL; the cheapest change keeps the other answers in proportion).
    progress = S / log2(1/P0(k)): share of the way from silence to unanimity on the target.
    focus = S / KL(Pz || P0): share of the controller's actual change needed for its gain.
    Both are signed: negative when the controller moved the target's probability DOWN (backfire)."""
    S = kl_binary(Pz[..., k], P0[..., k])
    with np.errstate(divide="ignore", invalid="ignore"):
        full = -np.log2(P0[..., k])
        progress = np.where((full >= 0.01) & np.isfinite(full) & np.isfinite(S), S / full, np.nan)   # < 0.01 bits left: undefined
        progress = np.where(Pz[..., k] < P0[..., k], -progress, progress)    # signed: moved away from target
        focus = np.where(np.isfinite(Cz) & (Cz >= 0.005) & np.isfinite(S), S / Cz, np.nan)
        focus = np.where(Pz[..., k] < P0[..., k], -focus, focus)          # signed like progress
    return {"S": S, "progress": progress, "focus": focus}


def measures(Pp, Pm, P0, targets=None) -> dict:
    Q = (Pp + Pm) / 2
    I = (kl(Pp, Q) + kl(Pm, Q)) / 2
    Cp, Cm = kl(Pp, P0), kl(Pm, P0)
    C = (Cp + Cm) / 2
    R = kl(Q, P0)
    M = (Q + P0) / 2
    RJS = (kl(Q, M) + kl(P0, M)) / 2
    M3 = (Pp + Pm + P0) / 3
    I3 = (kl(Pp, M3) + kl(Pm, M3) + kl(P0, M3)) / 3
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(np.isfinite(C) & (C > 1e-12), I / C, np.nan)
        eta_js = np.where(I + RJS > 1e-12, I / (I + RJS), np.nan)
    out = {"I": I, "C": C, "C_plus": Cp, "C_minus": Cm, "R": R, "eta": eta, "R_JS": RJS, "eta_JS": eta_js, "I3": I3}
    if targets is not None:
        for sign, Pz, Cz, k in (("plus", Pp, Cp, targets[0]), ("minus", Pm, Cm, targets[1])):
            for name, v in per_controller(Pz, P0, Cz, k).items():
                out[f"{name}_{sign}"] = v
    return out


def resample_counts(n: int, name: str, b: int = B) -> np.ndarray:
    """b x n matrix of how often each episode is drawn in each resample (shared by all arms)."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big"))
    idx = rng.integers(0, n, size=(b, n))
    counts = np.zeros((b, n))
    np.add.at(counts, (np.arange(b)[:, None], idx), 1)
    return counts


def summarise(point: dict, boot: dict) -> dict:
    out = {}
    for k, v in point.items():
        out[k] = float(v)
        bv = boot[k][np.isfinite(boot[k])]
        out[f"{k}_lo"] = float(np.percentile(bv, 2.5)) if len(bv) else np.nan
        out[f"{k}_hi"] = float(np.percentile(bv, 97.5)) if len(bv) else np.nan
        out[f"{k}_share_inf"] = float(np.mean(~np.isfinite(boot[k])))
    return out


def arm_arrays(states: pd.DataFrame):
    """{(sim, setup, rho, setting): {arm: frame sorted by episode, per time}} plus silent frames."""
    groups = {}
    for key, g in states.groupby(["sim", "setup", "rho"]):
        silent = g[g.arm == "silent"]
        for setting, h in g[g.arm != "silent"].groupby("setting"):
            groups[key + (setting,)] = {"+": h[h.arm == "truth"], "-": h[h.arm == "false"], "0": silent}
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    a = ap.parse_args()
    d = RES / (a.date or f"{datetime.date.today():%Y-%m-%d}")
    states = pd.read_parquet(d / "states.parquet")
    n = states.episode.nunique()
    counts = resample_counts(n, "efficiency_estimates")
    res_rows, unc_rows, null_rows = [], [], []
    rng = np.random.default_rng(20261012)
    for key, arms in arm_arrays(states).items():
        sim, setup, rho, setting = key
        times = sorted(arms["+"].t.unique())
        by_t = {a: {t: f.sort_values("episode") for t, f in arms[a].groupby("t")} for a in arms}
        for t in times:
            F = {a: by_t[a][t] for a in arms}
            assert all(len(F[x]) == n for x in F), (key, t)
            for name, level, source, fy in resolutions():
                Y = {x: fy(F[x]) for x in F}
                tg = (0, 2) if level == "agent" else None
                point = measures(*(Y[x].mean(axis=0) for x in ("+", "-", "0")), targets=tg)
                boot = measures(*((counts @ Y[x]) / n for x in ("+", "-", "0")), targets=tg)
                row = dict(sim=sim, setup=setup, rho=rho, setting=setting, t=t, resolution=name, level=level,
                           source=source, n=n)
                row.update(summarise(point, boot))
                res_rows.append(row)
                if t in (30.0, 40.0) or (sim == "sim1" and t == 30.0):
                    swap = rng.random((200, n)) < 0.5                  # swap + and - within episodes
                    I0 = []
                    for s in swap:
                        Pp = np.where(s[:, None], Y["-"], Y["+"]).mean(axis=0)
                        Pm = np.where(s[:, None], Y["+"], Y["-"]).mean(axis=0)
                        I0.append(float(measures(Pp, Pm, Y["0"].mean(axis=0))["I"]))
                    null_rows.append(dict(sim=sim, setup=setup, rho=rho, setting=setting, t=t, resolution=name,
                                          I=float(point["I"]), null_mean=np.mean(I0), null_q95=np.percentile(I0, 95)))
            # agents' own uncertainty: controlled - silent, same episode
            if F["0"].Hbar.notna().all():
                for arm, sign in (("truth", "+"), ("false", "-")):
                    for col in ("Hbar", "JSD"):
                        diff = F[sign][col].to_numpy() - F["0"][col].to_numpy()
                        bs = counts @ diff / n
                        unc_rows.append(dict(sim=sim, setup=setup, rho=rho, setting=setting, t=t, arm=arm,
                                             measure=col, silent=F["0"][col].mean(), controlled=F[sign][col].mean(),
                                             diff=diff.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    pd.DataFrame(res_rows).to_csv(d / "efficiency.csv", index=False)
    pd.DataFrame(unc_rows).to_csv(d / "uncertainty.csv", index=False)
    pd.DataFrame(null_rows).to_csv(d / "null_I.csv", index=False)
    print(f"{len(res_rows)} efficiency rows, {len(unc_rows)} uncertainty rows, {len(null_rows)} null rows in {d}")


if __name__ == "__main__":
    main()
