"""Coarse-grain the simulated trajectories and estimate information quantities.

Uses exactly the recipe validated on the LLM data in ../30_coarse_graining/:
the population coordinate e = mean_i P(A0 | K_i) binned into 4 states at
0.33 / 0.50 / 0.75. Because we generated the data, the estimates can be checked
against ground truth that the LLM study never had.
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CUTS = [0.33, 0.50, 0.75]
S = len(CUTS) + 1
EPS = 1e-12


def state(e):
    return np.digitize(e, CUTS)


def kernel(df):
    """First-order transition matrix from per-episode state sequences."""
    T = np.zeros((S, S))
    for _, g in df.groupby("episode"):
        s = state(g.sort_values("round_index")["e"].to_numpy())
        for a, b in zip(s[:-1], s[1:]):
            T[a, b] += 1
    counts = T.copy()
    T = (T + EPS) / (T + EPS).sum(1, keepdims=True)
    return T, counts


def init_dist(df):
    p = np.zeros(S)
    for _, g in df.groupby("episode"):
        p[state(g.sort_values("round_index")["e"].iloc[0])] += 1
    return (p + EPS) / (p + EPS).sum()


def occupancy(p0, T, h):
    occ, p = [], p0.copy()
    for _ in range(h):
        occ.append(p.copy()); p = p @ T
    return np.array(occ)


def path_kl(pP, TP, pQ, TQ, h):
    """Exact path KL on a finite-state Markov chain -- no critic needed."""
    kl = float(np.sum(pP * np.log((pP + EPS) / (pQ + EPS))))
    occ = occupancy(pP, TP, h)
    row = np.sum(TP * np.log((TP + EPS) / (TQ + EPS)), axis=1)
    for t in range(h):
        kl += float(occ[t] @ row)
    return kl


def order_gain(df):
    """Held-out log-likelihood gain of a 2nd-order model. ~0 means Markov."""
    seqs = [state(g.sort_values("round_index")["e"].to_numpy())
            for _, g in df.groupby("episode")]
    cut = len(seqs) // 2
    tr, te = seqs[:cut], seqs[cut:]
    T1 = np.full((S, S), EPS); T2 = np.full((S, S, S), EPS)
    for s in tr:
        for a, b in zip(s[:-1], s[1:]): T1[a, b] += 1
        for a, b, c in zip(s[:-2], s[1:-1], s[2:]): T2[a, b, c] += 1
    T1 /= T1.sum(1, keepdims=True); T2 /= T2.sum(2, keepdims=True)
    l1 = l2 = n = 0.0
    for s in te:
        for a, b, c in zip(s[:-2], s[1:-1], s[2:]):
            l1 += np.log(T1[b, c]); l2 += np.log(T2[a, b, c]); n += 1
    return (l2 - l1) / max(n, 1)


def ck_gap(df, n):
    """Chapman-Kolmogorov: || empirical n-step - (1-step)^n ||_TV."""
    T, _ = kernel(df)
    emp = np.zeros((S, S))
    for _, g in df.groupby("episode"):
        s = state(g.sort_values("round_index")["e"].to_numpy())
        for a, b in zip(s[:-n], s[n:]): emp[a, b] += 1
    rows = emp.sum(1)
    ok = rows > 0
    if not ok.any(): return np.nan
    emp = emp[ok] / rows[ok, None]
    Tn = np.linalg.matrix_power(T, n)[ok]
    return float(0.5 * np.abs(emp - Tn).sum(1).mean())


def mutual_info(dfs, labels, h):
    """I(label; state at round h) in bits, plug-in."""
    tab = np.zeros((len(dfs), S))
    for i, d in enumerate(dfs):
        for _, g in d.groupby("episode"):
            row = g[g.round_index == h]
            if len(row): tab[i, state(row["e"].iloc[0])] += 1
    p = tab / tab.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    m = p > 0
    return float((p[m] * np.log2(p[m] / (px @ py)[m])).sum())


def main():
    df = pd.read_parquet(OUT / "trajectories.parquet")
    H = int(df.round_index.max())
    recs, curves = [], []
    for (setup, rho, rule), g in df.groupby(["setup", "rho", "rule"]):
        sил = g[g.arm == "silent"]
        sil = g[g.arm == "silent"]
        if sil.empty: continue
        pQ, TQ = init_dist(sil), kernel(sil)[0]
        for arm in ("truth", "false"):
            a = g[g.arm == arm]
            if a.empty: continue
            pP, TP = init_dist(a), kernel(a)[0]
            for h in range(1, H + 1):
                curves.append({"setup": setup, "rho": rho, "rule": rule, "arm": arm,
                               "h": h, "path_kl": path_kl(pP, TP, pQ, TQ, h)})
            _, cnt = kernel(a)
            recs.append({
                "setup": setup, "rho": rho, "rule": rule, "arm": arm,
                "path_kl_h29": path_kl(pP, TP, pQ, TQ, H),
                "order_gain": order_gain(a),
                "ck_gap_n2": ck_gap(a, 2), "ck_gap_n5": ck_gap(a, 5),
                "transitions": int(cnt.sum()),
                "states_visited": int((cnt.sum(1) + cnt.sum(0) > 0).sum()),
            })
        # I(arm; state) and I(target; state) at the final round
        recs_mi = {
            "I_intervention_bits": mutual_info(
                [g[g.arm == "silent"], g[g.arm.isin(("truth", "false"))]], None, H),
            "I_target_bits": mutual_info(
                [g[g.arm == "truth"], g[g.arm == "false"]], None, H),
        }
        for r in recs[-2:]:
            r.update(recs_mi)
    pd.DataFrame(recs).to_csv(OUT / "estimates.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "path_kl_curves.csv", index=False)

    # occupancy of the 4 coarse states, per cell
    df["state"] = state(df["e"].to_numpy())
    occ = (df.groupby(["setup", "rho", "arm", "rule", "state"]).size()
             .unstack("state", fill_value=0))
    occ = occ.div(occ.sum(1), axis=0)
    occ.to_csv(OUT / "state_occupancy.csv")

    e = pd.DataFrame(recs)
    print("=== information estimates (h = 29) ===")
    print(e[["setup", "rho", "rule", "arm", "path_kl_h29", "I_target_bits",
             "order_gain", "ck_gap_n5", "transitions"]].to_string(index=False))


if __name__ == "__main__":
    main()
