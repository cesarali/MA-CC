"""Discriminating test (00_preregistration.md): does the effect of adding one controller fact depend on the count of agent
facts held (graded count response) or also on corrective combinations (combination response)?

Reads stored simulator memories (read-only) and the worlds; computes exact posteriors. Writes
results_by_setup.csv, within_count.csv, per_fact.csv in this folder.

Usage, from the repository root:
    caffeinate -i .venv/bin/python analysis/theory_claude/discriminating_test/marginal_persuasion.py
"""
from __future__ import annotations
import itertools, json, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
A = HERE.parents[1]
sys.path.insert(0, str(A / "task003_llm_free"))
from llmfree_core.kworld import load_world  # noqa: E402
from llmfree_core.setups import load_setup  # noqa: E402

RUNS = {"task003": (A / "task003_llm_free/results/simulation_2/2026-10-20_sim2_reading_mechanism",
                    A / "task003_llm_free/llmfree_core/world_task003.json", A / "task003_llm_free/experimental_setup",
                    {"symmetric": "task003_symmetric", "no-proof": "task003_nosolution"}),
        "Object Placements": (A / "object_placements_llm_free/results/simulation_2/2026-10-22_sim2_op_main",
                              A / "object_placements_llm_free/world/world_op1.json",
                              A / "object_placements_llm_free/experimental_setup",
                              {"symmetric": "op_symmetric", "no-proof": "op_nosolution"})}
CLIP = 1e-3


def logit(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def sig(x):
    return 1 / (1 + np.exp(-x))


def motifs(w, agent):
    cand = []
    for k in (2, 3):
        for S in itertools.combinations(agent, k):
            m = 0
            for f in S:
                m |= 1 << f
            cand.append((w.posterior(m)[2], k, S))
    cand.sort(key=lambda x: (x[0], x[1], x[2]))
    return [c[2] for c in cand[:5]], [c[0] for c in cand[:5]]


def collect(task, kind):
    run, wf, sd, names = RUNS[task]
    w = load_world(str(wf))
    st = load_setup(names[kind], sd, w)
    agent = sorted(set(st.agent_facts))
    ctrl = sorted(set(st.pools[2]) - set(agent))
    mot, mot_p = motifs(w, agent)
    amask = sum(1 << f for f in agent)
    cache = {}

    def P2(m):
        if m not in cache:
            cache[m] = w.posterior(m)[2]
        return cache[m]

    rows, proven = [], 0
    for c in sorted((run / "cells").iterdir()):
        p = json.loads((c / "params.json").read_text())
        if p["setup"] != names[kind] or p["arm"] != "silent" or int(p["q"]) not in (3, 6) or float(p["rho"]) not in (0.25, 0.5, 0.75):
            continue
        ag = pd.read_parquet(c / "agent_snapshots.parquet", columns=["episode", "grid_index", "active"], filters=[("grid_index", "==", 120)])
        for e, m in zip(ag.episode.to_numpy(), ag.active.to_numpy().astype(np.int64)):
            m = int(m)
            if w.posterior(m)[0] >= 1 - 1e-12:
                proven += 1
                continue
            a = bin(m & amask).count("1")
            g = max(sum((m >> f) & 1 for f in S) / len(S) for S in mot)
            p0 = P2(m)
            for f in ctrl:
                rows.append((int(p["q"]), float(p["rho"]), int(e), a, g, f, p0, P2(m | (1 << f))))
    d = pd.DataFrame(rows, columns=["q", "rho", "episode", "a", "g", "fact", "p0", "p1"])
    d["dP"] = d.p1 - d.p0
    return d, proven, mot_p


def fit_predict(train, test, with_comb):
    def design(d, c):
        cols = [np.ones(len(d)), -d.a.to_numpy(), np.full(len(d), c)]
        if with_comb:
            cols += [-d.g.to_numpy(), -c * d.g.to_numpy()]
        return np.column_stack(cols)
    X = np.vstack([design(train, 0.0), design(train, 1.0)])
    y = np.concatenate([logit(train.p0.to_numpy()), logit(train.p1.to_numpy())])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = sig(design(test, 1.0) @ coef) - sig(design(test, 0.0) @ coef)
    return pred, coef


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    out, within, perfact = [], [], []
    for task in RUNS:
        for kind in ("symmetric", "no-proof"):
            d, proven, mot_p = collect(task, kind)
            train = d[(d.q == 6) & (d.episode % 2 == 0)]
            test = d[(d.q == 3) & (d.episode % 2 == 1)].copy()
            pa, ca = fit_predict(train, test, False)
            pb, cb = fit_predict(train, test, True)
            test["predA"], test["predB"] = pa, pb
            row = dict(task=task, setup=kind, pairs_train=len(train), pairs_test=len(test), proven_memories_excluded=proven,
                       motif_P_A2=";".join(f"{x:.3f}" for x in mot_p), R2_count=r2(test.dP.to_numpy(), pa),
                       R2_count_comb=r2(test.dP.to_numpy(), pb), coef_count=";".join(f"{x:.3f}" for x in ca),
                       coef_comb=";".join(f"{x:.3f}" for x in cb), mean_dP=test.dP.mean())
            row["R2_gain"] = row["R2_count_comb"] - row["R2_count"]
            shares = []
            for a, gr in test.groupby("a"):
                if len(gr) < 200:
                    continue
                resA = gr.dP - gr.predA
                resB = gr.dP - gr.predB
                v = np.sum((resA - resA.mean()) ** 2)
                share = 1 - np.sum((resB - resB.mean()) ** 2) / v if v > 0 else np.nan
                shares.append(share)
                within.append(dict(task=task, setup=kind, a=a, pairs=len(gr), var_dP=gr.dP.var(), within_share_B_beyond_A=share,
                                   corr_dP_g=np.corrcoef(gr.dP, gr.g)[0, 1] if gr.g.std() > 0 else np.nan))
            row["within_share_mean"] = float(np.nanmean(shares)) if shares else np.nan
            row["combination_needed"] = bool(row["R2_gain"] >= 0.05 and row["within_share_mean"] >= 0.10)
            for rho, gr in test.groupby("rho"):
                row[f"R2_count_rho{rho}"] = r2(gr.dP.to_numpy(), gr.predA.to_numpy())
                row[f"R2_comb_rho{rho}"] = r2(gr.dP.to_numpy(), gr.predB.to_numpy())
            out.append(row)
            for f, gr in d.groupby("fact"):
                perfact.append(dict(task=task, setup=kind, fact=int(f), mean_dP=gr.dP.mean(), share_positive=(gr.dP > 1e-9).mean()))
            print("done", task, kind, flush=True)
    r = pd.DataFrame(out)
    r.to_csv(HERE / "results_by_setup.csv", index=False)
    pd.DataFrame(within).to_csv(HERE / "within_count.csv", index=False)
    pd.DataFrame(perfact).to_csv(HERE / "per_fact.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(r[["task", "setup", "pairs_test", "proven_memories_excluded", "mean_dP", "R2_count", "R2_count_comb", "R2_gain",
             "within_share_mean", "combination_needed"]].round(3).to_string(index=False))
    print(r[["task", "setup", "coef_count", "coef_comb", "motif_P_A2"]].to_string(index=False))


if __name__ == "__main__":
    main()
