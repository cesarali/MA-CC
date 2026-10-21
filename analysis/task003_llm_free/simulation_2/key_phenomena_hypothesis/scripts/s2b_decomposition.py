"""Step 2b: direct / indirect decomposition of the reading reversal, per episode, with bootstrap intervals.
Task: key_phenomena_hypothesis/reports/2026-10-21_step2_reading_mechanism (agent task of 21 October 2026).

Data: results/simulation_2/2026-10-20_sim2_reading_mechanism (both setups, 8 rho, q 3/6/9/12, false and silent arms,
qc 12, rate 1, gate on, B 30, W 48, 1,000 episodes). No new simulations.

Per episode e, setup, rho, q, at t = 30 (agent_snapshots grid_index 120: each agent's remembered facts):
  F_i^C  agent-origin facts remembered by agent i, controller arm (facts some agent starts with; the symmetric A2
         pool's 2 facts that agents also start with count as agent-origin)
  H_i^C  controller-exclusive facts remembered by agent i, controller arm (A2 pool minus agent-origin)
  F_i^S  facts remembered by agent i, silent arm (only agent-origin facts can circulate there; checked)
  p2(F)  exact P(A2 | F) (llmfree_core.world.World.posterior)
  g_direct   = mean_i [p2(F_i^C | H_i^C) - p2(F_i^C)]
  g_indirect = mean_i p2(F_i^C) - mean_i p2(F_i^S)
  g_belief   = g_direct + g_indirect            (identity checked per episode)
  g_vote     = x_A2(controller) - x_A2(silent)  (share of agents whose last vote is A2, among agents who have voted;
                                                 snapshots.parquet, the convention of all reports)
Reading increments D_X(q_lo -> q_hi) = mean_e g_X(q_hi) - mean_e g_X(q_lo), for X in direct, indirect, belief, vote,
and E = D_vote - D_belief.

Bootstrap: 2,000 resamples of episode IDs per (setup, rho), the same IDs drawn for all four q and both arms. This
keeps controller/silent pairing within each q and the pairing across q, which holds because every random stream is
keyed by (seed, episode, purpose) and never by q or arm (simulation_2.py, keyed_rng). 95% percentile intervals.
D_belief is computed from the summed episode values in each replicate.

Also: validation of the q = 3, 6, 12 light tables against the fine runs; time since each agent's last vote at t = 30
(vote_age), for the snapshot-versus-vote limitation.

Writes results/key_phenomena/s2b_episodes.parquet and, in key_phenomena_hypothesis/tables/, step2b_D.csv (every D
and E with intervals and signs), step2b_G.csv (levels), step2b_reversals.csv (Q3 classification),
step2b_validation.csv, step2b_vote_age.csv.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/scripts/s2b_decomposition.py
"""
from __future__ import annotations
import multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from s1_mechanism import classes  # noqa: E402
from s2_mechanism import cells, validate  # noqa: E402

OUT = ROOT / "results" / "key_phenomena"
TAB = HERE.parent / "tables"
QS = [3, 6, 9, 12]
STEPS = [(3, 6), (6, 9), (9, 12)]
RHOS = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
T_INDEX, N_BOOT, SEED = 120, 2000, 20261021
QUANT = ["direct", "indirect", "belief", "vote"]


def x2_at30(cell):
    s = pd.read_parquet(cell / "snapshots.parquet", columns=["episode", "grid_index", "share_A2", "share_voted"])
    s = s[s.grid_index == T_INDEX].set_index("episode").sort_index()
    return s.share_A2 / s.share_voted


def episode_values(job):
    (setup, rho, q), (fc, sc) = job
    world, agent, ctrl = classes(setup.replace("task003_", ""))
    cache = {}

    def p2(m):
        v = cache.get(m)
        if v is None:
            v = cache[m] = world.posterior(m)[2]
        return v

    def snaps(cell):
        a = pd.read_parquet(cell / "agent_snapshots.parquet", columns=["episode", "grid_index", "agent", "active",
                                                                          "vote_age"])
        return a[a.grid_index == T_INDEX].sort_values(["episode", "agent"])

    c, s = snaps(fc), snaps(sc)
    ca, sa = c.active.to_numpy(np.int64), s.active.to_numpy(np.int64)
    assert ((sa & ~agent) == 0).all(), "silent memory holds a non-agent-origin fact"
    assert ((ca & ~(agent | ctrl)) == 0).all(), "controller memory holds an unclassified fact"
    p_all = np.array([p2(int(m)) for m in ca])
    p_agent = np.array([p2(int(m) & agent) for m in ca])
    p_sil = np.array([p2(int(m)) for m in sa])
    ep = c.episode.to_numpy()
    df = pd.DataFrame({"episode": ep, "d": p_all - p_agent, "ca": p_agent, "cs": p_sil, "tot": p_all - p_sil})
    g = df.groupby("episode").mean()
    out = pd.DataFrame({"g_direct": g.d, "g_indirect": g.ca - g.cs, "g_belief_direct": g.tot})
    out["identity_error"] = (out.g_direct + out.g_indirect - out.g_belief_direct).abs()
    out["g_belief"] = out.g_direct + out.g_indirect
    out["g_vote"] = (x2_at30(fc) - x2_at30(sc)).reindex(out.index)
    out["vote_age_ctrl"] = c.groupby("episode").vote_age.mean()
    out["mem_ctrl_facts"] = pd.Series([bin(int(m) & ctrl).count("1") for m in ca], index=c.index).groupby(ep).mean().values
    out = out.reset_index()
    out.insert(0, "setup", setup.replace("task003_", ""))
    out.insert(1, "rho", rho)
    out.insert(2, "q", q)
    return out


def bootstrap(job):
    (setup, rho), d = job
    rng = np.random.default_rng([SEED, int(rho * 1000), 0 if setup == "symmetric" else 1])
    arr = {x: np.stack([d[d.q == q].sort_values("episode")[f"g_{x}"].to_numpy() for q in QS]) for x in QUANT}
    n = arr["vote"].shape[1]
    idx = rng.integers(0, n, size=(N_BOOT, n))
    rows = []
    means = {x: arr[x].mean(axis=1) for x in QUANT}
    boot = {x: arr[x][:, idx].mean(axis=2) for x in QUANT}                # (q, N_BOOT)
    boot["belief"] = boot["direct"] + boot["indirect"]                     # sum of episode-level parts per replicate
    for k, q in enumerate(QS):
        rows.append(("G", setup, rho, q, None, *[v for x in QUANT for v in
                                                  (means[x][k], *np.percentile(boot[x][k], [2.5, 97.5]))]))
    drows = []
    for lo, hi in STEPS:
        a, b = QS.index(lo), QS.index(hi)
        rec = dict(setup=setup, rho=rho, step=f"{lo}->{hi}")
        for x in QUANT:
            est = means[x][b] - means[x][a]
            bs = boot[x][b] - boot[x][a]
            l, u = np.percentile(bs, [2.5, 97.5])
            rec[f"D_{x}"], rec[f"D_{x}_lo"], rec[f"D_{x}_hi"] = est, l, u
            rec[f"D_{x}_sign"] = "+" if l > 0 else ("-" if u < 0 else "0")
        e = (means["vote"][b] - means["vote"][a]) - (means["belief"][b] - means["belief"][a])
        eb = (boot["vote"][b] - boot["vote"][a]) - (boot["belief"][b] - boot["belief"][a])
        rec["E"], (rec["E_lo"], rec["E_hi"]) = e, np.percentile(eb, [2.5, 97.5])
        rec["E_sign"] = "+" if rec["E_lo"] > 0 else ("-" if rec["E_hi"] < 0 else "0")
        drows.append(rec)
    G = pd.DataFrame(rows, columns=["kind", "setup", "rho", "q", "step"] +
                     [f"G_{x}{s}" for x in QUANT for s in ("", "_lo", "_hi")]).drop(columns=["kind", "step"])
    return G, pd.DataFrame(drows)


def reversals(D):
    """Q3: within each step column, every pair of neighbouring resolved cells (CI excludes 0) of the total response
    with opposite signs, after skipping unresolved cells in between; classified by D_direct and D_indirect."""
    rows = []
    for total in ("belief", "vote"):
        for (setup, step), d in D.groupby(["setup", "step"]):
            d = d.sort_values("rho")
            res = d[d[f"D_{total}_sign"] != "0"]
            for (_, a), (_, b) in zip(res.iloc[:-1].iterrows(), res.iloc[1:].iterrows()):
                if a[f"D_{total}_sign"] == b[f"D_{total}_sign"]:
                    continue
                prot = a if a[f"D_{total}_sign"] == "-" else b          # the protective side
                amp = b if prot is a else a
                ds = (amp.D_direct_sign, prot.D_direct_sign)
                if ds == ("+", "-"):
                    kind = "direct reversal"
                elif ds == ("+", "+") and prot.D_indirect_sign == "-":
                    kind = "indirect override"
                else:
                    kind = "mixed/uncertain"
                rows.append(dict(total=total, setup=setup, step=step, rho_low=a.rho, rho_high=b.rho,
                                 rho_amplifying=amp.rho, rho_protective=prot.rho, kind=kind,
                                 D_direct_amp=amp.D_direct, D_direct_prot=prot.D_direct,
                                 D_indirect_amp=amp.D_indirect, D_indirect_prot=prot.D_indirect,
                                 D_direct_sign_amp=amp.D_direct_sign, D_direct_sign_prot=prot.D_direct_sign,
                                 D_indirect_sign_prot=prot.D_indirect_sign))
    return pd.DataFrame(rows)


def main():
    cs = cells()
    jobs = [((s, r, q), (c, cs[(s, r, q, "silent")])) for (s, r, q, a), c in cs.items() if a == "false"]
    with multiprocessing.Pool(12) as pool:
        val = [v for v in pool.map(validate, cs.items()) if v is not None]
        ep = pd.concat(pool.map(episode_values, jobs), ignore_index=True)
        parts = pool.map(bootstrap, list(ep.groupby(["setup", "rho"])))
    v = pd.DataFrame(val, columns=["setup", "rho", "q", "arm", "identical"])
    v.to_csv(TAB / "step2b_validation.csv", index=False)
    print(f"validation: {int(v.identical.sum())}/{len(v)} light tables identical to the fine runs")
    print(f"identity: max |g_direct + g_indirect - g_belief| = {ep.identity_error.max():.2e}")
    ep.to_parquet(OUT / "s2b_episodes.parquet", index=False)
    G = pd.concat([p[0] for p in parts], ignore_index=True).sort_values(["setup", "rho", "q"])
    D = pd.concat([p[1] for p in parts], ignore_index=True).sort_values(["setup", "step", "rho"])
    G.round(5).to_csv(TAB / "step2b_G.csv", index=False)
    D.round(5).to_csv(TAB / "step2b_D.csv", index=False)
    reversals(D).round(5).to_csv(TAB / "step2b_reversals.csv", index=False)
    a = ep.groupby(["setup", "rho"]).vote_age_ctrl.mean().reset_index()
    a["gamma"] = -np.log(a.rho.where(a.rho < 1, np.nan))
    # an agent acts at rate 1, so the time since its last vote is about Exponential(1) (observed mean: vote_age_ctrl);
    # the expected share of the facts it held then that are forgotten by t = 30 is E[1 - rho^A] = gamma / (1 + gamma)
    a["expected_fact_loss_since_vote"] = (a.gamma / (1 + a.gamma)).fillna(0)
    a.round(4).to_csv(TAB / "step2b_vote_age.csv", index=False)
    print("->", TAB)


if __name__ == "__main__":
    main()
