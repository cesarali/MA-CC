"""Eight coarse-grained states of the swarm: are they Markov, and how readable is the target in them?

Data: Simulation 2, task003-symmetric, the cells with per-agent tables (q 6, qc 12, rate 1, gate on,
proof stop on): rho 0.75 (2026-10-09 grid) and rho 1 (2026-10-12 full run); silent, target A0, target A2;
1,000 episodes each. Snapshots at integer times t = 0..40.

Everything is computed from the facts each agent remembers (and, for V, from its last vote), using only
the world's logic: p_ik = P(A_k | K_i) is what agent i's facts imply, the same for any reasoner.

The eight candidate states (cut points identical across arms; medians per rho, pooled over arms and times; upper bin is >= median, so for b it is "the board proves A0"):
  FV  fact-implied votes: each agent's facts favour the allocation(s) with the largest p_ik (ties split);
      d = share(A0) - share(A2), 3 equal bins on [-1, 1]
  S1  m_A0 (mean of p_i0), 4 bins at 0.33 / 0.50 / 0.75
  S2  s = m_A0 - m_A2, 3 equal bins on [-1, 1]
  S3  (m_A0, m_A2), each in 3 equal bins on [0, 1]
  S4  S2 x disagreement D = (1/N) sum_i sum_k (p_ik - m_k)^2, 2 bins at the median
  S5  S2 x memory fragility F (r = 0.75): mean over agents of E[TV(P(.|K'), P(.|K))], K' keeps each fact
      with probability r (exact enumeration up to 8 facts, else 256 fixed-seed draws), 2 bins at the median
  S6  S2 x board evidence b = b_0 - b_2, b_k = mean over agents of P(A_k | facts in that agent's visible
      window: the 48 most recent posts not its own), 2 bins at the median
  V   votes: x = share voting A0 - share voting A2 (last votes), 3 equal bins on [-1, 1]

Markov tests (window t = 5..30, steps of 1; pass rules fixed before running):
  order   held-out log-likelihood gain of order 2 over order 1 (10 random 70/30 episode splits),
          pass if the mean gain < 0.005 nats per step
  CK      Chapman-Kolmogorov: weighted total-variation gap between T^5 and the empirical 5-step kernel,
          pass if < 0.05
  CMI     I(S_t+1; S_t-1 | S_t) against 200 within-stratum permutations, pass if p > 0.05
  lump    lumpability: within each state, median split by each hidden feature not in the state
          (D, F, b, mean facts held, decisive facts alive), weighted TV between the halves' next-state
          distributions; pass if the largest is < 0.05
A chain that sits in one state for >= 95% of the window is reported as 'absorbed' (the tests say nothing).

Readability: I(Z; X) at t = 10 and 30, Z = target (fair coin between the two controlled arms), plug-in from
1,000 episodes per arm, 95% bootstrap interval, and root-mean-square error at 100 paths (200 subsets).
Reference rows: the one-agent vote state used before, and its fact-implied analogue.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/coarse_graining/code/markov_study.py [--date 2026-10-15]
"""
from __future__ import annotations
import argparse, itertools, pathlib, sys, time, zlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from llmfree_core.world import World, bits  # noqa: E402
from llmfree_core.setups import load_setup  # noqa: E402

RES = ROOT / "results"
CELLS = {
    0.75: {"silent": RES / "simulation_2/2026-10-09_sim2_grid/cells/task003_symmetric__silent__q6__rho0.75",
           "truth": RES / "simulation_2/2026-10-09_sim2_grid/cells/task003_symmetric__truth__q6__qc12__rho0.75__lc1__gateon__stopon",
           "false": RES / "simulation_2/2026-10-09_sim2_grid/cells/task003_symmetric__false__q6__qc12__rho0.75__lc1__gateon__stopon"},
    1.0: {"silent": RES / "simulation_2/2026-10-12_sim2_main_rho1_full/cells/task003_symmetric__silent",
          "truth": RES / "simulation_2/2026-10-12_sim2_main_rho1_full/cells/task003_symmetric__truth__lc1__gateon__stopon",
          "false": RES / "simulation_2/2026-10-12_sim2_main_rho1_full/cells/task003_symmetric__false__lc1__gateon__stopon"},
}
N_AGENTS, W, R_FRAG, DT = 24, 48, 0.75, 0.25
TIMES = list(range(0, 41))
WINDOW = (5, 30)
CANDS = ["FV", "S1", "S2", "S3", "S4", "S5", "S6", "V"]
FEATURES = ["D", "F", "b", "nfacts", "dec_alive"]
HIDDEN = {"FV": FEATURES, "S1": FEATURES, "S2": FEATURES, "S3": FEATURES, "S4": ["F", "b", "nfacts", "dec_alive"],
          "S5": ["D", "b", "nfacts", "dec_alive"], "S6": ["D", "F", "nfacts", "dec_alive"], "V": FEATURES}
PASS = {"order": 0.005, "ck": 0.05, "cmi_p": 0.05, "lump": 0.05}


# ------------------------------------------------------------------ fact-based features
class Facts:
    def __init__(self):
        self.w = World(ROOT / "llmfree_core/world_task003.json")
        setup = load_setup("task003_symmetric", ROOT / "experimental_setup", self.w)
        import json
        ag = json.loads((ROOT / "experimental_setup/task003_symmetric/agents.json").read_text())
        self.decisive = [self.w.index[f["fact_id"]] for f in ag["facts"] if f.get("decisive")]
        assert len(self.decisive) == 6
        self.frag_cache: dict[int, float] = {}

    def post(self, s: int):
        return self.w.posterior(int(s))

    def fragility(self, s: int) -> float:
        """E over K' (each fact kept with probability r) of TV(P(.|K'), P(.|K))."""
        s = int(s)
        hit = self.frag_cache.get(s)
        if hit is not None:
            return hit
        p = np.array(self.post(s))
        fs = list(bits(s))
        n = len(fs)
        if n == 0:
            val = 0.0
        else:
            fbits = np.array([1 << f for f in fs], dtype=np.int64)
            if n <= 8:                                            # exact: every subset, with its probability
                keep = np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.int64)
                k = keep.sum(1)
                wts = R_FRAG ** k * (1 - R_FRAG) ** (n - k)
            else:                                                 # 256 fixed-seed draws
                rng = np.random.default_rng(zlib.crc32(s.to_bytes(8, "little")))
                keep = (rng.random((256, n)) < R_FRAG).astype(np.int64)
                wts = np.full(256, 1 / 256)
            subs = keep @ fbits
            tv = np.array([0.5 * np.abs(np.array(self.post(int(m))) - p).sum() for m in subs])
            val = float(wts @ tv)
        self.frag_cache[s] = float(val)
        return float(val)


def build_states(fx: Facts, cell: pathlib.Path) -> pd.DataFrame:
    """One row per (episode, t): continuous features of the swarm, all from facts or votes."""
    gis = [int(t / DT) for t in TIMES]
    a = pd.read_parquet(cell / "agent_snapshots.parquet", columns=["episode", "grid_index", "agent", "active", "last_vote"])
    a = a[a.grid_index.isin(gis)].sort_values(["episode", "grid_index", "agent"])
    a["t"] = a.grid_index * DT
    act = a.active.to_numpy().astype(np.int64)
    uniq = np.unique(act)
    P = {int(s): fx.post(int(s)) for s in uniq}
    Fr = {int(s): fx.fragility(int(s)) for s in uniq}
    p = np.array([P[int(s)] for s in act])                      # (rows, 3)
    a["p0"], a["p1"], a["p2"] = p[:, 0], p[:, 1], p[:, 2]
    a["frag"] = [Fr[int(s)] for s in act]
    a["nf"] = [int(s).bit_count() for s in act]
    mx = p.max(axis=1, keepdims=True)
    top = np.abs(p - mx) < 1e-9
    fv = top / top.sum(axis=1, keepdims=True)                     # ties split
    a["fv0"], a["fv1"], a["fv2"] = fv[:, 0], fv[:, 1], fv[:, 2]
    lv = a.last_vote.to_numpy()
    a["x0"], a["x1"], a["x2"], a["voted"] = (lv == 0) * 1.0, (lv == 1) * 1.0, (lv == 2) * 1.0, (lv >= 0) * 1.0
    dmask = np.zeros(len(a), np.int64)
    for f in fx.decisive:
        a[f"dec{f}"] = (act >> f) & 1
    g = a.groupby(["episode", "t"])
    st = g[["p0", "p1", "p2", "frag", "nf", "fv0", "fv1", "fv2", "x0", "x1", "x2", "voted"]].mean()
    st.columns = ["m0", "m1", "m2", "F", "nfacts", "fv0", "fv1", "fv2", "x0", "x1", "x2", "voted"]
    a = a.join(st[["m0", "m1", "m2"]], on=["episode", "t"])
    a["dev"] = (a.p0 - a.m0) ** 2 + (a.p1 - a.m1) ** 2 + (a.p2 - a.m2) ** 2
    st["D"] = a.groupby(["episode", "t"]).dev.mean()
    st["dec_alive"] = a.groupby(["episode", "t"])[[f"dec{f}" for f in fx.decisive]].max().sum(axis=1)
    b = board_evidence(fx, cell, st.index)
    st = st.join(b)
    del dmask
    return st.reset_index()


def board_evidence(fx: Facts, cell: pathlib.Path, index) -> pd.DataFrame:
    """b_k = mean over agents of P(A_k | union of facts in the agent's visible window) at each snapshot time."""
    posts = pd.read_parquet(cell / "posts.parquet", columns=["episode", "post_id", "t", "author", "fact"])
    posts = posts.sort_values(["episode", "t", "post_id"])
    out = []
    agents = np.arange(N_AGENTS)
    for ep, g in posts.groupby("episode", sort=True):
        tt, au, fb = g.t.to_numpy(), g.author.to_numpy(), (np.int64(1) << g.fact.to_numpy().astype(np.int64))
        for t in TIMES:
            n = np.searchsorted(tt, t, side="left")              # posts strictly before the snapshot
            lo = max(0, n - 2 * W - 2)
            ra, rf = au[lo:n][::-1], fb[lo:n][::-1]
            b0 = b2 = 0.0
            for i in agents:
                sel = rf[ra != i][:W]
                u = int(np.bitwise_or.reduce(sel)) if len(sel) else 0
                pp = fx.post(u)
                b0 += pp[0]
                b2 += pp[2]
            out.append((ep, float(t), b0 / N_AGENTS, b2 / N_AGENTS))
    b = pd.DataFrame(out, columns=["episode", "t", "b0", "b2"]).set_index(["episode", "t"])
    b["b"] = b.b0 - b.b2
    return b.reindex(index)


# ------------------------------------------------------------------ discretisation
def bin3(x):
    return np.digitize(x, [-1 / 3, 1 / 3])


def discretise(st: pd.DataFrame, med: dict) -> pd.DataFrame:
    s = st.copy()
    s["FV"] = bin3(s.fv0 - s.fv2)
    s["S1"] = np.digitize(s.m0, [0.33, 0.50, 0.75])
    s["S2"] = bin3(s.m0 - s.m2)
    s["S3"] = 3 * np.digitize(s.m0, [1 / 3, 2 / 3]) + np.digitize(s.m2, [1 / 3, 2 / 3])
    s["S4"] = 2 * s.S2 + (s.D >= med["D"])
    s["S5"] = 2 * s.S2 + (s.F >= med["F"])
    s["S6"] = 2 * s.S2 + (s.b >= med["b"])
    s["V"] = bin3(s.x0 - s.x2)
    return s


# ------------------------------------------------------------------ Markov tests
def sequences(s: pd.DataFrame, col: str):
    w = s[(s.t >= WINDOW[0]) & (s.t <= WINDOW[1])].sort_values(["episode", "t"])
    return {ep: g[col].to_numpy() for ep, g in w.groupby("episode")}


def test_order(seqs: dict, K: int, n_splits=10, alpha=0.5):
    eps = np.array(sorted(seqs))
    gains = []
    for sd in range(n_splits):
        rng = np.random.default_rng(sd)
        tr = set(eps[rng.random(len(eps)) < 0.7])
        c1, c2 = np.zeros((K, K)), np.zeros((K, K, K))
        test = []
        for ep, s in seqs.items():
            tri = np.stack([s[:-2], s[1:-1], s[2:]], axis=1)
            if ep in tr:
                np.add.at(c1, (tri[:, 1], tri[:, 2]), 1)
                np.add.at(c2, (tri[:, 0], tri[:, 1], tri[:, 2]), 1)
            else:
                test.append(tri)
        te = np.concatenate(test)
        p1 = (c1 + alpha) / (c1.sum(1, keepdims=True) + alpha * K)
        p2 = (c2 + alpha) / (c2.sum(2, keepdims=True) + alpha * K)
        gains.append(np.mean(np.log(p2[te[:, 0], te[:, 1], te[:, 2]])) - np.mean(np.log(p1[te[:, 1], te[:, 2]])))
    return float(np.mean(gains)), float(np.std(gains))


def test_ck(seqs: dict, K: int, nmax=5):
    T = np.zeros((K, K))
    for s in seqs.values():
        np.add.at(T, (s[:-1], s[1:]), 1)
    occ = T.sum(1)
    T = np.where(occ[:, None] > 0, T / np.maximum(occ[:, None], 1), np.eye(K))
    out = {}
    for n in range(2, nmax + 1):
        E = np.zeros((K, K))
        for s in seqs.values():
            np.add.at(E, (s[:-n], s[n:]), 1)
        w = E.sum(1)
        En = E / np.maximum(w[:, None], 1)
        tv = 0.5 * np.abs(np.linalg.matrix_power(T, n) - En).sum(1)
        m = w > 20
        out[n] = float((tv[m] * w[m]).sum() / w[m].sum()) if m.any() else np.nan
    return out


def cmi_bits(tri: np.ndarray, K: int) -> float:
    c = np.zeros((K, K, K))
    np.add.at(c, (tri[:, 1], tri[:, 0], tri[:, 2]), 1)       # [cur, prev, next]
    tot = c.sum()
    val = 0.0
    for k in range(K):
        j = c[k]
        nk = j.sum()
        if nk < 10:
            continue
        j = j / nk
        pm, nm = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
        nz = j > 0
        val += nk / tot * float(np.sum(j[nz] * np.log2(j[nz] / (pm @ nm)[nz])))
    return val


def test_cmi(seqs: dict, K: int, nperm=200, seed=0):
    tri = np.concatenate([np.stack([s[:-2], s[1:-1], s[2:]], axis=1) for s in seqs.values()])
    obs = cmi_bits(tri, K)
    rng = np.random.default_rng(seed)
    order = np.argsort(tri[:, 1], kind="stable")
    tri = tri[order]
    bounds = np.flatnonzero(np.diff(tri[:, 1])) + 1
    groups = np.split(np.arange(len(tri)), bounds)
    null = []
    for _ in range(nperm):
        sh = tri.copy()
        for g in groups:
            sh[g, 0] = rng.permutation(sh[g, 0])
        null.append(cmi_bits(sh, K))
    null = np.array(null)
    return dict(cmi=obs, excess=float(obs - null.mean()), p=float((1 + (null >= obs).sum()) / (1 + nperm)))


def test_lump(s: pd.DataFrame, col: str, K: int, feats: list[str]):
    w = s[(s.t >= WINDOW[0]) & (s.t <= WINDOW[1])].sort_values(["episode", "t"]).copy()
    w["nxt"] = w.groupby("episode")[col].shift(-1)
    w = w.dropna(subset=["nxt"])
    w["nxt"] = w.nxt.astype(int)
    res = {}
    for f in feats:
        tot, acc = 0, 0.0
        for k, g in w.groupby(col):
            if len(g) < 100 or g[f].nunique() < 2:
                continue
            cut = g[f].median()
            lo, hi = g[g[f] <= cut], g[g[f] > cut]
            if len(lo) < 30 or len(hi) < 30:
                continue
            a = np.bincount(lo.nxt, minlength=K) / len(lo)
            b = np.bincount(hi.nxt, minlength=K) / len(hi)
            acc += 0.5 * np.abs(a - b).sum() * len(g)
            tot += len(g)
        res[f] = acc / tot if tot else np.nan
    return res


# ------------------------------------------------------------------ readability
def mi_swarm(xp: np.ndarray, xm: np.ndarray, K: int) -> float:
    pp, pm = np.bincount(xp, minlength=K) / len(xp), np.bincount(xm, minlength=K) / len(xm)
    q = 0.5 * (pp + pm)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return 0.5 * kl(pp, q) + 0.5 * kl(pm, q)


def mi_agent(yp: np.ndarray, ym: np.ndarray) -> float:
    """One-agent state: per-episode share vectors, averaged."""
    pp, pm = yp.mean(0), ym.mean(0)
    pp, pm = pp / pp.sum(), pm / pm.sum()
    q = 0.5 * (pp + pm)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return 0.5 * kl(pp, q) + 0.5 * kl(pm, q)


def readability(states: dict, rho: float, t: float, B=500, nsub=200):
    rows = []
    sp = states[(rho, "truth")]
    sm = states[(rho, "false")]
    sp, sm = sp[sp.t == t].sort_values("episode"), sm[sm.t == t].sort_values("episode")
    rng = np.random.default_rng(1)
    n = len(sp)
    items = [(c, "swarm") for c in CANDS] + [("one agent, votes", "agent"), ("one agent, fact-implied votes", "agent")]
    for name, kind in items:
        if kind == "swarm":
            K = int(max(sp[name].max(), sm[name].max())) + 1
            xp, xm = sp[name].to_numpy(), sm[name].to_numpy()
            f = lambda ip, im: mi_swarm(xp[ip], xm[im], K)  # noqa: E731
            ncell = len(np.union1d(np.unique(xp), np.unique(xm)))
        else:
            cols = ["x0", "x1", "x2"] if "votes" in name and "fact" not in name else ["fv0", "fv1", "fv2"]
            yp, ym = sp[cols].to_numpy(), sm[cols].to_numpy()
            f = lambda ip, im: mi_agent(yp[ip], ym[im])  # noqa: E731
            ncell = 3
        full = f(np.arange(n), np.arange(len(sm)))
        boot = [f(rng.integers(0, n, n), rng.integers(0, len(sm), len(sm))) for _ in range(B)]
        sub = [f(rng.choice(n, 100, replace=False), rng.choice(len(sm), 100, replace=False)) for _ in range(nsub)]
        rows.append(dict(rho=rho, t=t, state=name, cells_used=ncell, I=full,
                         lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                         rmse_100=float(np.sqrt(np.mean((np.array(sub) - full) ** 2)))))
    return rows


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-15")
    a = ap.parse_args()
    out = RES / "coarse_graining" / a.date
    out.mkdir(parents=True, exist_ok=True)
    fx = Facts()

    raw = {}
    for rho, arms in CELLS.items():
        for arm, cell in arms.items():
            f = out / f"states_rho{rho:g}_{arm}.parquet"
            if f.exists():
                raw[(rho, arm)] = pd.read_parquet(f)
            else:
                t0 = time.time()
                raw[(rho, arm)] = build_states(fx, cell)
                raw[(rho, arm)].to_parquet(f)
                print(f"states rho {rho} {arm}: {time.time() - t0:.0f} s", flush=True)

    states, meds = {}, {}
    for rho in CELLS:
        pool = pd.concat([raw[(rho, arm)] for arm in CELLS[rho]])
        pool = pool[(pool.t >= WINDOW[0]) & (pool.t <= WINDOW[1])]
        meds[rho] = {k: float(pool[k].median()) for k in ("D", "F", "b")}
        for arm in CELLS[rho]:
            states[(rho, arm)] = discretise(raw[(rho, arm)], meds[rho])
    pd.DataFrame(meds).T.to_csv(out / "markov_medians.csv")

    rows = []
    for (rho, arm), s in states.items():
        for c in CANDS:
            K = int(s[c].max()) + 1
            seqs = sequences(s, c)
            allv = np.concatenate(list(seqs.values()))
            top = np.bincount(allv, minlength=K).max() / len(allv)
            r = dict(rho=rho, arm=arm, state=c, cells_visited=int(len(np.unique(allv))), top_share=top)
            if top >= 0.95:
                r.update(status="absorbed")
            else:
                g, gsd = test_order(seqs, K)
                ck = test_ck(seqs, K)
                ci = test_cmi(seqs, K)
                lump = test_lump(s, c, K, HIDDEN[c])
                worst = max(lump, key=lambda k: -1 if np.isnan(lump[k]) else lump[k])
                r.update(status="tested", order_gain=g, order_sd=gsd, ck2=ck[2], ck3=ck[3], ck5=ck[5],
                         cmi=ci["cmi"], cmi_excess=ci["excess"], cmi_p=ci["p"],
                         lump_max=lump[worst], lump_worst=worst, **{f"lump_{k}": v for k, v in lump.items()})
                r.update(pass_order=g < PASS["order"], pass_ck=ck[5] < PASS["ck"], pass_cmi=ci["p"] > PASS["cmi_p"],
                         pass_lump=lump[worst] < PASS["lump"])
            rows.append(r)
            print(rho, arm, c, r.get("status"), {k: (round(v, 4) if isinstance(v, float) else v)
                                                for k, v in r.items() if k.startswith(("pass", "order_gain", "ck5", "cmi_p", "lump_max"))}, flush=True)
    pd.DataFrame(rows).to_csv(out / "markov_tests.csv", index=False)

    mi = []
    for rho in CELLS:
        for t in (10.0, 30.0):
            mi += readability(states, rho, t)
    pd.DataFrame(mi).to_csv(out / "readability.csv", index=False)
    print("written to", out)


if __name__ == "__main__":
    main()
