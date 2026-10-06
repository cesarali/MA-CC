"""Four independent tests of whether the coarse-grained chain is Markovian.

Each probes a different implication of the Markov property, so they can
disagree -- and where they do, the disagreement is informative.

  1 ORDER        held-out log-likelihood of order-1 vs order-2 chains.
                 Tests: does the previous state add predictive power?
  2 CHAPMAN-KOLMOGOROV  does the fitted one-step kernel raised to the n-th power
                 match the empirical n-step kernel?  Tests longer horizons, which
                 the order test does not reach.
  3 CONDITIONAL INDEPENDENCE  estimate I(S_{t+1}; S_{t-1} | S_t) directly.
                 Zero under the Markov property.  Reported in bits with a
                 permutation null, since the plug-in estimate is biased upward.
  4 LUMPABILITY  within one macrostate, do microstates with different hidden
                 features have different next-macrostate distributions?  This is
                 the direct test of the mechanism: it names what is being lost.
"""
from __future__ import annotations
import sys, json
import numpy as np, pandas as pd

SC = "/Users/rsanchez/Projects/MA-CC/analysis/task003_LLMfree/"
EDGES = [0.33, 0.50, 0.75]
NS = 4
EPS = 1e-12


def load():
    d = pd.read_parquet(SC + "results/data/states_all.parquet")
    d["S"] = np.digitize(d.e, EDGES)
    return d


def seqs(df):
    return [g.sort_values("round_index").S.to_numpy()
            for _, g in df.groupby(["cell_id", "episode_id"])]


# ---------------------------------------------------------------- test 1
def test_order(df, seed=0, alpha=0.5):
    eps = df[["cell_id", "episode_id"]].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    tr_ids = {tuple(k) for k in eps[rng.random(len(eps)) < 0.7]}
    tr, te = [], []
    for k, g in df.sort_values("round_index").groupby(["cell_id", "episode_id"]):
        s = g.S.to_numpy()
        tri = [(s[i-1], s[i], s[i+1]) for i in range(1, len(s)-1)]
        (tr if tuple(k) in tr_ids else te).extend(tri)
    if len(te) < 50:
        return None
    def ll(order):
        cnt = {}
        for p, c, n in tr:
            key = (c,) if order == 1 else (p, c)
            cnt.setdefault(key, {}).setdefault(n, 0)
            cnt[key][n] += 1
        return sum(np.log((cnt.get((c,) if order == 1 else (p, c), {}).get(n, 0) + alpha)
                          / (sum(cnt.get((c,) if order == 1 else (p, c), {}).values()) + alpha*NS))
                   for p, c, n in te) / len(te)
    return ll(2) - ll(1)


# ---------------------------------------------------------------- test 2
def test_ck(df, nmax=5):
    """Total-variation gap between T^n and the empirical n-step kernel."""
    S = seqs(df)
    T = np.full((NS, NS), EPS)
    for s in S:
        for a, b in zip(s[:-1], s[1:]):
            T[a, b] += 1
    T /= T.sum(axis=1, keepdims=True)
    out = {}
    for n in range(2, nmax + 1):
        E = np.full((NS, NS), EPS)
        for s in S:
            for i in range(len(s) - n):
                E[s[i], s[i+n]] += 1
        w = E.sum(axis=1)
        E = E / E.sum(axis=1, keepdims=True)
        P = np.linalg.matrix_power(T, n)
        tv = 0.5 * np.abs(P - E).sum(axis=1)
        m = w > 20
        out[n] = float((tv[m] * w[m]).sum() / w[m].sum()) if m.any() else np.nan
    return out


# ---------------------------------------------------------------- test 3
def cmi(trip):
    """Plug-in I(next ; prev | cur) in bits."""
    df = pd.DataFrame(trip, columns=["p", "c", "n"])
    tot = 0.0
    for c, g in df.groupby("c"):
        w = len(g) / len(df)
        if len(g) < 10:
            continue
        j = pd.crosstab(g.p, g.n, normalize=True).to_numpy()
        pm, nm = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
        nz = j > 0
        tot += w * float(np.sum(j[nz] * np.log2(j[nz] / (pm @ nm)[nz])))
    return tot


def test_ci(df, nperm=200, seed=0):
    trip = [(s[i-1], s[i], s[i+1]) for s in seqs(df) for i in range(1, len(s)-1)]
    obs = cmi(trip)
    rng = np.random.default_rng(seed)
    t = pd.DataFrame(trip, columns=["p", "c", "n"])
    null = []
    for _ in range(nperm):          # shuffle prev WITHIN each cur stratum
        sh = t.copy()
        sh["p"] = t.groupby("c").p.transform(lambda x: rng.permutation(x.to_numpy()))
        null.append(cmi(list(sh.itertuples(index=False, name=None))))
    null = np.array(null)
    return dict(cmi_bits=obs, null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
                excess=float(obs - null.mean()),
                p_value=float((1 + (null >= obs).sum()) / (1 + len(null))))


# ---------------------------------------------------------------- test 4
def test_lumpability(df, feature, nbins=2):
    """Within each macrostate, split microstates by `feature` and compare their
    next-state distributions.  A large gap means the macrostate is not a
    sufficient summary and `feature` is part of what is missing."""
    d = df.sort_values("round_index").copy()
    d["nxt"] = d.groupby(["cell_id", "episode_id"]).S.shift(-1)
    d = d.dropna(subset=["nxt"])
    res = {}
    for s, g in d.groupby("S"):
        if len(g) < 100 or g[feature].nunique() < 2:
            continue
        cut = g[feature].median()
        lo, hi = g[g[feature] <= cut], g[g[feature] > cut]
        if len(lo) < 30 or len(hi) < 30:
            continue
        a = lo.nxt.value_counts(normalize=True).reindex(range(NS), fill_value=0).to_numpy()
        b = hi.nxt.value_counts(normalize=True).reindex(range(NS), fill_value=0).to_numpy()
        res[int(s)] = dict(tv=float(0.5*np.abs(a-b).sum()), n_lo=len(lo), n_hi=len(hi))
    if not res:
        return dict(weighted_tv=np.nan, per_state={})
    w = {k: v["n_lo"] + v["n_hi"] for k, v in res.items()}
    tot = sum(w.values())
    return dict(weighted_tv=float(sum(res[k]["tv"] * w[k] for k in res) / tot), per_state=res)
