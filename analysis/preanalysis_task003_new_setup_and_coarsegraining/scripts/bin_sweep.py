"""Does the number of epistemic bins matter?  Sweep 2..6 states.

Two binning schemes are compared at each size:
  quantile  equal-mass bins from the pooled empirical distribution
  semantic  cuts at interpretable points (the prior 1/3, a majority 1/2, 3/4)

Scored by the within-arm Markov-order test: held-out log-likelihood of a
first-order chain, and how much a second-order chain improves on it.  A small
gain means the state is close to sufficient.
"""
import json, sys
import numpy as np, pandas as pd

d = pd.read_parquet("/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/results/data/states.parquet")

def heldout(sub, col, alpha=0.5, seed=0):
    eps = sub[["cell_id", "episode_id"]].drop_duplicates()
    rng = np.random.default_rng(seed)
    tr_ids = set(map(tuple, eps[rng.random(len(eps)) < 0.7].to_numpy()))
    tr, te = [], []
    for k, g in sub.sort_values("round_index").groupby(["cell_id", "episode_id"]):
        s = g[col].to_numpy()
        tri = [(s[i-1], s[i], s[i+1]) for i in range(1, len(s) - 1)]
        (tr if tuple(k) in tr_ids else te).extend(tri)
    if len(te) < 50:
        return None
    states = sorted({x for t in tr + te for x in t})
    def ll(order):
        cnt = {}
        for p, c, n in tr:
            key = (c,) if order == 1 else (p, c)
            cnt.setdefault(key, {}).setdefault(n, 0)
            cnt[key][n] += 1
        tot = 0.0
        for p, c, n in te:
            row = cnt.get((c,) if order == 1 else (p, c), {})
            tot += np.log((row.get(n, 0) + alpha) / (sum(row.values()) + alpha * len(states)))
        return tot / len(te)
    a, b = ll(1), ll(2)
    return dict(n_test=len(te), ll1=a, ll2=b, gain=b - a, n_states=len(states))

SEMANTIC = {2: [0.50], 3: [0.40, 0.60], 4: [0.33, 0.50, 0.75],
            5: [0.25, 0.3334, 0.50, 0.75], 6: [0.20, 0.3334, 0.45, 0.60, 0.80]}
res = []
for k in (2, 3, 4, 5, 6):
    qs = np.quantile(d.e, np.linspace(0, 1, k + 1)[1:-1])
    for scheme, edges in (("quantile", list(qs)), ("semantic", SEMANTIC[k])):
        col = f"B{k}_{scheme}"
        d[col] = np.digitize(d.e, edges)
        for arm, sub in (("silent", d[d.arm == "silent"]), ("A2-target", d[d.arm == "A2"])):
            r = heldout(sub, col)
            if r:
                res.append(dict(bins=k, scheme=scheme, arm=arm, **r))
df = pd.DataFrame(res)
df.to_csv("/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/02_markov_tests/bin_sweep.csv", index=False)
for arm in ("silent", "A2-target"):
    print(f"\n=== {arm} ===")
    t = df[df.arm == arm].pivot_table(index="bins", columns="scheme", values=["ll1", "gain"])
    print(t.round(4).to_string())
print("\n=== combined ranking (A2-target: absolute fit and Markov gap) ===")
a = df[df.arm == "A2-target"].sort_values("gain")
print(a[["bins", "scheme", "n_states", "ll1", "gain"]].round(4).to_string(index=False))
