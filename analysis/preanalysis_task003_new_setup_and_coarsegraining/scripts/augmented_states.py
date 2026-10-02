"""Does adding a second coordinate to the macrostate restore the Markov property?

The lumpability test says the population mean discards something.  Here we put
each candidate back in as an explicit second dimension and re-run the order test.
A state is better if the order gain falls WITHOUT the state space growing so much
that estimation collapses.
"""
import sys
sys.path.insert(0, "/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/scripts")
import numpy as np, pandas as pd
from four_tests import test_order, test_ci, EDGES, SC

d = pd.read_parquet(SC + "results/data/states_full.parquet")
d["S_e"] = np.digitize(d.e, EDGES)
rows = []
for rho in (0.75, 1.00):
    for arm in ("silent", "A2"):
        sub = d[(d.epistemic_persistence == rho) & (d.arm == arm)].copy()
        if len(sub) < 300:
            continue
        # median splits computed within this arm, so each extra dimension is binary
        for feat in ("spread", "dec_frac", "pool_frac", "mean_nfacts"):
            sub[f"b_{feat}"] = (sub[feat] > sub[feat].median()).astype(int)
        variants = {"e only (4)": "S_e"}
        for feat in ("spread", "dec_frac", "pool_frac", "mean_nfacts"):
            col = f"S_e_{feat}"
            sub[col] = sub.S_e * 2 + sub[f"b_{feat}"]
            variants[f"e x {feat} (8)"] = col
        for name, col in variants.items():
            s2 = sub.rename(columns={col: "S"})
            gains = [g for g in (test_order(s2, seed=k) for k in range(12)) if g is not None]
            if not gains:
                continue
            rows.append(dict(rho=rho, arm=arm, state=name, n_states=int(s2.S.nunique()),
                             order_gain=float(np.mean(gains)), sd=float(np.std(gains))))
            print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv(SC + "30_coarse_graining/augmented_states.csv", index=False)
print("\n=== order gain (lower = more Markovian) ===", flush=True)
print(df.pivot_table(index="state", columns=["rho", "arm"], values="order_gain").round(4).to_string(), flush=True)
print("DONE", flush=True)
