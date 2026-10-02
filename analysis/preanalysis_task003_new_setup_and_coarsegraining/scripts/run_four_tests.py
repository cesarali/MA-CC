import sys, json
sys.path.insert(0, "/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/scripts")
import numpy as np, pandas as pd
from four_tests import test_order, test_ck, test_ci, test_lumpability, EDGES, NS, SC

d = pd.read_parquet(SC + "results/data/states_full.parquet")
d["S"] = np.digitize(d.e, EDGES)
rows, lump = [], []
for rho in (0.75, 1.00):
    for arm in ("silent", "A2", "A0"):
        sub = d[(d.epistemic_persistence == rho) & (d.arm == arm)]
        if len(sub) < 300:
            continue
        gains = [g for g in (test_order(sub, seed=k) for k in range(12)) if g is not None]
        ck = test_ck(sub)
        ci = test_ci(sub, nperm=200)
        rows.append(dict(rho=rho, arm=arm, n=len(sub),
                         order_gain=float(np.mean(gains)), order_sd=float(np.std(gains)),
                         ck_n2=ck.get(2), ck_n3=ck.get(3), ck_n5=ck.get(5),
                         cmi_bits=ci["cmi_bits"], cmi_null=ci["null_mean"],
                         cmi_excess=ci["excess"], cmi_p=ci["p_value"]))
        print(rows[-1], flush=True)
        for feat in ("spread", "dec_frac", "pool_frac", "mean_nfacts"):
            r = test_lumpability(sub, feat)
            lump.append(dict(rho=rho, arm=arm, feature=feat, weighted_tv=r["weighted_tv"]))
pd.DataFrame(rows).to_csv(SC + "30_coarse_graining/four_tests.csv", index=False)
pd.DataFrame(lump).to_csv(SC + "30_coarse_graining/lumpability.csv", index=False)
print("\n=== LUMPABILITY (TV gap between halves within a macrostate) ===", flush=True)
print(pd.DataFrame(lump).pivot_table(index=["rho", "arm"], columns="feature",
                                     values="weighted_tv").round(3).to_string(), flush=True)
print("DONE", flush=True)
