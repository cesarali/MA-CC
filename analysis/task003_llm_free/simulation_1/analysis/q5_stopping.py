"""Q5 — the stopping rule `silent_when_target_proved` (ANALYSIS_PLAN.md §4).

Only the truth controller in task003-symmetric can see its target proved on the board,
so the no-proof-stop runs contain exactly those cells (120 per variant). Each is paired
with the main-run cell of the same settings (rule on) and the main-run silent twin, and
compared episode by episode (same seed, so the same draws until the two rules first
differ). Per day, the rule-off minus rule-on difference in:
  share_A0                 the truth's vote share
  proof_rate_A0            share of agents whose memory proves A0 (all facts)
  proof_rate_A0_own        the same, using only the agents' original evidence
  dose                     facts posted so far
with 95% episode-bootstrap intervals; plus how often the rule mattered (nights with
decision `proved` under the rule, `posted_proved` without it).

Outputs: tables/q5_time.parquet, tables/q5_final.csv, figures/q5_*.png.
From the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_1/analysis/q5_stopping.py
"""
from __future__ import annotations
import hashlib, multiprocessing, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import VARIANT_LABEL, VARIANTS, cells_table, out_dir, write_index  # noqa: E402

N_BOOT = 1000
QUANTS = ("share_A0", "proof_rate_A0", "proof_rate_A0_own_evidence")
BS, QCS = (1, 2, 3, 6, 9), (3, 6, 12, 24)


def load(path):
    d = pd.read_parquet(pathlib.Path(path) / "days.parquet", columns=["episode", "day", *QUANTS]).sort_values(["episode", "day"])
    E, T = d.episode.nunique(), d.day.nunique()
    out = {k: d[k].to_numpy().reshape(E, T) for k in QUANTS}
    n = pd.read_parquet(pathlib.Path(path) / "nights.parquet", columns=["episode", "day", "n_posted", "decision"])
    per = np.zeros((E, T))
    per[n.episode.to_numpy(), n.day.to_numpy()] = n.n_posted.to_numpy()
    out["dose"] = per.cumsum(axis=1)
    out["decisions"] = n.decision.value_counts(normalize=True).to_dict()
    return out


def pair(job):
    key, off_path, on_path = job
    variant, q, qc, b, rho = key
    off, on = load(off_path), load(on_path)
    E, T = off["share_A0"].shape
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:8], "big"))
    idx = rng.integers(0, E, size=(N_BOOT, E))
    rec = {"variant": variant, "q": q, "qc": qc, "b": b, "rho": rho}
    time_rows, final = [], dict(rec)
    for k in (*QUANTS, "dose"):
        diff = off[k] - on[k]
        boots = diff[idx].mean(axis=1)
        m, lo, hi = diff.mean(axis=0), np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
        time_rows += [{**rec, "quantity": k, "day": t + 1, "mean": m[t], "lo": lo[t], "hi": hi[t]} for t in range(T)]
        last = T - 1 if k != "dose" else T - 2           # dose: nights 1..29
        final.update({f"diff_{k}": m[last], f"diff_{k}_lo": lo[last], f"diff_{k}_hi": hi[last],
                      f"on_{k}": on[k][:, last].mean(), f"off_{k}": off[k][:, last].mean()})
    final["nights_proved_rule_on"] = on["decisions"].get("proved", 0.0)
    final["nights_posted_proved_rule_off"] = off["decisions"].get("posted_proved", 0.0)
    return time_rows, final


def main():
    d = out_dir()
    c = cells_table()
    off = c[(c.purpose == "no_proof_stop")].set_index(["variant", "q", "qc", "b", "rho"]).path
    on = c[(c.purpose == "main") & (c.setup == "task003_symmetric") & (c.arm == "truth")].set_index(
        ["variant", "q", "qc", "b", "rho"]).path
    jobs = [(k, off[k], on[k]) for k in off.index]
    with multiprocessing.Pool(12) as pool:
        res = pool.map(pair, jobs)
    T = pd.DataFrame([r for t, _ in res for r in t]).sort_values(["variant", "q", "qc", "b", "rho", "quantity", "day"])
    F = pd.DataFrame([f for _, f in res]).sort_values(["variant", "rho", "q", "qc", "b"])
    T.to_parquet(d / "tables" / "q5_time.parquet", index=False)
    F.to_csv(d / "tables" / "q5_final.csv", index=False, float_format="%.5f")
    entries = [{"question": "Q5", "file": "tables/q5_time.parquet", "what": "rule off minus rule on, per day", "cells": "symmetric truth"},
               {"question": "Q5", "file": "tables/q5_final.csv", "what": "rule off minus rule on, day 30, and how often the rule mattered", "cells": "symmetric truth"}]

    colors = dict(zip(BS, plt.cm.viridis(np.linspace(0, 0.9, 5))))
    for k, label in (("proof_rate_A0", "share of agents whose memory proves A0"),
                     ("proof_rate_A0_own_evidence", "share proving A0 from the agents' own evidence only"),
                     ("share_A0", "A0 vote share")):
        fig, axes = plt.subplots(len(VARIANTS), 6, figsize=(18, 11), sharex=True, sharey=True)
        for i, v in enumerate(VARIANTS):
            for j, (rho, q) in enumerate([(r, qq) for r in (0.75, 1.0) for qq in (3, 6, 12)]):
                ax = axes[i, j]
                for b in BS:
                    t = T[(T.variant == v) & (T.q == q) & (T.qc == 12) & (T.b == b) & (T.rho == rho) & (T.quantity == k)]
                    ax.plot(t.day, t["mean"], color=colors[b], lw=1.4, label=f"b = {b}")
                    ax.fill_between(t.day, t.lo, t.hi, color=colors[b], alpha=0.18, lw=0)
                ax.axhline(0, color="k", lw=0.7)
                if i == 0:
                    ax.set_title(f"ρ = {rho}, q = {q}", fontsize=9)
                if j == 0:
                    ax.set_ylabel(VARIANT_LABEL[v], fontsize=8)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Q5 · task003-symmetric · truth controller: rule OFF minus rule ON, {label}, over days (qc = 12)",
                     fontsize=11)
        fig.tight_layout()
        name = f"figures/q5_{k}_time.png"
        fig.savefig(d / name, dpi=110)
        plt.close(fig)
        entries.append({"question": "Q5", "file": name, "what": f"rule off minus on: {k} over days", "cells": "symmetric truth"})
    write_index(d, entries)

    pd.set_option("display.width", 230)
    cols = ["nights_proved_rule_on", "nights_posted_proved_rule_off", "on_dose", "off_dose", "diff_share_A0",
            "on_proof_rate_A0", "diff_proof_rate_A0", "on_proof_rate_A0_own_evidence", "diff_proof_rate_A0_own_evidence"]
    print("DAY 30, mean over q, qc, b")
    print(F.groupby(["variant", "rho"])[cols].mean().round(4).to_string())
    print("\nBY b (qc = 12), mean over q: difference in proof rate (all facts / own evidence)")
    print(F[F.qc == 12].pivot_table(index=["variant", "rho"], columns="b",
                                    values=["diff_proof_rate_A0", "diff_proof_rate_A0_own_evidence"]).round(3).to_string())
    sig = F[(F.diff_proof_rate_A0_lo > 0) | (F.diff_proof_rate_A0_hi < 0)]
    print(f"\ncells where the proof-rate difference is significant: {len(sig)} of {len(F)}; "
          f"of those positive: {int((sig.diff_proof_rate_A0 > 0).sum())}; "
          f"at least one agent (0.042): {int((F.diff_proof_rate_A0.abs() >= 1/24).sum())}")


if __name__ == "__main__":
    main()
