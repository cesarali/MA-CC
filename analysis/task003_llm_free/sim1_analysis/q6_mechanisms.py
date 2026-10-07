"""Q6 — mechanisms (ANALYSIS_PLAN.md §4).

(a) Why uniform posting leans to A0 at full memory (task003-nosolution, silent, rho = 1).
    All 15 agent facts together give exactly (0.5, 0, 0.5), so a lean must come from
    partial memories. Decompose day-30 votes by memory size, and for memories missing
    exactly one agent fact, by which fact is missing and the lean of "all but that fact".
(b) Which facts spread, survive or die out: share of agents holding each agent fact in
    memory on day 30 (silent cells), against the allocation the fact favours.
(c) Why extra truth posts lower own-evidence proofs (Q5): rule off vs rule on, same
    episodes. Reading: share of posts read that are controller posts. Posting: share of
    agent posts that are agents' original facts vs controller-pool facts; posting reason.

Outputs: tables/q6_*.csv, figures/q6_*.png.
From the repository root:
    .venv/bin/python analysis/task003_llm_free/sim1_analysis/q6_mechanisms.py
"""
from __future__ import annotations
import pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (SETUPS_DIR, VARIANT_LABEL, VARIANTS, WORLD_FILE, World, cells_table, load_setup,  # noqa: E402
                    out_dir, write_index)

POP = 24


def favours(world, f):
    P = np.array(world.posterior(1 << f))
    return "/".join(f"A{k}" for k in range(3) if np.isclose(P[k], P.max()))


def part_a(world, cells, d):
    setup = load_setup("task003_nosolution", SETUPS_DIR, world)
    own = sorted(set(setup.agent_facts))
    full = sum(1 << f for f in own)
    holders = {f: setup.agent_facts.count(f) for f in own}
    rows_size, rows_missing = [], []
    for v in VARIANTS:
        c = cells[(cells.purpose == "main") & (cells.variant == v) & (cells.setup == "task003_nosolution")
                  & (cells.arm == "silent") & (cells.rho == 1.0)]
        for r in c.itertuples():
            a = pd.read_parquet(pathlib.Path(r.path) / "agents.parquet", columns=["day", "active_facts", "vote", "p_A0", "p_A2"])
            last = a[a.day == r.M - 1].copy()
            mem = last.active_facts.to_numpy()
            last["size"] = [int(m).bit_count() for m in mem]
            for k, g in last.groupby("size"):
                rows_size.append({"variant": v, "q": r.q, "facts_held": k, "share_of_agents": len(g) / len(last),
                                  "vote_margin_A2_minus_A0": (g.vote == 2).mean() - (g.vote == 0).mean(),
                                  "belief_margin_A2_minus_A0": (g.p_A2 - g.p_A0).mean()})
            nearly = last[last["size"] == len(own) - 1]
            for f in own:
                n = int(((nearly.active_facts.to_numpy() & (1 << f)) == 0).sum())
                P = world.posterior(full & ~(1 << f))
                rows_missing.append({"variant": v, "q": r.q, "missing_fact": world.fact_ids[f], "fact_favours": favours(world, f),
                                     "held_by_agents_at_start": holders[f],
                                     "share_of_all_agents": n / len(last),
                                     "lean_of_all_but_this_fact_A2_minus_A0": P[2] - P[0]})
    S, M = pd.DataFrame(rows_size), pd.DataFrame(rows_missing)
    S.to_csv(d / "tables" / "q6a_by_memory_size.csv", index=False, float_format="%.4f")
    M.to_csv(d / "tables" / "q6a_missing_one_fact.csv", index=False, float_format="%.4f")
    # contribution of each missing fact to the population's lean
    M["contribution"] = M.share_of_all_agents * M.lean_of_all_but_this_fact_A2_minus_A0
    print("(a) nosolution, silent, rho = 1: agents by number of facts held on day 30 (mean over q)")
    print(S.groupby(["variant", "facts_held"])[["share_of_agents", "vote_margin_A2_minus_A0"]].mean()
          .unstack("variant").round(3).to_string())
    print("\n    agents missing exactly one of the 15 agent facts: which fact, and that memory's lean (mean over q)")
    t = M.groupby(["variant", "missing_fact", "fact_favours", "held_by_agents_at_start",
                   "lean_of_all_but_this_fact_A2_minus_A0"]).share_of_all_agents.mean().unstack("variant")
    print(t.round(3).to_string())
    print("\n    sum over missing facts of (share x lean): predicted belief margin from one-fact-missing memories")
    print(M.groupby(["variant", "q"]).contribution.sum().unstack("q").round(3).to_string())
    return S, M


def part_b(world, cells, d):
    rows = []
    for setup_name in ("task003_nosolution", "task003_symmetric"):
        setup = load_setup(setup_name, SETUPS_DIR, world)
        own = sorted(set(setup.agent_facts))
        c = cells[(cells.purpose == "main") & (cells.setup == setup_name) & (cells.arm == "silent")]
        for r in c.itertuples():
            a = pd.read_parquet(pathlib.Path(r.path) / "agents.parquet", columns=["day", "active_facts"])
            mem = a[a.day == r.M - 1].active_facts.to_numpy()
            for f in own:
                rows.append({"setup": setup_name, "variant": r.variant, "rho": r.rho, "q": r.q,
                             "fact": world.fact_ids[f], "favours": favours(world, f),
                             "held_at_start": setup.agent_facts.count(f),
                             "share_holding_day30": float(((mem >> f) & 1).mean())})
    B = pd.DataFrame(rows)
    B.to_csv(d / "tables" / "q6b_fact_prevalence.csv", index=False, float_format="%.4f")
    print("\n(b) share of agents holding each agent fact on day 30, by the allocation the fact favours (silent, mean over q)")
    print(B.groupby(["setup", "rho", "favours", "variant"]).share_holding_day30.mean().unstack("variant")[list(VARIANTS)]
          .round(3).to_string())
    # figure: prevalence per fact, nosolution, rho = 0.75
    sub = B[(B.setup == "task003_nosolution") & (B.rho == 0.75)]
    order = sub.groupby(["favours", "fact"]).share_holding_day30.mean().sort_index().index
    fig, ax = plt.subplots(figsize=(13, 5))
    w = 0.2
    for i, v in enumerate(VARIANTS):
        m = sub[sub.variant == v].groupby(["favours", "fact"]).share_holding_day30.mean().reindex(order)
        ax.bar(np.arange(len(order)) + (i - 1.5) * w, m.to_numpy(), width=w, label=VARIANT_LABEL[v])
    ax.set_xticks(range(len(order)), [f"{f}\n({fav})" for fav, f in order], rotation=70, fontsize=7)
    ax.set_ylabel("share of agents holding it, day 30")
    ax.legend(fontsize=8)
    ax.set_title("Q6b · task003-nosolution · silent · ρ = 0.75: which agent facts survive (mean over q)")
    fig.tight_layout()
    fig.savefig(d / "figures" / "q6b_fact_prevalence_nosolution.png", dpi=115)
    plt.close(fig)
    return B


def part_c(world, cells, d):
    setup = load_setup("task003_symmetric", SETUPS_DIR, world)
    own = 0
    for f in setup.agent_facts:
        own |= 1 << f
    pool = 0
    for f in setup.pools[0]:
        pool |= 1 << f
    rows = []
    key = ["variant", "q", "qc", "b", "rho"]
    off = cells[cells.purpose == "no_proof_stop"].set_index(key).path
    on = cells[(cells.purpose == "main") & (cells.setup == "task003_symmetric") & (cells.arm == "truth")].set_index(key).path
    for v in ("sim1_base", "sim1_pm"):
        for rho in (0.75, 1.0):
            for b in (1, 3):
                k = (v, 6, 12, b, rho)
                for rule, path in (("on", on[k]), ("off", off[k])):
                    a = pd.read_parquet(pathlib.Path(path) / "agents.parquet",
                                        columns=["posts_read", "posted_fact", "post_reason", "active_facts"])
                    codes = np.concatenate(a.posts_read.to_numpy())
                    posted = a.posted_fact.to_numpy()
                    has = posted >= 0
                    pf = posted[has].astype(np.int64)
                    mem = a.active_facts.to_numpy()
                    rows.append({"variant": v, "rho": rho, "b": b, "rule": rule,
                                 "reads_from_controller": float((codes // 64 == 24).mean()),
                                 "agent_posts_own_evidence": float((((own >> pf) & 1) == 1).mean()),
                                 "agent_posts_pool_facts": float((((pool >> pf) & 1) == 1).mean()),
                                 "posts_by_fallback": float((a.post_reason[has] == 1).mean()),
                                 "own_evidence_facts_held": float(np.mean([int(m & own).bit_count() for m in mem]))})
    C = pd.DataFrame(rows)
    C.to_csv(d / "tables" / "q6c_crowding.csv", index=False, float_format="%.5f")
    print("\n(c) symmetric truth controller, q = 6, qc = 12: rule on vs off (all days)")
    print(C.set_index(["variant", "rho", "b", "rule"]).round(4).to_string())
    return C


def main():
    d = out_dir()
    world = World(WORLD_FILE)
    cells = cells_table()
    part_a(world, cells, d)
    part_b(world, cells, d)
    part_c(world, cells, d)
    write_index(d, [
        {"question": "Q6", "file": "tables/q6a_by_memory_size.csv", "what": "votes by memory size, nosolution silent rho 1", "cells": "nosolution silent rho=1"},
        {"question": "Q6", "file": "tables/q6a_missing_one_fact.csv", "what": "which fact is missing from nearly-full memories, and its lean", "cells": "nosolution silent rho=1"},
        {"question": "Q6", "file": "tables/q6b_fact_prevalence.csv", "what": "share of agents holding each agent fact on day 30", "cells": "silent"},
        {"question": "Q6", "file": "figures/q6b_fact_prevalence_nosolution.png", "what": "fact survival, nosolution rho 0.75", "cells": "nosolution silent"},
        {"question": "Q6", "file": "tables/q6c_crowding.csv", "what": "reading vs posting crowd-out, rule on vs off", "cells": "symmetric truth q6 qc12"},
    ])


if __name__ == "__main__":
    main()
