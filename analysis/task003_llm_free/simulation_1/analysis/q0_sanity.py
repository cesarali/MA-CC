"""Q0 — sanity checks at scale (ANALYSIS_PLAN.md §4).

1. Invariants on every cell of every run: posts come from memory; memory bookkeeping
   adds up; proof from own evidence never exceeds proof from all facts; the
   controller posts 0 or b facts, never after the last day, never when gated or
   (with the rule on) when the target is proved; abstention only where the variant
   allows it; vote shares sum to 1.
2. Starting votes: on day 1 the first agent in the order reads nothing, so its vote
   comes from its own fact alone. Compare those votes with the prediction from the
   voting rule (argmax with random ties, or probability matching).
3. The base run's silent cells against the 6 October reference values (plan §4).

From the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_1/analysis/q0_sanity.py
"""
from __future__ import annotations
import multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (HERE, SETUPS_DIR, VARIANTS, WORLD_FILE, World, cells_table, load_setup,  # noqa: E402
                    out_dir, write_index)

ABSTAINED, NO_MEMORY, RANDOM, UNIFORM = 4, 5, 2, 3
AGENT_COLS = ["episode", "day", "agent", "memory_after_dawn", "known_before_reading", "active_facts",
              "facts_read", "n_posts_read", "vote", "posted_fact", "post_reason", "proves_A0",
              "proves_A0_own_evidence"]


def check_cell(row: dict) -> dict:
    path = pathlib.Path(row["path"])
    a = pd.read_parquet(path / "agents.parquet", columns=AGENT_COLS)
    d = pd.read_parquet(path / "days.parquet")
    out = {"run": row["run"], "cell": row["cell"], "agent_rows": len(a)}
    posted = a.posted_fact.to_numpy().astype(np.int64)
    active = a.active_facts.to_numpy()
    has_post = posted >= 0
    out["post_not_in_memory"] = int((((active[has_post] >> posted[has_post]) & 1) == 0).sum())
    out["no_post_without_reason"] = int((~has_post & ~a.post_reason.isin([ABSTAINED, NO_MEMORY])).sum())
    out["dawn_not_subset_of_known"] = int(((a.memory_after_dawn & ~a.known_before_reading) != 0).sum())
    out["active_not_dawn_plus_read"] = int((a.active_facts != (a.memory_after_dawn | a.facts_read)).sum())
    out["reads_over_q"] = int((a.n_posts_read > row["q"]).sum())
    out["own_proof_without_proof"] = int((a.proves_A0_own_evidence & ~a.proves_A0).sum())
    reasons = a.post_reason.value_counts().to_dict()
    for code, name in ((0, "in_context"), (1, "fallback"), (2, "random"), (3, "uniform"), (4, "abstained"),
                       (5, "no_memory")):
        out[f"n_{name}"] = int(reasons.get(code, 0))
    # reasons the variant should never produce
    if row["agent_post_rule"] == "uniform_active":
        bad = ~a.post_reason.isin([UNIFORM, NO_MEMORY])
    elif row["agent_post_always"]:
        bad = a.post_reason.isin([ABSTAINED, UNIFORM])
    else:
        bad = a.post_reason.isin([RANDOM, UNIFORM])
    out["reason_not_allowed_by_variant"] = int(bad.sum())
    shares = d[["share_A0", "share_A1", "share_A2"]].sum(axis=1)
    out["shares_not_summing_to_1"] = int((np.abs(shares - 1) > 1e-9).sum())
    out["days_per_episode_wrong"] = int((d.groupby("episode").size() != row["M"]).sum())
    nf = path / "nights.parquet"
    if nf.exists():
        n = pd.read_parquet(nf, columns=["day", "n_posts_read", "decision", "n_posted", "p_target_given_read"])
        out["nights"] = len(n)
        out["night_after_last_day"] = int((n.day >= row["M"] - 1).sum())
        out["dose_not_0_or_b"] = int((~n.n_posted.isin([0, row["b"]])).sum())
        out["controller_reads_over_qc"] = int((n.n_posts_read > row["qc"]).sum())
        out["posted_while_silent_decision"] = int((n.decision.isin(["proved", "gate"]) & (n.n_posted > 0)).sum())
        out["proved_decision_without_proof"] = int(((n.decision == "proved") & (n.p_target_given_read < 1)).sum())
        rule_on = bool(row["silent_when_target_proved"])
        out["decision_inconsistent_with_rule"] = int(
            (n.decision == ("posted_proved" if rule_on else "proved")).sum())
        for dec in ("proved", "gate", "posted", "posted_proved", "fallback"):
            out[f"nights_{dec}"] = int((n.decision == dec).sum())
    elif row["arm"] != "silent":
        out["missing_nights_file"] = 1
    return out


def first_votes(cells: pd.DataFrame, world: World) -> pd.DataFrame:
    """Day-1 votes of the first agent in the order, against the voting rule's prediction.
    Day 1 is identical across arms and cells of a run (same seed, no controller yet,
    nobody has read anything), so one silent cell per variant and setup suffices."""
    rows = []
    for (variant, setup), g in cells[(cells.arm == "silent") & (cells.purpose == "main")].groupby(
            ["variant", "setup"]):
        cell = g.iloc[0]
        s = load_setup(setup, SETUPS_DIR, world)
        a = pd.read_parquet(pathlib.Path(cell.path) / "agents.parquet",
                            columns=["day", "position", "agent", "vote", "n_posts_read"])
        first = a[(a.day == 0) & (a.position == 0)]
        assert (first.n_posts_read == 0).all()
        expected = np.zeros(3)
        for ag in first.agent:
            P = np.array(world.posterior(1 << s.agent_facts[ag]))
            if cell.agent_sampling_mode == "argmax":
                win = np.isclose(P, P.max())
                expected += win / win.sum()
            else:
                expected += P
        observed = np.bincount(first.vote, minlength=3)
        n = len(first)
        sd = np.sqrt(expected * (1 - expected / n))
        # expected votes of the whole population before any reading, per 24 agents
        pop = np.zeros(3)
        for f in s.agent_facts:
            P = np.array(world.posterior(1 << f))
            if cell.agent_sampling_mode == "argmax":
                win = np.isclose(P, P.max())
                pop += win / win.sum()
            else:
                pop += P
        rows.append({"variant": variant, "setup": setup, "episodes": n,
                     **{f"observed_A{k}": int(observed[k]) for k in range(3)},
                     **{f"expected_A{k}": round(expected[k], 1) for k in range(3)},
                     **{f"z_A{k}": round((observed[k] - expected[k]) / sd[k], 2) for k in range(3)},
                     **{f"population_expected_votes_A{k}": round(pop[k], 2) for k in range(3)}})
    return pd.DataFrame(rows)


REFERENCE = {  # 6 October silent results, final-day mean vote shares (ANALYSIS_PLAN.md §4, Q0)
    ("task003_nosolution", 0.75): ([.229, .354, .433], [.763, .645, .563]),
    ("task003_nosolution", 1.0): ([.388, .422, .470], [.612, .578, .528]),
    ("task003_symmetric", 0.75): ([.958, .967, .940], [.039, .032, .060]),
    ("task003_symmetric", 1.0): ([1.0, .970, .930], [.0, .030, .070]),
}


def reference_check(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = cells[(cells.variant == "sim1_base") & (cells.arm == "silent") & (cells.purpose == "main")]
    for (setup, rho), (a0, a2) in REFERENCE.items():
        for i, q in enumerate((3, 6, 12)):
            c = base[(base.setup == setup) & (base.rho == rho) & (base.q == q)].iloc[0]
            d = pd.read_parquet(pathlib.Path(c.path) / "days.parquet")
            last = d[d.day == c.M - 1]
            rows.append({"setup": setup, "rho": rho, "q": q,
                         "share_A0": last.share_A0.mean(), "reference_A0": a0[i],
                         "share_A2": last.share_A2.mean(), "reference_A2": a2[i],
                         "matches_to_3_decimals": bool(round(last.share_A0.mean(), 3) == a0[i]
                                                       and round(last.share_A2.mean(), 3) == a2[i])})
    return pd.DataFrame(rows)


def main():
    d = out_dir()
    cells = cells_table()
    cells.to_csv(d / "tables" / "cells.csv", index=False)
    print(f"{len(cells)} cells in {cells.run.nunique()} runs")
    with multiprocessing.Pool(12) as pool:
        res = pd.DataFrame(pool.map(check_cell, cells.to_dict("records")))
    res.to_csv(d / "tables" / "q0_invariants.csv", index=False)
    violation_cols = ["post_not_in_memory", "no_post_without_reason", "dawn_not_subset_of_known",
                      "active_not_dawn_plus_read", "reads_over_q", "own_proof_without_proof",
                      "reason_not_allowed_by_variant", "shares_not_summing_to_1", "days_per_episode_wrong",
                      "night_after_last_day", "dose_not_0_or_b", "controller_reads_over_qc",
                      "posted_while_silent_decision", "proved_decision_without_proof",
                      "decision_inconsistent_with_rule", "missing_nights_file"]
    totals = res.reindex(columns=violation_cols).fillna(0).sum().astype(int)
    print("\nINVARIANT VIOLATIONS (summed over all cells; all should be 0):")
    print(totals.to_string())
    print(f"\nagent-day rows checked: {res.agent_rows.sum():,}; nights checked: {int(res.nights.fillna(0).sum()):,}")

    world = World(WORLD_FILE)
    fv = first_votes(cells, world)
    fv.to_csv(d / "tables" / "q0_first_votes.csv", index=False)
    print("\nFIRST AGENT ON DAY 1 (reads nothing): observed vs predicted votes, z = (obs - exp)/sd")
    print(fv.to_string(index=False))

    ref = reference_check(cells)
    ref.to_csv(d / "tables" / "q0_reference.csv", index=False)
    print(f"\nBASE SILENT vs 6 OCTOBER: {ref.matches_to_3_decimals.sum()} of {len(ref)} match to 3 decimals")
    print(ref.round(4).to_string(index=False))

    # abstention overview per variant (silent cells), for Q4 and the reader
    ab = res.merge(cells[["run", "cell", "variant", "arm", "setup"]], on=["run", "cell"])
    ab["abstention_rate"] = ab.n_abstained / ab.agent_rows
    ab["fallback_rate"] = ab.n_fallback / ab.agent_rows
    summ = ab.groupby(["variant", "setup", "arm"])[["abstention_rate", "fallback_rate"]].mean().round(4)
    summ.to_csv(d / "tables" / "q0_posting_reasons.csv")
    print("\nPOSTING REASONS (mean over cells)")
    print(summ.to_string())
    write_index(d, [
        {"question": "Q0", "file": "tables/cells.csv", "what": "every cell of every run with its factors", "cells": "all"},
        {"question": "Q0", "file": "tables/q0_invariants.csv", "what": "invariant violation counts per cell", "cells": "all"},
        {"question": "Q0", "file": "tables/q0_first_votes.csv", "what": "day-1 first-agent votes vs voting-rule prediction", "cells": "one silent cell per variant x setup"},
        {"question": "Q0", "file": "tables/q0_reference.csv", "what": "base silent final shares vs 6 October", "cells": "sim1_base silent"},
        {"question": "Q0", "file": "tables/q0_posting_reasons.csv", "what": "abstention and fallback rates", "cells": "all"},
    ])


if __name__ == "__main__":
    main()
