"""Agent-level (micro-update) analysis: does controller message dose move a vote?

Motivation.  The population-level effects in this archive are large and causally
established, but they do not say *how* the controller acts.  The micro-update
records give, for every one of the 497,520 agent updates, that agent's vote
before and after and the identifiers of the messages it actually read.  That
permits a within-population test: among agents facing the same board at the same
moment, does reading more controller messages make an agent likelier to adopt the
controller's target?

Two features of the design make the naive version of this test misleading.

1. ``message_lifetime_rounds`` is 1 and ``surviving_message_count`` is 0: the
   board is rebuilt every round, so dose measures the current round's output only.
2. Agents update sequentially (``micro_slot_index`` 0..23) and post as they go, so
   the controller -- which posts first -- is progressively diluted within the
   round.  Dose therefore falls with update order while the outcome falls too,
   and an unadjusted dose-response picks up that ordering rather than any effect
   of the messages.

So the estimator residualises both dose and outcome on (episode x round), which
holds the population and the board composition fixed, and on update order.  What
survives is variation in which messages an agent happened to sample, board
sampling being uniform.  Uncertainty is a cluster bootstrap over initializations,
the only independent unit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import ARM_OF_SEMANTICS, RECORDS

MICRO_COLUMNS = [
    "cell_id", "episode_id", "round_index", "micro_slot_index", "focal_agent_id",
    "focal_opinion_before", "focal_opinion_after", "sampled_message_ids",
    "sampled_controller_message_ids", "eligible_board_message_count",
    "eligible_controller_message_count", "eligible_peer_message_count",
]
TARGET_LABEL = {"correct": "ALLOCATION_0", "ALLOCATION_2": "ALLOCATION_2"}


def _n_ids(v: Any) -> int:
    return len(json.loads(v)) if isinstance(v, str) and v.startswith("[") else 0


def load_micro(records: Path = RECORDS) -> pd.DataFrame:
    """Controlled-arm agent updates where a controller message was on the board."""
    m = pd.read_parquet(records / "micro_slots.parquet", columns=MICRO_COLUMNS)
    cells = pd.read_parquet(records / "cells.parquet",
                            columns=["cell_id", "communication_profile", "epistemic_persistence",
                                     "intervention_budget", "controller_target_semantics"])
    link = pd.read_parquet(records / "rounds.parquet",
                           columns=["cell_id", "episode_id", "physical_initial_state_hash"]).drop_duplicates()
    d = m.merge(cells, on="cell_id").merge(link, on=["cell_id", "episode_id"], how="left")
    d = d[d.intervention_budget.notna()].copy()
    d["target"] = d.controller_target_semantics.map(TARGET_LABEL)
    d["dose"] = d.sampled_controller_message_ids.map(_n_ids)
    d["n_read"] = d.sampled_message_ids.map(_n_ids)
    d["share"] = d.dose / d.n_read.replace(0, np.nan)
    d["on_target_after"] = (d.focal_opinion_after == d.target).astype(float)
    # the agent's vote at the end of the following round, for the lagged test
    nxt = d[["cell_id", "episode_id", "focal_agent_id", "round_index", "on_target_after"]].copy()
    nxt["round_index"] -= 1
    nxt = nxt.rename(columns={"on_target_after": "on_target_next"})
    d = d.merge(nxt, on=["cell_id", "episode_id", "focal_agent_id", "round_index"], how="left")
    return d[d.eligible_controller_message_count.fillna(0) > 0].copy()


def dilution_table(d: pd.DataFrame) -> pd.DataFrame:
    """How the controller's share of the board decays within a round."""
    g = d.assign(slot_bin=d.micro_slot_index // 4 * 4).groupby("slot_bin")
    out = g.agg(eligible_board=("eligible_board_message_count", "mean"),
                eligible_controller=("eligible_controller_message_count", "mean"),
                messages_read=("n_read", "mean"), dose=("dose", "mean"),
                controller_share=("share", "mean"),
                on_target_after=("on_target_after", "mean"), n=("dose", "size"))
    return out.reset_index()


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    vx = x - x.mean()
    denom = float((vx * vx).sum())
    return float((vx * (y - y.mean())).sum() / denom) if denom > 0 else np.nan


def _residualise(df: pd.DataFrame, cols: list[str], by: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out.groupby(by)[c].transform("mean")
    return out


def dose_response(d: pd.DataFrame, outcome: str = "on_target_after",
                  n_boot: int = 400, seed: int = 20260923) -> dict[str, Any]:
    """Raw, population-adjusted and order-adjusted dose slopes with a cluster bootstrap.

    Restricted to agents not already voting the controller's target, so the
    outcome is adoption rather than retention.
    """
    a = d[(d.focal_opinion_before != d.target) & d[outcome].notna()].copy()
    a["cell"] = a.cell_id + "|" + a.episode_id + "|" + a.round_index.astype(str)
    a["y"] = a[outcome].astype(float)

    def variants(f: pd.DataFrame) -> dict[str, float]:
        raw = _slope(f.dose.to_numpy(), f.y.to_numpy())
        pop = _residualise(f, ["dose", "y"], ["cell"])
        s_pop = _slope(pop.dose.to_numpy(), pop.y.to_numpy())
        both = _residualise(pop, ["dose", "y"], ["micro_slot_index"])
        s_both = _slope(both.dose.to_numpy(), both.y.to_numpy())
        return {"raw": raw, "population_adjusted": s_pop, "order_adjusted": s_both,
                "resid_dose_sd": float(both.dose.std())}

    point = variants(a)
    inits = a.physical_initial_state_hash.dropna().unique()
    groups = {k: v for k, v in a.groupby("physical_initial_state_hash")}
    rng = np.random.default_rng(seed)
    draws: list[dict[str, float]] = []
    for _ in range(n_boot):
        s = pd.concat([groups[i] for i in rng.choice(inits, len(inits))], copy=False)
        draws.append(variants(s))
    out: dict[str, Any] = {"outcome": outcome, "n_updates": int(len(a)),
                           "n_initializations": int(len(inits)), "n_boot": n_boot}
    for k in ("raw", "population_adjusted", "order_adjusted"):
        v = np.array([x[k] for x in draws])
        out[k] = point[k]
        out[f"{k}_lo"], out[f"{k}_hi"] = np.percentile(v, [2.5, 97.5])
    out["resid_dose_sd"] = point["resid_dose_sd"]
    out["raw_dose_sd"] = float(a.dose.std())
    return out


def board_facts(records: Path = RECORDS) -> dict[str, Any]:
    """Message lifetime and board turnover, which set up the dilution problem."""
    cells = pd.read_parquet(records / "cells.parquet", columns=["message_lifetime_rounds"])
    r = pd.read_parquet(records / "rounds.parquet",
                        columns=["board_mean_size", "board_peak_size", "board_messages_created",
                                 "board_messages_expired", "surviving_message_count"])
    return {"message_lifetime_rounds": sorted(cells.message_lifetime_rounds.unique().tolist()),
            "surviving_message_count_max": float(r.surviving_message_count.max()),
            "board_mean_size": float(r.board_mean_size.mean()),
            "board_peak_size": float(r.board_peak_size.mean()),
            "messages_created_per_round": float(r.board_messages_created.mean())}


def build(out_dir: Path, n_boot: int = 400, seed: int = 20260923) -> None:
    tdir = out_dir / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    d = load_micro()
    dilution_table(d).to_csv(tdir / "dose_by_slot.csv", index=False)
    rows = [dose_response(d, outcome=o, n_boot=n_boot, seed=seed)
            for o in ("on_target_after", "on_target_next")]
    pd.DataFrame(rows).to_csv(tdir / "dose_response.csv", index=False)
    facts = board_facts()
    facts["n_agent_updates_with_controller_message"] = int(len(d))
    facts["exposure_rate"] = float((d.dose > 0).mean())
    (tdir / "board_facts.json").write_text(json.dumps(facts, indent=1))
    print("micro tables written to", tdir)
