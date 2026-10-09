"""Summarise a Simulation 2 run: one row per cell, with paired gains over its silent cell.

A controlled cell is paired with the silent cell of the same step and setup (or the earlier
silent cell it reuses, listed in manifest.json). Gains are means over episodes of
(controlled - silent) for the same episode number. Uncertainty is left to the analysis.

The runner calls this at the end of every run. To redo it by hand, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/simulator/summarize_simulation_2.py <run folder>
"""
from __future__ import annotations
import json, pathlib, sys

import pandas as pd

TARGET = {"truth": 0, "false": 2}
ABSTAINED = 4
SNAP = ["episode", "t", "mean_p_A0", "mean_p_A2", "share_A0", "share_A2", "share_voted", "proof_rate_A0"]


def _at(snap: pd.DataFrame, t: float) -> pd.DataFrame:
    return snap[(snap.t - t).abs() < 1e-9].set_index("episode")


def summarize(run: str | pathlib.Path) -> pd.DataFrame:
    run = pathlib.Path(run)
    manifest_file = run / "manifest.json"
    reused = json.loads(manifest_file.read_text()).get("reused_silent_cells", {}) if manifest_file.exists() else {}
    cells = {p.name: p for p in sorted((run / "cells").iterdir()) if p.is_dir()}
    snaps, rows = {}, []
    for name, folder in cells.items():
        snaps[name] = pd.read_parquet(folder / "snapshots.parquet", columns=SNAP)
    for name, folder in cells.items():
        prm = json.loads((folder / "params.json").read_text())
        snap = snaps[name]
        ep = pd.read_parquet(folder / "episodes.parquet")
        act = pd.read_parquet(folder / "actions.parquet", columns=["post_reason"])
        row = {k: prm[k] for k in ("step", "setup", "arm", "board", "forgetting", "agent_schedule",
                                   "controller_schedule", "lambda_c", "controller_gate",
                                   "silent_when_target_proved")}
        row["episodes"] = len(ep)
        for t in (30.0, 40.0):
            s = _at(snap, t)
            for col in ("mean_p_A0", "mean_p_A2", "share_A0", "share_A2", "share_voted", "proof_rate_A0"):
                row[f"{col}_t{int(t)}"] = s[col].mean()
        row["abstention_rate"] = (act.post_reason == ABSTAINED).mean()
        row["controller_messages"] = ep.controller_messages.mean()
        row["controller_reads"] = ep.controller_reads.mean()
        row["share_budget_exhausted"] = ep.budget_exhausted_at.notna().mean()
        row["agent_actions"] = ep.agent_actions.mean()
        if prm["arm"] in TARGET:
            k = TARGET[prm["arm"]]
            silent = f"{prm['step']}__{prm['setup']}__silent"
            silent = reused.get(silent, silent)
            row["silent_cell"] = silent
            if silent in snaps:
                for t in (30.0, 40.0):
                    c, s = _at(snap, t), _at(snaps[silent], t)
                    both = c.index.intersection(s.index)
                    row[f"gain_m_target_t{int(t)}"] = (c.loc[both, f"mean_p_A{k}"] - s.loc[both, f"mean_p_A{k}"]).mean()
                    row[f"gain_share_target_t{int(t)}"] = (c.loc[both, f"share_A{k}"] - s.loc[both, f"share_A{k}"]).mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(run / "summary.csv", index=False)
    return out


if __name__ == "__main__":
    print(summarize(sys.argv[1]).to_string(index=False))
