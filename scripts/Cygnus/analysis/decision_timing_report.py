"""Where does a run's wall time go, per decision?

Reads every ``decision_timing.jsonl`` under a study or run root (one row per
logical decision, written by the relational runtime) and prints the split
between provider latency, coordinator admission wait, local queue wait and the
remainder (prompt compile before the timer is excluded; validation, recorder
writes and event-loop scheduling are included), by decision stage and by round.

    python scripts/Cygnus/analysis/decision_timing_report.py <root> [--csv out.csv]

The gap this was built to explain: 25.5 s per call client-side against 8.6 s
of model time on the b9/b15 run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

COLUMNS = ["wall_seconds", "provider_latency_seconds", "queue_seconds", "admission_seconds"]


def load(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("decision_timing.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["source"] = str(path.parent)
                    rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in COLUMNS:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce").fillna(0.0)
    frame["client_overhead_seconds"] = (
        frame["wall_seconds"] - frame["provider_latency_seconds"]
    ).clip(lower=0.0)
    frame["backpressure_seconds"] = frame["queue_seconds"] + frame["admission_seconds"]
    frame["model_seconds"] = (frame["provider_latency_seconds"] - frame["backpressure_seconds"]).clip(lower=0.0)
    return frame


def summarize(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    metrics = ["wall_seconds", "model_seconds", "backpressure_seconds", "client_overhead_seconds",
               "provider_retries", "validation_attempts"]
    present = [m for m in metrics if m in frame.columns]
    grouped = frame.groupby(by, dropna=False)[present].agg(["mean", "median"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    grouped["decisions"] = frame.groupby(by, dropna=False).size()
    return grouped.reset_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)
    frame = load(args.root)
    if frame.empty:
        print("no decision_timing.jsonl under", args.root, file=sys.stderr)
        return 1
    total = frame[["wall_seconds", "model_seconds", "backpressure_seconds", "client_overhead_seconds"]].sum()
    print(f"decisions: {len(frame)}  outcomes: {frame['outcome'].value_counts().to_dict()}")
    print("share of total wall: " + ", ".join(f"{k.replace('_seconds', '')}={v / max(total['wall_seconds'], 1e-9):.1%}"
                                              for k, v in total.items() if k != "wall_seconds"))
    pd.set_option("display.width", 200)
    print("\n== by decision stage")
    print(summarize(frame, ["decision_stage"]).round(3).to_string(index=False))
    print("\n== by round (first 12)")
    print(summarize(frame, ["round_index"]).round(3).head(12).to_string(index=False))
    if args.csv:
        frame.to_csv(args.csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
