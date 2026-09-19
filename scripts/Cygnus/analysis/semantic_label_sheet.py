"""Draw a labelling sheet for the semantic-attribution calibration.

Takes the messages System One judged (the ``semantic_attribution.parquet`` a
run produced, or the ``--messages`` table it was fed) and writes a CSV with one
row per sampled message and one blank column per question for a human to fill
in. Two labellers each get their own copy; ``semantic_agreement.py`` scores the
sheets against the model and against each other.

    python scripts/Cygnus/analysis/semantic_label_sheet.py --judged <semantic_attribution.parquet> \
        --sample 200 --seed 1 --output labels-A.csv

The sample is the module's own stratified-by-cell draw, so the same seed gives
the same messages to every labeller. Human columns and their vocabularies:

    human_cites_controller   yes | no
    human_stance             one of the message's possible answers, or none
    human_pressure           0 | 1 | 2 | 3   (index into PRESSURE_LEVELS: no push … strong)
    human_provides_evidence  yes | no
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mas_cc.analysis import semantic_attribution as sa  # noqa: E402

HUMAN_COLUMNS = ("human_cites_controller", "human_stance", "human_pressure", "human_provides_evidence")
KEY_COLUMNS = ("episode_id", "message_id")


def build_sheet(frame: pd.DataFrame, *, sample: int | None, seed: int) -> pd.DataFrame:
    messages = sa.sample_messages(sa.messages_from_table(frame), sample, seed)
    rows = []
    for message in messages:
        rows.append({
            "episode_id": message["episode_id"], "message_id": message["message_id"], "cell_id": message["cell_id"],
            "intervention_budget": message.get("intervention_budget"), "message_type": message["message_type"],
            "possible_answers": json.dumps(message["possible_answers"]), "controller_target": message["controller_target"],
            "controller_message_visible": message["controller_message_exposed"],
            "vote_before": message["vote_before"], "vote_after": message["vote_after"], "text": message["text"],
            **{column: "" for column in HUMAN_COLUMNS}, "labeller": "", "notes": "",
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judged", type=Path, required=True, help="semantic_attribution.parquet or the messages table")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    frame = pd.read_parquet(args.judged) if args.judged.suffix == ".parquet" else pd.read_csv(args.judged)
    sheet = build_sheet(frame, sample=args.sample, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(args.output, index=False, quoting=csv.QUOTE_ALL)
    print(json.dumps({"messages": int(len(sheet)), "cells": int(sheet["cell_id"].nunique()), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
