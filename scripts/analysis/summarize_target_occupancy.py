#!/usr/bin/env python3
"""Summarize pre-round, post-round, or final-episode target shares."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def summarize(frame, mode="before"):
    if mode not in {"before", "after", "final"}:
        raise ValueError("Unknown summary mode")
    frame = frame.loc[frame.episode_complete].copy()
    if frame.duplicated(['cell_id', 'episode_id', 'round_index']).any():
        raise ValueError('Duplicate round identities')
    field = "x_t" if mode == "before" else "x_after"
    if mode == "final":
        frame = frame.sort_values("round_index").groupby(["cell_id", "episode_id"], as_index=False).tail(1)
    if frame[field].isna().any() or not frame[field].between(0, 1).all():
        raise ValueError('Invalid pre-intervention target shares')
    return frame.groupby('epistemic_persistence', as_index=False).agg(
        mean_target_share=(field, 'mean'), n_observations=(field, 'size'),
        n_cells=('cell_id', 'nunique'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument("--mode", choices=["before", "after", "final"], default="before")
    args = parser.parse_args()
    frame = pd.read_parquet(args.source)
    result = summarize(frame, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        source=str(args.source.resolve()), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        weighting=('equal weight per completed episode' if args.mode == 'final' else 'equal weight per retained round from completed episodes') + '; pooled across budgets',
        coordinate={'before': 'x_t: pre-intervention share', 'after': 'x_after: end-of-round share', 'final': 'x_after at last round of each completed episode'}[args.mode],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')
    print(result.to_string(index=False))
