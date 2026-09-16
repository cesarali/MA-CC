#!/usr/bin/env python3
"""Create descriptive no-control bar inputs from complete retained episodes."""
import argparse
import hashlib
import json
import pandas as pd
from pathlib import Path

METRICS = [('truth_before', 'truth_vote_share_before', 'round'),
           ('truth_after', 'truth_vote_share', 'round'),
           ('final_truth', 'truth_vote_share', 'episode'),
           ('collective', 'collective_solvable', 'round'),
           ('individual', 'symbolic_individual_solvability_share', 'round'),
           ('facts', 'active_union_fact_fraction', 'round')]


def summarize(source, output):
    validation = json.loads((source / 'validation.json').read_text())
    if not validation.get('valid') or not validation.get('complete'):
        raise ValueError('This summary requires a validated complete package')
    path = source / 'tables/epistemic_round_timeseries.parquet'
    frame = pd.read_parquet(path)
    if frame.duplicated(['cell_id', 'episode_id', 'round_index']).any():
        raise ValueError('Duplicate round identities')
    output.mkdir(parents=True, exist_ok=True)
    for key, field, unit in METRICS:
        rows = frame.sort_values('round_index').groupby(['cell_id', 'episode_id']).tail(1) if unit == 'episode' else frame
        values = pd.to_numeric(rows[field], errors='coerce')
        if values.isna().any() or not values.between(0, 1).all():
            raise ValueError(f'Invalid fraction: {field}')
        result = rows.groupby('epistemic_persistence', as_index=False).agg(mean_target_share=(field, 'mean'), n_observations=(field, 'size'))
        result.to_parquet(output / f'{key}.parquet', index=False)
    (output / 'provenance.json').write_text(json.dumps(dict(
        source=str(source.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metrics=METRICS, weighting='Equal weight per round, except final_truth: equal weight per episode',
        counts=validation['counts']), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.source, args.output)
