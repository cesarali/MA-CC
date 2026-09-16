#!/usr/bin/env python3
"""Extend a copied analysis package with x-binned existing causal estimates."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mas_cc.analysis.causal_response import estimate_causal_response
from mas_cc.studies.table_io import write_scientific_table


def extend(source: Path, output: Path) -> None:
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError('Use a separate output directory outside the source package')
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    recipe = yaml.safe_load((source / 'analysis_recipe.yaml').read_text())
    bins = int(recipe.get('state_local_x_bins', 8))
    settings = recipe['resampling']
    inputs = pd.read_parquet(source / 'tables/causal_response_round_inputs.parquet')
    if not inputs.x_t.dropna().between(0, 1).all():
        raise ValueError('Target state outside [0, 1]')
    indices = np.minimum(np.floor(inputs.x_t * bins), bins - 1)
    tables = {}
    for lag in (1, 2, 3):
        estimates = []
        diagnostics = []
        for index in range(bins):
            selected = inputs[(indices == index) & inputs[f'lag_{lag}_available']].copy()
            effect, support, _ = estimate_causal_response(
                selected, lags=(lag,),
                bootstrap_resamples=int(settings['bootstrap_resamples']),
                confidence=float(settings['confidence']), seed=int(settings['seed']),
            )
            for frame in (effect, support):
                frame['target_fraction_bin_index'] = index
                frame['target_fraction_bin_lower'] = index / bins
                frame['target_fraction_bin_upper'] = (index + 1) / bins
                frame['target_fraction_bin_center'] = (index + 0.5) / bins
                frame['target_fraction_bin_count'] = bins
            estimates.append(effect)
            diagnostics.append(support)
        tables[f'causal_response_state_local_lag_{lag}'] = pd.concat(estimates, ignore_index=True)
        tables[f'causal_response_state_local_support_lag_{lag}'] = pd.concat(diagnostics, ignore_index=True)
        print(f'Lag {lag}: {len(tables[f"causal_response_state_local_lag_{lag}"])} cell/state estimates', flush=True)
    shutil.copytree(source, output)
    manifest = json.loads((output / 'analysis_manifest.json').read_text())
    provenance = dict(
        operation='x_binned_existing_propensity_weighted_causal_response',
        source_analysis_hash=manifest.get('analysis_hash'),
        source_inputs_sha256=hashlib.sha256((source / 'tables/causal_response_round_inputs.parquet').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bins=bins, lags=[1, 2, 3], resampling=settings,
        selection='Completed episodes, initial x bin, and available lag outcome; bootstrap shared initialization blocks in each selected subset.',
        provider_calls=0,
    )
    for name, frame in tables.items():
        write_scientific_table(output / 'tables', name, frame)
    provenance['output_sha256'] = {name: hashlib.sha256((output / 'tables' / f'{name}.parquet').read_bytes()).hexdigest() for name in tables}
    (output / 'provenance/state_local_causal_response.json').write_text(json.dumps(provenance, indent=2) + '\n')
    manifest['state_local_causal_response_extension'] = provenance
    manifest['analysis_hash'] = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    if isinstance(manifest.get('tables'), list):
        manifest['tables'].extend(f'tables/{name}.parquet' for name in tables)
    (output / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extend(args.source, args.output)
