"""Compare epistemic bootstrap time/RSS against a saved pre-change module.

Example (run from the repository root in the project environment)::

    python scripts/benchmarks/benchmark_epistemic_bootstrap.py \
        --baseline /tmp/epistemic_phase_baseline.py --resamples 1000

Each implementation runs in a fresh process. Synthetic prepared causal inputs
isolate drift/modulation costs; this does not benchmark symbolic state building,
provider calls, or whole-study aggregation. Output equality is checked as well.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd


def prepared_frame(rows: int, cells: int, blocks: int) -> pd.DataFrame:
    rng = np.random.default_rng(832)
    x = rng.random(rows)
    phi = rng.random(rows)
    action = rng.integers(0, 2, rows)
    propensity = rng.uniform(0.15, 0.85, rows)
    return pd.DataFrame(
        {
            "cell_id": [f"cell-{i}" for i in rng.integers(cells, size=rows)],
            "episode_id": [
                f"episode-{i}" for i in rng.integers(blocks * cells, size=rows)
            ],
            "initialization_block_id": [
                f"block-{i}" for i in rng.integers(blocks, size=rows)
            ],
            "episode_complete": True,
            "lag_1_available": True,
            "x_t": x,
            "symbolic_individual_solvability_share": phi,
            "symbolic_individual_solvability_share_after": np.clip(
                phi + rng.normal(0, 0.1, rows), 0, 1
            ),
            "U_t": action,
            "e_t": propensity,
            "ipw_contrast_weight": action / propensity
            - (1 - action) / (1 - propensity),
            "delta_x_h1": rng.normal(0, 0.1, rows),
            "round_index": rng.integers(0, 30, rows),
        }
    )


def worker(args: argparse.Namespace) -> None:
    if args.worker == "baseline":
        spec = importlib.util.spec_from_file_location(
            "epistemic_baseline", args.baseline
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        import mas_cc.analysis.epistemic_phase as module
    frame = prepared_frame(args.rows, args.cells, args.blocks)
    module.build_causal_response_inputs = lambda *a, **kw: frame.copy()
    kwargs = dict(
        x_bins=3,
        phi_bands=2,
        bootstrap_resamples=args.resamples,
        confidence=0.95,
        seed=42,
    )
    start = perf_counter()
    drift = module.estimate_joint_drift(frame, pd.DataFrame(), **kwargs)
    drift_seconds = perf_counter() - start
    start = perf_counter()
    kwargs["seed"] += 1
    surface, regression = module.estimate_epistemic_modulation(
        frame, pd.DataFrame(), **kwargs
    )
    modulation_seconds = perf_counter() - start
    for name, table in [
        ("drift", drift),
        ("surface", surface),
        ("regression", regression),
    ]:
        table.to_parquet(Path(args.output) / f"{args.worker}-{name}.parquet")
    print(
        json.dumps(
            dict(
                implementation=args.worker,
                drift_seconds=drift_seconds,
                modulation_seconds=modulation_seconds,
                total_seconds=drift_seconds + modulation_seconds,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--cells", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=30)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--worker", choices=["baseline", "optimized"])
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory(prefix="epistemic-benchmark-") as output:
        results = []
        for implementation in ("baseline", "optimized"):
            command = [
                sys.executable,
                __file__,
                "--baseline",
                str(args.baseline.resolve()),
                "--rows",
                str(args.rows),
                "--cells",
                str(args.cells),
                "--blocks",
                str(args.blocks),
                "--resamples",
                str(args.resamples),
                "--worker",
                implementation,
                "--output",
                output,
            ]
            result = subprocess.run(
                command, check=True, capture_output=True, text=True, env=env
            )
            measurement = json.loads(result.stdout)
            results.append(measurement)
            print(json.dumps(measurement), flush=True)
        for name in ("drift", "surface", "regression"):
            pd.testing.assert_frame_equal(
                pd.read_parquet(Path(output) / f"baseline-{name}.parquet"),
                pd.read_parquet(Path(output) / f"optimized-{name}.parquet"),
                check_exact=False,
                rtol=1e-10,
                atol=1e-12,
            )
        print(
            json.dumps(
                dict(
                    outputs_match=True,
                    rows=args.rows,
                    cells=args.cells,
                    blocks=args.blocks,
                    resamples=args.resamples,
                    speedup=results[0]["total_seconds"] / results[1]["total_seconds"],
                    peak_rss_reduction=results[0]["peak_rss_mib"]
                    / results[1]["peak_rss_mib"],
                )
            )
        )


if __name__ == "__main__":
    main()
