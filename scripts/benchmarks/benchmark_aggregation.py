"""Measure a retained generation without publishing or modifying its fragments.

Run in the project environment, on an allocated compute node for full studies.
Use PYTHONPATH to select a frozen source tree for a baseline measurement.
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import resource
from time import perf_counter

from mas_cc.studies import aggregation
from mas_cc.analysis import causal_response, epistemic_phase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    args.output.mkdir(parents=True)
    measurements = []

    def instrument(module, name):
        original = getattr(module, name)

        @functools.wraps(original)
        def measured(*a, **kw):
            start = perf_counter()
            result = original(*a, **kw)
            record = dict(stage=name, seconds=perf_counter() - start,
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
            measurements.append(record)
            print(json.dumps(record), flush=True)
            return result
        setattr(module, name, measured)

    for module, names in (
        (aggregation, ("discover_runs", "discover_cells", "read_scientific_table",
                       "_information_tables", "_current_primary", "_affinity_primary",
                       "_derived", "_render_plots", "_package")),
        (causal_response, ("build_causal_response_inputs", "estimate_causal_response",
                           "estimate_available_causal_susceptibility", "build_communication_funnel",
                           "communication_efficiency", "analyze_causal_communication")),
        (epistemic_phase, ("load_symbolic_tasks", "build_epistemic_round_states",
                          "estimate_joint_drift", "estimate_epistemic_modulation",
                          "classify_capture_timing", "analyze_epistemic_phase_diagrams",
                          "render_epistemic_phase_plots")),
    ):
        for name in names:
            instrument(module, name)
    manifest = json.loads(args.manifest.read_text())
    start = perf_counter()
    result = aggregation._aggregate_study_local(
        manifest["study_dir"], allow_incomplete=manifest["allow_incomplete"],
        analysis_output_dir=args.output / "analysis",
        canonical_snapshot_dir=args.manifest.parent / "input",
        information_fragments_dir=args.manifest.parent / "groups",
        information_fragments_analysis_hash=manifest["analysis_hash"],
    )
    payload = dict(seconds=perf_counter() - start,
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   stages=measurements, result=result)
    (args.output / "measurements.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
