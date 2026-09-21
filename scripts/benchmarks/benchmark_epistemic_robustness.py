"""Compare literal and lookup mask evaluation using identical survival draws."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from mas_cc.analysis.epistemic_phase import load_symbolic_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--task", default="task_003")
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--draws", type=int, default=500)
    args = parser.parse_args()
    task = load_symbolic_tasks([args.task], args.tasks)[args.task]
    ids = sorted(task.facts)
    rng = np.random.default_rng(459)
    literal_seconds = lookup_seconds = 0.0
    for _ in range(args.states):
        retained = rng.random((args.draws, len(ids))) < rng.uniform(0, 1, len(ids))
        start = perf_counter()
        expected = np.asarray([
            task.solvable_from_masks([fact for fact, keep in zip(ids, draw) if keep])
            for draw in retained
        ])
        literal_seconds += perf_counter() - start
        start = perf_counter()
        actual = task.solvable_survival_draws(ids, retained)
        lookup_seconds += perf_counter() - start
        np.testing.assert_array_equal(actual, expected)
    print(json.dumps(dict(states=args.states, draws=args.draws, facts=len(ids),
                          literal_seconds=literal_seconds, lookup_seconds=lookup_seconds,
                          speedup=literal_seconds / lookup_seconds, exact_agreement=True), indent=2))


if __name__ == "__main__":
    main()
