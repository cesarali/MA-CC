#!/usr/bin/env python
"""Freeze a controller fact pool with no built-in lean toward any allocation.

Every fact in the pool is true. What differs between facts is which allocation
each one favours when read alone: the allocation whose probability it raises
most, under the task generator's exact world model. In task_004 the full
non-decisive pool leans 25 / 9 / 9 toward truth / A1 / A2, so a controller
given it starts with a menu slanted toward the answer it may be trying to hide.

This keeps every rival-leaning fact and matches them with an equal number of
truth-leaning ones. The truth-leaning facts are chosen to match the rivals'
*strength distribution* (minimum Wasserstein-1 distance), not at random
and not the strongest: keeping
the 9 strongest truth facts would bring the slant straight back.

    scripts/local/build_balanced_controller_pool.py <task-dir>

Writes <task-dir>/controller/balanced_fact_pool.json. The game only reads it.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

from mas_cc.musr_team_allocation_generator.ambiguity import TeamAllocationCompletionIndex
from mas_cc.musr_team_allocation_generator.symbolic_facts import CanonicalFact

SCHEMA_VERSION = 1
RULE = "keep_all_rival_leaning__wasserstein_matched_truth_leaning_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=pathlib.Path)
    args = parser.parse_args()
    root = args.task_dir

    task = json.loads((root / "task.json").read_text())
    truth = int(str(task["gold_target"]).rsplit("_", 1)[1])
    rows = json.loads((root / "facts/all_true_facts.json").read_text())
    decisive = {row["fact_id"] for row in json.loads((root / "facts/decisive_facts.json").read_text())}
    facts = [CanonicalFact.from_dict(row) for row in rows if row["fact_id"] not in decisive]

    index = TeamAllocationCompletionIndex()
    prior = index.metrics_for_facts(()).probabilities
    scored = []
    for fact in facts:
        posterior = index.metrics_for_facts((fact,)).probabilities
        lift = [posterior[k] - prior[k] for k in range(3)]
        lean = max(range(3), key=lambda k: (lift[k], -k))
        scored.append({
            "fact_id": fact.fact_id,
            "lean": f"ALLOCATION_{lean}",
            "strength": round(lift[lean], 6),
            "posterior": [round(p, 6) for p in posterior],
        })

    truthy = sorted((r for r in scored if r["lean"] == f"ALLOCATION_{truth}"),
                    key=lambda r: (r["strength"], r["fact_id"]))
    rivals = sorted((r for r in scored if r["lean"] != f"ALLOCATION_{truth}"),
                    key=lambda r: (r["strength"], r["fact_id"]))
    by_rival = collections.Counter(r["lean"] for r in rivals)
    per_allocation = min(by_rival.values())
    if len(by_rival) != 2 or len(set(by_rival.values())) != 1:
        # Unequal rival groups would need trimming too; task_004 does not.
        print(f"rival groups are not equal: {dict(by_rival)}", file=sys.stderr)
        return 1
    if len(truthy) < per_allocation:
        print("fewer truth-leaning facts than rival groups", file=sys.stderr)
        return 1

    # Strength matching: choose the truth-leaning facts whose strength
    # distribution is closest to the rivals', measured by the Wasserstein-1
    # distance (the least total strength that has to be moved to turn one
    # distribution into the other). Exhaustive over value multisets, which is
    # small because strengths sit on a coarse grid; ties go to the closer mean,
    # then to fact_id, so the result is deterministic. Picking one quantile at
    # a time instead drifts upward on this grid, i.e. quietly re-slants.
    rival_strengths = sorted(r["strength"] for r in rivals)
    rival_mean = sum(rival_strengths) / len(rival_strengths)

    def w1(a, b):
        points = sorted(set(a) | set(b))
        total = 0.0
        for left, right in zip(points, points[1:]):
            fa = sum(1 for x in a if x <= left) / len(a)
            fb = sum(1 for x in b if x <= left) / len(b)
            total += abs(fa - fb) * (right - left)
        return total

    groups: dict[float, list[dict]] = collections.defaultdict(list)
    for row in truthy:
        groups[row["strength"]].append(row)
    values = sorted(groups)
    best_key, best_counts = None, None

    def search(i, remaining, counts):
        nonlocal best_key, best_counts
        if remaining == 0:
            picked = [v for v, c in zip(values, counts) for _ in range(c)]
            key = (round(w1(picked, rival_strengths), 12),
                   round(abs(sum(picked) / len(picked) - rival_mean), 12))
            if best_key is None or key < best_key:
                best_key, best_counts = key, list(counts)
            return
        if i == len(values):
            return
        for c in range(min(remaining, len(groups[values[i]])), -1, -1):
            search(i + 1, remaining - c, counts + [c])

    search(0, per_allocation, [])
    best_counts += [0] * (len(values) - len(best_counts))
    chosen: list[dict] = []
    for value, count in zip(values, best_counts):
        chosen += sorted(groups[value], key=lambda r: r["fact_id"])[:count]
    match_quality = {"wasserstein_1": best_key[0], "abs_mean_gap": best_key[1]}

    selected = sorted(chosen + rivals, key=lambda r: r["fact_id"])

    def mean(xs):
        return round(sum(xs) / len(xs), 6)

    joint = {}
    for label, group in (("truth_leaning", chosen),
                         *((f"leaning_{k}", [r for r in rivals if r["lean"] == k]) for k in sorted(by_rival)),
                         ("whole_pool", selected)):
        cf = [f for f in facts if f.fact_id in {r["fact_id"] for r in group}]
        joint[label] = [round(p, 6) for p in index.metrics_for_facts(tuple(cf)).probabilities]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "rule": RULE,
        "task_id": task["task_id"],
        "gold_target": task["gold_target"],
        "prior": [round(p, 6) for p in prior],
        "source_pool": {
            "definition": "all true facts minus decisive facts",
            "size": len(scored),
            "lean_counts": dict(collections.Counter(r["lean"] for r in scored)),
        },
        "lean_counts": dict(collections.Counter(r["lean"] for r in selected)),
        "mean_strength": {
            "truth_leaning_selected": mean([r["strength"] for r in chosen]),
            "truth_leaning_available": mean([r["strength"] for r in truthy]),
            "rival_leaning": mean([r["strength"] for r in rivals]),
        },
        "strength_match": match_quality,
        "joint_posterior": joint,
        "fact_ids": [r["fact_id"] for r in selected],
        "facts": selected,
    }
    out = root / "controller/balanced_fact_pool.json"
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out.write_text(text)
    print(f"wrote {out}  sha256={hashlib.sha256(text.encode()).hexdigest()}")
    print(json.dumps({k: payload[k] for k in ("source_pool", "lean_counts", "mean_strength", "joint_posterior")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
