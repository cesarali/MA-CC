#!/usr/bin/env python
"""Remove duplicate episode traces from a Phoenix project.

An episode can end up with two traces: one from the retro importer and one
streamed live, or a stub left by importing it while it was still running. Both
carry the same ``mas_cc.run_id``.

The survivor is the trace with the most spans, then the most prompts. Spans
first because a span IS one agent decision: a live trace that lost calls to a
full export queue, or was cut short by a crash, is genuinely poorer than a
complete retro one even though live tracing keeps more prompts per call.
Between two complete traces the span counts tie and the prompt-rich live trace
wins, which is what you want.

    scripts/local/phoenix_dedup.py                 # report only
    scripts/local/phoenix_dedup.py --apply         # actually delete

Run it only once the study has finished. An episode still in flight looks
exactly like a stub, and would be deleted while it is still being written.

Deletion is per span, so a trace is removed by deleting every span in it.
"""

from __future__ import annotations

import argparse
import collections
import sys
from typing import Any

import httpx


def _spans(client: httpx.Client, project: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"limit": 1000}
        if cursor:
            params["cursor"] = cursor
        payload = client.get(f"/v1/projects/{project}/spans", params=params).json()
        out.extend(payload.get("data", []))
        cursor = payload.get("next_cursor")
        if not cursor:
            return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="mas-cc")
    parser.add_argument("--base-url", default="http://localhost:6006")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=300.0) as client:
        spans = _spans(client, args.project)
        print(f"{args.project}: {len(spans)} spans")

        by_trace: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for span in spans:
            by_trace[span["context"]["trace_id"]].append(span)

        # Identity comes from the root episode span. run_id alone is NOT
        # unique: it is just the episode directory name, so no_control
        # cell-0003-0004 and controlled cell-0003-0004 share it. Keying on
        # (arm, cell, run) keeps the two arms apart -- without it this would
        # delete one arm's trace as a "duplicate" of the other's.
        traces: dict[tuple, list[tuple[str, int, int, int]]] = collections.defaultdict(list)
        for trace_id, members in by_trace.items():
            root = next(
                (s for s in members if s["name"].startswith("episode ")), None
            )
            if root is None:
                continue
            attributes = root["attributes"]
            run_id = attributes.get("mas_cc.run_id")
            if not run_id:
                continue
            identity = (
                attributes.get("mas_cc.arm"),
                attributes.get("mas_cc.cell_id"),
                run_id,
            )
            rounds = sum(1 for s in members if s["name"].startswith("round "))
            prompts = sum(1 for s in members if s["attributes"].get("input.value"))
            traces[identity].append((trace_id, rounds, prompts, len(members)))

        doomed: list[str] = []
        for identity, found in sorted(traces.items(), key=lambda kv: str(kv[0])):
            if len(found) < 2:
                continue
            found.sort(key=lambda item: (item[3], item[2], item[1]), reverse=True)
            keep, *drop = found
            print(
                f"  {'/'.join(str(part) for part in identity)}: "
                f"keeping {keep[1]}r/{keep[2]}p/{keep[3]}s, "
                f"dropping {[f'{r}r/{p}p/{n}s' for _, r, p, n in drop]}"
            )
            doomed.extend(trace_id for trace_id, _, _, _ in drop)

        if not doomed:
            print("no duplicate episode traces")
            return 0
        victims = [s for t in doomed for s in by_trace[t]]
        print(f"\n{len(doomed)} duplicate trace(s), {len(victims)} spans")
        if not args.apply:
            print("dry run; pass --apply to delete")
            return 0
        # Children first, so a parent is never left pointing at nothing if the
        # pass is interrupted part-way.
        victims.sort(key=lambda s: s["start_time"], reverse=True)
        removed = 0
        for span in victims:
            try:
                client.delete(f"/v1/spans/{span['id']}")
                removed += 1
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  failed on {span['id']}: {exc}", file=sys.stderr)
        print(f"deleted {removed} spans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
