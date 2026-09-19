"""Measure System One throughput and error behaviour at increasing concurrency.

TypeSafe publishes no rate limits. Before running an instrument over tens of
thousands of messages, find the ceiling on our gateway path with a bounded
probe: distinct real messages (from a semantic_attribution dry-run table), no
cache, concurrency levels 1, 2, 4, 8, 16, 32, a fixed number of requests per
level, and a hard cap on total requests.

    python -m scripts.Cygnus.analysis.systemone_ramp --messages <dry-run parquet> \
        --output ramp.json [--per-level 40] [--max-requests 300] [--batch-size 1]

Prints one line per level: requests, wall seconds, requests/s, mean latency,
p95 latency, retries, HTTP errors. Writes the same as JSON. Cost is reported
from the usage the API returns.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

from mas_cc.analysis import semantic_attribution as sa
from mas_cc.analysis import systemone

LEVELS = (1, 2, 4, 8, 16, 32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--messages", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-level", type=int, default=40)
    parser.add_argument("--max-requests", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--levels", type=str, default=",".join(map(str, LEVELS)))
    args = parser.parse_args(argv)
    messages = sa.messages_from_table(pd.read_parquet(args.messages))
    levels = [int(x) for x in args.levels.split(",") if x]
    needed = args.per_level * len(levels) * args.batch_size
    if len(messages) < needed:
        print(f"need {needed} distinct messages, have {len(messages)}; lowering per-level", file=sys.stderr)
    cursor = 0
    sent = 0
    rows = []
    for level in levels:
        if sent >= args.max_requests:
            break
        count = min(args.per_level, (args.max_requests - sent))
        chunk = messages[cursor: cursor + count * args.batch_size]
        cursor += len(chunk)
        if not chunk:
            break
        client = systemone.SystemOneClient(cache_dir=None, workers=level)
        items = [(sa.message_state(m), sa.message_questions(m)) for m in chunk]
        started = time.monotonic()
        errors = 0
        try:
            records = client.ask_many(items, batch_size=args.batch_size, workers=level)
        except systemone.SystemOneError as exc:
            errors += 1
            records = []
            print(f"level {level}: {exc}", file=sys.stderr)
        wall = time.monotonic() - started
        latencies = sorted(r["seconds"] * r.get("batch_size", 1) for r in records) or [float("nan")]
        row = {
            "concurrency": level, "requests": client.usage.requests, "messages": len(records), "wall_seconds": round(wall, 2),
            "requests_per_second": round(client.usage.requests / wall, 2) if wall else None,
            "mean_latency_s": round(statistics.fmean(latencies), 3) if records else None,
            "p95_latency_s": round(latencies[int(0.95 * (len(latencies) - 1))], 3) if records else None,
            "retries": client.usage.retries, "errors": errors,
            "input_tokens": client.usage.input_tokens, "usd": round(client.usage.usd, 5),
        }
        rows.append(row)
        sent += client.usage.requests
        print(json.dumps(row))
    args.output.write_text(json.dumps({"levels": rows, "batch_size": args.batch_size, "model": systemone.DEFAULT_MODEL}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
