"""Exercise the provider coordinator concurrently from several SLURM nodes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import time
from pathlib import Path

from mas_cc.llm_runtime.providers.load_control import (
    ProviderLoadControlConfig,
    SharedProviderCoordinator,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--hold-seconds", type=float, default=0.03)
    parser.add_argument("--deadline-seconds", type=float, default=120.0)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> dict[str, object]:
    rank = os.environ.get("SLURM_PROCID", str(os.getpid()))
    config = ProviderLoadControlConfig.from_mapping(
        {
            "initial_concurrency": args.limit,
            "minimum_concurrency": args.limit,
            "maximum_concurrency": args.limit,
            "target_rpm": 100_000,
            "polling_seconds": 0.01,
        }
    )
    coordinators = [
        SharedProviderCoordinator(
            args.root,
            config,
            worker_id=f"smoke:{rank}:{worker}",
        )
        for worker in range(args.workers)
    ]
    maximum_seen = 0
    stop = asyncio.Event()

    async def monitor() -> None:
        nonlocal maximum_seen
        observer = SharedProviderCoordinator(args.root, config)
        while not stop.is_set():
            maximum_seen = max(maximum_seen, len(observer.snapshot()["leases"]))
            await asyncio.sleep(0.005)

    async def worker(coordinator: SharedProviderCoordinator) -> None:
        for _ in range(args.iterations):
            lease = await coordinator.acquire(
                deadline=time.monotonic() + args.deadline_seconds
            )
            await asyncio.sleep(args.hold_seconds)
            await coordinator.release(
                lease,
                success=True,
                retryable=False,
                status_code=200,
                latency_seconds=args.hold_seconds,
                deadline=time.monotonic() + args.deadline_seconds,
            )

    monitor_task = asyncio.create_task(monitor())
    started = time.monotonic()
    await asyncio.gather(*(worker(item) for item in coordinators))
    stop.set()
    await monitor_task
    state = coordinators[0].snapshot()
    return {
        "rank": rank,
        "node": socket.gethostname(),
        "operations": args.workers * args.iterations,
        "elapsed_seconds": time.monotonic() - started,
        "maximum_active_leases_seen": maximum_seen,
        "final_active_leases_seen": len(state["leases"]),
    }


def main() -> None:
    args = _arguments()
    args.root.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(_run(args))
    destination = args.root / f"result-{result['rank']}.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
