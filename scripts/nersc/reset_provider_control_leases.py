#!/usr/bin/env python3
"""Clear provider leases orphaned by a completed NERSC allocation.

The four-hour interactive wall can terminate workers before their ``finally``
blocks release shared provider leases.  The rollover launcher calls this only
after the prior allocation process has exited and while holding the study's
allocation lock.  Adaptive limits and provider-health history are preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - NERSC is POSIX-only
    fcntl = None  # type: ignore[assignment]


def _write_atomic(path: Path, state: Mapping[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix="state-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def reset_orphaned_leases(study_dir: str | Path, *, now: float | None = None) -> int:
    """Clear leases after the previous allocation has definitively exited."""

    if fcntl is None:
        raise RuntimeError("provider lease reset requires POSIX file locking")
    root = Path(study_dir).expanduser().resolve()
    if not (root / "execution_manifest.csv").is_file():
        raise ValueError(f"prepared execution manifest is missing: {root}")
    control = root / "runtime" / "provider-control"
    state_path = control / "state.json"
    if not state_path.is_file():
        return 0

    control.mkdir(parents=True, exist_ok=True)
    lock_path = control / "state.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError(f"provider load-control state is not an object: {state_path}")
        state = dict(loaded)
        leases = state.get("leases", {})
        if not isinstance(leases, Mapping):
            raise ValueError(f"provider load-control leases are not an object: {state_path}")
        cleared = len(leases)
        state["leases"] = {}
        state["last_allocation_lease_reset_at"] = float(time.time() if now is None else now)
        state["last_allocation_lease_reset_count"] = cleared
        _write_atomic(state_path, state)
    return cleared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        cleared = reset_orphaned_leases(args.study_dir)
    except (OSError, TypeError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"[nersc] cleared_orphaned_provider_leases={cleared}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
