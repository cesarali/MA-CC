"""Run cost from the LLM gateway, recorded with the study.

The gateway (LiteLLM) tracks spend per virtual key. ``/key/info`` returns the
key's own cumulative spend and is readable by the key itself on the in-cluster
Service (``LLM_BASE``) and on the tailnet host; the public edge blocks it. A
study's cost is therefore the difference between two snapshots of the key that
ran it: one taken at ``study submit`` (written to ``run_cost.json`` in the
study root) and one taken when the finalizer starts, written into
``analysis_manifest.json`` under ``run_cost``. Nothing here spends anything.

Caveats recorded in the output rather than hidden: the delta attributes every
call made with that key in the window to the study (a second study on the same
key at the same time is not separable this way), and if the gateway is
unreachable the record says ``status: unavailable`` instead of failing the run.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RUN_COST_FILE = "run_cost.json"


def _key() -> str | None:
    path = os.environ.get("LLM_KEY_FILE") or os.path.expanduser("~/.llm.key")
    if os.environ.get("LLM_KEY"):
        return os.environ["LLM_KEY"].strip()
    if os.path.isfile(path):
        return Path(path).read_text(encoding="utf-8").strip() or None
    return None


def _base() -> str | None:
    base = os.environ.get("MA_CC_GATEWAY_ADMIN_BASE") or os.environ.get("LLM_BASE")
    return base.rstrip("/") if base else None


def key_fingerprint(key: str) -> str:
    """Non-reversible identifier for the key in records (first 12 hex of sha256)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def key_spend(*, base: str | None = None, key: str | None = None, timeout: float = 15.0) -> dict[str, Any]:
    """One snapshot: ``{status, spend_usd, key_alias, user_id, key_fingerprint, base, fetched_at}``."""
    key = key or _key()
    base = base or _base()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not key or not base:
        return {"status": "unavailable", "reason": "no gateway key or base url in the environment", "fetched_at": stamp}
    request = urllib.request.Request(
        f"{base}/key/info", headers={"authorization": f"Bearer {key}", "accept": "application/json",
                                     "user-agent": "mas-cc-gateway-spend/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {str(exc)[:160]}",
                "key_fingerprint": key_fingerprint(key), "base": base, "fetched_at": stamp}
    info = body.get("info") if isinstance(body, Mapping) else None
    if not isinstance(info, Mapping) or "spend" not in info:
        return {"status": "unavailable", "reason": "unexpected /key/info body", "key_fingerprint": key_fingerprint(key),
                "base": base, "fetched_at": stamp}
    return {
        "status": "ok", "spend_usd": float(info.get("spend") or 0.0), "key_alias": info.get("key_alias"),
        "user_id": info.get("user_id"), "team_id": info.get("team_id"), "max_budget": info.get("max_budget"),
        "key_fingerprint": key_fingerprint(key), "base": base, "fetched_at": stamp,
    }


def record_run_start(study_dir: str | Path) -> dict[str, Any]:
    """Write the submit-time snapshot to ``<study>/run_cost.json`` (never raises)."""
    record = {"schema_version": 1, "start": key_spend()}
    path = Path(study_dir) / RUN_COST_FILE
    try:
        path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    except OSError:
        pass
    return record


def run_cost(study_dir: str | Path) -> dict[str, Any]:
    """Close the window: read the start snapshot, take the end snapshot, return the delta record."""
    path = Path(study_dir) / RUN_COST_FILE
    start: dict[str, Any] | None = None
    if path.is_file():
        try:
            start = json.loads(path.read_text(encoding="utf-8")).get("start")
        except (OSError, ValueError):
            start = None
    end = key_spend()
    record: dict[str, Any] = {"schema_version": 1, "start": start, "end": end, "status": "unavailable"}
    if start and start.get("status") == "ok" and end.get("status") == "ok":
        if start.get("key_fingerprint") == end.get("key_fingerprint"):
            record["usd"] = round(end["spend_usd"] - start["spend_usd"], 6)
            record["status"] = "ok"
            record["attribution"] = "all calls on this key between the submit and finalize snapshots"
        else:
            record["status"] = "key_changed"
    elif start is None:
        record["status"] = "no_start_snapshot"
    return record


__all__ = ["RUN_COST_FILE", "key_fingerprint", "key_spend", "record_run_start", "run_cost"]
