"""Transparent rough-latency estimates, not runtime guarantees."""

from __future__ import annotations

import math


DEFAULT_LATENCY_SECONDS = {
    "mock": 0.001,
    "openai": 2.0,
    "university": 3.0,
    "gemma_local": 2.0,
}


def estimate_runtime_seconds(
    provider: str,
    *,
    logical_calls: int,
    request_concurrency: int,
    seconds_per_call: float | None = None,
    one_time_load_seconds: float = 0.0,
) -> float:
    latency = (
        DEFAULT_LATENCY_SECONDS.get(provider, 3.0)
        if seconds_per_call is None
        else seconds_per_call
    )
    if latency < 0 or one_time_load_seconds < 0:
        raise ValueError("latency estimates cannot be negative")
    waves = math.ceil(logical_calls / request_concurrency)
    return one_time_load_seconds + waves * latency
