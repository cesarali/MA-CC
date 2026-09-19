"""Optional OpenTelemetry spans for per-decision timing.

Off unless ``MA_CC_OTEL_ENDPOINT`` is set *and* the OpenTelemetry SDK with the
OTLP/HTTP exporter is importable. When on, :func:`decision_span` opens one span
per logical decision with the same attributes the recorder writes to
``decision_timing.jsonl`` (agent, stage, round, latency split, tokens), exported
to the endpoint (on Cygnus: the Alloy gateway, which forwards to Tempo). The
runtime never depends on this module being functional: every failure to set up
falls back to the no-op path and is reported once on stderr.
"""
from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator, Mapping
from typing import Any

_TRACER: Any = None
_DISABLED = False


def _tracer() -> Any:
    global _TRACER, _DISABLED
    if _TRACER is not None or _DISABLED:
        return _TRACER
    endpoint = os.environ.get("MA_CC_OTEL_ENDPOINT")
    if not endpoint:
        _DISABLED = True
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({
            "service.name": os.environ.get("MA_CC_OTEL_SERVICE", "mas-cc-runner"),
            "mas_cc.slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        }))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces")))
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("mas_cc.runtime")
    except Exception as exc:  # pragma: no cover - depends on optional packages
        print(f"[otel] disabled: {type(exc).__name__}: {exc}", file=sys.stderr)
        _DISABLED = True
        _TRACER = None
    return _TRACER


@contextlib.contextmanager
def decision_span(name: str, attributes: Mapping[str, Any]) -> Iterator[Any]:
    """Context manager: a span when OTel is configured, otherwise a no-op."""
    tracer = _tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(f"mas_cc.{key}", value)
        yield span


def enabled() -> bool:
    return _tracer() is not None


__all__ = ["decision_span", "enabled"]
