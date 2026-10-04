"""Optional OpenInference tracing for live runs.

Off by default. Everything here is inert unless ``MAS_CC_TRACE_ENDPOINT`` is
set, and if the OpenTelemetry packages are not installed it stays inert even
then -- so no run can fail because tracing is unavailable.

    MAS_CC_TRACE_ENDPOINT=http://localhost:6006/v1/traces \
    MAS_CC_TRACE_PROJECT=mas-cc \
        mas-cc experiment run --config ...

One span per episode, one child span per provider call (including every
validation retry, which the on-disk artifacts only keep as a counter). The
attribute names match ``scripts/local/phoenix_import.py`` so a live run and a
re-imported finished run are directly comparable in the same Phoenix project.

Why a hand-rolled span instead of an auto-instrumentor: the adapters speak to
the provider through their own httpx session with a bespoke retry and lease
loop, so nothing standard has a seam to hook. Wrapping ``complete`` also keeps
the span aligned with one *logical* request rather than one socket write.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from typing import Any, Iterator, Mapping

_ENDPOINT_ENV = "MAS_CC_TRACE_ENDPOINT"
_PROJECT_ENV = "MAS_CC_TRACE_PROJECT"
_PROMPT_LIMIT = 60_000

_lock = threading.Lock()
_provider: Any | None = None
_tracer: Any | None = None
_unavailable = False

# Episode identity for the calls made inside it. A ContextVar rather than a
# global because episodes run concurrently on one event loop: each asyncio task
# gets its own copy, so four episodes in flight cannot overwrite each other.
try:  # pragma: no cover - trivial
    from contextvars import ContextVar

    _episode: ContextVar[Mapping[str, Any]] = ContextVar("mas_cc_episode", default={})
    # Set while a decision-level span is open, so the adapter does not nest a
    # second, poorer span inside it for the same call.
    _active: ContextVar[bool] = ContextVar("mas_cc_llm_span", default=False)
    # The open round span's context, so each agent call hangs under the round
    # it happened in. Carried explicitly rather than by attach/detach: the
    # round body is long and the explicit form cannot leak a stale context.
    _round: ContextVar[Any] = ContextVar("mas_cc_round_ctx", default=None)
except Exception:  # pragma: no cover
    _episode = None  # type: ignore[assignment]
    _active = None  # type: ignore[assignment]
    _round = None  # type: ignore[assignment]


# One round is ~400 recorded quantities. Only these become span attributes;
# the rest stay in round_trajectory.jsonl, where the analysis code reads them.
# The cut is "what you would want to see next to the agents' reasoning".
ROUND_ATTRIBUTES: tuple[str, ...] = (
    "absolute_round",
    "analysis_target",
    "correct_answer",
    "epistemic_persistence",
    "intervention_budget",
    "requested_b",
    # where the vote stands
    "truth_vote_share_before",
    "truth_vote_share",
    "vote_entropy_before",
    "vote_entropy",
    "occupation_counts_before",
    "occupation_counts_after",
    "delta_m_truth",
    "delta_m_ctrl",
    # the board itself
    "board_messages_created",
    "board_messages_expired",
    "board_mean_size",
    "board_peak_size",
    "surviving_message_count",
    "message_type_counts",
    "report_count",
    "request_count",
    "total_eligible_board_message_reads",
    "peer_report_exposures",
    # what the board did to what agents know
    "new_evidence_acquisitions",
    "new_peer_facts",
    "new_controller_facts",
    "persistence_deactivated_fact_count",
    "mean_known_fact_count",
    "mean_supporting_fact_coverage",
    "full_proof_agent_share",
    # the controller's turn
    "controller_enabled",
    "controller_target",
    "controller_target_share",
    "controller_posts",
    "actual_controller_posts",
    "controller_reports_requested",
    "controller_reports_admitted",
    "controller_report_read_share",
    "controller_unique_readers",
    "controller_report_target_adoptions",
    "controller_fact_id",
    "controller_fact_text",
    "chosen_message_mode",
    "communication_choice_reason",
    "controller_llm_fallback_used",
)


def enabled() -> bool:
    """True when tracing is configured and usable."""

    return bool(os.environ.get(_ENDPOINT_ENV)) and not _unavailable


def _get_tracer() -> Any | None:
    global _provider, _tracer, _unavailable
    if _unavailable or not os.environ.get(_ENDPOINT_ENV):
        return None
    if _tracer is not None:
        return _tracer
    with _lock:
        if _tracer is not None:
            return _tracer
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except Exception:
            _unavailable = True
            return None
        project = os.environ.get(_PROJECT_ENV) or "mas-cc"
        provider = TracerProvider(
            resource=Resource.create({"openinference.project.name": project})
        )
        # A queue big enough that a busy collector cannot cost us data. One
        # episode is ~250 spans and several run concurrently, so the 2048
        # default overflows under load -- and OpenTelemetry discards the
        # overflow silently, which showed up later as episodes whose traces
        # were missing most of their agent calls.
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=os.environ[_ENDPOINT_ENV]),
                max_queue_size=65_536,
                max_export_batch_size=512,
                schedule_delay_millis=2_000,
            )
        )
        _provider = provider
        _tracer = provider.get_tracer("mas_cc")
        return _tracer


def shutdown() -> None:
    """Flush buffered spans. Safe to call when tracing was never started."""

    global _provider, _tracer
    with _lock:
        provider, _provider, _tracer = _provider, None, None
    if provider is not None:
        with contextlib.suppress(Exception):
            provider.force_flush()
        with contextlib.suppress(Exception):
            provider.shutdown()


@contextlib.contextmanager
def episode_span(
    *, episode_id: str, cell_id: str | None, attributes: Mapping[str, Any]
) -> Iterator[None]:
    """Group one episode's provider calls under a single trace."""

    tracer = _get_tracer()
    if tracer is None:
        yield
        return
    payload = {
        "mas_cc.run_id": episode_id,
        "mas_cc.cell_id": cell_id or "run",
        **{f"mas_cc.{key}": _scalar(value) for key, value in attributes.items()},
    }
    token = _episode.set(payload) if _episode is not None else None
    try:
        with tracer.start_as_current_span(f"episode {episode_id}") as span:
            for key, value in payload.items():
                span.set_attribute(key, value)
            yield
    finally:
        if token is not None:
            _episode.reset(token)


def _scalar(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple, dict)):
        with contextlib.suppress(Exception):
            return json.dumps(value, default=str)[:8000]
    return str(value)



def actor(agent_id: Any, stage: str) -> str:
    """Who made this call.

    The controller's own completions are logged with the *string* "None" as
    their agent id (it is not one of the population), which would otherwise
    label its spans "None".
    """

    if agent_id in (None, "None", ""):
        return "controller" if "controller" in stage else "?"
    return str(agent_id)


def round_begin(round_index: int, previous: Any = None) -> Any:
    """Open a span for one game round; agent calls then nest inside it.

    Returns an opaque handle to pass to :func:`round_end`, or ``None`` when
    tracing is off. ``previous`` closes a round span that was left open by an
    episode that failed part-way through, so a crash cannot strand a span.
    """

    if previous is not None:
        round_end(previous)
    tracer = _get_tracer()
    if tracer is None:
        return None
    span = tracer.start_span(f"round {round_index + 1}")
    if _round is not None:
        from opentelemetry import trace as otel_trace

        _round.set(otel_trace.set_span_in_context(span))
    return span


def round_end(
    span: Any,
    *,
    event: Mapping[str, Any] | None = None,
    board: Any = None,
) -> None:
    """Attach the round's outcome and its blackboard, then close the span."""

    if span is None:
        return
    if _round is not None:
        _round.set(None)
    with contextlib.suppress(Exception):
        try:
            from openinference.semconv.trace import SpanAttributes as SA

            span.set_attribute(SA.OPENINFERENCE_SPAN_KIND, "CHAIN")
            output_key = SA.OUTPUT_VALUE
        except Exception:
            output_key = "output.value"
        if _episode is not None:
            for key, value in (_episode.get() or {}).items():
                span.set_attribute(key, value)
        for key in ROUND_ATTRIBUTES:
            value = (event or {}).get(key)
            if value is not None:
                span.set_attribute(f"mas_cc.{key}", _scalar(value))
        if board is not None:
            rendered = render_board(board)
            if rendered:
                span.set_attribute(output_key, rendered[:_PROMPT_LIMIT])
    with contextlib.suppress(Exception):
        span.end()


def render_board(messages: Any) -> str:
    """The blackboard as the agents would read it, one message per block."""

    lines: list[str] = []
    for message in messages or ():
        get = (
            message.get
            if isinstance(message, Mapping)
            else lambda key, _m=message: getattr(_m, key, None)
        )
        identifier = get("message_id")
        header = f"[{identifier}] " if identifier else ""
        header += str(get("author_id"))
        kind = get("author_kind")
        if kind and kind != "agent":
            header += f" ({kind})"
        header += f" — {get('message_type')}"
        if get("shared_fact_id"):
            header += f" · fact {get('shared_fact_id')}"
        if get("reply_to"):
            header += f" · reply to {get('reply_to')}"
        if get("vote"):
            header += f" · votes {get('vote')}"
        lines.append(f"{header}\n{get('text')}")
    return "\n\n".join(lines)


@contextlib.contextmanager
def llm_span(
    *, provider: str, model: str, request: Any, skip_if_active: bool = False
) -> Iterator[Any]:
    """Wrap one logical provider call.

    Yields a recorder; call ``record(response)`` on success and, where the
    caller knows it, ``decision(...)`` for the validated outcome. An exception
    escaping the block marks the span as an error and is re-raised untouched.

    ``skip_if_active`` lets the transport adapter stand down when a caller
    higher up (the decision loop) has already opened a richer span for this
    same call.
    """

    tracer = _get_tracer()
    if tracer is None:
        yield _NullRecorder()
        return
    if skip_if_active and _active is not None and _active.get():
        yield _NullRecorder()
        return

    meta: Mapping[str, Any] = getattr(request, "metadata", {}) or {}
    stage = str(meta.get("decision_stage") or "call")
    agent = actor(meta.get("agent_id"), stage)
    name = f"{agent} {stage}"

    try:
        from openinference.semconv.trace import SpanAttributes as SA
    except Exception:  # pragma: no cover - guarded by _get_tracer's import
        yield _NullRecorder()
        return

    # Parent under the open round span when there is one, so the trace tree
    # reads episode -> round -> agent decision.
    parent = _round.get() if _round is not None else None
    with tracer.start_as_current_span(name, context=parent) as span:
        span.set_attribute(SA.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SA.LLM_PROVIDER, provider)
        span.set_attribute(SA.LLM_MODEL_NAME, model)
        if _episode is not None:
            for key, value in _episode.get().items():
                span.set_attribute(key, value)
        for key in (
            "agent_id",
            "interaction_id",
            "decision_stage",
            "attempt",
            "validation_attempt",
            "prompt_family",
            "prompt_version",
            "game_type",
        ):
            if meta.get(key) is not None:
                span.set_attribute(f"mas_cc.{key}", _scalar(meta[key]))
        # The rendered prompt is the span input. On disk only the first N
        # interactions keep a prompt file; here every call has one.
        with contextlib.suppress(Exception):
            messages = request.wire_messages()
            span.set_attribute(
                SA.INPUT_VALUE,
                "\n\n".join(
                    f"[{m['role']}]\n{m['content']}" for m in messages
                )[:_PROMPT_LIMIT],
            )
            for index, message in enumerate(messages):
                span.set_attribute(
                    f"{SA.LLM_INPUT_MESSAGES}.{index}.message.role", message["role"]
                )
                span.set_attribute(
                    f"{SA.LLM_INPUT_MESSAGES}.{index}.message.content",
                    message["content"][:_PROMPT_LIMIT],
                )
        recorder = _SpanRecorder(span, SA)
        token = _active.set(True) if _active is not None else None
        try:
            yield recorder
        except Exception as exc:
            with contextlib.suppress(Exception):
                from opentelemetry import trace as otel_trace

                span.record_exception(exc)
                span.set_status(
                    otel_trace.Status(otel_trace.StatusCode.ERROR, str(exc))
                )
            raise
        finally:
            if token is not None:
                _active.reset(token)


class _NullRecorder:
    def __call__(self, response: Any) -> None:
        return None

    record = __call__

    def decision(self, **_: Any) -> None:
        return None


class _SpanRecorder:
    def __init__(self, span: Any, semconv: Any) -> None:
        self._span = span
        self._sa = semconv

    def __call__(self, response: Any) -> None:
        self.record(response)

    def record(self, response: Any) -> None:
        span, sa = self._span, self._sa
        with contextlib.suppress(Exception):
            # A reasoning model returns its chain of thought beside the answer,
            # not inside it: `content` holds only the final JSON. Without this
            # the whole point of a reasoning arm would be invisible in the
            # trace. Field name varies by provider, hence both spellings.
            message = (
                (getattr(response, "raw_response", None) or {})
                .get("choices", [{}])[0]
                .get("message", {})
            )
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                span.set_attribute("mas_cc.reasoning", reasoning[:_PROMPT_LIMIT])
                span.set_attribute(
                    f"{sa.LLM_OUTPUT_MESSAGES}.0.message.contents.0.message_content.type",
                    "text",
                )
        with contextlib.suppress(Exception):
            content = getattr(response, "content", None)
            if isinstance(content, str):
                span.set_attribute(sa.OUTPUT_VALUE, content[:_PROMPT_LIMIT])
                span.set_attribute(
                    f"{sa.LLM_OUTPUT_MESSAGES}.0.message.role", "assistant"
                )
                span.set_attribute(
                    f"{sa.LLM_OUTPUT_MESSAGES}.0.message.content",
                    content[:_PROMPT_LIMIT],
                )
            usage = getattr(response, "usage", None)
            for attribute, field in (
                (sa.LLM_TOKEN_COUNT_PROMPT, "input_tokens"),
                (sa.LLM_TOKEN_COUNT_COMPLETION, "output_tokens"),
                (sa.LLM_TOKEN_COUNT_TOTAL, "total_tokens"),
            ):
                value = getattr(usage, field, None)
                if isinstance(value, int):
                    span.set_attribute(attribute, value)
            for name, field in (
                ("mas_cc.finish_reason", "finish_reason"),
                ("mas_cc.retries", "retries"),
                ("mas_cc.status_code", "status_code"),
                ("mas_cc.latency_seconds", "latency_seconds"),
                ("mas_cc.request_id", "request_id"),
            ):
                value = getattr(response, field, None)
                if value is not None:
                    span.set_attribute(name, _scalar(value))

    def decision(
        self,
        *,
        valid: bool,
        action: Any = None,
        validation_error: str | None = None,
    ) -> None:
        """Record what the game made of this completion.

        A call can succeed at the provider and still be rejected by the
        response contract; that difference is the thing worth seeing in the
        trace, so it becomes the span's status.
        """

        span = self._span
        with contextlib.suppress(Exception):
            span.set_attribute("mas_cc.valid", bool(valid))
            if validation_error:
                span.set_attribute("mas_cc.validation_error", str(validation_error)[:4000])
            value = getattr(action, "value", None)
            if value is not None:
                span.set_attribute("mas_cc.vote", _scalar(value))
            metadata = getattr(action, "metadata", None) or {}
            for key in ("reason", "shared_fact_id", "public_message"):
                if metadata.get(key):
                    name = "private_reason" if key == "reason" else key
                    span.set_attribute(f"mas_cc.{name}", str(metadata[key])[:8000])
            if not valid:
                from opentelemetry import trace as otel_trace

                span.set_status(
                    otel_trace.Status(
                        otel_trace.StatusCode.ERROR,
                        str(validation_error or "invalid response"),
                    )
                )
