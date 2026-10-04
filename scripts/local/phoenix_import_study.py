#!/usr/bin/env python
"""Import a finished study's analysis archive into Phoenix.

A completed study is distributed as parquet tables, not episode folders, so
``phoenix_import.py`` cannot read it. This produces the same trace shape from
the archive:

    episode -> round -> agent decisions and controller calls

What the archive can and cannot supply is worth stating plainly. Agent prompts
and agents' private reasoning are NOT retained in the aggregate tables, so an
agent span carries its vote, the message it posted and the messages it read,
but no prompt and no reasoning. The controller's calls are retained in full,
prompt and response both.

    scripts/local/phoenix_import_study.py <archive.zip> --task-dir <task>
        --project cesar-v1 --per-condition 10
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import pathlib
import sys
import time
import zipfile
from typing import Any

import pandas as pd
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from mas_cc.llm_runtime.tracing import ROUND_ATTRIBUTES, render_board

LIMIT = 60_000


def _scalar(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    return json.dumps(value, default=str)[:8000]


def _json(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def select(rounds: pd.DataFrame, per_condition: int) -> dict[str, dict[str, Any]]:
    """Pick episodes per (arm, rho, budget, profile), balanced across profiles.

    Keyed on episode_key: episode_id repeats across cells and would silently
    merge two different episodes.
    """

    ordered = rounds.sort_values(["episode_key", "absolute_round"])
    g = ordered.groupby("episode_key")
    meta = pd.DataFrame({
        "name": g["episode_id"].last(),
        "cell": g["source_cell_id"].last(),
        "profile": g["communication_profile"].last(),
        "rho": g["epistemic_persistence"].last(),
        "budget": g["intervention_budget"].last(),
        "target": g["controller_target"].last(),
        "correct": g["correct_answer"].last(),
    }).reset_index()

    def arm_for(row) -> str:
        if pd.isna(row["target"]) or not row["target"] or int(row["budget"] or 0) == 0:
            return "no-controller"
        return "truth-controller" if row["target"] == row["correct"] else "false-controller"

    meta["arm"] = meta.apply(arm_for, axis=1)
    half = max(per_condition // 2, 1)
    chosen: dict[str, dict[str, Any]] = {}
    for _, block in meta.groupby(["arm", "rho", "budget", "profile"], dropna=False):
        for row in block.sort_values("episode_key").head(half).itertuples(index=False):
            chosen[row.episode_key] = {
                "name": row.name, "cell": row.cell, "arm": row.arm,
                "profile": row.profile, "rho": row.rho,
                "budget": None if row.arm == "no-controller" else int(row.budget or 0),
                "correct": row.correct,
            }
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=pathlib.Path)
    parser.add_argument("--task-dir", type=pathlib.Path, required=True)
    parser.add_argument("--project", default="cesar-v1")
    parser.add_argument("--endpoint", default="http://localhost:6006/v1/traces")
    parser.add_argument("--per-condition", type=int, default=10)
    parser.add_argument("--pace", type=float, default=0.5,
                        help="seconds to wait after flushing each episode")
    args = parser.parse_args()

    with zipfile.ZipFile(args.archive) as archive:
        rounds = pd.read_parquet(io.BytesIO(archive.read("tables/rounds.parquet")))
        micro = pd.read_parquet(io.BytesIO(archive.read("tables/micro_slots.parquet")))

    chosen = select(rounds, args.per_condition)
    if not chosen:
        print("no episodes selected", file=sys.stderr)
        return 1
    rounds = rounds[rounds["episode_key"].isin(chosen)]
    micro = micro[micro["episode_key"].isin(chosen)]
    print(f"{len(chosen)} episodes, {len(rounds)} rounds, {len(micro)} agent slots")

    provider = TracerProvider(
        resource=Resource.create({"openinference.project.name": args.project})
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=args.endpoint),
            # Phoenix is one process over SQLite. Pushed flat out it answers
            # 503 and the batch processor eventually drops the batch, which is
            # how the first attempt lost two thirds of its spans. Small batches
            # plus a flush-and-wait per episode keeps ingestion steady.
            max_queue_size=65_536, max_export_batch_size=256,
            schedule_delay_millis=1_000,
        )
    )
    tracer = provider.get_tracer("mas_cc.study_import")

    micro_by_round: dict[tuple, list] = collections.defaultdict(list)
    for row in micro.itertuples(index=False):
        micro_by_round[(row.episode_key, int(row.round_index) + 1)].append(row)

    spans = 0
    done = 0
    for key, info in chosen.items():
        block = rounds[rounds["episode_key"] == key].sort_values("absolute_round")
        if block.empty:
            continue
        with tracer.start_as_current_span(f"episode {info['name']}") as episode_span:
            for name, value in (
                ("run_id", info["name"]), ("cell_id", info["cell"]),
                ("arm", info["arm"]), ("communication_profile", info["profile"]),
                ("epistemic_persistence", info["rho"]), ("study", args.project),
                ("correct_answer", info["correct"]),
            ):
                if value is not None:
                    episode_span.set_attribute(f"mas_cc.{name}", _scalar(value))
            if info["budget"] is not None:
                episode_span.set_attribute("mas_cc.intervention_budget", info["budget"])
            spans += 1

            for event in block.itertuples(index=False):
                rnd = int(event.absolute_round)
                slots = micro_by_round.get((key, rnd), [])
                with tracer.start_as_current_span(f"round {rnd}") as round_span:
                    round_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
                    round_span.set_attribute("mas_cc.run_id", info["name"])
                    round_span.set_attribute("mas_cc.arm", info["arm"])
                    for field in ROUND_ATTRIBUTES:
                        value = getattr(event, field, None)
                        if value is not None and not (isinstance(value, float) and pd.isna(value)):
                            round_span.set_attribute(f"mas_cc.{field}", _scalar(value))
                    board = [_json(s.new_message, None) for s in slots
                             if getattr(s, "new_message", None) is not None]
                    board = [b for b in board if b]
                    for fact in _json(event.controller_report_fact_ids, []):
                        board.append({"author_id": "control-source", "author_kind": "controller",
                                      "message_type": "REPORT", "shared_fact_id": fact,
                                      "text": "(controller report)"})
                    rendered = render_board(board)
                    if rendered:
                        round_span.set_attribute(SpanAttributes.OUTPUT_VALUE, rendered[:LIMIT])
                    spans += 1

                    for slot in slots:
                        agent = str(getattr(slot, "focal_agent_id", "?"))
                        with tracer.start_as_current_span(f"{agent} focal_update") as span:
                            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
                            span.set_attribute("mas_cc.agent_id", agent)
                            span.set_attribute("mas_cc.round", rnd)
                            span.set_attribute("mas_cc.arm", info["arm"])
                            span.set_attribute("mas_cc.run_id", info["name"])
                            for name, field in (
                                ("vote", "focal_opinion_after"),
                                ("vote_before", "focal_opinion_before"),
                                ("shared_fact_id", "new_message_shared_fact_id"),
                                ("message_type", "new_message_type"),
                                ("posted", "focal_posted_message"),
                            ):
                                value = getattr(slot, field, None)
                                if value is not None and not (isinstance(value, float) and pd.isna(value)):
                                    span.set_attribute(f"mas_cc.{name}", _scalar(value))
                            message = _json(getattr(slot, "new_message", None), None)
                            if message and message.get("text"):
                                span.set_attribute(SpanAttributes.OUTPUT_VALUE, message["text"][:LIMIT])
                            read = _json(getattr(slot, "sampled_message_ids", None), [])
                            if read:
                                span.set_attribute("mas_cc.read_message_ids", _scalar(read))
                            spans += 1

                    # The controller's calls are retained in full, unlike the
                    # agents'. One span per attempt, prompt and response both.
                    for attempt in _json(event.controller_llm_attempts, []):
                        request = attempt.get("request") or {}
                        messages = request.get("messages") or []
                        with tracer.start_as_current_span("controller communication") as span:
                            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
                            span.set_attribute("mas_cc.agent_id", "controller")
                            span.set_attribute("mas_cc.round", rnd)
                            span.set_attribute("mas_cc.arm", info["arm"])
                            span.set_attribute("mas_cc.run_id", info["name"])
                            span.set_attribute("mas_cc.attempt", int(attempt.get("attempt") or 1))
                            if messages:
                                span.set_attribute(
                                    SpanAttributes.INPUT_VALUE,
                                    "\n\n".join(m.get("content", "") for m in messages)[:LIMIT],
                                )
                            raw = attempt.get("raw_output")
                            if raw:
                                span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(raw)[:LIMIT])
                                parsed = _json(raw, None)
                                if isinstance(parsed, dict) and parsed.get("reason"):
                                    span.set_attribute("mas_cc.reason", str(parsed["reason"])[:8000])
                            for name, field in (("validation_error", "validation_error"),
                                                ("provider_error", "provider_error")):
                                value = attempt.get(field)
                                if value:
                                    span.set_attribute(f"mas_cc.{name}", str(value)[:4000])
                                    span.set_status(otel_trace.Status(
                                        otel_trace.StatusCode.ERROR, str(value)))
                            spans += 1
        provider.force_flush()
        if args.pace:
            time.sleep(args.pace)
        done += 1
        print(f"  [{done}/{len(chosen)}] {info['arm']:<17} {info['name']:<18} "
              f"spans {spans:,}", flush=True)

    provider.force_flush()
    provider.shutdown()
    print(f"\nimported {spans:,} spans into project {args.project!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
