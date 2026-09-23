#!/usr/bin/env python
"""Import completed MA-CC episodes into Phoenix as OpenInference traces.

Reads the artifacts an ``artifact_profile: full`` run already writes and emits
one trace per episode, one span per game round, and one LLM span per agent
decision underneath the round it happened in:

    api_call_status.jsonl   agent, round, model, provider, validity, errors
    usage_cost.jsonl        token usage per call
    trajectory.jsonl        the decision itself: vote, private_reason, and the
                            message the agent posted to the blackboard
    round_trajectory.jsonl  the round's outcome: truth share, board sizes,
                            controller activity (see tracing.ROUND_ATTRIBUTES)
    prompts/round_NNN.md    the rendered prompt the agent saw

The round span carries that round's blackboard as its output, rebuilt from the
messages agents posted. A controller's own posts are not agent decisions and so
cannot be recovered this way -- live tracing reads the board object itself and
does not have that gap.

Nothing is read from the provider and nothing is re-run, so this is safe to
point at a study while it is still executing; incomplete episodes are skipped.

Usage:
    scripts/local/phoenix_import.py results/studies/task003_v0_deepseek/no_control
    scripts/local/phoenix_import.py <dir> --endpoint http://localhost:6006/v1/traces
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict
from typing import Any, Iterator, Mapping

from mas_cc.llm_runtime.tracing import ROUND_ATTRIBUTES, actor, render_board
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def _read_jsonl(path: pathlib.Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _call_key(record: dict[str, Any]) -> tuple:
    return (
        str(record.get("interaction_id")),
        str(record.get("agent_id")),
        int(record.get("attempt") or 1),
    )


def _decisions_by_interaction(episode: pathlib.Path) -> dict[tuple, dict[str, Any]]:
    """Map (interaction_id, agent_id) -> the decision that call produced."""
    out: dict[tuple, dict[str, Any]] = {}
    for row in _read_jsonl(episode / "trajectory.jsonl"):
        interaction = str(row.get("interaction_id"))
        for decision in row.get("decisions") or []:
            action = decision.get("action") or decision
            agent = action.get("agent_id") or decision.get("agent_id")
            if agent is None:
                continue
            meta = action.get("metadata") or {}
            out[(interaction, str(agent))] = {
                "vote": action.get("value"),
                "reason": meta.get("reason"),
                "shared_fact_id": meta.get("shared_fact_id"),
                "public_message": meta.get("public_message"),
            }
    return out


def _prompts_by_interaction(episode: pathlib.Path) -> dict[tuple[int, str], str]:
    """Map (interaction_index, agent_id) -> rendered prompt.

    "Round" in both the filename and the header is really the *interaction
    index* (1..population_size*rounds), the same number api_call_status.jsonl
    stores as ``round_index`` -- not the population round. Only the first
    logging.options.prompt_examples.count interactions keep a file, so most
    calls have no prompt on disk.

        prompts/round_001.md  ->  "# Round 1 - agent agent_011"
        api_call_status       ->  {"round_index": 1, "agent_id": "agent_011"}
    """
    out: dict[tuple[int, str], str] = {}
    directory = episode / "prompts"
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("round_*.md")):
        text = path.read_text(encoding="utf-8")
        match = re.search(r"^#\s*Round\s+(\d+)\s*[-—]\s*agent\s+(\S+)", text, re.M)
        if match:
            out[(int(match.group(1)), match.group(2))] = text
    return out



def _rounds(episode: pathlib.Path) -> dict[int, dict[str, Any]]:
    """Map population_round -> that round's recorded quantities."""

    out: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(episode / "round_trajectory.jsonl"):
        event = row.get("event") or row
        absolute = event.get("absolute_round")
        if absolute is not None:
            out[int(absolute)] = event
    return out


def _board_by_round(
    episode: pathlib.Path, population: int
) -> dict[int, list[dict[str, Any]]]:
    """Rebuild each round's blackboard from the messages agents posted.

    Only agent posts can be recovered this way: a controller's own posts are
    not decisions, so they are absent here. Live traces do not have this gap --
    they read the board object itself. Kept because it is the only way to see
    the board for episodes that finished before tracing existed.
    """

    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(episode / "trajectory.jsonl"):
        index = row.get("interaction_index")
        if index is None:
            continue
        game_round = (int(index) - 1) // population + 1
        for decision in row.get("decisions") or []:
            action = decision.get("action") or decision
            meta = action.get("metadata") or {}
            message = meta.get("public_message")
            if not isinstance(message, Mapping):
                continue
            out[game_round].append(
                {
                    "author_id": action.get("agent_id"),
                    "author_kind": "agent",
                    "message_type": message.get("type"),
                    "text": message.get("text"),
                    "vote": action.get("value"),
                    "shared_fact_id": message.get("shared_fact_id"),
                    "reply_to": message.get("reply_to"),
                }
            )
    return out


def _episode_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    """Finished episodes only.

    An episode writes trajectory.jsonl when it completes, so its absence marks
    one that is still running or was interrupted. Importing those would put a
    stub trace next to the full one the re-run later produces.
    """

    return sorted(
        {
            path.parent
            for path in root.rglob("api_call_status.jsonl")
            if (path.parent / "trajectory.jsonl").is_file()
        }
    )


def _cell_of(episode: pathlib.Path) -> str:
    for parent in episode.parents:
        if parent.name.startswith("cell-"):
            return parent.name
    return "unknown"


def _overrides_of(episode: pathlib.Path) -> dict[str, Any]:
    for parent in episode.parents:
        candidate = parent / "overrides.json"
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8")).get(
                    "overrides", {}
                )
            except (OSError, json.JSONDecodeError):
                return {}
    return {}


def import_episode(
    tracer: otel_trace.Tracer, episode: pathlib.Path, arm: str
) -> tuple[int, int]:
    usage = {_call_key(r): r for r in _read_jsonl(episode / "usage_cost.jsonl")}
    decisions = _decisions_by_interaction(episode)
    prompts = _prompts_by_interaction(episode)
    agents = {str(c.get("agent_id")) for c in _read_jsonl(episode / "api_call_status.jsonl")}
    population = max(len(agents), 1)
    overrides = _overrides_of(episode)
    cell = _cell_of(episode)
    run_id = episode.name
    calls = list(_read_jsonl(episode / "api_call_status.jsonl"))
    if not calls:
        return 0, 0
    round_events = _rounds(episode)
    boards = _board_by_round(episode, population)

    # Group by round first so each round becomes one span with the whole
    # blackboard on it, and that round's agent calls hang underneath.
    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        index = int(call.get("round_index") or 0)
        by_round[((index - 1) // population) + 1 if index else 0].append(call)

    spans = errors = 0
    with tracer.start_as_current_span(f"episode {run_id}") as episode_span:
        episode_span.set_attribute("mas_cc.run_id", run_id)
        episode_span.set_attribute("mas_cc.cell_id", cell)
        episode_span.set_attribute("mas_cc.arm", arm)
        for key, value in overrides.items():
            episode_span.set_attribute(f"mas_cc.{key}", str(value))

        for game_round in sorted(by_round):
            with tracer.start_as_current_span(f"round {game_round}") as round_span:
                round_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
                round_span.set_attribute("mas_cc.run_id", run_id)
                round_span.set_attribute("mas_cc.cell_id", cell)
                round_span.set_attribute("mas_cc.arm", arm)
                for key in ROUND_ATTRIBUTES:
                    value = (round_events.get(game_round) or {}).get(key)
                    if value is not None:
                        round_span.set_attribute(
                            f"mas_cc.{key}",
                            value
                            if isinstance(value, (str, bool, int, float))
                            else json.dumps(value, default=str)[:8000],
                        )
                rendered = render_board(boards.get(game_round) or ())
                if rendered:
                    round_span.set_attribute(SpanAttributes.OUTPUT_VALUE, rendered[:60000])

                for call in by_round[game_round]:
                    agent = str(call.get("agent_id"))
                    interaction = str(call.get("interaction_id"))
                    # api_call_status.round_index is the interaction index over the
                    # whole episode (1..population_size*rounds), not the population
                    # round; each block of population_size interactions is one round.
                    interaction_index = int(call.get("round_index") or 0)
                    round_index = (
                        ((interaction_index - 1) // population) + 1 if interaction_index else 0
                    )
                    decision = decisions.get((interaction, agent), {})
                    use = (usage.get(_call_key(call)) or {}).get("usage") or {}
                    valid = bool(call.get("valid"))
                    if not valid:
                        errors += 1

                    stage = str(call.get("decision_stage"))
                    name = f"{actor(agent, stage)} r{round_index} {stage}"
                    with tracer.start_as_current_span(name) as span:
                        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
                        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, str(call.get("model")))
                        span.set_attribute(SpanAttributes.LLM_PROVIDER, str(call.get("provider")))
                        prompt = prompts.get((interaction_index, agent))
                        if prompt:
                            span.set_attribute(SpanAttributes.INPUT_VALUE, prompt[:60000])
                        # The decision is this call's output: the vote plus the private
                        # reason the agent recorded for it.
                        output = {k: v for k, v in decision.items() if v is not None}
                        if output:
                            span.set_attribute(
                                SpanAttributes.OUTPUT_VALUE, json.dumps(output)[:60000]
                            )
                        if decision.get("reason"):
                            span.set_attribute("mas_cc.private_reason", decision["reason"])
                        if decision.get("vote") is not None:
                            span.set_attribute("mas_cc.vote", str(decision["vote"]))
                        if decision.get("shared_fact_id"):
                            span.set_attribute(
                                "mas_cc.shared_fact_id", str(decision["shared_fact_id"])
                            )
                        for attr, field in (
                            (SpanAttributes.LLM_TOKEN_COUNT_PROMPT, "input_tokens"),
                            (SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, "output_tokens"),
                            (SpanAttributes.LLM_TOKEN_COUNT_TOTAL, "total_tokens"),
                        ):
                            if use.get(field) is not None:
                                span.set_attribute(attr, int(use[field]))
                        span.set_attribute("mas_cc.agent_id", actor(agent, stage))
                        span.set_attribute("mas_cc.round", int(round_index))
                        span.set_attribute("mas_cc.interaction_index", interaction_index)
                        span.set_attribute("mas_cc.attempt", int(call.get("attempt") or 1))
                        span.set_attribute("mas_cc.cell_id", cell)
                        span.set_attribute("mas_cc.arm", arm)
                        span.set_attribute("mas_cc.valid", valid)
                        if call.get("validation_error"):
                            span.set_attribute(
                                "mas_cc.validation_error", str(call["validation_error"])
                            )
                        if call.get("provider_error"):
                            span.set_attribute(
                                "mas_cc.provider_error", str(call["provider_error"])
                            )
                        if not valid:
                            span.set_status(
                                otel_trace.Status(
                                    otel_trace.StatusCode.ERROR,
                                    str(call.get("validation_error") or call.get("provider_error")),
                                )
                            )
                        spans += 1
    return spans, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=pathlib.Path)
    parser.add_argument("--endpoint", default="http://localhost:6006/v1/traces")
    parser.add_argument("--project", default="mas-cc")
    args = parser.parse_args()

    provider = TracerProvider(
        resource=Resource.create({"openinference.project.name": args.project})
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=args.endpoint))
    )
    tracer = provider.get_tracer("mas_cc.phoenix_import")

    total_spans = total_errors = total_episodes = 0
    for root in args.roots:
        if not root.exists():
            print(f"  skip (missing): {root}", file=sys.stderr)
            continue
        arm = "controlled" if "controlled" in str(root) else "no_control"
        episodes = _episode_dirs(root)
        print(f"{root}: {len(episodes)} episode(s), arm={arm}")
        for episode in episodes:
            spans, errors = import_episode(tracer, episode, arm)
            if spans:
                total_episodes += 1
                total_spans += spans
                total_errors += errors
    provider.force_flush()
    provider.shutdown()
    print(
        f"\nimported {total_episodes} episodes, {total_spans} LLM spans "
        f"({total_errors} failed calls) -> {args.endpoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
