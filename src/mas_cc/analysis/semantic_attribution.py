"""Semantic attribution of posted board messages (optional, System One backed).

The numeric estimators measure *how much* a controller's messages move the
population. This instrument asks *what the agents' own messages do*: whether a
posted message leans on the controller's recommendation, which allocation it
argues for, and how hard it pushes toward the controller's target. It reads the
``dashboard_semantic.jsonl`` logs the runs already saved, builds one bounded
state block per posted message, and asks TypeSafe System One four atomic typed
questions per message through :mod:`mas_cc.analysis.systemone`.

It is deliberately not part of the finalizer. It never runs unless invoked, the
package it writes is labelled with ``provider_calls`` and the model that answered,
and every answer is cached by content hash so a re-run costs nothing and
reproduces the same table.

    python -m mas_cc.analysis.semantic_attribution --study-root <dir> \
        --output <dir> --sample 300 --seed 1 [--dry-run]

``--dry-run`` writes the state blocks and questions without calling the model,
which is how the prompt design is reviewed before any tokens are spent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from . import systemone

VERSION = "semantic_attribution_v1"
QUESTION_VERSION = "q1"
PRESSURE_LEVELS = (
    "no push: the message does not mention or favour the controller's target",
    "weak: the message mentions the target or its evidence neutrally, without recommending it",
    "moderate: the message gives one reason that favours the target",
    "strong: the message explicitly urges the team to adopt the target or dismisses the alternatives",
)


def iter_posted_messages(study_root: Path) -> Iterator[dict[str, Any]]:
    """Every agent-posted message in every episode under ``study_root``."""
    for path in sorted(study_root.rglob("dashboard_semantic.jsonl")):
        header: dict[str, Any] = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("record_type") == "header":
                    header = row
                    continue
                if row.get("record_type") != "update":
                    continue
                message = row.get("new_message")
                if not message or message.get("author_kind") != "agent" or not message.get("text"):
                    continue
                yield {
                    "source_path": str(path),
                    "cell_id": row.get("cell_id") or header.get("cell_id"),
                    "episode_id": row.get("episode_id") or header.get("episode_id"),
                    "round_index": row.get("round_index"),
                    "within_round_index": row.get("within_round_index"),
                    "focal_agent_id": row.get("focal_agent_id"),
                    "message_id": message.get("message_id"),
                    "message_type": message.get("message_type"),
                    "reply_to": message.get("reply_to"),
                    "text": message.get("text"),
                    "vote_before": row.get("focal_vote_before"),
                    "vote_after": row.get("focal_vote_after"),
                    "controller_target": row.get("round_controller_target") or row.get("controller_target"),
                    "correct_answer": row.get("correct_answer"),
                    "possible_answers": list(row.get("possible_answers") or []),
                    "controller_message_exposed": bool(row.get("controller_message_directly_exposed"))
                    or int(row.get("eligible_controller_message_count") or 0) > 0,
                    "intervention_budget": row.get("intervention_budget"),
                    "controller_action": row.get("round_controller_action"),
                    "sampled_message_types": list(row.get("sampled_message_types") or []),
                }


def messages_from_table(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rebuild message dicts from a semantic_attribution table (one row per message, any question)."""
    columns = ["source_path", "cell_id", "episode_id", "round_index", "within_round_index", "focal_agent_id",
               "message_id", "message_type", "reply_to", "text", "vote_before", "vote_after", "controller_target",
               "correct_answer", "controller_message_exposed", "intervention_budget", "controller_action"]
    seen: set[tuple] = set()
    messages: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        key = (row.get("episode_id"), row.get("message_id"))
        if key in seen:
            continue
        seen.add(key)
        message = {c: (None if pd.isna(row.get(c)) else row.get(c)) if not isinstance(row.get(c), (list, dict)) else row.get(c) for c in columns}
        message["possible_answers"] = list(json.loads(row["possible_answers_json"])) if row.get("possible_answers_json") else []
        message["controller_message_exposed"] = bool(message["controller_message_exposed"])
        message["sampled_message_types"] = []
        messages.append(message)
    return messages


def message_state(message: Mapping[str, Any]) -> dict[str, Any]:
    """The bounded state System One sees: the message and only the context needed."""
    return {
        "game": ("A team of agents votes each round on one allocation from `possible_answers`. "
                 "A controller may post recommendations on a shared board; agents see a sample of board messages. "
                 "One agent has just posted `message`."),
        "possible_answers": message["possible_answers"],
        "controller_target": message["controller_target"],
        "controller_message_visible_to_this_agent": message["controller_message_exposed"],
        "agent_vote_before_posting": message["vote_before"],
        "agent_vote_after_posting": message["vote_after"],
        "message": {"type": message["message_type"], "text": message["text"]},
    }


def message_questions(message: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    options = {answer: f"the message argues for {answer}" for answer in message["possible_answers"]}
    options["none"] = "the message asks for evidence, reports a fact, or otherwise does not argue for any allocation"
    return {
        "cites_controller": systemone.noul(
            "Does `message.text` refer to, quote, or rely on the controller's recommendation or the controller's target?",
            true="the message mentions the controller, its recommendation, or names `controller_target` as the recommended choice",
            false="the message does not reference the controller or its recommendation",
        ),
        "stance": systemone.choice("Which allocation does `message.text` argue for?", options),
        "pressure_toward_target": systemone.score(
            "How strongly does `message.text` push the team toward `controller_target`?", PRESSURE_LEVELS),
        "provides_evidence": systemone.noul(
            "Does `message.text` state a concrete fact or comparison about the candidates (as opposed to only asking or expressing a preference)?",
            true="the message asserts at least one specific fact or comparison", false="the message only asks a question or states a preference",
        ),
    }


def sample_messages(messages: Iterable[Mapping[str, Any]], sample: int | None, seed: int) -> list[dict[str, Any]]:
    """Deterministic sample, stratified by cell so every cell keeps its share."""
    rows = [dict(m) for m in messages]
    if sample is None or sample >= len(rows):
        return rows
    rng = random.Random(seed)
    by_cell: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault(row["cell_id"], []).append(row)
    picked: list[dict[str, Any]] = []
    quota = max(1, sample // max(1, len(by_cell)))
    for cell_id in sorted(by_cell, key=str):
        group = by_cell[cell_id]
        picked.extend(rng.sample(group, min(quota, len(group))))
    return picked[:sample]


def attribute(messages: list[dict[str, Any]], client: systemone.SystemOneClient | None) -> pd.DataFrame:
    """One row per (message, question); ``client=None`` records the request only."""
    rows = []
    for message in messages:
        state = message_state(message)
        questions = message_questions(message)
        rid = systemone.request_id(client.model if client else "dry-run", state, questions)
        base = {k: v for k, v in message.items() if k not in ("possible_answers", "sampled_message_types")}
        base.update(possible_answers_json=json.dumps(message["possible_answers"]), request_id=rid,
                    question_version=QUESTION_VERSION, estimator_version=VERSION)
        if client is None:
            for name in questions:
                rows.append({**base, "question": name, "answer_type": questions[name]["type"], "value": None,
                             "label": None, "confidence": None, "probabilities_json": None, "model": None, "cached": None})
            continue
        record = client.ask(state, questions)
        for name, answer in record["answers"].items():
            rows.append({**base, **systemone.flatten_answer(name, answer), "model": record["model"], "cached": record["cached"]})
    frame = pd.DataFrame(rows)
    if not frame.empty and "stance" in set(frame["question"]):
        # Identical messages in identical context share a request_id (and a cached
        # answer); one label per request_id is all the mapping needs.
        stance = frame[frame["question"] == "stance"].drop_duplicates("request_id").set_index("request_id")["label"]
        frame["stance_matches_target"] = frame["request_id"].map(stance) == frame["controller_target"]
        frame["stance_matches_truth"] = frame["request_id"].map(stance) == frame["correct_answer"]
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Per cell: mean noul/score values and stance shares, with message counts."""
    if frame.empty:
        return pd.DataFrame()
    numeric = frame[frame["answer_type"].isin(["noul", "score"])]
    means = numeric.groupby(["cell_id", "intervention_budget", "question"])["value"].agg(["mean", "count"]).reset_index()
    means = means.rename(columns={"mean": "estimate", "count": "n_messages"})
    means["statistic"] = "mean"
    stance = frame[frame["question"] == "stance"]
    shares = []
    for (cell, budget), group in stance.groupby(["cell_id", "intervention_budget"]):
        n = len(group)
        shares.append({"cell_id": cell, "intervention_budget": budget, "question": "stance_matches_target",
                       "estimate": float(group["stance_matches_target"].mean()), "n_messages": n, "statistic": "share"})
        shares.append({"cell_id": cell, "intervention_budget": budget, "question": "stance_matches_truth",
                       "estimate": float(group["stance_matches_truth"].mean()), "n_messages": n, "statistic": "share"})
        shares.append({"cell_id": cell, "intervention_budget": budget, "question": "stance_none",
                       "estimate": float((group["label"] == "none").mean()), "n_messages": n, "statistic": "share"})
    return pd.concat([means, pd.DataFrame(shares)], ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-root", type=Path, default=None, help="study runs directory to scan")
    parser.add_argument("--messages", type=Path, default=None,
                        help="instead of scanning, judge the messages recorded in a previous run's semantic_attribution.parquet (e.g. a --dry-run made on the cluster)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample", type=int, default=None, help="messages to judge (stratified by cell); default all")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cache-dir", type=Path, default=None, help="default: <output>/systemone_cache")
    parser.add_argument("--dry-run", action="store_true", help="write states and questions, call nothing")
    args = parser.parse_args(argv)
    if (args.study_root is None) == (args.messages is None):
        parser.error("give exactly one of --study-root or --messages")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.messages is not None:
        messages = sample_messages(messages_from_table(pd.read_parquet(args.messages)), args.sample, args.seed)
    else:
        messages = sample_messages(iter_posted_messages(args.study_root), args.sample, args.seed)
    client = None if args.dry_run else systemone.SystemOneClient(cache_dir=args.cache_dir or args.output / "systemone_cache")
    frame = attribute(messages, client)
    frame.to_parquet(args.output / "semantic_attribution.parquet", index=False)
    summary = summarize(frame) if client is not None else pd.DataFrame()
    summary.to_parquet(args.output / "semantic_attribution_summary.parquet", index=False)
    if args.dry_run:
        preview = [{"state": message_state(m), "questions": message_questions(m)} for m in messages[:3]]
        (args.output / "dry_run_preview.json").write_text(json.dumps(preview, indent=1, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "estimator_version": VERSION, "question_version": QUESTION_VERSION,
        "study_root": str(args.study_root) if args.study_root else None, "messages_source": str(args.messages) if args.messages else None,
        "messages_judged": len(messages), "sample": args.sample, "seed": args.seed,
        "dry_run": args.dry_run,
        "provider_calls": 0 if client is None else client.usage.requests,
        "usage": None if client is None else client.usage.as_dict(),
        "model": None if client is None else client.model,
        "endpoint": None if client is None else client.url,
        "questions_sha256": hashlib.sha256(json.dumps(message_questions(
            {"possible_answers": ["A", "B"], "controller_target": "A"}), sort_keys=True).encode()).hexdigest(),
    }
    (args.output / "semantic_attribution_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("messages_judged", "provider_calls", "usage", "model")}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
