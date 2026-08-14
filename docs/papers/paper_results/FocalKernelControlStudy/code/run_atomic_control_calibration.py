#!/usr/bin/env python3
"""Run one resumable, deterministically sharded worker over frozen prompts."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path[:0] = [str(SCRIPT_DIR), str(REPO_ROOT / "src")]

from atomic_control_common import (  # noqa: E402
    BUCKETS,
    atomic_write_json,
    read_jsonl,
    stable_shard,
    verify_frozen_dataset,
    write_jsonl,
)
from mas_cc.config.loader import load_component_config  # noqa: E402
from mas_cc.llm_runtime.config import LLMProviderConfig  # noqa: E402
from mas_cc.llm_runtime.messages import Message, MessageRole  # noqa: E402
from mas_cc.llm_runtime.providers import CompletionRequest, create_llm_provider  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_vote(raw: str, options: list[str]) -> tuple[str | None, str | None]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"response is not exact JSON: {exc.msg}"
    if not isinstance(value, dict) or set(value) != {"vote"}:
        return None, "response must be an object containing only the vote key"
    vote = value["vote"]
    if not isinstance(vote, str) or vote not in options:
        return None, f"vote must exactly match one of {options!r}"
    return vote, None


def load_provider_config(args: argparse.Namespace) -> LLMProviderConfig:
    path = args.provider_config
    if path is None:
        candidate = REPO_ROOT / "configs/components/llm_providers" / f"{args.provider}.yaml"
        if not candidate.is_file():
            raise ValueError(
                f"no default config for provider {args.provider!r}; pass --provider-config"
            )
        path = candidate
    config = load_component_config(path, "llm_provider")
    if not isinstance(config, LLMProviderConfig):
        raise TypeError(f"{path} did not resolve to an LLMProviderConfig")
    if config.type != args.provider:
        raise ValueError(
            f"provider config type {config.type!r} does not match --provider {args.provider!r}"
        )
    return replace(
        config,
        model=args.model,
        request_concurrency=args.concurrency,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )


def load_items(input_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        manifest_path = input_dir / bucket / "manifest.jsonl"
        if not manifest_path.is_file():
            raise ValueError(f"missing frozen bucket manifest: {manifest_path}")
        for row in read_jsonl(manifest_path):
            if row.get("bucket") != bucket:
                raise ValueError(f"bucket mismatch in {manifest_path}")
            item = dict(row)
            item["prompt_file"] = input_dir / bucket / str(row["prompt_path"])
            items.append(item)
    if len(items) != 600:
        raise ValueError(f"expected 600 frozen prompt items, found {len(items)}")
    return sorted(items, key=lambda row: (row["bucket"], row["state_id"]))


def result_path(run_dir: Path, item: dict[str, Any]) -> Path:
    return run_dir / "completed" / item["bucket"] / f"{item['state_id']}.json"


def failure_path(run_dir: Path, item: dict[str, Any]) -> Path:
    return run_dir / "failures" / item["bucket"] / f"{item['state_id']}.json"


def acquire_item_lock(run_dir: Path, item: dict[str, Any]) -> Path | None:
    lock = run_dir / "inflight" / f"{item['bucket']}__{item['state_id']}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()} started_at={utc_now()}\n")
    return lock


def mock_response_factory(mode: str):
    def respond(request: CompletionRequest) -> str:
        metadata = request.metadata
        if mode == "control_target":
            vote = metadata["control_target"]
        elif mode == "current_vote":
            vote = metadata["current_vote"]
        elif mode == "truth":
            vote = metadata["correct_answer"]
        else:
            vote = metadata["options"][0]
        return json.dumps({"vote": vote}, ensure_ascii=False)

    return respond


async def run_item(
    item: dict[str, Any],
    *,
    provider: Any,
    semaphore: asyncio.Semaphore,
    run_dir: Path,
    provider_name: str,
    model: str,
    dataset_hash: str,
    temperature: float,
    max_output_tokens: int,
    invalid_response_retries: int,
    rerun: bool,
) -> str:
    destination = result_path(run_dir, item)
    if destination.is_file() and not rerun:
        return "skipped"
    lock = acquire_item_lock(run_dir, item)
    if lock is None:
        return "locked"
    raw_attempts: list[str] = []
    validation_errors: list[str] = []
    try:
        if destination.is_file() and not rerun:
            return "skipped"
        prompt = Path(item["prompt_file"]).read_text(encoding="utf-8")
        request = CompletionRequest(
            messages=(Message(MessageRole.USER, prompt),),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=int.from_bytes(
                f"{item['bucket']}:{item['state_id']}".encode("utf-8")[:8].ljust(8, b"\0"),
                "big",
            ),
            metadata={
                "bucket": item["bucket"],
                "state_id": item["state_id"],
                "options": item["options"],
                "current_vote": item["current_vote"],
                "control_target": item["control_target"],
                "correct_answer": item["correct_answer"],
            },
        )
        last_response = None
        for attempt in range(1, invalid_response_retries + 2):
            try:
                async with semaphore:
                    last_response = await provider.complete(request)
            except Exception as exc:  # Provider errors are recorded, not format-retried.
                failure = {
                    "state_id": item["state_id"],
                    "task_id": item["task_id"],
                    "bucket": item["bucket"],
                    "provider": provider_name,
                    "model": model,
                    "dataset_hash": dataset_hash,
                    "valid_response": False,
                    "attempts": attempt,
                    "failure_type": "provider_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_attempts": raw_attempts,
                    "recorded_at": utc_now(),
                }
                atomic_write_json(failure_path(run_dir, item), failure)
                return "failed"
            raw_attempts.append(last_response.content)
            vote, error = parse_vote(last_response.content, item["options"])
            if error is not None:
                validation_errors.append(error)
                continue
            result = {
                "state_id": item["state_id"],
                "task_id": item["task_id"],
                "bucket": item["bucket"],
                "provider": provider_name,
                "model": model,
                "dataset_hash": dataset_hash,
                "current_vote": item["current_vote"],
                "control_target": item["control_target"],
                "control_alignment": item["control_alignment"],
                "correct_answer": item["correct_answer"],
                "vote_after": vote,
                "valid_response": True,
                "attempts": attempt,
                "raw_response": last_response.content,
                "invalid_attempts": raw_attempts[:-1],
                "validation_errors": validation_errors,
                "provider_response": last_response.to_dict(),
                "recorded_at": utc_now(),
            }
            atomic_write_json(destination, result)
            stale_failure = failure_path(run_dir, item)
            if stale_failure.exists():
                stale_failure.unlink()
            return "completed"

        failure = {
            "state_id": item["state_id"],
            "task_id": item["task_id"],
            "bucket": item["bucket"],
            "provider": provider_name,
            "model": model,
            "dataset_hash": dataset_hash,
            "current_vote": item["current_vote"],
            "control_target": item["control_target"],
            "control_alignment": item["control_alignment"],
            "correct_answer": item["correct_answer"],
            "valid_response": False,
            "attempts": len(raw_attempts),
            "failure_type": "invalid_response",
            "raw_response": raw_attempts[-1] if raw_attempts else "",
            "raw_attempts": raw_attempts,
            "validation_errors": validation_errors,
            "recorded_at": utc_now(),
        }
        atomic_write_json(failure_path(run_dir, item), failure)
        return "failed"
    finally:
        lock.unlink(missing_ok=True)


def collect_records(run_dir: Path, category: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / category).glob("bucket_*/state_*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def write_progress(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    dataset_hash: str,
    started_at: str,
    total: int,
    processed: int,
    stored_completed: int,
    stored_failed: int,
    skipped: int,
    locked: int,
) -> None:
    atomic_write_json(
        run_dir / "PROGRESS.json",
        {
            "provider": args.provider,
            "model": args.model,
            "dataset_hash": dataset_hash,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "started_at": started_at,
            "updated_at": utc_now(),
            "total_prompts": total,
            "processed_this_invocation": processed,
            "remaining_this_invocation": max(0, total - processed),
            "stored_completed_prompts": stored_completed,
            "stored_failed_prompts": stored_failed,
            "skipped_completed_prompts": skipped,
            "locked_prompts": locked,
            "fraction_processed": processed / total if total else 1.0,
        },
    )


def update_manifests(
    output_dir: Path,
    run_dir: Path,
    *,
    args: argparse.Namespace,
    dataset: dict[str, Any],
    started_at: str,
    selected_count: int,
) -> None:
    completed = collect_records(run_dir, "completed")
    failures = collect_records(run_dir, "failures")
    write_jsonl(run_dir / "results.jsonl", completed)
    write_jsonl(run_dir / "failures.jsonl", failures)
    shard_manifest = {
        "provider": args.provider,
        "model": args.model,
        "dataset_hash": dataset["dataset_hash"],
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "invalid_response_retries": args.invalid_response_retries,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_prompts": selected_count,
        "started_at": started_at,
        "updated_at": utc_now(),
        "completed_prompts": len(completed),
        "failed_prompts": len(failures),
    }
    atomic_write_json(run_dir / "RUN_MANIFEST.json", shard_manifest)
    shard_manifests = []
    if args.num_shards == 1:
        shard_manifests = [shard_manifest]
    else:
        for path in sorted(output_dir.glob("shard_*/RUN_MANIFEST.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("dataset_hash") == dataset["dataset_hash"] and value.get("model") == args.model:
                shard_manifests.append(value)
    aggregate = {
        "provider": args.provider,
        "model": args.model,
        "dataset_hash": dataset["dataset_hash"],
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "started_at": min((row["started_at"] for row in shard_manifests), default=started_at),
        "updated_at": utc_now(),
        "num_shards": args.num_shards,
        "reported_shards": sorted(row["shard_index"] for row in shard_manifests),
        "completed_prompts": sum(row["completed_prompts"] for row in shard_manifests),
        "failed_prompts": sum(row["failed_prompts"] for row in shard_manifests),
    }
    atomic_write_json(output_dir / "RUN_MANIFEST.json", aggregate)


async def run(args: argparse.Namespace) -> dict[str, int]:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    if args.concurrency < 1 or args.invalid_response_retries < 0:
        raise ValueError("concurrency must be positive and retries cannot be negative")
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    dataset = verify_frozen_dataset(input_dir)
    items = [
        item
        for item in load_items(input_dir)
        if stable_shard(item["bucket"], item["state_id"], args.num_shards) == args.shard_index
    ]
    run_dir = output_dir if args.num_shards == 1 else output_dir / f"shard_{args.shard_index:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = run_dir / "RUN_MANIFEST.json"
    if existing_manifest_path.is_file():
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        identity = (existing.get("provider"), existing.get("model"), existing.get("dataset_hash"))
        requested = (args.provider, args.model, dataset["dataset_hash"])
        if identity != requested:
            raise ValueError(f"output directory belongs to {identity}, not {requested}")
        settings = (existing.get("temperature"), existing.get("max_output_tokens"))
        requested_settings = (args.temperature, args.max_output_tokens)
        if settings != requested_settings:
            raise ValueError(
                f"output directory used settings {settings}, not requested {requested_settings}"
            )
        started_at = existing["started_at"]
    else:
        started_at = utc_now()
    config = load_provider_config(args)
    adapter_options: dict[str, Any] = {}
    if args.provider == "mock":
        adapter_options["response_factory"] = mock_response_factory(args.mock_response_mode)
    provider = create_llm_provider(config, **adapter_options)
    semaphore = asyncio.Semaphore(args.concurrency)
    completed_keys = {
        (row["bucket"], row["state_id"]) for row in collect_records(run_dir, "completed")
    }
    failed_keys = {
        (row["bucket"], row["state_id"]) for row in collect_records(run_dir, "failures")
    }
    if args.rerun:
        pending_items = items
        outcomes: list[str] = []
        initially_processed = 0
    else:
        pending_items = [
            item for item in items if (item["bucket"], item["state_id"]) not in completed_keys
        ]
        initially_processed = len(items) - len(pending_items)
        outcomes = ["skipped"] * initially_processed
    processed = initially_processed
    write_progress(
        run_dir,
        args=args,
        dataset_hash=dataset["dataset_hash"],
        started_at=started_at,
        total=len(items),
        processed=processed,
        stored_completed=len(completed_keys),
        stored_failed=len(failed_keys),
        skipped=outcomes.count("skipped"),
        locked=0,
    )
    disable_progress = args.progress == "never" or (
        args.progress == "auto" and not sys.stderr.isatty()
    )
    description = f"{args.provider}:{args.model}"
    if args.num_shards > 1:
        description += f" [{args.shard_index}/{args.num_shards}]"
    bar = tqdm(
        total=len(items),
        initial=initially_processed,
        desc=description,
        unit="prompt",
        dynamic_ncols=True,
        position=args.progress_position,
        disable=disable_progress,
    )
    try:
        async def tagged_run(item: dict[str, Any]) -> tuple[dict[str, Any], str]:
            outcome = await run_item(
                item,
                provider=provider,
                semaphore=semaphore,
                run_dir=run_dir,
                provider_name=args.provider,
                model=args.model,
                dataset_hash=dataset["dataset_hash"],
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                invalid_response_retries=args.invalid_response_retries,
                rerun=args.rerun,
            )
            return item, outcome

        tasks = {
            asyncio.create_task(tagged_run(item))
            for item in pending_items
        }
        for task in asyncio.as_completed(tasks):
            item, outcome = await task
            outcomes.append(outcome)
            processed += 1
            key = (item["bucket"], item["state_id"])
            if outcome == "completed":
                completed_keys.add(key)
                failed_keys.discard(key)
            elif outcome == "failed":
                failed_keys.add(key)
            bar.update(1)
            bar.set_postfix(
                completed=len(completed_keys),
                failed=len(failed_keys),
                skipped=outcomes.count("skipped"),
                refresh=False,
            )
            write_progress(
                run_dir,
                args=args,
                dataset_hash=dataset["dataset_hash"],
                started_at=started_at,
                total=len(items),
                processed=processed,
                stored_completed=len(completed_keys),
                stored_failed=len(failed_keys),
                skipped=outcomes.count("skipped"),
                locked=outcomes.count("locked"),
            )
    finally:
        bar.close()
        provider.close()
    update_manifests(
        output_dir,
        run_dir,
        args=args,
        dataset=dataset,
        started_at=started_at,
        selected_count=len(items),
    )
    return {key: outcomes.count(key) for key in ("completed", "skipped", "locked", "failed")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider-config", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--invalid-response-retries", type=int, default=2)
    parser.add_argument("--rerun", action="store_true", help="explicitly replace completed tuples")
    parser.add_argument(
        "--mock-response-mode",
        choices=("control_target", "current_vote", "truth", "first_option"),
        default="control_target",
        help="deterministic behavior used only by the repository mock provider",
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="show one tqdm bar for this model (auto shows it on an interactive terminal)",
    )
    parser.add_argument(
        "--progress-position",
        type=int,
        default=0,
        help="tqdm row to use when several model processes share one terminal",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outcomes = asyncio.run(run(args))
    print(json.dumps(outcomes, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
