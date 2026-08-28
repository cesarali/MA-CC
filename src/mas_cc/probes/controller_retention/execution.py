"""Bounded concurrent execution for the focused local probe."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mas_cc.games.relational_reasoning.data import RelationalTask, load_relational_task
from mas_cc.llm_runtime.providers import ProviderError
from mas_cc.llm_runtime.providers.registry import create_llm_provider
from mas_cc.llm_runtime.providers.requests import CompletionRequest

from . import vignette as vignette_module
from .config import ModelSpec, ProbeConfig
from .design import ARMS_BY_Q, PROBE_VERSION, Vignette

RAW_CALLS_FILENAME = "raw_calls.jsonl"


@dataclass(frozen=True, slots=True)
class CallSpec:
    call_id: str
    model: ModelSpec
    vignette: Vignette
    arm: str
    dataset_dir: str
    task_id: str

    @property
    def pair_id(self) -> str:
        return self.vignette.vignette_id


@dataclass(frozen=True, slots=True)
class CallResult:
    call_id: str
    model_label: str
    arm: str
    vignette_id: str
    response_text: str | None
    final_vote_semantic: str | None
    parse_ok: bool
    provider_error: str | None
    validation_error: str | None
    attempts: int
    latency: float
    input_tokens: int | None
    output_tokens: int | None
    ballot: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "model_label": self.model_label,
            "arm": self.arm,
            "vignette_id": self.vignette_id,
            "response_text": self.response_text,
            "final_vote_semantic": self.final_vote_semantic,
            "parse_ok": self.parse_ok,
            "provider_error": self.provider_error,
            "validation_error": self.validation_error,
            "attempts": self.attempts,
            "latency": self.latency,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "ballot": self.ballot,
        }


_TASK_CACHE: dict[tuple[str, str], RelationalTask] = {}
_LOCAL = threading.local()


def _worker_provider(model: ModelSpec) -> Any:
    cache = getattr(_LOCAL, "providers", None)
    if cache is None:
        cache = {}
        _LOCAL.providers = cache
    key = f"{model.provider}:{model.model}:{model.generation_settings_hash}"
    if key not in cache:
        cache[key] = create_llm_provider(model.provider_config())
    return cache[key]


def _worker_loop() -> asyncio.AbstractEventLoop:
    loop = getattr(_LOCAL, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _LOCAL.loop = loop
    return loop


def _worker_task(dataset_dir: str, task_id: str) -> RelationalTask:
    key = (dataset_dir, task_id)
    if key not in _TASK_CACHE:
        _TASK_CACHE[key] = load_relational_task(dataset_dir, task_id)
    return _TASK_CACHE[key]


def execute_call(spec: CallSpec, max_retries: int = 0) -> CallResult:
    """Make one logical call; transport retries belong to the provider adapter."""

    del max_retries
    task = _worker_task(spec.dataset_dir, spec.task_id)
    compiled = vignette_module.build_prompt(task, spec.vignette, spec.arm).compile()
    request = CompletionRequest(
        messages=tuple(compiled.messages),
        temperature=spec.model.temperature,
        max_output_tokens=spec.model.max_output_tokens,
        seed=spec.model.seed,
        metadata={"probe": PROBE_VERSION, "call_id": spec.call_id, "arm": spec.arm},
    )
    started = time.perf_counter()
    try:
        response = _worker_loop().run_until_complete(
            _worker_provider(spec.model).complete(request)
        )
    except ProviderError as exc:
        return _failure(spec, started, provider_error=f"{exc.code}: {exc}")
    except Exception as exc:  # noqa: BLE001 - one bad call must not crash the pool
        return _failure(spec, started, provider_error=f"{type(exc).__name__}: {exc}")

    try:
        compiled.response_contract.validate(response.content).raise_for_errors(
            context="controller-retention response contract"
        )
        vote, ballot = vignette_module.parse_vote(task, spec.vignette, response.content)
        if vote is None:
            raise ValueError("validated response did not resolve to a semantic vote")
    except (TypeError, ValueError) as exc:
        return CallResult(
            call_id=spec.call_id,
            model_label=spec.model.label,
            arm=spec.arm,
            vignette_id=spec.pair_id,
            response_text=response.content,
            final_vote_semantic=None,
            parse_ok=False,
            provider_error=None,
            validation_error=str(exc),
            attempts=1,
            latency=time.perf_counter() - started,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
    return CallResult(
        call_id=spec.call_id,
        model_label=spec.model.label,
        arm=spec.arm,
        vignette_id=spec.pair_id,
        response_text=response.content,
        final_vote_semantic=vote,
        parse_ok=True,
        provider_error=None,
        validation_error=None,
        attempts=1,
        latency=time.perf_counter() - started,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        ballot=ballot,
    )


def _failure(spec: CallSpec, started: float, *, provider_error: str) -> CallResult:
    return CallResult(
        call_id=spec.call_id,
        model_label=spec.model.label,
        arm=spec.arm,
        vignette_id=spec.pair_id,
        response_text=None,
        final_vote_semantic=None,
        parse_ok=False,
        provider_error=provider_error,
        validation_error=None,
        attempts=1,
        latency=time.perf_counter() - started,
        input_tokens=None,
        output_tokens=None,
    )


def _worker_entry(payload: tuple[CallSpec, int]) -> CallResult:
    return execute_call(*payload)


def build_call_specs(config: ProbeConfig, vignettes: tuple[Vignette, ...]) -> tuple[CallSpec, ...]:
    per_model: list[list[CallSpec]] = []
    for model in config.models:
        specs = [
            CallSpec(
                call_id=item.call_id(model.call_identity, arm),
                model=model,
                vignette=item,
                arm=arm,
                dataset_dir=config.dataset_dirs[item.reasoning_depth],
                task_id=item.task_id,
            )
            for item in vignettes
            for arm in ARMS_BY_Q[item.q]
        ]
        per_model.append(specs)
    interleaved: list[CallSpec] = []
    for index in range(max((len(group) for group in per_model), default=0)):
        for group in per_model:
            if index < len(group):
                interleaved.append(group[index])
    return tuple(interleaved)


def successful_call_ids(raw_path: Path) -> set[str]:
    latest = {str(row.get("call_id")): row for row in read_raw_calls(raw_path)}
    return {
        call_id
        for call_id, row in latest.items()
        if row.get("parse_ok") is True
        and not row.get("provider_error")
        and not row.get("validation_error")
    }


# Backwards-compatible name with corrected successful-only semantics.
completed_call_ids = successful_call_ids


def effective_workers(config: ProbeConfig) -> int:
    workers = config.execution.workers
    providers = {model.provider for model in config.models}
    caps = [
        config.execution.provider_concurrency_caps[name]
        for name in providers
        if name in config.execution.provider_concurrency_caps
    ]
    if caps:
        workers = min(workers, min(caps))
    if config.execution.backend == "serial":
        return 1
    return max(1, min(workers, (os.cpu_count() or 1) * 4))


@dataclass(slots=True)
class Progress:
    total: int
    completed: int = 0
    errors: int = 0
    parse_failures: int = 0
    started: float = field(default_factory=time.perf_counter)
    stream: Any = None
    every: int = 1

    def update(self, result: CallResult) -> None:
        self.completed += 1
        self.errors += int(result.provider_error is not None)
        self.parse_failures += int(result.validation_error is not None)
        if self.stream is None or self.completed % self.every:
            return
        elapsed = time.perf_counter() - self.started
        rate = self.completed / elapsed if elapsed else 0.0
        eta = (self.total - self.completed) / rate if rate else float("nan")
        self.stream.write(
            f"\r  {self.completed}/{self.total} calls | errors {self.errors} "
            f"| invalid {self.parse_failures} | ETA {eta / 60:.1f} min   "
        )
        self.stream.flush()

    def finish(self) -> None:
        if self.stream is not None:
            self.stream.write("\n")
            self.stream.flush()


def run_calls(
    config: ProbeConfig,
    specs: tuple[CallSpec, ...],
    raw_path: Path,
    *,
    stream: Any = sys.stderr,
) -> tuple[CallResult, ...]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    workers = effective_workers(config)
    progress = Progress(total=len(specs), stream=stream)
    results: list[CallResult] = []
    payloads = [(spec, 0) for spec in specs]
    with raw_path.open("a", encoding="utf-8") as handle:
        def record(result: CallResult) -> None:
            handle.write(json.dumps(result.to_row(), sort_keys=True) + "\n")
            handle.flush()
            results.append(result)
            progress.update(result)

        if config.execution.backend == "serial" or workers == 1:
            for payload in payloads:
                record(_worker_entry(payload))
        else:
            pool_type = ThreadPoolExecutor if config.execution.backend == "thread_pool" else ProcessPoolExecutor
            with pool_type(max_workers=workers) as pool:
                futures: dict[Future, CallSpec] = {
                    pool.submit(_worker_entry, payload): payload[0] for payload in payloads
                }
                for future in as_completed(futures):
                    record(future.result())
    progress.finish()
    return tuple(results)


def read_raw_calls(raw_path: Path) -> tuple[dict[str, Any], ...]:
    if not raw_path.is_file():
        return ()
    rows: dict[str, dict[str, Any]] = {}
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            call_id = row.get("call_id")
            if isinstance(call_id, str):
                rows[call_id] = row
    return tuple(rows[key] for key in sorted(rows))


__all__ = [
    "RAW_CALLS_FILENAME",
    "CallResult",
    "CallSpec",
    "Progress",
    "build_call_specs",
    "completed_call_ids",
    "effective_workers",
    "execute_call",
    "read_raw_calls",
    "run_calls",
    "successful_call_ids",
]
