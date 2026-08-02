#!/usr/bin/env python3
"""Run the canonical paper-style HiddenBench protocol through MAS-CC Gemma.

The command is preflight-only unless ``--execute`` is supplied.  Importing this
module and running preflight never constructs a provider, imports torch or
Transformers, checks CUDA, or loads a model.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mas_cc.config import LLMProviderConfig, PromptConfig, load_component_config
from mas_cc.core import Message, ValidationIssue, ValidationResult
from mas_cc.llm_providers import (
    CompletionRequest,
    LLMProvider,
    ProviderError,
    create_llm_provider,
)
from mas_cc.planning import LogicalCallSpec, static_preflight
from mas_cc.prompts import (
    PromptComposer,
    PromptContext,
    RegexTokenCounter,
    ResponseContract,
    create_default_prompt_registry,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "scripts/local_llms/hiddenbench_population_pipeline/data/hiddenbench/canonical/tasks.json"
)
DEFAULT_PROVIDER_CONFIG = (
    REPOSITORY_ROOT / "configs/components/llm_providers/gemma_local.yaml"
)
DEFAULT_DISCUSSION_PROMPT = (
    REPOSITORY_ROOT / "configs/components/prompts/hidden_profile_discussion_paper.yaml"
)
DEFAULT_VOTE_PROMPT = (
    REPOSITORY_ROOT / "configs/components/prompts/hidden_profile_vote_paper.yaml"
)


class HiddenBenchRunnerError(RuntimeError):
    """Safe configuration, dataset, or protocol failure."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_seed(*parts: Any) -> int:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(encoded).hexdigest()[:16], 16)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_canonical_tasks(
    path: str | Path, task_ids: Sequence[int] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HiddenBenchRunnerError(f"Cannot load canonical dataset {source}: {exc}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("tasks"), list):
        raise HiddenBenchRunnerError("Canonical input must be an object containing a tasks list.")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping) or metadata.get("kind") != "canonical":
        raise HiddenBenchRunnerError(
            "This runner accepts the canonical HiddenBench dataset only (metadata.kind=canonical)."
        )
    selected: list[dict[str, Any]] = []
    wanted = None if not task_ids else set(task_ids)
    for raw in payload["tasks"]:
        if not isinstance(raw, Mapping):
            raise HiddenBenchRunnerError("Every canonical task must be an object.")
        task = dict(raw)
        _validate_task(task)
        if wanted is None or int(task["task_id"]) in wanted:
            selected.append(task)
    if wanted is not None:
        missing = wanted - {int(task["task_id"]) for task in selected}
        if missing:
            raise HiddenBenchRunnerError(f"Unknown task IDs: {sorted(missing)}")
    if not selected:
        raise HiddenBenchRunnerError("No canonical tasks were selected.")
    return selected, {**dict(metadata), "path": str(source), "file_sha256": _sha256(source)}


def _validate_task(task: Mapping[str, Any]) -> None:
    required = {
        "task_id",
        "name",
        "scenario_description",
        "shared_information",
        "hidden_information",
        "possible_answers",
        "correct_answer",
    }
    missing = required - set(task)
    if missing:
        raise HiddenBenchRunnerError(
            f"Task {task.get('task_id', '<unknown>')} lacks {sorted(missing)}."
        )
    shared = task["shared_information"]
    hidden = task["hidden_information"]
    answers = task["possible_answers"]
    if not isinstance(shared, list) or not shared or not all(
        isinstance(item, str) and item.strip() for item in shared
    ):
        raise HiddenBenchRunnerError(f"Task {task['task_id']} has invalid shared information.")
    if not isinstance(hidden, list) or not hidden:
        raise HiddenBenchRunnerError(f"Task {task['task_id']} has no hidden information.")
    for item in hidden:
        if not isinstance(item, Mapping) or not isinstance(item.get("source_text"), str):
            raise HiddenBenchRunnerError(
                f"Task {task['task_id']} has invalid hidden information."
            )
    if not isinstance(answers, list) or len(answers) < 2 or not all(
        isinstance(item, str) and item.strip() for item in answers
    ):
        raise HiddenBenchRunnerError(f"Task {task['task_id']} has invalid answers.")
    if task["correct_answer"] not in answers:
        raise HiddenBenchRunnerError(
            f"Task {task['task_id']} correct answer is not an available answer."
        )


def canonical_agents(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One canonical agent per hidden evidence item, matching the source test."""

    return [
        {
            "agent_id": index,
            "evidence_type": int(item.get("evidence_type", index)),
            "private_information": [str(item["source_text"])],
        }
        for index, item in enumerate(task["hidden_information"])
    ]


def _shuffled_information(
    task: Mapping[str, Any], private_information: Sequence[str], seed: int
) -> list[str]:
    information = [
        *[str(item) for item in task["shared_information"]],
        *[str(item) for item in private_information],
    ]
    random.Random(seed).shuffle(information)
    return information


@dataclass(slots=True)
class HiddenBenchPromptCompiler:
    discussion_config: PromptConfig
    vote_config: PromptConfig
    _composer: PromptComposer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._composer = PromptComposer(
            create_default_prompt_registry(), RegexTokenCounter()
        )

    @classmethod
    def from_files(
        cls, discussion_path: str | Path, vote_path: str | Path
    ) -> "HiddenBenchPromptCompiler":
        discussion = load_component_config(
            Path(discussion_path).resolve(), "prompt", environment={}
        )
        vote = load_component_config(Path(vote_path).resolve(), "prompt", environment={})
        if not isinstance(discussion, PromptConfig) or not isinstance(vote, PromptConfig):
            raise HiddenBenchRunnerError("HiddenBench prompt files did not resolve correctly.")
        return cls(discussion, vote)

    def compile(
        self,
        task: Mapping[str, Any],
        private_information: Sequence[str],
        transcript: Sequence[Mapping[str, Any]],
        *,
        stage: str,
        task_seed: int,
        request_seed: int,
        session_index: int,
        agent_id: int,
        round_index: int | None = None,
        temperature: float,
        max_output_tokens: int,
    ) -> tuple[CompletionRequest, list[str], ResponseContract]:
        information = _shuffled_information(task, private_information, task_seed)
        is_vote = stage in {"pre_vote", "post_vote", "full_profile_vote"}
        config = self.vote_config if is_vote else self.discussion_config
        if is_vote:
            config = replace(
                config,
                response_contract={
                    "type": "json_vote",
                    "allowed_values": list(task["possible_answers"]),
                },
            )
        contract = ResponseContract.from_mapping(config.response_contract)
        context = PromptContext(
            task_description=str(task["scenario_description"]),
            game_rules=(
                "Use only your permitted shared/private evidence and the visible public transcript.",
            ),
            private_state={
                "information": information,
                "possible_answers": list(task["possible_answers"]),
            },
            recent_memory=tuple(
                {
                    "speaker_id": int(entry["speaker_id"]),
                    "message": str(entry["message"]),
                }
                for entry in transcript
            ),
            current_interaction={
                "phase": stage,
                "session_index": session_index,
                "agent_id": agent_id,
                "round_index": round_index,
            },
            decision_instruction=(
                "Return your vote using the required JSON contract."
                if is_vote
                else "Contribute one concise public message."
            ),
            metadata={"prompt_fixture": "hiddenbench_canonical_gemma_v1"},
        )
        prompt = self._composer.compose(config, context)
        request = CompletionRequest(
            messages=prompt.messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=request_seed,
            metadata={
                "benchmark": "HiddenBench",
                "protocol_version": 1,
                "task_id": int(task["task_id"]),
                "session_index": session_index,
                "agent_id": agent_id,
                "round_index": round_index,
                "stage": stage,
                "prompt_family": config.prompt_family,
                "prompt_version": config.prompt_version,
                "response_contract": contract.to_dict(),
            },
        )
        return request, information, contract


class AuditWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.existing_records = (
            sum(1 for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())
            if self.path.is_file()
            else 0
        )

    def write(self, record: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()


class HiddenBenchProtocolRunner:
    """Paper protocol orchestration; provider and prompt concerns remain external."""

    def __init__(
        self,
        provider: LLMProvider,
        compiler: HiddenBenchPromptCompiler,
        audit: AuditWriter,
        *,
        base_seed: int,
        temperature: float,
        discussion_max_tokens: int,
        vote_max_tokens: int,
        validation_retries: int,
    ) -> None:
        self.provider = provider
        self.compiler = compiler
        self.audit = audit
        self.base_seed = base_seed
        self.temperature = temperature
        self.discussion_max_tokens = discussion_max_tokens
        self.vote_max_tokens = vote_max_tokens
        self.validation_retries = validation_retries
        self._call_index = audit.existing_records

    async def _complete(self, request: CompletionRequest, contract: ResponseContract):
        current = request
        for attempt in range(self.validation_retries + 1):
            response = await self.provider.complete(current)
            self._call_index += 1
            audit_id = f"call-{self._call_index:08d}"
            validation = contract.validate(response.content)
            if contract.type == "free_text" and not response.content.strip():
                validation = ValidationResult.failure(
                    ValidationIssue("response", "discussion response must not be blank")
                )
            self.audit.write(
                {
                    "schema_version": 1,
                    "audit_id": audit_id,
                    "created_at": _now(),
                    "attempt": attempt,
                    "request": current.to_dict(),
                    "normalized_response": response.to_dict(),
                    "raw_response_redacted": response.redacted_raw_response(),
                    "validation": {
                        "valid": validation.is_valid,
                        "issues": [
                            {
                                "field": issue.field,
                                "message": issue.message,
                                "invalid_value": issue.invalid_value,
                            }
                            for issue in validation.issues
                        ],
                    },
                }
            )
            if validation.is_valid:
                return response, audit_id, attempt
            if attempt == self.validation_retries:
                raise HiddenBenchRunnerError(
                    f"Invalid {current.metadata.get('stage')} response after {attempt + 1} attempts."
                )
            current = CompletionRequest(
                messages=(
                    *request.messages,
                    Message("assistant", response.content or "<empty response>"),
                    Message(
                        "user",
                        "The preceding response was invalid. Return only a response satisfying: "
                        + contract.instruction(),
                    ),
                ),
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens,
                seed=stable_seed(request.seed, "validation_retry", attempt + 1),
                metadata={**dict(request.metadata), "validation_retry": attempt + 1},
            )
        raise AssertionError("unreachable")

    async def _vote(
        self,
        task: Mapping[str, Any],
        private_information: Sequence[str],
        transcript: Sequence[Mapping[str, Any]],
        *,
        session_index: int,
        agent_id: int,
        stage: str,
    ) -> dict[str, Any]:
        request, information, contract = self.compiler.compile(
            task,
            private_information,
            transcript,
            stage=stage,
            task_seed=stable_seed(
                self.base_seed, task["task_id"], session_index, stage, agent_id, "information"
            ),
            request_seed=stable_seed(
                self.base_seed, task["task_id"], session_index, stage, agent_id, "request"
            ),
            session_index=session_index,
            agent_id=agent_id,
            temperature=self.temperature,
            max_output_tokens=self.vote_max_tokens,
        )
        response, audit_id, retries = await self._complete(request, contract)
        parsed = json.loads(response.content)
        vote = parsed["vote"]
        return {
            "agent_id": agent_id,
            "vote": vote,
            "rationale": parsed["rationale"],
            "correct": vote == task["correct_answer"],
            "information": information,
            "audit_id": audit_id,
            "validation_retries": retries,
            "usage": response.usage.to_dict(),
        }

    async def _message(
        self,
        task: Mapping[str, Any],
        agent: Mapping[str, Any],
        transcript: Sequence[Mapping[str, Any]],
        *,
        session_index: int,
        round_index: int,
    ) -> dict[str, Any]:
        request, information, contract = self.compiler.compile(
            task,
            agent["private_information"],
            transcript,
            stage="discussion",
            task_seed=stable_seed(
                self.base_seed,
                task["task_id"],
                session_index,
                "discussion",
                round_index,
                agent["agent_id"],
                "information",
            ),
            request_seed=stable_seed(
                self.base_seed,
                task["task_id"],
                session_index,
                "discussion",
                round_index,
                agent["agent_id"],
                "request",
            ),
            session_index=session_index,
            agent_id=int(agent["agent_id"]),
            round_index=round_index,
            temperature=self.temperature,
            max_output_tokens=self.discussion_max_tokens,
        )
        response, audit_id, retries = await self._complete(request, contract)
        return {
            "round": round_index + 1,
            "speaker_id": int(agent["agent_id"]),
            "message": response.content.strip(),
            "information": information,
            "audit_id": audit_id,
            "validation_retries": retries,
            "usage": response.usage.to_dict(),
        }

    async def run_session(
        self,
        task: Mapping[str, Any],
        *,
        session_index: int,
        communication_rounds: int,
        speaker_order: str,
    ) -> dict[str, Any]:
        agents = canonical_agents(task)
        pre_votes = [
            await self._vote(
                task,
                agent["private_information"],
                (),
                session_index=session_index,
                agent_id=int(agent["agent_id"]),
                stage="pre_vote",
            )
            for agent in agents
        ]
        rng = random.Random(
            stable_seed(self.base_seed, task["task_id"], session_index, "speakers")
        )
        if speaker_order == "round_robin":
            offset = rng.randrange(len(agents))
            speakers = [
                (offset + index) % len(agents) for index in range(communication_rounds)
            ]
        elif speaker_order == "random":
            speakers = [rng.randrange(len(agents)) for _ in range(communication_rounds)]
        else:
            raise HiddenBenchRunnerError("speaker_order must be round_robin or random.")
        transcript: list[dict[str, Any]] = []
        for round_index, speaker_index in enumerate(speakers):
            transcript.append(
                await self._message(
                    task,
                    agents[speaker_index],
                    transcript,
                    session_index=session_index,
                    round_index=round_index,
                )
            )
        post_votes = [
            await self._vote(
                task,
                agent["private_information"],
                transcript,
                session_index=session_index,
                agent_id=int(agent["agent_id"]),
                stage="post_vote",
            )
            for agent in agents
        ]
        full_information = [
            str(item["source_text"]) for item in task["hidden_information"]
        ]
        full_votes = [
            await self._vote(
                task,
                full_information,
                (),
                session_index=session_index,
                agent_id=int(agent["agent_id"]),
                stage="full_profile_vote",
            )
            for agent in agents
        ]
        return {
            "session_index": session_index,
            "num_agents": len(agents),
            "communication_rounds": communication_rounds,
            "speaker_order": speaker_order,
            "pre_discussion_decisions": pre_votes,
            "discussion_history": transcript,
            "post_discussion_decisions": post_votes,
            "full_profile_decisions": full_votes,
            "metrics": {
                "pre": vote_metrics(pre_votes, str(task["correct_answer"])),
                "post": vote_metrics(post_votes, str(task["correct_answer"])),
                "full": vote_metrics(full_votes, str(task["correct_answer"])),
            },
        }


def vote_metrics(votes: Sequence[Mapping[str, Any]], correct_answer: str) -> dict[str, Any]:
    values = [item.get("vote") for item in votes]
    counts = Counter(values)
    majority = counts.most_common(1)[0][0] if counts else None
    return {
        "average_accuracy": (
            sum(value == correct_answer for value in values) / len(values) if values else 0.0
        ),
        "correct_count": sum(value == correct_answer for value in values),
        "num_votes": len(values),
        "majority_vote": majority,
        "majority_correct": majority == correct_answer,
        "unanimous": bool(values) and all(value == values[0] for value in values),
    }


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def task_summary(sessions: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        "average_pre_accuracy": _mean(
            [float(item["metrics"]["pre"]["average_accuracy"]) for item in sessions]
        ),
        "average_post_accuracy": _mean(
            [float(item["metrics"]["post"]["average_accuracy"]) for item in sessions]
        ),
        "average_full_profile_accuracy": _mean(
            [float(item["metrics"]["full"]["average_accuracy"]) for item in sessions]
        ),
    }


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "num_tasks": len(results),
        "average_pre_accuracy": _mean(
            [float(item["summary"]["average_pre_accuracy"]) for item in results]
        ),
        "average_post_accuracy": _mean(
            [float(item["summary"]["average_post_accuracy"]) for item in results]
        ),
        "average_full_profile_accuracy": _mean(
            [float(item["summary"]["average_full_profile_accuracy"]) for item in results]
        ),
    }


def planned_logical_calls(
    tasks: Sequence[Mapping[str, Any]], sessions: int, rounds: int
) -> tuple[int, list[dict[str, int]]]:
    per_task = []
    total = 0
    for task in tasks:
        agents = len(canonical_agents(task))
        calls = sessions * (3 * agents + rounds)
        per_task.append(
            {"task_id": int(task["task_id"]), "agents": agents, "logical_calls": calls}
        )
        total += calls
    return total, per_task


def build_preflight(
    tasks: Sequence[Mapping[str, Any]],
    compiler: HiddenBenchPromptCompiler,
    provider_config: LLMProviderConfig,
    *,
    sessions: int,
    rounds: int,
    seed: int,
    temperature: float,
    discussion_max_tokens: int,
    vote_max_tokens: int,
) -> dict[str, Any]:
    logical_calls, per_task = planned_logical_calls(tasks, sessions, rounds)
    first = tasks[0]
    agent = canonical_agents(first)[0]
    request, _, _ = compiler.compile(
        first,
        agent["private_information"],
        (),
        stage="pre_vote",
        task_seed=stable_seed(seed, "preflight", "information"),
        request_seed=stable_seed(seed, "preflight", "request"),
        session_index=0,
        agent_id=int(agent["agent_id"]),
        temperature=temperature,
        max_output_tokens=vote_max_tokens,
    )
    vote_calls = sum(sessions * 3 * len(canonical_agents(task)) for task in tasks)
    discussion_calls = sum(sessions * rounds for _ in tasks)
    assumed_output = max(
        1,
        round(
            (vote_calls * vote_max_tokens + discussion_calls * discussion_max_tokens)
            / logical_calls
        ),
    )
    estimate = static_preflight(
        request,
        provider_config,
        LogicalCallSpec(logical_calls),
        assumed_output_tokens=assumed_output,
    ).to_dict()
    estimate.update(
        {
            "dataset": "HiddenBench canonical",
            "tasks": len(tasks),
            "sessions_per_task": sessions,
            "communication_rounds": rounds,
            "vote_calls": vote_calls,
            "discussion_calls": discussion_calls,
            "per_task": per_task,
            "representative_request": request.to_dict(),
            "warning": (
                "Input tokens are extrapolated from the first pre-vote request. "
                "Discussion transcripts grow over time, so use this as a rough lower-fidelity estimate."
            ),
        }
    )
    return estimate


def _run_spec(
    args: argparse.Namespace,
    provider_config: LLMProviderConfig,
    dataset_metadata: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "hiddenbench_canonical_paper_style",
        "protocol_version": 1,
        "provider": provider_config.to_dict(),
        "dataset": dict(dataset_metadata),
        "task_ids": [int(task["task_id"]) for task in tasks],
        "sessions": args.sessions,
        "rounds": args.rounds,
        "speaker_order": args.speaker_order,
        "seed": args.seed,
        "temperature": args.temperature,
        "discussion_max_tokens": args.discussion_max_tokens,
        "vote_max_tokens": args.vote_max_tokens,
        "validation_retries": args.validation_retries,
        "prompts": {
            "discussion": str(args.discussion_prompt.resolve()),
            "vote": str(args.vote_prompt.resolve()),
        },
    }


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def execute_run(
    tasks: Sequence[Mapping[str, Any]],
    compiler: HiddenBenchPromptCompiler,
    provider_config: LLMProviderConfig,
    run_spec: Mapping[str, Any],
    output_dir: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    results_path = output_dir / "results.json"
    fingerprint = _fingerprint(run_spec)
    if resume:
        if not results_path.is_file():
            raise HiddenBenchRunnerError("--resume requires an existing results.json.")
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        if payload.get("metadata", {}).get("run_spec_fingerprint") != fingerprint:
            raise HiddenBenchRunnerError("Existing results do not match this resolved run spec.")
        results = list(payload.get("results", []))
    else:
        results = []
        (output_dir / "audit.jsonl").write_text("", encoding="utf-8")
        _write_json_atomic(
            results_path,
            {
                "metadata": {
                    "status": "in_progress",
                    "updated_at": _now(),
                    "run_spec_fingerprint": fingerprint,
                },
                "summary": aggregate_results(results),
                "results": results,
            },
        )
    completed = {int(item["task"]["task_id"]) for item in results}
    provider = create_llm_provider(provider_config)
    runner = HiddenBenchProtocolRunner(
        provider,
        compiler,
        AuditWriter(output_dir / "audit.jsonl"),
        base_seed=int(run_spec["seed"]),
        temperature=float(run_spec["temperature"]),
        discussion_max_tokens=int(run_spec["discussion_max_tokens"]),
        vote_max_tokens=int(run_spec["vote_max_tokens"]),
        validation_retries=int(run_spec["validation_retries"]),
    )
    try:
        for task in tasks:
            task_id = int(task["task_id"])
            if task_id in completed:
                continue
            sessions = []
            for session_index in range(int(run_spec["sessions"])):
                print(
                    f"task={task_id} session={session_index + 1}/{run_spec['sessions']}",
                    flush=True,
                )
                sessions.append(
                    await runner.run_session(
                        task,
                        session_index=session_index,
                        communication_rounds=int(run_spec["rounds"]),
                        speaker_order=str(run_spec["speaker_order"]),
                    )
                )
            results.append(
                {
                    "task": {
                        "task_id": task_id,
                        "name": task["name"],
                        "correct_answer": task["correct_answer"],
                    },
                    "summary": task_summary(sessions),
                    "sessions": sessions,
                }
            )
            payload = {
                "metadata": {
                    "status": "in_progress",
                    "updated_at": _now(),
                    "run_spec_fingerprint": fingerprint,
                },
                "summary": aggregate_results(results),
                "results": results,
            }
            _write_json_atomic(results_path, payload)
    finally:
        provider.close()
    payload = {
        "metadata": {
            "status": "complete",
            "updated_at": _now(),
            "run_spec_fingerprint": fingerprint,
        },
        "summary": aggregate_results(results),
        "results": results,
    }
    _write_json_atomic(results_path, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or explicitly execute the canonical HiddenBench protocol "
            "with MAS-CC's lazy local Gemma provider."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--provider-config", type=Path, default=DEFAULT_PROVIDER_CONFIG)
    parser.add_argument("--discussion-prompt", type=Path, default=DEFAULT_DISCUSSION_PROMPT)
    parser.add_argument("--vote-prompt", type=Path, default=DEFAULT_VOTE_PROMPT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/hiddenbench_canonical_gemma")
    )
    parser.add_argument("--task-ids", type=int, nargs="*")
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--speaker-order", choices=("round_robin", "random"), default="round_robin")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--discussion-max-tokens", type=int, default=256)
    parser.add_argument("--vote-max-tokens", type=int, default=128)
    parser.add_argument("--validation-retries", type=int, default=2)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually construct Gemma and run inference; omitted means preflight only.",
    )
    parser.add_argument(
        "--confirm-full-benchmark",
        action="store_true",
        help="Required with --execute when no --task-ids restriction is supplied.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "sessions",
        "rounds",
        "discussion_max_tokens",
        "vote_max_tokens",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.validation_retries < 0:
        parser.error("--validation-retries cannot be negative")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be combined")
    if args.execute and not args.task_ids and not args.confirm_full_benchmark:
        parser.error(
            "full execution requires --confirm-full-benchmark; use --task-ids for a subset"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        tasks, dataset_metadata = load_canonical_tasks(args.input, args.task_ids)
        provider_config = load_component_config(
            args.provider_config.resolve(), "llm_provider", environment={}
        )
        if not isinstance(provider_config, LLMProviderConfig):
            raise HiddenBenchRunnerError("Provider config did not resolve correctly.")
        if provider_config.type != "gemma_local":
            raise HiddenBenchRunnerError("This script requires a gemma_local provider config.")
        compiler = HiddenBenchPromptCompiler.from_files(
            args.discussion_prompt, args.vote_prompt
        )
        run_spec = _run_spec(args, provider_config, dataset_metadata, tasks)
        output_dir = args.output_dir.resolve()
        results_path = output_dir / "results.json"
        if args.execute and results_path.exists() and not (args.resume or args.overwrite):
            raise HiddenBenchRunnerError(
                f"{results_path} exists; use --resume or --overwrite explicitly."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(output_dir / "resolved_run.json", run_spec)
        preflight = build_preflight(
            tasks,
            compiler,
            provider_config,
            sessions=args.sessions,
            rounds=args.rounds,
            seed=args.seed,
            temperature=args.temperature,
            discussion_max_tokens=args.discussion_max_tokens,
            vote_max_tokens=args.vote_max_tokens,
        )
        _write_json_atomic(output_dir / "preflight.json", preflight)
        if not args.execute:
            print(
                f"Preflight only: {preflight['logical_calls']} logical calls across "
                f"{len(tasks)} task(s). No provider was created; Gemma was not loaded."
            )
            print(f"Wrote {output_dir / 'preflight.json'}")
            return 0
        if args.overwrite:
            for name in ("results.json", "audit.jsonl"):
                target = output_dir / name
                if target.exists():
                    target.unlink()
        asyncio.run(
            execute_run(
                tasks,
                compiler,
                provider_config,
                run_spec,
                output_dir,
                resume=args.resume,
            )
        )
        print(f"Completed HiddenBench run: {results_path}")
        return 0
    except (
        HiddenBenchRunnerError,
        ProviderError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
