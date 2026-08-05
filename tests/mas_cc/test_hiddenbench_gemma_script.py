import asyncio
import json
import sys
from pathlib import Path

from mas_cc.llm_runtime.providers import (
    CompletionResponse,
    ProviderCapabilities,
    ProviderUsage,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.local_llms.hiddenbench_gemma import run_hiddenbench_canonical as runner


class FakeHiddenBenchProvider:
    name = "fake"
    model = "fake-hiddenbench"
    capabilities = ProviderCapabilities(supports_seed=True, reports_usage=True)

    def __init__(self):
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        stage = request.metadata["stage"]
        if stage == "discussion":
            content = "My evidence rules out one of the unsafe routes."
        else:
            allowed = request.metadata["response_contract"]["allowed_values"]
            content = json.dumps(
                {"vote": allowed[0], "rationale": "This best fits my available evidence."}
            )
        return CompletionResponse(
            content=content,
            provider=self.name,
            model=self.model,
            usage=ProviderUsage(10, 3, 13),
            raw_response={"content": content},
        )

    def close(self):
        pass


def _task_one():
    tasks, metadata = runner.load_canonical_tasks(runner.DEFAULT_INPUT, [1])
    return tasks[0], metadata


def test_canonical_loader_and_call_plan_match_paper_protocol():
    task, metadata = _task_one()
    assert metadata["kind"] == "canonical"
    assert len(runner.canonical_agents(task)) == 4
    calls, rows = runner.planned_logical_calls([task], sessions=10, rounds=15)
    assert calls == 270
    assert rows == [{"task_id": 1, "agents": 4, "logical_calls": 270}]


def test_mocked_session_preserves_information_boundaries_and_audit(tmp_path: Path):
    task, _ = _task_one()
    compiler = runner.HiddenBenchPromptCompiler.from_files(
        runner.DEFAULT_DISCUSSION_PROMPT, runner.DEFAULT_VOTE_PROMPT
    )
    provider = FakeHiddenBenchProvider()
    audit_path = tmp_path / "audit.jsonl"
    protocol = runner.HiddenBenchProtocolRunner(
        provider,
        compiler,
        runner.AuditWriter(audit_path),
        base_seed=1729,
        temperature=0.0,
        discussion_max_tokens=64,
        vote_max_tokens=64,
        validation_retries=0,
    )
    result = asyncio.run(
        protocol.run_session(
            task, session_index=0, communication_rounds=2, speaker_order="round_robin"
        )
    )
    assert len(provider.requests) == 14
    assert len(result["pre_discussion_decisions"]) == 4
    assert len(result["discussion_history"]) == 2
    assert len(result["post_discussion_decisions"]) == 4
    assert len(result["full_profile_decisions"]) == 4
    assert all(
        len(item["information"]) == len(task["shared_information"]) + 1
        for item in result["pre_discussion_decisions"]
    )
    assert all(
        len(item["information"])
        == len(task["shared_information"]) + len(task["hidden_information"])
        for item in result["full_profile_decisions"]
    )
    records = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert len(records) == 14
    assert all(record["validation"]["valid"] for record in records)
    assert all("correct_answer" not in json.dumps(record["request"]) for record in records)
    assert all(
        set(message) == {"role", "content"}
        for request in provider.requests
        for message in request.wire_messages()
    )


def test_default_mode_is_preflight_only_and_never_constructs_provider(
    tmp_path: Path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run preflight must not construct a provider")

    monkeypatch.setattr(runner, "create_llm_provider", forbidden)
    output = tmp_path / "preflight"
    assert (
        runner.main(
            [
                "--task-ids",
                "1",
                "--sessions",
                "1",
                "--rounds",
                "2",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert {path.name for path in output.iterdir()} == {
        "preflight.json",
        "resolved_run.json",
    }
    estimate = json.loads((output / "preflight.json").read_text())
    assert estimate["logical_calls"] == 14
    resolved = json.loads((output / "resolved_run.json").read_text())
    assert resolved["provider"]["type"] == "gemma_local"
