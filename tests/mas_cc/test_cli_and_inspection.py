import json
from pathlib import Path

from mas_cc.cli.inspect import inspect_phase_2, inspect_phase_3
from mas_cc.cli.main import main
from mas_cc.cli.provider import run_provider_smoke_test
from mas_cc.cli.prompt import generate_paper_prompt_examples


def test_version_command(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0"


def test_phase_2_inspection_contract(tmp_path: Path):
    output = tmp_path / "phase_02"
    assert inspect_phase_2("configs/runs/provider_smoke_test.yaml", output)
    expected = {
        "report.md",
        "manifest.json",
        "input_config.yaml",
        "resolved_config.yaml",
        "config_schema.json",
        "validation_examples.md",
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "pass"
    assert all(manifest["checks"].values())
    assert {entry["path"] for entry in manifest["artifacts"]} == expected - {"manifest.json"}
    assert "POTSDAM_API_KEY" in (output / "resolved_config.yaml").read_text(encoding="utf-8")
    assert "<redacted>" not in (output / "validation_examples.md").read_text(encoding="utf-8")


def test_phase_3_inspection_contract(tmp_path: Path):
    output = tmp_path / "phase_03"
    assert inspect_phase_3(
        "configs/components/prompts/basic_binary_choice.yaml", output
    )
    expected = {
        "report.md",
        "manifest.json",
        "prompt_context.json",
        "prompt_blocks.json",
        "compiled_messages.json",
        "rendered_prompt.md",
        "token_breakdown.csv",
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "pass"
    assert all(manifest["checks"].values())
    blocks = json.loads((output / "prompt_blocks.json").read_text(encoding="utf-8"))
    assert [block["name"] for block in blocks] == [
        "task_description",
        "game_rules",
        "private_state",
        "recent_memory",
        "current_interaction",
        "decision_instruction",
        "output_contract",
    ]
    messages = json.loads(
        (output / "compiled_messages.json").read_text(encoding="utf-8")
    )
    assert len(messages) == 7
    assert {message["role"] for message in messages} == {"system", "user"}


def test_phase_4_mock_provider_inspection_contract(tmp_path: Path):
    output = tmp_path / "phase_04" / "mock"
    assert run_provider_smoke_test(
        "mock", "configs/components/prompts/basic_binary_choice.yaml", output
    )
    expected = {
        "report.md",
        "manifest.json",
        "request.json",
        "normalized_response.json",
        "raw_response_redacted.json",
        "usage.json",
        "preflight_estimate.json",
        "timing.csv",
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "pass"
    assert all(manifest["checks"].values())
    response = json.loads((output / "normalized_response.json").read_text(encoding="utf-8"))
    assert response["provider"] == "mock"
    assert response["content"] == "A"
    assert "Bearer " not in "".join(
        path.read_text(encoding="utf-8") for path in output.iterdir()
    )


def test_paper_prompt_example_bundle_is_readable_and_machine_inspectable(tmp_path: Path):
    output = tmp_path / "paper_prompts"
    data = Path(
        "scripts/local_llms/hiddenbench_population_pipeline/data/hiddenbench/"
        "scaled/exact_replication/N_32.json"
    )
    generate_paper_prompt_examples(
        output, hiddenbench_data=data, task_id=1, agent_id=0
    )
    examples = {
        "social_conventions",
        "hiddenbench_first_speaker",
        "hiddenbench_discussion",
        "hiddenbench_pre_vote",
        "hiddenbench_post_vote",
    }
    for name in examples:
        request = (output / name / "request.md").read_text(encoding="utf-8")
        messages = json.loads(
            (output / name / "compiled_messages.json").read_text(encoding="utf-8")
        )
        assert [message["role"] for message in messages] == ["system", "user"]
        assert all(message["content"] in request for message in messages)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "pass"
    assert all(manifest["checks"].values())
    assert "correct_answer" not in (
        output / "hiddenbench_discussion/prompt_context.json"
    ).read_text(encoding="utf-8")
    combined = (output / "all_requests.md").read_text(encoding="utf-8")
    assert "Social conventions paper" in combined
    assert "HiddenBench paper" in combined
