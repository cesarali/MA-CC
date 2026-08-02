import json
from pathlib import Path

from mas_cc.cli.inspect import inspect_phase_2, inspect_phase_3
from mas_cc.cli.main import main
from mas_cc.cli.game import run_game_inspection
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
        "prompt_schema_v2.json",
        "resolved_prompt_component.yaml",
        "v1_to_v2_migration_examples.md",
        "secret_scan.json",
        "validation_examples.md",
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "pass"
    assert all(manifest["checks"].values())
    assert {entry["path"] for entry in manifest["artifacts"]} == expected - {"manifest.json"}
    assert "POTSDAM_API_KEY" in (output / "resolved_config.yaml").read_text(encoding="utf-8")
    assert "<redacted>" not in (output / "validation_examples.md").read_text(encoding="utf-8")
    prompt_schema = json.loads((output / "prompt_schema_v2.json").read_text())
    assert prompt_schema["properties"]["schema_version"] == {"const": 2}
    assert "blocks" not in prompt_schema["properties"]


def test_phase_3_inspection_contract(tmp_path: Path):
    output = tmp_path / "phase_03"
    assert inspect_phase_3(
        "configs/components/prompts/basic_choice_v3.yaml", output
    )
    expected = {
        "report.md",
        "manifest.json",
        "compiled_messages.json",
        "rendered_prompt.md",
        "token_breakdown.csv",
        "full_prompt_definition.json",
        "unbound_prompt.json",
        "bound_prompt.json",
        "block_manifest.json",
        "rendered_blocks.json",
        "omitted_blocks.json",
        "fingerprints.json",
        "validation_examples.md",
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "pass"
    assert all(manifest["checks"].values())
    blocks = json.loads((output / "rendered_blocks.json").read_text(encoding="utf-8"))
    assert [block["name"] for block in blocks] == [
        "task",
        "rules",
        "private_state",
        "recent_memory",
        "current_interaction",
    ]
    messages = json.loads(
        (output / "compiled_messages.json").read_text(encoding="utf-8")
    )
    assert len(messages) == 2
    assert {message["role"] for message in messages} == {"system", "user"}


def test_phase_4_mock_provider_inspection_contract(tmp_path: Path):
    output = tmp_path / "phase_04" / "mock"
    assert run_provider_smoke_test(
        "mock", "configs/components/prompts/basic_choice_v3.yaml", output
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
        "compiled_prompt.json",
        "pricing_snapshot.json",
        "budget_status.json",
        "provider_boundary_diff.md",
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


def test_phase_5_game_inspection_contract_and_deterministic_artifacts(tmp_path: Path):
    first = tmp_path / "first" / "phase_05"
    second = tmp_path / "second" / "phase_05"
    assert run_game_inspection("configs/runs/toy_game_smoke_test.yaml", first)
    assert run_game_inspection("configs/runs/toy_game_smoke_test.yaml", second)
    expected = {
        "report.md",
        "manifest.json",
        "resolved_config.yaml",
        "initial_state.json",
        "interactions.jsonl",
        "final_state.json",
        "game_call_plan.json",
        "trajectory.csv",
        "trajectory.png",
        "observations.jsonl",
        "bound_prompts.jsonl",
        "compiled_prompts.jsonl",
        "prompt_scenarios.json",
    }
    assert {path.name for path in first.iterdir()} == expected
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == 5
    assert manifest["status"] == "pass"
    assert all(manifest["checks"].values())

    deterministic = expected - {"report.md", "manifest.json"}
    assert {
        name: (first / name).read_bytes() for name in deterministic
    } == {
        name: (second / name).read_bytes() for name in deterministic
    }
    interactions = [
        json.loads(line)
        for line in (first / "interactions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(interactions) == 3
    assert all(len(item["decisions"]) == 2 for item in interactions)
    plan = json.loads((first / "game_call_plan.json").read_text(encoding="utf-8"))
    assert plan["provider_requests"] == {"lower": 6, "expected": 6, "maximum": 6}
    assert plan["metadata"]["provider_prices_included"] is False
    assert (first / "trajectory.png").stat().st_size > 0


def test_phase_5_cli_aliases(tmp_path: Path, capsys):
    direct = tmp_path / "direct"
    assert main(
        [
            "game",
            "run",
            "--config",
            "configs/runs/toy_game_smoke_test.yaml",
            "--output-dir",
            str(direct),
        ]
    ) == 0
    assert "Game inspection passed" in capsys.readouterr().out

    standard = tmp_path / "standard"
    assert main(["inspect", "phase", "5", "--output-dir", str(standard)]) == 0
    assert "Phase 5 inspection passed" in capsys.readouterr().out


def test_phase_6_inspection_contract_audit_and_determinism(tmp_path: Path):
    first = tmp_path / "first" / "phase_06"
    second = tmp_path / "second" / "phase_06"
    assert run_game_inspection(
        "configs/runs/naming_convention_smoke_test.yaml", first
    )
    assert run_game_inspection(
        "configs/runs/naming_convention_smoke_test.yaml", second
    )
    expected = {
        "report.md",
        "manifest.json",
        "resolved_config.yaml",
        "agents_initial.json",
        "interactions.jsonl",
        "selected_audit_traces.jsonl",
        "game_call_plan.json",
        "prompt_token_scenarios.csv",
        "trajectory.csv",
        "action_share.png",
        "coordination_rate.png",
        "agents_final.json",
        "full_prompt_definition.json",
        "selected_block_traces.jsonl",
        "prompt_parity_report.md",
    }
    assert {path.name for path in first.iterdir()} == expected
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == 6
    assert manifest["status"] == "pass"
    assert all(manifest["checks"].values())

    deterministic = expected - {"report.md", "manifest.json"}
    assert {
        name: (first / name).read_bytes() for name in deterministic
    } == {
        name: (second / name).read_bytes() for name in deterministic
    }
    interactions = [
        json.loads(line)
        for line in (first / "interactions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(interactions) == 12
    first_decision = interactions[0]["decisions"]["player_1"]
    assert first_decision["visible_memory"] == []
    assert first_decision["compiled_messages"]
    assert first_decision["raw_response"]
    assert first_decision["parsed_action"] == "Q"
    assert first_decision["validation"]["valid"] is True
    assert interactions[0]["payoff"] == 100
    assert interactions[0]["post_interaction_private_memory"]["player_1"]

    selected = (first / "selected_audit_traces.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert [json.loads(line)["interaction_index"] for line in selected] == [1, 6, 12]
    plan = json.loads((first / "game_call_plan.json").read_text(encoding="utf-8"))
    assert plan["logical_decisions"]["expected"] == 24
    assert plan["provider_requests"] == {
        "lower": 24,
        "expected": 24,
        "maximum": 72,
    }
    report = (first / "report.md").read_text(encoding="utf-8")
    assert "not a paper replication" in report
    assert "Legacy fixed-fixture parity: passed" in report


def test_phase_6_standard_inspection_cli(tmp_path: Path, capsys):
    output = tmp_path / "phase_06"
    assert main(["inspect", "phase", "6", "--output-dir", str(output)]) == 0
    assert "Phase 6 inspection passed" in capsys.readouterr().out


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
        output / "hiddenbench_discussion/bound_prompt.json"
    ).read_text(encoding="utf-8")
    combined = (output / "all_requests.md").read_text(encoding="utf-8")
    assert "Social conventions paper" in combined
    assert "HiddenBench paper" in combined
