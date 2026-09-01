from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from mas_cc.llm_runtime.providers import CompletionResponse, ProviderCapabilities
from mas_cc.musr_team_allocation_generator.validation_study import (
    ValidationStudyConfig,
    _displayed_options,
    run_validation_study,
    validation_prompt,
)
from mas_cc.musr_team_allocation_generator.validation_comparison import (
    add_validation_model,
)


class ValidationStudyProvider:
    name = "mock-study"
    capabilities = ProviderCapabilities(supports_seed=True)

    def __init__(self, model: str = "mock-study-model") -> None:
        self.model = model

    async def complete(self, request):
        prompt = request.messages[-1].content
        if "Create independent indirect evidence branches" in prompt:
            target = json.loads(
                re.search(
                    r"Hidden target \(never copy into explicit statements\): (\{.*\})",
                    prompt,
                ).group(1)
            )
            branches = int(re.search(r"exactly (\d+) branches", prompt).group(1))
            statements = int(
                re.search(r"exactly (\d+) non-empty statements", prompt).group(1)
            )
            intermediates = int(
                re.search(r"exactly (\d+) intermediate_claims", prompt).group(1)
            )
            content = json.dumps(
                {
                    "branches": [
                        {
                            "intermediate_claims": [
                                f"Indirect inference {target['hidden_fact_id']} {branch} {index}."
                                for index in range(intermediates)
                            ],
                            "statements": [
                                f"Recorded event {target['hidden_fact_id']} {branch}-{index} produced a concrete outcome."
                                for index in range(statements)
                            ],
                            "commonsense_bridges": [
                                "Repeated concrete outcomes can indicate a relevant capability."
                            ],
                        }
                        for branch in range(branches)
                    ]
                }
            )
        else:
            content = json.dumps({"option_label": "A", "rationale": "Mock choice."})
        return CompletionResponse(content=content, provider=self.name, model=self.model)

    def close(self):
        pass


def test_validation_prompt_hides_exact_evaluation_metadata():
    task = {
        "scenario": "A public scenario.",
        "options": [
            {"id": f"ALLOCATION_{index}", "display_text": f"Display {index}"}
            for index in range(3)
        ],
        "evidence": [
            {"evidence_id": "e1", "text": ["Visible fact."], "latent_fact_id": "hidden"}
        ],
        "latent": {"secret": 3},
        "gold_index": 2,
        "gold_answer": "ALLOCATION_2",
    }
    displayed, mapping = _displayed_options(task, seed=_seed(5))
    prompt = validation_prompt(task, displayed, ["e1"], condition="partial")
    assert "Visible fact" in prompt
    assert "secret" not in prompt
    assert "gold_index" not in prompt
    assert "ALLOCATION_" not in prompt
    assert set(mapping) == {"A", "B", "C"}


def _seed(value):
    from mas_cc.core.random import Seed

    return Seed(value)


def test_complete_mock_validation_study_artifact(tmp_path: Path):
    config = ValidationStudyConfig(
        seed=44,
        candidate_limit=3,
        branches_per_latent_fact=3,
        tree_depth=2,
        skip_full_acceptance_for_testing=True,
    )
    result = asyncio.run(
        run_validation_study(
            ValidationStudyProvider(),
            config,
            output=tmp_path,
            repository_root=Path(__file__).resolve().parents[2],
        )
    )
    assert result["manifest"]["observed_validation_calls"] == 138
    assert len(list((tmp_path / "tasks").glob("task_*/base_task.json"))) == 3
    assert sum(1 for _ in (tmp_path / "raw/full_information.jsonl").open()) == 15
    assert sum(1 for _ in (tmp_path / "raw/zero_information.jsonl").open()) == 15
    assert sum(1 for _ in (tmp_path / "raw/partial_N12.jsonl").open()) == 36
    assert sum(1 for _ in (tmp_path / "raw/partial_N24.jsonl").open()) == 72
    assert (tmp_path / "analysis/validation_report.md").is_file()
    assert (
        tmp_path / "analysis/figures/accuracy_by_information_condition.png"
    ).is_file()
    assert (tmp_path / "analysis/figures/partial_accuracy_by_population.png").is_file()
    for task_dir in sorted((tmp_path / "tasks").iterdir()):
        base = json.loads((task_dir / "base_task.json").read_text())
        n12 = json.loads((task_dir / "distribution_N12.json").read_text())
        n24 = json.loads((task_dir / "distribution_N24.json").read_text())
        assert n12["semantic_world_sha256"] == base["semantic_world_sha256"]
        assert n24["semantic_world_sha256"] == base["semantic_world_sha256"]
        assert n12["no_single_agent_violations"] == 0
        assert n24["no_single_agent_violations"] == 0

    comparison = asyncio.run(
        add_validation_model(
            ValidationStudyProvider("mock-second-model"),
            study_dir=tmp_path,
            seed=45,
        )
    )
    assert len(comparison["rows"]) == 138
    assert (
        sum(1 for _ in (tmp_path / "raw/validation_mock_second_model.jsonl").open())
        == 138
    )
    combined = (tmp_path / "analysis/behavioral_summary_by_model.csv").read_text()
    assert "mock-study-model" in combined
    assert "mock-second-model" in combined
    report = (tmp_path / "analysis/validation_report.md").read_text()
    assert "## H. Validation-model comparison" in report
