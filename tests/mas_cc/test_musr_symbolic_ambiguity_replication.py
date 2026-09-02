from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from mas_cc.probes.musr_prompt_solvability.prompting import render
from mas_cc.probes.musr_symbolic_ambiguity_replication.analysis import (
    existing_task_diagnostic,
)
from mas_cc.probes.musr_symbolic_ambiguity_replication.config import load_config
from mas_cc.probes.musr_symbolic_ambiguity_replication.design import (
    additional_call_plan,
)
from mas_cc.probes.musr_symbolic_ambiguity_replication.runner import (
    load_tasks,
    verify_source_reference,
)

CONFIG = Path("configs/probes/musr_symbolic_ambiguity_replication_01.yaml")
SOURCE = Path("results/studies/musr_symbolic_ambiguity_calibration_01")


def _plan():
    config = load_config(CONFIG)
    tasks = load_tasks(config)
    return (
        config,
        tasks,
        additional_call_plan(
            tasks,
            private_start=config.private_existing_repetitions,
            private_count=config.private_additional_repetitions,
            endpoint_start=config.endpoint_existing_repetitions,
            endpoint_count=config.endpoint_additional_repetitions,
            seed=config.seed,
        ),
    )


def test_config_freezes_replication_contract():
    config = load_config(CONFIG)
    assert config.provider.model == "gwdg/openai-gpt-oss-120b"
    assert config.provider.temperature == 1.0
    assert config.prompt_variant == "P2"
    assert config.full_profile == "F9"
    assert config.additional_calls == 336
    assert config.final_calls == 672
    assert config.output_dir.name == "musr_symbolic_ambiguity_replication_01"


def test_source_reference_and_existing_diagnostic_are_valid():
    config = load_config(CONFIG)
    reference = verify_source_reference(config)
    rows = existing_task_diagnostic(config.calibration_root)
    assert (
        reference["manifest_content_sha256"] == config.expected_manifest_content_sha256
    )
    assert len(rows) == 6
    assert {row["task_id"] for row in rows} == {
        f"task_{index:03d}" for index in range(1, 7)
    }
    assert (
        next(row for row in rows if row["task_id"] == "task_004")["private_truth_rate"]
        == 29 / 36
    )


def test_additional_plan_has_exact_ranges_counts_and_disjoint_ids():
    config, _, specs = _plan()
    assert len(specs) == 336
    assert Counter(spec.packet_variant for spec in specs) == {
        "Private": 216,
        "Zero": 60,
        "F9": 60,
    }
    assert {spec.repetition for spec in specs if spec.packet_variant == "Private"} == {
        3,
        4,
        5,
    }
    assert {
        spec.repetition for spec in specs if spec.packet_variant != "Private"
    } == set(range(10, 20))
    source_plan = json.loads(
        (SOURCE / "preflight/behavioral_call_plan.json").read_text()
    )
    assert not {row["call_id"] for row in source_plan}.intersection(
        spec.call_id for spec in specs
    )
    assert (
        len({row["call_id"] for row in source_plan} | {spec.call_id for spec in specs})
        == 672
    )
    assert all(spec.to_dict()["replicate_id"] == spec.repetition for spec in specs)


def test_endpoint_pairing_and_plan_determinism():
    config, tasks, specs = _plan()
    repeated = additional_call_plan(
        tasks,
        private_start=3,
        private_count=3,
        endpoint_start=10,
        endpoint_count=10,
        seed=config.seed,
    )
    assert [spec.to_dict() for spec in specs] == [spec.to_dict() for spec in repeated]
    by_id = {spec.call_id: spec for spec in specs}
    for task_id in tasks:
        for repetition in range(10, 20):
            zero = by_id[f"{task_id}:Zero:{repetition:02d}"]
            full = by_id[f"{task_id}:F9:{repetition:02d}"]
            assert zero.option_mapping == full.option_mapping
            assert zero.provider_seed == full.provider_seed


def test_packets_latents_and_prompts_remain_frozen():
    _, tasks, specs = _plan()
    archived_full = json.loads(
        (SOURCE / "accepted_tasks/full_profile_packets.json").read_text()
    )
    archived_private = json.loads(
        (SOURCE / "accepted_tasks/private_assignments.json").read_text()
    )
    forbidden = (
        "skill_matrix",
        "cooperation_matrix",
        "candidate_scores",
        "gold_answer",
        "max_predictability",
        "normalized_entropy",
    )
    for spec in specs:
        if spec.packet_variant == "F9":
            assert list(spec.evidence_ids) == archived_full[spec.task_id]
            assert len(spec.latent_ids) == 9
        elif spec.packet_variant == "Private":
            assert (
                list(spec.evidence_ids)
                == archived_private[spec.task_id]["agent_evidence_ids"][
                    str(spec.agent_id - 1)
                ]
            )
            assert len(spec.latent_ids) == 4
        else:
            assert not spec.evidence_ids
            assert not spec.latent_ids
        prompt = "\n".join(
            message.content for message in render(tasks[spec.task_id], spec).messages
        )
        assert all(term not in prompt for term in forbidden)
