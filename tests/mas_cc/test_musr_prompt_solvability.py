"""Prompt ablation, task split, and staged selection contracts."""

from __future__ import annotations

from mas_cc.probes.musr_prompt_solvability.analysis import select_packet, select_prompt
from mas_cc.probes.musr_prompt_solvability.config import load_config
from mas_cc.probes.musr_prompt_solvability.design import (
    packet_definitions,
    phase_a,
    phase_b,
    phase_c,
)
from mas_cc.probes.musr_prompt_solvability.prompting import render
from mas_cc.probes.musr_prompt_solvability.runner import load_tasks

CONFIG = "configs/probes/musr_prompt_solvability_calibration_01.yaml"


def test_design_separates_development_and_heldout_and_counts_calls():
    config = load_config(CONFIG)
    tasks = load_tasks(config)
    assert config.development_tasks == ("task_001", "task_002")
    assert config.heldout_tasks == ("task_003",)
    a = phase_a(tasks, config.development_tasks, config.prompt_repetitions, config.seed)
    b = phase_b(
        tasks, config.development_tasks, "P3", config.packet_repetitions, config.seed
    )
    c = phase_c(
        tasks,
        config.heldout_tasks,
        "P3",
        "F9",
        config.heldout_repetitions,
        12,
        config.seed,
    )
    assert (len(a), len(b), len(c), len(a) + len(b) + len(c)) == (80, 60, 140, 280)
    assert not {s.task_id for s in (*a, *b)} & set(config.heldout_tasks)


def test_prompt_ablation_is_compositional_and_p0_is_unchanged():
    config = load_config(CONFIG)
    tasks = load_tasks(config)
    specs = phase_a(tasks, ("task_001",), 1, config.seed)
    by_variant = {
        spec.prompt_variant: render(tasks[spec.task_id], spec) for spec in specs
    }
    text = {
        key: "\n".join(m.content for m in value.messages)
        for key, value in by_variant.items()
    }
    assert "Some participants may have objectives" in text["P0"]
    assert "Evaluate each candidate allocation" not in text["P0"]
    assert "Evaluate each candidate allocation" in text["P1"]
    assert "Some participants may have objectives" not in text["P2"]
    assert "Evaluate each candidate allocation" in text["P3"]
    assert "Some participants may have objectives" not in text["P3"]
    assert all(
        "candidate_scores" not in value and "gold_answer" not in value
        for value in text.values()
    )
    mappings = {tuple(spec.option_mapping.items()) for spec in specs}
    seeds = {spec.provider_seed for spec in specs}
    assert len(mappings) == len(seeds) == 1


def test_full_profile_packets_have_fixed_breadth_and_redundancy():
    config = load_config(CONFIG)
    tasks = load_tasks(config)
    definitions = packet_definitions(tasks)
    for task_id, packets in definitions.items():
        assert [len(packets[name]) for name in ("F9", "F18", "F27")] == [9, 18, 27]
        assert set(packets["F9"]) < set(packets["F18"]) < set(packets["F27"])
        groups = tasks[task_id].supporting_fact_groups or {}
        card_group = {
            card: latent for latent, cards in groups.items() for card in cards
        }
        assert all(
            len({card_group[card] for card in packets[name]}) == 9 for name in packets
        )


def test_selection_rules_prefer_simpler_near_ties_and_smallest_passing_packet():
    prompts = [
        {"prompt_variant": "P0", "truth_rate": 0.85, "parse_rate": 1.0},
        {"prompt_variant": "P1", "truth_rate": 0.89, "parse_rate": 1.0},
        {"prompt_variant": "P2", "truth_rate": 0.88, "parse_rate": 1.0},
        {"prompt_variant": "P3", "truth_rate": 0.90, "parse_rate": 1.0},
    ]
    assert select_prompt(prompts) == "P0"
    packets = [
        {"packet_variant": "F9", "truth_rate": 0.79, "parse_rate": 1.0},
        {"packet_variant": "F18", "truth_rate": 0.85, "parse_rate": 1.0},
        {"packet_variant": "F27", "truth_rate": 0.95, "parse_rate": 1.0},
    ]
    assert select_packet(packets) == "F18"
