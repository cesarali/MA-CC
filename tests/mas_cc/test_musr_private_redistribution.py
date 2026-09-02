"""Structural assignment and call-plan contracts for redistribution."""
from __future__ import annotations
from mas_cc.probes.musr_private_redistribution.assignment import build_assignment,structural_summary
from mas_cc.probes.musr_private_redistribution.config import load_config
from mas_cc.probes.musr_private_redistribution.design import assignments,call_plan
from mas_cc.probes.musr_private_redistribution.analysis import select_regime
from mas_cc.probes.musr_private_redistribution.runner import load_tasks

CONFIG="configs/probes/musr_private_redistribution_calibration_01.yaml"

def test_natural_audit_matches_known_structure():
    cfg=load_config(CONFIG); tasks=load_tasks(cfg); summaries=[structural_summary(build_assignment(task,"NAT",cfg.seed)) for task in tasks.values()]
    assert sum(row["mean_latent_values_per_agent"] for row in summaries)/3==6.75
    assert sum(row["agents_fully_scoring_any_candidate"] for row in summaries)==24
    assert sum(row["fully_scoreable_candidate_incidences"] for row in summaries)==25

def test_redistributions_have_six_cards_exact_breadth_and_holder_profiles():
    cfg=load_config(CONFIG); tasks=load_tasks(cfg); amap=assignments(tasks,cfg.seed); expected={"R2":sorted([3]*6+[2]*3),"R3":[4]*9,"R4":sorted([6]*3+[5]*6)}
    for task_id,task in tasks.items():
        for regime in ("R2","R3","R4"):
            value=amap[(task_id,regime)]; rows=value["diagnostics"]
            assert {row["num_cards"] for row in rows}=={6}
            assert {row["num_latent_values"] for row in rows}=={int(regime[1])}
            assert sorted(value["latent_holder_counts"].values())==expected[regime]
            assert set(value["card_holder_counts"])==set(task.fact_order)
            if regime=="R4": assert all(row["num_fully_scoreable_allocations"]==0 for row in rows)

def test_assignments_are_seeded_and_reproducible():
    cfg=load_config(CONFIG); task=load_tasks(cfg)["task_001"]
    first=build_assignment(task,"R3",cfg.seed); second=build_assignment(task,"R3",cfg.seed); different=build_assignment(task,"R3",cfg.seed+1)
    assert first==second
    assert first["assignment_sha256"]!=different["assignment_sha256"]
    assert structural_summary(first)==structural_summary(different)

def test_call_plan_has_492_calls_and_matched_private_mappings():
    cfg=load_config(CONFIG); tasks=load_tasks(cfg); amap=assignments(tasks,cfg.seed); specs=call_plan(tasks,amap,cfg.private_repetitions,cfg.endpoint_repetitions,cfg.seed)
    assert len(specs)==492
    assert sum(s.packet_variant in {"NAT","R2","R3","R4"} for s in specs)==432
    assert sum(s.packet_variant in {"Zero","F9"} for s in specs)==60
    for task_id in tasks:
        for agent in range(1,13):
            for repetition in range(3):
                group=[s for s in specs if s.task_id==task_id and s.agent_id==agent and s.repetition==repetition]
                assert len(group)==4
                assert len({tuple(s.option_mapping.items()) for s in group})==1
                assert len({s.provider_seed for s in group})==1

def test_selection_prefers_largest_difficult_regime():
    rows=[{"regime":"R2","truth_rate":.2},{"regime":"R3","truth_rate":.45},{"regime":"R4","truth_rate":.55}]
    assert select_regime(rows)=="R3"
