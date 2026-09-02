"""Design, prompt-fidelity, execution, and analysis contracts for the probe."""

from __future__ import annotations

import json
from pathlib import Path

from mas_cc.probes.musr_local_evidence.analysis import (
    summarize_doses,
    summarize_prompt_equivalence,
)
from mas_cc.probes.musr_local_evidence.config import load_probe_config
from mas_cc.probes.musr_local_evidence.design import build_call_plan
from mas_cc.probes.musr_local_evidence.preflight import build_plan, preflight_payload
from mas_cc.probes.musr_local_evidence.prompting import render_call

CONFIG = Path("configs/probes/musr_local_evidence_probe_01.yaml")


def test_probe_design_has_exact_paired_and_nested_call_contract():
    config = load_probe_config(CONFIG)
    plan = build_plan(config)
    assert plan.passed
    assert len(plan.calls) == 123
    assert sum(c.experiment == "prompt_equivalence" for c in plan.calls) == 60
    assert sum(c.experiment == "evidence_dose" for c in plan.calls) == 63
    pairs = {}
    for call in plan.calls:
        if call.pair_id:
            pairs.setdefault(call.pair_id, []).append(call)
    assert len(pairs) == 30
    assert all(
        len(v) == 2
        and v[0].option_mapping == v[1].option_mapping
        and v[0].evidence_ids == v[1].evidence_ids
        and v[0].requested_seed == v[1].requested_seed
        for v in pairs.values()
    )
    for agent in config.agents:
        rows = sorted(
            (r for r in plan.dose_definitions if r["agent_id"] == agent),
            key=lambda r: r["dose"],
        )
        assert [r["distinct_latent_fact_count"] for r in rows[:4]] == [0, 3, 6, 9]
        assert all(
            set(a["evidence_ids"]) < set(b["evidence_ids"])
            for a, b in zip(rows, rows[1:])
        )
        assert len(rows[-1]["evidence_ids"]) == 27


def test_prompts_use_exact_existing_renderers_and_hide_latent_metadata():
    config = load_probe_config(CONFIG)
    plan = build_plan(config)
    pair = next(
        c
        for c in plan.calls
        if c.experiment == "prompt_equivalence"
        and c.agent_number == 1
        and c.prompt_family == "validation"
    )
    game = next(
        c
        for c in plan.calls
        if c.pair_id == pair.pair_id and c.prompt_family == "game_init"
    )
    validation = render_call(plan.task, pair)
    game_prompt = render_call(plan.task, game)
    assert validation.prompt_family == "musr_team_allocation_validation"
    assert game_prompt.prompt_family == "relational_public_ballot"
    assert pair.option_mapping == game.option_mapping
    assert pair.evidence_ids == game.evidence_ids
    for rendered in (validation, game_prompt):
        text = "\n".join(message.content for message in rendered.messages)
        assert "build the data pipeline" in text
        assert "skill_matrix" not in text
        assert "cooperation_matrix" not in text
        assert "hidden_claim" not in text


def test_preflight_accounts_for_every_rendered_call():
    config = load_probe_config(CONFIG)
    plan = build_plan(config)
    payload = preflight_payload(config, plan)
    assert payload["passed"]
    assert payload["calls"] == {
        "prompt_equivalence": 60,
        "evidence_dose": 63,
        "total": 123,
        "maximum_http_attempts": 123,
    }
    assert payload["tokens"]["estimated_input_total"] == sum(
        item.token_estimate for item in plan.rendered.values()
    )


def _finished(
    call_id, experiment, prompt_family, agent, dose, answer, correct, pair_id=None
):
    return {
        "event": "call_finished",
        "call_id": call_id,
        "experiment": experiment,
        "prompt_family": prompt_family,
        "agent_id": agent,
        "dose": dose,
        "repetition": 0,
        "pair_id": pair_id,
        "parse_success": True,
        "parsed_semantic_answer": answer,
        "correct": correct,
        "evidence_ids": [],
        "distinct_latent_facts": dose or 0,
        "usage": {},
    }


def test_analysis_uses_semantic_answers_and_preserves_pair_direction():
    eq = [
        _finished(
            "v", "prompt_equivalence", "validation", 1, None, "ALLOCATION_2", True, "p"
        ),
        _finished(
            "g", "prompt_equivalence", "game_init", 1, None, "ALLOCATION_1", False, "p"
        ),
    ]
    pairs, summary = summarize_prompt_equivalence(eq)
    assert pairs[0]["validation_correct_game_wrong"]
    assert summary[-1]["paired_disagreement_rate"] == 1.0
    dose = [
        _finished("d0", "evidence_dose", "game_init", 1, 0, "ALLOCATION_0", False),
        _finished("d3", "evidence_dose", "game_init", 1, 3, "ALLOCATION_2", True),
    ]
    observations, summaries = summarize_doses(dose)
    assert [r["truth_rate"] for r in summaries] == [0.0, 1.0]
