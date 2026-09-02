from __future__ import annotations

import json
from pathlib import Path

from mas_cc.probes.musr_blackboard_prompt_validation.config import load_config
from mas_cc.probes.musr_blackboard_prompt_validation.execution import call_plan
from mas_cc.probes.musr_blackboard_prompt_validation.runner import _sanity, _tasks
from mas_cc.probes.musr_blackboard_prompt_validation.states import build_frozen_states

FULL = Path("configs/probes/musr_blackboard_prompt_validation_01_full.yaml")
SMOKE = Path("configs/probes/musr_blackboard_prompt_validation_01_smoke.yaml")


def test_configs_freeze_smoke_and_full_counts():
    smoke = load_config(SMOKE)
    full = load_config(FULL)
    assert smoke.state_count == 6
    assert smoke.logical_calls == 12
    assert full.state_count == 72
    assert full.logical_calls == 360
    assert full.max_concurrency == 30
    assert full.max_rpm == 500
    assert full.fallback_concurrency == (30, 20, 10)


def test_frozen_states_use_actual_runtime_and_expected_coverage(tmp_path):
    config = load_config(SMOKE)
    states = build_frozen_states(config, _tasks(config))
    assert len(states) == 6
    for frozen in states.values():
        row = frozen.definition
        assert frozen.compiled_prompt.family == "relational_blackboard_ballot"
        assert (
            frozen.request.prompt.response_contract.type
            == "relational_blackboard_ballot"
        )
        assert (
            row["latent_coverage_count"] == {"S0": 4, "S1": 6, "S2": 9}[row["state_id"]]
        )
        assert len(row["sampled_message_ids"]) == (0 if row["state_id"] == "S0" else 1)
        assert set(row["original_evidence_ids"]).issubset(row["total_evidence_ids"])
        assert set(row["acquired_evidence_ids"]).issubset(row["total_evidence_ids"])
        if row["state_id"] == "S2":
            assert row["sampled_message_types"] == ["REPLY"]
            assert next(iter(row["reply_to_structure"].values())) is not None
    evidence, schema, lifetime = _sanity(tmp_path, states)
    assert all(
        all(bool(value) for key, value in row.items() if key != "state_key")
        for row in evidence
    )
    assert all(bool(row["passed"]) for row in schema)
    assert all(
        all(bool(value) for key, value in row.items() if key != "state_key")
        for row in lifetime
    )


def test_call_plan_matches_states_and_repetitions():
    config = load_config(SMOKE)
    states = build_frozen_states(config, _tasks(config))
    calls = call_plan(config, states)
    assert len(calls) == 12
    assert {call.repetition for call in calls} == {0, 1}
    assert len({call.call_id for call in calls}) == 12


def test_public_contract_rejects_no_post_with_fact_and_invisible_reply():
    config = load_config(SMOKE)
    states = build_frozen_states(config, _tasks(config))
    s1 = next(
        value for value in states.values() if value.definition["state_id"] == "S1"
    )
    contract = s1.request.prompt.response_contract
    letter = next(iter(s1.definition["option_letters"]))
    fact = s1.definition["total_evidence_ids"][0]
    no_post = json.dumps(
        {
            "vote": letter,
            "reason": "private",
            "shared_fact_id": fact,
            "public_message": None,
        }
    )
    bad_reply = json.dumps(
        {
            "vote": letter,
            "reason": "private",
            "shared_fact_id": "none",
            "public_message": {
                "type": "REPLY",
                "text": "public",
                "reply_to": "missing",
            },
        }
    )
    assert not contract.validate(no_post).valid
    assert not contract.validate(bad_reply).valid
    issues = contract.validate(no_post).issues
    guidance = contract.repair_guidance(issues)
    assert '"public_message" to null and "shared_fact_id" to "none"' in guidance
    assert "Do not attach a fact to a null public message" in guidance
