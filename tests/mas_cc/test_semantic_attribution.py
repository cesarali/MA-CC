"""The semantic-attribution instrument is opt-in, samples deterministically, and labels provenance."""

from __future__ import annotations

import json

import pandas as pd

from mas_cc.analysis import semantic_attribution as sa
from mas_cc.analysis import systemone


def _fake_transport(log: list):
    def transport(payload: dict) -> dict:
        log.append(payload)
        answers = {}
        for name, question in payload["questions"].items():
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.25}
            elif question["type"] == "choice":
                options = list(question["criteria"])
                answers[name] = {"type": "choice", "choice": options[0], "confidence": 0.9,
                                 "probabilities": {o: (0.9 if i == 0 else 0.1 / max(1, len(options) - 1)) for i, o in enumerate(options)}}
            else:
                levels = question["criteria"]
                answers[name] = {"type": "score", "score": 1.0, "confidence": 0.8,
                                 "probabilities": {str(i): (1.0 if i == 1 else 0.0) for i in range(len(levels))}}
        return {"model": "jev-test", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 3}}
    return transport


def _write_episode(root, cell: str, episode: str, messages: int):
    directory = root / "runs" / cell / "round_records" / episode
    directory.mkdir(parents=True)
    rows = [{"record_type": "header", "cell_id": cell, "episode_id": episode}]
    for i in range(messages):
        rows.append({
            "record_type": "update", "cell_id": cell, "episode_id": episode, "round_index": i // 3,
            "within_round_index": i % 3, "focal_agent_id": f"agent_{i:03d}",
            "new_message": {"author_kind": "agent", "message_id": f"m{i:06d}", "message_type": "REPORT" if i % 2 else "REQUEST",
                            "text": f"Evidence item {i} about Alice." if i % 2 else "Please provide evidence about Alice."},
            "focal_vote_before": "ALLOCATION_1", "focal_vote_after": "ALLOCATION_2" if i % 4 == 0 else "ALLOCATION_1",
            "round_controller_target": "ALLOCATION_2", "correct_answer": "ALLOCATION_0",
            "possible_answers": ["ALLOCATION_0", "ALLOCATION_1", "ALLOCATION_2"],
            "controller_message_directly_exposed": i % 5 == 0, "eligible_controller_message_count": 0,
            "intervention_budget": 9, "round_controller_action": "NO_OP", "sampled_message_types": [],
        })
        rows.append({"record_type": "validation", "cell_id": cell, "episode_id": episode})
        rows.append({"record_type": "update", "cell_id": cell, "episode_id": episode, "round_index": 0,
                     "within_round_index": 9, "new_message": None})
    with open(directory / "dashboard_semantic.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_iterates_only_agent_posted_messages(tmp_path):
    _write_episode(tmp_path, "cell-0000", "cell-0000-0001", 6)
    _write_episode(tmp_path, "cell-0001", "cell-0001-0001", 4)
    messages = list(sa.iter_posted_messages(tmp_path))
    assert len(messages) == 10
    assert {m["cell_id"] for m in messages} == {"cell-0000", "cell-0001"}
    assert all(m["text"] and m["controller_target"] == "ALLOCATION_2" for m in messages)
    assert sum(m["controller_message_exposed"] for m in messages) == 3  # i in {0, 5} for six messages, {0} for four


def test_sample_is_deterministic_and_stratified(tmp_path):
    _write_episode(tmp_path, "cell-0000", "e1", 30)
    _write_episode(tmp_path, "cell-0001", "e2", 30)
    a = sa.sample_messages(sa.iter_posted_messages(tmp_path), 10, seed=3)
    b = sa.sample_messages(sa.iter_posted_messages(tmp_path), 10, seed=3)
    assert [m["message_id"] for m in a] == [m["message_id"] for m in b]
    assert len(a) == 10 and {m["cell_id"] for m in a} == {"cell-0000", "cell-0001"}


def test_attribute_dry_run_records_requests_without_calls(tmp_path):
    _write_episode(tmp_path, "cell-0000", "e1", 4)
    frame = sa.attribute(list(sa.iter_posted_messages(tmp_path)), client=None)
    assert len(frame) == 4 * 4  # four questions per message
    assert frame["value"].isna().all() and frame["model"].isna().all()
    assert set(frame["question"]) == {"cites_controller", "stance", "pressure_toward_target", "provides_evidence"}


def test_attribute_with_fake_model_and_summary(tmp_path):
    _write_episode(tmp_path, "cell-0000", "e1", 4)
    log: list = []
    client = systemone.SystemOneClient(cache_dir=tmp_path / "cache", transport=_fake_transport(log), key="k")
    messages = list(sa.iter_posted_messages(tmp_path))
    frame = sa.attribute(messages, client)
    assert len(log) == 4 and client.usage.requests == 4
    stance = frame[frame["question"] == "stance"]
    # the fake picks the first option, which is ALLOCATION_0 == correct_answer here
    assert stance["stance_matches_truth"].all() and not stance["stance_matches_target"].any()
    summary = sa.summarize(frame)
    assert {"stance_matches_target", "stance_matches_truth", "stance_none", "cites_controller", "pressure_toward_target"} <= set(summary["question"])
    assert (summary["n_messages"] == 4).all()
    # second pass is fully cached
    again = sa.attribute(messages, client)
    assert again["cached"].all() and client.usage.requests == 4


def test_cli_dry_run_writes_manifest(tmp_path):
    _write_episode(tmp_path / "study", "cell-0000", "e1", 3)
    out = tmp_path / "out"
    assert sa.main(["--study-root", str(tmp_path / "study"), "--output", str(out), "--dry-run"]) == 0
    manifest = json.loads((out / "semantic_attribution_manifest.json").read_text())
    assert manifest["provider_calls"] == 0 and manifest["dry_run"] is True and manifest["messages_judged"] == 3
    preview = json.loads((out / "dry_run_preview.json").read_text())
    assert preview[0]["questions"]["stance"]["criteria"]["none"]
    assert pd.read_parquet(out / "semantic_attribution.parquet").shape[0] == 12
