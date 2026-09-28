#!/usr/bin/env python
"""Stream the four completed Task 004 suites as compact Phoenix import inputs.

Run this on Cesar from /shared/MA-CC. It writes one gzip tar stream to stdout;
redirect that stream to a local file over SSH. No provider calls are made.
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys
import tarfile
from typing import Any

from mas_cc.llm_runtime.tracing import ROUND_ATTRIBUTES


ARMS = (
    "false_allfacts_rho075",
    "false_allfacts_rho1",
    "truth_allfacts_rho075",
    "truth_allfacts_rho1",
)
SUITES = {
    "task004-episode-b90-report-only": {
        "false_allfacts_rho075": "task004-false-allfacts-rho075-b90-20260925",
        "false_allfacts_rho1": "task004-allfacts-b90-parallel-20260925",
        "truth_allfacts_rho075": "task004-allfacts-b90-parallel-20260925",
        "truth_allfacts_rho1": "task004-truth-allfacts-rho1-b90-20260925",
    },
    "task004-episode-b90-full-comms": {
        arm: "task004-allfacts-b90-full-communication-suite-20260925"
        for arm in ARMS
    },
    "task004-round-b3-report-only": {
        arm: "task004-allfacts-b3-soft-report-only-suite-20260926"
        for arm in ARMS
    },
    "task004-round-b3-full-comms": {
        arm: "task004-allfacts-b3-soft-full-communication-suite-20260926"
        for arm in ARMS
    },
}
ROUND_FIELDS = set(ROUND_ATTRIBUTES) | {
    "controller_action",
    "controller_action_probability",
    "controller_budget_scope",
    "controller_llm_attempts",
    "controller_report_fact_ids",
    "controller_round_budget_mode",
    "controller_target",
    "communication_profile",
    "epistemic_persistence",
    "intervention_budget",
    "correct_answer",
    "chosen_message_mode",
    "absolute_round",
}


def _add(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return ("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)).encode()


def _slim_trajectory(path: pathlib.Path) -> bytes:
    rows = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        decisions = []
        for decision in row.get("decisions") or []:
            action = decision.get("action") or decision
            metadata = action.get("metadata") or {}
            decisions.append({
                "action": {
                    "agent_id": action.get("agent_id"),
                    "value": action.get("value"),
                    "metadata": {
                        key: metadata.get(key)
                        for key in ("reason", "shared_fact_id", "public_message")
                    },
                }
            })
        rows.append({
            "interaction_id": row.get("interaction_id"),
            "interaction_index": row.get("interaction_index"),
            "decisions": decisions,
        })
    return _jsonl(rows)


def _slim_rounds(path: pathlib.Path) -> tuple[bytes, int]:
    rows = []
    prompts = 0
    for line in path.open(encoding="utf-8"):
        raw = json.loads(line)
        row = raw.get("event", raw)
        rows.append({key: row.get(key) for key in ROUND_FIELDS})
        prompts += sum(
            bool((attempt.get("request") or {}).get("messages"))
            for attempt in row.get("controller_llm_attempts") or []
        )
    if len(rows) != 30 or [row["absolute_round"] for row in rows] != list(range(1, 31)):
        raise ValueError(f"expected 30 complete rounds in {path}")
    return _jsonl(rows), prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=pathlib.Path,
                        default=pathlib.Path("/shared/MA-CC-results"))
    parser.add_argument("--suite", choices=tuple(SUITES), action="append",
                        help="export only selected suites; default: all four")
    args = parser.parse_args()
    chosen = args.suite or list(SUITES)
    manifest: dict[str, Any] = {"schema_version": 1, "suites": {}}
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
        for suite in chosen:
            suite_info: dict[str, Any] = {}
            for arm, study in SUITES[suite].items():
                root = args.results_root / study / "runs" / arm
                episodes = sorted(root.glob("**/data/episodes/*"))
                complete = [
                    episode for episode in episodes
                    if (episode / "manifest.json").is_file()
                    and json.loads((episode / "manifest.json").read_text()).get("status")
                    == "completed"
                ]
                if len(complete) != 10:
                    raise ValueError(f"{suite}/{arm}: expected 10 completed episodes, found {len(complete)}")
                arm_info = {"source_study": study, "episodes": []}
                for episode in complete:
                    prefix = f"{suite}/{arm}/{episode.name}"
                    rounds, controller_prompts = _slim_rounds(
                        episode / "round_trajectory.jsonl"
                    )
                    _add(archive, f"{prefix}/round_trajectory.jsonl", rounds)
                    _add(archive, f"{prefix}/trajectory.jsonl", _slim_trajectory(
                        episode / "trajectory.jsonl"
                    ))
                    for filename in ("api_call_status.jsonl", "usage_cost.jsonl"):
                        _add(archive, f"{prefix}/{filename}",
                             (episode / filename).read_bytes())
                    prompt_rows = [
                        {"name": prompt.name, "content": prompt.read_text(encoding="utf-8")}
                        for prompt in sorted((episode / "prompts").glob("round_*.md"))
                    ]
                    _add(archive, f"{prefix}/prompts.jsonl", _jsonl(prompt_rows))
                    arm_info["episodes"].append({
                        "name": episode.name,
                        "agent_prompt_artifacts": len(prompt_rows),
                        "controller_prompts": controller_prompts,
                    })
                suite_info[arm] = arm_info
            manifest["suites"][suite] = suite_info
        _add(archive, "bundle_manifest.json", json.dumps(manifest, indent=2).encode())


if __name__ == "__main__":
    main()
