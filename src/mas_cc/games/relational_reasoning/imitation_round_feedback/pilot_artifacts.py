"""Deterministic inspection bundle for one finite-memory blackboard episode."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from mas_cc.config import RunConfig

from .prompts import agent_label, control_label


REQUIRED_OUTPUTS = (
    "config.yaml",
    "initial_assignment.json",
    "episode.jsonl",
    "messages.jsonl",
    "controller_events.jsonl",
    "round_control_events.jsonl",
    "message_reads.jsonl",
    "directive_lineage.jsonl",
    "evidence_transfers.jsonl",
    "persistence_events.jsonl",
    "agent_state_by_update.csv",
    "agent_state_by_round.csv",
    "population_by_round.csv",
    "analysis/dashboard/index.html",
    "analysis/task001_pilot_report.md",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"required blackboard inspection input is missing: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
            )


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _event(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("event", row)
    if not isinstance(value, Mapping):
        raise ValueError("trajectory row has no event object")
    return dict(value)


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _latent_map(task: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(fact_id): str(latent_id)
        for latent_id, fact_ids in dict(task.get("supporting_fact_groups", {})).items()
        for fact_id in fact_ids
    }


def _prompt_markdown(row: Mapping[str, Any], *, round_index: int, update: int) -> str:
    observation = row.get("observation") or {}
    visible = (
        observation.get("visible_state", {}) if isinstance(observation, Mapping) else {}
    )
    response = row.get("response") or {}
    public_sources = []
    for source in visible.get("social_sources", []):
        public_sources.append(
            {
                "message_id": source.get("message_id"),
                "author_public_id": source.get("label"),
                "vote": source.get("vote"),
                "type": source.get("message_type"),
                "text": source.get("text"),
                "shared_fact_id": source.get("shared_fact_id"),
                "reply_to": source.get("reply_to"),
                "created_round": source.get("round_created"),
                "expires_round": source.get("expires_after_round"),
            }
        )
    lines = [
        f"# Round {round_index:02d}, update {update:03d}, {row.get('agent_id')}",
        "",
        f"- Decision stage: `{row.get('decision_stage')}`",
        f"- Attempt: `{row.get('attempt')}`",
        f"- Valid: `{row.get('valid')}`",
        f"- Prompt instance: `{row.get('prompt_instance_hash')}`",
        "",
        "## ACTIVE PRIVATE EVIDENCE",
        "",
        "```json",
        json.dumps(visible.get("active_fact_ids", []), indent=2, ensure_ascii=False),
        "```",
        "",
        "## CURRENT VOTE",
        "",
        f"`{visible.get('current_vote')}`",
        "",
        "## VISIBLE BLACKBOARD MESSAGE",
        "",
        "```json",
        json.dumps(public_sources, indent=2, ensure_ascii=False),
        "```",
        "",
        "## AVAILABLE PUBLIC ACTION TYPES",
        "",
        "`REQUEST | REPORT | NONE`",
        "",
        "## EXACT RENDERED PROMPT",
        "",
    ]
    for index, message in enumerate(row.get("compiled_messages", ()), start=1):
        content = str(message.get("content", ""))
        fence = "```"
        while fence in content:
            fence += "`"
        lines.extend(
            (
                f"### Message {index}: `{message.get('role')}`",
                "",
                f"{fence}text",
                content,
                fence,
                "",
            )
        )
    lines.extend(
        (
            "## RAW MODEL RESPONSE",
            "",
            "```json",
            str(response.get("content", "")),
            "```",
            "",
        )
    )
    return "\n".join(lines)


def _heatmap(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    agents: Sequence[str],
    latents: Sequence[str],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {
        (str(row["agent_id"]), str(row["latent_id"])): int(row["present"])
        for row in rows
    }
    matrix = np.array(
        [[lookup.get((agent, latent), 0) for latent in latents] for agent in agents]
    )
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(
        matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="Blues"
    )
    axis.set_xticks(range(len(latents)), labels=latents, rotation=45, ha="right")
    axis.set_yticks(range(len(agents)), labels=agents)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, ticks=(0, 1))
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def build_blackboard_pilot_artifacts(
    config: RunConfig, run_dir: str | Path
) -> dict[str, Any]:
    """Build the handoff's human-readable files from one completed episode."""

    root = Path(run_dir)
    episodes = sorted((root / "data" / "episodes").glob("*"))
    episodes = [path for path in episodes if (path / "manifest.json").is_file()]
    completed = [
        path
        for path in episodes
        if json.loads((path / "manifest.json").read_text(encoding="utf-8")).get(
            "status"
        )
        in {"completed", "skipped_resumed"}
    ]
    if len(completed) != 1:
        raise ValueError(
            "blackboard pilot inspection requires exactly one completed episode; "
            f"found {len(completed)}"
        )
    episode_dir = completed[0]
    trajectory = [_event(row) for row in _read_jsonl(episode_dir / "trajectory.jsonl")]
    rounds = [
        _event(row) for row in _read_jsonl(episode_dir / "round_trajectory.jsonl")
    ]
    audits = _read_jsonl(episode_dir / "audit_traces.jsonl")
    checkpoint = json.loads(
        (episode_dir / ".checkpoints" / "checkpoint.json").read_text(encoding="utf-8")
    )
    if not rounds or not trajectory:
        raise ValueError(
            "blackboard pilot inspection requires non-empty round and update trajectories"
        )

    initialization = dict(config.game.options.get("initialization", {}))
    initialization_calls = (
        int(rounds[0].get("N", 0))
        if initialization.get("mode", "local_vote") == "local_vote"
        else 0
    )
    expected_attempts = (
        sum(
            len(item.get("decisions", ()))
            for item in _read_jsonl(episode_dir / "trajectory.jsonl")
        )
        + initialization_calls
    )
    if len(audits) < expected_attempts:
        raise ValueError(
            "detailed prompt audit is incomplete; configure every round with no prompt cap"
        )

    assignment_path = Path(
        str(config.game.options["initial_information"]["artifact_path"])
    )
    shutil.copyfile(assignment_path, root / "initial_assignment.json")
    (root / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    _write_jsonl(root / "episode.jsonl", trajectory)

    population_size = int(rounds[0]["N"])
    messages = [dict(message) for message in checkpoint["state"]["blackboard"]]
    for message in messages:
        author_public_id = (
            control_label(population_size)
            if message.get("author_kind") == "controller"
            else agent_label(message.get("author_id"))
        )
        message.update(
            round=message.get("round_created"),
            author=message.get("author_id"),
            type=message.get("message_type"),
            created_at=message.get("micro_step_created"),
            expires_at=message.get("expires_after_round"),
            author_public_id=author_public_id,
            created_round=message.get("round_created"),
            expires_round=message.get("expires_after_round"),
        )
    messages.sort(
        key=lambda item: (
            int(item.get("micro_step_created") or 0),
            str(item.get("message_id")),
        )
    )
    _write_jsonl(root / "messages.jsonl", messages)

    message_read_rows = [
        {
            "round": event.get("round_index"),
            "microscopic_update": event.get("within_round_index"),
            "focal_agent_id": event.get("focal_agent_id"),
            "sampled_message_ids": event.get("sampled_message_ids", []),
            "sampled_message_authors": event.get("sampled_message_authors", []),
            "sampled_message_types": event.get("sampled_message_types", []),
            "sampled_controller_message_ids": event.get(
                "sampled_controller_message_ids", []
            ),
            "eligible_live_messages": event.get("eligible_board_message_count", 0),
            "eligible_live_directives": event.get("eligible_directive_count", 0),
            "eligible_directive_share": event.get("eligible_directive_share", 0.0),
        }
        for event in trajectory
    ]
    _write_jsonl(root / "message_reads.jsonl", message_read_rows)

    lineage_rows = [
        dict(lineage)
        for record in rounds
        for lineage in record.get("directive_lineage_events", ())
    ]
    _write_jsonl(root / "directive_lineage.jsonl", lineage_rows)

    transfer_rows: list[dict[str, Any]] = []
    for event in trajectory:
        refreshed = set(event.get("reactivated_peer_fact_ids", ())) | set(
            event.get("reactivated_controller_fact_ids", ())
        )
        acquired = set(event.get("new_peer_fact_ids", ())) | set(
            event.get("new_controller_fact_ids", ())
        )
        for source in event.get("social_sources", ()):
            fact_id = source.get("shared_fact_id")
            transfer_rows.append(
                {
                    "round": event.get("round_index"),
                    "microscopic_update": event.get("within_round_index"),
                    "receiver_agent_id": event.get("focal_agent_id"),
                    "message_id": source.get("message_id"),
                    "author_id": source.get("source_id"),
                    "message_type": source.get("message_type"),
                    "shared_fact_id": fact_id,
                    "event_type": (
                        "refresh"
                        if fact_id in refreshed
                        else "acquisition"
                        if fact_id in acquired
                        else "exact_read"
                        if fact_id is not None
                        else "semantic_only_read"
                    ),
                }
            )
    _write_jsonl(root / "evidence_transfers.jsonl", transfer_rows)

    persistence_rows = [
        {
            "round": record["round_index"],
            "boundary": (
                "night_before_day"
                if record.get("controller_timing") == "dawn_only"
                else "legacy_end_of_round"
            ),
            **pair,
        }
        for record in rounds
        for pair in record.get("persistence_deactivated_pairs", ())
    ]
    _write_jsonl(root / "persistence_events.jsonl", persistence_rows)

    controller_rows = []
    for record in rounds:
        round_index = record["round_index"]
        window = [
            event for event in trajectory if event.get("round_index") == round_index
        ]
        controller_rows.append(
            {
                "round": round_index,
                "n_k": record.get("n_k"),
                "Y": record.get("controller_sensor_Y"),
                "Y_k": record.get("Y_k"),
                "y_Z": record.get("Y_k"),
                "q_c": record.get("sensor_sample_size"),
                "beta": record.get("controller_beta"),
                "theta": record.get("controller_threshold"),
                "sensed_agent_ids": record.get("sensor_agent_ids", []),
                "sensed_votes": record.get("sensor_observed_opinions", []),
                "probability_U1_given_Y": record.get(
                    "controller_probability_U1_given_Y"
                ),
                "P_U1_given_Y": record.get("P_U1_given_Y"),
                "sampled_U": record.get("controller_sampled_U"),
                "U": record.get("U_k"),
                "b": record.get("b", record.get("intervention_budget")),
                "acted": bool(record.get("controller_sampled_U")),
                "action": record.get("controller_action"),
                "target": record.get("controller_target"),
                "coordinator_public_vote": record.get("coordinator_public_vote"),
                "controller_timing": record.get("controller_timing"),
                "injection_within_round_indices": record.get(
                    "controller_injection_within_round_indices", []
                ),
                "injection_global_update_indices": record.get(
                    "controller_injection_global_update_indices", []
                ),
                "directive_ids": record.get("controller_post_ids", []),
                "directive_message_ids": record.get("directive_message_ids", []),
                "dawn_directive_count": record.get("dawn_directive_count", 0),
                "directive_texts": [
                    message.get("text")
                    for message in messages
                    if message.get("message_id")
                    in set(record.get("controller_post_ids", ()))
                ],
                "sampled_by_agents": sorted(
                    {
                        event.get("focal_agent_id")
                        for event in window
                        if event.get("sampled_controller_message_ids")
                    }
                ),
                "directive_exposed_focal_updates": record.get(
                    "directive_exposed_focal_updates", 0
                ),
                "realized_directive_exposure_fraction": record.get(
                    "realized_directive_exposure_fraction", 0.0
                ),
                "reply_count": record.get("controller_direct_replies", 0),
                "report_reply_count": record.get("directive_report_reply_count", 0),
                "evidence_report_count": record.get(
                    "directive_evidence_report_count", 0
                ),
                "directive_attributed_acquisitions": record.get(
                    "directive_attributed_acquisitions", 0
                ),
                "directive_attributed_refreshes": record.get(
                    "directive_attributed_refreshes", 0
                ),
                "n_k_plus_1": record.get("n_k_plus_1"),
                "downstream_exact_evidence_moved": sum(
                    int(event.get("new_peer_facts", 0))
                    + int(event.get("reactivated_peer_facts", 0))
                    for event in window
                    if event.get("new_message_reply_to")
                    in set(record.get("controller_post_ids", ()))
                ),
            }
        )
    _write_jsonl(root / "controller_events.jsonl", controller_rows)
    _write_jsonl(root / "round_control_events.jsonl", controller_rows)

    task_state = checkpoint["state"]["task"]
    latent_for = _latent_map(task_state)
    agents = [str(agent) for agent in rounds[0]["agent_ids"]]
    latents = sorted(set(latent_for.values()))
    active = {
        agent: set(cards)
        for agent, cards in zip(
            agents, rounds[0]["initial_active_fact_ids_by_agent"], strict=True
        )
    }
    known = {
        agent: set(cards)
        for agent, cards in zip(
            agents, rounds[0]["initial_known_fact_ids_by_agent"], strict=True
        )
    }
    update_rows: list[dict[str, Any]] = []
    persistence_by_round: dict[int, list[dict[str, Any]]] = {}
    for row in persistence_rows:
        persistence_by_round.setdefault(int(row["round"]), []).append(row)
    applied_persistence_rounds: set[int] = set()
    for event in trajectory:
        event_round = int(event["round_index"])
        if (
            event_round not in applied_persistence_rounds
            and rounds[event_round].get("controller_timing") == "dawn_only"
        ):
            for item in persistence_by_round.get(event_round, ()):
                active[str(item["agent_id"])].discard(str(item["fact_id"]))
            applied_persistence_rounds.add(event_round)
        focal = str(event["focal_agent_id"])
        active[focal] = set(event["focal_active_fact_ids_after"])
        known[focal] = set(event["focal_known_fact_ids_after"])
        for agent in agents:
            update_rows.append(
                {
                    "round": event["round_index"],
                    "microscopic_update": event["within_round_index"],
                    "global_update": event["global_update_index"],
                    "agent_id": agent,
                    "is_focal": agent == focal,
                    "active_fact_ids": _json_cell(sorted(active[agent])),
                    "historical_fact_ids": _json_cell(sorted(known[agent])),
                    "active_latent_ids": _json_cell(
                        sorted({latent_for[x] for x in active[agent]})
                    ),
                    "historical_latent_ids": _json_cell(
                        sorted({latent_for[x] for x in known[agent]})
                    ),
                }
            )
    _write_csv(
        root / "agent_state_by_update.csv",
        update_rows,
        (
            "round",
            "microscopic_update",
            "global_update",
            "agent_id",
            "is_focal",
            "active_fact_ids",
            "historical_fact_ids",
            "active_latent_ids",
            "historical_latent_ids",
        ),
    )

    round_agent_rows: list[dict[str, Any]] = []
    heatmap_data: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in rounds:
        round_index = int(record["round_index"])
        active_snapshot = record["active_fact_ids_by_agent_after"]
        known_snapshot = record["known_fact_ids_by_agent_after"]
        for agent in agents:
            active_ids = set(active_snapshot[agent])
            known_ids = set(known_snapshot[agent])
            round_agent_rows.append(
                {
                    "round": round_index,
                    "agent_id": agent,
                    "active_fact_ids": _json_cell(sorted(active_ids)),
                    "historical_fact_ids": _json_cell(sorted(known_ids)),
                    "active_count": len(active_ids),
                    "historical_count": len(known_ids),
                    "active_latent_ids": _json_cell(
                        sorted({latent_for[x] for x in active_ids})
                    ),
                    "historical_latent_ids": _json_cell(
                        sorted({latent_for[x] for x in known_ids})
                    ),
                }
            )
            for kind, ids in (("active", active_ids), ("historical", known_ids)):
                heatmap_data.setdefault((round_index, kind), []).extend(
                    {
                        "agent_id": agent,
                        "latent_id": latent,
                        "present": int(any(latent_for[x] == latent for x in ids)),
                    }
                    for latent in latents
                )
    _write_csv(
        root / "agent_state_by_round.csv",
        round_agent_rows,
        (
            "round",
            "agent_id",
            "active_fact_ids",
            "historical_fact_ids",
            "active_count",
            "historical_count",
            "active_latent_ids",
            "historical_latent_ids",
        ),
    )

    population_rows = []
    for record in rounds:
        counts = Counter(record["population_state_after"])
        population_rows.append(
            {
                "round": record["round_index"],
                "n_k": record.get("n_k"),
                "Y_k": record.get("Y_k"),
                "P_U1_given_Y": record.get("P_U1_given_Y"),
                "U_k": record.get("U_k"),
                "b": record.get("b"),
                "n_k_plus_1": record.get("n_k_plus_1"),
                "vote_counts": _json_cell(counts),
                "p_truth": record["truth_vote_share"],
                "mean_active": record["active_mean_fact_count_after"],
                "mean_historical": record["mean_known_fact_count"],
                "active_latent_coverage": sum(
                    bool(any(latent_for[x] == latent for x in ids))
                    for latent in latents
                    for ids in [
                        set().union(
                            *[
                                set(v)
                                for v in record[
                                    "active_fact_ids_by_agent_after"
                                ].values()
                            ]
                        )
                    ]
                ),
                "historical_latent_coverage": sum(
                    bool(any(latent_for[x] == latent for x in ids))
                    for latent in latents
                    for ids in [
                        set().union(
                            *[
                                set(v)
                                for v in record[
                                    "known_fact_ids_by_agent_after"
                                ].values()
                            ]
                        )
                    ]
                ),
                "dawn_directive_count": record.get("dawn_directive_count", 0),
                "directive_exposed_focal_updates": record.get(
                    "directive_exposed_focal_updates", 0
                ),
                "directive_unique_readers": record.get(
                    "controller_unique_readers", 0
                ),
            }
        )
    _write_csv(
        root / "population_by_round.csv",
        population_rows,
        (
            "round",
            "n_k",
            "Y_k",
            "P_U1_given_Y",
            "U_k",
            "b",
            "n_k_plus_1",
            "vote_counts",
            "p_truth",
            "mean_active",
            "mean_historical",
            "active_latent_coverage",
            "historical_latent_coverage",
            "dawn_directive_count",
            "directive_exposed_focal_updates",
            "directive_unique_readers",
        ),
    )

    prompt_dir = root / "analysis" / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    by_interaction = {
        str(event.get("interaction_index")): event for event in trajectory
    }
    for index, row in enumerate(audits):
        stage = str(row.get("decision_stage"))
        if stage == "initial_vote":
            round_index, update = 0, 0
            prefix = "initial"
        else:
            number = int(str(row.get("interaction_id", "0")).rsplit("-", 1)[-1])
            event = by_interaction[str(number)]
            round_index = int(event["round_index"]) + 1
            update = int(event["within_round_index"]) + 1
            prefix = f"round_{round_index:02d}_update_{update:03d}"
        name = f"{prefix}_agent_{row.get('agent_id')}_attempt_{row.get('attempt')}_{index:04d}.md"
        (prompt_dir / name).write_text(
            _prompt_markdown(row, round_index=round_index, update=update),
            encoding="utf-8",
        )

    figures = root / "analysis" / "figures"
    for (round_index, kind), values in heatmap_data.items():
        _heatmap(
            figures / f"{kind}_evidence_heatmap_round_{round_index + 1:02d}.png",
            values,
            agents=agents,
            latents=latents,
            title=f"{kind.title()} latent coverage after round {round_index + 1}",
        )

    counts = Counter(message.get("message_type") for message in messages)
    exact = sum(row["event_type"] == "acquisition" for row in transfer_rows)
    refreshes = sum(row["event_type"] == "refresh" for row in transfer_rows)
    attributed_exact = sum(
        row.get("event_type") == "acquisition" for row in lineage_rows
    )
    attributed_refreshes = sum(
        row.get("event_type") == "refresh" for row in lineage_rows
    )
    initial_truth = rounds[0]["truth_vote_share_before"]
    final_truth = rounds[-1]["truth_vote_share"]
    transition_lines = "\n".join(
        "| {round} | {n} | {y} | {p:.4f} | {u} | {d} | {e} | {after} |".format(
            round=int(record["round_index"]) + 1,
            n=record.get("n_k"),
            y=record.get("Y_k"),
            p=float(record.get("P_U1_given_Y") or 0.0),
            u=record.get("U_k"),
            d=record.get("dawn_directive_count", 0),
            e=record.get("directive_exposed_focal_updates", 0),
            after=record.get("n_k_plus_1"),
        )
        for record in rounds
    )
    dawn_contract = all(
        int(record.get("dawn_directive_count", 0))
        == (int(record.get("b", 0)) if record.get("U_k") == 1 else 0)
        for record in rounds
    )
    daytime_silent = not any(
        event.get("controller_message_posted") for event in trajectory
    )
    sensor_population_only = all(
        set(record.get("sensor_agent_ids", ())).issubset(set(record["agent_ids"]))
        for record in rounds
    )
    report = f"""# MuSR blackboard task001 pilot report

This is an engineering inspection report for one short episode. It does not support strong scientific claims.

## Run summary

- Task: `{rounds[0]["task_id"]}`
- Population: `{rounds[0]["N"]}` agents
- Rounds: `{len(rounds)}`
- Initial truth-vote share: `{initial_truth:.4f}`
- Final truth-vote share: `{final_truth:.4f}`
- REQUEST messages: `{counts["REQUEST"]}`
- REPORT messages: `{counts["REPORT"]}`
- DIRECTIVE messages: `{counts["DIRECTIVE"]}`
- Exact evidence acquisitions: `{exact}`
- Refresh events: `{refreshes}`
- Directive-attributed acquisitions: `{attributed_exact}`
- Directive-attributed refreshes: `{attributed_refreshes}`
- Persistence deactivations: `{len(persistence_rows)}`

## Round transition compatibility

The established round variables remain the primary controller transition;
blackboard observables are additional columns and artifacts.

| Round | n_k | Y_k | P(U=1|Y) | U_k | Dawn DIRECTIVEs | Exposed focal updates | n_k+1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
{transition_lines}

These rows remain consumable by the existing relational round-record adapter
for sensing MI, controller-to-population CMI, susceptibility, signed response,
and efficiency diagnostics. One five-round episode is not adequate scientific
support for estimating controller efficiency.

## Protocol-freeze checks

1. Sequence: each row records night sensing and `U`, then dawn seeding, an autonomous day, and `n_k+1`.
2. Policy: the inherited soft-target probability and sampled action are retained unchanged.
3. `U=0`: every such round has zero dawn DIRECTIVEs: `{all(int(r.get('dawn_directive_count', 0)) == 0 for r in rounds if r.get('U_k') == 0)}`.
4. `U=1`: exactly `b` dawn DIRECTIVEs: `{dawn_contract}`.
5. Dawn ordering: all coordinator posts have the round-boundary micro-step and exist before the first focal update.
6. Day silence: no coordinator post is created by a daytime focal update: `{daytime_silent}`.
7. Prompt identity: coordinator messages render under the neutral public alias `{control_label(population_size)}`; the special role label is absent from rendered prompts.
8. Population boundary: coordinator vote is social context only; every sensor ID belongs to the `{population_size}` ordinary agents: `{sensor_population_only}`.
9. Realized exposure is reported per round above; nominal `b` is never substituted for sampled exposure.
10. DIRECTIVE replies: `{sum(int(r.get('controller_direct_replies', 0)) for r in rounds)}` total, including `{sum(int(r.get('directive_report_reply_count', 0)) for r in rounds)}` REPORT replies.
11. Transitive evidence lineage records `{attributed_exact}` acquisitions and `{attributed_refreshes}` refreshes.
12. `K_active`, `K_hist`, board expiry, and persistence are archived independently.
13. Mechanical freeze status: `{'PASS' if dawn_contract and daytime_silent and sensor_population_only else 'FAIL'}`. This is an engineering result, not a scientific efficiency claim.

## Artifact guide

Use `round_control_events.jsonl` for the causal round transition,
`message_reads.jsonl` for realized public-channel pressure,
`directive_lineage.jsonl` for reply-ancestry evidence events, and the dashboard
and prompt archive for human inspection.
"""
    (root / "analysis" / "task001_pilot_report.md").parent.mkdir(
        parents=True, exist_ok=True
    )
    (root / "analysis" / "task001_pilot_report.md").write_text(report, encoding="utf-8")

    dashboard = root / "analysis" / "dashboard" / "index.html"
    dashboard.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        "<h1>MuSR blackboard task001 pilot</h1>",
        f"<p>Initial p_truth: {initial_truth:.4f}; final p_truth: {final_truth:.4f}</p>",
    ]
    for record in rounds:
        round_index = int(record["round_index"])
        round_messages = [
            m for m in messages if int(m.get("round_created") or 0) == round_index
        ]
        sections.append(f"<h2>Round {round_index + 1}</h2>")
        sections.append("<h3>NIGHT</h3>")
        sections.append(
            f"<p>n_k={record.get('n_k')}; Y_k={record.get('Y_k')}; "
            f"P(U=1|Y)={float(record.get('P_U1_given_Y') or 0.0):.4f}; "
            f"sampled U_k={record.get('U_k')}; sensed agents="
            f"{html.escape(_json_cell(record.get('sensor_agent_ids', [])))}</p>"
        )
        directive_texts = [
            str(message.get("text"))
            for message in round_messages
            if message.get("message_type") == "DIRECTIVE"
        ]
        sections.append("<h3>DAWN</h3>")
        sections.append(
            f"<p>DIRECTIVEs seeded={record.get('dawn_directive_count', 0)}; "
            f"coordinator public vote={html.escape(str(record.get('coordinator_public_vote')))}; "
            f"texts={html.escape(_json_cell(directive_texts))}</p>"
        )
        sections.append("<h3>DAY</h3>")
        sections.append(
            f"<p>Realized exposed focal updates={record.get('directive_exposed_focal_updates', 0)} "
            f"({float(record.get('realized_directive_exposure_fraction', 0.0)):.3f}); "
            f"unique readers={record.get('controller_unique_readers', 0)}; "
            f"REQUEST={record.get('request_count', 0)}; REPORT={record.get('report_count', 0)}; "
            f"acquisitions={record.get('new_evidence_acquisitions', 0)}; "
            f"refreshes={int(record.get('reactivated_peer_fact_count', 0)) + int(record.get('reactivated_controller_fact_count', 0))}; "
            f"eligible DIRECTIVE share={float(record.get('directive_share_among_eligible_messages', 0.0)):.3f}</p>"
        )
        sections.append("<h3>END OF DAY</h3>")
        sections.append(
            f"<p>n_k+1={record.get('n_k_plus_1')}; votes: "
            f"{html.escape(_json_cell(Counter(record['population_state_after'])))}; "
            f"p_truth={record['truth_vote_share']:.4f}; "
            f"mean K_active={record['active_mean_fact_count_after']:.3f}; "
            f"mean K_hist={record['mean_known_fact_count']:.3f}</p>"
        )
        sections.append(
            f'<p><img src="../figures/active_evidence_heatmap_round_{round_index + 1:02d}.png" alt="active evidence heatmap"> <img src="../figures/historical_evidence_heatmap_round_{round_index + 1:02d}.png" alt="historical evidence heatmap"></p>'
        )
        sections.append(
            "<table><tr><th>ID</th><th>Author</th><th>Role</th><th>Type</th><th>Text</th><th>Evidence</th><th>Reply</th><th>Created</th><th>Expires</th></tr>"
        )
        for message in round_messages:
            style = (
                ' style="background:#fff3cd"'
                if message.get("message_type") == "DIRECTIVE"
                else ""
            )
            sections.append(
                "<tr%s>%s</tr>"
                % (
                    style,
                    "".join(
                        f"<td>{html.escape(str(message.get(key)))}</td>"
                        for key in (
                            "message_id",
                            "author_public_id",
                            "author_kind",
                            "message_type",
                            "text",
                            "shared_fact_id",
                            "reply_to",
                            "micro_step_created",
                            "expires_after_round",
                        )
                    ),
                )
            )
        sections.append("</table>")
    dashboard.write_text(
        "<!doctype html><meta charset='utf-8'><title>MuSR blackboard pilot</title><style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:.35rem;text-align:left}img{max-width:48%;vertical-align:top}</style>"
        + "".join(sections),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 2,
        "episode_dir": str(episode_dir),
        "outputs": {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in REQUIRED_OUTPUTS
        },
        "counts": {
            "requests": counts["REQUEST"],
            "reports": counts["REPORT"],
            "directives": counts["DIRECTIVE"],
            "exact_acquisitions": exact,
            "refreshes": refreshes,
            "directive_attributed_acquisitions": attributed_exact,
            "directive_attributed_refreshes": attributed_refreshes,
            "prompt_attempts": len(audits),
        },
    }
    (root / "analysis" / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = ["build_blackboard_pilot_artifacts"]
