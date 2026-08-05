"""Phase 6 scientific-game inspection workflow."""

from __future__ import annotations

import csv
import io
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from mas_cc.config import RunConfig, resolved_config_yaml
from mas_cc.core import AgentId
from mas_cc.games import Action, create_game
from mas_cc.games.naming_convention import (
    NamingConventionGame,
    run_naming_convention_game_sync,
)
from mas_cc.llm_runtime.providers import (
    BudgetGuardedProvider,
    BudgetLimits,
    PricingQuote,
    RuntimeBudgetGuard,
    create_llm_provider,
    resolve_budget_limits,
)
from mas_cc.planning import estimate_input_tokens, static_game_preflight
from mas_cc.llm_runtime.providers import CompletionRequest
from mas_cc.llm_runtime.prompts import RegexTokenCounter

from .inspect import _write, _write_manifest


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _jsonl(values: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n" for value in values
    )


def _frozen_prompt_wire_parity() -> bool:
    from mas_cc.games.naming_convention.prompts import bind_naming_convention_prompt

    for order in (("Q", "M"), ("M", "Q")):
        prompt = bind_naming_convention_prompt(
            presented_actions=order,
            visible_memory=(),
            visible_score=0,
            local_round=1,
        ).compile()
        expected_system = (
            "Player 1 is playing a repeated two-player partnership game with Player 2. "
            "In each round both players choose simultaneously.\n"
            "If both players choose the same value, both receive +100 points.\n"
            "If the players choose different values, both receive -50 points.\n"
            "Player 1 cannot see Player 2's current choice before deciding.\n"
            "The available values, in the order presented for this decision, are: "
            f"{json.dumps(list(order))}.\n"
            "Your objective is to maximize Player 1's accumulated points conditional "
            "on Player 2's behavior.\n\n"
            "Player 1 has no past rounds available in memory.\n\n"
            "It is now local round 1. Player 1's score over the visible memory window is 0.\n\n"
            "Put the decision before the explanation. Return only an object in this "
            "answer-first form: {'value': '<ACTION>'; 'reason': '<YOUR REASON>'}. "
            f"The value must be exactly {' or '.join(order)}. "
            "Do not add text outside the object."
        )
        if [message.content for message in prompt.messages] != [
            expected_system,
            "Answer saying which action Player 1 should play.",
        ]:
            return False
    return True


def _baseline_selected_wire_parity(selected_rows: list[dict[str, Any]]) -> bool:
    """Compare every selected V3 role/content pair with the frozen V2 traces."""

    baseline = (
        Path(__file__).resolve().parents[3]
        / "inspection"
        / "realignment_v3"
        / "baseline"
        / "phase_06"
        / "selected_audit_traces.jsonl"
    )
    if not baseline.is_file():
        return False
    frozen_rows = [
        json.loads(line)
        for line in baseline.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(frozen_rows) != len(selected_rows):
        return False
    current_by_index = {row["interaction_index"]: row for row in selected_rows}
    for frozen in frozen_rows:
        current = current_by_index.get(frozen["interaction_index"])
        if current is None:
            return False
        if frozen["presented_action_orders"] != current["presented_action_orders"]:
            return False
        for player in ("player_1", "player_2"):
            frozen_messages = [
                (message["role"], message["content"])
                for message in frozen["decisions"][player]["compiled_messages"]
            ]
            current_messages = [
                (message["role"], message["content"])
                for message in current["decisions"][player]["compiled_messages"]
            ]
            if frozen_messages != current_messages:
                return False
    return True


def _prompt_token_scenarios(
    game: NamingConventionGame, config: RunConfig
) -> tuple[str, list[dict[str, Any]]]:
    plan = game.call_plan(config.game)
    stage = plan.decision_stages[0]
    counter = RegexTokenCounter()
    rows: list[dict[str, Any]] = []
    for scenario in stage.prompt_scenarios:
        prompt = scenario.bound_prompt.compile(counter)
        request = CompletionRequest(
            prompt.messages,
            temperature=config.llm_provider.temperature,
            max_output_tokens=config.llm_provider.max_output_tokens,
        )
        canonical = json.dumps(
            request.wire_messages(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        import hashlib

        rows.append(
            {
                "scenario": scenario.name,
                "visible_memory_entries": len(
                    scenario.bound_prompt.block("visible_memory").value
                ),
                "presented_actions": json.dumps(
                    list(scenario.bound_prompt.block("presented_actions").value),
                    ensure_ascii=False,
                ),
                "tokenizer": "mas_cc_regex_v1_estimate",
                "estimated_input_tokens": estimate_input_tokens(request),
                "prompt_hash_sha256": hashlib.sha256(canonical).hexdigest(),
                "assumptions": " | ".join(scenario.assumptions),
            }
        )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue(), rows


def _trajectory(result, actions: tuple[str, ...], population_size: int) -> tuple[str, list[dict[str, Any]]]:
    output = io.StringIO(newline="")
    fieldnames = [
        "interaction_index",
        "population_round_project_convention",
        "agent_i",
        "agent_j",
        "action_i",
        "action_j",
        "success",
        "payoff",
        "rolling_window_interactions",
        "rolling_coordination_rate",
        *(f"rolling_share_{action}" for action in actions),
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    rows: list[dict[str, Any]] = []
    for index, interaction in enumerate(result.interactions, start=1):
        window = result.interactions[max(0, index - population_size) : index]
        window_actions = [
            decision.action.value for item in window for decision in item.decisions
        ]
        counts = Counter(window_actions)
        row: dict[str, Any] = {
            "interaction_index": index,
            "population_round_project_convention": index / population_size,
            "agent_i": str(interaction.selected_agents[0]),
            "agent_j": str(interaction.selected_agents[1]),
            "action_i": interaction.decisions[0].action.value,
            "action_j": interaction.decisions[1].action.value,
            "success": interaction.transition.success,
            "payoff": interaction.transition.payoff,
            "rolling_window_interactions": len(window),
            "rolling_coordination_rate": sum(item.transition.success for item in window)
            / len(window),
        }
        row.update(
            {
                f"rolling_share_{action}": counts[action] / len(window_actions)
                for action in actions
            }
        )
        rows.append(row)
        writer.writerow(row)
    return output.getvalue(), rows


def _plots(rows: list[dict[str, Any]], actions: tuple[str, ...], output_dir: Path, window: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    x = [row["interaction_index"] for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4), dpi=120)
    for action in actions:
        axis.plot(
            x,
            [row[f"rolling_share_{action}"] for row in rows],
            marker="o",
            label=action,
        )
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("Global pair interaction (evaluator only)")
    axis.set_ylabel("Rolling action share")
    axis.set_title(f"Action shares over the previous up to {window} interactions")
    axis.legend(title="Action")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "action_share.png", metadata={"Software": "MAS-CC"})
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4), dpi=120)
    axis.plot(
        x,
        [row["rolling_coordination_rate"] for row in rows],
        marker="o",
        color="tab:green",
    )
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("Global pair interaction (evaluator only)")
    axis.set_ylabel("Rolling coordination rate")
    axis.set_title(f"Coordination over the previous up to {window} interactions")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "coordination_rate.png", metadata={"Software": "MAS-CC"})
    plt.close(figure)


def _legacy_parity_fixture(game: NamingConventionGame, config: RunConfig) -> bool:
    """Compare two fixed transitions with the read-only legacy state records."""

    from naming_game.naming_convention_game import ConventionAgent

    fixture_config = replace(config.game, population_size=2, horizon=2)
    state = game.initialize(fixture_config, seed=7)
    pair = (AgentId("agent-000"), AgentId("agent-001"))
    legacy = (ConventionAgent(0), ConventionAgent(1))
    for interaction_index, values in enumerate((("Q", "Q"), ("Q", "M")), start=1):
        actions = (
            Action(pair[0], values[0], "pair_decision"),
            Action(pair[1], values[1], "pair_decision"),
        )
        transition = game.apply_transition(state, pair, actions, fixture_config)
        payoff = 100 if values[0] == values[1] else -50
        legacy[0].remember(
            interaction_index=interaction_index,
            own_action=values[0],
            partner_action=values[1],
            payoff=payoff,
            partner_id=1,
        )
        legacy[1].remember(
            interaction_index=interaction_index,
            own_action=values[1],
            partner_action=values[0],
            payoff=payoff,
            partner_id=0,
        )
        state = transition.next_state
    for index, agent_id in enumerate(pair):
        new_agent = state.convention_agent(agent_id)
        if new_agent.lifetime_score != legacy[index].score:
            return False
        normalized_new = [
            (entry.own_action, entry.partner_action, entry.payoff, entry.success)
            for entry in new_agent.private_history
        ]
        normalized_legacy = [
            (entry.own_action, entry.partner_action, entry.payoff, entry.success)
            for entry in legacy[index].history
        ]
        if normalized_new != normalized_legacy:
            return False
    return True


def run_naming_convention_inspection(
    config: RunConfig,
    source: Path,
    destination: Path,
    *,
    quote: PricingQuote,
    runtime_quote: PricingQuote,
    system_budget: BudgetLimits,
    run_budget: BudgetLimits,
) -> bool:
    game = create_game(config.game)
    if not isinstance(game, NamingConventionGame):
        raise TypeError("naming_convention registry returned an incompatible game")
    rules = game.rules(config.game)
    plan = game.call_plan(config.game)
    preflight = static_game_preflight(
        plan,
        config.prompt,
        config.llm_provider,
        assumed_output_tokens=config.llm_provider.max_output_tokens,
        pricing_quote=quote,
        system_budget=system_budget,
        run_budget=run_budget,
        explicit_override=config.pricing.explicit_unknown_price_override,
        allow_stale_pricing=not config.pricing.require_fresh_at_launch,
    )
    if preflight.launch_status != "permitted":
        raise ValueError(
            f"game preflight launch status is {preflight.launch_status!r}; no provider calls sent"
        )
    guard = RuntimeBudgetGuard(resolve_budget_limits(system_budget, run_budget))
    provider = create_llm_provider(config.llm_provider)
    guarded = BudgetGuardedProvider(
        provider,
        guard,
        runtime_quote.pricing,
        input_token_estimator=estimate_input_tokens,
        input_token_multiplier=1.0,
    )
    try:
        result = run_naming_convention_game_sync(game, config, guarded)
    finally:
        guarded.close()

    interaction_rows = [interaction.to_dict() for interaction in result.interactions]
    selected_indices = tuple(
        int(item) for item in config.game.options.get("selected_audit_interactions", (1,))
    )
    selected_rows = [
        row for row in interaction_rows if row["interaction_index"] in selected_indices
    ]
    selected_block_rows = [
        {
            "interaction_index": row["interaction_index"],
            "agent_id": decision["agent_id"],
            "definition_hash": decision["prompt_definition_hash"],
            "instance_hash": decision["prompt_instance_hash"],
            "rendered_blocks": decision["rendered_blocks"],
            "token_counts": decision["prompt_token_counts"],
        }
        for row in selected_rows
        for decision in row["decisions"].values()
    ]
    token_csv, token_rows = _prompt_token_scenarios(game, config)
    trajectory_csv, trajectory_rows = _trajectory(
        result, rules.actions, rules.population_size
    )
    frozen_wire_parity = (
        _frozen_prompt_wire_parity()
        and _baseline_selected_wire_parity(selected_rows)
    )

    _write(destination / "resolved_config.yaml", resolved_config_yaml(config))
    _write(
        destination / "agents_initial.json",
        _json({"agents": [agent.to_dict() for agent in result.initial_state.agents]}),
    )
    _write(destination / "interactions.jsonl", _jsonl(interaction_rows))
    _write(destination / "selected_audit_traces.jsonl", _jsonl(selected_rows))
    _write(destination / "selected_block_traces.jsonl", _jsonl(selected_block_rows))
    definition_prompt = plan.decision_stages[0].representative_prompt.bound_prompt
    _write(
        destination / "full_prompt_definition.json",
        _json(definition_prompt.definition_dict()),
    )
    _write(destination / "game_call_plan.json", _json(plan.to_dict()))
    _write(destination / "prompt_token_scenarios.csv", token_csv)
    _write(destination / "trajectory.csv", trajectory_csv)
    _write(
        destination / "prompt_parity_report.md",
        "# Naming prompt parity report\n\n"
        "The Version 3 prompt exposes five semantic blocks. Its normalized system and "
        f"user message content {'preserves' if frozen_wire_parity else 'does not preserve'} "
        "the frozen Version 2 empty, representative, and maximum-memory fixtures "
        "for both selected agents exactly; "
        "only local provenance metadata, family identity, and fingerprints changed.\n",
    )
    _plots(trajectory_rows, rules.actions, destination, rules.population_size)
    _write(
        destination / "agents_final.json",
        _json({"agents": [agent.to_dict() for agent in result.final_state.agents]}),
    )

    secret_values = [
        value
        for name, value in os.environ.items()
        if any(marker in name.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
        and len(value) >= 8
    ]
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in destination.iterdir()
        if path.is_file() and path.suffix != ".png"
    )
    artifacts_secret_free = (
        "Bearer " not in artifact_text
        and '"authorization"' not in artifact_text.lower()
        and not any(value in artifact_text for value in secret_values)
    )

    prompt_text = "\n".join(
        message["content"]
        for row in interaction_rows
        for decision in row["decisions"].values()
        for message in decision["compiled_messages"]
    ).lower()
    privacy_markers = (
        "agent-",
        "population size",
        "global interaction",
        "consensus",
        "committee",
        "committed",
    )
    guard_status = guard.status()
    checks = {
        "fixed_horizon_completed": len(result.interactions) == rules.max_interactions,
        "simultaneous_pre_state_barrier": all(
            item.decisions[0].request.interaction_id
            == item.decisions[1].request.interaction_id
            for item in result.interactions
        ),
        "private_prompt_boundary": not any(marker in prompt_text for marker in privacy_markers),
        "empty_memory_calls_provider": all(
            decision.validation_attempts >= 1
            for item in result.interactions
            for decision in item.decisions
            if not decision.request.visible_memory
        ),
        "bounded_visible_memory_and_full_audit_history": (
            all(
                len(decision.request.visible_memory) <= rules.memory_size
                for item in result.interactions
                for decision in item.decisions
            )
            and any(
                len(agent.private_history) > rules.memory_size
                for agent in result.final_state.agents
            )
        ),
        "action_orders_seeded_and_auditable": all(
            set(decision.request.presented_actions) == set(rules.actions)
            for item in result.interactions
            for decision in item.decisions
        ),
        "payoff_and_memory_transitions_valid": all(
            item.transition.payoff
            == (rules.success_payoff if item.transition.success else rules.failure_payoff)
            for item in result.interactions
        ),
        "legacy_fixed_fixture_parity": _legacy_parity_fixture(game, config),
        "frozen_prompt_wire_parity": frozen_wire_parity,
        "provider_independent_stage_aware_plan": (
            plan.metadata["provider_prices_included"] is False
            and plan.logical_decisions.expected == 2 * rules.max_interactions
            and plan.provider_requests.maximum
            == 2 * rules.max_interactions * (1 + rules.invalid_response_retries)
        ),
        "planned_and_actual_valid_mock_calls_match": (
            plan.provider_requests.expected == result.validation_attempts
            == guard_status["used_and_reserved"]["requests"]
        ),
        "memory_aware_token_scenarios": (
            {row["visible_memory_entries"] for row in token_rows}
            >= {0, rules.memory_size}
        ),
        "selected_audit_traces_complete": len(selected_rows) == len(selected_indices),
        "sampling_profile_deviations_audited": all(
            decision.attempts[0].completion_request.metadata["requested_sampling"]["top_k"]
            == config.llm_provider.options.get("top_k")
            and (
                config.llm_provider.options.get("top_k") is None
                or "top_k"
                in decision.attempts[0].completion_request.metadata[
                    "unsupported_or_adapter_omitted_parameters"
                ]
            )
            for item in result.interactions
            for decision in item.decisions
        ),
        "artifacts_are_secret_free": artifacts_secret_free,
        "plots_nonempty": all(
            (destination / name).stat().st_size > 0
            for name in ("action_share.png", "coordination_rate.png")
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    report = f"""# Phase 6 Naming Convention Game inspection report

- Status: **{status.upper()}**
- Command: `mas-cc game run --config {source} --output-dir {destination}`
- Scientific scope: repeated symmetric Ashery–Aiello–Baronchelli convention game, not the speaker/hearer inventory Naming Game.
- Run classification: **architecture smoke test; not a paper replication**.
- Code paths exercised: complete-mixing pair sampling, two frozen private views, concurrent validated provider decisions, answer-first parsing, isolated validation retries, pure +100/-50 transition, bounded prompt memory, complete evaluator history, stage-aware provider-neutral planning, Phase 4 pricing/budget composition, and deterministic plotting.
- Expected behavior: {rules.population_size} ordinary agents play {rules.max_interactions} sequential pair interactions with two simultaneous decisions per pair; empty-memory choices remain provider calls.
- Deviations from the paper profile: population and horizon are reduced for inspection; the deterministic mock provider replaces source-model stochasticity; `top_k=10` is requested and recorded but the current normalized adapters omit it; one project population round is labeled as N pair interactions.

## Results

- Pair interactions: {len(result.interactions)}
- Logical decisions: {result.logical_decisions}
- Validation attempts/provider requests: {result.validation_attempts}
- Provider transport retries: {result.provider_retries}
- Successful coordination interactions: {sum(item.transition.success for item in result.interactions)}
- Maximum full private history: {max(len(agent.private_history) for agent in result.final_state.agents)}
- Visible memory bound: {rules.memory_size}
- Planned requests lower/expected/maximum: {plan.provider_requests.lower}/{plan.provider_requests.expected}/{plan.provider_requests.maximum}
- Pricing composition: `{preflight.pricing['status']}`; launch `{preflight.launch_status}`
- Legacy fixed-fixture parity: {'passed' if checks['legacy_fixed_fixture_parity'] else 'failed'}

## Files to inspect manually

- `agents_initial.json` and `agents_final.json` — complete evaluator-side agent histories and lifetime scores.
- `interactions.jsonl` — every selected pair, private view, prompt, response, parser result, retry count, transition, and post-memory.
- `selected_audit_traces.jsonl` — deterministic full traces for interactions {', '.join(map(str, selected_indices))}.
- `game_call_plan.json` — pair stage, concurrency barrier, logical decisions, validation bounds, and memory scenarios without provider prices.
- `prompt_token_scenarios.csv` — empty, representative, and full-memory prompt estimates and hashes.
- `trajectory.csv` — raw outcomes plus rolling action shares and coordination over up to N interactions.
- `action_share.png` and `coordination_rate.png` — rolling evaluator-only diagnostics using window N={rules.population_size}.
- `manifest.json` — hashes and machine-readable acceptance checks.
"""
    _write(destination / "report.md", report)
    _write_manifest(destination, phase=6, status=status, checks=checks)
    return status == "pass"
