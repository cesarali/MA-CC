"""Phase 7 local-first observability inspection."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from mas_cc.config import load_run_config, resolved_config_yaml
from mas_cc.games import create_game
from mas_cc.games.naming_convention import NamingConventionGame, run_naming_convention_game_sync
from mas_cc.llm_providers import BudgetGuardedProvider, RuntimeBudgetGuard, create_llm_provider, resolve_budget_limits
from mas_cc.observability import DetailedAuditPolicy, RunRecorder, price_snapshot_hash
from mas_cc.planning import estimate_input_tokens, static_game_preflight

from .game import _budgets, _quote
from .inspect import _write, _write_manifest


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


class _ObservedRecorder:
    """Adds budget state at the runtime callback boundary without coupling modules."""

    def __init__(self, recorder: RunRecorder, guard: RuntimeBudgetGuard) -> None:
        self.recorder = recorder
        self.guard = guard

    def event(self, event_type: str, **payload: Any) -> None:
        self.recorder.event(event_type, **payload)

    def record_attempt(self, **payload: Any) -> None:
        self.recorder.record_attempt(**payload, budget_status=self.guard.status())

    def record_interaction(self, **payload: Any) -> None:
        self.recorder.record_interaction(**payload, budget_status=self.guard.checkpoint_state())


def _dashboard(metrics_path: Path, destination: Path) -> None:
    import csv
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    figure, axis = plt.subplots(figsize=(7, 4), dpi=120)
    axis.plot([int(row["round_index"]) for row in rows], [float(row["successful_interaction"]) for row in rows], marker="o", label="coordination")
    axis.plot([int(row["round_index"]) for row in rows], [float(row["provider_attempts"]) for row in rows], marker="s", label="provider attempts")
    axis.set_xlabel("Interaction round")
    axis.set_title("Phase 7 operational dashboard")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, metadata={"Software": "MAS-CC"})
    plt.close(figure)


def run_phase_7_inspection(config_path: str | Path, output_dir: str | Path) -> bool:
    """Run the convention game with bounded local audit and optional Comet metrics."""

    source = Path(config_path).resolve()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config = load_run_config(source)
    game = create_game(config.game)
    if not isinstance(game, NamingConventionGame):
        raise ValueError("Phase 7 inspection currently requires the naming_convention game")
    quote = _quote(config)
    system_budget, run_budget = _budgets(config, quote)
    preflight = static_game_preflight(
        game.call_plan(config.game), config.prompt, config.llm_provider,
        assumed_output_tokens=config.llm_provider.max_output_tokens, pricing_quote=quote,
        system_budget=system_budget, run_budget=run_budget,
        explicit_override=config.pricing.explicit_unknown_price_override,
        allow_stale_pricing=not config.pricing.require_fresh_at_launch,
    )
    if preflight.launch_status != "permitted":
        raise ValueError(f"Phase 7 preflight is {preflight.launch_status!r}; no provider calls sent")
    resolved = config.to_dict()
    policy = DetailedAuditPolicy.from_mapping(config.logging.options.get("detailed_prompt_audit"))
    snapshot = quote.to_dict()
    recorder = RunRecorder(
        destination, run_id=f"{config.experiment.name}-{config.execution.seed}",
        resolved_config=resolved, policy=policy, comet_enabled=config.logging.comet,
        project_name=str(config.logging.options.get("comet_project", "mas-cc")),
        checkpoint_enabled=config.storage.checkpoints, price_snapshot_hash=price_snapshot_hash(snapshot),
    )
    guard = RuntimeBudgetGuard(resolve_budget_limits(system_budget, run_budget))
    provider = create_llm_provider(config.llm_provider)
    guarded = BudgetGuardedProvider(provider, guard, quote.pricing, input_token_estimator=estimate_input_tokens, input_token_multiplier=1.0)
    try:
        result = run_naming_convention_game_sync(game, config, guarded, observer=_ObservedRecorder(recorder, guard))
    except Exception as exc:
        recorder.event("run_failed", error_type=type(exc).__name__, error=str(exc))
        recorder.finalize(status="failed", budget_status=guard.checkpoint_state())
        raise
    finally:
        guarded.close()
    summary = recorder.finalize(status="completed", budget_status=guard.checkpoint_state())
    _write(destination / "resolved_config.yaml", resolved_config_yaml(config))
    _dashboard(destination / "local_metrics.csv", destination / "observability_dashboard.png")
    interactions = [item.to_dict() for item in result.interactions]
    # No prompt text appears in normal events, Comet, checkpoints, or the manifest.
    public_text = "\n".join(
        path.read_text(encoding="utf-8") for path in destination.iterdir()
        if path.is_file() and path.name not in {"audit_traces.jsonl", "prompt_block_traces.jsonl"}
        and path.suffix in {".json", ".jsonl", ".log", ".yaml", ".csv", ".md"}
    )
    checkpoint = json.loads((destination / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    checks = {
        "three_or_more_rounds_completed": len(interactions) >= 3,
        "one_provider_attempt_per_participant_per_round": all(
            decision.validation_attempts == 1
            for item in result.interactions for decision in item.decisions
        ),
        "all_attempts_have_prompt_provenance": all(
            decision.prompt_definition_hash and decision.prompt_instance_hash
            for item in result.interactions for decision in item.decisions
        ),
        "lightweight_events_cover_every_attempt": len((destination / "api_call_status.jsonl").read_text(encoding="utf-8").splitlines()) == result.validation_attempts,
        "checkpoint_is_atomic_and_resume_compatible": checkpoint["completed_rounds"] == len(interactions) and checkpoint["prompt_state_restored"] is False,
        "detailed_audit_is_bounded": summary["audit"]["selected_prompt_records"] <= (policy.max_logged_prompts_per_run if policy.max_logged_prompts_per_run is not None else result.validation_attempts),
        "remote_summary_excludes_prompt_content": "compiled_messages" not in (destination / "comet_summary.json").read_text(encoding="utf-8"),
        "normal_artifacts_exclude_rendered_prompts": "compiled_messages" not in public_text and "rendered_blocks" not in public_text,
        "dashboard_nonempty": (destination / "observability_dashboard.png").stat().st_size > 0,
    }
    status = "pass" if all(checks.values()) else "fail"
    report = f"""# Phase 7 observability inspection report

- Status: **{status.upper()}**
- Run: `{config.experiment.name}`; {len(interactions)} pair interactions and {result.validation_attempts} provider attempts.
- Detailed audit policy: `{json.dumps(policy.to_dict(), sort_keys=True)}`.
- Comet: `{summary['comet']['status']}`. Comet is optional; only aggregate metrics are eligible for it.
- Resume boundary: checkpoint state contains game state, budget accounting, and hashes; prompts are reconstructed as immutable values and are never restored from a checkpoint.

## Files

- `events.jsonl`, `api_call_status.jsonl`, `usage_cost.jsonl`, and `budget_events.jsonl` are complete lightweight streams.
- `audit_traces.jsonl` and `prompt_block_traces.jsonl` exist only when the policy selected records.
- `checkpoint.json` is atomically replaced after each completed interaction; `checkpoint_manifest.json` records its compatibility boundary.
"""
    _write(destination / "report.md", report)
    _write_manifest(destination, phase=7, status=status, checks=checks, warnings=[] if summary["comet"]["status"] != "unavailable" else ["Comet requested but unavailable; local logging completed."])
    return status == "pass"
