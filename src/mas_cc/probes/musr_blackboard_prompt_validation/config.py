"""Configuration for actual-runtime blackboard prompt validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from mas_cc.config import GameConfig
from mas_cc.llm_runtime.config import LLMProviderConfig

PROBE_NAME = "musr_blackboard_prompt_validation"


def _map(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


@dataclass(frozen=True, slots=True)
class BlackboardValidationConfig:
    source_path: str
    mode: str
    calibration_root: Path
    expected_manifest_file_sha256: str
    expected_manifest_content_sha256: str
    expected_symbolic_selection_sha256: str
    provider: LLMProviderConfig
    task_ids: tuple[str, ...]
    agents_by_task: Mapping[str, tuple[str, ...]]
    state_ids: tuple[str, ...]
    q: int
    private_breadth: int
    max_predictability: float
    min_normalized_entropy: float
    min_score_margin: int
    round_zero_prompt: str
    full_profile: str
    current_vote_policy: str
    state_seed: int
    repetitions: int
    local_workers: int
    max_concurrency: int
    max_rpm: int
    fallback_concurrency: tuple[int, ...]
    invalid_response_retries: int
    static_comparison: bool
    output_dir: Path
    max_provider_requests: int
    max_input_tokens: int
    max_output_tokens_total: int
    max_cost: float
    accounting_unit: str

    @property
    def logical_calls(self) -> int:
        multiplier = 2 if self.static_comparison else 1
        return (
            len(self.task_ids)
            * sum(len(self.agents_by_task[x]) for x in self.task_ids)
            // len(self.task_ids)
            * len(self.state_ids)
            * self.repetitions
            * multiplier
        )

    @property
    def state_count(self) -> int:
        return sum(len(self.agents_by_task[x]) for x in self.task_ids) * len(
            self.state_ids
        )

    def game_config(self, task_id: str) -> GameConfig:
        return GameConfig(
            type="relational_imitation_round_feedback",
            population_size=12,
            horizon=1,
            topology="complete",
            options={
                "task_family": "musr_team_allocation",
                "task_dataset_dir": str(self.calibration_root / "accepted_tasks"),
                "task_id": task_id,
                "dynamics_mode": "reasoning",
                "rounds": 1,
                "social_group_size": self.q,
                "social_mode": "board",
                "board": {
                    "sampling": "uniform",
                    "message_lifetime_rounds": 1,
                    "exclude_self_authored": True,
                    "allow_no_post": True,
                },
                "vote_visibility": "public",
                "prompt_version": 1,
                "receiver_epistemic_disposition": "vigilant",
                "stop_on_consensus": False,
                "invalid_response_retries": self.invalid_response_retries,
                "expected_validation_failure_rate": 0.05,
                "epistemic_persistence": 1.0,
                "initialization": {"mode": "local_vote"},
                "local_prompt_variant": self.round_zero_prompt,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": PROBE_NAME,
            "mode": self.mode,
            "source": {
                "calibration_root": str(self.calibration_root),
                "expected_manifest_file_sha256": self.expected_manifest_file_sha256,
                "expected_manifest_content_sha256": self.expected_manifest_content_sha256,
                "expected_symbolic_selection_sha256": self.expected_symbolic_selection_sha256,
            },
            "provider": self.provider.to_dict(),
            "frozen_design": {
                "task_ids": list(self.task_ids),
                "agents_by_task": {
                    key: list(value) for key, value in self.agents_by_task.items()
                },
                "state_ids": list(self.state_ids),
                "q": self.q,
                "private_breadth": self.private_breadth,
                "max_predictability": self.max_predictability,
                "min_normalized_entropy": self.min_normalized_entropy,
                "min_score_margin": self.min_score_margin,
                "round_zero_prompt": self.round_zero_prompt,
                "full_profile": self.full_profile,
                "current_vote_policy": self.current_vote_policy,
                "state_seed": self.state_seed,
            },
            "execution": {
                "repetitions": self.repetitions,
                "local_workers": self.local_workers,
                "max_concurrency": self.max_concurrency,
                "max_rpm": self.max_rpm,
                "fallback_concurrency": list(self.fallback_concurrency),
                "invalid_response_retries": self.invalid_response_retries,
            },
            "analysis": {"static_comparison": {"enabled": self.static_comparison}},
            "storage": {"output_dir": str(self.output_dir)},
            "budget": {
                "accounting_unit": self.accounting_unit,
                "max_provider_requests": self.max_provider_requests,
                "max_input_tokens": self.max_input_tokens,
                "max_output_tokens": self.max_output_tokens_total,
                "max_cost": self.max_cost,
            },
        }


def load_config(path: str | Path) -> BlackboardValidationConfig:
    source_path = Path(path)
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("probe") != PROBE_NAME:
        raise ValueError(f"probe must be {PROBE_NAME}")
    source = _map(raw.get("source"), "source")
    provider_raw = _map(raw.get("provider"), "provider")
    design = _map(raw.get("frozen_design"), "frozen_design")
    execution = _map(raw.get("execution"), "execution")
    analysis = _map(raw.get("analysis", {}), "analysis")
    static = _map(analysis.get("static_comparison", {}), "analysis.static_comparison")
    storage = _map(raw.get("storage"), "storage")
    budget = _map(raw.get("budget"), "budget")
    provider = LLMProviderConfig(
        type=str(provider_raw.get("type", "university")),
        model=str(provider_raw["model"]),
        credentials_env=str(provider_raw.get("credentials_env", "POTSDAM_API_KEY")),
        base_url_env=str(provider_raw.get("base_url_env", "BASE_POTSDAM_LLM_URL")),
        timeout_seconds=float(provider_raw.get("timeout_seconds", 180)),
        max_retries=int(provider_raw.get("max_retries", 2)),
        request_concurrency=int(provider_raw.get("request_concurrency", 30)),
        temperature=float(provider_raw.get("temperature", 1.0)),
        max_output_tokens=int(provider_raw.get("max_output_tokens", 4096)),
        options=dict(provider_raw.get("options") or {}),
    )
    tasks = tuple(str(value) for value in design.get("task_ids", ()))
    agents_raw = _map(design.get("agents_by_task"), "frozen_design.agents_by_task")
    config = BlackboardValidationConfig(
        source_path=str(source_path),
        mode=str(raw.get("mode", "full")),
        calibration_root=Path(str(source["calibration_root"])),
        expected_manifest_file_sha256=str(source["expected_manifest_file_sha256"]),
        expected_manifest_content_sha256=str(
            source["expected_manifest_content_sha256"]
        ),
        expected_symbolic_selection_sha256=str(
            source["expected_symbolic_selection_sha256"]
        ),
        provider=provider,
        task_ids=tasks,
        agents_by_task={
            task: tuple(str(agent) for agent in agents_raw[task]) for task in tasks
        },
        state_ids=tuple(str(value) for value in design.get("state_ids", ())),
        q=int(design.get("q", 1)),
        private_breadth=int(design.get("private_breadth", 4)),
        max_predictability=float(design.get("max_predictability", 0.45)),
        min_normalized_entropy=float(design.get("min_normalized_entropy", 0.90)),
        min_score_margin=int(design.get("min_score_margin", 2)),
        round_zero_prompt=str(design.get("round_zero_prompt", "P2")),
        full_profile=str(design.get("full_profile", "F9")),
        current_vote_policy=str(
            design.get("current_vote_policy", "calibration_private_repetition_0")
        ),
        state_seed=int(design.get("state_seed", 20260902)),
        repetitions=int(execution.get("repetitions", 5)),
        local_workers=int(execution.get("local_workers", 4)),
        max_concurrency=int(execution.get("max_concurrency", 30)),
        max_rpm=int(execution.get("max_rpm", 500)),
        fallback_concurrency=tuple(
            int(value) for value in execution.get("fallback_concurrency", (30, 20, 10))
        ),
        invalid_response_retries=int(execution.get("invalid_response_retries", 1)),
        static_comparison=bool(static.get("enabled", False)),
        output_dir=Path(str(storage["output_dir"])),
        max_provider_requests=int(budget["max_provider_requests"]),
        max_input_tokens=int(budget["max_input_tokens"]),
        max_output_tokens_total=int(budget["max_output_tokens"]),
        max_cost=float(budget["max_cost"]),
        accounting_unit=str(budget.get("accounting_unit", "proxy_accounting_unit")),
    )
    if config.mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    if (
        provider.type,
        provider.model,
        provider.temperature,
        provider.max_output_tokens,
    ) != ("university", "gwdg/openai-gpt-oss-120b", 1.0, 4096):
        raise ValueError("provider settings differ from the frozen benchmark")
    if (
        config.state_ids,
        config.q,
        config.private_breadth,
        config.max_predictability,
        config.min_normalized_entropy,
        config.min_score_margin,
        config.round_zero_prompt,
        config.full_profile,
    ) != (("S0", "S1", "S2"), 1, 4, 0.45, 0.90, 2, "P2", "F9"):
        raise ValueError("blackboard state design differs from the frozen handoff")
    if config.current_vote_policy != "calibration_private_repetition_0":
        raise ValueError("unsupported current-vote policy")
    if config.fallback_concurrency != (30, 20, 10):
        raise ValueError("concurrency fallback must be exactly 30 -> 20 -> 10")
    if (
        config.max_concurrency != 30
        or config.max_rpm != 500
        or config.local_workers != 4
    ):
        raise ValueError(
            "execution controls must freeze workers=4, concurrency=30, RPM=500"
        )
    if provider.request_concurrency > config.max_concurrency:
        raise ValueError("provider request concurrency exceeds the global cap")
    expected = 12 if config.mode == "smoke" else 360
    expected_states = 6 if config.mode == "smoke" else 72
    if config.logical_calls != expected or config.state_count != expected_states:
        raise ValueError(
            f"{config.mode} design must contain {expected_states} states and {expected} calls"
        )
    if config.mode == "smoke" and config.logical_calls > 24:
        raise ValueError("development smoke exceeds 24 logical calls")
    if config.max_provider_requests < config.logical_calls:
        raise ValueError("request budget is below the logical call count")
    return config


__all__ = ["BlackboardValidationConfig", "PROBE_NAME", "load_config"]
