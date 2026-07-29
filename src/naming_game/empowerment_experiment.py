"""Configuration-driven committee experiments for the convention game.

This module owns intervention schedules and analysis-ready observations.  It
does not put population state or intervention metadata into agent prompts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import shutil
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd
import yaml

from .api_client import LLMClient
from .models import ConfigurationError
from .naming_convention_game import (
    ConventionGameConfig,
    ConventionHistoryEntry,
    ConventionIntervention,
    NamingConventionGame,
)

SCHEMA_VERSION = 2
PROMPT_VERSION = "convention-answer-first-v1"
Regime = Literal["neutral", "consensus_attack", "pulse"]


@dataclass(frozen=True)
class ReplicationConfig:
    unit: Literal["per_policy", "per_stratum"] = "per_policy"
    count: int = 100

    def __post_init__(self) -> None:
        if self.unit not in {"per_policy", "per_stratum"}:
            raise ConfigurationError("replications.unit must be per_policy or per_stratum.")
        if self.count < 1:
            raise ConfigurationError("replications.count must be positive.")


@dataclass(frozen=True)
class ConventionRolesConfig:
    """Experiment-supplied convention roles; never inferred from outcomes."""

    strong_name: str
    weak_name: str
    source: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str)
            for value in (self.strong_name, self.weak_name, self.source)
        ):
            raise ConfigurationError("convention_roles values must be strings.")
        if self.strong_name == self.weak_name:
            raise ConfigurationError("strong_name and weak_name must be distinct.")
        if not self.source.strip():
            raise ConfigurationError("convention_roles.source must be non-empty.")


@dataclass(frozen=True)
class EmpowermentExperimentConfig:
    population_size: int = 24
    names: tuple[str, str] = ("A", "B")
    memory_length: int = 5
    max_population_rounds: int = 30
    committee_sizes: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8)
    pulse_rounds: tuple[int, ...] = (1, 3, 5, 10)
    regimes: tuple[Regime, ...] = ("neutral", "consensus_attack", "pulse")
    replications: ReplicationConfig = ReplicationConfig()
    auto_analyze: bool = True
    quick_bootstrap_resamples: int = 200
    quick_null_permutations: int = 200
    convention_roles: ConventionRolesConfig | None = None
    temperature: float = 0.5
    max_tokens: int = 15
    seed: int = 1
    window_interactions: int | None = None
    resolution_threshold: float = 0.95
    provider: str = "university"
    model: str = "gwdg/qwen3-30b-a3b-instruct-2507"
    fallback_provider: str = "openai"
    fallback_model: str | None = None
    allow_fallback: bool = False
    request_concurrency: int = 20
    episode_concurrency: int = 1
    timeout_seconds: float = 60.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ConfigurationError("population_size must be at least 2.")
        if self.names != ("A", "B"):
            raise ConfigurationError("The first probe requires names: [A, B].")
        if self.memory_length < 0 or self.max_population_rounds < 1:
            raise ConfigurationError("memory_length and max_population_rounds are invalid.")
        if any(k < 0 or k > self.population_size for k in self.committee_sizes):
            raise ConfigurationError("committee sizes must be between 0 and population_size.")
        if "pulse" in self.regimes and any(
            duration < 1 or duration > self.max_population_rounds
            for duration in self.pulse_rounds
        ):
            raise ConfigurationError("pulse_rounds must lie inside the episode horizon.")
        if not 0.5 < self.resolution_threshold <= 1.0:
            raise ConfigurationError("resolution_threshold must be in (0.5, 1].")
        if self.episode_concurrency < 1 or self.request_concurrency < 1:
            raise ConfigurationError("concurrency settings must be positive.")
        if not set(self.regimes) <= {"neutral", "consensus_attack", "pulse"}:
            raise ConfigurationError("Unknown experiment regime.")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (
                self.quick_bootstrap_resamples,
                self.quick_null_permutations,
            )
        ):
            raise ConfigurationError("Quick analysis resample counts must be integers.")
        if self.quick_bootstrap_resamples < 0 or self.quick_null_permutations < 0:
            raise ConfigurationError("Quick analysis resample counts cannot be negative.")
        if not isinstance(self.auto_analyze, bool):
            raise ConfigurationError("auto_analyze must be true or false.")
        if self.convention_roles is not None and {
            self.convention_roles.strong_name,
            self.convention_roles.weak_name,
        } != set(self.names):
            raise ConfigurationError(
                "convention_roles strong_name and weak_name must match names."
            )

    @property
    def rolling_window(self) -> int:
        return self.window_interactions or 3 * self.population_size

    @property
    def max_interactions(self) -> int:
        return self.population_size * self.max_population_rounds


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    seed: int
    regime: Regime
    committee_size: int
    committee_policy: str
    initial_condition: str
    incumbent: str | None
    alternative: str | None
    pulse_rounds: int | None
    replicate: int


@dataclass(frozen=True)
class CommitteeSchedule(ConventionIntervention):
    committee_ids: tuple[int, ...]
    policy: str
    action: str | None
    active_through_interaction: int | None = None

    def window_active(self, interaction_index: int) -> bool:
        return self.active_through_interaction is not None and interaction_index <= self.active_through_interaction

    def action_for(self, agent_id: int, interaction_index: int) -> str | None:
        if agent_id not in self.committee_ids or self.action is None:
            return None
        if self.active_through_interaction is not None and interaction_index > self.active_through_interaction:
            return None
        return self.action


def load_experiment_config(path: str | Path) -> EmpowermentExperimentConfig:
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Could not load experiment config: {source}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("Experiment configuration must be a YAML mapping.")
    replication_raw = raw.get("replications", {})
    if not isinstance(replication_raw, dict):
        raise ConfigurationError("replications must be a mapping.")
    roles_raw = raw.get("convention_roles")
    if roles_raw is not None and not isinstance(roles_raw, dict):
        raise ConfigurationError("convention_roles must be a mapping or null.")
    try:
        values = dict(raw)
        values["names"] = tuple(str(value) for value in raw.get("names", ("A", "B")))
        values["committee_sizes"] = tuple(int(value) for value in raw.get("committee_sizes", (0, 1, 2, 3, 4, 6, 8)))
        values["pulse_rounds"] = tuple(int(value) for value in raw.get("pulse_rounds", (1, 3, 5, 10)))
        values["regimes"] = tuple(raw.get("regimes", ("neutral", "consensus_attack", "pulse")))
        values["replications"] = ReplicationConfig(**replication_raw)
        values["convention_roles"] = (
            ConventionRolesConfig(**roles_raw) if roles_raw is not None else None
        )
        return EmpowermentExperimentConfig(**values)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("Experiment configuration has invalid fields.") from exc


def _policy_replicates(policies: Sequence[str], replications: ReplicationConfig) -> list[tuple[str, int]]:
    if replications.unit == "per_policy":
        return [(policy, replicate) for policy in policies for replicate in range(replications.count)]
    return [
        (policies[index % len(policies)], index // len(policies))
        for index in range(replications.count)
    ]


def build_episode_specs(config: EmpowermentExperimentConfig) -> tuple[EpisodeSpec, ...]:
    raw_specs: list[dict[str, Any]] = []
    first, second = config.names
    for regime in config.regimes:
        if regime == "neutral":
            strata = [("empty", None, None, None)]
            policies = ("always_A", "always_B", "no_committee")
        elif regime == "consensus_attack":
            strata = [
                (f"consensus_{incumbent}", incumbent, second if incumbent == first else first, None)
                for incumbent in config.names
            ]
            policies = ("support_incumbent", "promote_alternative", "no_committee")
        else:
            strata = [
                (f"consensus_{incumbent}", incumbent, second if incumbent == first else first, duration)
                for incumbent in config.names
                for duration in config.pulse_rounds
            ]
            policies = ("alternative_pulse", "no_pulse")
        for committee_size in config.committee_sizes:
            for initial, incumbent, alternative, duration in strata:
                for policy, replicate in _policy_replicates(policies, config.replications):
                    raw_specs.append(
                        {
                            "regime": regime,
                            "committee_size": committee_size,
                            "committee_policy": policy,
                            "initial_condition": initial,
                            "incumbent": incumbent,
                            "alternative": alternative,
                            "pulse_rounds": duration,
                            "replicate": replicate,
                        }
                    )
    specs: list[EpisodeSpec] = []
    for ordinal, raw in enumerate(raw_specs):
        episode_seed = config.seed + ordinal
        identity = json.dumps({**raw, "seed": episode_seed}, sort_keys=True)
        specs.append(
            EpisodeSpec(
                episode_id=hashlib.sha256(identity.encode()).hexdigest()[:20],
                seed=episode_seed,
                **raw,
            )
        )
    return tuple(specs)


def _schedule_for(spec: EpisodeSpec, config: EmpowermentExperimentConfig) -> CommitteeSchedule:
    import random

    rng = random.Random(spec.seed ^ 0xC01117EE)
    ids = tuple(sorted(rng.sample(range(1, config.population_size + 1), spec.committee_size)))
    action: str | None = None
    active_through: int | None = None
    if spec.regime == "neutral":
        action = {"always_A": config.names[0], "always_B": config.names[1]}.get(spec.committee_policy)
    elif spec.regime == "consensus_attack":
        if spec.committee_policy == "support_incumbent":
            action = spec.incumbent
        elif spec.committee_policy == "promote_alternative":
            action = spec.alternative
    elif spec.committee_policy == "alternative_pulse":
        action = spec.alternative
        active_through = (spec.pulse_rounds or 0) * config.population_size
    return CommitteeSchedule(ids, spec.committee_policy, action, active_through)


def _memory_json(memory: Sequence[ConventionHistoryEntry]) -> str:
    return json.dumps([asdict(entry) for entry in memory], sort_keys=True, separators=(",", ":"))


def _prompt_hash(config: EmpowermentExperimentConfig) -> str:
    body = {
        "version": PROMPT_VERSION,
        "names": config.names,
        "memory_length": config.memory_length,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "success_reward": 100,
        "failure_payoff": -50,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _direction_metadata(
    spec: EpisodeSpec, config: EmpowermentExperimentConfig
) -> dict[str, str | None]:
    roles = config.convention_roles
    direction: str | None = None
    if spec.incumbent is not None and spec.alternative is not None:
        if roles is None:
            direction = f"{spec.incumbent}_to_{spec.alternative}"
        elif (
            spec.incumbent == roles.strong_name
            and spec.alternative == roles.weak_name
        ):
            direction = "strong_to_weak"
        elif (
            spec.incumbent == roles.weak_name
            and spec.alternative == roles.strong_name
        ):
            direction = "weak_to_strong"
    return {
        "strong_name": roles.strong_name if roles is not None else None,
        "weak_name": roles.weak_name if roles is not None else None,
        "convention_role_source": roles.source if roles is not None else None,
        "incumbent_name": spec.incumbent,
        "promoted_name": spec.alternative,
        "attack_direction": direction,
    }


def _experiment_fingerprint(config: EmpowermentExperimentConfig) -> str:
    """Hash data-generating settings while allowing grid/concurrency expansion."""

    body = {
        "schema_version": SCHEMA_VERSION,
        "population_size": config.population_size,
        "names": config.names,
        "memory_length": config.memory_length,
        "max_population_rounds": config.max_population_rounds,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "window_interactions": config.rolling_window,
        "resolution_threshold": config.resolution_threshold,
        "provider": config.provider,
        "model": config.model,
        "prompt_hash": _prompt_hash(config),
        "convention_roles": (
            asdict(config.convention_roles)
            if config.convention_roles is not None
            else None
        ),
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def _label_from_share(share: float, threshold: float, full_window: bool) -> str | None:
    if not full_window:
        return None
    if share >= threshold:
        return "A"
    if share <= 1.0 - threshold:
        return "B"
    return "unresolved"


def derive_episode(
    rows: list[dict[str, Any]], spec: EpisodeSpec, config: EmpowermentExperimentConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    window: deque[tuple[str, str]] = deque(maxlen=config.rolling_window)
    previous_binary: int | None = None
    first_consensus_row: dict[str, Any] | None = None
    takeover_row: dict[str, Any] | None = None
    for row in rows:
        window.append((row["output_i"], row["output_j"]))
        outputs = [value for pair in window for value in pair]
        share = outputs.count(config.names[0]) / len(outputs)
        full = len(window) == config.rolling_window
        if share > 0.5:
            binary = 1
        elif share < 0.5:
            binary = 0
        else:
            binary = previous_binary
        if binary is not None:
            previous_binary = binary
        sensitivity = "B_dominant" if share < 0.4 else "A_dominant" if share > 0.6 else "mixed"
        resolved = _label_from_share(share, config.resolution_threshold, full)
        row.update(
            rolling_share_A=share,
            rolling_window_count=len(window),
            insufficient_window=not full,
            macrostate_binary=binary,
            macrostate_three=sensitivity,
            resolved_state=resolved,
        )
        if resolved in config.names and first_consensus_row is None:
            first_consensus_row = row
        if (
            spec.alternative is not None
            and spec.alternative == resolved
            and takeover_row is None
        ):
            takeover_row = row

    terminal = rows[-1]
    terminal_outcome = terminal["resolved_state"] or "unresolved"
    committee_actions = sum(int(row["forced_i"]) + int(row["forced_j"]) for row in rows)
    pulse_removal_index = (spec.pulse_rounds or 0) * config.population_size if spec.regime == "pulse" else None
    recovery_row: dict[str, Any] | None = None
    recovery_time: int | None = None
    if pulse_removal_index is not None and spec.incumbent is not None:
        at_removal = next((row for row in rows if row["interaction_index"] == pulse_removal_index), None)
        if at_removal is not None and at_removal["resolved_state"] == spec.incumbent:
            recovery_row, recovery_time = at_removal, 0
        else:
            recovery_row = next(
                (row for row in rows if row["interaction_index"] > pulse_removal_index and row["resolved_state"] == spec.incumbent),
                None,
            )
            if recovery_row is not None:
                recovery_time = recovery_row["interaction_index"] - pulse_removal_index

    if spec.alternative is None:
        peak_share = None
        peak_row = None
    else:
        alternative_shares = [
            (1.0 - row["rolling_share_A"] if spec.alternative == config.names[1] else row["rolling_share_A"], row)
            for row in rows
        ]
        peak_share, peak_row = max(alternative_shares, key=lambda item: item[0])

    same_after_consensus: float | None = None
    if first_consensus_row is not None:
        target = first_consensus_row["resolved_state"]
        tail = [row for row in rows if row["interaction_index"] >= first_consensus_row["interaction_index"]]
        round_endpoints = {
            row["population_round"]: row for row in tail
        }
        same_after_consensus = sum(
            row["resolved_state"] == target for row in round_endpoints.values()
        ) / len(round_endpoints)

    summary = {
        "schema_version": SCHEMA_VERSION,
        **asdict(spec),
        **_direction_metadata(spec, config),
        "provider": terminal["provider"],
        "model": terminal["model"],
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": terminal["prompt_hash"],
        "N": config.population_size,
        "W": 2,
        "H": config.memory_length,
        "committee_ids": terminal["committee_ids"],
        "max_interactions": config.max_interactions,
        "stopping_interaction": first_consensus_row["interaction_index"] if first_consensus_row else None,
        "stopping_population_round": first_consensus_row["population_round"] if first_consensus_row else None,
        "final_convention": terminal_outcome,
        "terminal_share_A": terminal["rolling_share_A"],
        "unresolved": terminal_outcome == "unresolved",
        "takeover": takeover_row is not None,
        "ever_crossed": takeover_row is not None,
        "terminal_takeover": (
            spec.alternative is not None and terminal_outcome == spec.alternative
        ),
        "incumbent_survives": (
            spec.incumbent is not None and terminal_outcome == spec.incumbent
        ),
        "takeover_interaction": takeover_row["interaction_index"] if takeover_row else None,
        "takeover_population_round": takeover_row["population_round"] if takeover_row else None,
        "recovery_time_interactions": recovery_time,
        "recovery_time_population_rounds": recovery_time / config.population_size if recovery_time is not None else None,
        "recovery_censored": pulse_removal_index is not None and recovery_row is None,
        "permanent_flip": spec.alternative is not None and terminal_outcome == spec.alternative,
        "peak_displacement": peak_share,
        "time_to_peak_interactions": peak_row["interaction_index"] if peak_row else None,
        "time_to_peak_population_rounds": peak_row["interaction_index"] / config.population_size if peak_row else None,
        "total_committee_actions": committee_actions,
        "post_consensus_persistence": same_after_consensus,
        "pulse_removal_interaction": pulse_removal_index,
    }
    for row in rows:
        row["terminal_outcome"] = terminal_outcome
    return rows, summary


async def run_episode(
    spec: EpisodeSpec,
    config: EmpowermentExperimentConfig,
    client: LLMClient,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schedule = _schedule_for(spec, config)
    game = NamingConventionGame(
        client=client,
        config=ConventionGameConfig(
            num_agents=config.population_size,
            actions=config.names,
            memory_size=config.memory_length,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        ),
        seed=spec.seed,
        intervention=schedule,
        request_seed_base=spec.seed * 1_000_000,
    )
    if spec.incumbent is not None:
        game.seed_consensus_history(spec.incumbent)
    result = await game.run(config.max_interactions, stop_on_convergence=False)
    prompt_hash = _prompt_hash(config)
    provider = getattr(client, "provider_name", client.__class__.__name__)
    actual_models = [
        decision.response.model
        for record in result.interactions
        for decision in (record.player_1_decision, record.player_2_decision)
        if decision.response is not None
    ]
    actual_model = actual_models[-1] if actual_models else client.model
    rows: list[dict[str, Any]] = []
    direction_metadata = _direction_metadata(spec, config)
    for record in result.interactions:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "episode_id": spec.episode_id,
                "seed": spec.seed,
                "regime": spec.regime,
                "provider": provider,
                "model": actual_model,
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": prompt_hash,
                "N": config.population_size,
                "W": 2,
                "H": config.memory_length,
                "committee_size": spec.committee_size,
                "committee_ids": json.dumps(schedule.committee_ids),
                "committee_policy": spec.committee_policy,
                "initial_condition": spec.initial_condition,
                "incumbent": spec.incumbent,
                "alternative": spec.alternative,
                **direction_metadata,
                "pulse_rounds": spec.pulse_rounds,
                "pulse_active": schedule.window_active(record.interaction_index) and schedule.action is not None,
                "interaction_index": record.interaction_index,
                "population_round": math.ceil(record.interaction_index / config.population_size),
                "agent_i": record.player_1_id,
                "agent_j": record.player_2_id,
                "output_i": record.player_1_action,
                "output_j": record.player_2_action,
                "forced_i": record.player_1_decision.forced,
                "forced_j": record.player_2_decision.forced,
                "success": record.success,
                "payoff_i": record.payoff,
                "payoff_j": record.payoff,
                "memory_i_before": _memory_json(record.player_1_memory_before),
                "memory_j_before": _memory_json(record.player_2_memory_before),
            }
        )
    return derive_episode(rows, spec, config)


async def run_experiment(
    config: EmpowermentExperimentConfig,
    client: LLMClient,
    output_dir: str | Path,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    destination = Path(output_dir)
    fingerprint = _experiment_fingerprint(config)
    shards = destination / ".episode_shards" / fingerprint
    shards.mkdir(parents=True, exist_ok=True)
    specs = build_episode_specs(config)
    semaphore = asyncio.Semaphore(config.episode_concurrency)

    async def execute(spec: EpisodeSpec) -> None:
        interactions_path = shards / f"{spec.episode_id}.interactions.parquet"
        summary_path = shards / f"{spec.episode_id}.episode.parquet"
        if resume and interactions_path.exists() and summary_path.exists():
            return
        async with semaphore:
            rows, summary = await run_episode(spec, config, client)
            temp_interactions = interactions_path.with_suffix(".tmp.parquet")
            temp_summary = summary_path.with_suffix(".tmp.parquet")
            pd.DataFrame(rows).to_parquet(temp_interactions, index=False)
            pd.DataFrame([summary]).to_parquet(temp_summary, index=False)
            temp_interactions.replace(interactions_path)
            temp_summary.replace(summary_path)

    await asyncio.gather(*(execute(spec) for spec in specs))
    interaction_frames = [pd.read_parquet(path) for path in sorted(shards.glob("*.interactions.parquet"))]
    episode_frames = [pd.read_parquet(path) for path in sorted(shards.glob("*.episode.parquet"))]
    interactions = pd.concat(interaction_frames, ignore_index=True) if interaction_frames else pd.DataFrame()
    episodes = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()
    destination.mkdir(parents=True, exist_ok=True)
    interactions_path = destination / "interactions.parquet"
    episodes_path = destination / "episodes.parquet"
    interactions_temp = destination / "interactions.compacting.parquet"
    episodes_temp = destination / "episodes.compacting.parquet"
    config_path = destination / "experiment_config.json"
    config_temp = destination / "experiment_config.compacting.json"
    try:
        interactions.to_parquet(interactions_temp, index=False)
        episodes.to_parquet(episodes_temp, index=False)
        compacted_interactions = pd.read_parquet(interactions_temp)
        compacted_episodes = pd.read_parquet(episodes_temp)
        if len(compacted_interactions) != len(interactions):
            raise RuntimeError("Interaction Parquet compaction row-count mismatch.")
        if len(compacted_episodes) != len(episodes):
            raise RuntimeError("Episode Parquet compaction row-count mismatch.")
        interaction_ids = set(compacted_interactions.get("episode_id", ()))
        episode_ids = set(compacted_episodes.get("episode_id", ()))
        if interaction_ids != episode_ids:
            raise RuntimeError("Compacted interaction and episode IDs do not match.")
        expected_ids = {spec.episode_id for spec in specs}
        if not expected_ids <= episode_ids:
            raise RuntimeError("Compacted histories are missing completed episodes.")
        config_temp.write_text(
            json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
        )
        interactions_temp.replace(interactions_path)
        episodes_temp.replace(episodes_path)
        config_temp.replace(config_path)
    finally:
        interactions_temp.unlink(missing_ok=True)
        episodes_temp.unlink(missing_ok=True)
        config_temp.unlink(missing_ok=True)
    return {
        "episodes": len(episodes),
        "interactions": len(interactions),
        "interactions_path": str(interactions_path),
        "episodes_path": str(episodes_path),
        "experiment_fingerprint": fingerprint,
    }


def clear_completed_shards(output_dir: str | Path) -> None:
    """Remove recoverable checkpoint shards after successful archival."""

    shards = Path(output_dir) / ".episode_shards"
    if shards.exists():
        shutil.rmtree(shards)


__all__ = [
    "CommitteeSchedule",
    "ConventionRolesConfig",
    "EmpowermentExperimentConfig",
    "EpisodeSpec",
    "ReplicationConfig",
    "build_episode_specs",
    "derive_episode",
    "load_experiment_config",
    "run_episode",
    "run_experiment",
]
