"""Immutable parent checkpoints and resumable continuation bundles.

These artifacts are scientific fork points.  They are deliberately separate
from the recorder's replaceable provider-failure checkpoint.
"""

from __future__ import annotations

import inspect
import json
import os
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mas_cc.core import Seed
from mas_cc.storage import canonical_hash

from .state import RelationalGameState

PARENT_CHECKPOINT_SCHEMA_VERSION = 1
CONTINUATION_STREAM_DERIVATION_VERSION = 1
PARENT_BUNDLE_SCHEMA_VERSION = 1
BRANCH_POLICIES = (
    "none",
    "always_truth",
    "always_false",
    "sensing_truth",
    "sensing_false",
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _task_hash(state: RelationalGameState) -> str:
    return canonical_hash(state.to_dict()["task"])


@dataclass(frozen=True, slots=True)
class ParentCheckpoint:
    """Versioned, content-addressed state at a population-round boundary."""

    parent_id: str
    checkpoint_id: str
    state: Mapping[str, Any]
    absolute_round: int
    micro_update_boundary: int
    q: int
    rho: float
    population_size: int
    preparation_rounds: int
    task_hash: str
    initialization_hash: str
    prompt_hash: str
    model_hash: str
    preparation_config_hash: str
    parent_seed: int
    parent_stream_provenance: Mapping[str, Any]
    runtime_state: Mapping[str, Any]
    controller_history: Sequence[Mapping[str, Any]] = ()
    schema_version: int = PARENT_CHECKPOINT_SCHEMA_VERSION
    checkpoint_hash: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "parent_id": self.parent_id,
            "checkpoint_id": self.checkpoint_id,
            "state": dict(self.state),
            "absolute_round": self.absolute_round,
            "micro_update_boundary": self.micro_update_boundary,
            "q": self.q,
            "rho": self.rho,
            "population_size": self.population_size,
            "preparation_rounds": self.preparation_rounds,
            "task_hash": self.task_hash,
            "initialization_hash": self.initialization_hash,
            "prompt_hash": self.prompt_hash,
            "model_hash": self.model_hash,
            "preparation_config_hash": self.preparation_config_hash,
            "parent_seed": self.parent_seed,
            "parent_stream_provenance": dict(self.parent_stream_provenance),
            "runtime_state": dict(self.runtime_state),
            "controller_history": [dict(item) for item in self.controller_history],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "checkpoint_hash": self.checkpoint_hash}

    @classmethod
    def create(
        cls,
        *,
        parent_id: str,
        checkpoint_id: str,
        state: RelationalGameState,
        absolute_round: int,
        preparation_rounds: int,
        initialization_hash: str,
        prompt_hash: str,
        model_hash: str,
        preparation_config_hash: str,
        parent_seed: int,
        parent_stream_provenance: Mapping[str, Any],
        runtime_state: Mapping[str, Any],
        controller_history: Sequence[Mapping[str, Any]] = (),
    ) -> "ParentCheckpoint":
        rules = dict(state.data.get("rules", {}))
        checkpoint = cls(
            parent_id=parent_id,
            checkpoint_id=checkpoint_id,
            state=state.to_dict(),
            absolute_round=absolute_round,
            micro_update_boundary=state.turn,
            q=int(rules["social_group_size"]),
            rho=float(rules.get("epistemic_persistence", 1.0)),
            population_size=len(state.agents),
            preparation_rounds=preparation_rounds,
            task_hash=_task_hash(state),
            initialization_hash=initialization_hash,
            prompt_hash=prompt_hash,
            model_hash=model_hash,
            preparation_config_hash=preparation_config_hash,
            parent_seed=parent_seed,
            parent_stream_provenance=dict(parent_stream_provenance),
            runtime_state=dict(runtime_state),
            controller_history=tuple(dict(item) for item in controller_history),
        )
        checkpoint.validate()
        return replace(checkpoint, checkpoint_hash=canonical_hash(checkpoint.payload()))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParentCheckpoint":
        checkpoint = cls(
            parent_id=str(value["parent_id"]),
            checkpoint_id=str(value["checkpoint_id"]),
            state=dict(value["state"]),
            absolute_round=int(value["absolute_round"]),
            micro_update_boundary=int(value["micro_update_boundary"]),
            q=int(value["q"]),
            rho=float(value["rho"]),
            population_size=int(value["population_size"]),
            preparation_rounds=int(value["preparation_rounds"]),
            task_hash=str(value["task_hash"]),
            initialization_hash=str(value["initialization_hash"]),
            prompt_hash=str(value["prompt_hash"]),
            model_hash=str(value["model_hash"]),
            preparation_config_hash=str(value["preparation_config_hash"]),
            parent_seed=int(value["parent_seed"]),
            parent_stream_provenance=dict(value["parent_stream_provenance"]),
            runtime_state=dict(value.get("runtime_state", {})),
            controller_history=tuple(dict(item) for item in value.get("controller_history", ())),
            schema_version=int(value.get("schema_version", -1)),
            checkpoint_hash=str(value.get("checkpoint_hash", "")),
        )
        checkpoint.validate()
        expected = canonical_hash(checkpoint.payload())
        if checkpoint.checkpoint_hash != expected:
            raise ValueError("parent checkpoint hash mismatch")
        return checkpoint

    def validate(
        self,
        *,
        task_hash: str | None = None,
        initialization_hash: str | None = None,
        q: int | None = None,
        rho: float | None = None,
    ) -> RelationalGameState:
        if self.schema_version != PARENT_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported parent checkpoint schema version")
        if self.checkpoint_hash and self.checkpoint_hash != canonical_hash(self.payload()):
            raise ValueError("parent checkpoint hash mismatch")
        if not self.parent_id or not self.checkpoint_id:
            raise ValueError("parent_id and checkpoint_id must be non-empty")
        if self.absolute_round != self.preparation_rounds:
            raise ValueError("checkpoint must be at the configured preparation boundary")
        state = RelationalGameState.from_dict(self.state)
        if state.terminated:
            raise ValueError("a parent checkpoint cannot contain a terminated state")
        if state.turn != self.micro_update_boundary:
            raise ValueError("checkpoint micro-update boundary does not match state.turn")
        if state.turn != self.absolute_round * self.population_size:
            raise ValueError("checkpoint is not at a complete population-round boundary")
        if len(state.agents) != self.population_size:
            raise ValueError("checkpoint population size mismatch")
        if _task_hash(state) != self.task_hash:
            raise ValueError("checkpoint task hash mismatch")
        rules = dict(state.data.get("rules", {}))
        if int(rules.get("social_group_size", -1)) != self.q:
            raise ValueError("checkpoint q does not match restored state")
        if float(rules.get("epistemic_persistence", -1.0)) != self.rho:
            raise ValueError("checkpoint rho does not match restored state")
        for message in state.blackboard.messages:
            if message.round_created >= self.absolute_round:
                raise ValueError("checkpoint contains a board message from a future round")
            if message.micro_step_created > self.micro_update_boundary:
                raise ValueError("checkpoint contains a board message from a future update")
        if self.controller_history:
            raise ValueError("uncontrolled preparation must have empty controller history")
        if task_hash is not None and task_hash != self.task_hash:
            raise ValueError("branch task differs from the prepared parent")
        if initialization_hash is not None and initialization_hash != self.initialization_hash:
            raise ValueError("branch initialization differs from the prepared parent")
        if q is not None and q != self.q:
            raise ValueError("branch q differs from the prepared parent")
        if rho is not None and float(rho) != self.rho:
            raise ValueError("branch rho differs from the prepared parent")
        return state

    @property
    def checkpoint_votes(self) -> tuple[str, ...]:
        state = self.validate()
        return tuple(str(agent.committed_action) for agent in state.agents)

    @property
    def checkpoint_option_counts(self) -> Mapping[str, int]:
        state = self.validate()
        counts = Counter(self.checkpoint_votes)
        return {option: int(counts.get(option, 0)) for option in state.possible_answers}

    def h0_observation(
        self,
        *,
        continuation_rounds: int,
        copy_id: int = 1,
    ) -> Mapping[str, Any]:
        """Canonical shared trajectory row at the branch horizon ``h=0``."""

        state = self.validate()
        return {
            "parent_id": self.parent_id,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_hash": self.checkpoint_hash,
            "branch_policy": "checkpoint",
            "posting_budget": None,
            "copy_id": copy_id,
            "absolute_round": self.absolute_round,
            "post_branch_horizon": 0,
            "q": self.q,
            "rho": self.rho,
            "N": self.population_size,
            "L": self.preparation_rounds,
            "M": continuation_rounds,
            "correct_answer": state.correct_answer,
            "semantic_votes": list(self.checkpoint_votes),
            "option_counts": dict(self.checkpoint_option_counts),
            "preparation_seed": self.parent_seed,
            "branch_status": "checkpoint_sealed",
        }


class ParentCheckpointStore:
    """Content-addressed immutable storage for parent fork points."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def path_for(self, checkpoint_hash: str) -> Path:
        return self.directory / f"{checkpoint_hash}.json"

    def write(self, checkpoint: ParentCheckpoint) -> Path:
        checkpoint = ParentCheckpoint.from_dict(checkpoint.to_dict())
        path = self.path_for(checkpoint.checkpoint_hash)
        if path.exists():
            existing = ParentCheckpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if existing.to_dict() != checkpoint.to_dict():
                raise ValueError("content-addressed checkpoint collision")
            return path
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                checkpoint.to_dict(),
                stream,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = ParentCheckpoint.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
            if existing.to_dict() != checkpoint.to_dict():
                raise ValueError("content-addressed checkpoint collision")
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def load(self, checkpoint_hash: str) -> ParentCheckpoint:
        path = self.path_for(checkpoint_hash)
        if not path.exists():
            raise FileNotFoundError(path)
        return ParentCheckpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class ContinuationBranch:
    policy: str
    posting_budget: int | None
    copy_id: int = 1

    def __post_init__(self) -> None:
        if self.policy not in BRANCH_POLICIES:
            raise ValueError(f"unknown checkpoint branch policy {self.policy!r}")
        if self.copy_id < 1:
            raise ValueError("copy_id must be positive")
        if self.policy == "none":
            if self.posting_budget not in {None, 0}:
                raise ValueError("the none branch has canonical posting budget zero/none")
            object.__setattr__(self, "posting_budget", None)
        elif self.posting_budget is None or self.posting_budget < 0:
            raise ValueError("controlled branches require a non-negative posting budget")

    @property
    def branch_id(self) -> str:
        budget = "none" if self.posting_budget is None else str(self.posting_budget)
        return f"{self.policy}--b-{budget}--copy-{self.copy_id:03d}"

    def derive_stream(self, checkpoint: ParentCheckpoint) -> tuple[int, Mapping[str, Any]]:
        identity = {
            "q": checkpoint.q,
            "rho": checkpoint.rho,
            "parent_id": checkpoint.parent_id,
            "policy": self.policy,
            "posting_budget": self.posting_budget,
            "copy_id": self.copy_id,
        }
        namespace = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        seed = int(Seed(checkpoint.parent_seed).derive(f"checkpoint-continuation-v1:{namespace}"))
        return seed, {
            "derivation_version": CONTINUATION_STREAM_DERIVATION_VERSION,
            "identity": identity,
            "seed": seed,
        }

    def metadata(
        self,
        checkpoint: ParentCheckpoint,
        *,
        continuation_rounds: int,
        truth_target: str,
        false_target: str,
    ) -> Mapping[str, Any]:
        state = checkpoint.validate()
        if truth_target != state.correct_answer:
            raise ValueError("truth controller target must equal the semantic correct answer")
        if false_target == state.correct_answer or false_target not in state.possible_answers:
            raise ValueError("false controller target must be an incorrect semantic answer")
        target = (
            None
            if self.policy == "none"
            else truth_target
            if self.policy.endswith("truth")
            else false_target
        )
        seed, stream = self.derive_stream(checkpoint)
        return {
            "parent_id": checkpoint.parent_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_hash": checkpoint.checkpoint_hash,
            "branch_policy": self.policy,
            "posting_budget": self.posting_budget,
            "copy_id": self.copy_id,
            "q": checkpoint.q,
            "rho": checkpoint.rho,
            "N": checkpoint.population_size,
            "L": checkpoint.preparation_rounds,
            "M": continuation_rounds,
            "controller_target_semantic_id": target,
            "false_target_semantic_id": false_target,
            "correct_answer_semantic_id": state.correct_answer,
            "preparation_seed": checkpoint.parent_seed,
            "continuation_seed": seed,
            "continuation_stream_derivation_version": stream["derivation_version"],
            "continuation_stream_identity": stream["identity"],
            "branch_status": "in_progress",
            "branch_recovery_status": "restored_parent",
        }


async def run_checkpoint_continuation(
    *,
    game: Any,
    config: Any,
    provider: Any,
    checkpoint: ParentCheckpoint,
    branch: ContinuationBranch,
    continuation_rounds: int,
    truth_target: str,
    false_target: str,
    token_counter: Any | None = None,
    observer: Any | None = None,
) -> Any:
    """Execute one named child with the policy implied by its branch identity."""

    from .controller import SCHEDULE_ALWAYS, SCHEDULE_SOFT, RelationalRoundBudgetedControl
    from .runtime import run_relational_imitation_round_feedback_game

    state = checkpoint.validate(q=checkpoint.q, rho=checkpoint.rho)
    metadata = branch.metadata(
        checkpoint,
        continuation_rounds=continuation_rounds,
        truth_target=truth_target,
        false_target=false_target,
    )
    seed = int(metadata["continuation_seed"])
    control = None
    if branch.policy != "none":
        options = {
            **dict(config.control.options),
            "target": metadata["controller_target_semantic_id"],
            "intervention_budget": branch.posting_budget,
            "advocacy_schedule": (
                SCHEDULE_ALWAYS if branch.policy.startswith("always_") else SCHEDULE_SOFT
            ),
            "policy": "soft_target",
        }
        control = RelationalRoundBudgetedControl.from_options(options)
    return await run_relational_imitation_round_feedback_game(
        game,
        config,
        provider,
        token_counter=token_counter,
        observer=observer,
        control=control,
        initial_state=state,
        start_round=checkpoint.absolute_round,
        continuation_length=continuation_rounds,
        continuation_seed=seed,
        restored_runtime_state=checkpoint.runtime_state,
        continuation_metadata=metadata,
    )


def standard_continuation_branches(
    *, budgets: Sequence[int] = (3, 12), copy_id: int = 1
) -> tuple[ContinuationBranch, ...]:
    branches = [ContinuationBranch("none", None, copy_id)]
    for policy in ("always_truth", "always_false", "sensing_truth", "sensing_false"):
        branches.extend(ContinuationBranch(policy, int(budget), copy_id) for budget in budgets)
    return tuple(branches)


class ParentBundleWorker:
    """Resume-safe one-parent/all-descendants execution contract.

    ``execute_branch`` receives ``(checkpoint, branch, branch_directory)`` and
    returns a JSON-serializable summary.  A valid completed sibling is never
    invoked again when another branch is retried.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.branches_dir = self.directory / "branches"

    async def run(
        self,
        checkpoint: ParentCheckpoint,
        branches: Sequence[ContinuationBranch],
        execute_branch: Callable[
            [ParentCheckpoint, ContinuationBranch, Path],
            Mapping[str, Any] | Awaitable[Mapping[str, Any]],
        ],
        drain_controller: Any | None = None,
    ) -> Mapping[str, Any]:
        unique = {branch.branch_id: branch for branch in branches}
        if len(unique) != len(branches):
            raise ValueError("parent bundle contains duplicate branch identities")
        copy_ids = {branch.copy_id for branch in branches}
        none_copies = {
            branch.copy_id for branch in branches if branch.policy == "none"
        }
        if none_copies != copy_ids:
            raise ValueError(
                "parent bundle must contain exactly one no-control branch per copy"
            )
        completed: dict[str, Any] = {}
        for branch_id, branch in unique.items():
            branch_dir = self.branches_dir / branch_id
            seal_path = branch_dir / "branch_seal.json"
            if seal_path.exists():
                seal = json.loads(seal_path.read_text(encoding="utf-8"))
                if seal.get("checkpoint_hash") != checkpoint.checkpoint_hash:
                    raise ValueError("completed branch belongs to a different checkpoint")
                if seal.get("branch_id") != branch_id or seal.get("status") != "complete":
                    raise ValueError("completed branch seal has incompatible identity/status")
                recorded_hash = str(seal.get("branch_hash", ""))
                unhashed = {key: value for key, value in seal.items() if key != "branch_hash"}
                if recorded_hash != canonical_hash(unhashed):
                    raise ValueError("completed branch seal hash mismatch")
                completed[branch_id] = seal
                continue
            if drain_controller is not None:
                drain_controller.raise_if_safe("branch")
            branch_dir.mkdir(parents=True, exist_ok=True)
            if drain_controller is not None:
                drain_controller.work_started("branch")
            try:
                result = execute_branch(checkpoint, branch, branch_dir)
                if inspect.isawaitable(result):
                    result = await result
                seed, stream = branch.derive_stream(checkpoint)
                seal = {
                    "schema_version": PARENT_BUNDLE_SCHEMA_VERSION,
                    "status": "complete",
                    "parent_id": checkpoint.parent_id,
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "checkpoint_hash": checkpoint.checkpoint_hash,
                    "branch_id": branch_id,
                    "branch_policy": branch.policy,
                    "posting_budget": branch.posting_budget,
                    "copy_id": branch.copy_id,
                    "continuation_seed": seed,
                    "continuation_stream": stream,
                    "result": dict(result),
                }
                seal["branch_hash"] = canonical_hash(seal)
                _atomic_json(seal_path, seal)
            finally:
                if drain_controller is not None:
                    drain_controller.work_finished("branch")
            completed[branch_id] = seal
            if drain_controller is not None and len(completed) < len(unique):
                drain_controller.raise_if_safe("branch")
        bundle = {
            "schema_version": PARENT_BUNDLE_SCHEMA_VERSION,
            "status": "complete",
            "parent_id": checkpoint.parent_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_hash": checkpoint.checkpoint_hash,
            "expected_branches": sorted(unique),
            "completed_branches": sorted(completed),
            "branch_hashes": {
                branch_id: completed[branch_id]["branch_hash"] for branch_id in sorted(completed)
            },
        }
        bundle["bundle_hash"] = canonical_hash(bundle)
        _atomic_json(self.directory / "parent_bundle_seal.json", bundle)
        return bundle


@dataclass(frozen=True, slots=True)
class CheckpointBundleResult:
    """Minimal result contract consumed by the generic experiment runner."""

    interactions: tuple[Any, ...]
    termination_reason: str
    logical_decisions: int
    bundle: Mapping[str, Any]


async def run_checkpoint_parent_bundle(
    *,
    game: Any,
    config: Any,
    provider: Any,
    observer: Any,
    token_counter: Any | None = None,
) -> CheckpointBundleResult:
    """Prepare one parent and execute every configured continuation branch.

    The generic orchestrator still owns the episode, provider guard, recorder,
    and retry boundary. This function only supplies the reusable one-parent /
    many-descendants topology within that ordinary episode.
    """
    drain_controller = getattr(observer, "drain_controller", None)
    if drain_controller is not None:
        drain_controller.raise_if_safe("branch")

    from mas_cc.storage import prompt_definition_hash

    from .initialization import physical_initial_state_projection
    from .runtime import run_relational_imitation_round_feedback_game

    ensemble = config.ensemble
    if not ensemble.enabled:
        raise ValueError("checkpoint parent bundle requires ensemble.enabled")
    episode_label = str(getattr(observer, "episode_label", config.execution.seed))
    recorder = getattr(observer, "recorder", None)
    if recorder is None or not hasattr(recorder, "output_dir"):
        raise ValueError("checkpoint parent bundle requires a recorder-backed observer")
    scientific_identity = getattr(recorder, "scientific_identity", None)
    parent_id = (
        f"{scientific_identity.cell_id}:{episode_label}"
        if scientific_identity is not None
        else episode_label
    )
    bundle_dir = Path(recorder.output_dir) / "checkpoint_ensembles" / episode_label
    store = ParentCheckpointStore(bundle_dir / "parent_checkpoints")
    pointer_path = bundle_dir / "parent_checkpoint.json"

    checkpoint: ParentCheckpoint
    preparation_logical_decisions = 0
    interactions: list[Any] = []
    if pointer_path.exists():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        checkpoint = store.load(str(pointer["checkpoint_hash"]))
        checkpoint.validate(
            q=int(config.game.options["social_group_size"]),
            rho=float(config.game.options["epistemic_persistence"]),
        )
    else:
        preparation = await run_relational_imitation_round_feedback_game(
            game,
            config,
            provider,
            token_counter=token_counter,
            observer=observer,
            control=None,
            continuation_length=ensemble.preparation_rounds,
            continuation_metadata={
                "parent_id": parent_id,
                "branch_policy": "preparation",
                "posting_budget": None,
                "copy_id": None,
                "L": ensemble.preparation_rounds,
                "M": ensemble.continuation_rounds,
                "branch_status": "preparing_parent",
            },
        )
        initialization_hash = canonical_hash(
            physical_initial_state_projection(preparation.initial_state)
        )
        checkpoint = ParentCheckpoint.create(
            parent_id=parent_id,
            checkpoint_id=f"{parent_id}--h0",
            state=preparation.final_state,
            absolute_round=ensemble.preparation_rounds,
            preparation_rounds=ensemble.preparation_rounds,
            initialization_hash=initialization_hash,
            prompt_hash=prompt_definition_hash(config),
            model_hash=canonical_hash(config.llm_provider.to_dict()),
            preparation_config_hash=canonical_hash(config.to_dict()),
            parent_seed=int(config.execution.seed),
            parent_stream_provenance={
                "derivation_version": 1,
                "seed": int(config.execution.seed),
                "scope": "uncontrolled_preparation",
            },
            runtime_state=preparation.runtime_state,
        )
        store.write(checkpoint)
        _atomic_json(
            pointer_path,
            {
                "schema_version": 1,
                "parent_id": checkpoint.parent_id,
                "checkpoint_id": checkpoint.checkpoint_id,
                "checkpoint_hash": checkpoint.checkpoint_hash,
            },
        )
        preparation_logical_decisions = preparation.logical_decisions
        interactions.extend(preparation.interactions)

    state = checkpoint.validate()
    truth_target = str(state.correct_answer)
    false_target = str(ensemble.false_target)
    branches = tuple(
        ContinuationBranch(policy, None if policy == "none" else budget, copy_id)
        for copy_id in range(1, ensemble.continuation_copies + 1)
        for policy in ensemble.branch_policies
        for budget in ((None,) if policy == "none" else ensemble.posting_budgets)
    )
    for copy_id in range(1, ensemble.continuation_copies + 1):
        observer.record_round_trajectory(
            record={
                "record_type": "relational_checkpoint_h0",
                "round_index": ensemble.preparation_rounds - 1,
                **checkpoint.h0_observation(
                    continuation_rounds=ensemble.continuation_rounds,
                    copy_id=copy_id,
                ),
                "false_target_semantic_id": false_target,
                "correct_answer_semantic_id": truth_target,
            }
        )

    branch_logical_decisions = 0

    async def execute_branch(
        parent: ParentCheckpoint, branch: ContinuationBranch, branch_dir: Path
    ) -> Mapping[str, Any]:
        nonlocal branch_logical_decisions
        result = await run_checkpoint_continuation(
            game=game,
            config=config,
            provider=provider,
            checkpoint=parent,
            branch=branch,
            continuation_rounds=ensemble.continuation_rounds,
            truth_target=truth_target,
            false_target=false_target,
            token_counter=token_counter,
            observer=observer,
        )
        branch_logical_decisions += int(result.logical_decisions)
        interactions.extend(result.interactions)
        return {
            "termination_reason": result.termination_reason,
            "logical_decisions": int(result.logical_decisions),
            "validation_attempts": int(result.validation_attempts),
            "final_state_hash": canonical_hash(result.final_state.to_dict()),
            "rounds": ensemble.continuation_rounds,
            "branch_directory": str(branch_dir),
        }

    bundle = await ParentBundleWorker(bundle_dir).run(
        checkpoint, branches, execute_branch, drain_controller=drain_controller
    )
    if not ensemble.retain_parent_artifacts:
        store.path_for(checkpoint.checkpoint_hash).unlink(missing_ok=True)
    return CheckpointBundleResult(
        interactions=tuple(interactions),
        termination_reason="checkpoint_parent_bundle_complete",
        logical_decisions=preparation_logical_decisions + branch_logical_decisions,
        bundle=bundle,
    )


__all__ = [
    "BRANCH_POLICIES",
    "CONTINUATION_STREAM_DERIVATION_VERSION",
    "ContinuationBranch",
    "CheckpointBundleResult",
    "PARENT_CHECKPOINT_SCHEMA_VERSION",
    "ParentBundleWorker",
    "ParentCheckpoint",
    "ParentCheckpointStore",
    "run_checkpoint_continuation",
    "run_checkpoint_parent_bundle",
    "standard_continuation_branches",
]
