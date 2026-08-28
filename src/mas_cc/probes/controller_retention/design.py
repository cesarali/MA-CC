"""Deterministic local vignettes for the focused controller-retention probe."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
import hashlib
import json

from mas_cc.core import Seed
from mas_cc.games.relational_reasoning.data import RelationalTask

PROBE_VERSION = "controller_retention_probe_v2"

NO_OP = "NO_OP"
ONE_SLOT = "one_slot"
TWO_SLOTS = "two_slots"
ARMS_BY_Q: Mapping[int, tuple[str, ...]] = {
    2: (NO_OP, ONE_SLOT),
    3: (NO_OP, ONE_SLOT, TWO_SLOTS),
}
ARMS = (NO_OP, ONE_SLOT, TWO_SLOTS)

TARGET_TRUTH = "truth"
TARGET_FALSE = "false"
TARGET_SEMANTICS = (TARGET_TRUTH, TARGET_FALSE)

ZERO_SUPPORT = "zero_support"
VOTE_CONTROLLER_TARGET = "controller_target"
VOTE_TRUTH = "truth"
VOTE_OTHER = "other_non_target"
INITIAL_VOTE_ROLES = (VOTE_CONTROLLER_TARGET, VOTE_TRUTH, VOTE_OTHER)


def _digest(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def controller_slot_count(q: int, arm: str) -> int:
    """Return the number of controller-controlled visible slots for an arm."""

    if q not in ARMS_BY_Q:
        raise ValueError(f"q must be one of {list(ARMS_BY_Q)}")
    if arm not in ARMS_BY_Q[q]:
        raise ValueError(f"arm {arm!r} is not supported for q={q}")
    return {NO_OP: 0, ONE_SLOT: 1, TWO_SLOTS: 2}[arm]


@dataclass(frozen=True, slots=True)
class Vignette:
    """One local situation shared by all valid exposure arms at one q."""

    task_id: str
    task_fingerprint: str
    dataset_dir: str
    reasoning_depth: int
    truth_semantic: str
    controller_target_semantic: str
    target_semantics: str
    receiver_epistemic_disposition: str
    message_mode: str
    known_fact_ids: tuple[str, ...]
    initial_vote_role: str
    initial_vote_semantic: str
    q: int
    ordinary_peer_votes: tuple[str, ...]
    option_permutation_seed: int
    replicate: int

    @property
    def vignette_id(self) -> str:
        return _digest(
            {
                "probe_version": PROBE_VERSION,
                "task_id": self.task_id,
                "task_fingerprint": self.task_fingerprint,
                "L": self.reasoning_depth,
                "target_semantics": self.target_semantics,
                "controller_target": self.controller_target_semantic,
                "receiver": self.receiver_epistemic_disposition,
                "message_mode": self.message_mode,
                "known_fact_ids": list(self.known_fact_ids),
                "initial_vote_semantic": self.initial_vote_semantic,
                "q": self.q,
                "ordinary_peer_votes": list(self.ordinary_peer_votes),
                "option_permutation_seed": self.option_permutation_seed,
                "replicate": self.replicate,
            }
        )[:24]

    @property
    def pair_id(self) -> str:
        """Compatibility alias for the shared matched-vignette identity."""

        return self.vignette_id

    def call_id(self, model_identity: str, arm: str) -> str:
        if arm not in ARMS_BY_Q[self.q]:
            raise ValueError(f"arm {arm!r} is not supported for q={self.q}")
        return f"{self.vignette_id}:{model_identity}:{arm}"

    def controller_slots(self, arm: str) -> int:
        return controller_slot_count(self.q, arm)

    def controller_exposure_fraction(self, arm: str) -> float:
        return self.controller_slots(arm) / self.q

    def vote_count_vector(self, semantic_answers: Sequence[str], arm: str) -> dict[str, int]:
        counts = {str(option): 0 for option in semantic_answers}
        controlled = self.controller_slots(arm)
        for vote in self.ordinary_peer_votes[controlled:]:
            counts[vote] = counts.get(vote, 0) + 1
        return counts


@dataclass(frozen=True, slots=True)
class DesignSpec:
    """The deliberately small, fixed scientific design."""

    seed: int
    reasoning_depths: tuple[int, ...] = (1, 2)
    q_values: tuple[int, ...] = (2, 3)
    receivers: tuple[str, ...] = ("naive",)
    targets: tuple[str, ...] = TARGET_SEMANTICS
    tasks_per_depth: int = 12
    replicates: int = 1
    tasks: Mapping[int, tuple[str, ...]] = field(default_factory=dict)


def focal_known_fact_ids(task: RelationalTask) -> tuple[str, ...]:
    """Use distractors only, leaving all supporting facts unavailable."""

    distractors = set(task.distractor_fact_ids)
    return tuple(fact_id for fact_id in task.fact_order if fact_id in distractors)


def controller_target_for(task: RelationalTask, target_semantics: str) -> str:
    if target_semantics == TARGET_TRUTH:
        return task.correct_relation
    if target_semantics == TARGET_FALSE:
        for relation in task.semantic_answers:
            if relation != task.correct_relation:
                return relation
        raise ValueError(f"task {task.task_id!r} has no incorrect option")
    raise ValueError(f"target_semantics must be one of {list(TARGET_SEMANTICS)}")


def initial_vote_semantic(task: RelationalTask, controller_target: str, role: str) -> str:
    if role == VOTE_CONTROLLER_TARGET:
        return controller_target
    if role == VOTE_TRUTH:
        return task.correct_relation
    if role == VOTE_OTHER:
        for relation in task.semantic_answers:
            if relation not in (controller_target, task.correct_relation):
                return relation
        for relation in task.semantic_answers:
            if relation != controller_target:
                return relation
        raise ValueError(f"task {task.task_id!r} has no alternative option")
    raise ValueError(f"initial vote role must be one of {list(INITIAL_VOTE_ROLES)}")


def initial_vote_role_for_task(task_index: int) -> str:
    return INITIAL_VOTE_ROLES[task_index % len(INITIAL_VOTE_ROLES)]


def _background_votes(
    task: RelationalTask, controller_target: str, *, q: int, rng: Any
) -> tuple[str, ...]:
    """Build one frozen peer panel with no peer already supporting the target."""

    alternatives = [
        relation for relation in task.semantic_answers if relation != controller_target
    ]
    votes = [alternatives[index % len(alternatives)] for index in range(q)]
    rng.shuffle(votes)
    return tuple(votes)


def build_vignettes(
    spec: DesignSpec,
    tasks: Mapping[int, Sequence[RelationalTask]],
    task_fingerprints: Mapping[tuple[int, str], str],
) -> tuple[Vignette, ...]:
    """Build 96 base vignettes for the requested 12-task design."""

    root = Seed(spec.seed)
    vignettes: list[Vignette] = []
    for depth in spec.reasoning_depths:
        for task_index, task in enumerate(tasks.get(depth, ())):
            role = initial_vote_role_for_task(task_index)
            for target_semantics in spec.targets:
                target = controller_target_for(task, target_semantics)
                initial_vote = initial_vote_semantic(task, target, role)
                for receiver in spec.receivers:
                    for q in spec.q_values:
                        for replicate in range(spec.replicates):
                            namespace = (
                                f"controller-retention-v2:{task.task_id}:{depth}:"
                                f"{target_semantics}:{receiver}:{q}:{replicate}"
                            )
                            cell = root.derive(namespace)
                            vignettes.append(
                                Vignette(
                                    task_id=task.task_id,
                                    task_fingerprint=task_fingerprints.get(
                                        (depth, task.task_id), ""
                                    ),
                                    dataset_dir=str(task.source_path),
                                    reasoning_depth=depth,
                                    truth_semantic=task.correct_relation,
                                    controller_target_semantic=target,
                                    target_semantics=target_semantics,
                                    receiver_epistemic_disposition=receiver,
                                    message_mode="recommendation_only",
                                    known_fact_ids=focal_known_fact_ids(task),
                                    initial_vote_role=role,
                                    initial_vote_semantic=initial_vote,
                                    q=q,
                                    ordinary_peer_votes=_background_votes(
                                        task,
                                        target,
                                        q=q,
                                        rng=cell.derive("peer-panel").create_random(),
                                    ),
                                    option_permutation_seed=int(
                                        cell.derive("option-permutation")
                                    ),
                                    replicate=replicate,
                                )
                            )
    return tuple(vignettes)


__all__ = [
    "ARMS",
    "ARMS_BY_Q",
    "CONTROLLED",
    "INITIAL_VOTE_ROLES",
    "NO_OP",
    "ONE_SLOT",
    "PROBE_VERSION",
    "TARGET_FALSE",
    "TARGET_SEMANTICS",
    "TARGET_TRUTH",
    "TWO_SLOTS",
    "DesignSpec",
    "Vignette",
    "build_vignettes",
    "controller_slot_count",
    "controller_target_for",
    "focal_known_fact_ids",
    "initial_vote_role_for_task",
    "initial_vote_semantic",
]

# Historical name retained for imports outside this package. Controlled calls
# are now represented by the explicit one_slot and two_slots arms.
CONTROLLED = ONE_SLOT
