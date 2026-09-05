"""Round-level budgeted controller for the relational reasoning game.

The **sensing and policy machinery is reused unchanged** from the HiddenBench
round-feedback controller: one decision per population round, a hypergeometric
vote sensor of size ``q_c``, a soft (logistic) policy over ``{NO_OP,
ADVOCATE_Z}``, and an exact budget ``b`` of randomly placed controlled
positions.  The controller senses **votes only** and never receives any agent's
knowledge state.

For ``coordination_request`` with ``controller_timing: dawn_only``, the same
binary policy is retained but ``b`` means dawn board mass: the runtime posts
``b`` factless DIRECTIVEs before any focal update and uses no position schedule.
The default microscopic timing preserves historical configs and artifacts.

What this subclass adds is *what the controller may say*.  Because a relational
task's symbolic content is exact, the linguistic content of an intervention can
be manipulated as a controlled variable:

``message_mode: recommendation_only``
    The controller occupies its social slot with a recommendation and no
    evidence.  No fact enters anybody's ``K_i`` from the controller, ever.

``message_mode: recommendation_plus_fact``
    The same recommendation, plus exactly one fact of the frozen task, chosen
    **before the episode runs** and fixed for its whole duration.

``message_mode: silent``
    The **occlusion placebo**.  A controlled position still consumes one of the
    focal's ordinary peer slots on exactly the schedule the other modes use, but
    nothing is substituted into it: the focal simply sees one peer fewer.  No
    vote, no recommendation, no fact and no new speaker enter the prompt, so the
    only thing this mode removes is a peer-information opportunity.  It exists
    to separate that loss from the directional influence of an adversarial
    recommendation - run it against ``recommendation_only`` at the same budget
    and the difference is the directional part.

Fact selection is deterministic and never delegated to a model:
``controller_fact_id`` names one explicitly, or ``controller_fact_selector:
supporting`` resolves to the task's first supporting fact in task order.  One
fixed fact per episode is the default on purpose - varying the citation per slot
would add an uncontrolled stochastic channel to an experiment whose whole point
is measuring what one message does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar
import hashlib
import math

from mas_cc.config import ControlConfig
from mas_cc.control import Control
from mas_cc.llm_runtime.validation import ValidationIssue

from ...hidden_bench.imitation.controller import (
    ADVOCATE_TARGET,
    NO_OP,
    SoftTargetControl,
)
from ...hidden_bench.imitation_round_feedback.controller import (
    EVIDENCE_NONE,
    RoundSoftTargetBudgetedControl,
)
from ..data import RelationalTask
from .adaptive_communication import (
    COMMUNICATION_POLICY,
    COMMUNICATION_POLICY_VERSION,
)

RECOMMENDATION_ONLY = "recommendation_only"
RECOMMENDATION_PLUS_FACT = "recommendation_plus_fact"
SILENT = "silent"
MESSAGE_MODES = (RECOMMENDATION_ONLY, RECOMMENDATION_PLUS_FACT, SILENT)
FACTLESS_MESSAGE_MODES = (RECOMMENDATION_ONLY, SILENT)
"""Modes that never put a fact in front of a focal agent."""

SELECTOR_SUPPORTING = "supporting"
FACT_SELECTORS = (SELECTOR_SUPPORTING,)

EVIDENCE_NEUTRAL = "neutral"
EVIDENCE_STRATEGIC = "strategic"
CONTROLLER_EVIDENCE_STRATEGIES = (EVIDENCE_NEUTRAL, EVIDENCE_STRATEGIC)

DIRECT_RECOMMENDATION = "direct_recommendation"
COORDINATION_REQUEST = "coordination_request"
TRUTHFUL_STRATEGIC_REPORT = "truthful_strategic_report"
ADAPTIVE_COMMUNICATION = "adaptive_communication"
CONTROLLER_ACTUATION_MODES = (
    DIRECT_RECOMMENDATION,
    COORDINATION_REQUEST,
    TRUTHFUL_STRATEGIC_REPORT,
    ADAPTIVE_COMMUNICATION,
)

STRATEGIC_REPORT_SELECTION_V1 = "target_preserving_v1"
STRATEGIC_REPORT_SELECTION_STRATEGIES = (STRATEGIC_REPORT_SELECTION_V1,)

TIMING_MICROSCOPIC = "microscopic"
TIMING_DAWN_ONLY = "dawn_only"
CONTROLLER_TIMINGS = (TIMING_MICROSCOPIC, TIMING_DAWN_ONLY)

_DIRECTION_VECTORS = {
    "NORTH": (0, 1),
    "NORTHEAST": (1, 1),
    "EAST": (1, 0),
    "SOUTHEAST": (1, -1),
    "SOUTH": (0, -1),
    "SOUTHWEST": (-1, -1),
    "WEST": (-1, 0),
    "NORTHWEST": (-1, 1),
}

SCHEDULE_SOFT = "soft"
SCHEDULE_ALWAYS = "always"
SCHEDULE_NEVER = "never"
ADVOCACY_SCHEDULES = (SCHEDULE_SOFT, SCHEDULE_ALWAYS, SCHEDULE_NEVER)
"""``control.options.advocacy_schedule``.

``soft`` is the default and the closed loop: the sensed target share feeds the
logistic policy, so whether the controller acts depends on the population.
``always`` is open loop - it advocates every round regardless of what it sensed.
``never`` is the corresponding open-loop no-op schedule.

Open loop is what a *controllability* study wants. Under the soft policy the
actuation a population receives is a function of that population's own state,
so a cell where the population converged early is a cell where the controller
mostly stopped, and "did control move it" and "did it need moving" are
confounded. ``always`` fixes the intervention and lets the population vary.

The sensor still runs and is still logged under ``always``; only the decision
ignores it, which keeps the sensing columns available for comparison."""


@dataclass(frozen=True, slots=True)
class StrategicReportSelection:
    """One auditable true-fact choice; only the fact text reaches participants."""

    fact_id: str
    score: float
    strategy_class: str
    novel_on_live_board: bool
    cooldown_eligible: bool
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "score": self.score,
            "strategy_class": self.strategy_class,
            "novel_on_live_board": self.novel_on_live_board,
            "cooldown_eligible": self.cooldown_eligible,
            "rank": self.rank,
        }


@dataclass(frozen=True, slots=True)
class RelationalRoundBudgetedControl(RoundSoftTargetBudgetedControl):
    """Soft target policy, exact per-round slot budget, explicit evidence choice."""

    message_mode: str = RECOMMENDATION_ONLY
    controller_fact_id: str | None = None
    controller_fact_selector: str | None = None
    controller_evidence_strategy: str | None = None
    advocacy_schedule: str = SCHEDULE_SOFT
    controller_actuation_mode: str = DIRECT_RECOMMENDATION
    controller_timing: str = TIMING_MICROSCOPIC
    controller_report_cooldown_rounds: int = 1
    controller_report_selection_strategy: str = STRATEGIC_REPORT_SELECTION_V1
    allow_controller_requests: bool = True
    allow_controller_directives: bool = True
    controller_communication_policy: str = COMMUNICATION_POLICY
    controller_communication_policy_version: int = COMMUNICATION_POLICY_VERSION

    policy: ClassVar[str] = "soft_target"
    default_template_version: ClassVar[int] = 3

    def select_action(self, sampled_target_share: float, rng: Any) -> tuple[str, float]:
        """The policy step, with an open-loop escape hatch.

        Under ``always`` the sensed share is deliberately not consulted and the
        seeded stream is deliberately not drawn from, so the schedule is exactly
        reproducible and independent of the population trajectory.
        """

        if self.advocacy_schedule == SCHEDULE_ALWAYS:
            return ADVOCATE_TARGET, 1.0
        if self.advocacy_schedule == SCHEDULE_NEVER:
            return NO_OP, 0.0
        # Explicit rather than `super()`: `@dataclass(slots=True)` rebuilds the
        # class, which leaves the zero-argument form's `__class__` cell pointing
        # at the discarded original.  The parent does the same for the same
        # reason.
        return SoftTargetControl.select_action(self, sampled_target_share, rng)

    @property
    def transmits_fact(self) -> bool:
        return self.message_mode == RECOMMENDATION_PLUS_FACT

    @property
    def transmits_recommendation(self) -> bool:
        """Whether an actuated slot is occupied at all.

        ``False`` only under ``silent``, where the slot is vacated rather than
        filled: the runtime omits the source entirely instead of substituting
        one, so nothing about the controller reaches the rendered prompt.
        """

        return self.message_mode != SILENT

    def resolved_target_for_task(self, task: RelationalTask, episode_seed: int) -> str:
        if isinstance(self.target, int):
            if self.target < 0 or self.target >= len(task.semantic_answers):
                raise ValueError(
                    f"controller target index {self.target} is outside task options"
                )
            return task.semantic_answers[self.target]
        if self.target == "correct":
            return task.correct_relation
        if self.target == "random_incorrect":
            from mas_cc.core import Seed

            candidates = tuple(
                x for x in task.semantic_answers if x != task.correct_relation
            )
            rng = (
                Seed(episode_seed)
                .derive(
                    f"hidden-bench-imitation-random-incorrect-target:{task.task_id}"
                )
                .create_random()
            )
            return rng.choice(candidates)
        return str(self.target)

    def resolve_fact_id(
        self, task: RelationalTask, episode_seed: int = 0
    ) -> str | None:
        """The one fact this controller cites for the whole episode, or ``None``.

        Resolution is deterministic and validated against the frozen task, so a
        configuration that names a fact the task does not contain fails at
        launch rather than quietly advocating without evidence.
        """

        if not self.transmits_fact:
            return None
        if self.controller_fact_id is not None:
            if self.controller_fact_id not in task.facts:
                raise ValueError(
                    f"control.options.controller_fact_id {self.controller_fact_id!r} is "
                    f"not a fact of task {task.task_id!r}"
                )
            return self.controller_fact_id
        if self.controller_evidence_strategy == EVIDENCE_NEUTRAL:
            # The first frozen fact is a stable rule independent of target and seed.
            return task.fact_order[0]
        if self.controller_evidence_strategy == EVIDENCE_STRATEGIC:
            target = self.resolved_target_for_task(task, episode_seed)
            if target not in _DIRECTION_VECTORS:
                raise ValueError(
                    f"task {task.task_id!r} target {target!r} has no symbolic direction"
                )
            tx, ty = _DIRECTION_VECTORS[target]
            scored: list[tuple[float, int, str]] = []
            for index, fact_id in enumerate(task.fact_order):
                relation = task.fact(fact_id).relation
                if relation not in _DIRECTION_VECTORS:
                    continue
                fx, fy = _DIRECTION_VECTORS[relation]
                score = (tx * fx + ty * fy) / math.sqrt(
                    (tx * tx + ty * ty) * (fx * fx + fy * fy)
                )
                if score > 0:
                    scored.append((score, -index, fact_id))
            if not scored:
                raise ValueError(
                    f"task {task.task_id!r}/target {target!r} is strategically "
                    "inadmissible: no real fact has positive directional alignment"
                )
            return max(scored)[2]
        if self.controller_fact_selector == SELECTOR_SUPPORTING:
            if not task.supporting_fact_ids:
                raise ValueError(
                    f"task {task.task_id!r} has no supporting facts for "
                    "controller_fact_selector 'supporting'"
                )
            return task.supporting_fact_ids[0]
        raise ValueError(
            "message_mode 'recommendation_plus_fact' requires either "
            "control.options.controller_fact_id or control.options.controller_fact_selector"
        )

    def coordination_request_text(
        self,
        target: str,
        sampled_opinion_counts: Mapping[str, Any],
        answer_display_texts: Mapping[str, str] | None = None,
    ) -> str:
        """Create an auditable request without inventing task evidence."""

        rivals = {
            str(option): int(count)
            for option, count in sampled_opinion_counts.items()
            if str(option) != target
        }
        rival: str | None = None
        if rivals:
            maximum = max(rivals.values())
            strongest = sorted(
                option for option, count in rivals.items() if count == maximum
            )
            if len(strongest) == 1:
                rival = strongest[0]
        target_text = (answer_display_texts or {}).get(target, target)
        if rival is None:
            return (
                f"Please share evidence that bears on whether {target_text} is the "
                "best allocation. Report evidence that supports or contradicts it."
            )
        rival_text = (answer_display_texts or {}).get(rival, rival)
        return (
            f"Please share evidence that helps distinguish {target_text} from "
            f"{rival_text}. If you have evidence supporting or contradicting "
            "either allocation, report it."
        )

    def coordination_directive_text(
        self,
        target: str,
        sampled_opinion_counts: Mapping[str, Any],
        answer_display_texts: Mapping[str, str] | None = None,
    ) -> str:
        """Coordinate comparison work without presenting a fact or asking a question."""

        rivals = {
            str(option): int(count)
            for option, count in sampled_opinion_counts.items()
            if str(option) != target
        }
        rival = max(sorted(rivals), key=rivals.get) if rivals else None
        target_text = (answer_display_texts or {}).get(target, target)
        if rival is None:
            return (
                f"Please compare the available evidence for and against {target_text} "
                "before deciding."
            )
        rival_text = (answer_display_texts or {}).get(rival, rival)
        return (
            f"Please compare the evidence for {target_text} and {rival_text}, "
            "including both ability and cooperation evidence, before deciding."
        )

    def validate_truthful_report_task(
        self, task: RelationalTask, episode_seed: int = 0
    ) -> None:
        """Fail before provider use unless the task has a valid frozen pool."""

        if self.controller_actuation_mode not in {
            TRUTHFUL_STRATEGIC_REPORT,
            ADAPTIVE_COMMUNICATION,
        }:
            return
        target = self.resolved_target_for_task(task, episode_seed)
        if task.task_family != "musr_team_allocation":
            raise ValueError(
                "truthful_strategic_report requires a MuSR Team Allocation task"
            )
        if task.controller_target is None:
            raise ValueError(
                f"task {task.task_id!r} has no frozen truthful-controller design"
            )
        if target != task.controller_target:
            raise ValueError(
                f"controller target {target!r} does not match task-declared target "
                f"{task.controller_target!r}"
            )
        if target == task.correct_relation:
            raise ValueError("truthful strategic controller target must be false")
        pool = task.controller_reportable_fact_ids
        if self.intervention_budget > len(pool):
            raise ValueError(
                "control.options.intervention_budget exceeds the distinct "
                f"controller-reportable pool ({self.intervention_budget} > {len(pool)})"
            )
        missing = set(pool) - set(task.facts)
        if missing:
            raise ValueError(
                f"controller-reportable pool references unknown facts: {sorted(missing)}"
            )

    def select_truthful_reports(
        self,
        task: RelationalTask,
        *,
        episode_seed: int,
        round_index: int,
        live_fact_counts: Mapping[str, int],
        selected_rounds: Mapping[str, Sequence[int]],
    ) -> tuple[StrategicReportSelection, ...]:
        """Rank true frozen facts without consuming the action-policy RNG stream."""

        self.validate_truthful_report_task(task, episode_seed)
        if self.controller_actuation_mode not in {
            TRUTHFUL_STRATEGIC_REPORT,
            ADAPTIVE_COMMUNICATION,
        }:
            return ()
        classes = task.controller_fact_classes or {}
        base_scores = task.controller_fact_scores or {}
        ranked: list[tuple[tuple[Any, ...], str, float, bool, bool]] = []
        for fact_id in task.controller_reportable_fact_ids:
            prior_rounds = tuple(
                int(value) for value in selected_rounds.get(fact_id, ())
            )
            cooldown_eligible = not prior_rounds or (
                round_index - max(prior_rounds)
                >= self.controller_report_cooldown_rounds
            )
            live_count = int(live_fact_counts.get(fact_id, 0))
            novel = live_count == 0
            reuse_count = len(prior_rounds)
            base_score = float(base_scores.get(fact_id, 0.0))
            score = base_score + float(novel) - live_count - reuse_count
            tie = hashlib.sha256(
                f"{episode_seed}:{task.task_id}:{round_index}:{fact_id}".encode("utf-8")
            ).hexdigest()
            ranked.append(
                (
                    (
                        not cooldown_eligible,
                        reuse_count,
                        live_count,
                        -base_score,
                        tie,
                        fact_id,
                    ),
                    fact_id,
                    score,
                    novel,
                    cooldown_eligible,
                )
            )
        ranked.sort(key=lambda item: item[0])
        selected = ranked[: self.intervention_budget]
        return tuple(
            StrategicReportSelection(
                fact_id=fact_id,
                score=score,
                strategy_class=str(classes[fact_id]),
                novel_on_live_board=novel,
                cooldown_eligible=cooldown_eligible,
                rank=rank,
            )
            for rank, (_, fact_id, score, novel, cooldown_eligible) in enumerate(
                selected, start=1
            )
        )

    def select_adaptive_truthful_reports(
        self,
        task: RelationalTask,
        *,
        episode_seed: int,
        round_index: int,
        live_fact_counts: Mapping[str, int],
        selected_rounds: Mapping[str, Sequence[int]],
    ) -> tuple[StrategicReportSelection, ...]:
        """Return at most b useful reports without repeating prior controller facts."""

        selected = replace(
            self, intervention_budget=len(task.controller_reportable_fact_ids)
        ).select_truthful_reports(
            task,
            episode_seed=episode_seed,
            round_index=round_index,
            live_fact_counts=live_fact_counts,
            selected_rounds=selected_rounds,
        )
        useful = tuple(
            row
            for row in selected
            if row.fact_id not in selected_rounds
            and row.novel_on_live_board
            and row.cooldown_eligible
        )
        return useful[: self.intervention_budget]

    @classmethod
    def _extra_from_options(
        cls, options: Mapping[str, Any], issues: list[ValidationIssue]
    ) -> dict[str, Any]:
        # `beta` from the soft policy, deliberately *not* the HiddenBench
        # round controller's `_extra_from_options`: that one validates
        # `evidence_mode` and pins `template_version`, neither of which means
        # anything on a relational task.
        values = SoftTargetControl._extra_from_options(options, issues)

        budget = options.get("intervention_budget", 0)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            issues.append(
                ValidationIssue(
                    "control.options.intervention_budget",
                    "must be a non-negative integer",
                )
            )
        else:
            values["intervention_budget"] = budget

        if "evidence_mode" in options:
            issues.append(
                ValidationIssue(
                    "control.options.evidence_mode",
                    "is a HiddenBench option; this controller uses message_mode "
                    f"(one of {list(MESSAGE_MODES)})",
                )
            )
        values["evidence_mode"] = EVIDENCE_NONE

        mode = options.get("message_mode", RECOMMENDATION_ONLY)
        if mode not in MESSAGE_MODES:
            issues.append(
                ValidationIssue(
                    "control.options.message_mode",
                    f"must be one of {list(MESSAGE_MODES)}",
                )
            )
            mode = RECOMMENDATION_ONLY
        values["message_mode"] = str(mode)

        fact_id = options.get("controller_fact_id")
        if fact_id is not None and (
            not isinstance(fact_id, str) or not fact_id.strip()
        ):
            issues.append(
                ValidationIssue(
                    "control.options.controller_fact_id",
                    "must be a fact identifier such as f1",
                )
            )
            fact_id = None
        values["controller_fact_id"] = None if fact_id is None else str(fact_id).strip()

        schedule = options.get("advocacy_schedule", SCHEDULE_SOFT)
        if schedule not in ADVOCACY_SCHEDULES:
            issues.append(
                ValidationIssue(
                    "control.options.advocacy_schedule",
                    f"must be one of {list(ADVOCACY_SCHEDULES)}",
                )
            )
            schedule = SCHEDULE_SOFT
        values["advocacy_schedule"] = str(schedule)

        actuation_mode = options.get("controller_actuation_mode", DIRECT_RECOMMENDATION)
        if actuation_mode not in CONTROLLER_ACTUATION_MODES:
            issues.append(
                ValidationIssue(
                    "control.options.controller_actuation_mode",
                    f"must be one of {list(CONTROLLER_ACTUATION_MODES)}",
                )
            )
            actuation_mode = DIRECT_RECOMMENDATION
        values["controller_actuation_mode"] = str(actuation_mode)

        timing = options.get("controller_timing", TIMING_MICROSCOPIC)
        if timing not in CONTROLLER_TIMINGS:
            issues.append(
                ValidationIssue(
                    "control.options.controller_timing",
                    f"must be one of {list(CONTROLLER_TIMINGS)}",
                )
            )
            timing = TIMING_MICROSCOPIC
        if timing == TIMING_DAWN_ONLY and actuation_mode not in {
            COORDINATION_REQUEST,
            TRUTHFUL_STRATEGIC_REPORT,
            ADAPTIVE_COMMUNICATION,
        }:
            issues.append(
                ValidationIssue(
                    "control.options.controller_timing",
                    "dawn_only requires controller_actuation_mode "
                    "'coordination_request', 'truthful_strategic_report', or "
                    "'adaptive_communication'",
                )
            )
        values["controller_timing"] = str(timing)

        allow_requests = options.get("allow_controller_requests", True)
        if not isinstance(allow_requests, bool):
            issues.append(
                ValidationIssue(
                    "control.options.allow_controller_requests",
                    "must be a boolean",
                )
            )
            allow_requests = True
        values["allow_controller_requests"] = allow_requests

        allow_directives = options.get("allow_controller_directives", True)
        if not isinstance(allow_directives, bool):
            issues.append(
                ValidationIssue(
                    "control.options.allow_controller_directives",
                    "must be a boolean",
                )
            )
            allow_directives = True
        values["allow_controller_directives"] = allow_directives

        communication_policy = options.get(
            "controller_communication_policy", COMMUNICATION_POLICY
        )
        if communication_policy != COMMUNICATION_POLICY:
            issues.append(
                ValidationIssue(
                    "control.options.controller_communication_policy",
                    f"must be {COMMUNICATION_POLICY!r}",
                )
            )
            communication_policy = COMMUNICATION_POLICY
        values["controller_communication_policy"] = str(communication_policy)

        communication_policy_version = options.get(
            "controller_communication_policy_version",
            COMMUNICATION_POLICY_VERSION,
        )
        if communication_policy_version != COMMUNICATION_POLICY_VERSION:
            issues.append(
                ValidationIssue(
                    "control.options.controller_communication_policy_version",
                    f"must be {COMMUNICATION_POLICY_VERSION}",
                )
            )
            communication_policy_version = COMMUNICATION_POLICY_VERSION
        values["controller_communication_policy_version"] = int(
            communication_policy_version
        )

        selector = options.get("controller_fact_selector")
        if selector is not None and selector not in FACT_SELECTORS:
            issues.append(
                ValidationIssue(
                    "control.options.controller_fact_selector",
                    f"must be one of {list(FACT_SELECTORS)}",
                )
            )
            selector = None
        values["controller_fact_selector"] = None if selector is None else str(selector)

        strategy = options.get("controller_evidence_strategy")
        if strategy is not None and strategy not in CONTROLLER_EVIDENCE_STRATEGIES:
            issues.append(
                ValidationIssue(
                    "control.options.controller_evidence_strategy",
                    f"must be one of {list(CONTROLLER_EVIDENCE_STRATEGIES)}",
                )
            )
            strategy = None
        values["controller_evidence_strategy"] = strategy

        cooldown = options.get("controller_report_cooldown_rounds", 1)
        if isinstance(cooldown, bool) or not isinstance(cooldown, int) or cooldown < 0:
            issues.append(
                ValidationIssue(
                    "control.options.controller_report_cooldown_rounds",
                    "must be a non-negative integer",
                )
            )
            cooldown = 1
        values["controller_report_cooldown_rounds"] = cooldown

        report_strategy = options.get(
            "controller_report_selection_strategy", STRATEGIC_REPORT_SELECTION_V1
        )
        if report_strategy not in STRATEGIC_REPORT_SELECTION_STRATEGIES:
            issues.append(
                ValidationIssue(
                    "control.options.controller_report_selection_strategy",
                    f"must be one of {list(STRATEGIC_REPORT_SELECTION_STRATEGIES)}",
                )
            )
            report_strategy = STRATEGIC_REPORT_SELECTION_V1
        values["controller_report_selection_strategy"] = str(report_strategy)

        evidence_sources = sum(
            bool(value)
            for value in (
                values["controller_fact_id"],
                values["controller_fact_selector"],
                strategy,
            )
        )
        if evidence_sources > 1:
            issues.append(
                ValidationIssue(
                    "control.options.controller_fact_id",
                    "cannot be combined with controller_fact_selector or "
                    "controller_evidence_strategy; the citation "
                    "must have exactly one deterministic source",
                )
            )
        if mode == RECOMMENDATION_PLUS_FACT and not (
            values["controller_fact_id"]
            or values["controller_fact_selector"]
            or strategy
        ):
            issues.append(
                ValidationIssue(
                    "control.options.message_mode",
                    "'recommendation_plus_fact' requires controller_evidence_strategy "
                    "(or the legacy controller_fact_id/controller_fact_selector)",
                )
            )
        if mode in FACTLESS_MESSAGE_MODES and (
            values["controller_fact_id"]
            or values["controller_fact_selector"]
            or strategy
        ):
            issues.append(
                ValidationIssue(
                    "control.options.message_mode",
                    f"{mode!r} transmits no fact; remove "
                    "controller_fact_id/controller_fact_selector or switch mode",
                )
            )
        if actuation_mode == COORDINATION_REQUEST and mode != RECOMMENDATION_ONLY:
            issues.append(
                ValidationIssue(
                    "control.options.message_mode",
                    "coordination_request requires recommendation_only because it "
                    "never injects controller evidence",
                )
            )
        if actuation_mode == TRUTHFUL_STRATEGIC_REPORT:
            if timing != TIMING_DAWN_ONLY:
                issues.append(
                    ValidationIssue(
                        "control.options.controller_timing",
                        "truthful_strategic_report requires dawn_only timing",
                    )
                )
            if mode != RECOMMENDATION_ONLY:
                issues.append(
                    ValidationIssue(
                        "control.options.message_mode",
                        "truthful_strategic_report uses the frozen report pool; "
                        "message_mode must be recommendation_only",
                    )
                )
        if actuation_mode == ADAPTIVE_COMMUNICATION:
            if isinstance(budget, int) and not isinstance(budget, bool) and budget == 0:
                issues.append(
                    ValidationIssue(
                        "control.options.intervention_budget",
                        "adaptive_communication requires at least one communication slot",
                    )
                )
            if timing != TIMING_DAWN_ONLY:
                issues.append(
                    ValidationIssue(
                        "control.options.controller_timing",
                        "adaptive_communication requires dawn_only timing",
                    )
                )
            if mode != RECOMMENDATION_ONLY:
                issues.append(
                    ValidationIssue(
                        "control.options.message_mode",
                        "adaptive_communication uses typed board messages; "
                        "message_mode must be recommendation_only",
                    )
                )
        return values


def create_relational_round_budgeted_control(config: ControlConfig) -> Control:
    return RelationalRoundBudgetedControl.from_options(config.options)


__all__ = [
    "ADAPTIVE_COMMUNICATION",
    "CONTROLLER_ACTUATION_MODES",
    "CONTROLLER_TIMINGS",
    "COORDINATION_REQUEST",
    "DIRECT_RECOMMENDATION",
    "TRUTHFUL_STRATEGIC_REPORT",
    "ADVOCACY_SCHEDULES",
    "FACTLESS_MESSAGE_MODES",
    "FACT_SELECTORS",
    "CONTROLLER_EVIDENCE_STRATEGIES",
    "EVIDENCE_NEUTRAL",
    "EVIDENCE_STRATEGIC",
    "MESSAGE_MODES",
    "SILENT",
    "RECOMMENDATION_ONLY",
    "RECOMMENDATION_PLUS_FACT",
    "SCHEDULE_ALWAYS",
    "SCHEDULE_NEVER",
    "SCHEDULE_SOFT",
    "TIMING_DAWN_ONLY",
    "TIMING_MICROSCOPIC",
    "SELECTOR_SUPPORTING",
    "STRATEGIC_REPORT_SELECTION_STRATEGIES",
    "STRATEGIC_REPORT_SELECTION_V1",
    "StrategicReportSelection",
    "RelationalRoundBudgetedControl",
    "create_relational_round_budgeted_control",
]
