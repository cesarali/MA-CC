"""One-sense/one-action budgeted feedback controller.

`template_version` is **not** an experimental axis here and the `message` this
controller carries is **not** what a reasoning-mode agent sees. Under atomic
public ballots the controller's whole visible contribution is one social slot's
`(vote, reason)` pair, rendered by
`prompts.render_control_reason` - one fixed, fact-free renderer per target.
The option is still validated (and pinned to `3`) so the shipped configs and
the classical reference path keep loading unchanged.

`evidence_mode` is the one thing that varies that rendering, and it varies only
the *justification*: `shared_fact` appends one verbatim fact from the task's
shared set `Is` to the advocacy line. `Is` is what every agent already holds,
so a controlled prompt still contains no information the population lacked -
see `select_shared_evidence_fact`.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from mas_cc.config import ControlConfig
from mas_cc.control import Control, RoundControlSignal
from mas_cc.games.protocols import GameState
from mas_cc.llm_runtime.validation import ValidationIssue

from ..imitation.controller import (
    FORBIDDEN_MESSAGE_TERMS,
    SoftTargetControl,
    ThresholdTargetControl,
    _preferred_facts,
    _shared_information,
)

EVIDENCE_NONE = "none"
EVIDENCE_SHARED_FACT = "shared_fact"
CONTROL_EVIDENCE_MODES = (EVIDENCE_NONE, EVIDENCE_SHARED_FACT)
"""`control.options.evidence_mode`.  `none` is the historical behaviour and the
default, so an existing config renders byte-identical advocacy."""


def select_shared_evidence_fact(
    state: GameState, target: str, rng: random.Random
) -> tuple[int | None, str | None]:
    """One verbatim shared fact the controller may cite for `target`, or nothing.

    The pool is `Is` alone, read off the agents' own `shared_information`, which
    is the same text their prompts already contain - the unshared pool `Iu` is
    not reachable from this function, so citing a fact cannot add information to
    the episode.

    Relevance is the parent module's `_preferred_facts`: facts that name `Z`
    first, short ones next, the whole shared pool as the last resort.  That is
    the only mapping from facts to options the corpus actually carries, so on a
    task whose facts never name an option this degrades to a uniform draw rather
    than to a semantic guess.  The draw uses the caller's seeded stream and
    therefore replays.

    Returns `(index into Is, fact)`, or `(None, None)` when there is nothing
    citable - an empty shared set, or one whose every fact would name the
    apparatus and blow the controller's cover.
    """

    candidates = tuple(
        item
        for item in _preferred_facts(_shared_information(state), target)
        if not any(term in item[1].lower() for term in FORBIDDEN_MESSAGE_TERMS)
    )
    if not candidates:
        return None, None
    index, fact = rng.choice(candidates)
    return index, fact


@dataclass(frozen=True, slots=True)
class RoundSoftTargetBudgetedControl(SoftTargetControl):
    """Soft target policy sampled once, with an exact per-round slot budget."""

    intervention_budget: int = 0
    evidence_mode: str = EVIDENCE_NONE
    policy: ClassVar[str] = "soft_target"
    default_template_version: ClassVar[int] = 3

    def interaction_signal(self, **_: Any) -> None:
        """This controller deliberately has no event-clock behavior."""

        return None

    def round_signal(
        self,
        *,
        round_index: int,
        state: GameState,
        rng: Any,
    ) -> RoundControlSignal:
        # Call the event controller's implementation as a pure sensor/policy
        # helper.  The round runtime invokes this method once and owns all
        # schedule/actuation semantics after it returns.
        signal = ThresholdTargetControl.interaction_signal(
            self,
            agent_id=state.agents[0].agent_id,
            interaction_index=round_index + 1,
            state=state,
            rng=rng,
        )
        return RoundControlSignal(
            action=signal.action,
            target=signal.target,
            message=signal.message,
            observation=signal.observation,
            metadata={
                **dict(signal.metadata),
                "intervention_budget": self.intervention_budget,
                "round_index": round_index,
            },
        )

    @classmethod
    def _extra_from_options(
        cls, options: Mapping[str, Any], issues: list[ValidationIssue]
    ) -> dict[str, Any]:
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
        mode = options.get("evidence_mode", EVIDENCE_NONE)
        if mode not in CONTROL_EVIDENCE_MODES:
            issues.append(
                ValidationIssue(
                    "control.options.evidence_mode",
                    f"must be one of {list(CONTROL_EVIDENCE_MODES)}",
                )
            )
        else:
            values["evidence_mode"] = str(mode)
        if options.get("template_version", cls.default_template_version) != 3:
            issues.append(
                ValidationIssue(
                    "control.options.template_version",
                    "round feedback requires fixed peer-style template version 3",
                )
            )
        return values


def create_round_soft_target_budgeted_control(config: ControlConfig) -> Control:
    return RoundSoftTargetBudgetedControl.from_options(config.options)


__all__ = [
    "CONTROL_EVIDENCE_MODES",
    "EVIDENCE_NONE",
    "EVIDENCE_SHARED_FACT",
    "RoundSoftTargetBudgetedControl",
    "create_round_soft_target_budgeted_control",
    "select_shared_evidence_fact",
]
