"""Post-action communication chooser for the MuSR blackboard controller.

The chooser never decides whether the controller acts.  It is called only after
the existing binary policy has produced ``U_k = 1`` and uses an independent
random-number stream, so it cannot perturb later sensor samples or binary
actions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class CommunicationMode(str, Enum):
    REPORT = "REPORT"
    REQUEST = "REQUEST"
    DIRECTIVE = "DIRECTIVE"


COMMUNICATION_POLICY = "contextual_weighted_v1"
COMMUNICATION_POLICY_VERSION = 1


@dataclass(frozen=True, slots=True)
class ControllerCommunicationContext:
    """The explicit, public-information-only input to the chooser."""

    round_index: int
    target: str
    sampled_opinion_counts: Mapping[str, int]
    live_message_type_counts: Mapping[str, int]
    previous_modes: tuple[CommunicationMode, ...] = ()


@dataclass(frozen=True, slots=True)
class CommunicationChoice:
    mode: CommunicationMode
    reason: str
    policy: str = COMMUNICATION_POLICY
    policy_version: int = COMMUNICATION_POLICY_VERSION


def allowed_communication_modes(
    *, allow_requests: bool, allow_directives: bool
) -> tuple[CommunicationMode, ...]:
    """Return the adaptive vocabulary; truthful REPORT is always available."""

    modes = [CommunicationMode.REPORT]
    if allow_requests:
        modes.append(CommunicationMode.REQUEST)
    if allow_directives:
        modes.append(CommunicationMode.DIRECTIVE)
    return tuple(modes)


def choose_communication_mode(
    controller_context: ControllerCommunicationContext,
    allowed_modes: Sequence[CommunicationMode | str],
    rng: Any,
) -> CommunicationChoice:
    """Choose one allowed mode from current public context with a seeded draw.

    The weights favor requests when reports are scarce, reports after requests,
    and coordination after evidence accumulates.  They are contextual rather
    than tied to a particular round number or intervention budget.
    """

    allowed = tuple(CommunicationMode(value) for value in allowed_modes)
    if not allowed:
        raise ValueError("adaptive communication requires at least one allowed mode")
    if len(set(allowed)) != len(allowed):
        raise ValueError("allowed communication modes must be unique")

    counts = controller_context.live_message_type_counts
    reports = int(counts.get(CommunicationMode.REPORT.value, 0))
    requests = int(counts.get(CommunicationMode.REQUEST.value, 0))
    directives = int(counts.get(CommunicationMode.DIRECTIVE.value, 0))
    weights: dict[CommunicationMode, float] = {
        CommunicationMode.REPORT: 2.0 + 1.5 * requests,
        CommunicationMode.REQUEST: 1.0 + (3.0 if reports == 0 else 0.5),
        CommunicationMode.DIRECTIVE: 1.0 + min(3.0, float(reports)) + 0.5 * directives,
    }
    selected = rng.choices(allowed, weights=[weights[mode] for mode in allowed], k=1)[0]
    reason = {
        CommunicationMode.REPORT: "share_verified_evidence",
        CommunicationMode.REQUEST: "seek_missing_public_evidence",
        CommunicationMode.DIRECTIVE: "coordinate_evidence_comparison",
    }[selected]
    return CommunicationChoice(mode=selected, reason=reason)


__all__ = [
    "COMMUNICATION_POLICY",
    "COMMUNICATION_POLICY_VERSION",
    "CommunicationChoice",
    "CommunicationMode",
    "ControllerCommunicationContext",
    "allowed_communication_modes",
    "choose_communication_mode",
]
