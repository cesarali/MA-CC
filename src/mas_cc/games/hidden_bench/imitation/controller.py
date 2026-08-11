"""Provider-free measurement and feedback control for the imitation game."""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mas_cc.config import ControlConfig
from mas_cc.control import Control, InteractionControlSignal
from mas_cc.core import AgentId
from mas_cc.games.protocols import GameState
from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.llm_runtime.validation import ValidationIssue

NO_OP = "NO_OP"
ADVOCATE_TARGET = "ADVOCATE_Z"
ADVOCACY_TEMPLATE_VERSION = 1
ADVOCACY_TEMPLATE_VERSIONS = (1, 2)


def advocacy_message(target: str) -> str:
    """`template_version: 1` - the original text, which announces itself.

    Kept because every controlled run recorded before 2026-08-11 used it and
    those numbers have to stay reproducible. It is also leak 1 of §2.1 of
    `docs/tdd/misselaneous/11082026_prompt_modifications_v2.md`: an agent told
    that an "external controller" is advocating can discount the turn as an
    outside voice, so `R_ctrl` measured under it is not social control.
    """

    return (
        f"The external controller currently advocates option {target}. "
        "Reconsider your current position before committing your next vote."
    )


# --------------------------------------------------------------------------
# template_version 2 - the controller as an indistinguishable peer (§2)
# --------------------------------------------------------------------------

FORBIDDEN_FRAME_TERMS = (
    "controller",
    "system",
    "experiment",
    "external",
    "your position",
    "reconsider",
    "commit",
)
"""§2.4, checked against the *frame* - the words the controller itself
contributes. It cannot be checked against the whole message: nine corpus facts
legitimately say "air purification system" and two say "committed"."""

FORBIDDEN_MESSAGE_TERMS = ("controller", "external", "experiment", "simulation")
"""The identity words, checked against the rendered message. No shared fact in
the corpus contains any of them, so a hit here is a real leak."""

PEER_ADVOCACY_FRAMES: tuple[tuple[str, str], ...] = (
    (
        "peer-advocacy-v2-a",
        "{fact} That makes **{target}** look like the safest option to me. "
        "I'd rather we all end up in the same place. I'm voting **{target}**.",
    ),
    (
        "peer-advocacy-v2-b",
        "Going back over what we all have, **{target}** still holds up best. {fact} "
        "I think that's the most reliable choice for everyone. I'm voting **{target}**.",
    ),
    (
        "peer-advocacy-v2-c",
        "{fact} On balance **{target}** seems like the option that works out best for "
        "the group. I'm voting **{target}**.",
    ),
    (
        "peer-advocacy-v2-d",
        "I keep coming back to **{target}**. {fact} It would be good if we landed on "
        "this together. I'm voting **{target}**.",
    ),
)
"""§2.3/2.5: a reason drawn only from the shared facts, a soft coordination
appeal, then an explicit vote - four paraphrases with stable IDs so no single
wording can silently carry the whole effect. There is no `{fact}` invention
slot on purpose: the controller never receives the hidden facts (§2.2), because
a controller that manufactures evidence turns "the population moved toward Z"
into "an agent believed a new fact"."""

FACTLESS_ADVOCACY_FRAMES: tuple[tuple[str, str], ...] = (
    (
        "peer-advocacy-v2-plain-a",
        "Weighing it all up, **{target}** still looks like the safest option to me. "
        "I'd rather we all end up in the same place. I'm voting **{target}**.",
    ),
    (
        "peer-advocacy-v2-plain-b",
        "I keep coming back to **{target}** as the most reliable choice for the group. "
        "I'm voting **{target}**.",
    ),
)
"""Used only for a task with no shared information at all, where there is no
fact the controller is allowed to cite."""


def _sentence_count(text: str) -> int:
    return len([part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part])


def check_peer_style(text: str, *, target: str) -> tuple[str, ...]:
    """§2.4 as assertions rather than prose. Returns the violated rules.

    Length is checked on the *frame* (`check_frames`), not here: a cited fact is
    a verbatim corpus sentence or two, and trimming one to hit a sentence count
    would mean the controller paraphrasing evidence, which is exactly the thing
    §2.2 forbids.
    """

    problems: list[str] = []
    lowered = text.lower()
    for term in FORBIDDEN_MESSAGE_TERMS:
        if term in lowered:
            problems.append(f"message contains the forbidden term {term!r}")
    if f"**{target}**" not in text:
        problems.append("message does not bold the option name")
    if not text.rstrip().endswith(f"I'm voting **{target}**."):
        problems.append("message does not end with an explicit first-person vote")
    return tuple(problems)


def check_frames() -> tuple[str, ...]:
    """Every shipped frame obeys §2.4 once a one-sentence fact is slotted in."""

    problems: list[str] = []
    for template_id, frame in (*PEER_ADVOCACY_FRAMES, *FACTLESS_ADVOCACY_FRAMES):
        lowered = frame.lower()
        for term in FORBIDDEN_FRAME_TERMS:
            if term in lowered:
                problems.append(f"{template_id}: frame contains the forbidden term {term!r}")
        rendered = frame.format(fact="A single sentence of shared evidence.", target="Z")
        problems.extend(f"{template_id}: {item}" for item in check_peer_style(rendered, target="Z"))
        if not 2 <= _sentence_count(rendered) <= 4:
            problems.append(f"{template_id}: frame is not two to four sentences")
    return tuple(problems)


def _preferred_facts(shared: Sequence[str], target: str) -> tuple[tuple[int, str], ...]:
    """Shared facts that make a short, on-target reason.

    Preference order, each falling through only when it would leave nothing:
    facts that name the target (a reason has to point at Z to be a reason for
    Z), then facts of at most two sentences (so the controller reads at the
    length a peer writes at). The whole pool is the last resort, which keeps the
    controller working on tasks whose facts never name an option outright.
    """

    indexed = tuple((index, str(fact).strip()) for index, fact in enumerate(shared))
    on_target = tuple(item for item in indexed if target.lower() in item[1].lower()) or indexed
    return tuple(item for item in on_target if _sentence_count(item[1]) <= 2) or on_target


@dataclass(frozen=True, slots=True)
class PeerAdvocacy:
    template_id: str
    fact_index: int | None
    text: str


def peer_advocacy_message(
    target: str, shared_information: Sequence[str], rng: random.Random
) -> PeerAdvocacy:
    """One paraphrase of "a peer argues for Z from the shared facts"."""

    candidates = _preferred_facts(shared_information, target)
    if candidates:
        template_id, frame = rng.choice(PEER_ADVOCACY_FRAMES)
        fact_index, fact = rng.choice(candidates)
        text = frame.format(fact=fact.strip(), target=target)
    else:
        template_id, frame = rng.choice(FACTLESS_ADVOCACY_FRAMES)
        fact_index, text = None, frame.format(target=target)
    problems = check_peer_style(text, target=target)
    if problems:
        raise ValueError(
            f"controller message for target {target!r} violates the peer-style rules: "
            + "; ".join(problems)
        )
    return PeerAdvocacy(template_id=template_id, fact_index=fact_index, text=text)


def _shared_information(state: GameState) -> tuple[str, ...]:
    for agent in state.agents:
        shared = agent.attributes.get("shared_information")
        if shared:
            return tuple(str(fact) for fact in shared)
    return ()


@dataclass(frozen=True, slots=True)
class ThresholdTargetControl(Control):
    """Hypergeometric opinion sensor followed by a deterministic threshold policy."""

    target: str | int = "correct"
    sensor_sample_size: int = 1
    threshold: float = 0.5
    policy: str = "threshold_target"
    template_version: int = ADVOCACY_TEMPLATE_VERSION

    def override(
        self, *, agent_id: AgentId, interaction_index: int, state: GameState
    ) -> str | None:
        # Feedback shapes the interaction.  It never overwrites the vote.
        return None

    def _resolved_target(self, state: GameState) -> str:
        if isinstance(self.target, int):
            task = state.data.get("task", {})
            options = task.get("possible_answers", ()) if isinstance(task, Mapping) else ()
            if self.target < 0 or self.target >= len(options):
                raise ValueError(
                    f"controller target index {self.target} is outside [0, {len(options)})"
                )
            return str(options[self.target])
        if self.target != "correct":
            return self.target
        task = state.data.get("task", {})
        if not isinstance(task, Mapping) or not task.get("correct_answer"):
            raise ValueError("target 'correct' requires state.data.task.correct_answer")
        return str(task["correct_answer"])

    def interaction_signal(
        self,
        *,
        agent_id: AgentId,
        interaction_index: int,
        state: GameState,
        rng: Any,
    ) -> InteractionControlSignal:
        if self.sensor_sample_size > len(state.agents):
            raise ValueError(
                "controller sensor_sample_size cannot exceed the population size"
            )
        target = self._resolved_target(state)
        sampled = rng.sample(list(state.agents), self.sensor_sample_size)
        opinions = [
            None
            if agent.attributes.get("committed_action") is None
            else str(agent.attributes["committed_action"])
            for agent in sampled
        ]
        counts = Counter(value for value in opinions if value is not None)
        support = counts.get(target, 0) / self.sensor_sample_size
        action = ADVOCATE_TARGET if support < self.threshold else NO_OP
        message, template_id, fact_index = None, None, None
        if action == ADVOCATE_TARGET:
            if self.template_version == 1:
                message, template_id = advocacy_message(target), "labelled-advocacy-v1"
            else:
                advocacy = peer_advocacy_message(target, _shared_information(state), rng)
                message = advocacy.text
                template_id, fact_index = advocacy.template_id, advocacy.fact_index
        return InteractionControlSignal(
            action=action,
            target=target,
            message=message,
            observation={
                "sampled_agent_ids": [str(agent.agent_id) for agent in sampled],
                "sampled_opinions": opinions,
                "sampled_opinion_counts": dict(counts),
                "sample_size": self.sensor_sample_size,
            },
            metadata={
                "policy": self.policy,
                "threshold": self.threshold,
                "target_support": support,
                "message_template_version": self.template_version,
                "message_template_id": template_id,
                "message_fact_index": fact_index,
            },
        )

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "ThresholdTargetControl":
        target = options.get("target", "correct")
        sample_size = options.get("sensor_sample_size", 1)
        threshold = options.get("threshold", 0.5)
        policy = options.get("policy", "threshold_target")
        template_version = options.get("template_version", ADVOCACY_TEMPLATE_VERSION)
        issues: list[ValidationIssue] = []
        if (
            isinstance(target, bool)
            or not isinstance(target, (str, int))
            or (isinstance(target, str) and not target.strip())
            or (isinstance(target, int) and target < 0)
        ):
            issues.append(
                ValidationIssue(
                    "control.options.target",
                    "must be 'correct', an option label, or a non-negative zero-based index",
                )
            )
        if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
            issues.append(
                ValidationIssue("control.options.sensor_sample_size", "must be a positive integer")
            )
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0 <= float(threshold) <= 1
        ):
            issues.append(ValidationIssue("control.options.threshold", "must be between 0 and 1"))
        if policy != "threshold_target":
            issues.append(
                ValidationIssue(
                    "control.options.policy", "only 'threshold_target' is implemented"
                )
            )
        if (
            isinstance(template_version, bool)
            or not isinstance(template_version, int)
            or template_version not in ADVOCACY_TEMPLATE_VERSIONS
        ):
            issues.append(
                ValidationIssue(
                    "control.options.template_version",
                    f"must be one of {list(ADVOCACY_TEMPLATE_VERSIONS)}: "
                    "1 announces the controller, 2 speaks as a peer",
                )
            )
        if issues:
            raise ConfigurationError(issues, context="control creation")
        return cls(
            target=target,
            sensor_sample_size=int(sample_size),
            threshold=float(threshold),
            policy=str(policy),
            template_version=int(template_version),
        )


def create_threshold_target_control(config: ControlConfig) -> Control:
    return ThresholdTargetControl.from_options(config.options)


def controller_from_game_options(options: Mapping[str, Any]) -> ThresholdTargetControl | None:
    """Compatibility adapter for the plan's nested ``game.options.controller`` form."""

    raw = options.get("controller")
    if not isinstance(raw, Mapping) or not bool(raw.get("enabled", False)):
        return None
    return ThresholdTargetControl.from_options(raw)
