"""Paper-verbatim HiddenBench prompts (brief §1.5).

**Source of truth.** The brief quotes the paper's templates from the PDF, with
markdown line wrapping applied. This module instead reproduces
`scripts/local_llms/hiddenbench_population_pipeline/scripts/hiddenbench_evaluation.py`,
which is the already-checked-in, already-run implementation of the same
templates in this repository. Where the two disagree the script wins, because it
is what produced the numbers the user already has. The differences are small but
real and are all in `docs/hidden_bench/data_provenance.md`; the ones that change
rendered characters:

- the line break falls after "these information", not after "are";
- there is **no** blank line between "please reason carefully:" and the facts;
- facts render as `- ` bullets, one per line;
- `%possible_answers%` renders as a Python list repr (`['A', 'B', 'C']`), and
  the JSON skeleton is indented with **two** spaces, not the brief's four;
- shared and private facts are shuffled **together** into one list, so an agent
  cannot recover which of its facts were the private ones from their position.

The paper's own typos ("randomly shuffle", "concise-just") are reproduced
deliberately. A golden-file test pins every one of these
(`tests/games/hidden_bench/test_prompt_fidelity.py`); if you "fix" the grammar,
that test fails, which is the intended behaviour.

`%extra%` (the prompting-strategy hook of §1.6) is a first-class bound value,
never a hardcoded string: it is what cooperative/conflictual/CoT/share-all and
per-agent adversarial-dissenter ablations are injected through.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mas_cc.llm_runtime.messages import MessageRole
from mas_cc.llm_runtime.prompts import UNBOUND, FullPrompt, PromptBlock, ResponseContract, Unbound
from mas_cc.llm_runtime.prompts._values import thaw
from mas_cc.llm_runtime.validation import ValidationIssue, ValidationResult

INFORMATION_PREAMBLE = (
    "You have received the following information, notice the order of these information\n"
    "are randomly shuffle, the order of facts does not indicate importance or relationship,\n"
    "please reason carefully:"
)
"""Verbatim, including the paper's "randomly shuffle"."""

RESPONSE_STYLE = "Keep your response concise-just one or two sentences."
"""Verbatim, including the paper's missing space in "concise-just"."""

FIRST_SPEAKER = "You are the first to speak."
TRANSCRIPT_HEADER = "Previous messages from other people:"
TURN_INSTRUCTION = "It's your turn to speak."


def render_extra(extra: str | None) -> str:
    """`%extra%` as the paper appends it: a leading space, or nothing at all.

    An unset hook must leave *no* trailing whitespace, otherwise every
    no-ablation run carries a prompt that differs from the paper's by one
    character and hashes differently.
    """

    return f" {extra.strip()}" if extra and extra.strip() else ""


def render_facts(information: Sequence[str]) -> str:
    return "\n".join(f"- {fact}" for fact in information)


def render_transcript(transcript: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(f"Agent {item['speaker_id']}: {item['message']}" for item in transcript)


def shuffled_information(shared: Sequence[str], private: Sequence[str], *, seed: int) -> tuple[str, ...]:
    """One shuffled fact list per agent, seeded from the episode RNG (§1.5).

    Shared and private facts go into the *same* shuffle. That is not incidental:
    if private facts always landed last, an agent could infer which of its facts
    nobody else had, and the paper's core claim is that agents are never told
    the information is asymmetric.
    """

    information = [*(str(item) for item in shared), *(str(item) for item in private)]
    random.Random(seed).shuffle(information)
    return tuple(information)


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------


def _text_issues(field_name: str, value: Any) -> tuple[ValidationIssue, ...]:
    if not isinstance(value, str) or not value.strip():
        return (ValidationIssue(f"prompt.blocks.{field_name}.value", "must be non-empty text", value),)
    return ()


@dataclass(frozen=True, slots=True)
class ScenarioBlock(PromptBlock[str]):
    """`%description%` - the scenario, identical for every agent."""

    name: str = field(init=False, default="scenario")
    title: str = field(init=False, default="Scenario")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: str | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: str) -> tuple[ValidationIssue, ...]:
        return _text_issues("scenario", value)

    def render(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class InformationBlock(PromptBlock[tuple[str, ...]]):
    """The preamble plus `%information%` - this agent's own shuffled facts.

    `sensitive=True`: this is the one block whose value is another agent's
    business to never see, so audit records redact it rather than writing every
    agent's private information into a shared log.
    """

    name: str = field(init=False, default="information")
    title: str = field(init=False, default="Received information")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: tuple[str, ...] | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")
    sensitive: bool = field(init=False, default=True)

    def value_issues(self, value: tuple[str, ...]) -> tuple[ValidationIssue, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
            return (
                ValidationIssue("prompt.blocks.information.value", "must be a non-empty sequence", value),
            )
        if any(not isinstance(item, str) or not item.strip() for item in value):
            return (
                ValidationIssue("prompt.blocks.information.value", "items must be non-empty text", value),
            )
        return ()

    def render(self) -> str:
        return f"{INFORMATION_PREAMBLE}\n{render_facts(self.value)}"  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ResponseStyleBlock(PromptBlock[str]):
    """`Keep your response concise-just one or two sentences. %extra%`.

    Bound with the *rendered* extra (already space-prefixed or empty), so the
    block's value is exactly what appears in the message.
    """

    name: str = field(init=False, default="response_style")
    title: str = field(init=False, default="Response style and strategy hook")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: str | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: str) -> tuple[ValidationIssue, ...]:
        return _text_issues("response_style", value)

    def render(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class SpeakingTurnBlock(PromptBlock[str]):
    """The discussion user message: first-speaker or transcript-then-turn."""

    name: str = field(init=False, default="speaking_turn")
    title: str = field(init=False, default="Speaking turn")
    role: MessageRole = field(init=False, default=MessageRole.USER)
    value: str | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: str) -> tuple[ValidationIssue, ...]:
        return _text_issues("speaking_turn", value)

    def render(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class TranscriptBlock(PromptBlock[tuple[Mapping[str, Any], ...]]):
    """`Previous messages from other people:` for the post-discussion vote.

    `required=False`: the *pre*-discussion vote legitimately has no transcript,
    and an unbound optional block is omitted from the compiled prompt entirely
    rather than rendering an empty header - which is what makes one vote prompt
    serve both §1.5 vote variants.
    """

    name: str = field(init=False, default="transcript")
    title: str = field(init=False, default="Group discussion")
    role: MessageRole = field(init=False, default=MessageRole.USER)
    value: tuple[Mapping[str, Any], ...] | Unbound = UNBOUND
    required: bool = field(init=False, default=False)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: tuple[Mapping[str, Any], ...]) -> tuple[ValidationIssue, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return (ValidationIssue("prompt.blocks.transcript.value", "must be a sequence", value),)
        for index, item in enumerate(value):
            if not isinstance(item, Mapping) or {"speaker_id", "message"} - set(item):
                return (
                    ValidationIssue(
                        f"prompt.blocks.transcript.value[{index}]",
                        "must contain speaker_id and message",
                        thaw(item) if isinstance(item, Mapping) else item,
                    ),
                )
        return ()

    def render(self) -> str:
        return f"{TRANSCRIPT_HEADER}\n{render_transcript(self.value)}"  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Response contracts
# --------------------------------------------------------------------------

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(response: str) -> Mapping[str, Any] | None:
    """The first JSON object in `response`, tolerating fences and stray prose.

    Models routinely wrap the vote in ```json fences or precede it with a
    sentence. Treating that as a validation failure would burn a retry on a
    response that is, semantically, a perfectly good vote - and a vote that is
    silently counted as wrong is one of the three failure modes the brief's §9.5
    tells us to suspect when the reproduction numbers drift.
    """

    if not isinstance(response, str):
        return None
    match = _JSON_OBJECT.search(response)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def normalize_vote(value: Any, possible_answers: Sequence[str]) -> str | None:
    """`hiddenbench_common.normalize_vote` - case/whitespace-insensitive match.

    Exact normalized equality first, then a *conservative* substring fallback
    that gives up when more than one option matches, so "not East Town, I say
    West City" never resolves to whichever option happened to be listed first.
    """

    if value is None:
        return None
    text = _norm(str(value))
    for answer in possible_answers:
        if text == _norm(answer):
            return answer
    matches = [answer for answer in possible_answers if _norm(answer) in text]
    return matches[0] if len(matches) == 1 else None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


@dataclass(frozen=True, slots=True)
class DiscussionContract(ResponseContract):
    """A free-text discussion turn: no appended instruction at all.

    Overriding `instruction_messages` to return nothing is the point. The base
    class always appends one trailing instruction message; the paper's
    discussion prompt ends with "It's your turn to speak.", so anything appended
    after it would be a departure from the template.
    """

    type: str = "hidden_bench_discussion_turn"

    def instruction_messages(self) -> tuple[tuple[MessageRole, str], ...]:
        return ()

    def validate(self, response: str) -> ValidationResult:
        if not isinstance(response, str) or not response.strip():
            return ValidationResult.failure(
                ValidationIssue("response", "a discussion turn must be non-empty text", response)
            )
        return ValidationResult.success()


@dataclass(frozen=True, slots=True)
class VoteContract(ResponseContract):
    """The §1.5 vote instruction, rendered exactly as the pipeline renders it."""

    type: str = "hidden_bench_json_vote"

    def instruction(self) -> str:
        answers = list(self.allowed_values)
        return (
            "Please decide and provide your rationale in the following JSON format:\n"
            "{\n"
            f'  "vote": <A string, {answers}>,\n'
            '  "rationale": <A string, representing your rationale>\n'
            "}"
        )

    def instruction_messages(self) -> tuple[tuple[MessageRole, str], ...]:
        return ((MessageRole.USER, self.instruction()),)

    def validate(self, response: str) -> ValidationResult:
        parsed = extract_json_object(response)
        if parsed is None:
            return ValidationResult.failure(
                ValidationIssue("response", "must contain a JSON object with vote and rationale", response)
            )
        if normalize_vote(parsed.get("vote"), self.allowed_values) is None:
            return ValidationResult.failure(
                ValidationIssue(
                    "response.vote",
                    f"must resolve to exactly one of: {', '.join(self.allowed_values)}",
                    parsed.get("vote"),
                )
            )
        if not isinstance(parsed.get("rationale"), str):
            # Rationales are data (§5): a vote without one is a lost datum for
            # the qualitative asymmetry-signalling analysis, so it is retried.
            return ValidationResult.failure(
                ValidationIssue("response.rationale", "must be a string")
            )
        return ValidationResult.success()


# --------------------------------------------------------------------------
# Full prompts
# --------------------------------------------------------------------------


class HiddenBenchDiscussionPrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return "hidden_bench_discussion"


class HiddenBenchVotePrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return "hidden_bench_vote"


def hidden_bench_discussion_prompt() -> HiddenBenchDiscussionPrompt:
    return HiddenBenchDiscussionPrompt(
        "hidden_bench_discussion",
        1,
        (ScenarioBlock(), InformationBlock(), ResponseStyleBlock(), SpeakingTurnBlock()),
        DiscussionContract(),
    )


def hidden_bench_vote_prompt(possible_answers: Sequence[str] = ("A", "B", "C")) -> HiddenBenchVotePrompt:
    return HiddenBenchVotePrompt(
        "hidden_bench_vote",
        1,
        (ScenarioBlock(), InformationBlock(), ResponseStyleBlock(), TranscriptBlock()),
        VoteContract(allowed_values=tuple(possible_answers)),
    )


def bind_discussion_prompt(
    *,
    scenario: str,
    information: Sequence[str],
    extra: str | None,
    transcript: Sequence[Mapping[str, Any]],
    is_first_speaker: bool,
) -> HiddenBenchDiscussionPrompt:
    rendered_extra = render_extra(extra)
    if is_first_speaker:
        turn = FIRST_SPEAKER
    else:
        turn = (
            f"{TRANSCRIPT_HEADER}\n{render_transcript(transcript)}\n\n{TURN_INSTRUCTION}{rendered_extra}"
        )
    return hidden_bench_discussion_prompt().bind(
        scenario=scenario,
        information=tuple(information),
        response_style=f"{RESPONSE_STYLE}{rendered_extra}",
        speaking_turn=turn,
    )


def bind_vote_prompt(
    *,
    scenario: str,
    information: Sequence[str],
    possible_answers: Sequence[str],
    transcript: Sequence[Mapping[str, Any]] | None = None,
) -> HiddenBenchVotePrompt:
    """A vote prompt; `transcript=None` is the pre-discussion vote.

    The vote's system prompt carries no `%extra%`: the pipeline calls
    `discussion_system_prompt(task, information)` with the default empty
    `extra` for votes, so a strategy ablation conditions the *discussion*
    and leaves the measurement instrument untouched.
    """

    prompt = hidden_bench_vote_prompt(possible_answers).bind(
        scenario=scenario,
        information=tuple(information),
        response_style=RESPONSE_STYLE,
    )
    if transcript:
        prompt = prompt.bind(transcript=tuple(transcript))
    return prompt
