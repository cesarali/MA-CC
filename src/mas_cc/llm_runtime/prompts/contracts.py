"""Provider-independent output contracts and local response validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import re
from typing import Any

from ..messages import MessageRole
from ..validation import ValidationIssue, ValidationResult
from ._values import freeze, thaw


@dataclass(frozen=True, slots=True)
class ResponseContract:
    """The explicit output shape attached to a compiled prompt."""

    type: str
    allowed_values: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.type, str) or not self.type.strip():
            raise ValueError("ResponseContract.type must be a non-empty string")
        if isinstance(self.allowed_values, (str, bytes)) or not isinstance(
            self.allowed_values, Sequence
        ):
            raise TypeError("ResponseContract.allowed_values must be a sequence of strings")
        values = tuple(self.allowed_values)
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("ResponseContract.allowed_values must contain non-empty strings")
        if len(set(values)) != len(values):
            raise ValueError("ResponseContract.allowed_values must be unique")
        if self.type in {"choice_only", "paper_choice_reason", "json_vote"} and not values:
            raise ValueError(
                f"ResponseContract.allowed_values is required for {self.type}"
            )
        if not isinstance(self.options, Mapping):
            raise TypeError("ResponseContract.options must be a mapping")
        object.__setattr__(self, "allowed_values", values)
        object.__setattr__(self, "options", freeze(self.options))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ResponseContract":
        if not isinstance(values, Mapping):
            raise TypeError("response contract must be a mapping")
        unknown = set(values) - {"type", "allowed_values", "options"}
        if unknown:
            raise ValueError(f"ResponseContract.{sorted(unknown)[0]} is an unknown field")
        return cls(
            type=values.get("type", ""),
            allowed_values=tuple(values.get("allowed_values", ())),
            options=values.get("options", {}),
        )

    def instruction(self) -> str:
        if self.type == "choice_only":
            choices = ", ".join(self.allowed_values)
            return (
                f"Return exactly one of these values: {choices}. "
                "Return only the value, with no explanation or punctuation."
            )
        if self.type == "paper_choice_reason":
            choices = " or ".join(self.allowed_values)
            return (
                "Write your answer using the following answer-first format: "
                "{'value': <"
                f"{choices}"
                ">; 'reason': <YOUR REASON>}."
            )
        if self.type == "json_vote":
            choices = json.dumps(list(self.allowed_values), ensure_ascii=False)
            return (
                "Please decide and provide your rationale in the following JSON format:\n"
                "{\n"
                f'  "vote": <A string, {choices}>,\n'
                '  "rationale": <A string, representing your rationale>\n'
                "}"
            )
        return f"Return a response satisfying the {self.type!r} contract."

    def instruction_messages(self) -> tuple[tuple[MessageRole, str], ...]:
        """Return local prompt instructions owned by this response contract.

        Concrete game contracts may override this to preserve a paper-specific
        split between system formatting constraints and the user decision.
        """

        role_value = self.options.get("instruction_role", MessageRole.USER.value)
        role = MessageRole(str(role_value))
        decision = self.options.get("decision_instruction")
        instruction = self.instruction()
        if decision is not None:
            if not isinstance(decision, str) or not decision.strip():
                raise ValueError(
                    "ResponseContract.options.decision_instruction must be a non-empty string"
                )
            instruction = f"{decision.strip()}\n\n{instruction}"
        return ((role, instruction),)

    def validate(self, response: str) -> ValidationResult:
        if not isinstance(response, str):
            return ValidationResult.failure(
                ValidationIssue("response", "must be a string", response)
            )
        if self.type == "choice_only" and response.strip() not in self.allowed_values:
            return ValidationResult.failure(
                ValidationIssue(
                    "response",
                    f"must be exactly one of: {', '.join(self.allowed_values)}",
                    response,
                )
            )
        if self.type == "paper_choice_reason":
            value_match = re.search(
                r"[\"']value[\"']\s*:\s*[\"']([^\"']+)[\"']",
                response,
            )
            reason_match = re.search(r"[\"']reason[\"']\s*:", response)
            if value_match is None:
                return ValidationResult.failure(
                    ValidationIssue(
                        "response.value",
                        "must contain an answer-first quoted value field",
                        response,
                    )
                )
            if value_match.group(1) not in self.allowed_values:
                return ValidationResult.failure(
                    ValidationIssue(
                        "response.value",
                        f"must be exactly one of: {', '.join(self.allowed_values)}",
                        value_match.group(1),
                    )
                )
            if reason_match is None:
                return ValidationResult.failure(
                    ValidationIssue("response.reason", "must contain a reason field")
                )
        if self.type == "json_vote":
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                return ValidationResult.failure(
                    ValidationIssue("response", "must be a valid JSON object", response)
                )
            if not isinstance(parsed, Mapping):
                return ValidationResult.failure(
                    ValidationIssue("response", "must be a JSON object", response)
                )
            if parsed.get("vote") not in self.allowed_values:
                return ValidationResult.failure(
                    ValidationIssue(
                        "response.vote",
                        f"must be exactly one of: {', '.join(self.allowed_values)}",
                        parsed.get("vote"),
                    )
                )
            if not isinstance(parsed.get("rationale"), str):
                return ValidationResult.failure(
                    ValidationIssue("response.rationale", "must be a string")
                )
        return ValidationResult.success()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type}
        if self.allowed_values:
            result["allowed_values"] = list(self.allowed_values)
        if self.options:
            result["options"] = thaw(self.options)
        result["instruction_messages"] = [
            {"role": role.value, "content": content}
            for role, content in self.instruction_messages()
        ]
        return result
