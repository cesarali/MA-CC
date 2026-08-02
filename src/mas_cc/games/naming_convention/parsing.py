"""Versioned, non-inferential parsing for answer-first convention responses."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Sequence
from typing import Any

from .records import ParsedConventionResponse


def _strip_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|python)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _mapping_candidates(content: str) -> tuple[str, ...]:
    stripped = _strip_fence(content)
    candidates = [stripped]
    start, end = stripped.find("{"), stripped.rfind("}")
    if 0 <= start < end and stripped[start : end + 1] != stripped:
        candidates.append(stripped[start : end + 1])
    return tuple(candidates)


def _validated(
    raw: str, mapping: dict[str, Any], actions: Sequence[str], mode: str
) -> ParsedConventionResponse:
    value = mapping.get("value", mapping.get("action"))
    if not isinstance(value, str) or value not in actions:
        raise ValueError("response value is not exactly one configured action")
    reason_value = mapping.get("reason")
    reason = reason_value.strip() if isinstance(reason_value, str) else None
    if not reason:
        raise ValueError("paper-style response requires a non-blank reason")
    return ParsedConventionResponse(raw, value, reason, mode)


def parse_convention_response(
    content: str,
    actions: Sequence[str],
    parser_contract: str = "tolerant_paper_object_v1",
) -> ParsedConventionResponse:
    """Extract only an explicit answer field; never infer from free-form prose."""

    if not isinstance(content, str):
        raise TypeError("provider response must be text")
    legal = tuple(actions)
    if parser_contract == "choice_only_v1":
        value = content.strip()
        if value not in legal:
            raise ValueError("choice-only response must be exactly one configured action")
        return ParsedConventionResponse(content, value, None, parser_contract)
    if parser_contract not in {"strict_json_reason_v1", "tolerant_paper_object_v1"}:
        raise ValueError(f"unknown convention parser contract {parser_contract!r}")

    errors: list[str] = []
    for candidate in _mapping_candidates(content):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
        else:
            if isinstance(value, dict):
                return _validated(content, value, legal, "strict_json_reason_v1")
        if parser_contract == "strict_json_reason_v1":
            continue
        try:
            value = ast.literal_eval(candidate)
        except (SyntaxError, ValueError) as exc:
            errors.append(str(exc))
        else:
            if isinstance(value, dict):
                return _validated(content, value, legal, "python_object_reason_v1")

        # The supplement's displayed object uses a semicolon between fields,
        # which is neither strict JSON nor a Python literal.
        match = re.fullmatch(
            r"\{\s*['\"]value['\"]\s*:\s*['\"](?P<value>[^'\"]+)['\"]\s*;\s*"
            r"['\"]reason['\"]\s*:\s*['\"](?P<reason>.*?)['\"]\s*\}",
            candidate,
            flags=re.DOTALL,
        )
        if match:
            return _validated(content, match.groupdict(), legal, "paper_semicolon_reason_v1")
    raise ValueError("response does not contain a valid answer-first action/reason object")
