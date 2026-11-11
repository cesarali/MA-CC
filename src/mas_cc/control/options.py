"""Report control options that the selected mechanism never reads.

A control mechanism parses ``control.options`` with ``options.get(...)``, so a key
it does not know used to be dropped with no error. A misspelled key, or a key
that only exists on another branch (``controller_budget_scope`` on a checkout
without it), then ran the study on the default instead, and nothing in the run
record said so. ``ReadTrackingOptions`` records every key a parser looks up and
turns each key that was never looked up into a validation issue, so the mistake
fails at control creation, before any provider call.

A grid that shares one options block across mechanisms (``beta`` for
``soft_target`` cells, ignored by ``threshold_target`` cells) declares the keys a
mechanism may skip under ``control.options.ignored_options``.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from mas_cc.llm_runtime.validation import ValidationIssue

IGNORED_OPTIONS_KEY = "ignored_options"


class ReadTrackingOptions(Mapping[str, Any]):
    """Read-only view of control options that records which keys were looked up."""

    __slots__ = ("_options", "_read")

    def __init__(self, options: Mapping[str, Any]) -> None:
        self._options = options
        self._read: set[str] = set()

    def __getitem__(self, key: str) -> Any:
        self._read.add(key)
        return self._options[key]

    def get(self, key: str, default: Any = None) -> Any:
        self._read.add(key)
        return self._options.get(key, default)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            self._read.add(key)
        return key in self._options

    def __iter__(self) -> Iterator[str]:
        # A caller that iterates sees every key, so none of them is unread.
        self._read.update(self._options)
        return iter(self._options)

    def __len__(self) -> int:
        return len(self._options)

    def unread_issues(self) -> list[ValidationIssue]:
        """One issue per supplied key that no parser looked up and none declared ignored."""

        ignored = self._options.get(IGNORED_OPTIONS_KEY, ())
        if (
            isinstance(ignored, (str, bytes))
            or not isinstance(ignored, Sequence)
            or not all(isinstance(item, str) for item in ignored)
        ):
            return [
                ValidationIssue(
                    f"control.options.{IGNORED_OPTIONS_KEY}", "must be a list of option names"
                )
            ]
        known = sorted(self._read)
        issues: list[ValidationIssue] = []
        for key in sorted(set(self._options) - self._read - set(ignored) - {IGNORED_OPTIONS_KEY}):
            close = difflib.get_close_matches(key, known, n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            issues.append(
                ValidationIssue(
                    f"control.options.{key}",
                    "is not read by this control mechanism, so the run would silently use "
                    f"the default instead{hint}. If another mechanism in the same grid needs "
                    f"it, list it under control.options.{IGNORED_OPTIONS_KEY}.",
                )
            )
        return issues
