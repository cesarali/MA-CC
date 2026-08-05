"""Small tokenizer protocol and deterministic dependency-free estimator."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class TokenCounter(Protocol):
    name: str

    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class RegexTokenCounter:
    name: str = "mas_cc_regex_v1_estimate"

    def count_tokens(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("token counter input must be a string")
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
