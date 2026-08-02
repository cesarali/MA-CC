"""Dependency-free static request token estimation."""

from __future__ import annotations

import re

from mas_cc.llm_providers import CompletionRequest


TOKENIZER_NAME = "mas_cc_regex_v1_estimate"


def estimate_input_tokens(request: CompletionRequest) -> int:
    content = sum(
        len(re.findall(r"\w+|[^\w\s]", message.content, flags=re.UNICODE))
        for message in request.messages
    )
    return content + 4 * len(request.messages)
