"""The synthetic agent, standing in for an LLM at the provider boundary.

It is a normal `LLMProvider`: the decision loop, the budget guard, the
recorder and the audit trail cannot tell it apart from a model adapter, which
is what lets a synthetic run rehearse the real pipeline rather than a
simplified copy of it.

What it does *not* do is take a shortcut. It receives only a
`CompletionRequest` - the same compiled messages a real provider would be
sent - and recovers its decision input by reading the marked payload line out
of them. So a prompt that fails to carry this round's observation, or carries
last round's, produces a wrong action here and a mutual information that
misses its closed-form value. A provider handed the observation through a side
channel would have quietly passed that same bug.

There is no randomness in this file. Every coin was drawn up front into the
episode's tape (`noise.py`) and reaches the agent through the prompt, so the
same seed replays exactly and speed mode can reproduce it without imitating a
provider.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from typing import Any, Callable, Mapping

from mas_cc.llm_runtime.config import LLMProviderConfig
from mas_cc.llm_runtime.providers.capabilities import ProviderCapabilities
from mas_cc.llm_runtime.providers.requests import CompletionRequest
from mas_cc.llm_runtime.providers.responses import CompletionResponse, ProviderUsage

from .prompts import OBSERVATION_MARKER

_PAYLOAD = re.compile(re.escape(OBSERVATION_MARKER) + r"\s*(\{.*?\})\s*$", re.MULTILINE)

DecodingPolicy = Callable[[Mapping[str, Any]], str]


class SyntheticPromptError(ValueError):
    """The compiled prompt did not carry a decodable observation.

    Raised rather than guessed at: a missing or malformed payload means the
    prompt-construction path is broken, and that is exactly the failure these
    games exist to surface.
    """


def bernoulli_xor_v1(observation: Mapping[str, Any]) -> str:
    """Report the signal, flipped iff this agent's private coin came up."""

    actions = tuple(str(item) for item in observation["actions"])
    signal = str(observation["signal"])
    if signal not in actions:
        raise SyntheticPromptError(f"signal {signal!r} is not one of the declared actions")
    if len(actions) != 2:
        raise SyntheticPromptError("the XOR policy is defined for a binary alphabet")
    if not observation["flip"]:
        return signal
    return next(action for action in actions if action != signal)


POLICIES: dict[str, DecodingPolicy] = {"bernoulli_xor_v1": bernoulli_xor_v1}
"""Decoding rules by name. Each synthetic game names the one its prompt asks for.

Dispatching on a name carried in the payload - rather than on the provider's
own configuration - keeps the agent honest: it applies the rule the prompt
actually stated, so a game that renders the wrong rule gets the wrong answer.
"""


def read_observation(request: CompletionRequest) -> Mapping[str, Any]:
    """Recover the observation payload from the compiled messages."""

    for message in request.messages:
        match = _PAYLOAD.search(message.content)
        if match is None:
            continue
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise SyntheticPromptError(f"observation payload is not valid JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise SyntheticPromptError("observation payload must be a JSON object")
        return payload
    raise SyntheticPromptError(
        f"no {OBSERVATION_MARKER} line in the compiled prompt; the game did not "
        "render this round's observation into the messages the provider was sent"
    )


def decide(request: CompletionRequest) -> str:
    """The whole agent: read the prompt, apply the named rule, answer."""

    observation = read_observation(request)
    policy_name = str(observation.get("policy", ""))
    try:
        policy = POLICIES[policy_name]
    except KeyError as exc:
        raise SyntheticPromptError(
            f"unknown decoding policy {policy_name!r}; known: {', '.join(sorted(POLICIES))}"
        ) from exc
    return policy(observation)


class SyntheticAgentProvider:
    """A lookup table behind the `LLMProvider` interface."""

    name = "synthetic_agent"
    capabilities = ProviderCapabilities(
        supports_seed=True,
        reports_usage=True,
        max_request_concurrency=None,
    )

    def __init__(self, config: LLMProviderConfig) -> None:
        self.model = config.model
        self._closed = False

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        if self._closed:
            raise RuntimeError("synthetic agent provider is closed")
        started = time.perf_counter()
        await asyncio.sleep(0)
        content = decide(request)
        input_tokens = sum(_count_tokens(message.content) for message in request.messages)
        output_tokens = _count_tokens(content)
        digest = hashlib.sha256(
            repr((request.wire_messages(), self.model, content)).encode("utf-8")
        ).hexdigest()[:16]
        latency = time.perf_counter() - started
        raw = {
            "id": f"synthetic-{digest}",
            "model": self.model,
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
        return CompletionResponse(
            content=content,
            provider=self.name,
            model=self.model,
            usage=ProviderUsage(input_tokens, output_tokens, input_tokens + output_tokens),
            finish_reason="stop",
            request_id=raw["id"],
            latency_seconds=latency,
            inference_seconds=latency,
            status_code=200,
            raw_response=raw,
        )

    def close(self) -> None:
        self._closed = True


def _count_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
