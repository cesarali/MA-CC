"""TypeSafe System One as a ballot provider: a calibrated typed Choice, no text.

Experimental agent arm. The MA-CC ballot asks an agent to pick one presented
option letter and to justify it; System One answers the *choice* with a
calibrated probability vector in one round trip (~0.6 s on the estate
gateway) and generates no text. This adapter therefore:

* sends the compiled prompt (every message) as the ``state`` and asks one
  Choice over the presented letters, parsed from the ballot instruction the
  prompt itself carries (``"vote": "<A | B | C>"``);
* returns a ``CompletionResponse`` whose content is the exact JSON the
  ballot contract accepts: the chosen letter, a machine-written
  ``private_reason`` that states the probability, and a ``NONE`` public
  message (System One cannot write board text);
* records the whole probability vector and confidence in ``raw_response`` so
  the calibration engine can later read the population's vote distribution
  directly instead of estimating it.

It is a *different agent*, not a drop-in: an arm that uses it must be compared
on the same paired seeds as a gpt-oss arm and described as such. Nothing in the
repository references this provider type; selecting it is a config decision
(``llm_provider.type: typesafe``, see ``configs/components/cygnus/gateway_typesafe_jev.yaml``).

Credentials and endpoint come from the same environment as the instruments
(``LLM_KEY`` / ``MA_CC_SYSTEMONE_KEY_FILE``, ``LLM_BASE`` / ``MA_CC_SYSTEMONE_URL``).
``options.dry_run: true`` answers without any network call (uniform pick, for
plans and tests).
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Sequence
from typing import Any

from mas_cc.analysis import systemone
from mas_cc.llm_runtime.config import LLMProviderConfig

from ..capabilities import ProviderCapabilities
from ..errors import ProviderError
from ..requests import CompletionRequest
from ..responses import CompletionResponse, ProviderUsage

_VOTE_OPTIONS = re.compile(r'"vote":\s*"<([^>]+)>"')


def presented_letters(request: CompletionRequest, fallback: Sequence[str] = ()) -> tuple[str, ...]:
    """The option letters the ballot instruction offers (``"vote": "<A | B | C>"``)."""
    for message in reversed(request.messages):
        match = _VOTE_OPTIONS.search(message.content)
        if match:
            letters = tuple(part.strip() for part in match.group(1).split("|") if part.strip())
            if letters:
                return letters
    if fallback:
        return tuple(str(item) for item in fallback)
    raise ProviderError("typesafe ballot provider found no option letters in the prompt", provider="typesafe",
                        code="no_options")


class TypeSafeChoiceProvider:
    name = "typesafe"
    capabilities = ProviderCapabilities(
        supports_seed=False,
        reports_usage=True,
        supports_system_messages=True,
        supports_parallel_requests=True,
        max_request_concurrency=None,
    )

    def __init__(self, config: LLMProviderConfig, *, client: systemone.SystemOneClient | None = None) -> None:
        self.model = config.model or systemone.DEFAULT_MODEL
        options = dict(config.options)
        self._dry_run = bool(options.get("dry_run", False))
        self._fallback_letters = tuple(str(x) for x in (options.get("allowed_values") or ()))
        self._question = str(options.get(
            "instructions",
            "You are the agent addressed by `conversation` (the system and user messages it was given, in order). "
            "Which presented option letter should this agent vote for, given everything in `conversation`?",
        ))
        self._client = client
        if client is None and not self._dry_run:
            self._client = systemone.SystemOneClient(
                model=self.model,
                cache_dir=options.get("cache_dir"),
                budget=systemone.Budget(max_usd=options.get("max_usd"), max_input_tokens=options.get("max_input_tokens"))
                if options.get("max_usd") is not None or options.get("max_input_tokens") is not None else None,
                timeout=float(config.timeout_seconds),
            )
        self._closed = False

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _state(request: CompletionRequest) -> dict[str, Any]:
        return {"conversation": [{"role": m.role.value, "content": m.content} for m in request.messages]}

    def _questions(self, letters: Sequence[str]) -> dict[str, dict[str, Any]]:
        return {"vote": systemone.choice(self._question, {letter: f"vote for option {letter}" for letter in letters})}

    @staticmethod
    def _ballot(letter: str, probabilities: dict[str, float], confidence: float | None, source: str) -> str:
        p = probabilities.get(letter)
        reason = (f"Typed choice by {source}: P({letter})={p:.3f}" if p is not None else f"Typed choice by {source}")
        if confidence is not None:
            reason += f", confidence {confidence:.3f}"
        reason += ". No free-text reasoning is produced by this provider."
        return json.dumps({
            "vote": letter,
            "private_reason": reason,
            "public_message": {"type": "NONE", "text": None, "shared_fact_id": None, "reply_to": None},
        })

    # -- protocol ----------------------------------------------------------
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        if self._closed:
            raise ProviderError("typesafe provider is closed", provider=self.name, code="closed")
        letters = presented_letters(request, self._fallback_letters)
        started = time.perf_counter()
        if self._dry_run:
            letter = letters[0]
            probabilities = {item: 1.0 / len(letters) for item in letters}
            content = self._ballot(letter, probabilities, None, "dry-run")
            return CompletionResponse(content=content, provider=self.name, model=self.model,
                                      usage=ProviderUsage(0, 0, 0), finish_reason="stop", request_id="dry-run",
                                      latency_seconds=time.perf_counter() - started, status_code=200,
                                      raw_response={"dry_run": True, "probabilities": probabilities, "letters": list(letters)})
        assert self._client is not None
        state, questions = self._state(request), self._questions(letters)
        try:
            record = await asyncio.to_thread(self._client.ask, state, questions)
        except systemone.BudgetExceeded as exc:
            raise ProviderError(str(exc), provider=self.name, code="budget") from exc
        except systemone.SystemOneError as exc:
            raise ProviderError(str(exc), provider=self.name, code="transport") from exc
        answer = record["answers"]["vote"]
        letter = str(answer.get("choice"))
        if letter not in letters:
            raise ProviderError(f"typesafe returned a letter outside the ballot: {letter!r}", provider=self.name,
                                code="bad_choice")
        probabilities = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
        confidence = answer.get("confidence")
        usage = record.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        content = self._ballot(letter, probabilities, None if confidence is None else float(confidence), record.get("model", self.model))
        return CompletionResponse(
            content=content, provider=self.name, model=str(record.get("model", self.model)),
            usage=ProviderUsage(input_tokens, output_tokens, input_tokens + output_tokens), finish_reason="stop",
            request_id=record.get("request_id"), latency_seconds=time.perf_counter() - started,
            inference_seconds=float(record.get("seconds") or 0.0), status_code=200,
            raw_response={"typesafe": {"probabilities": probabilities, "confidence": confidence, "letters": list(letters),
                                       "cached": record.get("cached"), "request_id": record.get("request_id")}},
        )

    def close(self) -> None:
        self._closed = True


__all__ = ["TypeSafeChoiceProvider", "presented_letters"]
