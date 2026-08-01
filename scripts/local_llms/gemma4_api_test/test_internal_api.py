#!/usr/bin/env python3
"""Opt-in live public-API diagnostic. This file is not a pytest test."""

import asyncio
import json
import math

from naming_game import GemmaLocalAsyncLLMClient
from naming_game.gemma_local_client import _selected_index

PROMPT = """Question: Which number is larger, 7 or 3?

A. 7
B. 3
C. They are equal

Follow the requested response contract exactly."""


def response_report(response):
    return {
        "output_format": response.output_format,
        "semantic_choices": [score.choice for score in response.scores],
        "token_ids": [list(score.token_ids) for score in response.scores],
        "log_likelihoods": [score.log_likelihood for score in response.scores],
        "probabilities": [score.probability for score in response.scores],
        "selected_choice": response.selected_choice,
        "selected_probability": next(
            score.probability
            for score in response.scores
            if score.choice == response.selected_choice
        ),
        "reason": response.reason,
        "reason_valid": response.reason_valid,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


async def main():
    client = GemmaLocalAsyncLLMClient()
    assert client.model == "google/gemma-4-12B-it"
    assert client.provider_name == "gemma_local"
    messages = [{"role": "user", "content": PROMPT}]

    response = await client.complete_decision(
        messages,
        choices=["A", "B", "C"],
        output_format="choice_only",
        choice_temperature=1.0,
        selection_policy="argmax",
    )
    assert [score.choice for score in response.scores] == ["A", "B", "C"]
    assert all(
        score.token_ids
        and math.isfinite(score.log_likelihood)
        and math.isfinite(score.probability)
        for score in response.scores
    )
    assert abs(sum(score.probability for score in response.scores) - 1.0) < 1e-5
    assert response.content == response.selected_choice
    assert response.reason is None and response.reason_valid is None
    assert client.stats["actual_calls"] == 1
    assert client.stats["successful_calls"] == 1
    assert client.stats["failed_calls"] == 0
    assert client.diagnostics["reason_generation_calls"] == 0
    runtime_id = id(client._runtime)

    explained = await client.complete_decision(
        messages,
        choices=["A", "B", "C"],
        output_format="choice_reason",
        choice_temperature=1.0,
        selection_policy="argmax",
        generation_temperature=0,
        max_reason_tokens=32,
        seed=7,
    )
    assert explained.content.startswith(explained.selected_choice + "\nReason: ")
    assert explained.reason and explained.reason_valid
    assert abs(sum(score.probability for score in explained.scores) - 1.0) < 1e-5
    assert client.stats["actual_calls"] == 2
    assert client.stats["successful_calls"] == 2
    assert client.stats["failed_calls"] == 0
    assert client.diagnostics["reason_generation_calls"] == 1
    assert id(client._runtime) == runtime_id

    multi = await client.complete_decision(
        messages,
        choices=["A", "MULTI_TOKEN_DIAGNOSTIC_CHOICE"],
        output_format="choice_only",
        choice_temperature=1.0,
        selection_policy="argmax",
    )
    assert any(len(score.token_ids) > 1 for score in multi.scores)
    assert client.diagnostics["sequence_score_calls"] >= 1

    # Selection mechanics are checkpoint-independent.  The live score calls
    # above establish probabilities; these checks establish order/tie and seed
    # behavior without requiring a real checkpoint to produce an exact tie.
    assert _selected_index((0.5, 0.5), "argmax", None) == 0
    assert _selected_index((0.7, 0.3), "sample", 1) == _selected_index(
        (0.7, 0.3), "sample", 1
    )
    assert _selected_index((0.7, 0.3), "sample", 1) != _selected_index(
        (0.7, 0.3), "sample", 2
    )

    ordinary = await client.complete(messages, temperature=0, max_tokens=8)
    assert ordinary.content.strip() and id(client._runtime) == runtime_id
    diagnostics = client.diagnostics
    assert diagnostics and "cuda_peak_allocated_bytes" in diagnostics
    print(
        json.dumps(
            {
                "choice_only": response_report(response),
                "choice_reason": response_report(explained),
                "multi_token": response_report(multi),
                "statistics": client.stats,
                "runtime_diagnostics": diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    client.close()
    client.close()
    try:
        await client.complete(messages, temperature=0, max_tokens=1)
    except RuntimeError as exc:
        assert str(exc) == "client is closed."
    else:
        raise AssertionError("closed client unexpectedly accepted a request")


if __name__ == "__main__":
    asyncio.run(main())
