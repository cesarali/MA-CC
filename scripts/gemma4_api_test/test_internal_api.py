#!/usr/bin/env python3
"""Opt-in live public-API diagnostic. This file is not a pytest test."""
import asyncio
import math

from naming_game import GemmaLocalAsyncLLMClient

PROMPT = """Question: Which number is larger, 7 or 3?

A. 7
B. 3
C. They are equal

Return only A, B, or C."""


async def main():
    client = GemmaLocalAsyncLLMClient()
    assert client.model == "google/gemma-4-12B-it" and client.provider_name == "gemma_local"
    messages = [{"role": "user", "content": PROMPT}]
    response = await client.complete_decision(messages, choices=["A", "B", "C"],
        output_format="choice_only", choice_temperature=1.0, selection_policy="argmax")
    assert [score.choice for score in response.scores] == ["A", "B", "C"]
    assert all(score.token_ids and math.isfinite(score.log_likelihood) and math.isfinite(score.probability) for score in response.scores)
    assert abs(sum(score.probability for score in response.scores) - 1) < 1e-5
    assert response.selected_choice == "A"
    assert response.content == "A" and response.reason is None
    assert client.stats["attempts"] == 1 and client.stats["successes"] == 1
    explained = await client.complete_decision(messages, choices=["A", "B", "C"],
        output_format="choice_reason", choice_temperature=1.0, selection_policy="argmax",
        generation_temperature=0, max_reason_tokens=32, seed=7)
    assert explained.content.startswith(explained.selected_choice + "\nReason: ")
    assert explained.reason and explained.reason_valid
    assert abs(sum(score.probability for score in explained.scores) - 1) < 1e-5
    assert client.stats["attempts"] == 2 and client.stats["successes"] == 2
    runtime_id = id(client._runtime)
    ordinary = await client.complete(messages, temperature=0, max_tokens=8)
    assert ordinary.content.strip() and id(client._runtime) == runtime_id
    assert client.diagnostics and "cuda_peak_allocated_bytes" in client.diagnostics
    print(client.diagnostics)
    client.close()


if __name__ == "__main__":
    asyncio.run(main())
