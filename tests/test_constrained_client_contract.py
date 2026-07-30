import asyncio
import math

import pytest

from naming_game.gemma_local_client import GemmaLocalAsyncLLMClient, GenerationResult, ScoringResult
from naming_game.local_model_types import ConstrainedLLMClient


class FakeRuntime:
    diagnostics = {"kind": "fake"}
    def generate(self, messages, **kwargs): return GenerationResult("answer", 4, 2)
    def score(self, messages, choices):
        return ScoringResult(4, tuple((i + 1,) * (i + 1) for i in range(len(choices))), tuple(-1.0 for _ in choices))


def test_structural_contract_order_tie_multitoken_and_usage():
    async def check():
        client = GemmaLocalAsyncLLMClient(runtime_factory=FakeRuntime)
        assert isinstance(client, ConstrainedLLMClient)
        response = await client.complete_constrained([{"role": "user", "content": "choose"}], choices=["B", "AA"])
        assert [score.choice for score in response.scores] == ["B", "AA"]
        assert response.selected_choice == "B"
        assert response.scores[1].token_ids == (2, 2)
        assert math.isclose(sum(score.probability for score in response.scores), 1)
        assert response.usage.prompt_tokens == 8
        assert response.usage.completion_tokens == 3
    asyncio.run(check())


@pytest.mark.parametrize("choices,temp", [([], 1), ([""], 1), (["A", "A"], 1), (["A"], 0), (["A"], float("nan"))])
def test_invalid_choices(choices, temp):
    with pytest.raises(ValueError):
        asyncio.run(GemmaLocalAsyncLLMClient(runtime_factory=FakeRuntime).complete_constrained(
            [{"role": "user", "content": "x"}], choices=choices, temperature=temp))
