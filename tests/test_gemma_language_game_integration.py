import asyncio
import json

from naming_game.gemma_local_client import GemmaLocalAsyncLLMClient, GenerationResult, ScoringResult
from naming_game.interaction import execute_pair_interaction
from naming_game.local_model_types import ChoiceScore, ConstrainedLLMResponse
from naming_game.models import AgentSnapshot, LLMResponse, RunSpec, TokenUsage, UpdateMode
from naming_game.runner import run_single


class GameRuntime:
    diagnostics = {}
    def generate(self, messages, **kwargs):
        # Both parsers accept these labels; prompts distinguish listener requests.
        text = messages[-1]["content"].lower()
        return GenerationResult("YES" if "listener" in text or "already" in text else "A", 2, 1)
    def score(self, messages, choices): return ScoringResult(2, tuple((i,) for i, _ in enumerate(choices)), tuple(-i for i, _ in enumerate(choices)))


def test_fake_local_game_records_provider(tmp_path):
    async def check():
        client = GemmaLocalAsyncLLMClient(runtime_factory=GameRuntime)
        spec = RunSpec("google/gemma-4-12B-it", 2, 0, UpdateMode.SEQUENTIAL, 2, 2, None, 7, 1)
        run = await run_single(spec=spec, client=client, output_dir=tmp_path)
        assert run.summary.api_backend == "gemma_local"
        assert client.stats["successful_calls"] == 4
        config = json.loads(run.output_files["config"].read_text())
        assert config["api_backend"] == "gemma_local"
    asyncio.run(check())


class CapableNonGemmaClient:
    model = "fake/capable"
    provider_name = "future-provider"
    concurrency = 2

    def __init__(self):
        self.temperatures = []
        self.choices = []
        self.prompts = []

    async def complete_constrained(self, messages, *, choices, temperature, seed=None):
        self.temperatures.append(temperature)
        self.choices.append(tuple(choices))
        self.prompts.append(messages)
        scores = tuple(
            ChoiceScore(choice, (index + 1,), -float(index), 1.0 if index == 0 else 0.0)
            for index, choice in enumerate(choices)
        )
        return ConstrainedLLMResponse(
            choices[0], scores, self.model, 0.0, TokenUsage(2, 1, 3), temperature
        )

    async def complete(self, messages, **kwargs):
        return LLMResponse('{"already_known":false}', self.model, 0.0)

    def close(self):
        pass


def test_basic_game_uses_capability_not_provider_name_and_preserves_positive_temperature():
    async def check():
        client = CapableNonGemmaClient()
        result = await execute_pair_interaction(
            client=client,
            speaker=AgentSnapshot(1, frozenset({"A", "B"})),
            listener=AgentSnapshot(2, frozenset({"B"})),
            interaction_index=1,
            round_index=None,
            pair_index=None,
            interaction_kind="basic",
            choice_seed=3,
            temperature=0.5,
            max_tokens_speaker=3,
            max_tokens_listener=3,
        )
        assert client.temperatures == [0.5]
        assert client.choices == [("A", "B")]
        assert "bare name" in client.prompts[0][0]["content"]
        assert "JSON" in client.prompts[0][0]["content"]
        assert result.selected_name == "A"
        assert result.decision_method == "constrained_sequence"

    asyncio.run(check())


def test_basic_game_legacy_zero_uses_explicit_positive_choice_temperature():
    async def check():
        client = CapableNonGemmaClient()
        await execute_pair_interaction(
            client=client,
            speaker=AgentSnapshot(1, frozenset({"A"})),
            listener=AgentSnapshot(2, frozenset({"A"})),
            interaction_index=1,
            round_index=None,
            pair_index=None,
            interaction_kind="basic",
            choice_seed=3,
            temperature=0,
            max_tokens_speaker=3,
            max_tokens_listener=3,
        )
        assert client.temperatures == [1.0]

    asyncio.run(check())
