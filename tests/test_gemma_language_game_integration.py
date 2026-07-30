import asyncio
import json

from naming_game.gemma_local_client import GemmaLocalAsyncLLMClient, GenerationResult, ScoringResult
from naming_game.models import RunSpec, UpdateMode
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
