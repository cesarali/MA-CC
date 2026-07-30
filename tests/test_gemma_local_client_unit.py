import asyncio
import sys
import threading
import time

import pytest

from naming_game.gemma_local_client import GemmaLocalAsyncLLMClient, GenerationResult, ScoringResult


class Runtime:
    diagnostics = {"fake": True}
    active = maximum = 0
    lock = threading.Lock()
    def generate(self, messages, **kwargs):
        with self.lock:
            self.active += 1; self.maximum = max(self.maximum, self.active)
        time.sleep(.01)
        with self.lock: self.active -= 1
        return GenerationResult("new only", 3, 2)
    def score(self, messages, choices): return ScoringResult(3, tuple((1,) for _ in choices), tuple(range(len(choices))))


def test_lazy_once_serialized_usage_stats_and_close():
    async def check():
        count = 0; runtime = Runtime()
        def factory():
            nonlocal count; count += 1; return runtime
        client = GemmaLocalAsyncLLMClient(runtime_factory=factory)
        assert "transformers" not in sys.modules
        results = await asyncio.gather(*(client.complete([{"role":"user", "content":"x"}], temperature=0, max_tokens=2) for _ in range(4)))
        assert count == 1 and runtime.maximum == 1
        assert results[0].content == "new only" and results[0].usage.total_tokens == 5
        assert client.stats["successful_calls"] == 4
        client.close(); client.close()
        with pytest.raises(RuntimeError):
            await client.complete([{"role":"user", "content":"x"}], temperature=0, max_tokens=1)
        assert client.stats["failed_calls"] == 1
    asyncio.run(check())


@pytest.mark.parametrize("messages", [[], [{"role":"user"}], [{"role":" ", "content":"x"}], [{"role":"user", "content":""}]])
def test_exact_messages(messages):
    with pytest.raises(ValueError):
        asyncio.run(GemmaLocalAsyncLLMClient(runtime_factory=Runtime).complete(messages, temperature=0, max_tokens=1))
