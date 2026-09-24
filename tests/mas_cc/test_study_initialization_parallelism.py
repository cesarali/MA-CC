from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from mas_cc.studies import initialization


@dataclass(frozen=True)
class _FakeGame:
    def initialize(self, _config, seed):
        return seed

    def initial_vote_requests(self, state, _config):
        return (state,)


def test_artifact_parallelism_is_bounded_and_materializes_each_entry(
    monkeypatch, tmp_path
):
    config_path = Path(
        "configs/runs/relational_reasoning/blackboard_game/iclr_experiments/"
        "21-09-2026-full-vs-report-v1/controlled.yaml"
    )
    representative = initialization._representative_configs((config_path,))[0]
    entries = tuple(
        initialization.InitializationPlanEntry(
            repetition_index=index,
            episode_seed=100 + index,
            compatibility_key=f"key-{index}",
            artifact_path=str(tmp_path / f"artifact-{index}.json"),
        )
        for index in range(4)
    )
    monkeypatch.setattr(
        initialization, "_representative_configs", lambda _paths: (representative,)
    )
    monkeypatch.setattr(
        initialization, "build_initialization_plan", lambda _paths, _dir: entries
    )
    monkeypatch.setattr(initialization, "create_game", lambda _config: _FakeGame())
    monkeypatch.setattr(
        initialization,
        "artifact_from_actions",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        initialization,
        "write_initialization_artifact",
        lambda path, _artifact: path.write_text("ok", encoding="utf-8"),
    )

    active = 0
    maximum_active = 0

    async def fake_execute(*_args, **_kwargs):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return SimpleNamespace(action="A")

    monkeypatch.setattr(initialization, "_execute_decision", fake_execute)
    result = asyncio.run(
        initialization.materialize_initializations(
            (config_path,),
            tmp_path,
            lambda _config: SimpleNamespace(close=lambda: None),
            artifact_parallelism=2,
        )
    )

    assert result == entries
    assert maximum_active == 2
    assert all(Path(entry.artifact_path).read_text(encoding="utf-8") == "ok" for entry in entries)
