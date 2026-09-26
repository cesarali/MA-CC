"""Phase-1 contracts for forkable blackboard parent checkpoints."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from mas_cc.config import load_run_config
from mas_cc.games import create_game
from mas_cc.games.relational_reasoning.imitation_round_feedback import (
    ContinuationBranch,
    ParentBundleWorker,
    ParentCheckpoint,
    ParentCheckpointStore,
    RelationalGameState,
    run_checkpoint_continuation,
    run_relational_imitation_round_feedback_game,
    standard_continuation_branches,
)
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider
from mas_cc.storage import canonical_hash

CONFIG = (
    "configs/runs/relational_reasoning/misselaneous/"
    "relational_blackboard_no_control_smoke.yaml"
)


def _config(*, rounds: int = 4, rho: float = 0.7):
    config = load_run_config(CONFIG, environment={})
    options = {
        **dict(config.game.options),
        "rounds": rounds,
        "epistemic_persistence": rho,
    }
    return replace(
        config,
        game=replace(config.game, horizon=rounds, options=options),
    )


async def _run(config, **kwargs):
    return await run_relational_imitation_round_feedback_game(
        create_game(config.game),
        config,
        MockLLMProvider(config.llm_provider),
        **kwargs,
    )


def _checkpoint(config, preparation):
    return ParentCheckpoint.create(
        parent_id="parent-0001",
        checkpoint_id="parent-0001-after-r02",
        state=preparation.final_state,
        absolute_round=2,
        preparation_rounds=2,
        initialization_hash=canonical_hash(preparation.initial_state.to_dict()),
        prompt_hash="prompt-v2-test-hash",
        model_hash="deterministic-smoke-test-hash",
        preparation_config_hash=canonical_hash(config.to_dict()),
        parent_seed=config.execution.seed,
        parent_stream_provenance={
            "derivation_version": 1,
            "seed": config.execution.seed,
        },
        runtime_state=preparation.runtime_state,
    )


@pytest.fixture(scope="module")
def prepared():
    config = _config()
    full = asyncio.run(_run(config))
    preparation = asyncio.run(_run(config, continuation_length=2))
    return config, full, preparation, _checkpoint(config, preparation)


def test_state_and_checkpoint_round_trip_preserve_hash(prepared, tmp_path):
    _, _, preparation, checkpoint = prepared
    restored_state = RelationalGameState.from_dict(preparation.final_state.to_dict())
    assert restored_state.to_dict() == preparation.final_state.to_dict()

    store = ParentCheckpointStore(tmp_path / "parents")
    path = store.write(checkpoint)
    restored = store.load(checkpoint.checkpoint_hash)
    assert path.name == f"{checkpoint.checkpoint_hash}.json"
    assert restored.checkpoint_hash == checkpoint.checkpoint_hash
    assert restored.state == checkpoint.state
    assert store.write(checkpoint) == path


def test_checkpoint_rejects_hash_coordinate_and_state_tampering(prepared):
    _, _, _, checkpoint = prepared
    with pytest.raises(ValueError, match="q does not match"):
        replace(checkpoint, q=checkpoint.q + 1, checkpoint_hash="").validate()

    value = checkpoint.to_dict()
    value["checkpoint_hash"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        ParentCheckpoint.from_dict(value)


def test_save_restore_matches_uninterrupted_mock_continuation(prepared):
    config, full, preparation, checkpoint = prepared
    continuation = asyncio.run(
        _run(
            config,
            initial_state=checkpoint.validate(),
            start_round=2,
            continuation_length=2,
            continuation_seed=config.execution.seed,
            restored_runtime_state=checkpoint.runtime_state,
        )
    )

    assert continuation.final_state.to_dict() == full.final_state.to_dict()
    provenance_only = {
        "post_branch_horizon",
        "continuation_seed",
    }
    uninterrupted = [
        {k: v for k, v in row.event.items() if k not in provenance_only}
        for row in full.rounds[2:]
    ]
    restored = [
        {k: v for k, v in row.event.items() if k not in provenance_only}
        for row in continuation.rounds
    ]
    assert restored == uninterrupted
    assert [row.event["absolute_round"] for row in continuation.rounds] == [3, 4]
    assert [row.event["post_branch_horizon"] for row in continuation.rounds] == [1, 2]
    # Equality with the uninterrupted records also proves first-round expiry
    # and persistence were neither omitted nor applied twice.
    assert continuation.rounds[0].event["persistence_deactivated_fact_count"] == full.rounds[2].event[
        "persistence_deactivated_fact_count"
    ]
    assert continuation.rounds[0].event["expired_message_count"] == full.rounds[2].event[
        "expired_message_count"
    ]


def test_nine_branches_share_one_checkpoint_and_distinct_streams(prepared):
    _, _, _, checkpoint = prepared
    branches = standard_continuation_branches()
    assert len(branches) == 9
    assert sum(branch.policy == "none" for branch in branches) == 1
    streams = [branch.derive_stream(checkpoint) for branch in branches]
    assert len({seed for seed, _ in streams}) == 9
    assert {
        stream["identity"]["parent_id"] for _, stream in streams
    } == {checkpoint.parent_id}
    assert checkpoint.h0_observation(continuation_rounds=10)["option_counts"] == dict(
        checkpoint.checkpoint_option_counts
    )


def test_none_branch_is_canonical_and_metadata_uses_semantic_targets(prepared):
    config, _, _, checkpoint = prepared
    state = checkpoint.validate()
    truth = state.correct_answer
    false = next(option for option in state.possible_answers if option != truth)
    none = ContinuationBranch("none", 0)
    truth_branch = ContinuationBranch("always_truth", 3)
    false_branch = ContinuationBranch("sensing_false", 12)
    assert none.posting_budget is None
    assert none.metadata(
        checkpoint,
        continuation_rounds=10,
        truth_target=truth,
        false_target=false,
    )["controller_target_semantic_id"] is None
    assert truth_branch.metadata(
        checkpoint,
        continuation_rounds=10,
        truth_target=truth,
        false_target=false,
    )["controller_target_semantic_id"] == truth
    assert false_branch.metadata(
        checkpoint,
        continuation_rounds=10,
        truth_target=truth,
        false_target=false,
    )["controller_target_semantic_id"] == false
    with pytest.raises(ValueError, match="canonical posting budget"):
        ContinuationBranch("none", 3)

    result = asyncio.run(
        run_checkpoint_continuation(
            game=create_game(config.game),
            config=config,
            provider=MockLLMProvider(config.llm_provider),
            checkpoint=checkpoint,
            branch=none,
            continuation_rounds=1,
            truth_target=truth,
            false_target=false,
        )
    )
    row = result.rounds[0].event
    assert row["checkpoint_hash"] == checkpoint.checkpoint_hash
    assert row["branch_policy"] == "none"
    assert row["U_k"] == 0
    assert row["P_U1_given_Y"] == 0.0


def test_failed_branch_retry_keeps_parent_and_completed_siblings(prepared, tmp_path):
    _, _, _, checkpoint = prepared
    branches = standard_continuation_branches()
    worker = ParentBundleWorker(tmp_path / "bundle")
    calls: list[str] = []
    fail_id = branches[1].branch_id

    async def initially_failing(parent, branch, branch_dir):
        calls.append(branch.branch_id)
        assert parent.checkpoint_hash == checkpoint.checkpoint_hash
        if branch.branch_id == fail_id:
            raise RuntimeError("simulated branch interruption")
        return {"observation_count": 10}

    with pytest.raises(RuntimeError, match="simulated branch interruption"):
        asyncio.run(worker.run(checkpoint, branches, initially_failing))
    first_sibling = branches[0].branch_id
    sibling_seal = tmp_path / "bundle" / "branches" / first_sibling / "branch_seal.json"
    original_sibling = sibling_seal.read_bytes()

    async def succeeding(parent, branch, branch_dir):
        calls.append(branch.branch_id)
        return {"observation_count": 10}

    bundle = asyncio.run(worker.run(checkpoint, branches, succeeding))
    assert bundle["status"] == "complete"
    assert len(bundle["completed_branches"]) == 9
    assert sibling_seal.read_bytes() == original_sibling
    assert calls.count(first_sibling) == 1
    assert json.loads(
        (tmp_path / "bundle" / "parent_bundle_seal.json").read_text()
    )["checkpoint_hash"] == checkpoint.checkpoint_hash


def test_graceful_branch_drain_seals_active_branch_and_resumes(prepared, tmp_path):
    from mas_cc.studies.drain import Drained

    _, _, _, checkpoint = prepared
    branches = standard_continuation_branches()
    worker = ParentBundleWorker(tmp_path / "drained-bundle")

    class Controller:
        requested = False

        def raise_if_safe(self, boundary):
            if self.requested:
                raise Drained(boundary)

        def work_started(self, _kind):
            pass

        def work_finished(self, _kind):
            pass

    controller = Controller()
    calls = []

    async def execute(_parent, branch, _branch_dir):
        calls.append(branch.branch_id)
        controller.requested = True
        return {"observation_count": 1}

    with pytest.raises(Drained, match="branch"):
        asyncio.run(worker.run(checkpoint, branches, execute, drain_controller=controller))
    assert calls == [branches[0].branch_id]
    first_seal = (tmp_path / "drained-bundle" / "branches" /
                  branches[0].branch_id / "branch_seal.json")
    original = first_seal.read_bytes()
    assert not (tmp_path / "drained-bundle" / "parent_bundle_seal.json").exists()
    controller.requested = False

    async def resume(_parent, branch, _branch_dir):
        calls.append(branch.branch_id)
        return {"observation_count": 1}

    bundle = asyncio.run(worker.run(checkpoint, branches, resume, drain_controller=controller))
    assert bundle["status"] == "complete"
    assert first_seal.read_bytes() == original
    assert calls.count(branches[0].branch_id) == 1


def test_downscaled_mock_parent_seals_all_nine_real_continuations(prepared, tmp_path):
    config, _, _, checkpoint = prepared
    state = checkpoint.validate()
    truth = state.correct_answer
    false = next(option for option in state.possible_answers if option != truth)
    worker = ParentBundleWorker(tmp_path / "real-bundle")

    async def execute(parent, branch, branch_dir):
        result = await run_checkpoint_continuation(
            game=create_game(config.game),
            config=config,
            provider=MockLLMProvider(config.llm_provider),
            checkpoint=parent,
            branch=branch,
            continuation_rounds=1,
            truth_target=truth,
            false_target=false,
        )
        row = result.rounds[0].event
        assert row["checkpoint_hash"] == checkpoint.checkpoint_hash
        return {
            "observation_count": len(result.rounds),
            "final_state_hash": canonical_hash(result.final_state.to_dict()),
        }

    bundle = asyncio.run(worker.run(checkpoint, standard_continuation_branches(), execute))
    assert bundle["status"] == "complete"
    assert len(bundle["branch_hashes"]) == 9
