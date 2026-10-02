from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from slime.rollout import sglang_rollout
from slime.utils.types import Sample
from slime_plugins.agent_tasks.alfworld import envs as alfworld_envs
from slime_plugins.agent_tasks.alfworld.frozen.capabilities import capabilities_for_args
from slime_plugins.agent_tasks.alfworld.frozen import corpus as frozen_corpus
from slime_plugins.agent_tasks.alfworld.frozen import data_source as frozen_data_source
from slime_plugins.agent_tasks.alfworld.frozen.corpus import _manifest_from_args
from slime_plugins.agent_tasks.alfworld.frozen.generate import generate
from slime_plugins.agent_tasks.alfworld.generate import _collection_turn_sampling_seed
from types import SimpleNamespace


NUM_GPUS = 0








@pytest.mark.unit
def test_collection_turn_seed_is_bound_to_cycle_stream_and_turn():
    args = SimpleNamespace(rollout_seed=43)
    baseline = _collection_turn_sampling_seed(args, stream_index=7, turn_idx=2)
    assert baseline == _collection_turn_sampling_seed(args, stream_index=7, turn_idx=2)
    assert baseline != _collection_turn_sampling_seed(args, stream_index=8, turn_idx=2)
    assert baseline != _collection_turn_sampling_seed(args, stream_index=7, turn_idx=3)
    assert baseline != _collection_turn_sampling_seed(
        SimpleNamespace(rollout_seed=44), stream_index=7, turn_idx=2
    )




@pytest.mark.unit
def test_alfworld_pool_prewarm_creates_workers_in_bounded_parallel_batches(monkeypatch, tmp_path):
    active = 0
    peak = 0

    class FakeWorker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.dirty = False

        def ping(self):
            return None

        def close(self):
            return None

    original_to_thread = asyncio.to_thread

    async def tracked_to_thread(function, *args, **kwargs):
        nonlocal active, peak
        if function is FakeWorker:
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            try:
                return function(*args, **kwargs)
            finally:
                active -= 1
        return await original_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(alfworld_envs, "ProcessAlfWorldEnvWorker", FakeWorker)
    monkeypatch.setattr(alfworld_envs.asyncio, "to_thread", tracked_to_thread)
    pool = alfworld_envs.AlfWorldEnvPool(
        pool_size=65,
        prewarm_batch_size=32,
        split="train",
        cache_dir=tmp_path,
        max_episode_steps=30,
    )
    asyncio.run(pool.prewarm_async(seed=43))
    assert len(pool._workers) == 65
    assert peak == 32


@pytest.mark.unit
def test_streaming_collection_reorders_once_and_writes_fifteen_atomic_shards(monkeypatch, tmp_path):
    writes = []
    manifest = {"corpus_schema_version": 4}
    monkeypatch.setattr(frozen_corpus, "_manifest_from_args", lambda _args: manifest)
    monkeypatch.setattr(frozen_corpus, "_manifest_write_kwargs", lambda _manifest: {})
    monkeypatch.setattr(frozen_corpus, "write_frozen_corpus_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        frozen_corpus,
        "_trajectory_from_rows",
        lambda _rows, *, stream_index, manifest: {"stream_index": stream_index},
    )
    monkeypatch.setattr(
        frozen_corpus,
        "write_frozen_trajectory_shard",
        lambda path, trajectories, **_kwargs: writes.append((Path(path).name, trajectories)),
    )
    samples = [Sample(index=index, metadata={"sample_group_index": index}) for index in reversed(range(960))]
    frozen_corpus.process(
        SimpleNamespace(alfworld_frozen_corpus_dir=str(tmp_path), alfworld_frozen_shard_size=64),
        samples,
        None,
    )
    assert [name for name, _ in writes] == [f"batch_{index:03d}.pt" for index in range(15)]
    assert [[row["stream_index"] for row in rows] for _, rows in writes] == [
        list(range(index * 64, (index + 1) * 64)) for index in range(15)
    ]


@pytest.mark.unit
def test_generic_streaming_scheduler_materializes_all_but_caps_live_coroutines(monkeypatch):
    peak = 0
    active = 0
    submitted = []
    completed = []

    class FakeState:
        remaining_batch_size = 0
        pendings = set()
        aborted = False

        def submit_generate_tasks(self, groups):
            nonlocal peak
            for group in groups:
                submitted.append(group[0].index)
                self.pendings.add(asyncio.create_task(run_group(group)))
            self.remaining_batch_size += len(groups)
            peak = max(peak, len(self.pendings))

        def reset(self):
            self.remaining_batch_size = 0
            self.pendings = set()
            self.aborted = False

    async def run_group(group):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep((20 - group[0].index) % 5 / 1000)
        active -= 1
        completed.append(group[0].index)
        return group

    state = FakeState()
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "reset_generate_request_metrics", lambda: None)
    monkeypatch.setattr(sglang_rollout, "get_generate_request_metrics", lambda: {})
    monkeypatch.setattr(sglang_rollout, "abort", lambda *_args, **_kwargs: asyncio.sleep(0, result=[]))
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=20,
        n_samples_per_prompt=1,
        rollout_trajectory_max_inflight=5,
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
        debug_train_only=False,
    )
    materialization_calls = []

    def data_source(count):
        materialization_calls.append(count)
        return [[Sample(index=index, response="ok", reward=0.0)] for index in range(count)]

    output, aborted = asyncio.run(sglang_rollout.generate_rollout_async(args, 0, data_source))
    assert len(output.samples) == 20
    assert [group[0].index for group in output.samples] == list(range(20))
    assert aborted == []
    assert materialization_calls == [20]
    assert submitted == list(range(20))
    assert completed != list(range(20))
    assert peak == 5


















@pytest.mark.unit
def test_iterative_collection_manifest_is_ordinary_only_and_has_no_integrity_hashes(monkeypatch):
    monkeypatch.setenv("ALFWORLD_FROZEN_PROMPT_NAME", "alfworld_ordinary_1024")
    manifest = _manifest_from_args(
        SimpleNamespace(
            alfworld_frozen_behavior_model="alfworld/c1/collection",
            alfworld_frozen_collection_schema="iterative",
            rollout_max_response_len=1024,
            rollout_temperature=1.0,
            rollout_top_p=1.0,
            rollout_top_k=-1,
            rollout_stop=None,
            rollout_stop_token_ids=None,
            rollout_skip_special_tokens=True,
            rollout_seed=43,
            alfworld_train_split="train",
        )
    )
    assert manifest["manifest_schema_version"] == 3
    assert manifest["corpus_schema_version"] == 4
    assert "expectation_enabled" not in manifest["prompt_contract"]
    assert not any("hash" in key or "digest" in key for key in manifest)


@pytest.mark.unit
def test_iterative_generation_emits_plain_teacher_view_metadata_for_sgs_scoring():
    class CharacterTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [ord(char) for char in text]

        def decode(self, token_ids, **_kwargs):
            return "".join(chr(token_id) for token_id in token_ids)

    response_text = "<thinking>inspect</thinking><action>look</action>"

    async def generator(**_kwargs):
        return {
            "text": response_text,
            "meta_info": {
                "output_token_logprobs": [[-0.1, ord(char)] for char in response_text],
                "finish_reason": {"type": "stop"},
            },
        }

    trajectory = {
        "trajectory_uid": "iterative-trajectory",
        "task_id": "task-1",
        "task_description": "look around",
        "split": "train",
        "success": True,
        "outcome": "success",
        "turns": [
            {
                "turn_idx": 0,
                "prompt_ids": [1, 2],
                "messages": [{"role": "user", "content": "ordinary prompt"}],
                "frozen_action": "look",
                "format_valid": True,
                "errors": [],
                "current_observation": "room",
                "next_observation": "room contents",
            }
        ],
    }
    sample = Sample(
        index=0,
        rollout_id=0,
        metadata={
            "frozen_trajectory": trajectory,
            "frozen_arm": "iterative_trajectory_distillation",
            "source_draw_id": 0,
            "branch_idx": 0,
            "traj_uid": "iterative-trajectory",
        },
    )
    rows = asyncio.run(
        generate(
            SimpleNamespace(
                alfworld_tokenizer=CharacterTokenizer(),
                alfworld_frozen_generator=generator,
                sgs_selection_fraction=0.05,
            ),
            sample,
            {},
        )
    )
    sdpo = rows[0].train_metadata["sdpo"]
    assert sdpo["sdpo_current_prompt_text"] == "ordinary prompt"
    assert sdpo["sdpo_current_raw_prompt"] == trajectory["turns"][0]["messages"]
    assert any(sdpo["sgs_action_token_mask"])
    assert "expectation_enabled" not in sdpo


@pytest.mark.unit
def test_offline_replay_emits_plain_teacher_view_metadata_for_sgs_scoring():
    class CharacterTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [ord(char) for char in text]

        def decode(self, token_ids, **_kwargs):
            return "".join(chr(token_id) for token_id in token_ids)

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            del add_special_tokens
            payload = {"input_ids": [ord(char) for char in text]}
            if return_offsets_mapping:
                payload["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
            return payload

    response_text = "<thinking>inspect</thinking><action>look</action>"
    trajectory = {
        "trajectory_uid": "offline-trajectory",
        "task_id": "task-1",
        "task_description": "look around",
        "split": "train",
        "success": True,
        "outcome": "success",
        "turns": [
            {
                "turn_idx": 0,
                "prompt_ids": [1, 2],
                "messages": [{"role": "user", "content": "ordinary prompt"}],
                "response_text": response_text,
                "response_ids": [ord(char) for char in response_text],
                "deployment_behavior_log_probs": np.asarray([-0.1] * len(response_text)),
                "finish_reason": "stop",
                "frozen_action": "look",
                "format_valid": True,
                "errors": [],
                "current_observation": "room",
                "next_observation": "room contents",
            }
        ],
    }
    sample = Sample(
        index=0,
        rollout_id=0,
        metadata={
            "frozen_trajectory": trajectory,
            "frozen_arm": "iterative_offline_sdpo",
            "source_draw_id": 0,
            "branch_idx": 0,
            "traj_uid": "offline-trajectory",
        },
    )

    rows = asyncio.run(
        generate(
            SimpleNamespace(
                alfworld_tokenizer=CharacterTokenizer(),
                sgs_selection_fraction=0.05,
                agent_frozen_guidance_summary_dir=None,
            ),
            sample,
            {},
        )
    )

    sdpo = rows[0].train_metadata["sdpo"]
    assert sdpo["sdpo_current_prompt_text"] == "ordinary prompt"
    assert sdpo["sdpo_current_raw_prompt"] == trajectory["turns"][0]["messages"]
    assert any(sdpo["sgs_action_token_mask"])
    assert sdpo["sgs_action_match"] is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("arm", "finish_reason", "expected_reward", "expected_status"),
    [
        ("iterative_offline_sdpo", "stop", 0.0, Sample.Status.COMPLETED),
        ("iterative_offline_sdpo", "length", 0.0, Sample.Status.TRUNCATED),
        ("iterative_success_filtered_sft", "stop", 1.0, Sample.Status.COMPLETED),
        ("iterative_success_filtered_sft", "length", 1.0, Sample.Status.COMPLETED),
    ],
)
def test_deployment_replay_preserves_offline_sdpo_and_rft_rewards_and_status(
    arm, finish_reason, expected_reward, expected_status
):
    class CharacterTokenizer:
        def decode(self, token_ids, **_kwargs):
            return "".join(chr(token_id) for token_id in token_ids)

    trajectory = {
        "trajectory_uid": "deployment-trajectory",
        "task_id": "task-1",
        "task_description": "look around",
        "split": "train",
        "success": True,
        "outcome": "success",
        "turns": [
            {
                "turn_idx": 0,
                "prompt_ids": [1, 2],
                "messages": [{"role": "user", "content": "ordinary prompt"}],
                "response_text": "look",
                "response_ids": [ord(char) for char in "look"],
                "deployment_behavior_log_probs": np.asarray([-0.1] * 4),
                "finish_reason": finish_reason,
                "frozen_action": "look",
                "format_valid": True,
                "errors": [],
                "current_observation": "room",
                "next_observation": "room contents",
            }
        ],
    }
    sample = Sample(
        index=0,
        rollout_id=0,
        metadata={
            "frozen_trajectory": trajectory,
            "frozen_arm": arm,
            "source_draw_id": 0,
            "branch_idx": 0,
            "traj_uid": "deployment-trajectory",
        },
    )
    rows = asyncio.run(
        generate(
            SimpleNamespace(alfworld_tokenizer=CharacterTokenizer(), sgs_selection_fraction=None),
            sample,
            {},
        )
    )
    assert len(rows) == 1
    assert rows[0].reward == expected_reward
    assert rows[0].status == expected_status
