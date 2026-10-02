from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slime_plugins.agent_tasks.alfworld.frozen.epd import materialize_records


NUM_GPUS = 0


@pytest.mark.unit
def test_alfworld_epd_canonicalizes_iterative_schema_corpus(tmp_path: Path):
    from slime_plugins.agent_tasks.alfworld.frozen.data_source import (
        ITERATIVE_FROZEN_SCHEMA_VERSION,
        verify_frozen_trajectory,
        write_frozen_trajectory_shard,
    )
    from slime_plugins.agent_tasks.alfworld.frozen.epd import canonical_steps_from_corpus

    turn = {
        "turn_idx": 0,
        "messages": [{"role": "user", "content": "inspect the room"}],
        "prompt_ids": [11, 12],
        "response_text": "<action>look</action>",
        "response_ids": [21],
        "deployment_behavior_log_probs": torch.tensor([-0.1], dtype=torch.float32),
        "current_observation": "room",
        "next_observation": "room",
        "frozen_action": "look",
        "env_reward": 0.0,
        "finish_reason": "stop",
        "prompt_overlength": False,
        "prompt_truncated": False,
        "is_terminal": False,
        "format_valid": True,
        "format_invalid_reason": None,
        "errors": {},
    }
    trajectory = {
        "trajectory_uid": "alfworld-iterative-00000000",
        "task_id": "train:0",
        "task_description": "inspect the room",
        "split": "train",
        "success": False,
        "outcome": "failure",
        "episode_reward": 0.0,
        "termination_reason": "incomplete",
        "truncated": False,
        "horizon_reached": False,
        "errors": {},
        "runtime_task_identity": {"task_description": "inspect the room"},
        "provenance": {
            "stream_index": 0,
            "task_seed": 43,
            "source_group_index": 0,
            "seed": 43,
            "task_identity": {"task_description": "inspect the room"},
        },
        "turns": [turn],
    }
    corpus = tmp_path / "corpus"
    write_frozen_trajectory_shard(
        corpus / "batch_000.pt",
        [trajectory],
        schema_version=ITERATIVE_FROZEN_SCHEMA_VERSION,
        require_full_shard=False,
    )

    steps = canonical_steps_from_corpus(corpus, expected_trajectories=1, expected_steps=1)

    assert steps[0]["identity"] == "alfworld-iterative-00000000/turn/0000"
    assert steps[0]["source_shard"] == "batch_000.pt"
    for malformed_provenance in (None, []):
        malformed = {**trajectory, "provenance": malformed_provenance}
        with pytest.raises(TypeError, match="provenance must be a mapping"):
            verify_frozen_trajectory(malformed)


@pytest.mark.unit
def test_stream_bound_sampling_uses_public_collection_flag():
    from slime_plugins.agent_tasks.alfworld.generate import _stream_bound_sampling_enabled

    assert _stream_bound_sampling_enabled(SimpleNamespace(alfworld_stream_bound_sampling=True))
    assert not _stream_bound_sampling_enabled(
        SimpleNamespace(alfworld_stream_bound_sampling=False)
    )
    assert not _stream_bound_sampling_enabled(SimpleNamespace())


def _step() -> dict:
    turn = {
        "turn_idx": 0,
        "messages": [{"role": "user", "content": "ordinary prompt"}],
        "prompt_ids": [11, 12, 13],
        "response_text": "<action>look</action>",
        "response_ids": [21],
        "current_observation": "room",
        "next_observation": "room",
        "frozen_action": "look",
        "env_reward": 0.0,
        "finish_reason": "stop",
        "is_terminal": False,
    }
    trajectory = {
        "trajectory_uid": "alfworld-uid-0000",
        "task_id": "train:0",
        "task_description": "inspect the room",
        "split": "train",
        "success": False,
        "outcome": "failure",
        "turns": [turn],
    }
    return {"trajectory": trajectory, "turn": turn, "trajectory_uid": trajectory["trajectory_uid"], "turn_idx": 0}


@pytest.mark.unit
def test_alfworld_epd_direct_materialization_has_no_integrity_hash_fields(tmp_path: Path):
    async def generate(_request, _endpoint):
        return {
            "text": "<action>look</action>",
            "meta_info": {"output_ids": [42], "finish_reason": {"type": "stop"}},
        }

    binding = {
        "corpus_dir": "/workspace/slime/corpus",
        "materialization_identity": "alfworld-epd-c1-teacher-iter-29",
        "model_path": "/workspace/slime/model",
        "teacher_iteration": 29,
    }
    records = asyncio.run(
        materialize_records(
            [_step()],
            output_dir=tmp_path,
            endpoints=["fake://0"],
            tokenizer=None,
            generate_request=generate,
            expected_count=1,
            worker_count=1,
            direct_binding=binding,
        )
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["binding_mode"] == "direct_v1"
    assert manifest["direct_binding"] == binding
    assert records[0]["direct_binding"] == binding
    assert {(tmp_path / name).stat().st_mode & 0o777 for name in ("manifest.json", "index.json", "shard_0000.jsonl")} == {
        0o644
    }
    assert not {
        key
        for row in [manifest, *manifest["shards"], *records]
        for key in row
        if "hash" in key or "sha256" in key
    }


@pytest.mark.unit
def test_alfworld_epd_direct_loader_requires_exact_binding(tmp_path: Path, monkeypatch):
    import slime_plugins.agent_tasks.alfworld.frozen.epd as epd
    import slime_plugins.agent_tasks.alfworld.frozen.generate as generate

    target = tmp_path / "targets"
    target.mkdir()
    binding = {
        "corpus_dir": "/workspace/slime/corpus",
        "materialization_identity": "alfworld-epd-c1-teacher-iter-29",
        "model_path": "/workspace/slime/model",
        "teacher_iteration": 29,
    }
    manifest = {
        "schema_version": epd.EPD_MANIFEST_SCHEMA_VERSION,
        "kind": "epd_teacher_targets",
        "status": "complete",
        "binding_mode": "direct_v1",
        "direct_binding": binding,
        "target_count": 1,
        "canonical_turn_count": 1,
        "trajectory_count": 960,
    }
    verification = {
        "status": "verified",
        "binding_mode": "direct_v1",
        "direct_binding": binding,
        "corpus_bound": True,
        "trajectory_count": 960,
        "canonical_turn_count": 1,
        "target_count": 1,
    }
    (target / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    (target / "verification.json").write_text(json.dumps(verification) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        epd,
        "load_target_records",
        lambda _root: [
            {
                "identity": "uid/turn/0000",
                "trajectory_uid": "uid",
                "turn_idx": 0,
                "canonical_index": 0,
                "original_prompt_ids": [11],
                "response_ids": [42],
                "response_text": "<action>look</action>",
                "validation": {"valid": True, "format_valid": True, "invalid_reason": None},
            }
        ],
    )
    args = SimpleNamespace(
        alfworld_epd_target_dir=str(target),
        alfworld_epd_direct_binding=binding,
        alfworld_epd_target_count=1,
        alfworld_frozen_expected_trajectories=960,
    )
    generate._EPD_TARGET_CACHE.clear()
    _, index = generate._load_epd_target_index(args)
    assert list(index) == ["uid/turn/0000"]
    args.alfworld_epd_direct_binding = {**binding, "teacher_iteration": 28}
    with pytest.raises(ValueError, match="direct target verification"):
        generate._load_epd_target_index(args)





