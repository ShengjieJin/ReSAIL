from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.frozen.corpus import process_iterative_corpus, verify_iterative_corpus
from slime_plugins.agent_tasks.common.frozen.data_source import FrozenDataSource
from slime_plugins.agent_tasks.common.frozen import epd as common_epd
from slime_plugins.agent_tasks.common.frozen import generate as common_generate
from slime_plugins.agent_tasks.common.frozen.epd import build_sampling_params, validate_teacher_output
from slime_plugins.agent_tasks.common.frozen.sampling import eval_turn_seed, stream_turn_seed
from slime_plugins.agent_tasks.common.frozen.contracts import load_shard
from slime_plugins.agent_tasks.common.frozen.guidance import SUMMARY_SCHEMA_VERSION, build_summary_request


NUM_GPUS = 0


class _CharacterTokenizer:
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


def _row(stream_index: int, *, success: bool) -> Sample:
    task_id = f"textcraft_{stream_index % 2}"
    seed = stream_turn_seed(task="textcraft", cycle_seed=42, stream_index=stream_index, turn_idx=0)
    return Sample(
        index=stream_index,
        rollout_id=stream_index,
        group_index=stream_index,
        tokens=[10, 20],
        response="<action>inventory</action>",
        response_length=1,
        reward=float(success),
        loss_mask=[1],
        rollout_log_probs=[-0.2],
        status=Sample.Status.COMPLETED,
        metadata={
            "uid": f"uid-{stream_index}",
            "traj_uid": f"traj-{stream_index}",
            "sample_group_index": stream_index,
            "source_group_index": stream_index % 2,
            "seed": 42 + stream_index % 2,
            "split": "train",
            "turn_idx": 0,
            "task_id": task_id,
            "task_description": "Goal: inventory",
            "runtime_task_identity": {
                "task_id": task_id,
                "data_idx": stream_index % 2,
                "task_description": "Goal: inventory",
                "split": "train",
            },
            "messages": [{"role": "user", "content": "Goal: inventory"}],
            "current_observation": "Inventory: empty",
            "next_observation": "Inventory: empty",
            "projected_action": "inventory",
            "score": float(success),
            "finish_reason": "stop",
            "prompt_overlength": False,
            "history_auto_truncated": False,
            "is_terminal": success,
            "format_valid": True,
            "invalid_reason": None,
            "episode_error": 0.0,
            "env_step_failed": False,
            "episode_reward": float(success),
            "env_horizon_reached": not success,
            "success": success,
            "sampling_seed": seed,
        },
    )


@pytest.mark.unit
def test_textcraft_offline_sdpo_replay_emits_sgs_teacher_views():
    response = "<thinking>inspect inventory</thinking><action>inventory</action>"
    trajectory = {
        "trajectory_uid": "textcraft-iterative-00000000",
        "task_id": "textcraft_0",
        "task_description": "Goal: inventory",
        "split": "train",
        "success": True,
        "outcome": "success",
        "turns": [
            {
                "turn_idx": 0,
                "prompt_ids": [1, 2],
                "messages": [{"role": "user", "content": "Goal: inventory"}],
                "response_text": response,
                "response_ids": [ord(char) for char in response],
                "deployment_behavior_log_probs": np.asarray([-0.1] * len(response)),
                "finish_reason": "stop",
                "frozen_action": "inventory",
                "format_valid": True,
                "errors": [],
                "current_observation": "Inventory: empty",
                "next_observation": "Inventory: empty",
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
            "training_update": 0,
            "traj_uid": trajectory["trajectory_uid"],
        },
    )
    rows = asyncio.run(
        common_generate.generate(
            SimpleNamespace(
                agent_frozen_task="textcraft",
                textcraft_tokenizer=_CharacterTokenizer(),
                sgs_selection_fraction=0.25,
                agent_frozen_guidance_summary_dir=None,
            ),
            sample,
            {},
        )
    )
    sdpo = rows[0].train_metadata["sdpo"]
    assert sdpo["sgs_action_match"] is True
    assert sdpo["sgs_online_action"] == "inventory"
    assert sdpo["sgs_frozen_action"] == "inventory"
    assert any(sdpo["sgs_action_token_mask"])


@pytest.mark.unit
def test_textcraft_iterative_corpus_is_task_namespaced_and_success_repeats(tmp_path: Path):
    corpus = tmp_path / "corpus"
    args = SimpleNamespace(
        agent_frozen_corpus_dir=str(corpus),
        agent_frozen_shard_size=2,
        agent_frozen_behavior_model="qwen3-4b/base",
        rollout_seed=42,
        rollout_max_response_len=512,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        textcraft_train_split="train",
    )
    process_iterative_corpus(args, [[_row(0, success=True)], [_row(1, success=False)]], task="textcraft")

    stats = verify_iterative_corpus(
        corpus,
        expected_trajectories=2,
        expected_task="textcraft",
        expected_cycle_seed=42,
    )
    assert stats["trajectory_count"] == 2
    assert stats["success_count"] == 1
    assert json.loads((corpus / "manifest.json").read_text())["prompt_contract"]["response_max_tokens"] == 512
    steps = common_epd.canonical_steps_from_corpus(corpus, expected_trajectories=2)
    assert [step["identity"] for step in steps] == [
        "textcraft-iterative-00000000/turn/0000",
        "textcraft-iterative-00000001/turn/0000",
    ]

    source = FrozenDataSource(
        SimpleNamespace(
            agent_frozen_arm="iterative_success_filtered_sft",
            agent_frozen_task="textcraft",
            agent_frozen_corpus_dir=str(corpus),
            agent_frozen_expected_trajectories=2,
            n_samples_per_prompt=1,
            rollout_seed=42,
            rollout_shuffle=True,
        )
    )
    draws = source.get_samples(3)
    assert [group[0].metadata["source_trajectory_uid"] for group in draws] == [
        "textcraft-iterative-00000000"
    ] * 3

    trajectories = [trajectory for path in sorted(corpus.glob("batch_*.pt")) for trajectory in load_shard(path)]
    kept = trajectories[0]
    skipped = trajectories[1]
    request = build_summary_request(kept)
    record = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        **request,
        "summary": (
            "Guidance summary:\n"
            "- Minimal plan: inspect inventory.\n"
            "- Critical actions: use `inventory`.\n"
            "- Checks: verify the observation.\n"
            "- Avoid: invalid commands."
        ),
        "attempt_count": 1,
    }
    guidance = tmp_path / "guidance"
    guidance.mkdir()
    manifest = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "complete",
        "trajectory_count": 2,
        "valid_guideline_count": 1,
        "empty_guideline_count": 1,
        "empty_guideline_uids": [skipped["trajectory_uid"]],
        "empty_guideline_policy": "skip_trajectory",
        "corpus_dir": str(corpus),
    }
    (guidance / "manifest.json").write_text(json.dumps(manifest) + "\n")
    (guidance / "verification.json").write_text(
        json.dumps({"status": "verified", "manifest": manifest}) + "\n"
    )
    (guidance / "summaries.jsonl").write_text(json.dumps(record) + "\n")
    filtered = FrozenDataSource(
        SimpleNamespace(
            agent_frozen_arm="iterative_offline_sdpo",
            agent_frozen_task="textcraft",
            agent_frozen_corpus_dir=str(corpus),
            agent_frozen_expected_trajectories=2,
            agent_frozen_guidance_summary_dir=str(guidance),
            agent_frozen_empty_guideline_policy="skip_trajectory",
            n_samples_per_prompt=1,
            rollout_seed=42,
            rollout_shuffle=False,
        )
    )
    assert len(filtered) == 1
    assert [group[0].metadata["source_trajectory_uid"] for group in filtered.get_samples(2)] == [
        kept["trajectory_uid"],
        kept["trajectory_uid"],
    ]


@pytest.mark.unit
def test_textcraft_sampling_and_epd_projection_contracts():
    first = eval_turn_seed(task="textcraft", replicate_seed=314159, task_id="textcraft_7", turn_idx=0)
    assert first == eval_turn_seed(task="textcraft", replicate_seed=314159, task_id="textcraft_7", turn_idx=0)
    assert first != eval_turn_seed(task="textcraft", replicate_seed=314160, task_id="textcraft_7", turn_idx=0)
    assert build_sampling_params(temperature=0.4, max_new_tokens=512)["max_new_tokens"] == 512
    assert validate_teacher_output(
        "<thinking>gather</thinking><action>get 1 lilac</action>", "stop", [1]
    )["valid"] is True
    invalid = validate_teacher_output("<action>look</action>", "stop", [1])
    assert invalid["valid"] is False
    assert "invalid_textcraft_action" in str(invalid["invalid_reason"])


@pytest.mark.unit
def test_common_epd_materializer_binds_textcraft_profile_projection_and_512_limit(monkeypatch):
    captured = {}

    async def fake_materialize(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(common_epd._backend, "materialize_records", fake_materialize)
    assert asyncio.run(common_epd.materialize_records([])) == []
    assert captured["metadata_profile"] == "textcraft"
    assert captured["teacher_output_validator"] is common_epd.validate_teacher_output
    assert captured["sampling_params_overrides"] == {"max_new_tokens": 512}


@pytest.mark.unit
def test_epd_target_index_is_loaded_once_per_rollout_process(tmp_path: Path, monkeypatch):
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    binding = {"materialization_identity": "test-direct-binding"}
    manifest = {
        "schema_version": common_epd.EPD_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "binding_mode": "direct_v1",
        "direct_binding": binding,
        "trajectory_count": 1,
        "target_count": 1,
    }
    verification = {
        "status": "verified",
        "direct_binding": binding,
        "corpus_bound": True,
        "trajectory_count": 1,
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (target_dir / "verification.json").write_text(json.dumps(verification), encoding="utf-8")
    calls = []
    records = [{"identity": "trajectory/turn/0000"}]

    def fake_load_target_records(root):
        calls.append(root)
        return records

    monkeypatch.setattr(common_generate, "load_target_records", fake_load_target_records)
    args = SimpleNamespace(
        agent_frozen_epd_target_dir=str(target_dir),
        agent_frozen_epd_direct_binding=binding,
        agent_frozen_expected_trajectories=1,
    )

    first = common_generate._epd_targets(args)
    second = common_generate._epd_targets(args)

    assert first is not second
    assert first[0] is second[0]
    assert first[1] is second[1]
    assert calls == [target_dir]
