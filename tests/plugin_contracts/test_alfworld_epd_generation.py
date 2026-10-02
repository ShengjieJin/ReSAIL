from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from slime.utils.types import Sample


class _Tokenizer:
    def decode(self, token_ids, **_kwargs):
        return "".join(chr(int(token_id)) for token_id in token_ids)


def _trajectory():
    return {
        "trajectory_uid": "uid-1",
        "task_id": "task-1",
        "split": "train",
        "outcome": "success",
        "success": True,
        "turns": [
            {
                "turn_idx": 0,
                "prompt_ids": [11, 12],
                "frozen_action": "look",
                "current_observation": "obs0",
                "next_observation": "obs1",
            },
            {
                "turn_idx": 1,
                "prompt_ids": [21, 22],
                "frozen_action": "open",
                "current_observation": "obs1",
                "next_observation": "obs2",
            },
        ],
    }


@pytest.mark.unit
def test_epd_generation_uses_original_prefix_and_strips_teacher_prompt(monkeypatch):
    import slime_plugins.agent_tasks.alfworld.frozen.generate as module

    manifest = {"schema_version": 1, "direct_binding": {"materialization_identity": "test-epd"}}
    targets = {
        "uid-1/turn/0000": {
            "canonical_index": 0,
            "original_prompt_ids": [11, 12],
            "response_ids": [31, 32],
            "response_text": "AB",
            "response_log_probs": None,
            "target_finish_reason": "stop",
            "target_retry_count": 0,
            "target_primary_seed": 42,
            "target_teacher_signal_type": "solution_demo",
            "target_teacher_reference_scope": "full_frozen_trajectory",
        },
        "uid-1/turn/0001": {
            "canonical_index": 1,
            "original_prompt_ids": [21, 22],
            "response_ids": [41],
            "response_text": "C",
            "response_log_probs": [-0.2],
            "target_finish_reason": "stop",
            "target_retry_count": 1,
            "target_primary_seed": 42,
            "target_teacher_signal_type": "solution_demo",
            "target_teacher_reference_scope": "full_frozen_trajectory",
        },
    }
    monkeypatch.setattr(module, "_get_tokenizer", lambda _args: _Tokenizer())

    def fake_index(_args):
        return manifest, targets

    monkeypatch.setattr(module, "_load_epd_target_index", fake_index)
    sample = Sample(
        group_index=3,
        index=7,
        rollout_id=7,
        prompt="uid-1",
        metadata={
            "traj_uid": "uid-1-bundle-00000007-branch-00",
            "source_draw_id": 3,
            "source_turn_idx": None,
            "branch_idx": 0,
            "frozen_arm": "iterative_epd",
            "frozen_trajectory": _trajectory(),
        },
    )
    rows = asyncio.run(module.generate(SimpleNamespace(), sample, {}, evaluation=False))

    assert [row.tokens for row in rows] == [[11, 12, 31, 32], [21, 22, 41]]
    assert [row.rollout_id for row in rows] == [7, 7]
    assert all(row.reward == 0.0 and row.loss_mask == [1] * row.response_length for row in rows)
    assert all("frozen_trajectory" not in row.metadata for row in rows)
    assert all("teacher_prompt_ids" not in row.metadata for row in rows)
    assert all("teacher_messages" not in row.metadata for row in rows)


@pytest.mark.unit
def test_epd_generation_rejects_prefix_mismatch(monkeypatch):
    import slime_plugins.agent_tasks.alfworld.frozen.generate as module

    monkeypatch.setattr(module, "_get_tokenizer", lambda _args: _Tokenizer())
    monkeypatch.setattr(
        module,
        "_load_epd_target_index",
        lambda _args: (
            {"schema_version": 1, "direct_binding": {"materialization_identity": "test-epd"}},
            {
                "uid-1/turn/0000": {
                    "canonical_index": 0,
                    "original_prompt_ids": [99],
                    "response_ids": [31],
                    "response_text": "A",
                    "response_log_probs": [-0.1],
                    "target_finish_reason": "stop",
                    "target_retry_count": 0,
                    "target_primary_seed": 42,
                    "target_teacher_signal_type": None,
                    "target_teacher_reference_scope": None,
                }
            },
        ),
    )
    sample = Sample(
        index=1,
        rollout_id=1,
        metadata={
            "traj_uid": "uid-1-bundle-00000001-branch-00",
            "source_draw_id": 1,
            "source_turn_idx": 0,
            "branch_idx": 0,
            "frozen_arm": "iterative_epd",
            "frozen_trajectory": _trajectory(),
        },
    )
    with pytest.raises(ValueError, match="original prompt mismatch"):
        asyncio.run(module.generate(SimpleNamespace(), sample, {}, evaluation=False))
