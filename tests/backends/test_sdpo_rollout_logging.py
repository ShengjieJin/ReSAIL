from __future__ import annotations

from argparse import Namespace

import pytest
import torch


@pytest.mark.unit
def test_log_metric_key_keeps_explicit_top_level_metrics():
    from slime.backends.megatron_utils import data as data_module

    key = f"{data_module.TOP_LEVEL_LOG_METRIC_PREFIX}sdpo_token_weights/weights/active_mean"

    assert data_module._log_metric_key("rollout", key) == "sdpo_token_weights/weights/active_mean"
    assert data_module._log_metric_key("rollout", "self_distillation/token_weights_active_mean") == (
        "rollout/self_distillation/token_weights_active_mean"
    )


@pytest.mark.unit
def test_log_rollout_data_preserves_explicit_top_level_metrics(monkeypatch):
    from slime.backends.megatron_utils import data as data_module

    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(data_module.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_world_size", lambda with_context_parallel=False: 1)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_src_rank", lambda with_context_parallel=True: 0)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group_gloo", lambda with_context_parallel=True: None)

    def fake_reduce(log_dict, *, dp_size, dp_src_rank, dp_group):
        del dp_size, dp_src_rank, dp_group
        reduced = {}
        for key, value in log_dict.items():
            reduced[key] = value[0] / value[1] if isinstance(value, tuple) else value
        return reduced

    logged = {}
    monkeypatch.setattr(data_module, "gather_and_reduce_log_dict", fake_reduce)
    monkeypatch.setattr(data_module.logging_utils, "log", lambda args, log_dict, step_key=None: logged.update(log_dict))

    top_level_key = f"{data_module.TOP_LEVEL_LOG_METRIC_PREFIX}sdpo_token_weights/weights/active_mean"
    rollout_data = {
        "partition": [0],
        "response_lengths": [1],
        "loss_masks": [torch.tensor([1.0])],
        "total_lengths": [2],
        "rewards": [1.0],
        "truncated": [0],
        "global_batch_sizes": [1],
        "num_microbatches": [1],
        "micro_batch_indices": [[0]],
        "sdpo_token_weights": [torch.tensor([1.0])],
        top_level_key: torch.tensor(0.75),
    }
    args = Namespace(
        ci_test=False,
        log_correct_samples=False,
        log_multi_turn=False,
        log_passrate=False,
        qkv_format="thd",
        wandb_always_use_train_step=False,
    )

    data_module.log_rollout_data(7, args, rollout_data)

    assert logged["sdpo_token_weights/weights/active_mean"] == pytest.approx(0.75)
    assert "rollout/sdpo_token_weights/weights/active_mean" not in logged
    assert "rollout/sdpo_token_weights" not in logged
    assert logged["rollout/step"] == 7


@pytest.mark.unit
def test_log_rollout_data_skips_prompt_bearing_sdpo_fields(monkeypatch):
    from slime.backends.megatron_utils import data as data_module

    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(data_module.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_world_size", lambda with_context_parallel=False: 1)

    captured = {}

    def fake_gather_log_data(prefix, args, rollout_id, log_dict):
        captured.update(log_dict)
        return {}

    monkeypatch.setattr(data_module, "gather_log_data", fake_gather_log_data)

    rollout_data = {
        "partition": [0, 1],
        "response_lengths": [1, 1],
        "loss_masks": [[1], [1]],
        "total_lengths": [2, 2],
        "rewards": [1.0, 0.0],
        "truncated": [0, 0],
        "global_batch_sizes": [2],
        "num_microbatches": [2],
        "micro_batch_indices": [[0], [1]],
        "sdpo_metadata": [{"uid": "u0"}, {"uid": "u1"}],
        "sdpo_teacher_prompt_text": ["hidden prompt 0", "hidden prompt 1"],
        "sdpo_teacher_messages": [
            [{"role": "user", "content": "hidden prompt 0"}],
            [{"role": "user", "content": "hidden prompt 1"}],
        ],
        "sdpo_teacher_signal_type": ["none", "none"],
        "sdpo_selected_success_traj_uid": ["success-0", None],
        "sdpo_teacher_representations": [[[1.0, 0.0]], [[0.0, 1.0]]],
        "self_distillation_mask": [0.0, 1.0],
        "sdpo_loss_weights": [0.0, 1.0],
        "sgs_plain_prompt_text": ["plain 0", "plain 1"],
        "sgs_plain_messages": [[], []],
        "sgs_action_token_mask": [[1], [1]],
        "sgs_action_alignment_valid": [True, True],
        "sgs_action_alignment_reason": [None, None],
        "sgs_action_match": [True, False],
        "sgs_online_action": ["go north", "go south"],
        "sgs_online_format_valid": [True, True],
        "sgs_frozen_action": ["go north", "go north"],
        "sgs_source_draw_id": [0, 1],
        "sgs_task_id": ["task-0", "task-1"],
        "sgs_split": ["train", "train"],
    }

    args = Namespace(
        ci_test=False,
        log_correct_samples=False,
        log_multi_turn=False,
        log_passrate=False,
        qkv_format="thd",
    )
    data_module.log_rollout_data(0, args, rollout_data)

    assert "sdpo_metadata" not in captured
    assert "sdpo_teacher_prompt_text" not in captured
    assert "sdpo_teacher_messages" not in captured
    assert "sdpo_teacher_signal_type" not in captured
    assert "sdpo_selected_success_traj_uid" not in captured
    assert "sdpo_teacher_representations" not in captured
    assert not any(key.startswith("sgs_") for key in captured)
    assert captured["self_distillation_mask"] == (1.0, 2)
    assert captured["sdpo_loss_weights"] == (1.0, 2)


@pytest.mark.unit
def test_debug_train_data_summarizes_sdpo_teacher_representations():
    from slime.utils import train_dump_utils

    rollout_data = {"sdpo_teacher_representations": [torch.zeros(2, 4, dtype=torch.bfloat16)]}

    safe = train_dump_utils._debug_safe_rollout_data(rollout_data)

    assert safe["sdpo_teacher_representations"] == [
        {"shape": (2, 4), "dtype": "torch.bfloat16", "device": "cpu"}
    ]
    assert rollout_data["sdpo_teacher_representations"][0].shape == (2, 4)
