from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from slime.agent.trajectory import TokenSegment, write_segment_to_sample
from slime.utils.types import Sample


def build_step_segment_sample(
    *,
    base_sample: Sample,
    tokenizer,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    rollout_log_probs: Sequence[float],
    reward: float,
    sample_index: int,
    rollout_id: int,
    metadata: dict[str, Any],
    response_text: str | None = None,
    session_id: str | None = None,
    non_generation_time: float = 0.0,
    extra_train_metadata: dict[str, Any] | None = None,
) -> Sample:
    response_id_list = list(response_ids)
    logprob_list = list(rollout_log_probs)
    if len(logprob_list) != len(response_id_list):
        raise ValueError(f"rollout_log_probs length {len(logprob_list)} != response length {len(response_id_list)}")

    segment = TokenSegment(
        prompt_ids=list(prompt_ids),
        response_ids=response_id_list,
        loss_mask=[1] * len(response_id_list),
        rollout_log_probs=logprob_list,
        metadata=dict(metadata),
    )
    sample = copy.copy(base_sample)
    write_segment_to_sample(sample, segment, reward, tokenizer)
    if response_text is not None:
        sample.response = response_text
    sample.index = int(sample_index)
    sample.rollout_id = int(rollout_id)
    sample.group_index = base_sample.group_index
    sample.metadata = {**(base_sample.metadata or {}), **metadata}
    sample.session_id = session_id
    sample.non_generation_time = float(non_generation_time)
    sample.train_metadata = {
        "uid": sample.metadata.get("uid"),
        "traj_uid": sample.metadata.get("traj_uid"),
        "turn_idx": sample.metadata.get("turn_idx"),
        "agent_task": sample.metadata.get("agent_task"),
        "agent_task_trace_dir": sample.metadata.get("agent_task_trace_dir"),
        "sample_rollout_id": sample.metadata.get("sample_rollout_id", sample.metadata.get("rollout_id")),
        "sample_index": sample.metadata.get("sample_index", sample.index),
        "group_index": sample.metadata.get("group_index", sample.group_index),
        "task_id": sample.metadata.get("task_id"),
        "eval_dataset_name": sample.metadata.get("eval_dataset_name"),
        "split": sample.metadata.get("split"),
    }
    if extra_train_metadata:
        sample.train_metadata.update(copy.deepcopy(extra_train_metadata))
    validate_segment_sample(sample, require_rollout_log_probs=True)
    return sample


def validate_segment_sample(sample: Sample, *, require_rollout_log_probs: bool) -> None:
    if not isinstance(sample.index, int):
        raise TypeError("segment sample.index must be an int")
    if not isinstance(sample.rollout_id, int):
        raise TypeError("segment sample.rollout_id must be an int")
    if sample.response_length < 0:
        raise ValueError("sample.response_length must be non-negative")
    if sample.loss_mask is None:
        raise ValueError("step segment samples must carry loss_mask")
    if len(sample.loss_mask) != sample.response_length:
        raise ValueError(f"loss_mask length {len(sample.loss_mask)} != response_length {sample.response_length}")
    if require_rollout_log_probs:
        if sample.rollout_log_probs is None:
            raise ValueError("formal ALFWorld samples must carry rollout_log_probs")
        if len(sample.rollout_log_probs) != sample.response_length:
            raise ValueError(
                f"rollout_log_probs length {len(sample.rollout_log_probs)} != response_length {sample.response_length}"
            )


def backfill_terminal_rewards(
    samples: Sequence[Sample],
    *,
    episode_reward: float,
    success: bool,
    invalid_format_penalty: float,
    done: bool,
    max_steps: int,
) -> None:
    episode_length = len(samples)
    for sample in samples:
        metadata = sample.metadata
        if "reward_penalty" in metadata:
            apply_format_penalty = bool(metadata["reward_penalty"])
        elif "format_penalty" in metadata:
            apply_format_penalty = bool(metadata["format_penalty"])
        elif "is_action_valid" in metadata:
            apply_format_penalty = not bool(metadata["is_action_valid"])
        else:
            apply_format_penalty = not bool(metadata.get("format_valid", True))
        shaped_reward = float(episode_reward) - float(invalid_format_penalty) * int(apply_format_penalty)
        sample.reward = shaped_reward
        metadata["raw_reward"] = float(episode_reward)
        metadata["episode_reward"] = float(episode_reward)
        metadata["success"] = bool(success)
        metadata["episode_length"] = episode_length
        metadata["is_terminal"] = int(metadata.get("turn_idx", -1)) == episode_length - 1 and bool(done)
        metadata["env_horizon_reached"] = (not done) and episode_length >= int(max_steps)
        metadata["grpo_reward"] = shaped_reward
