from __future__ import annotations

import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from slime.utils.types import Sample


def flatten_samples(samples: Any) -> list[Sample]:
    if not samples:
        return []
    if isinstance(samples[0], Sample):
        return list(samples)
    out: list[Sample] = []
    for item in samples:
        out.extend(flatten_samples(item))
    return out


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]], *, limit: int | None = None) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as writer:
        for row in rows:
            if limit is not None and count >= limit:
                break
            writer.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def sample_log_record(sample: Sample, *, rollout_id: int, evaluation: bool) -> dict[str, Any]:
    metadata = dict(sample.metadata or {})
    status = sample.status.value if isinstance(sample.status, Sample.Status) else str(sample.status)
    model_input = (
        metadata.get("rendered_prompt")
        or metadata.get("raw_prompt")
        or metadata.get("prompt_text")
        or metadata.get("prompt_text_preview")
        or metadata.get("raw_prompt_preview")
        or sample.prompt
    )
    model_input_text = "" if model_input is None else str(model_input)
    return {
        "rollout_id": rollout_id,
        "sample_rollout_id": sample.rollout_id,
        "sample_index": sample.index,
        "group_index": sample.group_index,
        "evaluation": evaluation,
        "uid": metadata.get("uid"),
        "traj_uid": metadata.get("traj_uid"),
        "turn_idx": metadata.get("turn_idx"),
        "repeat_idx": metadata.get("repeat_idx"),
        "seed": metadata.get("seed"),
        "requested_seed": metadata.get("requested_seed"),
        "actual_seed": metadata.get("actual_seed"),
        "retry_count": metadata.get("retry_count"),
        "split": metadata.get("split"),
        "task_id": metadata.get("task_id"),
        "eval_dataset_name": metadata.get("eval_dataset_name"),
        "status": status,
        "reward": sample.reward,
        "raw_reward": metadata.get("raw_reward"),
        "episode_reward": metadata.get("episode_reward"),
        "score": metadata.get("score"),
        "env_reward_sum": metadata.get("env_reward_sum"),
        "won": metadata.get("won"),
        "state_text": metadata.get("state_text"),
        "prompt_agent_coord": metadata.get("prompt_agent_coord"),
        "prompt_box_coords": metadata.get("prompt_box_coords"),
        "prompt_target_coords": metadata.get("prompt_target_coords"),
        "success": metadata.get("success"),
        "is_terminal": metadata.get("is_terminal"),
        "episode_length": metadata.get("episode_length"),
        "episode_seconds": metadata.get("episode_seconds"),
        "env_horizon_reached": metadata.get("env_horizon_reached"),
        "episode_error": metadata.get("episode_error"),
        "episode_error_count": metadata.get("episode_error_count"),
        "episode_error_type": metadata.get("episode_error_type"),
        "episode_error_stage": metadata.get("episode_error_stage"),
        "episode_error_message": metadata.get("episode_error_message"),
        "env_step_failed": metadata.get("env_step_failed"),
        "response_length": sample.response_length,
        "loss_mask_sum": sum(sample.loss_mask or []),
        "rollout_log_probs_len": len(sample.rollout_log_probs or []),
        "format_valid": metadata.get("format_valid"),
        "format_penalty": metadata.get("format_penalty"),
        "action_penalty": metadata.get("action_penalty"),
        "ineffective_action_penalty": metadata.get("ineffective_action_penalty"),
        "reward_penalty": metadata.get("reward_penalty"),
        "reward_source": metadata.get("reward_source"),
        "terminate_reward_enabled": metadata.get("terminate_reward_enabled"),
        "terminate_selected": metadata.get("terminate_selected"),
        "terminate_correct": metadata.get("terminate_correct"),
        "terminate_false": metadata.get("terminate_false"),
        "terminate_reward": metadata.get("terminate_reward"),
        "solver_unknown": metadata.get("solver_unknown"),
        "solver_solvable": metadata.get("solver_solvable"),
        "solver_reason": metadata.get("solver_reason"),
        "solver_states_explored": metadata.get("solver_states_explored"),
        "solver_seconds": metadata.get("solver_seconds"),
        "termination_reason": metadata.get("termination_reason"),
        "is_action_valid": metadata.get("is_action_valid"),
        "missing_action_tag": metadata.get("missing_action_tag"),
        "missing_thinking_tag": metadata.get("missing_thinking_tag"),
        "contains_chinese": metadata.get("contains_chinese"),
        "admissible_member": metadata.get("admissible_member"),
        "projected_action": metadata.get("projected_action"),
        "projected_action_id": metadata.get("projected_action_id"),
        "action_is_effective": metadata.get("action_is_effective"),
        "invalid_reason": metadata.get("invalid_reason"),
        "finish_reason": metadata.get("finish_reason"),
        "prompt_tokens": metadata.get("prompt_tokens"),
        "raw_prompt_tokens": metadata.get("raw_prompt_tokens"),
        "prompt_overlength": metadata.get("prompt_overlength"),
        "prompt_max_tokens": metadata.get("prompt_max_tokens"),
        "failed_prompt_tokens": metadata.get("failed_prompt_tokens"),
        "failed_raw_prompt_tokens": metadata.get("failed_raw_prompt_tokens"),
        "history_steps_total": metadata.get("history_steps_total"),
        "history_steps_kept": metadata.get("history_steps_kept"),
        "history_auto_truncated": metadata.get("history_auto_truncated"),
        "history_auto_truncated_dropped": metadata.get("history_auto_truncated_dropped"),
        "history_assistant_content": metadata.get("history_assistant_content"),
        "history_format": metadata.get("history_format"),
        "full_prompt_tokens_before_truncation": metadata.get("full_prompt_tokens_before_truncation"),
        "observation_truncated": metadata.get("observation_truncated"),
        "observation_tokens": metadata.get("observation_tokens"),
        "raw_observation_tokens": metadata.get("raw_observation_tokens"),
        "max_obs_length": metadata.get("max_obs_length"),
        "effective_max_obs_length": metadata.get("effective_max_obs_length"),
        "observation_auto_truncated_to_fit_prompt": metadata.get("observation_auto_truncated_to_fit_prompt"),
        "response_tokens": metadata.get("response_tokens"),
        "cached_tokens": metadata.get("cached_tokens"),
        "prefix_cache_hit_rate": metadata.get("prefix_cache_hit_rate"),
        "generation_seconds": metadata.get("generation_seconds"),
        "generation_request_seconds": metadata.get("generation_request_seconds"),
        "generation_url": metadata.get("generation_url"),
        "multimodal_train_input_keys": metadata.get("multimodal_train_input_keys"),
        "multimodal_train_input_shapes": metadata.get("multimodal_train_input_shapes"),
        "image_count": metadata.get("image_count"),
        "image_width": metadata.get("image_width"),
        "image_height": metadata.get("image_height"),
        "image_size": metadata.get("image_size"),
        "session_id": metadata.get("session_id"),
        "worker_id": metadata.get("worker_id"),
        "worker_reused": metadata.get("worker_reused"),
        "worker_created": metadata.get("worker_created"),
        "env_step_seconds": metadata.get("env_step_seconds"),
        "reset_seconds": metadata.get("reset_seconds"),
        "task_description": metadata.get("task_description"),
        "admissible_actions_preview": metadata.get("admissible_actions_preview"),
        "sampling_seed": metadata.get("sampling_seed"),
        "eval_replicate_seed": metadata.get("eval_replicate_seed"),
        "eval_sampling_seed_namespace": metadata.get("eval_sampling_seed_namespace"),
        "model_input": model_input_text,
        "response": sample.response,
    }


def summarize_agent_task_samples(
    samples: list[Sample],
    *,
    metric_root: str,
    max_episode_errors: int | None = None,
    max_episode_error_rate: float | None = None,
) -> dict[str, float]:
    if not samples:
        return {}
    step_rows = [sample.metadata or {} for sample in samples]
    episode_rows = _episode_rows(samples)
    episode_count = float(len(episode_rows))
    episode_error_count = sum(
        _float(row.get("episode_error_count", row.get("episode_error", 0.0))) for row in episode_rows
    )
    episode_error_rate = episode_error_count / episode_count if episode_count else 0.0
    budget_exceeded = _budget_exceeded(
        episode_error_count=episode_error_count,
        episode_error_rate=episode_error_rate,
        max_episode_errors=max_episode_errors,
        max_episode_error_rate=max_episode_error_rate,
    )
    prompt_tokens = [_float(row.get("prompt_tokens")) for row in step_rows if row.get("prompt_tokens") is not None]
    cached_tokens = [_float(row.get("cached_tokens")) for row in step_rows if row.get("cached_tokens") is not None]
    prompt_token_sum = sum(prompt_tokens)
    cached_token_sum = sum(cached_tokens)
    root = metric_root.rstrip("/")
    return {
        f"{root}/episode/count": episode_count,
        f"{root}/episode/success_rate": _rate(episode_rows, "success"),
        f"{root}/episode/reward_mean": _mean(episode_rows, "episode_reward"),
        f"{root}/episode/length_mean": _mean(episode_rows, "episode_length"),
        f"{root}/episode/horizon_rate": _rate(episode_rows, "env_horizon_reached"),
        f"{root}/error/episode_count": float(episode_error_count),
        f"{root}/error/episode_rate": episode_error_rate,
        f"{root}/error/env_step_count": float(sum(int(bool(row.get("env_step_failed"))) for row in step_rows)),
        f"{root}/error/budget_exceeded": float(budget_exceeded),
        f"{root}/action/format_valid_rate": _rate(step_rows, "format_valid"),
        f"{root}/action/missing_action_tag_rate": _rate(step_rows, "missing_action_tag"),
        f"{root}/action/missing_thinking_tag_rate": _rate(step_rows, "missing_thinking_tag"),
        f"{root}/action/admissible_member_rate": _rate(step_rows, "admissible_member"),
        f"{root}/action/chinese_rate": _rate(step_rows, "contains_chinese"),
        f"{root}/perf/episode_seconds_mean": _mean(episode_rows, "episode_seconds"),
        f"{root}/perf/generation_seconds_mean": _mean(step_rows, "generation_seconds"),
        f"{root}/perf/generation_request_seconds_mean": _mean(step_rows, "generation_request_seconds"),
        f"{root}/perf/env_step_seconds_mean": _mean(step_rows, "env_step_seconds"),
        f"{root}/perf/reset_seconds_mean": _mean(episode_rows, "reset_seconds"),
        f"{root}/perf/prefix_cache_hit_rate": cached_token_sum / prompt_token_sum if prompt_token_sum else 0.0,
        f"{root}/tokens/prompt_mean": _mean(step_rows, "prompt_tokens"),
        f"{root}/tokens/raw_prompt_mean": _mean(step_rows, "raw_prompt_tokens"),
        f"{root}/tokens/response_mean": _mean(step_rows, "response_tokens"),
        f"{root}/tokens/cached_mean": _mean(step_rows, "cached_tokens"),
        f"{root}/prompt/overlength_rate": _rate(step_rows, "prompt_overlength"),
        f"{root}/prompt/history_auto_truncated_rate": _rate(step_rows, "history_auto_truncated"),
    }


def summarize_diversity_samples(
    samples: list[Sample],
    *,
    metric_root: str,
    low_conf_nll_threshold: float = 5.0,
    max_turn_groups: int = 64,
) -> dict[str, float]:
    root = metric_root.rstrip("/")
    log_dict: dict[str, float] = {}
    log_dict.update(
        _sampled_logprob_metrics(
            samples,
            metric_root=root,
            low_conf_nll_threshold=low_conf_nll_threshold,
            max_turn_groups=max_turn_groups,
        )
    )
    log_dict.update(_actor_entropy_metrics(samples, metric_root=root, max_turn_groups=max_turn_groups))
    return log_dict


def summarize_actor_entropy_records(
    records: list[dict[str, Any]],
    *,
    metric_root: str,
    expected_count: int | None = None,
) -> dict[str, float]:
    root = metric_root.rstrip("/")
    summaries = []
    for record in records:
        entropy = record.get("actor_entropy")
        if not isinstance(entropy, dict) or not entropy.get("enabled"):
            continue
        token_count = int(entropy.get("token_count") or 0)
        token_mean = _optional_float(entropy.get("token_mean"))
        token_std = _optional_float(entropy.get("token_std"))
        if token_count <= 0 or token_mean is None:
            continue
        summaries.append((token_count, token_mean, token_std or 0.0))
    denominator = expected_count if expected_count is not None else len(records)
    metrics = {f"{root}/actor_entropy/coverage": len(summaries) / denominator if denominator else 0.0}
    if not summaries:
        return metrics
    total_count = sum(item[0] for item in summaries)
    token_mean = sum(count * mean_value for count, mean_value, _ in summaries) / total_count
    second_moment = (
        sum(count * (std_value * std_value + mean_value * mean_value) for count, mean_value, std_value in summaries)
        / total_count
    )
    metrics[f"{root}/actor_entropy/token_mean"] = token_mean
    metrics[f"{root}/actor_entropy/token_std"] = math.sqrt(max(second_moment - token_mean * token_mean, 0.0))
    return metrics


def low_conf_nll_threshold_from_args(args: Any, default: float = 5.0) -> float:
    raw = getattr(args, "agent_task_low_conf_nll_threshold", None)
    if raw is None:
        raw = os.environ.get("AGENT_TASK_LOW_CONF_NLL_THRESHOLD", default)
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AGENT_TASK_LOW_CONF_NLL_THRESHOLD must be a float, got {raw!r}") from exc


def summarize_alfworld_samples(
    samples: list[Sample],
    *,
    metric_root: str = "alfworld",
    max_episode_errors: int | None = None,
    max_episode_error_rate: float | None = None,
) -> dict[str, float]:
    return summarize_agent_task_samples(
        samples,
        metric_root=metric_root,
        max_episode_errors=max_episode_errors,
        max_episode_error_rate=max_episode_error_rate,
    )


def _sampled_logprob_metrics(
    samples: list[Sample],
    *,
    metric_root: str,
    low_conf_nll_threshold: float,
    max_turn_groups: int,
) -> dict[str, float]:
    per_step = []
    token_logprobs = []
    token_nll = []
    low_conf_count = 0
    token_count = 0
    response_sums = []
    response_means = []
    for sample in samples:
        values = _valid_logprobs(sample.rollout_log_probs)
        if not values:
            continue
        nll_values = [-value for value in values]
        low_conf_count += sum(1 for value in nll_values if value > low_conf_nll_threshold)
        token_count += len(values)
        token_logprobs.extend(values)
        token_nll.extend(nll_values)
        response_sums.append(sum(nll_values))
        response_means.append(sum(values) / len(values))
        per_step.append((sample, nll_values))

    if not token_logprobs:
        return {}

    root = metric_root.rstrip("/")
    metrics = {
        f"{root}/sampled_logprob/token_mean": _mean_values(token_logprobs),
        f"{root}/sampled_logprob/response_mean": _mean_values(response_means),
        f"{root}/sampled_nll/token_mean": _mean_values(token_nll),
        f"{root}/sampled_nll/token_std": _std_values(token_nll),
        f"{root}/sampled_nll/token_max": max(token_nll),
        f"{root}/sampled_nll/response_sum_mean": _mean_values(response_sums),
        f"{root}/sampled_nll/low_conf_token_rate": low_conf_count / token_count if token_count else 0.0,
    }
    metrics.update(_grouped_token_mean_metrics(per_step, root=f"{root}/sampled_nll", max_turn_groups=max_turn_groups))
    return metrics


def _actor_entropy_metrics(
    samples: list[Sample],
    *,
    metric_root: str,
    max_turn_groups: int,
) -> dict[str, float]:
    valid_response_steps = [sample for sample in samples if int(getattr(sample, "response_length", 0) or 0) > 0]
    summaries = []
    per_step = []
    for sample in valid_response_steps:
        entropy = (sample.metadata or {}).get("actor_entropy")
        if not isinstance(entropy, dict) or not entropy.get("enabled"):
            continue
        token_count = int(entropy.get("token_count") or 0)
        token_mean = _optional_float(entropy.get("token_mean"))
        token_std = _optional_float(entropy.get("token_std"))
        if token_count <= 0 or token_mean is None:
            continue
        summaries.append((token_count, token_mean, token_std or 0.0))
        per_step.append((sample, [token_mean] * token_count))

    if not summaries:
        return {}

    root = metric_root.rstrip("/")
    metrics = {
        f"{root}/actor_entropy/coverage": len(summaries) / len(valid_response_steps) if valid_response_steps else 0.0
    }
    if not summaries:
        return metrics

    total_count = sum(item[0] for item in summaries)
    token_mean = sum(count * mean_value for count, mean_value, _ in summaries) / total_count
    second_moment = (
        sum(count * (std_value * std_value + mean_value * mean_value) for count, mean_value, std_value in summaries)
        / total_count
    )
    metrics[f"{root}/actor_entropy/token_mean"] = token_mean
    metrics[f"{root}/actor_entropy/token_std"] = math.sqrt(max(second_moment - token_mean * token_mean, 0.0))
    metrics.update(
        _grouped_token_mean_metrics(per_step, root=f"{root}/actor_entropy", max_turn_groups=max_turn_groups)
    )
    return metrics


def _grouped_token_mean_metrics(
    per_step: list[tuple[Sample, list[float]]],
    *,
    root: str,
    max_turn_groups: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    groups: dict[str, list[float]] = {}
    for sample, values in per_step:
        metadata = sample.metadata or {}
        if metadata.get("success") is not None:
            success = bool(metadata.get("success"))
            groups.setdefault(
                "by_success/success_token_mean" if success else "by_success/failure_token_mean", []
            ).extend(values)
        if metadata.get("format_valid") is not None:
            format_valid = bool(metadata.get("format_valid"))
            groups.setdefault(
                "by_format_valid/valid_token_mean" if format_valid else "by_format_valid/invalid_token_mean",
                [],
            ).extend(values)
        if metadata.get("is_action_valid") is not None:
            valid = bool(metadata.get("is_action_valid"))
            groups.setdefault(
                "by_action_valid/valid_token_mean" if valid else "by_action_valid/invalid_token_mean",
                [],
            ).extend(values)
        outcome = _terminal_outcome(metadata)
        if outcome is not None:
            groups.setdefault(f"by_outcome/{outcome}_token_mean", []).extend(values)
        turn_idx = metadata.get("turn_idx")
        if turn_idx is not None:
            try:
                turn = int(turn_idx)
            except (TypeError, ValueError):
                turn = None
            if turn is not None and 0 <= turn < max_turn_groups:
                groups.setdefault(f"by_turn/{turn}_token_mean", []).extend(values)

    for name, values in groups.items():
        metrics[f"{root}/{name}"] = _mean_values(values)
    return metrics


def _terminal_outcome(metadata: dict[str, Any]) -> str | None:
    if metadata.get("episode_error"):
        return "error"
    if metadata.get("success"):
        return "success"
    if metadata.get("env_horizon_reached"):
        return "horizon"
    if metadata.get("success") is not None:
        return "failure"
    return None


def _valid_logprobs(values: list[float] | None) -> list[float]:
    if not values:
        return []
    out = []
    for value in values:
        number = _optional_float(value)
        if number is not None and math.isfinite(number):
            out.append(number)
    return out


def _mean_values(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std_values(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean_values(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return mean([float(bool(value)) for value in values]) if values else 0.0


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [_float(row.get(key)) for row in rows if row.get(key) is not None]
    return mean(values) if values else 0.0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _episode_rows(samples: list[Sample]) -> list[dict[str, Any]]:
    groups: dict[Any, list[tuple[int, dict[str, Any]]]] = {}
    for idx, sample in enumerate(samples):
        metadata = sample.metadata or {}
        key = metadata.get("traj_uid") or metadata.get("rollout_id") or sample.rollout_id or sample.index or idx
        groups.setdefault(key, []).append((idx, metadata))

    rows = []
    for items in groups.values():
        terminal_items = [item for item in items if item[1].get("is_terminal")]
        if terminal_items:
            rows.append(terminal_items[-1][1])
            continue
        rows.append(max(items, key=lambda item: (int(item[1].get("turn_idx", -1)), item[0]))[1])
    return rows


def _budget_exceeded(
    *,
    episode_error_count: float,
    episode_error_rate: float,
    max_episode_errors: int | None,
    max_episode_error_rate: float | None,
) -> bool:
    if max_episode_errors is None and max_episode_error_rate is None:
        return False
    max_errors = float(max(max_episode_errors or 0, 0))
    max_rate = float(max(max_episode_error_rate or 0.0, 0.0))
    return episode_error_count > max_errors or episode_error_rate > max_rate
