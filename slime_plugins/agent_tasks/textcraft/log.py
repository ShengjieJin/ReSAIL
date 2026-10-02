from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step
from slime_plugins.agent_tasks.common.logging import (
    flatten_samples,
    low_conf_nll_threshold_from_args,
    sample_log_record,
    summarize_agent_task_samples,
    summarize_diversity_samples,
    write_jsonl,
)
from slime_plugins.agent_tasks.common.frozen.audits import write_eval_result_audit
from slime_plugins.agent_tasks.common.trace import (
    drain_trace_records_for_samples,
    get_agent_trace_config,
    write_trace_sidecar,
)

from .config import get_textcraft_config

logger = logging.getLogger(__name__)


def log_rollout_samples(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    flat = flatten_samples(samples)
    config = get_textcraft_config(args)
    summary = summarize_agent_task_samples(
        flat,
        metric_root="textcraft",
        max_episode_errors=config.max_episode_errors,
        max_episode_error_rate=config.max_episode_error_rate,
    )
    diversity = summarize_diversity_samples(
        flat,
        metric_root="diversity/rollout/textcraft",
        low_conf_nll_threshold=low_conf_nll_threshold_from_args(args),
    )
    if rollout_extra_metrics is not None:
        for key, value in summary.items():
            rollout_extra_metrics[f"rollout/{key}"] = value
        rollout_extra_metrics.update(diversity)
    _raise_if_error_budget_exceeded(summary, metric_root="textcraft", rollout_id=rollout_id)
    _write_trace(args, config, "train", rollout_id, flat)
    _write_samples(
        config.sample_log_dir,
        f"rollout_{rollout_id}.jsonl",
        flat,
        rollout_id=rollout_id,
        evaluation=False,
        limit=config.sample_log_limit,
    )
    return False


def log_eval_samples(rollout_id, args, data, extra_metrics) -> bool:
    config = get_textcraft_config(args, evaluation=True)
    if extra_metrics is None:
        extra_metrics = {}
    samples_by_dataset = {}
    for dataset_name, dataset_data in data.items():
        flat = _eval_step_samples(dataset_data)
        samples_by_dataset[dataset_name] = flat
        summary = summarize_agent_task_samples(
            flat,
            metric_root=dataset_name,
            max_episode_errors=config.max_episode_errors,
            max_episode_error_rate=config.max_episode_error_rate,
        )
        for key, value in summary.items():
            extra_metrics[f"eval/{key}"] = value
        extra_metrics.update(
            summarize_diversity_samples(
                flat,
                metric_root=f"diversity/eval/{dataset_name}",
                low_conf_nll_threshold=low_conf_nll_threshold_from_args(args),
            )
        )
        _raise_if_error_budget_exceeded(summary, metric_root=dataset_name, rollout_id=rollout_id)
    write_eval_result_audit(
        args,
        rollout_id,
        samples_by_dataset,
        extra_metrics,
        audit_attr="textcraft_eval_result_audit_dir",
        identity_seed_attr="textcraft_eval_identity_seed",
        replicate_seeds_attr="textcraft_eval_replicate_seeds",
    )
    _write_trace(
        args,
        config,
        "eval",
        rollout_id,
        [sample for samples in samples_by_dataset.values() for sample in samples],
    )
    _write_samples(
        config.sample_log_dir,
        f"eval_{rollout_id}.jsonl",
        [sample for samples in samples_by_dataset.values() for sample in samples],
        rollout_id=rollout_id,
        evaluation=True,
        limit=config.sample_log_limit,
    )
    step = compute_rollout_step(args, rollout_id)
    extra_metrics["eval/step"] = step
    logger.info("eval %s: %s", rollout_id, extra_metrics)
    if getattr(args, "use_wandb", False) or getattr(args, "use_tensorboard", False):
        logging_utils.log(args, extra_metrics, step_key="eval/step")
    return True


def _write_samples(
    sample_log_dir: Path | None,
    name: str,
    samples: list[Any],
    *,
    rollout_id: int,
    evaluation: bool,
    limit: int,
) -> None:
    if sample_log_dir is None:
        return
    write_jsonl(
        sample_log_dir / name,
        (_textcraft_sample_log_record(sample, rollout_id=rollout_id, evaluation=evaluation) for sample in samples),
        limit=limit,
    )


def _eval_step_samples(dataset_data: dict[str, Any]) -> list[Any]:
    return dataset_data.get("step_samples") or dataset_data.get("samples", [])


def _write_trace(args: Any, config, phase: str, rollout_id: int, samples: list[Any]) -> None:
    trace_config = get_agent_trace_config(args, task="textcraft", sample_log_dir=config.sample_log_dir)
    records = drain_trace_records_for_samples("textcraft", phase, samples)
    write_trace_sidecar(
        trace_config,
        task="textcraft",
        phase=phase,
        outer_rollout_id=int(rollout_id),
        records=records,
    )


def _textcraft_sample_log_record(sample, *, rollout_id: int, evaluation: bool) -> dict[str, Any]:
    record = sample_log_record(sample, rollout_id=rollout_id, evaluation=evaluation)
    metadata = sample.metadata or {}
    record.update(
        {
            "task_id": metadata.get("task_id"),
            "data_idx": metadata.get("data_idx"),
            "goal": metadata.get("goal"),
            "goal_text": metadata.get("goal_text"),
            "goal_depth": metadata.get("goal_depth"),
            "action_kind": metadata.get("action_kind"),
            "action_failed": metadata.get("action_failed"),
            "commands_count": metadata.get("commands_count"),
            "inventory_size": metadata.get("inventory_size"),
            "turn_idx": metadata.get("turn_idx"),
            "sampling_seed": metadata.get("sampling_seed"),
            "eval_base_dataset_name": metadata.get("eval_base_dataset_name"),
            "eval_replicate_index": metadata.get("eval_replicate_index"),
            "eval_replicate_seed": metadata.get("eval_replicate_seed"),
        }
    )
    return record


def _raise_if_error_budget_exceeded(summary: dict[str, float], *, metric_root: str, rollout_id: int) -> None:
    budget_exceeded = bool(summary.get(f"{metric_root}/error/budget_exceeded", 0.0))
    if not budget_exceeded:
        return
    error_count = summary.get(f"{metric_root}/error/episode_count", 0.0)
    error_rate = summary.get(f"{metric_root}/error/episode_rate", 0.0)
    raise RuntimeError(
        f"TextCraft episode error budget exceeded: rollout_id={rollout_id} "
        f"errors={error_count:g} rate={error_rate:.4f}"
    )
