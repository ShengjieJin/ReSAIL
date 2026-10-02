from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import Any

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step
from slime_plugins.agent_tasks.common.config import as_bool, get_arg_or_env
from slime_plugins.agent_tasks.common.logging import (
    flatten_samples,
    low_conf_nll_threshold_from_args,
    sample_log_record,
    summarize_agent_task_samples,
    summarize_diversity_samples,
    write_jsonl,
)
from slime_plugins.agent_tasks.common.trace import (
    drain_trace_records_for_samples,
    get_agent_trace_config,
    keep_prompt_metadata_enabled,
    write_trace_sidecar,
)

from .config import get_alfworld_config
from .frozen.audits import student_input_privilege_violation as _student_prompt_privilege_violation
from .frozen.audits import summarize_action_match as _summarize_action_match
from .frozen.audits import write_action_match_audit as _write_action_match_audit
from .frozen.audits import write_eval_result_audit as _write_eval_result_audit
from .frozen.audits import write_sglang_request_audit as _write_sglang_request_audit

logger = logging.getLogger(__name__)


def log_rollout_samples(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    flat = flatten_samples(samples)
    config = get_alfworld_config(args)
    summary = summarize_agent_task_samples(
        flat,
        metric_root="alfworld",
        max_episode_errors=config.max_episode_errors,
        max_episode_error_rate=config.max_episode_error_rate,
    )
    action_metrics = _summarize_action_match(
        flat,
        outcome_split=bool(getattr(args, "alfworld_action_match_outcome_split", False)),
    )
    summary.update(action_metrics["rates"])
    if action_metrics["counts"]:
        _write_action_match_audit(args, rollout_id, action_metrics["counts"])
    _write_sglang_request_audit(args, rollout_id, rollout_extra_metrics)
    if rollout_extra_metrics is not None:
        for key, value in summary.items():
            rollout_extra_metrics[f"rollout/{key}"] = value
        if _diversity_metrics_enabled(args):
            rollout_extra_metrics.update(
                summarize_diversity_samples(
                    flat,
                    metric_root="diversity/rollout/alfworld",
                    low_conf_nll_threshold=low_conf_nll_threshold_from_args(args),
                )
            )
    _raise_if_error_budget_exceeded(summary, metric_root="alfworld", rollout_id=rollout_id)
    _write_trace(args, config, "train", rollout_id, flat)
    _write_samples(
        config.sample_log_dir,
        f"rollout_{rollout_id}.jsonl",
        flat,
        rollout_id=rollout_id,
        evaluation=False,
        limit=config.sample_log_limit,
    )
    _release_logged_observations(args, flat)
    return False


def log_eval_samples(rollout_id, args, data, extra_metrics) -> bool:
    config = get_alfworld_config(args, evaluation=True)
    if extra_metrics is None:
        extra_metrics = {}
    samples_by_dataset = {}
    replicated_success_rates: dict[str, list[tuple[int, float]]] = {}
    for dataset_name, dataset_data in data.items():
        flat = dataset_data.get("step_samples") or dataset_data.get("samples", [])
        samples_by_dataset[dataset_name] = flat
        base_dataset_name, replicate_idx = _parse_replicate_dataset_name(dataset_name)
        metric_prefix, metric_root = _eval_metric_namespace(base_dataset_name, config)
        summary = summarize_agent_task_samples(
            flat,
            metric_root=metric_root,
            max_episode_errors=config.max_episode_errors,
            max_episode_error_rate=config.max_episode_error_rate,
        )
        for key, value in summary.items():
            output_prefix = metric_prefix if replicate_idx is None else f"{metric_prefix}_replicate_{replicate_idx}"
            extra_metrics[f"{output_prefix}/{key}"] = value
        success_key = f"{metric_root}/episode/success_rate"
        if replicate_idx is not None:
            replicated_success_rates.setdefault(base_dataset_name, []).append(
                (replicate_idx, float(summary[success_key]))
            )
        if _diversity_metrics_enabled(args):
            extra_metrics.update(
                summarize_diversity_samples(
                    flat,
                    metric_root=f"diversity/eval/{dataset_name}",
                    low_conf_nll_threshold=low_conf_nll_threshold_from_args(args),
                )
            )
        _raise_if_error_budget_exceeded(summary, metric_root=metric_root, rollout_id=rollout_id)
    for base_dataset_name, indexed_values in replicated_success_rates.items():
        replicate_indices = [index for index, _ in indexed_values]
        if len(set(replicate_indices)) != len(replicate_indices):
            raise ValueError(f"replicated evaluation contains duplicate indices for {base_dataset_name}")
        values = [value for _, value in sorted(indexed_values)]
        if len(values) < 2:
            raise ValueError("replicated evaluation requires at least two results per dataset")
        metric_prefix, metric_root = _eval_metric_namespace(base_dataset_name, config)
        base_key = f"{metric_prefix}/{metric_root}/episode/success_rate"
        extra_metrics[f"{base_key}_mean"] = statistics.fmean(values)
        extra_metrics[f"{base_key}_sample_std"] = statistics.stdev(values)
        extra_metrics[f"{base_key}_replicate_count"] = float(len(values))
    extra_metrics.update(_paired_pass_at_k_metrics(samples_by_dataset, config))
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
        _balanced_eval_sample_log_subset(
            samples_by_dataset,
            None if bool(getattr(args, "alfworld_eval_log_all_episodes", False)) else config.sample_log_limit,
        ),
        rollout_id=rollout_id,
        evaluation=True,
        limit=None,
    )
    step = compute_rollout_step(args, rollout_id)
    extra_metrics["eval/step"] = step
    _write_eval_result_audit(args, rollout_id, samples_by_dataset, extra_metrics)
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
        (_alfworld_sample_log_record(sample, rollout_id=rollout_id, evaluation=evaluation) for sample in samples),
        limit=limit,
    )


def _cache_relative_eval_gamefile(metadata: dict[str, Any]) -> str | None:
    identity = metadata.get("runtime_task_identity")
    gamefile = identity.get("gamefile") if isinstance(identity, dict) else None
    if not isinstance(gamefile, str) or not gamefile:
        return None
    parts = Path(gamefile).parts
    if "json_2.1.1" not in parts:
        return None
    relative = parts[parts.index("json_2.1.1") :]
    if (len(relative) != 5 or relative[1] not in {"valid_seen", "valid_unseen"}
            or relative[-1] != "game.tw-pddl" or ".." in relative):
        return None
    return "/".join(relative)


def _alfworld_sample_log_record(sample: Any, *, rollout_id: int, evaluation: bool) -> dict[str, Any]:
    record = sample_log_record(sample, rollout_id=rollout_id, evaluation=evaluation)
    metadata = sample.metadata or {}
    if evaluation:
        record["alfworld_gamefile"] = _cache_relative_eval_gamefile(metadata)
    if evaluation and metadata.get("eval_replicate_index") is not None:
        record.update(
            {
                "eval_base_dataset_name": metadata.get("eval_base_dataset_name"),
                "eval_replicate_index": metadata.get("eval_replicate_index"),
                "eval_replicate_seed": metadata.get("eval_replicate_seed"),
            }
        )
    return record


def _diversity_metrics_enabled(args: Any) -> bool:
    return as_bool(
        get_arg_or_env(
            args,
            "agent_task_diversity_metrics_enabled",
            "AGENT_TASK_DIVERSITY_METRICS_ENABLED",
            False,
        )
    )


def _write_trace(args: Any, config, phase: str, rollout_id: int, samples: list[Any]) -> None:
    trace_config = get_agent_trace_config(args, task="alfworld", sample_log_dir=config.sample_log_dir)
    records = drain_trace_records_for_samples("alfworld", phase, samples)
    write_trace_sidecar(
        trace_config,
        task="alfworld",
        phase=phase,
        outer_rollout_id=int(rollout_id),
        records=records,
    )


def _release_logged_observations(args: Any, samples: list[Any]) -> None:
    """Drop bulky prompt-derived observations once trace/sample artifacts exist."""

    if keep_prompt_metadata_enabled(args):
        return
    for sample in samples:
        metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
        metadata.pop("current_observation", None)
        metadata.pop("next_observation", None)


def _eval_metric_namespace(dataset_name: str, config) -> tuple[str, str]:
    if dataset_name == config.eval_out_of_distribution_dataset_name:
        return dataset_name, config.eval_dataset_name
    return "eval", dataset_name


def _parse_replicate_dataset_name(dataset_name: str) -> tuple[str, int | None]:
    marker = "__replicate_"
    if marker not in dataset_name:
        return dataset_name, None
    base, suffix = dataset_name.rsplit(marker, 1)
    replicate, separator, seed = suffix.partition("_seed_")
    if not base or not separator or not replicate.isdigit() or not seed.lstrip("-").isdigit():
        raise ValueError(f"invalid replicated eval dataset name: {dataset_name}")
    return base, int(replicate)


def _paired_pass_at_k_metrics(samples_by_dataset: dict[str, list[Any]], config) -> dict[str, float]:
    grouped: dict[str, dict[int, dict[str, bool]]] = {}
    for dataset_name, samples in samples_by_dataset.items():
        base, replicate_idx = _parse_replicate_dataset_name(dataset_name)
        if replicate_idx is None:
            continue
        terminal = {}
        for sample in samples:
            metadata = sample.metadata or {}
            if metadata.get("is_terminal") is not True:
                continue
            uid = str(metadata.get("uid") or "")
            if not uid or uid in terminal:
                raise ValueError(f"replicate {dataset_name} has missing or duplicate terminal uid")
            success = metadata.get("success")
            terminal[uid] = bool(float(sample.reward or 0.0) > 0.0 if success is None else success)
        grouped.setdefault(base, {})[replicate_idx] = terminal
    metrics = {}
    for base, replicates in grouped.items():
        if sorted(replicates) != [0, 1, 2]:
            raise ValueError(f"paired pass@3 requires replicate indices 0,1,2 for {base}")
        uid_sets = [set(replicates[index]) for index in range(3)]
        if not uid_sets[0] or uid_sets[1:] != [uid_sets[0], uid_sets[0]]:
            raise ValueError(f"paired pass@3 requires identical non-empty task identities for {base}")
        passed = sum(any(replicates[index][uid] for index in range(3)) for uid in uid_sets[0])
        metric_prefix, metric_root = _eval_metric_namespace(base, config)
        metrics[f"{metric_prefix}/{metric_root}/episode/pass_at_3"] = passed / len(uid_sets[0])
    return metrics


def _balanced_eval_sample_log_subset(samples_by_dataset: dict[str, list[Any]], limit: int | None) -> list[Any]:
    populated = [(dataset_name, samples) for dataset_name, samples in samples_by_dataset.items() if samples]
    if limit is None:
        return [sample for _, samples in populated for sample in samples]
    if limit <= 0 or not populated:
        return []

    quota, remainder = divmod(limit, len(populated))
    selected = []
    overflow = []
    for dataset_idx, (_, samples) in enumerate(populated):
        take = quota + int(dataset_idx < remainder)
        selected.extend(samples[:take])
        overflow.extend(samples[take:])

    if len(selected) < limit:
        selected.extend(overflow[: limit - len(selected)])
    return selected[:limit]


def _raise_if_error_budget_exceeded(summary: dict[str, float], *, metric_root: str, rollout_id: int) -> None:
    budget_exceeded = bool(summary.get(f"{metric_root}/error/budget_exceeded", 0.0))
    if not budget_exceeded:
        return
    error_count = summary.get(f"{metric_root}/error/episode_count", 0.0)
    error_rate = summary.get(f"{metric_root}/error/episode_rate", 0.0)
    raise RuntimeError(
        f"ALFWorld episode error budget exceeded: rollout_id={rollout_id} "
        f"errors={error_count:g} rate={error_rate:.4f}"
    )
