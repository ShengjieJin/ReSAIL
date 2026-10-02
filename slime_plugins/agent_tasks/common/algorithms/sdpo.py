from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import time
import urllib.request
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.rewards import post_process_grouped_rewards

from .sdpo_context import (
    _ROW_PRECOMPUTED_GUIDANCE_OUTPUT_KEY,
    _guidance_output_has_schema,
    clean_guidance_summary,
    collect_guidance_summary_prompt_payloads,
    collect_guidance_summary_prompts,
    compute_sdpo_context_metrics,
    finalize_sdpo_context_plan,
    prepare_sdpo_context_plan,
)

logger = logging.getLogger(__name__)

_PRECOMPUTED_GUIDANCE_OUTPUTS_KEY = "sdpo_guidance_summary_outputs"
_PRECOMPUTED_GUIDANCE_STATS_KEY = "sdpo_guidance_summary_stats"
_HEAVY_SDPO_METADATA_KEYS = {"sdpo_guidance_image_data", "sdpo_privileged_trajectory"}
_NO_SUCCESS_FILTER_REASON = "no_success_trajectory"
_ALL_SUCCESS_FILTER_REASON = "all_success_trajectory"
_SDPO_FILTER_REASONS = {_NO_SUCCESS_FILTER_REASON, _ALL_SUCCESS_FILTER_REASON}


@dataclass
class GuidanceSummaryStats:
    prompt_count: int = 0
    prompt_overlength_count: int = 0
    prompt_truncated_count: int = 0
    retry_count: int = 0
    failure_count: int = 0
    prompt_tokens_total: int = 0
    prompt_tokens_max: int = 0
    image_count_total: int = 0
    image_count_max: int = 0

    def observe_prompt(self, token_count: int, *, image_count: int = 0) -> None:
        self.prompt_count += 1
        self.prompt_tokens_total += int(token_count)
        self.prompt_tokens_max = max(self.prompt_tokens_max, int(token_count))
        self.image_count_total += int(image_count)
        self.image_count_max = max(self.image_count_max, int(image_count))

    def merge(self, other: "GuidanceSummaryStats") -> None:
        self.prompt_count += other.prompt_count
        self.prompt_overlength_count += other.prompt_overlength_count
        self.prompt_truncated_count += other.prompt_truncated_count
        self.retry_count += other.retry_count
        self.failure_count += other.failure_count
        self.prompt_tokens_total += other.prompt_tokens_total
        self.prompt_tokens_max = max(self.prompt_tokens_max, other.prompt_tokens_max)
        self.image_count_total += other.image_count_total
        self.image_count_max = max(self.image_count_max, other.image_count_max)

    def metrics(self) -> dict[str, float]:
        prompt_count = max(self.prompt_count, 0)
        return {
            "self_distillation/guidance_summary_prompt_count": float(prompt_count),
            "self_distillation/guidance_summary_prompt_overlength_count": float(self.prompt_overlength_count),
            "self_distillation/guidance_summary_prompt_overlength_rate": (
                self.prompt_overlength_count / prompt_count if prompt_count else 0.0
            ),
            "self_distillation/guidance_summary_prompt_truncated_count": float(self.prompt_truncated_count),
            "self_distillation/guidance_summary_retry_count": float(self.retry_count),
            "self_distillation/guidance_summary_failure_count": float(self.failure_count),
            "self_distillation/guidance_summary_prompt_tokens_mean": (
                self.prompt_tokens_total / prompt_count if prompt_count else 0.0
            ),
            "self_distillation/guidance_summary_prompt_tokens_max": float(self.prompt_tokens_max),
            "self_distillation/guidance_summary_image_count_mean": (
                self.image_count_total / prompt_count if prompt_count else 0.0
            ),
            "self_distillation/guidance_summary_image_count_max": float(self.image_count_max),
        }

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_count": self.prompt_count,
            "prompt_overlength_count": self.prompt_overlength_count,
            "prompt_truncated_count": self.prompt_truncated_count,
            "retry_count": self.retry_count,
            "failure_count": self.failure_count,
            "prompt_tokens_total": self.prompt_tokens_total,
            "prompt_tokens_max": self.prompt_tokens_max,
            "image_count_total": self.image_count_total,
            "image_count_max": self.image_count_max,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GuidanceSummaryStats":
        stats = cls()
        for key in stats.as_dict():
            setattr(stats, key, int(raw.get(key, 0) or 0))
        return stats


@dataclass
class FilterCompactionResult:
    samples: list[Sample]
    plan: list[dict[str, Any]]
    full_sample_count: int
    filtered_sample_count: int
    compacted_sample_count: int
    zero_loss_placeholder_count: int
    full_token_count: int
    filtered_token_count: int
    compacted_token_count: int
    active_token_count: int
    schedule_alignment_placeholder_count: int = 0
    schedule_alignment_shortfall_count: int = 0
    empty_batch_fallback: bool = False


def convert_samples_to_train_data(args: Any, samples: list[Sample] | list[list[Sample]]) -> dict[str, Any]:
    flat_samples = _flatten_samples(samples)
    if not flat_samples:
        raise ValueError("SDPO convert received no samples.")

    _ensure_default_sdpo_args(args)
    plan = prepare_sdpo_context_plan(args, flat_samples)
    if (
        str(getattr(args, "sdpo_teacher_context_mode", "original")) == "own_outcome"
        and str(getattr(args, "sdpo_solution_context_format", "guidance_plan")) == "guidance_plan"
        and str(getattr(args, "sdpo_guidance_summary_source", "success_priority")) == "self_trajectory"
    ):
        plan = _bind_precomputed_self_trajectory_outputs(flat_samples, plan)
    guidance_prompts = collect_guidance_summary_prompts(plan)
    guidance_prompt_payloads = collect_guidance_summary_prompt_payloads(plan)
    guidance_outputs = _collect_precomputed_guidance_outputs(flat_samples, guidance_prompts)
    missing_guidance_prompts = _missing_guidance_prompts(plan, guidance_outputs)
    guidance_outputs.update(
        _generate_guidance_summaries(args, missing_guidance_prompts, prompt_payloads=guidance_prompt_payloads)
    )
    guidance_summary_stats = _collect_guidance_summary_stats(flat_samples)
    if missing_guidance_prompts:
        guidance_summary_stats.merge(_last_guidance_summary_stats(args))
    finalized_plan = finalize_sdpo_context_plan(plan, guidance_outputs)
    guidance_debug_logged_count = _write_guidance_debug_records(
        args, flat_samples, plan, finalized_plan, guidance_outputs, prompt_payloads=guidance_prompt_payloads
    )
    guidance_output_metrics = _guidance_output_metrics(
        flat_samples,
        plan,
        guidance_outputs,
        debug_logged_count=guidance_debug_logged_count,
    )
    plan = finalized_plan
    _deactivate_removed_samples(flat_samples, plan)
    metrics_plan = plan
    compaction = _compact_sdpo_filtered_rows(args, flat_samples, plan)
    flat_samples = compaction.samples
    plan = compaction.plan
    _ensure_reward_group_metadata(flat_samples, plan)
    raw_rewards, rewards = post_process_grouped_rewards(args, flat_samples, group_key="uid", trajectory_key="traj_uid")
    train_data = _build_train_data(flat_samples, raw_rewards=raw_rewards, rewards=rewards)
    for key in (
        "sdpo_metadata",
        "sdpo_teacher_prompt_text",
        "sdpo_teacher_prompt_texts",
        "sdpo_teacher_messages",
        "sdpo_teacher_messages_list",
        "sdpo_teacher_signal_type",
        "self_distillation_mask",
        "sdpo_loss_weights",
        "sdpo_representation_success_aggregation",
        "sdpo_representation_success_count",
        "sdpo_selected_success_traj_uid",
        "sdpo_token_weight_contrast_prompt_text",
        "sdpo_token_weight_contrast_messages",
        "sdpo_token_weight_contrast_source",
        "sdpo_token_weight_contrast_traj_uid",
    ):
        if key == "sdpo_metadata":
            train_data[key] = [_sanitize_sdpo_metadata(row[key]) for row in plan]
        elif key == "sdpo_selected_success_traj_uid":
            train_data[key] = [row.get(key) for row in plan]
        else:
            train_data[key] = [row[key] for row in plan]
    _add_sgs_fields(train_data, plan)
    metrics = compute_sdpo_context_metrics(
        metrics_plan,
        success_reward_threshold=float(getattr(args, "sdpo_success_reward_threshold", 1.0)),
    )
    metrics.update(_filter_compaction_metrics(compaction))
    metrics.update(guidance_summary_stats.metrics())
    metrics.update(guidance_output_metrics)
    train_data.update({key: [float(value)] * len(flat_samples) for key, value in metrics.items()})
    return train_data


def _add_sgs_fields(train_data: dict[str, Any], plan: list[dict[str, Any]]) -> None:
    """Promote plain teacher context and action-alignment fields for SGS scoring."""
    metadata_rows = [row["sdpo_metadata"] for row in plan]
    if not any("sgs_action_token_mask" in row for row in metadata_rows):
        return
    required = (
        "sdpo_current_prompt_text",
        "sdpo_current_raw_prompt",
        "sgs_action_token_mask",
        "sgs_action_alignment_valid",
        "sgs_action_alignment_reason",
        "sgs_action_match",
        "sgs_online_action",
        "sgs_online_format_valid",
        "sgs_frozen_action",
        "sgs_source_draw_id",
        "sgs_task_id",
        "sgs_split",
    )
    missing = [
        (row_index, key)
        for row_index, row in enumerate(metadata_rows)
        for key in required
        if key not in row
    ]
    if missing:
        raise ValueError(f"SGS scoring rows are missing metadata: {missing[:5]}.")
    train_data["sgs_plain_prompt_text"] = [row["sdpo_current_prompt_text"] for row in metadata_rows]
    train_data["sgs_plain_messages"] = [row["sdpo_current_raw_prompt"] for row in metadata_rows]
    for key in required[2:]:
        train_data[key] = [row[key] for row in metadata_rows]


def _flatten_samples(samples: list[Sample] | list[list[Sample]]) -> list[Sample]:
    if not samples:
        return []
    if isinstance(samples[0], Sample):
        return list(samples)  # type: ignore[arg-type]
    flat: list[Sample] = []
    for group in samples:
        flat.extend(_flatten_samples(group))  # type: ignore[arg-type]
    return flat


def _compact_sdpo_filtered_rows(
    args: Any,
    samples: Sequence[Sample],
    plan: list[dict[str, Any]],
) -> FilterCompactionResult:
    full_sample_count = len(samples)
    full_token_count = sum(_sample_token_count(sample) for sample in samples)
    filtered_positions = [
        idx
        for idx, row in enumerate(plan)
        if str(row.get("sdpo_filter_reason") or "") in _SDPO_FILTER_REASONS
    ]
    active_token_count = sum(
        _sample_token_count(sample)
        for sample, row in zip(samples, plan, strict=True)
        if float(row.get("self_distillation_mask", 0.0)) > 0.0
        and str(row.get("sdpo_teacher_signal_type") or "") != "none"
    )
    if not filtered_positions:
        return FilterCompactionResult(
            samples=list(samples),
            plan=list(plan),
            full_sample_count=full_sample_count,
            filtered_sample_count=0,
            compacted_sample_count=0,
            zero_loss_placeholder_count=0,
            full_token_count=full_token_count,
            filtered_token_count=0,
            compacted_token_count=0,
            active_token_count=active_token_count,
        )

    filtered_set = set(filtered_positions)
    keep_positions = {idx for idx in range(full_sample_count) if idx not in filtered_set}
    placeholder_positions = _select_zero_loss_placeholders(samples, filtered_positions)
    keep_positions.update(placeholder_positions)
    alignment_positions, alignment_shortfall = _select_schedule_alignment_placeholders(
        args,
        samples,
        filtered_positions=filtered_positions,
        keep_positions=keep_positions,
    )
    placeholder_positions.update(alignment_positions)
    keep_positions.update(alignment_positions)
    for idx in placeholder_positions:
        sample = samples[idx]
        sample.remove_sample = True
        row = plan[idx]
        row["sdpo_teacher_signal_type"] = "none"
        row["self_distillation_mask"] = 0.0
        row["sdpo_loss_weights"] = 0.0
        row["sdpo_zero_loss_placeholder"] = True
        metadata = row.get("sdpo_metadata")
        if isinstance(metadata, dict):
            metadata["algorithm_active_mask"] = False
    empty_batch_fallback = False
    if not keep_positions:
        empty_batch_fallback = True
        keep_positions = set(range(full_sample_count))
        for idx, sample in enumerate(samples):
            sample.remove_sample = True
            row = plan[idx]
            row["sdpo_teacher_signal_type"] = "none"
            row["self_distillation_mask"] = 0.0
            row["sdpo_loss_weights"] = 0.0
            row["sdpo_zero_loss_placeholder"] = True

    kept = sorted(keep_positions)
    compacted_positions = [idx for idx in range(full_sample_count) if idx not in keep_positions]
    return FilterCompactionResult(
        samples=[samples[idx] for idx in kept],
        plan=[plan[idx] for idx in kept],
        full_sample_count=full_sample_count,
        filtered_sample_count=len(filtered_positions),
        compacted_sample_count=len(compacted_positions),
        zero_loss_placeholder_count=len(placeholder_positions),
        full_token_count=full_token_count,
        filtered_token_count=sum(_sample_token_count(samples[idx]) for idx in filtered_positions),
        compacted_token_count=sum(_sample_token_count(samples[idx]) for idx in compacted_positions),
        active_token_count=active_token_count,
        schedule_alignment_placeholder_count=len(alignment_positions),
        schedule_alignment_shortfall_count=alignment_shortfall,
        empty_batch_fallback=empty_batch_fallback,
    )


def _select_zero_loss_placeholders(samples: Sequence[Sample], filtered_positions: list[int]) -> set[int]:
    by_rollout: dict[int, int] = {}
    for idx in filtered_positions:
        rollout_id = _sample_rollout_id(samples[idx], idx)
        current = by_rollout.get(rollout_id)
        if current is None or idx < current:
            by_rollout[rollout_id] = idx
    return set(by_rollout.values())


def _select_schedule_alignment_placeholders(
    args: Any,
    samples: Sequence[Sample],
    *,
    filtered_positions: list[int],
    keep_positions: set[int],
) -> tuple[set[int], int]:
    alignment_unit = _schedule_alignment_unit(args)
    global_batch_size = int(getattr(args, "global_batch_size", 0) or 0)
    if alignment_unit <= 1 or global_batch_size <= 0:
        return set(), 0

    kept_rollout_ids: list[int] = []
    seen_rollout_ids: set[int] = set()
    for idx in sorted(keep_positions):
        rollout_id = _sample_rollout_id(samples[idx], idx)
        if rollout_id in seen_rollout_ids:
            continue
        seen_rollout_ids.add(rollout_id)
        kept_rollout_ids.append(rollout_id)

    extra_positions: set[int] = set()
    shortfall = 0
    num_steps = len(kept_rollout_ids) // global_batch_size
    for step_i in range(num_steps):
        step_rollouts = set(kept_rollout_ids[step_i * global_batch_size : (step_i + 1) * global_batch_size])
        step_kept = [
            idx
            for idx in sorted(keep_positions | extra_positions)
            if _sample_rollout_id(samples[idx], idx) in step_rollouts
        ]
        if not step_kept:
            continue
        target_count = max(
            alignment_unit,
            ((len(step_kept) + alignment_unit - 1) // alignment_unit) * alignment_unit,
        )
        need = target_count - len(step_kept)
        if need <= 0:
            continue
        candidates = [
            idx
            for idx in filtered_positions
            if idx not in keep_positions
            and idx not in extra_positions
            and _sample_rollout_id(samples[idx], idx) in step_rollouts
        ]
        candidates.sort(key=lambda idx: _placeholder_sort_key(samples[idx], idx))
        selected = candidates[:need]
        extra_positions.update(selected)
        shortfall += max(need - len(selected), 0)
    return extra_positions, shortfall


def _schedule_alignment_unit(args: Any) -> int:
    train_parallel_config = getattr(args, "train_parallel_config", None)
    if not isinstance(train_parallel_config, Mapping):
        return 1
    dp_size = max(int(train_parallel_config.get("dp_size", 1) or 1), 1)
    vpp_size = max(int(train_parallel_config.get("vpp_size", 1) or 1), 1)
    mb_group = max(int(train_parallel_config.get("microbatch_group_size_per_vp_stage", 1) or 1), 1)
    align_to = dp_size * (mb_group if vpp_size > 1 else 1)
    if bool(getattr(args, "use_dynamic_batch_size", False)):
        return align_to
    micro_batch_size = max(int(getattr(args, "micro_batch_size", 1) or 1), 1)
    return align_to * micro_batch_size


def _sample_rollout_id(sample: Sample, idx: int) -> int:
    rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
    return int(rollout_id if rollout_id is not None else idx)


def _placeholder_sort_key(sample: Sample, idx: int) -> tuple[int, int, int]:
    return (_sample_token_count(sample), int(sample.response_length), idx)


def _sample_token_count(sample: Sample) -> int:
    return len(sample.tokens or [])


def _filter_compaction_metrics(compaction: FilterCompactionResult) -> dict[str, float]:
    full_count = max(float(compaction.full_sample_count), 1.0)
    full_tokens = max(float(compaction.full_token_count), 1.0)
    return {
        "self_distillation/full_sample_count": float(compaction.full_sample_count),
        "self_distillation/train_sample_count": float(len(compaction.samples)),
        "self_distillation/filtered_sample_fraction": compaction.filtered_sample_count / full_count,
        "self_distillation/filtered_token_fraction": compaction.filtered_token_count / full_tokens,
        "self_distillation/compacted_sample_fraction": compaction.compacted_sample_count / full_count,
        "self_distillation/compacted_token_fraction": compaction.compacted_token_count / full_tokens,
        "self_distillation/zero_loss_placeholder_fraction": compaction.zero_loss_placeholder_count / full_count,
        "self_distillation/schedule_alignment_placeholder_fraction": (
            compaction.schedule_alignment_placeholder_count / full_count
        ),
        "self_distillation/schedule_alignment_shortfall_count": float(compaction.schedule_alignment_shortfall_count),
        "self_distillation/active_token_fraction": compaction.active_token_count / full_tokens,
        "self_distillation/compaction_empty_batch_fallback": 1.0 if compaction.empty_batch_fallback else 0.0,
    }


def _ensure_reward_group_metadata(samples: Sequence[Sample], plan: list[dict[str, Any]]) -> None:
    for sample, row in zip(samples, plan, strict=True):
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sdpo_metadata = row["sdpo_metadata"]
        sample.metadata.setdefault("uid", sdpo_metadata["uid"])
        sample.metadata.setdefault("traj_uid", sdpo_metadata["traj_uid"])


def _build_train_data(samples: Sequence[Sample], *, raw_rewards: list[float], rewards: list[float]) -> dict[str, Any]:
    rollout_ids = [sample.rollout_id if sample.rollout_id is not None else sample.index for sample in samples]
    loss_masks = []
    for sample in samples:
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        if len(sample.loss_mask) != sample.response_length:
            raise ValueError(f"loss_mask length {len(sample.loss_mask)} != response_length {sample.response_length}")
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        loss_masks.append(list(sample.loss_mask))

    mask_sums_per_sample = [sum(mask) for mask in loss_masks]
    rollout_total_mask: dict[int, int] = {}
    for rollout_id, mask_sum in zip(rollout_ids, mask_sums_per_sample, strict=True):
        rollout_id = int(rollout_id)
        rollout_total_mask[rollout_id] = rollout_total_mask.get(rollout_id, 0) + mask_sum

    train_data: dict[str, Any] = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
        "rollout_ids": [int(rollout_id) for rollout_id in rollout_ids],
        "loss_masks": loss_masks,
        "rollout_mask_sums": [rollout_total_mask[int(rollout_id)] for rollout_id in rollout_ids],
    }
    if any(sample.train_metadata is not None for sample in samples):
        train_data["metadata"] = [_sanitize_train_metadata(sample.train_metadata) for sample in samples]
    if any(getattr(sample, "multimodal_train_inputs", None) is not None for sample in samples):
        train_data["multimodal_train_inputs"] = [
            getattr(sample, "multimodal_train_inputs", None) for sample in samples
        ]
    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]
    return train_data


def _deactivate_removed_samples(samples: Sequence[Sample], plan: list[dict[str, Any]]) -> None:
    for sample, row in zip(samples, plan, strict=True):
        if not getattr(sample, "remove_sample", False):
            continue
        metadata = row.get("sdpo_metadata")
        if isinstance(metadata, dict):
            metadata["algorithm_active_mask"] = False
        row["sdpo_teacher_signal_type"] = "none"
        row["self_distillation_mask"] = 0.0
        row["sdpo_loss_weights"] = 0.0


def _ensure_default_sdpo_args(args: Any) -> None:
    if not hasattr(args, "agent_task_sdpo_metadata_profile"):
        args.agent_task_sdpo_metadata_profile = "agent"
    if not hasattr(args, "sdpo_guidance_summary_source"):
        args.sdpo_guidance_summary_source = "success_priority"
    if not hasattr(args, "sdpo_no_success_context_mode"):
        args.sdpo_no_success_context_mode = "feedback"
    if not hasattr(args, "sdpo_representation_success_aggregation"):
        args.sdpo_representation_success_aggregation = "sample"
    if not hasattr(args, "sdpo_filter_all_success_groups"):
        args.sdpo_filter_all_success_groups = False


def _sanitize_train_metadata(metadata: Any) -> Any:
    if not isinstance(metadata, Mapping):
        return metadata
    sanitized = dict(metadata)
    sdpo = sanitized.get("sdpo")
    if isinstance(sdpo, Mapping):
        sanitized["sdpo"] = _sanitize_sdpo_metadata(sdpo)
    return sanitized


def _sanitize_sdpo_metadata(metadata: Any) -> Any:
    if not isinstance(metadata, Mapping):
        return metadata
    sanitized = {}
    for key, value in metadata.items():
        if key in _HEAVY_SDPO_METADATA_KEYS:
            continue
        sanitized[key] = _redact_embedded_image_data(value)
    return sanitized


def _redact_embedded_image_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted = {}
        for key, item in value.items():
            if key == "image" and isinstance(item, str) and item.startswith("data:image"):
                redacted[key] = f"redacted_image_sha256:{hashlib.sha256(item.encode('utf-8')).hexdigest()[:16]}"
            else:
                redacted[key] = _redact_embedded_image_data(item)
        return redacted
    if isinstance(value, list):
        return [_redact_embedded_image_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_embedded_image_data(item) for item in value)
    if isinstance(value, str) and value.startswith("data:image"):
        return f"redacted_image_sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"
    return value


def _generate_guidance_summaries(
    args: Any,
    prompts: list[str],
    *,
    prompt_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    if not prompts:
        setattr(args, "_sdpo_guidance_summary_last_stats", GuidanceSummaryStats())
        return {}

    injected_generator = getattr(args, "sdpo_guidance_summary_generator", None)
    if callable(injected_generator):
        stats = GuidanceSummaryStats()
        tokenizer = _load_task_tokenizer(args)
        for prompt in prompts:
            payload = _guidance_prompt_payload(prompt, prompt_payloads)
            rendered = _render_chat_prompt(args, tokenizer, prompt, messages=payload.get("messages"))
            stats.observe_prompt(_token_count(tokenizer, rendered), image_count=len(payload.get("image_data") or []))
        setattr(args, "_sdpo_guidance_summary_last_stats", stats)
        outputs = injected_generator(prompts)
        if isinstance(outputs, dict):
            missing = [prompt for prompt in prompts if prompt not in outputs]
            if missing:
                raise ValueError(f"sdpo_guidance_summary_generator returned no output for {len(missing)} prompts")
            return {prompt: _guidance_output_to_text(outputs.get(prompt)) for prompt in prompts}
        outputs = list(outputs)
        if len(outputs) != len(prompts):
            raise ValueError(
                f"sdpo_guidance_summary_generator returned {len(outputs)} outputs for {len(prompts)} prompts"
            )
        return {prompt: _guidance_output_to_text(output) for prompt, output in zip(prompts, outputs, strict=True)}

    mode = str(getattr(args, "sdpo_guidance_generation_mode", "model_summary"))
    if mode == "disabled":
        raise ValueError(
            "SDPO guidance_plan produced guidance summary prompts, but sdpo_guidance_generation_mode is disabled. "
            "Set model_summary or use trajectory_demo."
        )
    if mode != "model_summary":
        raise ValueError(f"Unsupported sdpo_guidance_generation_mode: {mode}")

    endpoints = _guidance_summary_endpoints(args)
    if not endpoints:
        raise ValueError(
            "SDPO guidance_plan produced guidance summary prompts, but no SGLang endpoint is configured for "
            "guidance summary generation."
        )

    max_tokens, timeout, tokenizer, stats, jobs = _prepare_guidance_summary_jobs(
        args, prompts, endpoints=endpoints, prompt_payloads=prompt_payloads
    )
    concurrency = _guidance_summary_concurrency(args, endpoints=endpoints, prompt_count=len(prompts))
    outputs: dict[str, str] = {}
    if concurrency == 1:
        for prompt, endpoint, rendered_prompt, image_data in jobs:
            outputs[prompt] = _safe_post_guidance_summary(
                endpoint,
                rendered_prompt,
                image_data=image_data,
                prompt=prompt,
                args=args,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                timeout=timeout,
                stats=stats,
            )
        return outputs

    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="sdpo-guidance-summary") as executor:
        future_to_prompt = {
            executor.submit(
                _safe_post_guidance_summary,
                endpoint,
                rendered_prompt,
                image_data=image_data,
                prompt=prompt,
                args=args,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                timeout=timeout,
                stats=stats,
            ): prompt
            for prompt, endpoint, rendered_prompt, image_data in jobs
        }
        for future in as_completed(future_to_prompt):
            prompt = future_to_prompt[future]
            outputs[prompt] = future.result()
    return outputs


def _prepare_guidance_summary_jobs(
    args: Any,
    prompts: list[str],
    *,
    endpoints: Sequence[str],
    prompt_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[int, float, Any, GuidanceSummaryStats, list[tuple[str, str, str, list[str]]]]:
    max_tokens = int(getattr(args, "sdpo_guidance_summary_max_tokens", 256) or 256)
    timeout = float(getattr(args, "sdpo_guidance_summary_timeout", 120.0) or 120.0)
    tokenizer = _load_task_tokenizer(args)
    stats = GuidanceSummaryStats()
    setattr(args, "_sdpo_guidance_summary_last_stats", stats)
    jobs: list[tuple[str, str, str, list[str]]] = []
    for idx, prompt in enumerate(prompts):
        payload = _guidance_prompt_payload(prompt, prompt_payloads)
        image_data = [str(image) for image in (payload.get("image_data") or [])]
        rendered_prompt = _render_chat_prompt(args, tokenizer, prompt, messages=payload.get("messages"))
        rendered_prompt, image_data = _prepare_guidance_summary_prompt(
            args,
            tokenizer,
            prompt,
            rendered_prompt,
            image_data=image_data,
            max_new_tokens=max_tokens,
            stats=stats,
        )
        jobs.append((prompt, endpoints[idx % len(endpoints)], rendered_prompt, image_data))
    return max_tokens, timeout, tokenizer, stats, jobs


def _guidance_prompt_payload(
    prompt: str,
    prompt_payloads: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    if not prompt_payloads:
        return {}
    payload = prompt_payloads.get(prompt)
    return payload if isinstance(payload, Mapping) else {}


def _load_task_tokenizer(args: Any) -> Any:
    cached = getattr(args, "_sdpo_guidance_tokenizer", None)
    if cached is not None:
        return cached
    for attr in ("alfworld_tokenizer", "textcraft_tokenizer", "tokenizer"):
        tokenizer = getattr(args, attr, None)
        if tokenizer is not None:
            setattr(args, "_sdpo_guidance_tokenizer", tokenizer)
            return tokenizer
    try:
        from slime.rollout.sglang_rollout import GenerateState

        tokenizer = GenerateState(args).tokenizer
        setattr(args, "_sdpo_guidance_tokenizer", tokenizer)
        return tokenizer
    except Exception as exc:
        logger.warning("Falling back to raw guidance summary prompts; tokenizer load failed: %r", exc)
        return None


def _render_chat_prompt(
    args: Any,
    tokenizer: Any,
    prompt: str,
    *,
    messages: Any | None = None,
) -> str:
    if tokenizer is not None and getattr(args, "apply_chat_template", False):
        try:
            chat_messages = (
                list(messages) if isinstance(messages, Sequence) and not isinstance(messages, str) else None
            )
            if not chat_messages:
                chat_messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
                **(getattr(args, "apply_chat_template_kwargs", {}) or {}),
            )
        except Exception as exc:
            logger.warning("Falling back to raw guidance summary prompt; chat template failed: %r", exc)
    return prompt


def _guidance_summary_endpoints(args: Any) -> list[str]:
    raw_urls: list[Any] = [
        getattr(args, "agent_task_sdpo_guidance_sglang_urls", None),
        getattr(args, "agent_task_sdpo_guidance_sglang_url", None),
        getattr(args, "alfworld_sglang_urls", None),
        getattr(args, "alfworld_sglang_url", None),
        getattr(args, "textcraft_sglang_urls", None),
        getattr(args, "textcraft_sglang_url", None),
    ]
    if getattr(args, "sglang_router_ip", None) is not None and getattr(args, "sglang_router_port", None) is not None:
        raw_urls.append(f"http://{args.sglang_router_ip}:{args.sglang_router_port}")

    urls: list[str] = []
    for raw in raw_urls:
        if isinstance(raw, str):
            urls.extend(part.strip() for part in raw.replace(",", " ").split() if part.strip())
        elif raw:
            urls.extend(str(part).strip() for part in raw if str(part).strip())

    normalized: list[str] = []
    seen: set[str] = set()
    for url in urls:
        url = url.rstrip("/")
        if url and url not in seen:
            seen.add(url)
            normalized.append(url)
    return normalized


def _guidance_summary_concurrency(args: Any, *, endpoints: Sequence[str], prompt_count: int) -> int:
    configured = int(getattr(args, "agent_task_sdpo_guidance_summary_concurrency", 0) or 0)
    if configured <= 0:
        configured = len(endpoints)
    return max(1, min(configured, prompt_count))


def _post_sglang_generate(
    endpoint: str,
    prompt: str,
    *,
    image_data: Sequence[str] | None = None,
    max_tokens: int,
    timeout: float,
) -> str:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": max_tokens,
        },
    }
    if image_data:
        payload["image_data"] = list(image_data)
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    text = data.get("text")
    if text is None and isinstance(data.get("choices"), list) and data["choices"]:
        text = data["choices"][0].get("text", "")
    return str(text or "")


def _safe_post_guidance_summary(
    endpoint: str,
    rendered_prompt: str,
    *,
    image_data: Sequence[str] | None = None,
    prompt: str,
    args: Any,
    tokenizer: Any,
    max_tokens: int,
    timeout: float,
    stats: GuidanceSummaryStats,
) -> str:
    try:
        output = _post_sglang_generate(
            endpoint,
            rendered_prompt,
            image_data=image_data,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if output.strip():
            return output
        raise ValueError("SGLang returned empty guidance summary")
    except Exception as exc:
        stats.retry_count += 1
        retry_prompt, retry_image_data = _prepare_guidance_summary_prompt(
            args,
            tokenizer,
            prompt,
            rendered_prompt,
            image_data=list(image_data or []),
            max_new_tokens=max_tokens,
            stats=stats,
            force_shorter=True,
        )
        logger.warning(
            "SDPO guidance summary generation failed; retrying with shorter input. endpoint=%s prompt_prefix=%r "
            "error=%r",
            endpoint,
            prompt[:120],
            exc,
        )
        try:
            output = _post_sglang_generate(
                endpoint,
                retry_prompt,
                image_data=retry_image_data,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            if output.strip():
                return output
            raise ValueError("SGLang returned empty guidance summary after retry")
        except Exception:
            stats.failure_count += 1
            raise


def _prepare_guidance_summary_prompt(
    args: Any,
    tokenizer: Any,
    prompt: str,
    rendered_prompt: str,
    *,
    image_data: Sequence[str] | None = None,
    max_new_tokens: int,
    stats: GuidanceSummaryStats,
    force_shorter: bool = False,
) -> tuple[str, list[str]]:
    image_data = list(image_data or [])
    budget = _guidance_summary_max_prompt_tokens(args, max_new_tokens=max_new_tokens)
    if force_shorter:
        budget = max(1, budget // 2)
    token_count = _token_count(tokenizer, rendered_prompt)
    stats.observe_prompt(token_count, image_count=len(image_data))
    if token_count <= budget and not force_shorter:
        return rendered_prompt, _match_image_data_to_prompt(rendered_prompt, image_data)

    stats.prompt_overlength_count += int(token_count > budget)
    if image_data:
        return rendered_prompt, _match_image_data_to_prompt(rendered_prompt, image_data)

    for candidate in _guidance_summary_prompt_candidates(prompt, force_shorter=force_shorter):
        rendered_candidate = _render_chat_prompt(args, tokenizer, candidate)
        if _token_count(tokenizer, rendered_candidate) <= budget:
            stats.prompt_truncated_count += 1
            return rendered_candidate, _match_image_data_to_prompt(rendered_candidate, image_data)

    stats.prompt_truncated_count += 1
    truncated = _truncate_text_to_token_budget(tokenizer, rendered_prompt, budget)
    return truncated, _match_image_data_to_prompt(truncated, image_data)


def _guidance_summary_max_prompt_tokens(args: Any, *, max_new_tokens: int) -> int:
    configured = int(getattr(args, "sdpo_guidance_summary_max_prompt_tokens", 0) or 0)
    if configured > 0:
        return configured
    context_length = int(
        getattr(args, "sdpo_guidance_summary_context_length", None)
        or getattr(args, "rollout_max_context_len", None)
        or 40960
    )
    return max(1, context_length - int(max_new_tokens) - 512)


def _token_count(tokenizer: Any, text: str) -> int:
    if tokenizer is None:
        return len(text.split())
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def _truncate_text_to_token_budget(tokenizer: Any, text: str, budget: int) -> str:
    if tokenizer is None:
        return " ".join(text.split()[: max(1, budget)])
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    if len(token_ids) <= budget:
        return text
    try:
        return tokenizer.decode(token_ids[: max(1, budget)], skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(token_ids[: max(1, budget)])


def _guidance_summary_prompt_candidates(prompt: str, *, force_shorter: bool) -> list[str]:
    variants = [(32, 800), (16, 500), (8, 320), (4, 220), (2, 160)]
    if force_shorter:
        variants = variants[2:]
    return [
        _compact_guidance_summary_prompt(prompt, max_steps=steps, max_line_chars=chars) for steps, chars in variants
    ]


def _match_image_data_to_prompt(prompt: str, image_data: Sequence[str]) -> list[str]:
    images = list(image_data)
    if not images:
        return []
    marker_count = _count_image_markers(prompt)
    if marker_count <= 0:
        return []
    return images[:marker_count]


def _count_image_markers(prompt: str) -> int:
    text = str(prompt)
    counts = [
        text.count("<image>"),
        text.count("<|vision_start|>"),
        text.count("<|image_pad|>"),
    ]
    return max(counts)


def _compact_guidance_summary_prompt(prompt: str, *, max_steps: int, max_line_chars: int) -> str:
    lines = prompt.splitlines()
    marker_index = None
    for idx, line in enumerate(lines):
        if line.strip() in {
            "Successful trajectory evidence:",
            "Unsuccessful trajectory evidence:",
            "Successful trajectory:",
            "Unsuccessful trajectory:",
        }:
            marker_index = idx
            break
    if marker_index is None:
        return _limit_lines(prompt, max_lines=max_steps + 32, max_line_chars=max_line_chars)

    header = "\n".join(lines[: marker_index + 1]).strip()
    trajectory = lines[marker_index + 1 :]
    compact_lines: list[str] = []
    action_count = 0
    for line in trajectory:
        stripped = line.strip()
        if not stripped:
            if compact_lines and compact_lines[-1]:
                compact_lines.append("")
            continue
        if _looks_like_numbered_step(stripped):
            action_count += 1
            if action_count > max_steps:
                continue
            compact_lines.append(_limit_text(stripped, max_line_chars))
            continue
        if stripped.startswith("Result:") and action_count <= max_steps:
            compact_lines.append(_limit_text(stripped, max_line_chars))
            continue
        if len(compact_lines) < 8:
            compact_lines.append(_limit_text(stripped, max_line_chars))
    return f"{header}\n{chr(10).join(compact_lines).strip()}".strip()


def _looks_like_numbered_step(text: str) -> bool:
    first = text.split(".", 1)[0]
    return bool(first) and first.isdigit()


def _limit_lines(text: str, *, max_lines: int, max_line_chars: int) -> str:
    return "\n".join(_limit_text(line, max_line_chars) for line in text.splitlines()[:max_lines]).strip()


def _limit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 16)].rstrip() + " ... [truncated]"


def _bind_precomputed_self_trajectory_outputs(
    samples: Sequence[Any], plan: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if len(plan) != len(samples):
        raise ValueError("self-trajectory guidance binding requires one context-plan row per sample")
    bound: list[dict[str, Any]] = []
    prompt_fields = ("sdpo_success_guidance_prompt", "sdpo_failure_guidance_prompt")
    for sample, original_row in zip(samples, plan, strict=True):
        row = dict(original_row)
        active_fields = [field for field in prompt_fields if str(row.get(field) or "").strip()]
        if not active_fields:
            bound.append(row)
            continue
        if len(active_fields) != 1:
            raise ValueError("self-trajectory guidance plan row has multiple summary prompts")
        train_metadata = getattr(sample, "train_metadata", None)
        precomputed = (
            train_metadata.get(_PRECOMPUTED_GUIDANCE_OUTPUTS_KEY)
            if isinstance(train_metadata, Mapping)
            else None
        )
        if isinstance(precomputed, Mapping) and len(precomputed) == 1:
            # Prompt text is not a trajectory identity: distinct trajectories
            # can produce the same full or sparse prompt but have different
            # validated summaries. Keep the output on this row so finalization
            # never resolves it through the global prompt->output map.
            row[_ROW_PRECOMPUTED_GUIDANCE_OUTPUT_KEY] = _guidance_output_to_text(
                next(iter(precomputed.values()))
            )
        bound.append(row)
    return bound


def _missing_guidance_prompts(
    plan: Sequence[Mapping[str, Any]], guidance_outputs: Mapping[str, str]
) -> list[str]:
    missing: list[str] = []
    seen: set[str] = set()
    for row in plan:
        has_row_output = _ROW_PRECOMPUTED_GUIDANCE_OUTPUT_KEY in row
        for field in ("sdpo_success_guidance_prompt", "sdpo_failure_guidance_prompt"):
            prompt = str(row.get(field) or "").strip()
            if prompt and not has_row_output and prompt not in guidance_outputs and prompt not in seen:
                seen.add(prompt)
                missing.append(prompt)
    return missing


def _collect_precomputed_guidance_outputs(samples: Sequence[Any], prompts: Sequence[str]) -> dict[str, str]:
    prompt_set = set(prompts)
    outputs: dict[str, str] = {}
    for sample in samples:
        train_metadata = getattr(sample, "train_metadata", None)
        if not isinstance(train_metadata, dict):
            continue
        precomputed = train_metadata.get(_PRECOMPUTED_GUIDANCE_OUTPUTS_KEY)
        if not isinstance(precomputed, Mapping):
            continue
        for prompt, output in precomputed.items():
            prompt_text = str(prompt)
            if prompt_text in prompt_set and prompt_text not in outputs:
                outputs[prompt_text] = _guidance_output_to_text(output)
    return outputs


def _guidance_output_to_text(value: Any) -> str:
    return "" if value is None else str(value)


def _collect_guidance_summary_stats(samples: Sequence[Any]) -> GuidanceSummaryStats:
    stats = GuidanceSummaryStats()
    seen_stats: set[int] = set()
    for sample in samples:
        train_metadata = getattr(sample, "train_metadata", None)
        if not isinstance(train_metadata, dict):
            continue
        raw = train_metadata.get(_PRECOMPUTED_GUIDANCE_STATS_KEY)
        if not isinstance(raw, Mapping):
            continue
        identity = id(raw)
        if identity in seen_stats:
            continue
        seen_stats.add(identity)
        stats.merge(GuidanceSummaryStats.from_mapping(raw))
    return stats


def _last_guidance_summary_stats(args: Any) -> GuidanceSummaryStats:
    stats = getattr(args, "_sdpo_guidance_summary_last_stats", None)
    return stats if isinstance(stats, GuidanceSummaryStats) else GuidanceSummaryStats()


def _guidance_output_metrics(
    samples: Sequence[Any],
    plan: Sequence[Mapping[str, Any]],
    guidance_outputs: Mapping[str, str],
    *,
    debug_logged_count: int,
) -> dict[str, float]:
    entries = _resolved_guidance_output_entries(samples, plan, guidance_outputs)
    if not entries:
        return {
            "self_distillation/guidance_summary_output_count": 0.0,
            "self_distillation/guidance_summary_output_chars_mean": 0.0,
            "self_distillation/guidance_summary_output_chars_max": 0.0,
            "self_distillation/guidance_summary_schema_valid_fraction": 0.0,
            "self_distillation/guidance_summary_debug_logged_count": float(debug_logged_count),
        }

    clean_outputs = [clean_guidance_summary(entry["raw_output"]) for entry in entries]
    lengths = [len(output) for output in clean_outputs]
    valid = [
        _guidance_schema_valid(output, str(entry["kind"]))
        for entry, output in zip(entries, clean_outputs, strict=True)
    ]
    count = len(entries)
    return {
        "self_distillation/guidance_summary_output_count": float(count),
        "self_distillation/guidance_summary_output_chars_mean": sum(lengths) / count,
        "self_distillation/guidance_summary_output_chars_max": float(max(lengths)),
        "self_distillation/guidance_summary_schema_valid_fraction": sum(1 for item in valid if item) / count,
        "self_distillation/guidance_summary_debug_logged_count": float(debug_logged_count),
    }


def _resolved_guidance_output_entries(
    samples: Sequence[Any],
    plan: Sequence[Mapping[str, Any]],
    guidance_outputs: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Resolve summary outputs without treating prompt text as trajectory identity."""
    if len(plan) != len(samples):
        raise ValueError("guidance output resolution requires one context-plan row per sample")
    entries: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row_index, (sample, row) in enumerate(zip(samples, plan, strict=True)):
        for kind, field in (
            ("success", "sdpo_success_guidance_prompt"),
            ("failure", "sdpo_failure_guidance_prompt"),
        ):
            prompt = str(row.get(field) or "").strip()
            if not prompt:
                continue
            has_row_output = _ROW_PRECOMPUTED_GUIDANCE_OUTPUT_KEY in row
            metadata = row.get("sdpo_metadata") or {}
            if has_row_output:
                identity = (
                    "row_precomputed",
                    str(metadata.get("uid")),
                    str(metadata.get("traj_uid")),
                    kind,
                )
                raw_output = _guidance_output_to_text(row[_ROW_PRECOMPUTED_GUIDANCE_OUTPUT_KEY])
                source_prompt = _single_precomputed_guidance_prompt(sample)
                binding = "row_precomputed"
            else:
                identity = ("prompt", prompt)
                raw_output = _guidance_output_to_text(guidance_outputs.get(prompt))
                source_prompt = None
                binding = "prompt"
            existing = entries.get(identity)
            if existing is None:
                entries[identity] = {
                    "prompt": prompt,
                    "kind": kind,
                    "raw_output": raw_output,
                    "binding": binding,
                    "source_prompt": source_prompt,
                    "row_indices": [row_index],
                }
                continue
            if (
                existing["kind"] != kind
                or existing["raw_output"] != raw_output
                or existing["source_prompt"] != source_prompt
            ):
                raise ValueError("conflicting guidance outputs within one trajectory identity")
            existing["row_indices"].append(row_index)
    return list(entries.values())


def _single_precomputed_guidance_prompt(sample: Any) -> str | None:
    train_metadata = getattr(sample, "train_metadata", None)
    precomputed = (
        train_metadata.get(_PRECOMPUTED_GUIDANCE_OUTPUTS_KEY)
        if isinstance(train_metadata, Mapping)
        else None
    )
    if not isinstance(precomputed, Mapping) or len(precomputed) != 1:
        return None
    return str(next(iter(precomputed)))


def _write_guidance_debug_records(
    args: Any,
    samples: Sequence[Sample],
    pre_finalize_plan: list[dict[str, Any]],
    finalized_plan: list[dict[str, Any]],
    guidance_outputs: Mapping[str, str],
    *,
    prompt_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> int:
    debug_dir = _guidance_debug_dir(args)
    if debug_dir is None:
        return 0
    entries = _resolved_guidance_output_entries(samples, pre_finalize_plan, guidance_outputs)
    if not entries:
        return 0
    prompts = [str(entry["prompt"]) for entry in entries]

    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        output_path = debug_dir / _guidance_debug_filename(samples, prompts)
        max_records = int(getattr(args, "sdpo_guidance_debug_max_records", 0) or 0)
        written = 0
        with gzip.open(output_path, "wt", encoding="utf-8") as handle:
            for entry in entries:
                if max_records > 0 and written >= max_records:
                    break
                record = _guidance_debug_record(
                    samples,
                    pre_finalize_plan,
                    finalized_plan,
                    entry,
                    prompt_payloads=prompt_payloads,
                )
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
        return written
    except Exception as exc:
        logger.warning("Failed to write SDPO guidance debug records: %r", exc)
        return 0


def _guidance_debug_record(
    samples: Sequence[Sample],
    pre_finalize_plan: list[dict[str, Any]],
    finalized_plan: list[dict[str, Any]],
    entry: Mapping[str, Any],
    *,
    prompt_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    prompt = str(entry["prompt"])
    raw_output = _guidance_output_to_text(entry["raw_output"])
    model_clean_output = clean_guidance_summary(raw_output)
    row_indices = [int(index) for index in entry["row_indices"]]
    fallback_output = _guidance_fallback_output_for_rows(finalized_plan, pre_finalize_plan, row_indices)
    clean_output = fallback_output or model_clean_output
    payload = _guidance_prompt_payload(prompt, prompt_payloads)
    image_data = payload.get("image_data") or []
    image_hashes = payload.get("image_hashes") or []
    prompt_rows = []
    for row_index in row_indices:
        sample = samples[row_index]
        before = pre_finalize_plan[row_index]
        after = finalized_plan[row_index]
        kind = _guidance_prompt_kind_for_row(before, prompt)
        if kind is None:
            kind = str(entry["kind"])
        metadata = before.get("sdpo_metadata") or {}
        prompt_rows.append(
            {
                "kind": kind,
                "uid": metadata.get("uid"),
                "target_traj_uid": metadata.get("traj_uid"),
                "source_traj_uid": (
                    before.get("sdpo_selected_success_traj_uid") if kind == "success" else metadata.get("traj_uid")
                ),
                "turn_idx": metadata.get("turn_idx"),
                "sample_index": getattr(sample, "index", None),
                "rollout_id": getattr(sample, "rollout_id", None),
                "teacher_signal_type": after.get("sdpo_teacher_signal_type"),
                "used": float(after.get("self_distillation_mask", 0.0)) > 0.0,
            }
        )
    kinds = sorted({str(row["kind"]) for row in prompt_rows})
    kind = kinds[0] if len(kinds) == 1 else "mixed"
    profile = ""
    for row in pre_finalize_plan:
        metadata = row.get("sdpo_metadata") or {}
        profile = str(metadata.get("sdpo_metadata_profile") or "")
        if profile:
            break
    return {
        "schema_name": "sdpo_guidance_debug",
        "schema_version": 1,
        "task_profile": profile,
        "prompt_kind": kind,
        "output_binding": entry["binding"],
        "frozen_source_prompt_text": entry.get("source_prompt"),
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "prompt_token_count": _token_count(None, prompt),
        "image_count": len(image_data),
        "image_hashes": list(image_hashes),
        "prompt_text": prompt,
        "raw_output_text": raw_output,
        "model_clean_output_text": model_clean_output,
        "clean_output_text": clean_output,
        "fallback_used": bool(fallback_output),
        "schema_valid": _guidance_schema_valid(clean_output, kind),
        "used_count": sum(1 for row in prompt_rows if row["used"]),
        "teacher_signal_types": sorted({str(row["teacher_signal_type"]) for row in prompt_rows}),
        "uids": sorted({str(row["uid"]) for row in prompt_rows if row["uid"] is not None}),
        "source_traj_uids": sorted(
            {str(row["source_traj_uid"]) for row in prompt_rows if row["source_traj_uid"] is not None}
        ),
        "target_traj_uids": sorted(
            {str(row["target_traj_uid"]) for row in prompt_rows if row["target_traj_uid"] is not None}
        ),
        "turn_indices": [row["turn_idx"] for row in prompt_rows],
        "sample_indices": [row["sample_index"] for row in prompt_rows],
        "rollout_ids": [row["rollout_id"] for row in prompt_rows],
    }


def _guidance_fallback_output(
    finalized_plan: Sequence[Mapping[str, Any]],
    pre_finalize_plan: Sequence[Mapping[str, Any]],
    prompt: str,
) -> str:
    for before, after in zip(pre_finalize_plan, finalized_plan, strict=True):
        if before.get("sdpo_success_guidance_prompt") == prompt and after.get("sdpo_success_guidance_fallback_used"):
            return str(after.get("sdpo_success_guidance_text") or "")
        if before.get("sdpo_failure_guidance_prompt") == prompt and after.get("sdpo_failure_guidance_fallback_used"):
            return str(after.get("sdpo_failure_guidance_text") or "")
    return ""


def _guidance_fallback_output_for_rows(
    finalized_plan: Sequence[Mapping[str, Any]],
    pre_finalize_plan: Sequence[Mapping[str, Any]],
    row_indices: Sequence[int],
) -> str:
    for row_index in row_indices:
        before = pre_finalize_plan[row_index]
        after = finalized_plan[row_index]
        if before.get("sdpo_success_guidance_prompt") and after.get("sdpo_success_guidance_fallback_used"):
            return str(after.get("sdpo_success_guidance_text") or "")
        if before.get("sdpo_failure_guidance_prompt") and after.get("sdpo_failure_guidance_fallback_used"):
            return str(after.get("sdpo_failure_guidance_text") or "")
    return ""


def _guidance_debug_dir(args: Any) -> Path | None:
    if not _truthy(getattr(args, "sdpo_guidance_debug_enabled", False)):
        return None
    raw_dir = getattr(args, "sdpo_guidance_debug_dir", None)
    if raw_dir:
        path = Path(str(raw_dir))
        if path.is_absolute():
            return path
        base_dir = _guidance_debug_base_dir(args)
        return base_dir / path
    for attr in ("alfworld_sample_log_dir", "textcraft_sample_log_dir"):
        sample_log_dir = getattr(args, attr, None)
        if sample_log_dir:
            return Path(str(sample_log_dir)).parent / "sdpo_guidance"
    return None


def _guidance_debug_base_dir(args: Any) -> Path:
    for attr in ("alfworld_sample_log_dir", "textcraft_sample_log_dir"):
        sample_log_dir = getattr(args, attr, None)
        if sample_log_dir:
            return Path(str(sample_log_dir)).parent
    return Path.cwd()


def _guidance_debug_filename(samples: Sequence[Sample], prompts: Sequence[str]) -> str:
    prompt_hash = hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()[:10]
    return f"rollout_{_guidance_debug_rollout_label(samples)}_{os.getpid()}_{time.time_ns()}_{prompt_hash}.jsonl.gz"


def _guidance_debug_rollout_label(samples: Sequence[Sample]) -> str:
    rollout_ids = sorted(
        {
            int(sample.rollout_id if sample.rollout_id is not None else sample.index)
            for sample in samples
            if getattr(sample, "index", None) is not None
        }
    )
    if not rollout_ids:
        return "unknown"
    if len(rollout_ids) == 1:
        return f"{rollout_ids[0]:07d}"
    return f"mixed_{rollout_ids[0]:07d}_{rollout_ids[-1]:07d}"


def _guidance_prompt_kind_for_prompt(plan: list[dict[str, Any]], prompt: str) -> str:
    for row in plan:
        kind = _guidance_prompt_kind_for_row(row, prompt)
        if kind is not None:
            return kind
    return "unknown"


def _guidance_prompt_kind_for_row(row: Mapping[str, Any], prompt: str) -> str | None:
    if str(row.get("sdpo_success_guidance_prompt") or "") == prompt:
        return "success"
    if str(row.get("sdpo_failure_guidance_prompt") or "") == prompt:
        return "failure"
    return None


def _guidance_schema_valid(text: str, kind: str) -> bool:
    if kind == "success":
        return _guidance_output_has_schema(text, "success")
    if kind == "failure":
        return _guidance_output_has_schema(text, "failure")
    return _guidance_output_has_schema(text, "success") or _guidance_output_has_schema(text, "failure")


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
