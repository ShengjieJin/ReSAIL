from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step
from slime_plugins.agent_tasks.common.config import as_bool, get_arg_or_env
from slime_plugins.agent_tasks.common.logging import summarize_actor_entropy_records
from slime_plugins.agent_tasks.common.trace import (
    AgentTraceConfig,
    DEFAULT_TRACE_COMPRESSION,
    patch_trace_sidecar_actor_entropy,
)


def postprocess_rollout_entropy(args: Any, rollout_id: int, rollout_data: dict[str, Any]) -> None:
    if not as_bool(
        get_arg_or_env(args, "agent_task_diversity_entropy_enabled", "AGENT_TASK_DIVERSITY_ENTROPY_ENABLED", False)
    ):
        return

    local_records, local_expected_count = entropy_records_from_rollout_data(args, rollout_data)
    local_expected_by_task = entropy_expected_counts_by_task(rollout_data)
    rollout_data.pop("actor_entropy", None)
    payloads = _all_gather_payloads(
        {
            "records": local_records,
            "expected_count": local_expected_count,
            "expected_by_task": local_expected_by_task,
        }
    )
    if _rank() != 0:
        return

    records = [record for payload in payloads for record in payload["records"]]
    expected_count = sum(int(payload["expected_count"]) for payload in payloads)
    expected_by_task: dict[str, int] = {}
    for payload in payloads:
        for task, count in (payload.get("expected_by_task") or {}).items():
            expected_by_task[str(task)] = expected_by_task.get(str(task), 0) + int(count)
    grouped_records = _group_by_task(records)
    for task in sorted(set(expected_by_task) | set(grouped_records)):
        task_records = grouped_records.get(task, [])
        task_expected_count = expected_by_task.get(task, len(task_records))
        metrics = summarize_actor_entropy_records(
            task_records,
            metric_root=f"diversity/rollout/{task}",
            expected_count=task_expected_count,
        )
        _patch_task_trace(args, task=task, rollout_id=rollout_id, records=task_records, required=False)
        step = compute_rollout_step(args, rollout_id)
        metrics["rollout/step"] = step
        logging_utils.log(args, metrics, step_key="rollout/step")


def entropy_records_from_rollout_data(
    args: Any,
    rollout_data: dict[str, Any],
    *,
    source: str | None = None,
    temperature: float | None = None,
) -> tuple[list[dict[str, Any]], int]:
    _raise_if_context_parallel_entropy_unsupported()
    if not _is_entropy_source_rank():
        return [], 0
    entropy_values = rollout_data.get("actor_entropy") or rollout_data.get("entropy")
    metadata_rows = rollout_data.get("metadata") or []
    if not entropy_values or not metadata_rows:
        return [], 0
    loss_masks = rollout_data.get("loss_masks") or [None] * len(metadata_rows)
    response_lengths = rollout_data.get("response_lengths") or [None] * len(metadata_rows)
    source = source or ("old_actor_forward" if getattr(args, "keep_old_actor", False) else "actor_forward")
    temperature = float(getattr(args, "rollout_temperature", 1.0) if temperature is None else temperature)
    records = []
    expected_count = 0
    for entropy, metadata, loss_mask, response_length in zip(
        entropy_values,
        metadata_rows,
        loss_masks,
        response_lengths,
        strict=False,
    ):
        token_count = int(response_length if response_length is not None else _tensor_len(entropy))
        if token_count <= 0:
            continue
        expected_count += 1
        summary = _summarize_entropy_tensor(entropy, loss_mask)
        if summary is None:
            continue
        records.append(
            {
                "sample_rollout_id": metadata.get("sample_rollout_id", metadata.get("rollout_id")),
                "traj_uid": metadata.get("traj_uid"),
                "turn_idx": metadata.get("turn_idx"),
                "sample_index": metadata.get("sample_index"),
                "agent_task": metadata.get("agent_task"),
                "agent_task_trace_dir": metadata.get("agent_task_trace_dir"),
                "actor_entropy": {
                    "enabled": True,
                    "source": source,
                    "temperature": temperature,
                    "token_count": summary["token_count"],
                    "token_mean": summary["token_mean"],
                    "token_std": summary["token_std"],
                },
            }
        )
    return records, expected_count


def entropy_expected_counts_by_task(rollout_data: dict[str, Any]) -> dict[str, int]:
    metadata_rows = rollout_data.get("metadata") or []
    response_lengths = rollout_data.get("response_lengths") or [None] * len(metadata_rows)
    counts: dict[str, int] = {}
    for metadata, response_length in zip(metadata_rows, response_lengths, strict=False):
        if int(response_length or 0) <= 0:
            continue
        task = metadata.get("agent_task") if isinstance(metadata, dict) else None
        if task:
            counts[str(task)] = counts.get(str(task), 0) + 1
    return counts


def _summarize_entropy_tensor(entropy: Any, loss_mask: Any) -> dict[str, float] | None:
    values = _to_float_tensor(entropy)
    if values.numel() == 0:
        return None
    if loss_mask is not None:
        mask = _to_float_tensor(loss_mask).to(dtype=torch.bool)
        usable = min(values.numel(), mask.numel())
        values = values[:usable][mask[:usable]]
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None
    return {
        "token_count": int(values.numel()),
        "token_mean": float(values.mean().item()),
        "token_std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
    }


def _patch_task_trace(args: Any, *, task: str, rollout_id: int, records: list[dict[str, Any]], required: bool) -> None:
    trace_dir = _trace_dir(args, records)
    if trace_dir is None:
        if required:
            raise RuntimeError(f"trace dir unavailable for required rollout actor entropy task={task}")
        return
    config = AgentTraceConfig(
        enabled=True,
        trace_dir=trace_dir,
        phases=frozenset({"train", "eval"}),
        compression=str(
            get_arg_or_env(
                args, "agent_task_trace_compression", "AGENT_TASK_TRACE_COMPRESSION", DEFAULT_TRACE_COMPRESSION
            )
        )
        .strip()
        .lower()
        or "none",
    )
    patch_trace_sidecar_actor_entropy(
        config,
        task=task,
        phase="train",
        outer_rollout_id=int(rollout_id),
        records=records,
        required=required,
    )


def _trace_dir(args: Any, records: list[dict[str, Any]]) -> Path | None:
    for record in records:
        value = record.get("agent_task_trace_dir")
        if value:
            return Path(str(value))
    raw = get_arg_or_env(args, "agent_task_trace_dir", "AGENT_TASK_TRACE_DIR", None)
    return Path(str(raw)) if raw else None


def _group_by_task(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        task = record.get("agent_task")
        if task:
            grouped.setdefault(str(task), []).append(record)
    return grouped


def _all_gather_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not dist.is_available() or not dist.is_initialized():
        return [payload]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, payload)
    return [item for item in gathered if isinstance(item, dict)]


def _is_entropy_source_rank() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    try:
        from megatron.core import mpu

        return mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0
    except Exception:
        return True


def _raise_if_context_parallel_entropy_unsupported() -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    try:
        from megatron.core import mpu

        cp_size = int(mpu.get_context_parallel_world_size())
    except Exception:
        return
    if cp_size > 1:
        raise RuntimeError(
            "agent-task actor entropy does not support context_parallel_size > 1 yet; "
            "set --context-parallel-size 1 or disable diversity entropy"
        )


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _to_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().flatten()
    return torch.as_tensor(value, dtype=torch.float32).flatten()


def _tensor_len(value: Any) -> int:
    try:
        return int(value.numel())
    except AttributeError:
        try:
            return len(value)
        except TypeError:
            return 0
