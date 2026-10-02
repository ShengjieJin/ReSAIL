from __future__ import annotations

import gzip
import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from slime_plugins.agent_tasks.common.config import as_bool, get_arg_or_env, optional_str

TRACE_SCHEMA_NAME = "agent_task_step_trace"
TRACE_SCHEMA_VERSION = 1
DEFAULT_TRACE_COMPRESSION = "gzip"
_PROMPT_TRACE_KEYS = {"raw_prompt", "rendered_prompt", "prompt_text", "messages"}
_PROTECTED_EXTRA_KEYS = {
    "schema_name",
    "schema_version",
    "task",
    "phase",
    "outer_rollout_id",
    "uid",
    "traj_uid",
    "turn_idx",
    "sample_rollout_id",
    "sample_index",
    "group_index",
    "eval_dataset_name",
    "task_id",
    "seed",
    "split",
    "prompt_hash",
    "prompt_token_count",
    "message_role_summary",
    "response_text",
    "response_token_ids",
    "rollout_log_probs",
    "loss_mask_sum",
    "response_token_count",
    "finish_reason",
    "projected_action",
    "format_valid",
    "missing_action_tag",
    "invalid_reason",
    "is_action_valid",
    "score",
    "reward",
    "raw_reward",
    "grpo_reward",
    "success",
    "won",
    "done",
    "is_terminal",
    "episode_length",
    "env_step_failed",
    "episode_error",
    "termination_reason",
    "actor_entropy",
}


@dataclass(frozen=True)
class AgentTraceConfig:
    enabled: bool
    trace_dir: Path | None
    phases: frozenset[str]
    compression: str = DEFAULT_TRACE_COMPRESSION
    include_prompts: bool = False
    include_token_entropy: bool = False

    def enabled_for_phase(self, phase: str) -> bool:
        return self.enabled and phase in self.phases and self.trace_dir is not None


@dataclass(frozen=True)
class TraceWriteResult:
    path: Path
    row_count: int
    manifest_path: Path


_TRACE_RECORDS: dict[str, list[dict[str, Any]]] = {}
_TRACE_LOCK = threading.Lock()


def get_agent_trace_config(args: Any, *, task: str, sample_log_dir: str | Path | None = None) -> AgentTraceConfig:
    enabled = as_bool(get_arg_or_env(args, "agent_task_trace_enabled", "AGENT_TASK_TRACE_ENABLED", False))
    phases = _parse_phases(get_arg_or_env(args, "agent_task_trace_phases", "AGENT_TASK_TRACE_PHASES", "train,eval"))
    compression = (
        str(
            get_arg_or_env(
                args, "agent_task_trace_compression", "AGENT_TASK_TRACE_COMPRESSION", DEFAULT_TRACE_COMPRESSION
            )
        )
        .strip()
        .lower()
    )
    if compression not in {"gzip", "none", ""}:
        raise ValueError(f"agent task trace compression must be 'gzip' or 'none', got {compression!r}")
    trace_dir = _resolve_trace_dir(args, task=task, sample_log_dir=sample_log_dir)
    return AgentTraceConfig(
        enabled=enabled,
        trace_dir=trace_dir,
        phases=phases,
        compression="none" if compression == "" else compression,
        include_prompts=as_bool(
            get_arg_or_env(args, "agent_task_trace_include_prompts", "AGENT_TASK_TRACE_INCLUDE_PROMPTS", False)
        ),
        include_token_entropy=as_bool(
            get_arg_or_env(
                args,
                "agent_task_trace_include_token_entropy",
                "AGENT_TASK_TRACE_INCLUDE_TOKEN_ENTROPY",
                False,
            )
        ),
    )


def keep_prompt_metadata_enabled(args: Any) -> bool:
    return as_bool(get_arg_or_env(args, "agent_task_keep_prompt_metadata", "AGENT_TASK_KEEP_PROMPT_METADATA", False))


def compact_prompt_metadata(
    args: Any,
    *,
    raw_prompt: str | None,
    rendered_prompt: str | None,
    messages: list[dict[str, Any]] | None,
    prompt_text: str | None = None,
) -> dict[str, Any]:
    prompt_for_preview = rendered_prompt or prompt_text or raw_prompt
    preview_chars = int(
        get_arg_or_env(args, "agent_task_prompt_preview_chars", "AGENT_TASK_PROMPT_PREVIEW_CHARS", 4096)
    )
    metadata: dict[str, Any] = {
        "prompt_text_preview": _preview_text(prompt_for_preview, max_chars=preview_chars),
        "raw_prompt_preview": _preview_text(raw_prompt, max_chars=preview_chars),
        "message_roles": _message_role_summary(messages),
    }
    if keep_prompt_metadata_enabled(args):
        metadata.update(
            {
                "raw_prompt": raw_prompt,
                "rendered_prompt": rendered_prompt,
                "messages": messages,
            }
        )
        if prompt_text is not None:
            metadata["prompt_text"] = prompt_text
    return metadata


def add_trace_record(task: str, record: dict[str, Any]) -> None:
    with _TRACE_LOCK:
        _TRACE_RECORDS.setdefault(task, []).append(record)


def clear_trace_records(task: str | None = None) -> None:
    with _TRACE_LOCK:
        if task is None:
            _TRACE_RECORDS.clear()
        else:
            _TRACE_RECORDS.pop(task, None)


def drain_trace_records_for_samples(task: str, phase: str, samples: Iterable[Any]) -> list[dict[str, Any]]:
    keys = {_sample_trace_key(sample) for sample in samples}
    keys.discard(None)
    with _TRACE_LOCK:
        records = _TRACE_RECORDS.get(task, [])
        selected: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        for record in records:
            if record.get("phase") == phase and _record_trace_key(record) in keys:
                selected.append(record)
            else:
                remaining.append(record)
        if remaining:
            _TRACE_RECORDS[task] = remaining
        else:
            _TRACE_RECORDS.pop(task, None)
    return selected


def build_trace_record(
    *,
    task: str,
    phase: str,
    metadata: dict[str, Any],
    response_text: str | None,
    response_token_ids: list[int],
    rollout_log_probs: list[float],
    prompt_text: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    include_prompts: bool = False,
    include_token_entropy: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_name": TRACE_SCHEMA_NAME,
        "schema_version": TRACE_SCHEMA_VERSION,
        "task": task,
        "phase": phase,
        "uid": metadata.get("uid"),
        "traj_uid": metadata.get("traj_uid"),
        "turn_idx": metadata.get("turn_idx"),
        "sample_rollout_id": metadata.get("sample_rollout_id", metadata.get("rollout_id")),
        "sample_index": metadata.get("sample_index"),
        "group_index": metadata.get("group_index"),
        "task_id": metadata.get("task_id"),
        "eval_dataset_name": metadata.get("eval_dataset_name"),
        "seed": metadata.get("seed"),
        "split": metadata.get("split"),
        "prompt_token_count": metadata.get("prompt_tokens") or metadata.get("raw_prompt_tokens"),
        "message_role_summary": _message_role_summary(messages),
        "response_text": response_text or "",
        "response_token_ids": list(response_token_ids),
        "rollout_log_probs": [_json_float(value) for value in rollout_log_probs],
        "loss_mask_sum": metadata.get("loss_mask_sum"),
        "response_token_count": metadata.get("response_tokens") or len(response_token_ids),
        "finish_reason": metadata.get("finish_reason"),
        "projected_action": metadata.get("projected_action"),
        "format_valid": metadata.get("format_valid"),
        "missing_action_tag": metadata.get("missing_action_tag"),
        "invalid_reason": metadata.get("invalid_reason"),
        "is_action_valid": metadata.get("is_action_valid"),
        "score": metadata.get("score"),
        "reward": metadata.get("reward"),
        "raw_reward": metadata.get("raw_reward"),
        "grpo_reward": metadata.get("grpo_reward"),
        "success": metadata.get("success"),
        "won": metadata.get("won"),
        "done": metadata.get("done"),
        "is_terminal": metadata.get("is_terminal"),
        "episode_length": metadata.get("episode_length"),
        "env_step_failed": metadata.get("env_step_failed"),
        "episode_error": metadata.get("episode_error"),
        "termination_reason": metadata.get("termination_reason"),
        "actor_entropy": _actor_entropy_from_metadata(metadata, include_token_entropy=include_token_entropy),
    }
    if include_prompts:
        record["raw_prompt"] = prompt_text
        record["messages"] = messages
    if extra:
        overlap = set(extra) & (_PROTECTED_EXTRA_KEYS | (_PROMPT_TRACE_KEYS if not include_prompts else set()))
        if overlap:
            raise ValueError(f"trace extra fields cannot override protected keys: {sorted(overlap)!r}")
        record.update(extra)
    return sanitize_for_json(record)


def add_sample_trace_record(
    *,
    config: AgentTraceConfig,
    task: str,
    phase: str,
    sample: Any,
    response_token_ids: list[int],
    rollout_log_probs: list[float],
    prompt_text: str | None,
    messages: list[dict[str, Any]] | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not config.enabled_for_phase(phase):
        return None
    metadata = dict(getattr(sample, "metadata", None) or {})
    metadata.setdefault("rollout_id", getattr(sample, "rollout_id", None))
    metadata.setdefault("sample_rollout_id", getattr(sample, "rollout_id", None))
    metadata.setdefault("sample_index", getattr(sample, "index", None))
    metadata.setdefault("group_index", getattr(sample, "group_index", None))
    metadata.setdefault("loss_mask_sum", sum(getattr(sample, "loss_mask", None) or []))
    metadata.setdefault("response_tokens", len(response_token_ids))
    record = build_trace_record(
        task=task,
        phase=phase,
        metadata=metadata,
        response_text=getattr(sample, "response", None),
        response_token_ids=response_token_ids,
        rollout_log_probs=rollout_log_probs,
        prompt_text=prompt_text,
        messages=messages,
        include_prompts=config.include_prompts,
        include_token_entropy=config.include_token_entropy,
        extra=extra,
    )
    add_trace_record(task, record)
    return record


def default_actor_entropy(reason: str) -> dict[str, Any]:
    return {"enabled": False, "reason": reason}


def write_trace_sidecar(
    config: AgentTraceConfig,
    *,
    task: str,
    phase: str,
    outer_rollout_id: int,
    records: Iterable[dict[str, Any]],
) -> TraceWriteResult | None:
    if not config.enabled_for_phase(phase):
        return None
    rows = [
        sanitize_for_json(
            _prepare_record_for_write(
                record,
                config=config,
                phase=phase,
                outer_rollout_id=outer_rollout_id,
            )
        )
        for record in records
    ]
    if not rows:
        return None
    task_dir = config.trace_dir / task
    prefix = "rollout" if phase == "train" else "eval"
    suffix = ".jsonl.gz" if config.compression == "gzip" else ".jsonl"
    path = task_dir / f"{prefix}_{outer_rollout_id}{suffix}"
    _write_jsonl_atomic(path, rows, gzip_output=config.compression == "gzip")
    manifest_path = task_dir / "manifest.jsonl"
    append_manifest(
        manifest_path,
        {
            "schema_name": TRACE_SCHEMA_NAME,
            "schema_version": TRACE_SCHEMA_VERSION,
            "task": task,
            "phase": phase,
            "outer_rollout_id": outer_rollout_id,
            "path": str(path),
            "row_count": len(rows),
            "compression": config.compression,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "written",
        },
    )
    return TraceWriteResult(path=path, row_count=len(rows), manifest_path=manifest_path)


def patch_trace_sidecar_actor_entropy(
    config: AgentTraceConfig,
    *,
    task: str,
    phase: str,
    outer_rollout_id: int,
    records: Iterable[dict[str, Any]],
    required: bool = False,
) -> TraceWriteResult | None:
    if not config.enabled_for_phase(phase):
        return None
    entropy_by_key = {
        _record_trace_key(record): sanitize_for_json(record.get("actor_entropy"))
        for record in records
        if _record_trace_key(record) is not None and record.get("actor_entropy") is not None
    }
    if not entropy_by_key:
        if required:
            raise RuntimeError(f"no actor entropy records available for {task} {phase} rollout_id={outer_rollout_id}")
        return None
    task_dir = config.trace_dir / task
    prefix = "rollout" if phase == "train" else "eval"
    suffix = ".jsonl.gz" if config.compression == "gzip" else ".jsonl"
    path = task_dir / f"{prefix}_{outer_rollout_id}{suffix}"
    if not path.exists():
        if required:
            raise FileNotFoundError(f"trace sidecar missing for actor entropy patch: {path}")
        return None
    rows = _read_jsonl(path, gzip_input=config.compression == "gzip")
    patched = 0
    for row in rows:
        entropy = entropy_by_key.get(_record_trace_key(row))
        if entropy is None:
            continue
        row["actor_entropy"] = entropy
        patched += 1
    if required and patched < len(entropy_by_key):
        raise RuntimeError(
            f"actor entropy trace patch incomplete for {task} {phase} rollout_id={outer_rollout_id}: "
            f"patched={patched} records={len(entropy_by_key)}"
        )
    _write_jsonl_atomic(path, rows, gzip_output=config.compression == "gzip")
    manifest_path = task_dir / "manifest.jsonl"
    append_manifest(
        manifest_path,
        {
            "schema_name": TRACE_SCHEMA_NAME,
            "schema_version": TRACE_SCHEMA_VERSION,
            "task": task,
            "phase": phase,
            "outer_rollout_id": outer_rollout_id,
            "path": str(path),
            "row_count": len(rows),
            "patched_actor_entropy_rows": patched,
            "compression": config.compression,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "actor_entropy_patched",
        },
    )
    return TraceWriteResult(path=path, row_count=len(rows), manifest_path=manifest_path)


def append_manifest(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as writer:
        writer.write(json.dumps(sanitize_for_json(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, set):
        return [sanitize_for_json(item) for item in sorted(value, key=str)]
    if hasattr(value, "item"):
        try:
            return sanitize_for_json(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            size = int(getattr(value, "size", 0) or getattr(value, "numel", lambda: 0)())
        except Exception:
            size = 0
        if 0 < size <= 4096:
            try:
                return sanitize_for_json(value.tolist())
            except Exception:
                pass
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"type": type(value).__name__, "shape": sanitize_for_json(list(shape))}
    return str(value)


def _parse_phases(value: Any) -> frozenset[str]:
    raw = str(value or "").replace("rollout", "train")
    phases = {item.strip().lower() for item in raw.split(",") if item.strip()}
    invalid = phases - {"train", "eval"}
    if invalid:
        raise ValueError(f"agent task trace phases must be train/eval, got {sorted(invalid)!r}")
    return frozenset(phases or {"train", "eval"})


def _resolve_trace_dir(args: Any, *, task: str, sample_log_dir: str | Path | None) -> Path | None:
    raw = optional_str(get_arg_or_env(args, "agent_task_trace_dir", "AGENT_TASK_TRACE_DIR", None))
    if raw is not None:
        return Path(raw)
    if sample_log_dir is not None:
        return Path(sample_log_dir).parent / "traces"
    save_dir = optional_str(getattr(args, "save", None))
    if save_dir is not None:
        return Path(save_dir) / "agent_traces"
    return Path("agent_traces")


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]], *, gzip_output: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        if gzip_output:
            with gzip.open(tmp_path, "wt", encoding="utf-8") as writer:
                _write_jsonl_rows(writer, rows)
        else:
            with tmp_path.open("w", encoding="utf-8") as writer:
                _write_jsonl_rows(writer, rows)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_jsonl_rows(writer: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        writer.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _read_jsonl(path: Path, *, gzip_input: bool) -> list[dict[str, Any]]:
    if gzip_input:
        with gzip.open(path, "rt", encoding="utf-8") as reader:
            return [json.loads(line) for line in reader if line.strip()]
    with path.open("r", encoding="utf-8") as reader:
        return [json.loads(line) for line in reader if line.strip()]


def _prepare_record_for_write(
    record: dict[str, Any],
    *,
    config: AgentTraceConfig,
    phase: str,
    outer_rollout_id: int,
) -> dict[str, Any]:
    row = dict(record)
    row["phase"] = phase
    row["outer_rollout_id"] = outer_rollout_id
    if not config.include_prompts:
        for key in _PROMPT_TRACE_KEYS:
            row.pop(key, None)
    if not config.include_token_entropy:
        entropy = row.get("actor_entropy")
        if isinstance(entropy, dict):
            entropy = dict(entropy)
            entropy.pop("actor_entropy_tokens", None)
            entropy.pop("token_entropy", None)
            row["actor_entropy"] = entropy
    return row


def _actor_entropy_from_metadata(metadata: dict[str, Any], *, include_token_entropy: bool) -> dict[str, Any]:
    entropy = metadata.get("actor_entropy")
    if not isinstance(entropy, dict):
        return default_actor_entropy("pending_scoring")
    result = sanitize_for_json(dict(entropy))
    if not include_token_entropy:
        result.pop("actor_entropy_tokens", None)
        result.pop("token_entropy", None)
    return result


def _json_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sample_trace_key(sample: Any) -> tuple[Any, Any, Any, Any] | None:
    metadata = getattr(sample, "metadata", None) or {}
    return _trace_key(
        metadata.get("sample_rollout_id", metadata.get("rollout_id", getattr(sample, "rollout_id", None))),
        metadata.get("traj_uid"),
        metadata.get("turn_idx"),
        metadata.get("sample_index", getattr(sample, "index", None)),
    )


def _record_trace_key(record: dict[str, Any]) -> tuple[Any, Any, Any, Any] | None:
    return _trace_key(
        record.get("sample_rollout_id"), record.get("traj_uid"), record.get("turn_idx"), record.get("sample_index")
    )


def _trace_key(
    sample_rollout_id: Any, traj_uid: Any, turn_idx: Any, sample_index: Any
) -> tuple[Any, Any, Any, Any] | None:
    if sample_rollout_id is None or traj_uid is None or turn_idx is None or sample_index is None:
        return None
    return int(sample_rollout_id), traj_uid, int(turn_idx), int(sample_index)


def _preview_text(text: str | None, *, max_chars: int) -> str | None:
    if text is None:
        return None
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _message_role_summary(messages: list[dict[str, Any]] | None) -> list[str] | None:
    if messages is None:
        return None
    return [str(message.get("role", "")) for message in messages]
