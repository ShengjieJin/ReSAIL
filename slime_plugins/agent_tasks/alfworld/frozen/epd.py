"""Materialize frozen-trajectory teacher targets for EPD.

This module deliberately keeps corpus enumeration, prompt construction, sampling,
validation, and durable output in one small contract surface.  The teacher prompt
is built by the existing SDPO context implementation; this file does not define a
second privileged-prompt template.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Iterable

from slime.algorithms.sdpo.teacher_alignment import (
    DEFAULT_SDPO_MAX_REPROMPT_TOKENS,
    tokenize_sdpo_teacher_prompt,
)
from slime_plugins.agent_tasks.common.algorithms.sdpo_context import _build_own_outcome_trajectory_context

from ..projection import project_response
from .data_source import load_frozen_trajectory_shard, verify_frozen_trajectory

EPD_TARGET_SCHEMA_VERSION = 1
EPD_MANIFEST_SCHEMA_VERSION = 1
EPD_PRIMARY_SEED = 42
EPD_MAX_ATTEMPTS = 3
EPD_MAX_NEW_TOKENS = 1024
EPD_EXPECTED_TRAJECTORIES = 960
EPD_PROMPT_MAX_TOKENS = DEFAULT_SDPO_MAX_REPROMPT_TOKENS
EPD_SHARD_SIZE = 1024

SamplingRequest = Callable[[dict[str, Any], str], Awaitable[dict[str, Any]]]
TeacherOutputValidator = Callable[[str, str, list[int] | tuple[int, ...]], dict[str, Any]]
TeacherContextBuilder = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def build_teacher_prompt_args(*, metadata_profile: str = "alfworld") -> SimpleNamespace:
    """Return the exact controlled own-outcome SDPO prompt configuration."""

    return SimpleNamespace(
        sdpo_context_prompt_style="controlled",
        sdpo_teacher_context_mode="own_outcome",
        sdpo_solution_context_format="trajectory_demo",
        sdpo_no_success_context_mode="failed_negative",
        sdpo_own_outcome_label_mode="explicit",
        sdpo_success_reward_threshold=1.0,
        agent_task_sdpo_metadata_profile=str(metadata_profile),
    )


def canonical_identity(trajectory_uid: str, turn_idx: int) -> str:
    return f"{trajectory_uid}/turn/{int(turn_idx):04d}"


def retry_seed(trajectory_uid: str, turn_idx: int, retry_index: int) -> int:
    """Derive a stable positive 31-bit seed for retry number 1 or 2."""

    if int(retry_index) < 1:
        raise ValueError("retry_index must be >= 1; attempt 1 uses the primary seed")
    digest = hashlib.sha256(f"epd-retry-v1:{trajectory_uid}:{int(turn_idx)}:{int(retry_index)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1) or 1


def build_sampling_params(seed: int = EPD_PRIMARY_SEED) -> dict[str, Any]:
    return {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": EPD_MAX_NEW_TOKENS,
        "sampling_seed": int(seed),
    }


def _message_text(turn: dict[str, Any]) -> str:
    messages = turn.get("messages") or []
    if not messages:
        return ""
    return str(messages[-1].get("content", ""))


def _sdpo_row(
    trajectory: dict[str, Any], turn: dict[str, Any], *, metadata_profile: str = "alfworld"
) -> dict[str, Any]:
    trajectory_uid = str(trajectory["trajectory_uid"])
    turn_idx = int(turn["turn_idx"])
    messages = [dict(message) for message in turn["messages"]]
    return {
        "uid": trajectory_uid,
        "traj_uid": trajectory_uid,
        "turn_idx": turn_idx,
        "task_text": str(trajectory.get("task_description") or trajectory["task_id"]),
        "anchor_obs": str(turn.get("current_observation", "")),
        "next_anchor_obs": str(turn.get("next_observation", "")),
        "projected_action": str(turn.get("frozen_action", "")),
        "is_action_valid": not bool(turn.get("errors")),
        "is_terminal": bool(turn.get("is_terminal", turn_idx == len(trajectory["turns"]) - 1)),
        "episode_rewards": float(trajectory["success"]),
        "episode_lengths": len(trajectory["turns"]),
        "sdpo_current_prompt_text": _message_text(turn),
        "sdpo_current_raw_prompt": messages,
        "algorithm_active_mask": True,
        "sdpo_metadata_profile": str(metadata_profile),
    }


def build_teacher_context(args: Any, trajectory: dict[str, Any], turn: dict[str, Any]) -> dict[str, Any]:
    """Build a teacher request through the existing controlled SDPO context path."""

    metadata_profile = str(getattr(args, "agent_task_sdpo_metadata_profile", "alfworld"))
    rows = [_sdpo_row(trajectory, candidate, metadata_profile=metadata_profile) for candidate in trajectory["turns"]]
    current = next((row for row in rows if int(row["turn_idx"]) == int(turn["turn_idx"])), None)
    if current is None:
        raise ValueError(f"turn_idx={turn.get('turn_idx')} is not present in trajectory")
    planned = _build_own_outcome_trajectory_context(
        args,
        current,
        {(current["uid"], current["traj_uid"]): rows},
    )
    teacher_messages = planned.get("sdpo_teacher_messages")
    teacher_prompt_text = planned.get("sdpo_teacher_prompt_text")
    if not isinstance(teacher_messages, list) or not teacher_messages:
        raise ValueError("controlled SDPO context did not return teacher messages")
    if not isinstance(teacher_prompt_text, str) or not teacher_prompt_text:
        raise ValueError("controlled SDPO context did not return teacher prompt text")
    return {
        "teacher_messages": teacher_messages,
        "teacher_prompt_text": teacher_prompt_text,
        "teacher_signal_type": planned.get("sdpo_teacher_signal_type"),
        "teacher_reference_scope": planned.get("sdpo_context_reference_scope"),
        "outcome": str(trajectory["outcome"]),
        "success": bool(trajectory["success"]),
        "trajectory_uid": str(trajectory["trajectory_uid"]),
        "turn_idx": int(turn["turn_idx"]),
        "prompt_builder": "sdpo_context._build_own_outcome_trajectory_context",
        "context_mode": "own_outcome",
        "context_prompt_style": "controlled",
    }


def canonical_steps_from_trajectories(
    trajectories: Iterable[dict[str, Any]],
    *,
    expected_trajectories: int | None = None,
    expected_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Expand every frozen turn, preserving both successful and failed trajectories."""

    values = list(trajectories)
    if expected_trajectories is not None and len(values) != int(expected_trajectories):
        raise ValueError(f"expected {expected_trajectories} trajectories, found {len(values)}")
    seen: set[str] = set()
    steps: list[dict[str, Any]] = []
    for trajectory in values:
        verify_frozen_trajectory(trajectory)
        uid = str(trajectory["trajectory_uid"])
        if uid in seen:
            raise ValueError(f"duplicate trajectory_uid: {uid}")
        seen.add(uid)
        for turn in trajectory["turns"]:
            turn_idx = int(turn["turn_idx"])
            steps.append(
                {
                    "identity": canonical_identity(uid, turn_idx),
                    "trajectory_uid": uid,
                    "turn_idx": turn_idx,
                    "trajectory": trajectory,
                    "turn": turn,
                    "success": bool(trajectory["success"]),
                    "outcome": str(trajectory["outcome"]),
                    "split": str(trajectory["split"]),
                    "task_id": str(trajectory["task_id"]),
                }
            )
    steps.sort(key=lambda item: (item["trajectory_uid"], item["turn_idx"]))
    for canonical_index, step in enumerate(steps):
        step["canonical_index"] = canonical_index
    if expected_steps is not None and len(steps) != int(expected_steps):
        raise ValueError(f"expected {expected_steps} canonical turns, found {len(steps)}")
    return steps


def canonical_steps_from_corpus(
    corpus_dir: str | Path,
    *,
    expected_trajectories: int = EPD_EXPECTED_TRAJECTORIES,
    expected_steps: int | None = None,
) -> list[dict[str, Any]]:
    root = Path(corpus_dir)
    shard_paths = sorted(root.glob("batch_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"no batch_*.pt shards found under {root}")
    for shard_idx, path in enumerate(shard_paths):
        expected_name = f"batch_{shard_idx:03d}.pt"
        if path.name != expected_name:
            raise ValueError(f"frozen corpus shard sequence has a gap: expected {expected_name}, got {path.name}")
    trajectories = []
    source_locations: dict[str, tuple[str, int]] = {}
    for path in shard_paths:
        shard_trajectories = load_frozen_trajectory_shard(path)
        for trajectory_index, trajectory in enumerate(shard_trajectories):
            trajectories.append(trajectory)
            source_locations[str(trajectory["trajectory_uid"])] = (path.name, trajectory_index)
    steps = canonical_steps_from_trajectories(
        trajectories,
        expected_trajectories=expected_trajectories,
        expected_steps=expected_steps,
    )
    for step in steps:
        step["source_shard"], step["source_trajectory_index"] = source_locations[step["trajectory_uid"]]
    return steps


def validate_teacher_output(
    response_text: str,
    finish_type: str,
    response_ids: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    response_text = str(response_text or "")
    response_ids = [int(token_id) for token_id in response_ids]
    projection = project_response(response_text)
    reasons: list[str] = []
    if not response_text.strip():
        reasons.append("empty_response")
    if not response_ids:
        reasons.append("empty_response_ids")
    if str(finish_type) != "stop":
        reasons.append(f"finish_reason_{finish_type or 'missing'}")
    if not projection.format_valid:
        reasons.append(projection.invalid_reason or "invalid_action_format")
    if not projection.projected_action:
        reasons.append("empty_action")
    return {
        "valid": not reasons,
        "truncated": str(finish_type) == "length",
        "format_valid": bool(projection.format_valid),
        "projected_action": projection.projected_action,
        "invalid_reason": ",".join(reasons) if reasons else None,
        "response_token_count": len(response_ids),
    }


def _finish_type(meta_info: dict[str, Any]) -> str:
    finish_reason = meta_info.get("finish_reason") or {}
    return str(finish_reason.get("type", "")) if isinstance(finish_reason, dict) else str(finish_reason)


def parse_generation_output(output: dict[str, Any]) -> tuple[str, str, list[int], list[float] | None]:
    if not isinstance(output, dict):
        raise ValueError("generation output must be an object")
    meta_info = dict(output.get("meta_info") or {})
    values = meta_info.get("output_token_logprobs")
    response_ids: list[int] = []
    response_log_probs: list[float] | None = None
    if values:
        response_log_probs = []
        for item in values:
            if isinstance(item, dict):
                logprob = item.get("logprob", item.get("log_prob"))
                token_id = item.get("token_id", item.get("id"))
            else:
                if len(item) < 2:
                    raise ValueError("output_token_logprobs entries must contain logprob and token id")
                logprob, token_id = item[0], item[1]
            if logprob is None or token_id is None:
                raise ValueError("output_token_logprobs entry is missing logprob or token id")
            response_log_probs.append(float(logprob))
            response_ids.append(int(token_id))
    else:
        raw_ids = meta_info.get("output_ids", output.get("output_ids"))
        if raw_ids:
            response_ids = [int(token_id) for token_id in raw_ids]
    if response_log_probs is not None and len(response_log_probs) != len(response_ids):
        raise ValueError("sampled logprob count does not match response token count")
    return str(output.get("text", "")), _finish_type(meta_info), response_ids, response_log_probs


def _normalize_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = value.get("input_ids")
        if hasattr(value, "tolist"):
            value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError("tokenizer did not return a token-id sequence")
    return [int(token_id) for token_id in value]


def _teacher_prompt_ids(
    tokenizer: Any,
    teacher_messages: list[dict[str, Any]],
    fallback: list[int],
    *,
    max_prompt_tokens: int | None = None,
    truncation_side: str = "right",
) -> list[int]:
    if tokenizer is None:
        return list(fallback)
    # Historical materializers omit max_prompt_tokens and retain their hard-fail
    # contract. New callers may opt into the same explicit truncation contract
    # used by online SDPO teacher alignment.
    raw_ids = tokenize_sdpo_teacher_prompt(
        tokenizer,
        "",
        messages=teacher_messages,
        apply_chat_template_kwargs={"enable_thinking": False},
        max_prompt_tokens=max_prompt_tokens,
        truncation_side=truncation_side,
    )
    if max_prompt_tokens is None and len(raw_ids) > EPD_PROMPT_MAX_TOKENS:
        raise ValueError(
            f"teacher prompt exceeds DEFAULT_SDPO_MAX_REPROMPT_TOKENS={EPD_PROMPT_MAX_TOKENS}: {len(raw_ids)}"
        )
    return raw_ids


def _record_progress_repair(path: Path, *, offset: int, size: int, reason: str) -> None:
    """Record a repaired trailing progress write without making resume depend on it."""

    _atomic_json(
        path.with_name("progress_repair.json"),
        {
            "path": path.name,
            "removed_bytes": size - offset,
            "repair_offset": offset,
            "reason": reason,
        },
    )


def _read_jsonl_records(path: Path, *, repair_trailing_progress: bool) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("rb") as handle:
        while True:
            line_offset = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            try:
                line = raw_line.decode("utf-8")
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record is not a JSON object")
                identity = str(value.get("identity", ""))
                if not identity:
                    raise ValueError("missing identity")
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                if not repair_trailing_progress:
                    raise ValueError(f"malformed committed EPD record in {path}") from exc
                remainder = handle.read()
                if remainder.strip():
                    raise ValueError(f"malformed middle EPD progress record in {path}") from exc
                file_size = line_offset + len(raw_line) + len(remainder)
                fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
                temporary_path = Path(temporary_name)
                os.close(fd)
                try:
                    with path.open("rb") as source, temporary_path.open("wb") as destination:
                        remaining = line_offset
                        while remaining:
                            chunk = source.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise OSError(f"progress file changed while repairing {path}")
                            destination.write(chunk)
                            remaining -= len(chunk)
                        destination.flush()
                        os.fsync(destination.fileno())
                    temporary_path.chmod(0o644)
                    os.replace(temporary_path, path)
                finally:
                    temporary_path.unlink(missing_ok=True)
                _record_progress_repair(path, offset=line_offset, size=file_size, reason=str(exc))
                break
            prior = records.get(identity)
            if prior is not None and prior != value:
                raise ValueError(f"conflicting duplicate EPD target identity: {identity}")
            records[identity] = value
    return records


def _read_existing_records(output_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    paths = [
        (output_dir / "progress.jsonl", True),
        *[(path, False) for path in sorted(output_dir.glob("shard_*.jsonl"))],
    ]
    for path, repair_trailing_progress in paths:
        existing = _read_jsonl_records(path, repair_trailing_progress=repair_trailing_progress)
        for identity, value in existing.items():
            prior = records.get(identity)
            if prior is not None and prior != value:
                raise ValueError(f"conflicting duplicate EPD target identity: {identity}")
            records[identity] = value
    return records


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _record_is_complete(value: dict[str, Any]) -> bool:
    return (
        bool(value.get("identity"))
        and bool(value.get("response_ids"))
        and isinstance(value.get("validation", {}).get("valid"), bool)
    )


async def materialize_records(
    steps: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    endpoints: list[str],
    tokenizer: Any,
    generate_request: SamplingRequest,
    expected_count: int | None = None,
    worker_count: int = 8,
    global_concurrency: int | None = None,
    shard_size: int = EPD_SHARD_SIZE,
    config: dict[str, Any] | None = None,
    max_attempts: int = EPD_MAX_ATTEMPTS,
    direct_binding: dict[str, Any] | None = None,
    metadata_profile: str = "alfworld",
    teacher_output_validator: TeacherOutputValidator = validate_teacher_output,
    sampling_params_overrides: dict[str, Any] | None = None,
    teacher_prompt_max_tokens: int | None = None,
    teacher_prompt_truncation_side: str = "right",
    teacher_context_builder: TeacherContextBuilder | None = None,
    teacher_context_contract: dict[str, str] | None = None,
    attempt_seed_offset: int = 0,
) -> list[dict[str, Any]]:
    """Materialize independent requests with bounded global concurrency and resume support."""

    if not endpoints:
        raise ValueError("at least one EPD generation endpoint is required")
    if worker_count != len(endpoints):
        raise ValueError("worker_count is the engine count and must equal the endpoint count")
    if max_attempts != EPD_MAX_ATTEMPTS:
        raise ValueError(f"EPD requires max_attempts={EPD_MAX_ATTEMPTS}")
    if int(attempt_seed_offset) < 0:
        raise ValueError("EPD attempt_seed_offset must be nonnegative")
    if teacher_prompt_max_tokens is not None and int(teacher_prompt_max_tokens) <= 0:
        raise ValueError("teacher_prompt_max_tokens must be positive when set")
    if teacher_prompt_truncation_side not in {"left", "right"}:
        raise ValueError("teacher_prompt_truncation_side must be left or right")
    context_contract = {
        "teacher_prompt_builder": "sdpo_context._build_own_outcome_trajectory_context",
        "teacher_context_mode": "own_outcome",
        "teacher_context_prompt_style": "controlled",
    }
    if teacher_context_contract is not None:
        required_context = set(context_contract)
        if set(teacher_context_contract) != required_context or any(
            not str(teacher_context_contract[key]).strip() for key in required_context
        ):
            raise ValueError("EPD teacher_context_contract must define the complete non-empty context contract")
        context_contract = {key: str(teacher_context_contract[key]) for key in context_contract}
    if not isinstance(direct_binding, dict):
        raise ValueError("EPD requires a direct_binding mapping")
    required_binding = {"model_path", "corpus_dir", "teacher_iteration", "materialization_identity"}
    if not required_binding.issubset(direct_binding) or any(
        not str(direct_binding[key]).strip() for key in required_binding
    ):
        raise ValueError("EPD direct binding is incomplete")
    direct_binding = {key: direct_binding[key] for key in sorted(direct_binding)}
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    failure_marker = root / "failure.json"
    if failure_marker.exists():
        try:
            previous_failure = json.loads(failure_marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"EPD materialization failure marker is unreadable: {failure_marker}") from exc
        _append_jsonl(root / "failure_history.jsonl", previous_failure)
        failure_marker.unlink()
    expected_count = len(steps) if expected_count is None else int(expected_count)
    canonical = []
    for raw_step in steps:
        step = dict(raw_step)
        step.setdefault("identity", canonical_identity(str(step["trajectory_uid"]), int(step["turn_idx"])))
        trajectory = step.get("trajectory") or {}
        step.setdefault("task_id", str(trajectory.get("task_id", "")))
        step.setdefault("split", str(trajectory.get("split", "")))
        step.setdefault("success", bool(trajectory.get("success", False)))
        step.setdefault("outcome", str(trajectory.get("outcome", "failure")))
        canonical.append(step)
    canonical.sort(key=lambda item: (str(item["trajectory_uid"]), int(item["turn_idx"])))
    for canonical_index, step in enumerate(canonical):
        step.setdefault("canonical_index", canonical_index)
    if len(canonical) != expected_count:
        raise ValueError(f"expected {expected_count} materialization steps, found {len(canonical)}")
    expected_by_identity = {str(step["identity"]): step for step in canonical}
    if len(expected_by_identity) != len(canonical):
        raise ValueError("duplicate canonical EPD identity in input steps")
    existing = _read_existing_records(root)
    unknown = set(existing) - set(expected_by_identity)
    if unknown:
        raise ValueError(f"existing EPD output contains unknown identities: {sorted(unknown)[:3]}")
    for identity, value in existing.items():
        step = expected_by_identity[identity]
        expected_prompt_ids = [int(token_id) for token_id in step["turn"]["prompt_ids"]]
        if value.get("canonical_index") != step["canonical_index"]:
            raise ValueError(f"existing EPD target canonical_index mismatch: {identity}")
        if value.get("trajectory_uid") != step["trajectory_uid"] or int(value.get("turn_idx", -1)) != int(
            step["turn_idx"]
        ):
            raise ValueError(f"existing EPD target identity fields mismatch: {identity}")
        if value.get("original_prompt_ids") != expected_prompt_ids:
            raise ValueError(f"existing EPD target original prompt_ids mismatch: {identity}")
        expected_bindings = (("binding_mode", "direct_v1"), ("direct_binding", direct_binding))
        for key, expected in expected_bindings:
            if value.get(key) != expected:
                raise ValueError(f"existing EPD target {key} mismatch: {identity}")
    completed = {identity: value for identity, value in existing.items() if _record_is_complete(value)}
    pending = [step for step in canonical if str(step["identity"]) not in completed]
    progress_path = root / "progress.jsonl"
    engine_count = len(endpoints)
    global_concurrency = int(global_concurrency or engine_count * 64)
    if global_concurrency < engine_count:
        raise ValueError("global_concurrency must be at least the engine count")
    queue: asyncio.Queue[tuple[dict[str, Any], str] | None] = asyncio.Queue(maxsize=global_concurrency)
    records = dict(completed)
    record_lock = asyncio.Lock()
    stats_lock = asyncio.Lock()
    stop_event = asyncio.Event()
    fatal_error: list[BaseException] = []
    in_flight = 0
    observed_max_in_flight = 0
    in_flight_by_endpoint = {endpoint: 0 for endpoint in endpoints}
    observed_max_by_endpoint = {endpoint: 0 for endpoint in endpoints}
    request_count_by_endpoint = {endpoint: 0 for endpoint in endpoints}
    valid_target_count_by_endpoint = {endpoint: 0 for endpoint in endpoints}
    invalid_target_count_by_endpoint = {endpoint: 0 for endpoint in endpoints}
    for value in completed.values():
        endpoint = value.get("generation_endpoint")
        if endpoint in request_count_by_endpoint:
            if value.get("validation", {}).get("valid") is True:
                valid_target_count_by_endpoint[endpoint] += 1
            else:
                invalid_target_count_by_endpoint[endpoint] += 1
            request_count_by_endpoint[endpoint] += int(value.get("attempt_count", 1))
            # Historical progress has no live in-flight sample to observe; the
            # nonzero sentinel records that this engine did participate.
            observed_max_by_endpoint[endpoint] = max(observed_max_by_endpoint[endpoint], 1)
    if completed:
        observed_max_in_flight = 1

    async def request(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal in_flight, observed_max_in_flight
        async with stats_lock:
            in_flight += 1
            observed_max_in_flight = max(observed_max_in_flight, in_flight)
            in_flight_by_endpoint[endpoint] += 1
            observed_max_by_endpoint[endpoint] = max(
                observed_max_by_endpoint[endpoint], in_flight_by_endpoint[endpoint]
            )
            request_count_by_endpoint[endpoint] += 1
        try:
            return await generate_request(payload, endpoint)
        finally:
            async with stats_lock:
                in_flight -= 1
                in_flight_by_endpoint[endpoint] -= 1

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                if stop_event.is_set():
                    return
                step, endpoint = item
                uid, turn_idx = str(step["trajectory_uid"]), int(step["turn_idx"])
                trajectory, turn = step["trajectory"], step["turn"]
                context = (
                    teacher_context_builder(trajectory, turn)
                    if teacher_context_builder is not None
                    else build_teacher_context(
                        build_teacher_prompt_args(metadata_profile=metadata_profile), trajectory, turn
                    )
                )
                original_prompt_ids = [int(token_id) for token_id in turn["prompt_ids"]]
                teacher_prompt_ids = _teacher_prompt_ids(
                    tokenizer,
                    context["teacher_messages"],
                    original_prompt_ids,
                    max_prompt_tokens=teacher_prompt_max_tokens,
                    truncation_side=teacher_prompt_truncation_side,
                )
                if not teacher_prompt_ids:
                    raise ValueError(f"empty teacher prompt tokenization for {step['identity']}")
                attempts: list[dict[str, Any]] = []
                record = None
                selected_attempt = None
                for attempt_number in range(1, max_attempts + 1):
                    retry_index = int(attempt_seed_offset) + attempt_number - 1
                    seed = (
                        EPD_PRIMARY_SEED
                        if retry_index == 0
                        else retry_seed(uid, turn_idx, retry_index)
                    )
                    request_sampling_params = build_sampling_params(seed)
                    request_sampling_params.update(sampling_params_overrides or {})
                    payload = {
                        "input_ids": teacher_prompt_ids,
                        "sampling_params": request_sampling_params,
                        "return_logprob": True,
                        # These two fields let test/in-process adapters correlate
                        # requests; the host HTTP adapter strips them before POST.
                        "identity": str(step["identity"]),
                        "turn_idx": turn_idx,
                    }
                    try:
                        output = await request(endpoint, payload)
                    except Exception as exc:  # transport failures are retryable and are recorded separately
                        attempts.append(
                            {
                                "attempt": attempt_number,
                                "seed": seed,
                                "failure_class": "transport",
                                "valid": False,
                                "invalid_reason": f"request_error:{type(exc).__name__}:{exc}",
                            }
                        )
                        continue
                    try:
                        text, finish_type, response_ids, response_log_probs = parse_generation_output(output)
                        validation = teacher_output_validator(text, finish_type, response_ids)
                        attempt = {
                            "attempt": attempt_number,
                            "seed": seed,
                            "failure_class": "semantic" if not validation["valid"] else None,
                            "finish_reason": finish_type,
                            "response_token_count": len(response_ids),
                            "truncated": validation["truncated"],
                            "valid": validation["valid"],
                            "invalid_reason": validation["invalid_reason"],
                        }
                        attempts.append(attempt)
                        # Any non-empty sampled token sequence is a trainable target.
                        # Keep replacing the candidate so that, if all three attempts
                        # are format-invalid, the final available generation is frozen
                        # and trained with the same weight as every other EPD target.
                        if response_ids:
                            record = {
                                "schema_version": EPD_TARGET_SCHEMA_VERSION,
                                "identity": str(step["identity"]),
                                "canonical_index": int(step["canonical_index"]),
                                "trajectory_uid": uid,
                                "turn_idx": turn_idx,
                                "source": {
                                    "task_id": str(step["task_id"]),
                                    "split": str(step["split"]),
                                    "success": bool(step["success"]),
                                    "outcome": str(step["outcome"]),
                                    "shard": step.get("source_shard"),
                                    "trajectory_index": step.get("source_trajectory_index"),
                                },
                                "original_prompt_ids": original_prompt_ids,
                                "teacher_prompt_ids": teacher_prompt_ids,
                                "teacher_prompt_builder": context["prompt_builder"],
                                "teacher_context_mode": context["context_mode"],
                                "teacher_context_prompt_style": context["context_prompt_style"],
                                "teacher_signal_type": context["teacher_signal_type"],
                                "teacher_reference_scope": context["teacher_reference_scope"],
                                "response_text": text,
                                "response_ids": response_ids,
                                "response_log_probs": response_log_probs,
                                "finish_reason": finish_type,
                                "truncated": bool(validation["truncated"]),
                                "projected_action": validation["projected_action"],
                                "action_valid": bool(validation["format_valid"]),
                                "validation": validation,
                                "primary_seed": EPD_PRIMARY_SEED,
                                "generation_endpoint": endpoint,
                            }
                            if "outcome" in context:
                                record["teacher_outcome"] = context["outcome"]
                            if "success" in context:
                                record["teacher_success"] = context["success"]
                            record.update({"binding_mode": "direct_v1", "direct_binding": direct_binding})
                            selected_attempt = attempt_number
                        if validation["valid"] and record is not None:
                            break
                    except Exception as exc:  # retries are part of the durable scientific contract
                        attempts.append(
                            {
                                "attempt": attempt_number,
                                "seed": seed,
                                "failure_class": "semantic",
                                "valid": False,
                                "invalid_reason": f"output_error:{type(exc).__name__}:{exc}",
                            }
                        )
                if record is None:
                    raise RuntimeError(
                        f"EPD teacher target failed after {max_attempts} attempts with no trainable "
                        f"response_ids: {step['identity']}"
                    )
                record["attempt_count"] = len(attempts)
                record["retry_count"] = max(0, len(attempts) - 1)
                record["selected_attempt"] = int(selected_attempt)
                record["attempts"] = attempts
                async with record_lock:
                    prior = records.get(record["identity"])
                    if prior is not None and prior != record:
                        raise ValueError(f"conflicting duplicate generated identity: {record['identity']}")
                    if prior is None:
                        records[record["identity"]] = record
                        async with stats_lock:
                            if record["validation"]["valid"] is True:
                                valid_target_count_by_endpoint[endpoint] += 1
                            else:
                                invalid_target_count_by_endpoint[endpoint] += 1
                        _append_jsonl(progress_path, record)
            except Exception as exc:
                if not fatal_error:
                    fatal_error.append(exc)
                stop_event.set()
                return
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(global_concurrency)]

    async def producer() -> None:
        for pending_index, step in enumerate(pending):
            endpoint = endpoints[pending_index % engine_count]
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(queue.put((step, endpoint)), timeout=0.2)
                    break
                except asyncio.TimeoutError:
                    continue
            if stop_event.is_set():
                raise fatal_error[0] if fatal_error else RuntimeError("EPD worker stopped")
        for _ in workers:
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(queue.put(None), timeout=0.2)
                    break
                except asyncio.TimeoutError:
                    continue
            if stop_event.is_set():
                raise fatal_error[0] if fatal_error else RuntimeError("EPD worker stopped")

    try:
        await asyncio.gather(producer(), *workers)
        if fatal_error:
            raise fatal_error[0]
    except Exception as exc:
        stop_event.set()
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        _atomic_json(
            failure_marker,
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "target_count": expected_count},
        )
        raise

    if len(records) != expected_count:
        raise RuntimeError(f"EPD materialization completed with {len(records)}/{expected_count} records")
    ordered = sorted(records.values(), key=lambda item: (item["trajectory_uid"], int(item["turn_idx"])))
    shards = []
    target_index: dict[str, dict[str, Any]] = {}
    for shard_index, start in enumerate(range(0, len(ordered), shard_size)):
        shard_records = ordered[start : start + shard_size]
        shard_name = f"shard_{shard_index:04d}.jsonl"
        shard_path = root / shard_name
        fd, temporary_name = tempfile.mkstemp(prefix=f".{shard_name}.", suffix=".tmp", dir=root)
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                for line_number, value in enumerate(shard_records):
                    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                    target_index[value["identity"]] = {
                        "shard": shard_name,
                        "line": line_number,
                        "canonical_index": value["canonical_index"],
                    }
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.chmod(0o644)
            os.replace(temporary_path, shard_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        shard = {
            "name": shard_name,
            "count": len(shard_records),
            "first_identity": shard_records[0]["identity"],
            "last_identity": shard_records[-1]["identity"],
        }
        shards.append(shard)
    _atomic_json(root / "index.json", target_index)
    config = config or {}
    valid_target_count = sum(valid_target_count_by_endpoint.values())
    invalid_target_count = sum(invalid_target_count_by_endpoint.values())
    if valid_target_count + invalid_target_count != expected_count:
        raise RuntimeError("EPD valid/invalid target accounting does not match the materialized target count")
    manifest = {
        "schema_version": EPD_MANIFEST_SCHEMA_VERSION,
        "kind": "epd_teacher_targets",
        "status": "complete",
        "target_count": expected_count,
        "trajectory_count": len({step["trajectory_uid"] for step in canonical}),
        "canonical_turn_count": len(canonical),
        "shard_size": shard_size,
        "records_format": "jsonl",
        "sampling_params": build_sampling_params(EPD_PRIMARY_SEED),
        "seed": EPD_PRIMARY_SEED,
        "primary_seed": EPD_PRIMARY_SEED,
        "max_attempts": EPD_MAX_ATTEMPTS,
        "success_filter": False,
        "branch_packing": False,
        "student_visible_prefix": "original",
        **context_contract,
        "config": config,
        "index_file": "index.json",
        "engine_count": engine_count,
        "global_concurrency": global_concurrency,
        "observed_max_inflight": observed_max_in_flight,
        "observed_max_inflight_by_endpoint": observed_max_by_endpoint,
        "request_count_by_endpoint": request_count_by_endpoint,
        "valid_target_count_by_endpoint": valid_target_count_by_endpoint,
        "invalid_target_count_by_endpoint": invalid_target_count_by_endpoint,
        "valid_target_count": valid_target_count,
        "invalid_target_count": invalid_target_count,
        "invalid_target_fraction": invalid_target_count / expected_count if expected_count else 0.0,
        "shards": shards,
    }
    manifest.update({"binding_mode": "direct_v1", "direct_binding": direct_binding})
    _atomic_json(root / "manifest.json", manifest)
    return ordered


def load_target_records(output_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for shard in manifest.get("shards", []):
        with (root / shard["name"]).open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records
