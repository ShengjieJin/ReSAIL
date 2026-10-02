from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from contextlib import suppress
from typing import Any

from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.history import TextHistory, assistant_history_content, format_chat_messages
from slime_plugins.agent_tasks.common.segments import backfill_terminal_rewards, build_step_segment_sample
from slime_plugins.agent_tasks.common.trace import (
    add_sample_trace_record,
    compact_prompt_metadata,
    get_agent_trace_config,
)

from .config import AlfWorldConfig, get_alfworld_config
from .envs import build_alfworld_env, get_or_create_alfworld_env_pool
from .frozen.contracts import student_prompt_privilege_violation
from .frozen.sampling import collection_turn_seed
from .projection import project_response
from .prompts import build_messages, extract_task_description
from .runtime import assert_alfworld_live_environment_allowed

logger = logging.getLogger(__name__)


async def generate(args, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> list[Sample]:
    assert_alfworld_live_environment_allowed(args)
    config = get_alfworld_config(args, evaluation=evaluation)
    if evaluation:
        split = config.eval_split
    else:
        split = str(sample.metadata.get("split") or config.train_split)
    rows = await run_alfworld_episode(
        args,
        sample=sample,
        sampling_params=sampling_params,
        config=config,
        split=split,
        evaluation=evaluation,
    )
    if not rows:
        raise RuntimeError("ALFWorld custom_generate produced no step samples")
    return rows


async def run_alfworld_episode(
    args: Any,
    *,
    sample: Sample,
    sampling_params: dict[str, Any],
    config: AlfWorldConfig,
    split: str,
    evaluation: bool = False,
) -> list[Sample]:
    tokenizer = _get_tokenizer(args)
    env, pool = await _acquire_env(args, config=config, split=split, seed=_seed(sample))
    session_id = str(sample.metadata.get("traj_uid") or uuid.uuid4())
    rollout_id = int(sample.index if sample.index is not None else _seed(sample))
    history = TextHistory(max_length=config.history_max_steps)
    rows: list[Sample] = []
    trace_config = get_agent_trace_config(args, task="alfworld", sample_log_dir=config.sample_log_dir)
    trace_phase = "eval" if evaluation else "train"
    trace_enabled = trace_config.enabled_for_phase(trace_phase)
    trace_entries: list[dict[str, Any]] = []
    done = False
    success = False
    episode_reward = 0.0
    episode_start = time.monotonic()

    try:
        reset_start = time.monotonic()
        reset_result = await asyncio.to_thread(env.reset, _seed(sample))
        reset_seconds = float(getattr(reset_result, "reset_seconds", time.monotonic() - reset_start))
        observation = str(getattr(reset_result, "observation"))
        info = dict(getattr(reset_result, "info"))
        worker_id = str(getattr(reset_result, "worker_id", "unknown"))
        worker_reused = bool(getattr(reset_result, "reused_worker", False))
        worker_created = bool(getattr(reset_result, "worker_created", not worker_reused))
        task_description = extract_task_description(observation)
        gamefile = str(_info_value(info, "extra.gamefile", _info_value(info, "gamefile", "")) or "")
        for turn_idx in range(config.max_episode_steps):
            admissible_actions = _admissible_actions(info)
            messages, raw_prompt, rendered_prompt, prompt_ids, prompt_budget_metadata = _build_turn_prompt(
                args,
                tokenizer,
                config=config,
                current_observation=observation,
                admissible_actions=admissible_actions,
                turn_idx=turn_idx,
                task_description=task_description,
                history=history,
            )
            if _stream_bound_sampling_enabled(args):
                prompt_text = "\n".join(str(message.get("content", "")) for message in messages)
                if student_prompt_privilege_violation(prompt_text):
                    raise RuntimeError("ALFWorld collection student prompt contains privileged teacher text")

            gen_start = time.monotonic()
            turn_sampling_params = dict(sampling_params)
            if evaluation:
                turn_sampling_params["sampling_seed"] = _eval_turn_sampling_seed(
                    args,
                    split=split,
                    rollout_id=rollout_id,
                    uid=str(sample.metadata.get("uid")),
                    turn_idx=turn_idx,
                )
            elif _stream_bound_sampling_enabled(args):
                turn_sampling_params["sampling_seed"] = _collection_turn_sampling_seed(
                    args,
                    stream_index=int((sample.metadata or {}).get("sample_group_index", -1)),
                    turn_idx=turn_idx,
                )
            output = await _generate_step(
                args,
                prompt_ids=prompt_ids,
                rendered_prompt=rendered_prompt,
                sampling_params=turn_sampling_params,
                session_id=session_id,
            )
            generation_seconds = time.monotonic() - gen_start
            response_text = str(output.get("text", ""))
            meta_info = dict(output.get("meta_info") or {})
            response_ids, response_log_probs = _response_tokens_and_log_probs(meta_info)
            if not response_ids:
                raise RuntimeError("ALFWorld generation returned no response tokens")
            finish_type = _finish_type(meta_info)
            _validate_finish_type(finish_type)
            projection = project_response(response_text)
            action = projection.projected_action
            admissible_action_set = {item.lower() for item in admissible_actions}
            sdpo_action_valid = bool(projection.format_valid and action in admissible_action_set)
            step_start = time.monotonic()
            env_step_error = None
            try:
                next_observation, score, done, next_info = await asyncio.to_thread(env.step, action)
                env_step_seconds = time.monotonic() - step_start
                next_observation = str(next_observation)
                info = dict(next_info)
                success = bool(_info_value(info, "won", False) or score > 0)
                episode_reward = config.success_reward * float(success)
            except Exception as exc:
                env_step_seconds = time.monotonic() - step_start
                env_step_error = exc
                next_observation = observation
                score = 0.0
                done = True
                info = dict(info)
                success = False
                episode_reward = 0.0
                logger.warning(
                    "Recovered ALFWorld env.step failure: uid=%s rollout_id=%s turn_idx=%s worker_id=%s action=%r",
                    sample.metadata.get("uid"),
                    rollout_id,
                    turn_idx,
                    worker_id,
                    action,
                    exc_info=True,
                )

            metadata = {
                **(sample.metadata or {}),
                **prompt_budget_metadata,
                "uid": sample.metadata.get("uid"),
                "traj_uid": sample.metadata.get("traj_uid", f"{sample.metadata.get('uid', 'alfworld')}-traj"),
                "agent_task": "alfworld",
                "agent_task_trace_dir": str(trace_config.trace_dir) if trace_config.trace_dir is not None else None,
                "turn_idx": turn_idx,
                "rollout_id": rollout_id,
                "seed": _seed(sample),
                "split": split,
                "session_id": session_id,
                "task_description": task_description,
                "runtime_task_identity": {
                    "task_description": task_description,
                    "gamefile": gamefile,
                },
                "current_observation": observation,
                "next_observation": next_observation,
                **compact_prompt_metadata(
                    args,
                    raw_prompt=raw_prompt,
                    rendered_prompt=rendered_prompt,
                    messages=messages,
                ),
                "format_valid": projection.format_valid,
                "missing_action_tag": projection.missing_action_tag,
                "missing_thinking_tag": projection.missing_thinking_tag,
                "contains_chinese": projection.contains_chinese,
                "invalid_reason": projection.invalid_reason,
                "projected_action": projection.projected_action,
                "admissible_member": action in admissible_action_set,
                "admissible_actions_preview": admissible_actions[:20],
                "finish_reason": finish_type,
                "prompt_tokens": len(prompt_ids),
                "response_tokens": len(response_ids),
                "sampling_seed": turn_sampling_params.get("sampling_seed"),
                "cached_tokens": int(meta_info.get("cached_tokens", 0) or 0),
                "prefix_cache_hit_rate": (
                    (int(meta_info.get("cached_tokens", 0) or 0) / len(prompt_ids)) if prompt_ids else 0.0
                ),
                "generation_seconds": generation_seconds,
                "generation_request_seconds": generation_seconds,
                "generation_url": _generation_url(args),
                "worker_id": worker_id,
                "worker_reused": worker_reused,
                "worker_created": worker_created,
                "reset_seconds": reset_seconds,
                "env_step_seconds": env_step_seconds,
                "score": float(score),
                "won": bool(success),
                "episode_error": 0.0,
                "episode_error_count": 0.0,
                "env_step_failed": False,
            }
            if env_step_error is not None:
                metadata.update(
                    {
                        "episode_error": 1.0,
                        "episode_error_count": 1.0,
                        "episode_error_type": type(env_step_error).__name__,
                        "episode_error_stage": "env_step",
                        "episode_error_message": _short_error_message(env_step_error),
                        "env_step_failed": True,
                    }
                )
            sample_index = _segment_sample_index(sample, turn_idx, config.max_episode_steps)
            metadata.update(
                {
                    "sample_index": sample_index,
                    "sample_rollout_id": rollout_id,
                    "group_index": sample.group_index,
                }
            )
            extra_train_metadata = None
            sdpo_metadata_enabled = _sdpo_training_metadata_enabled(args, evaluation=evaluation)
            grpo_context_enabled = not evaluation and bool(getattr(args, "grpo_token_weights", False))
            if sdpo_metadata_enabled or grpo_context_enabled:
                context_metadata = {
                    "uid": metadata["uid"],
                    "traj_uid": metadata["traj_uid"],
                    "turn_idx": turn_idx,
                    "task_text": task_description,
                    "anchor_obs": observation,
                    "next_anchor_obs": next_observation,
                    "projected_action": projection.projected_action,
                    "is_action_valid": sdpo_action_valid,
                    "is_terminal": False,
                    "episode_rewards": 0.0,
                    "episode_lengths": 0,
                    "sdpo_current_prompt_text": _sdpo_current_prompt_text(messages, raw_prompt),
                    "sdpo_current_raw_prompt": messages,
                    "sdpo_metadata_profile": "alfworld",
                }
                extra_train_metadata = {}
                if sdpo_metadata_enabled:
                    extra_train_metadata["sdpo"] = dict(context_metadata)
                if grpo_context_enabled:
                    extra_train_metadata["grpo_token_weight_context"] = dict(context_metadata)
            row = build_step_segment_sample(
                base_sample=sample,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                rollout_log_probs=response_log_probs,
                reward=0.0,
                sample_index=sample_index,
                rollout_id=rollout_id,
                metadata=metadata,
                response_text=response_text,
                session_id=session_id,
                non_generation_time=reset_seconds + env_step_seconds,
                extra_train_metadata=extra_train_metadata,
            )
            if finish_type == "length":
                row.status = Sample.Status.TRUNCATED
            rows.append(row)
            if trace_enabled:
                trace_entries.append(
                    {
                        "sample": row,
                        "response_token_ids": response_ids,
                        "rollout_log_probs": response_log_probs,
                        "prompt_text": rendered_prompt,
                        "messages": messages,
                        "extra": {
                            "prompt_ids": prompt_ids,
                            "prompt_overlength": bool(prompt_budget_metadata.get("prompt_overlength", False)),
                            "prompt_truncated": bool(prompt_budget_metadata.get("history_auto_truncated", False)),
                            "current_observation": observation,
                            "next_observation": next_observation,
                            "task_description": task_description,
                            "gamefile": gamefile,
                            "admissible_actions": admissible_actions,
                            "next_admissible_actions": _admissible_actions(info),
                            "env_action": action,
                        },
                    }
                )
            if env_step_error is not None:
                break
            history.append(
                observation,
                action,
                assistant_history_content(
                    mode=config.history_assistant_content,
                    action=action,
                    response=response_text,
                ),
                user_message=messages[-1]["content"],
            )
            observation = next_observation
            if done:
                break

        backfill_terminal_rewards(
            rows,
            episode_reward=episode_reward,
            success=success,
            invalid_format_penalty=config.invalid_action_penalty,
            done=done,
            max_steps=config.max_episode_steps,
        )
        episode_seconds = time.monotonic() - episode_start
        for row in rows:
            row.metadata.setdefault("episode_error", 0.0)
            row.metadata.setdefault("episode_error_count", 0.0)
            row.metadata.setdefault("env_step_failed", False)
            row.metadata["episode_seconds"] = episode_seconds
            train_metadata = row.train_metadata if isinstance(row.train_metadata, dict) else {}
            for key in ("sdpo", "grpo_token_weight_context"):
                context_metadata = train_metadata.get(key)
                if isinstance(context_metadata, dict):
                    context_metadata["episode_rewards"] = float(row.metadata["episode_reward"])
                    context_metadata["episode_lengths"] = int(row.metadata["episode_length"])
                    context_metadata["is_terminal"] = bool(row.metadata["is_terminal"])
        if trace_enabled:
            for entry in trace_entries:
                add_sample_trace_record(
                    config=trace_config,
                    task="alfworld",
                    phase=trace_phase,
                    sample=entry["sample"],
                    response_token_ids=entry["response_token_ids"],
                    rollout_log_probs=entry["rollout_log_probs"],
                    prompt_text=entry["prompt_text"],
                    messages=entry["messages"],
                    extra=entry["extra"],
                )
        return rows
    finally:
        if pool is not None:
            pool.release(env)
        else:
            close = getattr(env, "close", None)
            if close is not None:
                with suppress(Exception):
                    close()


async def _acquire_env(args: Any, *, config: AlfWorldConfig, split: str, seed: int):
    assert_alfworld_live_environment_allowed(args)
    factory = getattr(args, "alfworld_env_factory", None)
    if factory is not None:
        return factory(seed=seed, split=split, max_episode_steps=config.max_episode_steps), None
    if not config.use_process_env_pool:
        return (
            build_alfworld_env(
                seed=seed,
                split=split,
                cache_dir=config.cache_dir,
                max_episode_steps=config.max_episode_steps,
                suppress_output=config.suppress_env_output,
            ),
            None,
        )
    from slime.rollout.sglang_rollout import GenerateState

    state = GenerateState(args)
    pool = get_or_create_alfworld_env_pool(
        state,
        split=split,
        cache_dir=config.cache_dir,
        max_episode_steps=config.max_episode_steps,
        pool_size=config.env_pool_size,
        prewarm_batch_size=config.env_prewarm_batch_size,
        suppress_output=config.suppress_env_output,
    )
    if config.prewarm_env_pool:
        await pool.prewarm_async(seed=seed)
    env = await pool.acquire_async(seed=seed)
    return env, pool


def _get_tokenizer(args: Any):
    tokenizer = getattr(args, "alfworld_tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    from slime.rollout.sglang_rollout import GenerateState

    return GenerateState(args).tokenizer


def _build_turn_prompt(
    args: Any,
    tokenizer,
    *,
    config: AlfWorldConfig,
    current_observation: str,
    admissible_actions: list[str],
    turn_idx: int,
    task_description: str,
    history: TextHistory,
) -> tuple[list[dict[str, str]], str, str, list[int], dict[str, Any]]:
    messages = build_messages(
        current_observation=current_observation,
        admissible_actions=admissible_actions,
        turn_idx=turn_idx,
        task_description=task_description,
        history=history,
        history_format=config.history_format,
    )
    raw_prompt = format_chat_messages(messages)
    rendered_prompt, prompt_ids, metadata = _render_prompt(
        args, tokenizer, messages, raw_prompt, config, enforce_policy=False
    )
    metadata.update(
        {
            "history_steps_total": len(history),
            "history_steps_kept": len(history),
            "history_auto_truncated": False,
            "history_auto_truncated_dropped": 0,
            "history_assistant_content": config.history_assistant_content,
        }
    )
    if not metadata["prompt_overlength"]:
        return messages, raw_prompt, rendered_prompt, prompt_ids, metadata
    if config.prompt_overlength_policy != "error":
        return messages, raw_prompt, rendered_prompt, prompt_ids, metadata
    if history.max_length is not None or len(history) == 0:
        _raise_prompt_overlength(len(prompt_ids), getattr(args, "rollout_max_prompt_len", None))

    full_prompt_tokens = len(prompt_ids)
    for keep_steps in range(len(history) - 1, -1, -1):
        truncated_history = history.tail(keep_steps)
        candidate_messages = build_messages(
            current_observation=current_observation,
            admissible_actions=admissible_actions,
            turn_idx=turn_idx,
            task_description=task_description,
            history=truncated_history,
            history_format=config.history_format,
            )
        candidate_raw_prompt = format_chat_messages(candidate_messages)
        candidate_rendered_prompt, candidate_prompt_ids, candidate_metadata = _render_prompt(
            args, tokenizer, candidate_messages, candidate_raw_prompt, config, enforce_policy=False
        )
        if candidate_metadata["prompt_overlength"]:
            continue
        candidate_metadata.update(
            {
                "history_steps_total": len(history),
                "history_steps_kept": keep_steps,
                "history_auto_truncated": True,
                "history_auto_truncated_dropped": len(history) - keep_steps,
                "full_prompt_tokens_before_truncation": full_prompt_tokens,
                "history_assistant_content": config.history_assistant_content,
            }
        )
        return (
            candidate_messages,
            candidate_raw_prompt,
            candidate_rendered_prompt,
            candidate_prompt_ids,
            candidate_metadata,
        )

    _raise_prompt_overlength(full_prompt_tokens, getattr(args, "rollout_max_prompt_len", None))


def _render_prompt(
    args: Any,
    tokenizer,
    messages: list[dict[str, str]],
    raw_prompt: str,
    config: AlfWorldConfig,
    *,
    enforce_policy: bool = True,
) -> tuple[str, list[int], dict[str, Any]]:
    if getattr(args, "apply_chat_template", False):
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(getattr(args, "apply_chat_template_kwargs", {}) or {}),
        )
    else:
        rendered_prompt = raw_prompt
    prompt_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    max_prompt_len = getattr(args, "rollout_max_prompt_len", None)
    prompt_overlength = max_prompt_len is not None and len(prompt_ids) > int(max_prompt_len)
    if prompt_overlength and enforce_policy and config.prompt_overlength_policy == "error":
        _raise_prompt_overlength(len(prompt_ids), max_prompt_len)
    return (
        rendered_prompt,
        prompt_ids,
        {
            "prompt_overlength": bool(prompt_overlength),
            "prompt_overlength_policy": config.prompt_overlength_policy,
            "raw_prompt_tokens": len(prompt_ids),
            "prompt_max_tokens": max_prompt_len,
        },
    )


def _raise_prompt_overlength(prompt_tokens: int, max_prompt_len: Any) -> None:
    raise ValueError(
        f"alfworld prompt overlength: {prompt_tokens} > rollout_max_prompt_len {max_prompt_len}; "
        "raise rollout_max_prompt_len or reduce ALFWORLD_HISTORY_MAX_STEPS"
    )


async def _generate_step(
    args: Any,
    *,
    prompt_ids: list[int],
    rendered_prompt: str,
    sampling_params: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    generator = getattr(args, "alfworld_generator", None)
    if generator is not None:
        return await generator(
            args=args,
            prompt_ids=prompt_ids,
            rendered_prompt=rendered_prompt,
            sampling_params=sampling_params,
            session_id=session_id,
        )

    from slime.utils.http_utils import post

    headers = None
    if session_id and getattr(args, "router_policy", None) == "consistent_hashing":
        headers = {"X-SMG-Routing-Key": session_id}
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    return await post(_generation_url(args), payload, headers=headers)


def _generation_url(args: Any) -> str:
    if getattr(args, "alfworld_generator", None) is not None:
        return "fake://alfworld_generator"
    direct_url = getattr(args, "alfworld_sglang_url", None)
    if direct_url:
        return f"{str(direct_url).rstrip('/')}/generate"
    from slime.rollout.sglang_rollout import get_model_url

    return get_model_url(args, str(getattr(args, "alfworld_sglang_model_name", "default")), "/generate")


def _eval_turn_sampling_seed(
    args: Any,
    *,
    split: str,
    rollout_id: int,
    uid: str,
    turn_idx: int,
) -> int:
    sampling_rollout_id = int(getattr(args, "alfworld_eval_sampling_rollout_id", rollout_id))
    payload = (
        f"alfworld-eval-v1:{int(getattr(args, 'eval_sampling_seed', 314159))}:"
        f"{split}:{sampling_rollout_id}:{uid}:{int(turn_idx)}"
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**31 - 1) or 1


def _collection_turn_sampling_seed(args: Any, *, stream_index: int, turn_idx: int) -> int:
    return collection_turn_seed(
        cycle_seed=int(getattr(args, "rollout_seed", 42)),
        stream_index=stream_index,
        turn_idx=turn_idx,
    )


def _stream_bound_sampling_enabled(args: Any) -> bool:
    return bool(getattr(args, "alfworld_stream_bound_sampling", False))


def _sdpo_training_metadata_enabled(args: Any, *, evaluation: bool) -> bool:
    if evaluation:
        return False
    if bool(getattr(args, "agent_task_sdpo_enabled", False)):
        return True
    if str(getattr(args, "loss_type", "") or "") == "sdpo_loss":
        return True
    convert_path = str(getattr(args, "custom_convert_samples_to_train_data_path", "") or "")
    return "algorithms.sdpo" in convert_path


def _sdpo_current_prompt_text(messages: list[dict[str, str]], raw_prompt: str) -> str:
    if messages:
        last_message = messages[-1]
        if last_message.get("role") == "user":
            return str(last_message.get("content", ""))
    return raw_prompt


def _response_tokens_and_log_probs(meta_info: dict[str, Any]) -> tuple[list[int], list[float]]:
    values = meta_info.get("output_token_logprobs")
    if not values:
        return [], []
    response_ids = [int(item[1]) for item in values]
    response_log_probs = [float(item[0]) for item in values]
    return response_ids, response_log_probs


def _exact_response_token_offsets(
    tokenizer,
    *,
    response_text: str,
    response_ids: list[int],
) -> list[tuple[int, int]] | None:
    try:
        encoded = tokenizer(response_text, add_special_tokens=False, return_offsets_mapping=True)
        encoded_ids = [int(token_id) for token_id in encoded["input_ids"]]
        offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        if encoded_ids == response_ids and len(offsets) == len(response_ids):
            return offsets
    except (KeyError, TypeError, ValueError, AttributeError):
        pass

    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        return None
    try:
        decoded = decode(response_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        decoded = decode(response_ids, skip_special_tokens=False)
    if decoded != response_text:
        return None

    offsets: list[tuple[int, int]] = []
    previous_end = 0
    for token_idx in range(1, len(response_ids) + 1):
        try:
            prefix = decode(
                response_ids[:token_idx],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            prefix = decode(response_ids[:token_idx], skip_special_tokens=False)
        current_end = len(prefix)
        if current_end < previous_end or not response_text.startswith(prefix):
            return None
        offsets.append((previous_end, current_end))
        previous_end = current_end
    return offsets if previous_end == len(response_text) else None


def _contained_span_token_mask(offsets: list[tuple[int, int]], span: tuple[int, int] | None) -> list[int]:
    if span is None:
        return [0] * len(offsets)
    span_start, span_end = span
    return [
        int(token_start >= span_start and token_end <= span_end and token_end > token_start)
        for token_start, token_end in offsets
    ]


def _build_action_content_token_metadata(
    tokenizer,
    *,
    response_text: str,
    response_ids: list[int],
    format_valid: bool,
) -> dict[str, Any]:
    """Build an exact mask for tokens fully contained in the first action content."""
    empty = [0] * len(response_ids)
    if not format_valid:
        return {
            "sgs_action_token_mask": empty,
            "sgs_action_alignment_valid": False,
            "sgs_action_alignment_reason": "response_format_invalid",
        }
    lowered = response_text.lower()
    open_tag = "<action>"
    close_tag = "</action>"
    content_start = lowered.find(open_tag)
    content_end = lowered.find(close_tag, content_start + len(open_tag))
    if content_start < 0 or content_end < 0:
        return {
            "sgs_action_token_mask": empty,
            "sgs_action_alignment_valid": False,
            "sgs_action_alignment_reason": "action_content_missing",
        }
    content_start += len(open_tag)
    while content_start < content_end and response_text[content_start].isspace():
        content_start += 1
    while content_end > content_start and response_text[content_end - 1].isspace():
        content_end -= 1
    offsets = _exact_response_token_offsets(tokenizer, response_text=response_text, response_ids=response_ids)
    if offsets is None:
        return {
            "sgs_action_token_mask": empty,
            "sgs_action_alignment_valid": False,
            "sgs_action_alignment_reason": "response_token_alignment_failed",
        }
    mask = _contained_span_token_mask(offsets, (content_start, content_end))
    if not any(mask):
        return {
            "sgs_action_token_mask": mask,
            "sgs_action_alignment_valid": False,
            "sgs_action_alignment_reason": "no_contained_action_content_tokens",
        }
    return {
        "sgs_action_token_mask": mask,
        "sgs_action_alignment_valid": True,
        "sgs_action_alignment_reason": None,
    }


def _finish_type(meta_info: dict[str, Any]) -> str:
    finish_reason = meta_info.get("finish_reason") or {}
    if isinstance(finish_reason, dict):
        return str(finish_reason.get("type", ""))
    return str(finish_reason)


def _validate_finish_type(finish_type: str) -> None:
    if finish_type in {"stop", "length"}:
        return
    if finish_type == "abort":
        raise RuntimeError("ALFWorld generation aborted before env step; refusing to train partial response")
    raise RuntimeError(f"Unexpected ALFWorld generation finish_reason={finish_type!r}")


def _admissible_actions(info: dict[str, Any]) -> list[str]:
    value = info.get("admissible_commands") or info.get("admissible_actions") or []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _info_value(info: dict[str, Any], key: str, default: Any) -> Any:
    value = info.get(key, default)
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def _seed(sample: Sample) -> int:
    return int((sample.metadata or {}).get("seed", 0))


def _segment_sample_index(sample: Sample, turn_idx: int, max_episode_steps: int) -> int:
    if sample.index is None:
        raise ValueError("ALFWorld placeholder sample.index must be set")
    return int(sample.index) * (int(max_episode_steps) + 1) + int(turn_idx)


def _short_error_message(exc: Exception, *, limit: int = 500) -> str:
    message = str(exc) or repr(exc)
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."
