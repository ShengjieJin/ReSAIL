from __future__ import annotations

import asyncio
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
from slime_plugins.agent_tasks.common.frozen.sampling import eval_turn_seed, stream_turn_seed

from .config import TextCraftConfig, get_textcraft_config
from .envs import build_textcraft_env, item_id_to_str
from .projection import project_response
from .prompts import build_messages

logger = logging.getLogger(__name__)


async def generate(args, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> list[Sample]:
    config = get_textcraft_config(args, evaluation=evaluation)
    split = config.eval_split if evaluation else str(sample.metadata.get("split") or config.train_split)
    rows = await run_textcraft_episode(
        args,
        sample=sample,
        sampling_params=sampling_params,
        config=config,
        split=split,
        evaluation=evaluation,
    )
    if not rows:
        raise RuntimeError("TextCraft custom_generate produced no step samples")
    return rows


async def run_textcraft_episode(
    args: Any,
    *,
    sample: Sample,
    sampling_params: dict[str, Any],
    config: TextCraftConfig,
    split: str,
    evaluation: bool = False,
) -> list[Sample]:
    tokenizer = _get_tokenizer(args)
    data_idx = int((sample.metadata or {}).get("data_idx", 0))
    env_reset_seed = int(config.env_reset_seed)
    env = await _acquire_env(args, config=config, split=split, seed=env_reset_seed, data_idx=data_idx)
    session_id = str(sample.metadata.get("traj_uid") or uuid.uuid4())
    rollout_id = int(sample.index if sample.index is not None else _seed(sample))
    history = TextHistory(max_length=config.history_max_steps)
    rows: list[Sample] = []
    trace_config = get_agent_trace_config(args, task="textcraft", sample_log_dir=config.sample_log_dir)
    trace_phase = "eval" if evaluation else "train"
    trace_enabled = trace_config.enabled_for_phase(trace_phase)
    trace_entries: list[dict[str, Any]] = []
    done = False
    success = False
    episode_reward = 0.0
    episode_start = time.monotonic()

    try:
        reset_start = time.monotonic()
        reset_result = await asyncio.to_thread(env.reset, env_reset_seed, data_idx)
        reset_seconds = float(getattr(reset_result, "reset_seconds", time.monotonic() - reset_start))
        observation = str(getattr(reset_result, "observation"))
        info = dict(getattr(reset_result, "info"))
        worker_id = str(getattr(reset_result, "worker_id", "unknown"))
        worker_reused = bool(getattr(reset_result, "reused_worker", False))
        worker_created = bool(getattr(reset_result, "worker_created", not worker_reused))
        goal = str(info.get("goal", ""))
        goal_text = str(info.get("goal_text") or item_id_to_str(goal)) if goal else ""
        goal_depth = info.get("goal_depth")
        crafting_commands = str(info.get("crafting_commands") or info.get("commands") or "")
        if not crafting_commands:
            crafting_commands = _extract_crafting_commands(observation)
        task_text = _sdpo_task_text(
            goal_text=goal_text,
            crafting_commands=crafting_commands,
            fallback=str(sample.metadata.get("task_id", f"textcraft_{data_idx}")),
        )

        for turn_idx in range(config.max_episode_steps):
            messages, raw_prompt, rendered_prompt, prompt_ids, prompt_budget_metadata = _build_turn_prompt(
                args,
                tokenizer,
                config=config,
                current_observation=observation,
                turn_idx=turn_idx,
                history=history,
                goal_text=goal_text,
                crafting_commands=crafting_commands,
            )

            gen_start = time.monotonic()
            turn_sampling_params = dict(sampling_params)
            task_id = str(sample.metadata.get("task_id", f"textcraft_{data_idx}"))
            if evaluation:
                turn_sampling_params["sampling_seed"] = eval_turn_seed(
                    task="textcraft",
                    replicate_seed=int(getattr(args, "eval_sampling_seed", 314159)),
                    task_id=task_id,
                    turn_idx=turn_idx,
                )
            elif bool(getattr(args, "agent_frozen_stream_bound_sampling", False)):
                turn_sampling_params["sampling_seed"] = stream_turn_seed(
                    task="textcraft",
                    cycle_seed=int(getattr(args, "rollout_seed", 42)),
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
                raise RuntimeError("TextCraft generation returned no response tokens")
            finish_type = _finish_type(meta_info)
            _validate_finish_type(finish_type)
            projection = project_response(response_text)
            action = projection.projected_action

            step_start = time.monotonic()
            env_step_error = None
            inventory_before = _compact_inventory(env) if trace_enabled else {}
            try:
                next_observation, score, done, next_info = await asyncio.to_thread(env.step, action)
                env_step_seconds = time.monotonic() - step_start
                next_observation = str(next_observation)
                info = dict(next_info)
                success = bool(float(score) > 0 or info.get("done") and info.get("reward", 0.0))
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
                    "Recovered TextCraft env.step failure: uid=%s rollout_id=%s turn_idx=%s action=%r",
                    sample.metadata.get("uid"),
                    rollout_id,
                    turn_idx,
                    action,
                    exc_info=True,
                )

            action_failed = bool(info.get("action_failed", False))
            sdpo_action_valid = bool(projection.format_valid and not action_failed)
            inventory_after = _compact_inventory(env) if trace_enabled else {}
            metadata = {
                **(sample.metadata or {}),
                **prompt_budget_metadata,
                "uid": sample.metadata.get("uid"),
                "traj_uid": sample.metadata.get("traj_uid", f"{sample.metadata.get('uid', 'textcraft')}-traj"),
                "agent_task": "textcraft",
                "agent_task_trace_dir": str(trace_config.trace_dir) if trace_config.trace_dir is not None else None,
                "turn_idx": turn_idx,
                "rollout_id": rollout_id,
                "seed": _seed(sample),
                "env_reset_seed": env_reset_seed,
                "split": split,
                "session_id": session_id,
                "data_idx": data_idx,
                "task_id": sample.metadata.get("task_id", f"textcraft_{data_idx}"),
                "task_description": task_text,
                "runtime_task_identity": {
                    "task_id": task_id,
                    "data_idx": data_idx,
                    "goal": goal,
                    "goal_text": goal_text,
                    "task_description": task_text,
                    "split": split,
                },
                "goal": goal,
                "goal_text": goal_text,
                "goal_depth": goal_depth,
                "crafting_commands_present": bool(crafting_commands),
                **compact_prompt_metadata(
                    args,
                    raw_prompt=raw_prompt,
                    rendered_prompt=rendered_prompt,
                    messages=messages,
                ),
                "format_valid": projection.format_valid,
                "missing_action_tag": projection.missing_action_tag,
                "contains_chinese": projection.contains_chinese,
                "invalid_reason": projection.invalid_reason,
                "projected_action": projection.projected_action,
                "action_kind": projection.action_kind,
                "action_failed": action_failed,
                "is_action_valid": sdpo_action_valid,
                "commands_count": info.get("commands_count"),
                "inventory_size": info.get("inventory_size"),
                "finish_reason": finish_type,
                "sampling_seed": turn_sampling_params.get("sampling_seed"),
                "prompt_tokens": len(prompt_ids),
                "response_tokens": len(response_ids),
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
            if bool(getattr(args, "agent_frozen_stream_bound_sampling", False)):
                metadata.update(
                    {
                        "current_observation": observation,
                        "next_observation": next_observation,
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
            if _sdpo_training_metadata_enabled(args, evaluation=evaluation):
                extra_train_metadata = {
                    "sdpo": {
                        "uid": metadata["uid"],
                        "traj_uid": metadata["traj_uid"],
                        "turn_idx": turn_idx,
                        "task_text": task_text,
                        "anchor_obs": observation,
                        "next_anchor_obs": next_observation,
                        "projected_action": projection.projected_action,
                        "is_action_valid": sdpo_action_valid,
                        "is_terminal": False,
                        "episode_rewards": 0.0,
                        "episode_lengths": 0,
                        "sdpo_current_prompt_text": _sdpo_current_prompt_text(messages, raw_prompt),
                        "sdpo_current_raw_prompt": messages,
                        "sdpo_metadata_profile": "textcraft",
                    }
                }
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
                            "current_observation": observation,
                            "next_observation": next_observation,
                            "goal": goal,
                            "goal_text": goal_text,
                            "crafting_commands": crafting_commands,
                            "next_crafting_commands": str(info.get("crafting_commands") or info.get("commands") or ""),
                            "env_action": action,
                            "inventory_before": inventory_before,
                            "inventory_after": inventory_after,
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
            sdpo_metadata = (row.train_metadata or {}).get("sdpo") if isinstance(row.train_metadata, dict) else None
            if isinstance(sdpo_metadata, dict):
                sdpo_metadata["episode_rewards"] = float(row.metadata["episode_reward"])
                sdpo_metadata["episode_lengths"] = int(row.metadata["episode_length"])
                sdpo_metadata["is_terminal"] = bool(row.metadata["is_terminal"])
        if trace_enabled:
            for entry in trace_entries:
                add_sample_trace_record(
                    config=trace_config,
                    task="textcraft",
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
        close = getattr(env, "close", None)
        if close is not None:
            with suppress(Exception):
                close()


async def _acquire_env(args: Any, *, config: TextCraftConfig, split: str, seed: int, data_idx: int):
    factory = getattr(args, "textcraft_env_factory", None)
    if factory is not None:
        return factory(seed=seed, split=split, data_idx=data_idx, max_episode_steps=config.max_episode_steps)
    return build_textcraft_env(
        seed=seed,
        split=split,
        cache_dir=config.cache_dir,
        data_idx=data_idx,
        max_episode_steps=config.max_episode_steps,
    )


def _get_tokenizer(args: Any):
    tokenizer = getattr(args, "textcraft_tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    from slime.rollout.sglang_rollout import GenerateState

    return GenerateState(args).tokenizer


def _build_turn_prompt(
    args: Any,
    tokenizer,
    *,
    config: TextCraftConfig,
    current_observation: str,
    turn_idx: int,
    history: TextHistory,
    goal_text: str,
    crafting_commands: str,
) -> tuple[list[dict[str, str]], str, str, list[int], dict[str, Any]]:
    messages = build_messages(
        current_observation=current_observation,
        turn_idx=turn_idx,
        history=history,
        goal_text=goal_text,
        history_format=config.history_format,
        crafting_commands=crafting_commands,
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
            turn_idx=turn_idx,
            history=truncated_history,
            goal_text=goal_text,
            history_format=config.history_format,
            crafting_commands=crafting_commands,
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
    config: TextCraftConfig,
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
        f"textcraft prompt overlength: {prompt_tokens} > rollout_max_prompt_len {max_prompt_len}; "
        "raise rollout_max_prompt_len or reduce TEXTCRAFT_HISTORY_MAX_STEPS"
    )


def _extract_crafting_commands(observation: str) -> str:
    marker = "Crafting commands:\n"
    start = observation.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    end = observation.find("\n\nGoal:", start)
    if end == -1:
        end = len(observation)
    return observation[start:end].strip()


def _sdpo_task_text(*, goal_text: str, crafting_commands: str, fallback: str) -> str:
    goal = str(goal_text or fallback or "textcraft").strip()
    commands = str(crafting_commands or "").strip()
    if not commands:
        return goal
    return f"Goal: {goal}\nCrafting commands:\n{commands}"


def _compact_inventory(env: Any) -> dict[str, int | float | str]:
    for candidate in [env, getattr(env, "env", None), getattr(env, "game", None)]:
        inventory = getattr(candidate, "inventory", None)
        if not isinstance(inventory, dict):
            continue
        compact: dict[str, int | float | str] = {}
        for key, value in sorted(inventory.items(), key=lambda item: str(item[0])):
            try:
                compact[str(key)] = int(value)
            except (TypeError, ValueError):
                try:
                    compact[str(key)] = float(value)
                except (TypeError, ValueError):
                    compact[str(key)] = str(value)
        return compact
    return {}


async def _generate_step(
    args: Any,
    *,
    prompt_ids: list[int],
    rendered_prompt: str,
    sampling_params: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    generator = getattr(args, "textcraft_generator", None)
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
    if getattr(args, "textcraft_generator", None) is not None:
        return "fake://textcraft_generator"
    direct_url = getattr(args, "textcraft_sglang_url", None)
    if direct_url:
        return f"{str(direct_url).rstrip('/')}/generate"
    from slime.rollout.sglang_rollout import get_model_url

    return get_model_url(args, str(getattr(args, "textcraft_sglang_model_name", "default")), "/generate")


def _sdpo_training_metadata_enabled(args: Any, *, evaluation: bool) -> bool:
    if evaluation or bool(getattr(args, "debug_rollout_only", False)):
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


def _finish_type(meta_info: dict[str, Any]) -> str:
    finish_reason = meta_info.get("finish_reason") or {}
    if isinstance(finish_reason, dict):
        return str(finish_reason.get("type", ""))
    return str(finish_reason)


def _validate_finish_type(finish_type: str) -> None:
    if finish_type in {"stop", "length"}:
        return
    if finish_type == "abort":
        raise RuntimeError("TextCraft generation aborted before env step; refusing to train partial response")
    raise RuntimeError(f"Unexpected TextCraft generation finish_reason={finish_type!r}")


def _seed(sample: Sample) -> int:
    return int((sample.metadata or {}).get("seed", 0))


def _segment_sample_index(sample: Sample, turn_idx: int, max_episode_steps: int) -> int:
    if sample.index is None:
        raise ValueError("TextCraft placeholder sample.index must be set")
    return int(sample.index) * (int(max_episode_steps) + 1) + int(turn_idx)


def _short_error_message(exc: Exception, *, limit: int = 500) -> str:
    message = str(exc) or repr(exc)
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."
