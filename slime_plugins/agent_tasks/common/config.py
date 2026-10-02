from __future__ import annotations

import os
from typing import Any


def get_arg_or_env(args: Any, attr: str, env_key: str, default: Any) -> Any:
    value = getattr(args, attr, None)
    if value is not None:
        return value
    return os.environ.get(env_key, default)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def parse_history_format(value: Any) -> str:
    history_format = str(value).strip().lower()
    if history_format not in {"inline", "chat"}:
        raise ValueError(f"history_format must be 'inline' or 'chat', got {value!r}")
    return history_format


def parse_history_max_steps(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "full":
        return None
    max_steps = int(value)
    if max_steps < 0:
        raise ValueError(f"history_max_steps must be a non-negative integer or 'full', got {value!r}")
    return max_steps


def parse_history_assistant_content(value: Any) -> str:
    assistant_content = str(value).strip().lower()
    if assistant_content in {"action", "action_only", "projected_action"}:
        return "action_only"
    if assistant_content in {"full", "full_response", "response"}:
        return "full_response"
    if assistant_content in {"summary", "step_summary"}:
        return "summary"
    raise ValueError(
        "history_assistant_content must be 'action_only', 'full_response', or 'summary', "
        f"got {value!r}"
    )


def history_assistant_content_from_arg_or_env(
    args: Any, attr: str, env_key: str, default: str = "full_response"
) -> str:
    return parse_history_assistant_content(get_arg_or_env(args, attr, env_key, default))


def history_format_from_arg_or_env(
    args: Any,
    attr: str,
    env_key: str,
    max_steps_attr: str,
    max_steps_env_key: str,
    legacy_attr: str,
    legacy_env_key: str,
) -> str:
    value = getattr(args, attr, None)
    if value is None:
        value = os.environ.get(env_key)
    if value is not None:
        return parse_history_format(value)
    new_max_steps = getattr(args, max_steps_attr, None)
    if new_max_steps is None:
        new_max_steps = os.environ.get(max_steps_env_key)
    if new_max_steps is not None:
        return "inline"
    legacy_value = getattr(args, legacy_attr, None)
    if legacy_value is None:
        legacy_value = os.environ.get(legacy_env_key)
    if legacy_value is not None:
        return "inline"
    return "inline"


def history_max_steps_from_arg_or_env(
    args: Any,
    attr: str,
    env_key: str,
    legacy_attr: str,
    legacy_env_key: str,
) -> int | None:
    value = getattr(args, attr, None)
    if value is None:
        value = os.environ.get(env_key)
    if value is None:
        value = getattr(args, legacy_attr, None)
    if value is None:
        value = os.environ.get(legacy_env_key)
    if value is None:
        value = "full"
    return parse_history_max_steps(value)


def sampling_params_from_args(args: Any, *, evaluation: bool = False) -> dict[str, Any]:
    prefix = "eval" if evaluation else "rollout"
    temperature = getattr(args, f"{prefix}_temperature", None)
    top_p = getattr(args, f"{prefix}_top_p", None)
    top_k = getattr(args, f"{prefix}_top_k", None)
    max_response_len = getattr(args, f"{prefix}_max_response_len", None)
    return {
        "temperature": temperature if temperature is not None else getattr(args, "rollout_temperature", 1.0),
        "top_p": top_p if top_p is not None else getattr(args, "rollout_top_p", 1.0),
        "top_k": top_k if top_k is not None else getattr(args, "rollout_top_k", -1),
        "max_new_tokens": (
            max_response_len if max_response_len is not None else getattr(args, "rollout_max_response_len", 512)
        ),
        "stop": getattr(args, "rollout_stop", None),
        "stop_token_ids": getattr(args, "rollout_stop_token_ids", None),
        "skip_special_tokens": getattr(args, "rollout_skip_special_tokens", False),
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }
