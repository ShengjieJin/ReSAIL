from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slime_plugins.agent_tasks.common.config import (
    as_bool,
    get_arg_or_env,
    history_assistant_content_from_arg_or_env,
    history_format_from_arg_or_env,
    history_max_steps_from_arg_or_env,
    optional_str,
    sampling_params_from_args,
)


@dataclass(frozen=True)
class AlfWorldConfig:
    cache_dir: Path
    max_episode_steps: int = 30
    history_format: str = "inline"
    history_max_steps: int | None = None
    history_assistant_content: str = "full_response"
    invalid_action_penalty: float = 0.1
    success_reward: float = 1.0
    train_split: str = "train"
    eval_split: str = "eval_in_distribution"
    eval_dataset_name: str = "alfworld_eval"
    eval_out_of_distribution_split: str = "eval_out_of_distribution"
    eval_out_of_distribution_dataset_name: str | None = None
    eval_episodes: int = 128
    sample_log_dir: Path | None = None
    sample_log_limit: int = 32
    env_pool_size: int = 128
    use_process_env_pool: bool = True
    prewarm_env_pool: bool = True
    env_prewarm_batch_size: int = 32
    eval_concurrency: int = 128
    prompt_overlength_policy: str = "error"
    debug_logging: bool = False
    suppress_env_output: bool = True
    max_episode_errors: int = 8
    max_episode_error_rate: float = 0.05


def get_alfworld_config(args: Any, *, evaluation: bool = False) -> AlfWorldConfig:
    cache_dir = Path(get_arg_or_env(args, "alfworld_cache_dir", "ALFWORLD_DATA", ".cache/alfworld")).expanduser()
    sample_log_dir_raw = get_arg_or_env(args, "alfworld_sample_log_dir", "ALFWORLD_SAMPLE_LOG_DIR", None)
    sample_log_dir = Path(sample_log_dir_raw).expanduser() if sample_log_dir_raw else None
    env_pool_size = int(get_arg_or_env(args, "alfworld_env_pool_size", "ALFWORLD_ENV_POOL_SIZE", 128))
    config = AlfWorldConfig(
        cache_dir=cache_dir,
        max_episode_steps=int(get_arg_or_env(args, "alfworld_max_episode_steps", "ALFWORLD_MAX_EPISODE_STEPS", 30)),
        history_format=history_format_from_arg_or_env(
            args,
            "alfworld_history_format",
            "ALFWORLD_HISTORY_FORMAT",
            "alfworld_history_max_steps",
            "ALFWORLD_HISTORY_MAX_STEPS",
            "alfworld_history_length",
            "ALFWORLD_HISTORY_LENGTH",
        ),
        history_max_steps=history_max_steps_from_arg_or_env(
            args,
            "alfworld_history_max_steps",
            "ALFWORLD_HISTORY_MAX_STEPS",
            "alfworld_history_length",
            "ALFWORLD_HISTORY_LENGTH",
        ),
        history_assistant_content=history_assistant_content_from_arg_or_env(
            args,
            "alfworld_history_assistant_content",
            "ALFWORLD_HISTORY_ASSISTANT_CONTENT",
        ),
        invalid_action_penalty=float(
            get_arg_or_env(args, "alfworld_invalid_action_penalty", "ALFWORLD_INVALID_ACTION_PENALTY", 0.1)
        ),
        success_reward=float(get_arg_or_env(args, "alfworld_success_reward", "ALFWORLD_SUCCESS_REWARD", 1.0)),
        train_split=str(get_arg_or_env(args, "alfworld_train_split", "ALFWORLD_TRAIN_SPLIT", "train")),
        eval_split=str(get_arg_or_env(args, "alfworld_eval_split", "ALFWORLD_EVAL_SPLIT", "eval_in_distribution")),
        eval_dataset_name=str(
            get_arg_or_env(args, "alfworld_eval_dataset_name", "ALFWORLD_EVAL_DATASET_NAME", "alfworld_eval")
        ),
        eval_out_of_distribution_split=str(
            get_arg_or_env(
                args,
                "alfworld_eval_out_of_distribution_split",
                "ALFWORLD_EVAL_OUT_OF_DISTRIBUTION_SPLIT",
                "eval_out_of_distribution",
            )
        ),
        eval_out_of_distribution_dataset_name=optional_str(
            get_arg_or_env(
                args,
                "alfworld_eval_out_of_distribution_dataset_name",
                "ALFWORLD_EVAL_OUT_OF_DISTRIBUTION_DATASET_NAME",
                None,
            )
        ),
        eval_episodes=int(get_arg_or_env(args, "alfworld_eval_episodes", "ALFWORLD_EVAL_EPISODES", 128)),
        sample_log_dir=sample_log_dir,
        sample_log_limit=int(get_arg_or_env(args, "alfworld_sample_log_limit", "ALFWORLD_SAMPLE_LOG_LIMIT", 32)),
        env_pool_size=env_pool_size,
        use_process_env_pool=as_bool(
            get_arg_or_env(args, "alfworld_use_process_env_pool", "ALFWORLD_USE_PROCESS_ENV_POOL", True)
        ),
        prewarm_env_pool=as_bool(
            get_arg_or_env(args, "alfworld_prewarm_env_pool", "ALFWORLD_PREWARM_ENV_POOL", True)
        ),
        env_prewarm_batch_size=int(
            get_arg_or_env(
                args,
                "alfworld_env_prewarm_batch_size",
                "ALFWORLD_ENV_PREWARM_BATCH_SIZE",
                32,
            )
        ),
        eval_concurrency=int(
            get_arg_or_env(args, "alfworld_eval_concurrency", "ALFWORLD_EVAL_CONCURRENCY", env_pool_size)
        ),
        prompt_overlength_policy=str(
            get_arg_or_env(args, "agent_task_prompt_overlength_policy", "AGENT_TASK_PROMPT_OVERLENGTH_POLICY", "error")
        ),
        debug_logging=as_bool(get_arg_or_env(args, "alfworld_debug_logging", "ALFWORLD_DEBUG_LOGGING", False)),
        suppress_env_output=as_bool(
            get_arg_or_env(args, "alfworld_suppress_env_output", "ALFWORLD_SUPPRESS_ENV_OUTPUT", True)
        ),
        max_episode_errors=int(get_arg_or_env(args, "alfworld_max_episode_errors", "ALFWORLD_MAX_EPISODE_ERRORS", 8)),
        max_episode_error_rate=float(
            get_arg_or_env(args, "alfworld_max_episode_error_rate", "ALFWORLD_MAX_EPISODE_ERROR_RATE", 0.05)
        ),
    )
    if config.env_prewarm_batch_size <= 0:
        raise ValueError("alfworld_env_prewarm_batch_size must be positive")
    return config
