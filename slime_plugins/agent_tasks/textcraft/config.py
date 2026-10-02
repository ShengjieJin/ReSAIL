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
    sampling_params_from_args,
)

from .splits import data_indices_from_file, default_split_file


@dataclass(frozen=True)
class TextCraftConfig:
    cache_dir: Path
    max_episode_steps: int = 30
    history_format: str = "inline"
    history_max_steps: int | None = None
    history_assistant_content: str = "full_response"
    invalid_action_penalty: float = 0.1
    success_reward: float = 1.0
    env_reset_seed: int = 42
    train_split: str = "train"
    eval_split: str = "eval"
    eval_dataset_name: str = "textcraft_eval"
    split_source: str = "agentgym_rl_data_id"
    train_data_file: Path | None = None
    eval_data_file: Path | None = None
    train_task_count: int = 374
    eval_task_count: int = 100
    eval_data_idx_offset: int = 374
    eval_episodes: int = 100
    sample_log_dir: Path | None = None
    sample_log_limit: int = 32
    eval_concurrency: int = 128
    prompt_overlength_policy: str = "error"
    debug_logging: bool = False
    max_episode_errors: int = 8
    max_episode_error_rate: float = 0.05


def get_textcraft_config(args: Any, *, evaluation: bool = False) -> TextCraftConfig:
    cache_dir = Path(get_arg_or_env(args, "textcraft_cache_dir", "TEXTCRAFT_DATA", ".cache/textcraft")).expanduser()
    sample_log_dir_raw = get_arg_or_env(args, "textcraft_sample_log_dir", "TEXTCRAFT_SAMPLE_LOG_DIR", None)
    sample_log_dir = Path(sample_log_dir_raw).expanduser() if sample_log_dir_raw else None
    train_data_file_raw = get_arg_or_env(args, "textcraft_train_data_file", "TEXTCRAFT_TRAIN_DATA_FILE", None)
    eval_data_file_raw = get_arg_or_env(args, "textcraft_eval_data_file", "TEXTCRAFT_EVAL_DATA_FILE", None)
    train_task_count = int(get_arg_or_env(args, "textcraft_train_task_count", "TEXTCRAFT_TRAIN_TASK_COUNT", 374))
    eval_task_count = int(get_arg_or_env(args, "textcraft_eval_task_count", "TEXTCRAFT_EVAL_TASK_COUNT", 100))
    return TextCraftConfig(
        cache_dir=cache_dir,
        max_episode_steps=int(get_arg_or_env(args, "textcraft_max_episode_steps", "TEXTCRAFT_MAX_EPISODE_STEPS", 30)),
        history_format=history_format_from_arg_or_env(
            args,
            "textcraft_history_format",
            "TEXTCRAFT_HISTORY_FORMAT",
            "textcraft_history_max_steps",
            "TEXTCRAFT_HISTORY_MAX_STEPS",
            "textcraft_history_length",
            "TEXTCRAFT_HISTORY_LENGTH",
        ),
        history_max_steps=history_max_steps_from_arg_or_env(
            args,
            "textcraft_history_max_steps",
            "TEXTCRAFT_HISTORY_MAX_STEPS",
            "textcraft_history_length",
            "TEXTCRAFT_HISTORY_LENGTH",
        ),
        history_assistant_content=history_assistant_content_from_arg_or_env(
            args,
            "textcraft_history_assistant_content",
            "TEXTCRAFT_HISTORY_ASSISTANT_CONTENT",
        ),
        invalid_action_penalty=float(
            get_arg_or_env(args, "textcraft_invalid_action_penalty", "TEXTCRAFT_INVALID_ACTION_PENALTY", 0.1)
        ),
        success_reward=float(get_arg_or_env(args, "textcraft_success_reward", "TEXTCRAFT_SUCCESS_REWARD", 1.0)),
        env_reset_seed=int(get_arg_or_env(args, "textcraft_env_reset_seed", "TEXTCRAFT_ENV_RESET_SEED", 42)),
        train_split=str(get_arg_or_env(args, "textcraft_train_split", "TEXTCRAFT_TRAIN_SPLIT", "train")),
        eval_split=str(get_arg_or_env(args, "textcraft_eval_split", "TEXTCRAFT_EVAL_SPLIT", "eval")),
        eval_dataset_name=str(
            get_arg_or_env(args, "textcraft_eval_dataset_name", "TEXTCRAFT_EVAL_DATASET_NAME", "textcraft_eval")
        ),
        split_source=str(
            get_arg_or_env(args, "textcraft_split_source", "TEXTCRAFT_SPLIT_SOURCE", "agentgym_rl_data_id")
        ),
        train_data_file=Path(train_data_file_raw).expanduser() if train_data_file_raw else None,
        eval_data_file=Path(eval_data_file_raw).expanduser() if eval_data_file_raw else None,
        train_task_count=max(1, train_task_count),
        eval_task_count=max(1, eval_task_count),
        eval_data_idx_offset=int(
            get_arg_or_env(args, "textcraft_eval_data_idx_offset", "TEXTCRAFT_EVAL_DATA_IDX_OFFSET", train_task_count)
        ),
        eval_episodes=int(get_arg_or_env(args, "textcraft_eval_episodes", "TEXTCRAFT_EVAL_EPISODES", eval_task_count)),
        sample_log_dir=sample_log_dir,
        sample_log_limit=int(get_arg_or_env(args, "textcraft_sample_log_limit", "TEXTCRAFT_SAMPLE_LOG_LIMIT", 32)),
        eval_concurrency=int(get_arg_or_env(args, "textcraft_eval_concurrency", "TEXTCRAFT_EVAL_CONCURRENCY", 128)),
        prompt_overlength_policy=str(
            get_arg_or_env(args, "agent_task_prompt_overlength_policy", "AGENT_TASK_PROMPT_OVERLENGTH_POLICY", "error")
        ),
        debug_logging=as_bool(get_arg_or_env(args, "textcraft_debug_logging", "TEXTCRAFT_DEBUG_LOGGING", False)),
        max_episode_errors=int(get_arg_or_env(args, "textcraft_max_episode_errors", "TEXTCRAFT_MAX_EPISODE_ERRORS", 8)),
        max_episode_error_rate=float(
            get_arg_or_env(args, "textcraft_max_episode_error_rate", "TEXTCRAFT_MAX_EPISODE_ERROR_RATE", 0.05)
        ),
    )


def split_data_indices(config: TextCraftConfig, split: str) -> tuple[int, ...]:
    split_source = config.split_source.strip().lower()
    if split_source == "agentgym_rl_data_id":
        if split == "train":
            return data_indices_from_file(config.train_data_file or default_split_file(config.cache_dir, "train"))
        if split == "eval":
            return data_indices_from_file(config.eval_data_file or default_split_file(config.cache_dir, "eval"))
        raise ValueError(f"Unsupported TextCraft split: {split!r}")
    if split_source == "generated":
        if split == "train":
            return tuple(range(config.train_task_count))
        if split == "eval":
            return tuple(
                config.eval_data_idx_offset + episode_idx for episode_idx in range(max(1, config.eval_task_count))
            )
        raise ValueError(f"Unsupported TextCraft split: {split!r}")
    raise ValueError(
        f"Unsupported TEXTCRAFT_SPLIT_SOURCE={config.split_source!r}; expected 'agentgym_rl_data_id' or 'generated'"
    )


def train_data_idx(args: Any, group_index: int) -> int:
    indices = split_data_indices(get_textcraft_config(args), "train")
    return indices[int(group_index) % len(indices)]


def eval_data_idx(config: TextCraftConfig, episode_idx: int) -> int:
    indices = split_data_indices(config, "eval")
    return indices[int(episode_idx) % len(indices)]
