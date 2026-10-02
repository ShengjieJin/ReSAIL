from __future__ import annotations

import asyncio
import copy
from typing import Any

from slime.rollout.base_types import RolloutFnEvalOutput
from slime.utils.async_utils import run
from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.eval import EvalTarget, build_eval_dataset_output, build_eval_placeholder_sample

from .config import get_alfworld_config, sampling_params_from_args
from .generate import run_alfworld_episode


def generate_eval_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnEvalOutput:
    return run(_generate_eval_rollout_async(args, rollout_id))


async def _generate_eval_rollout_async(args: Any, rollout_id: int) -> RolloutFnEvalOutput:
    config = get_alfworld_config(args, evaluation=True)
    sampling_params = sampling_params_from_args(args, evaluation=True)
    targets = _eval_targets(config)
    replicate_seeds = _eval_replicate_seeds(args)
    replicated = len(replicate_seeds) > 1
    episode_rows_by_dataset: dict[str, list[list[Sample] | None]] = {
        _replicate_dataset_name(target.dataset_name, replicate_idx, replicate_seed, replicated=replicated): [
            None
        ]
        * config.eval_episodes
        for replicate_idx, replicate_seed in enumerate(replicate_seeds)
        for target in targets
    }
    work = [
        (dataset_idx, target, episode_idx, replicate_idx, replicate_seed)
        for replicate_idx, replicate_seed in enumerate(replicate_seeds)
        for episode_idx in range(config.eval_episodes)
        for dataset_idx, target in enumerate(targets)
    ]
    queue: asyncio.Queue[tuple[int, EvalTarget, int, int, int]] = asyncio.Queue()
    for item in work:
        queue.put_nowait(item)

    async def consume() -> None:
        while True:
            try:
                dataset_idx, target, episode_idx, replicate_idx, replicate_seed = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            dataset_name, result_episode_idx, episode_rows = await _generate_eval_episode(
                args,
                rollout_id,
                config,
                sampling_params,
                len(targets),
                dataset_idx,
                target,
                episode_idx,
                replicate_idx,
                replicate_seed,
                replicated,
            )
            episode_rows_by_dataset[dataset_name][result_episode_idx] = episode_rows

    worker_count = min(max(1, config.eval_concurrency), len(work))
    await asyncio.gather(*(consume() for _ in range(worker_count)))

    data = {
        dataset_name: build_eval_dataset_output([rows for rows in episode_rows if rows is not None])
        for dataset_name, episode_rows in episode_rows_by_dataset.items()
    }
    return RolloutFnEvalOutput(data=data, metrics={})


async def _generate_eval_episode(
    args: Any,
    rollout_id: int,
    config,
    sampling_params: dict[str, Any],
    target_count: int,
    dataset_idx: int,
    target: EvalTarget,
    episode_idx: int,
    replicate_idx: int,
    replicate_seed: int,
    replicated: bool,
) -> tuple[str, int, list[Sample]]:
    episode_args = copy.copy(args)
    setattr(episode_args, "eval_sampling_seed", int(replicate_seed))
    setattr(
        episode_args,
        "alfworld_eval_sampling_rollout_id",
        int(rollout_id) * 10_000_000 + int(dataset_idx) * 1_000_000 + int(episode_idx),
    )
    independent_rng = bool(getattr(args, "alfworld_eval_independent_rng", False))
    environment_seed = _eval_environment_seed(
        args,
        dataset_idx=dataset_idx,
        episode_idx=episode_idx,
        independent_rng=independent_rng,
    )
    sample = build_eval_placeholder_sample(
        rollout_id=rollout_id,
        dataset_idx=replicate_idx * target_count + dataset_idx,
        dataset_name=target.dataset_name,
        episode_idx=episode_idx,
        seed=environment_seed,
        split=target.split,
    )
    rows = await run_alfworld_episode(
        episode_args,
        sample=sample,
        sampling_params=sampling_params,
        config=config,
        split=target.split,
        evaluation=True,
    )
    terminal = rows[-1]
    terminal.reward = float(terminal.metadata.get("episode_reward", 0.0))
    for row in rows:
        row.metadata.update(
            {
                "eval_base_dataset_name": target.dataset_name,
                "eval_replicate_index": int(replicate_idx),
                "eval_replicate_seed": int(replicate_seed),
            }
        )
    dataset_name = _replicate_dataset_name(
        target.dataset_name,
        replicate_idx,
        replicate_seed,
        replicated=replicated,
    )
    return dataset_name, episode_idx, rows


def _eval_targets(config) -> list[EvalTarget]:
    targets = [EvalTarget(config.eval_dataset_name, config.eval_split)]
    if config.eval_out_of_distribution_dataset_name:
        if config.eval_out_of_distribution_dataset_name == config.eval_dataset_name:
            raise ValueError("ALFWorld eval dataset names must be unique.")
        targets.append(EvalTarget(config.eval_out_of_distribution_dataset_name, config.eval_out_of_distribution_split))
    return targets


def _eval_replicate_seeds(args: Any) -> tuple[int, ...]:
    configured = getattr(args, "alfworld_eval_replicate_seeds", None)
    if configured is None:
        return (int(getattr(args, "eval_sampling_seed", 314159)),)
    if not isinstance(configured, (list, tuple)) or not configured:
        raise ValueError("alfworld_eval_replicate_seeds must be a non-empty list of integers")
    seeds = tuple(int(value) for value in configured)
    if len(set(seeds)) != len(seeds):
        raise ValueError("alfworld_eval_replicate_seeds must be unique")
    return seeds


def _eval_environment_seed(
    args: Any,
    *,
    dataset_idx: int,
    episode_idx: int,
    independent_rng: bool,
) -> int:
    """Keep task identities paired while replicate seeds vary only model sampling."""
    if independent_rng:
        identity_seed = int(
            getattr(args, "alfworld_eval_identity_seed", getattr(args, "eval_sampling_seed", 314159))
        )
        return identity_seed + int(dataset_idx) * 1_000_000 + int(episode_idx)
    return int(getattr(args, "rollout_seed", 42)) + int(episode_idx)


def _replicate_dataset_name(
    dataset_name: str,
    replicate_idx: int,
    replicate_seed: int,
    *,
    replicated: bool,
) -> str:
    if not replicated:
        return dataset_name
    return f"{dataset_name}__replicate_{int(replicate_idx):02d}_seed_{int(replicate_seed)}"
