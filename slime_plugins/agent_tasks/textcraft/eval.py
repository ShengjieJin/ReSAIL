from __future__ import annotations

import asyncio
import copy
from typing import Any

from slime.rollout.base_types import RolloutFnEvalOutput
from slime.utils.async_utils import run
from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.eval import EvalTarget, build_eval_dataset_output, build_eval_placeholder_sample
from slime_plugins.agent_tasks.common.frozen.eval import replicate_dataset_name, replicate_seeds

from .config import eval_data_idx, get_textcraft_config, sampling_params_from_args
from .generate import run_textcraft_episode


def generate_eval_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnEvalOutput:
    return run(_generate_eval_rollout_async(args, rollout_id))


async def _generate_eval_rollout_async(args: Any, rollout_id: int) -> RolloutFnEvalOutput:
    config = get_textcraft_config(args, evaluation=True)
    sampling_params = sampling_params_from_args(args, evaluation=True)
    target = EvalTarget(config.eval_dataset_name, config.eval_split)
    seeds = replicate_seeds(args, attr="textcraft_eval_replicate_seeds")
    replicated = len(seeds) > 1
    episode_rows = {
        replicate_dataset_name(target.dataset_name, idx, seed, replicated=replicated): [None] * config.eval_episodes
        for idx, seed in enumerate(seeds)
    }
    work = [(idx, seed, episode_idx) for idx, seed in enumerate(seeds) for episode_idx in range(config.eval_episodes)]
    queue: asyncio.Queue[tuple[int, int, int]] = asyncio.Queue()
    for item in work:
        queue.put_nowait(item)

    async def consume() -> None:
        while True:
            try:
                replicate_idx, replicate_seed, episode_idx = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            dataset_name, result_idx, rows = await _generate_eval_episode(
                args,
                rollout_id,
                config,
                sampling_params,
                target,
                episode_idx,
                replicate_idx,
                replicate_seed,
                replicated,
            )
            episode_rows[dataset_name][result_idx] = rows

    await asyncio.gather(*(consume() for _ in range(min(max(1, config.eval_concurrency), len(work)))))
    data = {
        dataset_name: build_eval_dataset_output([rows for rows in rows_by_episode if rows is not None])
        for dataset_name, rows_by_episode in episode_rows.items()
    }
    return RolloutFnEvalOutput(data=data, metrics={})


async def _generate_eval_episode(
    args: Any,
    rollout_id: int,
    config,
    sampling_params: dict[str, Any],
    target: EvalTarget,
    episode_idx: int,
    replicate_idx: int,
    replicate_seed: int,
    replicated: bool,
) -> tuple[str, int, list[Sample]]:
    data_idx = eval_data_idx(config, episode_idx)
    episode_args = copy.copy(args)
    setattr(episode_args, "eval_sampling_seed", int(replicate_seed))
    identity_seed = int(getattr(args, "textcraft_eval_identity_seed", 314159))
    sample = build_eval_placeholder_sample(
        rollout_id=rollout_id,
        dataset_idx=replicate_idx,
        dataset_name=target.dataset_name,
        episode_idx=episode_idx,
        seed=identity_seed + episode_idx,
        split=target.split,
    )
    sample.metadata.update({"task_id": f"textcraft_{data_idx}", "data_idx": data_idx})
    rows = await run_textcraft_episode(
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
    dataset_name = replicate_dataset_name(
        target.dataset_name, replicate_idx, replicate_seed, replicated=replicated
    )
    return dataset_name, episode_idx, rows
