from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Hashable

from slime.utils.types import Sample

GroupKey = str | tuple[str, ...] | Callable[[Sample], Hashable]


def post_process_grouped_rewards(
    args: Any,
    samples: Sequence[Sample],
    *,
    group_key: GroupKey = "uid",
    trajectory_key: str | None = "traj_uid",
) -> tuple[list[float], list[float]]:
    raw_rewards = [_reward_value(args, sample) for sample in samples]
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, sample in enumerate(samples):
        groups[_group_value(sample, group_key)][_trajectory_value(sample, trajectory_key, i)].append(i)

    normalized = [0.0] * len(samples)
    use_std = bool(getattr(args, "grpo_std_normalization", True))
    for trajectories in groups.values():
        trajectory_scores = {
            trajectory_id: _trajectory_reward(raw_rewards, indices) for trajectory_id, indices in trajectories.items()
        }
        values = list(trajectory_scores.values())
        if len(trajectory_scores) == 1:
            mean = 0.0
            std = 1.0
        else:
            mean = sum(values) / len(values)
            if use_std:
                variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                std = variance**0.5
            else:
                std = 1.0
        for trajectory_id, indices in trajectories.items():
            centered = trajectory_scores[trajectory_id] - mean
            value = centered / (std + 1e-6) if use_std else centered
            for i in indices:
                normalized[i] = value
    return raw_rewards, normalized


def _reward_value(args: Any, sample: Sample) -> float:
    reward_key = getattr(args, "reward_key", None)
    reward = sample.reward if not reward_key else sample.reward[reward_key]
    return float(reward)


def _group_value(sample: Sample, group_key: GroupKey) -> str:
    if callable(group_key):
        return str(group_key(sample))
    metadata = sample.metadata or {}
    if isinstance(group_key, str):
        value = metadata.get(group_key)
        if value is None:
            raise ValueError(f"sample.metadata['{group_key}'] is required for grouped reward normalization")
        return str(value)
    values = []
    for key in group_key:
        value = metadata.get(key)
        if value is None:
            raise ValueError(f"sample.metadata['{key}'] is required for grouped reward normalization")
        values.append(str(value))
    return "\x1f".join(values)


def _trajectory_value(sample: Sample, trajectory_key: str | None, sample_index: int) -> str:
    if trajectory_key is None:
        return str(sample_index)
    metadata = sample.metadata or {}
    value = metadata.get(trajectory_key)
    return str(value) if value is not None else str(sample_index)


def _trajectory_reward(raw_rewards: list[float], indices: list[int]) -> float:
    if not indices:
        return 0.0
    return sum(raw_rewards[i] for i in indices) / len(indices)
