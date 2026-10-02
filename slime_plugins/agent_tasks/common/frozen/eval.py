from __future__ import annotations

from typing import Any


def replicate_seeds(args: Any, *, attr: str, default: tuple[int, ...] = (314159,)) -> tuple[int, ...]:
    configured = getattr(args, attr, None)
    seeds = default if configured is None else tuple(int(value) for value in configured)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"{attr} must contain unique integer seeds")
    return seeds


def replicate_dataset_name(dataset_name: str, replicate_idx: int, replicate_seed: int, *, replicated: bool) -> str:
    if not replicated:
        return dataset_name
    return f"{dataset_name}__replicate_{int(replicate_idx):02d}_seed_{int(replicate_seed)}"
