from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

from slime.utils.types import Sample


@dataclass(frozen=True)
class EvalTarget:
    dataset_name: str
    split: str


@dataclass(frozen=True)
class EvalReplicatePlan:
    """Logical eval replicates executed inside one physical model-service job."""

    seeds: tuple[int, ...]
    reuse_engine: bool
    deterministic_sampling: bool


T = TypeVar("T")


def resolve_eval_replicate_plan(
    args: Any,
    *,
    legacy_seed_attr: str | None = None,
    default_seed: int = 314159,
) -> EvalReplicatePlan:
    configured = getattr(args, "eval_replicate_seeds", None)
    legacy_seed = getattr(args, legacy_seed_attr, None) if legacy_seed_attr else None
    deterministic_sampling = configured is not None or legacy_seed is not None
    if configured is None:
        seeds = (int(legacy_seed if legacy_seed is not None else getattr(args, "eval_sampling_seed", default_seed)),)
    else:
        if not isinstance(configured, (list, tuple)):
            raise ValueError("eval_replicate_seeds must be a non-empty list of unique integers")
        seeds = tuple(int(seed) for seed in configured)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("eval_replicate_seeds must be a non-empty list of unique integers")
    reuse_engine = bool(getattr(args, "reuse_eval_engine_across_replicates", False))
    if len(seeds) > 1 and not reuse_engine:
        raise ValueError("multiple eval_replicate_seeds require reuse_eval_engine_across_replicates=true")
    return EvalReplicatePlan(
        seeds=seeds,
        reuse_engine=reuse_engine,
        deterministic_sampling=deterministic_sampling,
    )


async def run_eval_replicates_sequentially(
    plan: EvalReplicatePlan,
    generate_one: Callable[[int, int], Awaitable[T]],
) -> list[T]:
    """Run logical replicates serially while the caller keeps its engines alive."""

    results = []
    for replicate_index, replicate_seed in enumerate(plan.seeds):
        results.append(await generate_one(replicate_index, replicate_seed))
    return results


def replicate_dataset_name(
    dataset_name: str,
    replicate_index: int,
    replicate_seed: int,
    *,
    replicated: bool,
) -> str:
    if not replicated:
        return dataset_name
    return f"{dataset_name}__replicate_{int(replicate_index):02d}_seed_{int(replicate_seed)}"


def build_eval_placeholder_sample(
    *,
    rollout_id: int,
    dataset_idx: int,
    dataset_name: str,
    episode_idx: int,
    seed: int,
    split: str,
) -> Sample:
    uid = f"{dataset_name}-{rollout_id:04d}-{episode_idx:06d}"
    return Sample(
        group_index=episode_idx,
        index=rollout_id * 10_000_000 + dataset_idx * 1_000_000 + episode_idx,
        prompt=uid,
        metadata={
            "uid": uid,
            "traj_uid": f"{uid}-traj-00",
            "eval_dataset_name": dataset_name,
            "seed": seed,
            "repeat_idx": 0,
            "split": split,
        },
    )


def build_eval_dataset_output(episode_rows: list[list[Sample]]) -> dict[str, list[Any]]:
    step_samples = [row for rows in episode_rows for row in rows]
    terminal_samples = [rows[-1] for rows in episode_rows if rows]
    rewards = [float(sample.reward or 0.0) for sample in terminal_samples]
    truncated = [sample.status == Sample.Status.TRUNCATED for sample in terminal_samples]
    return {
        "rewards": rewards,
        "truncated": truncated,
        "samples": terminal_samples,
        "step_samples": step_samples,
    }
