from __future__ import annotations

from typing import Any
import hashlib

from slime_plugins.agent_tasks.common.sampling import deterministic_rollout_seed

from .capabilities import capabilities_for_args


def sampling_seed_namespace(args: Any, arm: str) -> str | None:
    explicit = str(getattr(args, "alfworld_frozen_sampling_seed_namespace", "") or "").strip()
    return explicit or capabilities_for_args(args, arm).sampling_namespace


def sampling_seed(
    args: Any,
    *,
    arm: str,
    base_seed: int,
    update: int,
    trajectory_uid: str,
    turn_idx: int,
    branch_idx: int,
) -> int:
    namespace = sampling_seed_namespace(args, arm)
    if namespace is None:
        raise ValueError(f"frozen contract has no deterministic rollout seed: {arm!r}")
    capabilities = capabilities_for_args(args, arm)
    include_branch = capabilities.objective == "grpo" or (
        capabilities.configured_fanout and capabilities.fanout > 1 and int(branch_idx) > 0
    )
    return deterministic_rollout_seed(
        namespace=namespace,
        base_seed=base_seed,
        update=update,
        trajectory_uid=trajectory_uid,
        turn_idx=turn_idx,
        branch_idx=branch_idx,
        include_branch=include_branch,
    )


def collection_turn_seed(*, cycle_seed: int, stream_index: int, turn_idx: int) -> int:
    if stream_index < 0 or turn_idx < 0:
        raise ValueError("collection sampling requires non-negative stream_index and turn_idx")
    payload = f"alfworld-collection-v1:{int(cycle_seed)}:{int(stream_index)}:{int(turn_idx)}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**31 - 1) or 1


__all__ = ["collection_turn_seed", "sampling_seed", "sampling_seed_namespace"]
