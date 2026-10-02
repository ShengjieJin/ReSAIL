from __future__ import annotations

import hashlib

from slime_plugins.agent_tasks.common.sampling import deterministic_rollout_seed


def stream_turn_seed(*, task: str, cycle_seed: int, stream_index: int, turn_idx: int) -> int:
    if not task or stream_index < 0 or turn_idx < 0:
        raise ValueError("stream-bound sampling requires a task and non-negative indices")
    payload = f"{task}-iterative-collection-v1:{int(cycle_seed)}:{int(stream_index)}:{int(turn_idx)}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**31 - 1) or 1


def eval_turn_seed(*, task: str, replicate_seed: int, task_id: str, turn_idx: int) -> int:
    if not task or not task_id or turn_idx < 0:
        raise ValueError("eval sampling requires a task identity and non-negative turn index")
    payload = f"{task}-eval-v1:{int(replicate_seed)}:{task_id}:{int(turn_idx)}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**31 - 1) or 1


def branch_turn_seed(
    *,
    namespace: str,
    base_seed: int,
    update: int,
    trajectory_uid: str,
    turn_idx: int,
    branch_idx: int,
    include_branch: bool = False,
) -> int:
    return deterministic_rollout_seed(
        namespace=namespace,
        base_seed=base_seed,
        update=update,
        trajectory_uid=trajectory_uid,
        turn_idx=turn_idx,
        branch_idx=branch_idx,
        include_branch=include_branch,
    )
