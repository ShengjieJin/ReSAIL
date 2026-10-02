from __future__ import annotations

import hashlib


def deterministic_rollout_seed(
    *,
    namespace: str,
    base_seed: int,
    update: int,
    trajectory_uid: str,
    turn_idx: int,
    branch_idx: int = 0,
    include_branch: bool = False,
) -> int:
    if not namespace:
        raise ValueError("deterministic rollout seed requires a non-empty namespace")
    payload = f"{namespace}:{int(base_seed)}:{int(update)}:{trajectory_uid}:{int(turn_idx)}"
    if include_branch:
        payload = f"{payload}:{int(branch_idx)}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**31 - 1) or 1
