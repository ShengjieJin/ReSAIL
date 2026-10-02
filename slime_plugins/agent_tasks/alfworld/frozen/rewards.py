from __future__ import annotations
import logging
from collections import defaultdict
from typing import Any

from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.rewards import post_process_grouped_rewards
from .capabilities import capabilities_for_args

logger = logging.getLogger(__name__)


def post_process_action_match_rewards(args: Any, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """Normalize per source-draw turn and make policy loss a turn mean per bundle.

    Slime's policy reducer computes a token-weighted mean over rows sharing one
    ``rollout_id``. Scaling each row advantage by
    ``bundle_tokens / (bundle_turns * row_tokens)`` makes that reducer exactly
    ``mean_turn(mean_response_token(policy_loss))`` for every branch bundle.
    The train step then averages the 128 bundle rollout IDs.
    """

    groups: dict[tuple[int, int], list[Sample]] = defaultdict(list)
    for sample in samples:
        metadata = sample.metadata or {}
        capabilities = capabilities_for_args(args, str(metadata.get("frozen_arm", "")))
        if capabilities.objective != "grpo":
            raise ValueError("action-match reward postprocess received a non-GRPO sample")
        if metadata.get("source_draw_id") is None or metadata.get("source_turn_idx") is None:
            raise ValueError("GRPO sample is missing source draw/turn grouping metadata")
        if metadata.get("branch_idx") is None or metadata.get("bundle_rollout_id") is None:
            raise ValueError("GRPO sample is missing branch bundle identity")
        groups[(int(metadata["source_draw_id"]), int(metadata["source_turn_idx"]))].append(sample)

    for key, group in groups.items():
        branch_indices = [int(sample.metadata["branch_idx"]) for sample in group]
        trajectory_uids = [str(sample.metadata["traj_uid"]) for sample in group]
        bundle_ids = [int(sample.metadata["bundle_rollout_id"]) for sample in group]
        source_uids = {str(sample.metadata.get("source_trajectory_uid")) for sample in group}
        if len(group) != 8:
            raise ValueError(f"GRPO group {key} must contain exactly 8 branches, got {len(group)}")
        if set(branch_indices) != set(range(8)):
            raise ValueError(f"GRPO group {key} must contain branch_idx 0..7 exactly once")
        if len(set(trajectory_uids)) != 8 or len(set(bundle_ids)) != 8:
            raise ValueError(f"GRPO group {key} must contain 8 distinct traj_uid and bundle_rollout_id values")
        if len(source_uids) != 1:
            raise ValueError(f"GRPO source-draw turn group {key} crosses frozen source trajectories")
    _validate_source_draw_bundles(groups)

    raw_rewards, advantages = post_process_grouped_rewards(
        args,
        samples,
        group_key=("source_draw_id", "source_turn_idx"),
        trajectory_key="traj_uid",
    )
    for sample in samples:
        sample.metadata["grpo_step_equal_advantage_scale"] = 1.0
    scaled_advantages = advantages
    nonzero_std_groups = sum(int(len({float(sample.reward) for sample in group}) > 1) for group in groups.values())
    metrics = {
        "rollout/alfworld/action_match/nonzero_std_group_fraction": nonzero_std_groups / len(groups),
        "rollout/alfworld/action_match/effective_advantage_fraction": sum(
            abs(value) > 1e-12 for value in scaled_advantages
        )
        / len(scaled_advantages),
    }
    logger.info("GRPO reward metrics: %s", metrics)
    return raw_rewards, scaled_advantages


def _validate_source_draw_bundles(groups: dict[tuple[int, int], list[Sample]]) -> None:
    draws: dict[int, dict[int, list[Sample]]] = defaultdict(dict)
    for (source_draw_id, source_turn_idx), group in groups.items():
        draws[source_draw_id][source_turn_idx] = group
    for source_draw_id, turn_groups in draws.items():
        if sorted(turn_groups) != list(range(len(turn_groups))):
            raise ValueError(f"GRPO source draw {source_draw_id} must contain contiguous source turns")
        expected_bundles = None
        for source_turn_idx in sorted(turn_groups):
            bundles = {
                int(sample.metadata["branch_idx"]): (
                    int(sample.metadata["bundle_rollout_id"]),
                    str(sample.metadata["traj_uid"]),
                )
                for sample in turn_groups[source_turn_idx]
            }
            if expected_bundles is None:
                expected_bundles = bundles
            elif bundles != expected_bundles:
                raise ValueError(
                    f"GRPO source draw {source_draw_id} must preserve the same eight branch bundles on every turn"
                )
