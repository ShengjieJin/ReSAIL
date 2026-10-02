from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FrozenArmCapabilities:
    response_source: Literal["deployment", "current_actor", "epd"]
    objective: Literal["sft", "grpo", "sdpo"]
    success_only: bool = False
    fanout: int = 1
    pair_audit: bool = False
    action_match_audit: bool = False
    privileged_teacher_trajectory: bool = False
    sampling_namespace: str | None = None


FROZEN_ARMS = {
    "iterative_success_filtered_sft": FrozenArmCapabilities("deployment", "sft", success_only=True),
    "iterative_success_only_grpo": FrozenArmCapabilities(
        "current_actor",
        "grpo",
        success_only=True,
        fanout=8,
        action_match_audit=True,
        sampling_namespace="iterative-success-grpo-v1",
    ),
    "iterative_offline_sdpo": FrozenArmCapabilities(
        "deployment", "sdpo", pair_audit=True, privileged_teacher_trajectory=True
    ),
    "iterative_oel": FrozenArmCapabilities(
        "current_actor",
        "sdpo",
        pair_audit=True,
        action_match_audit=True,
        privileged_teacher_trajectory=True,
        sampling_namespace="iterative-current-actor-v1",
    ),
    "iterative_epd": FrozenArmCapabilities("epd", "sft", pair_audit=True),
    "iterative_trajectory_distillation": FrozenArmCapabilities(
        "current_actor",
        "sdpo",
        pair_audit=True,
        action_match_audit=True,
        privileged_teacher_trajectory=True,
        sampling_namespace="iterative-current-actor-v1",
    ),
}


def capabilities_for_arm(arm: str) -> FrozenArmCapabilities:
    try:
        return FROZEN_ARMS[str(arm)]
    except KeyError as exc:
        raise ValueError(f"unsupported iterative frozen arm: {arm!r}") from exc
