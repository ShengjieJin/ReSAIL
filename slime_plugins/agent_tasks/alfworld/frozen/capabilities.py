from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ResponseSource = Literal["deployment", "epd", "current_actor"]
Objective = Literal["sft", "sdpo", "grpo"]


@dataclass(frozen=True)
class FrozenArmCapabilities:
    response_source: ResponseSource
    objective: Objective
    success_only: bool = False
    privileged_teacher_trajectory: bool = False
    sampling_namespace: str | None = None
    fanout: int = 1
    configured_fanout: bool = False
    pair_audit: bool = False
    source_audit: bool = False
    strict_action_format_match: bool = False
    step_equal_policy_loss: bool = False
    action_match_audit: bool = False


_CAPABILITIES: dict[str, FrozenArmCapabilities] = {
    "iterative_trajectory_distillation": FrozenArmCapabilities(
        "current_actor",
        "sdpo",
        privileged_teacher_trajectory=True,
        sampling_namespace="iterative-current-actor-v1",
        configured_fanout=True,
        pair_audit=True,
        source_audit=True,
        action_match_audit=True,
    ),
    "iterative_success_filtered_sft": FrozenArmCapabilities(
        "deployment",
        "sft",
        success_only=True,
        source_audit=True,
    ),
    "iterative_success_only_grpo": FrozenArmCapabilities(
        "current_actor",
        "grpo",
        success_only=True,
        sampling_namespace="iterative-success-grpo-v1",
        fanout=8,
        source_audit=True,
        strict_action_format_match=True,
        step_equal_policy_loss=True,
        action_match_audit=True,
    ),
    "iterative_offline_sdpo": FrozenArmCapabilities(
        "deployment",
        "sdpo",
        privileged_teacher_trajectory=True,
        pair_audit=True,
        source_audit=True,
    ),
    "iterative_oel": FrozenArmCapabilities(
        "current_actor",
        "sdpo",
        privileged_teacher_trajectory=True,
        sampling_namespace="iterative-current-actor-v1",
        pair_audit=True,
        source_audit=True,
        action_match_audit=True,
    ),
    "iterative_epd": FrozenArmCapabilities(
        "epd",
        "sft",
        pair_audit=True,
        source_audit=True,
    ),
}

FROZEN_ARMS = frozenset(_CAPABILITIES)


def capabilities_for_args(args: Any, arm: str | None = None) -> FrozenArmCapabilities:
    """Resolve one immutable runtime contract at the datasource boundary."""

    name = str(arm if arm is not None else getattr(args, "alfworld_frozen_arm", ""))
    try:
        base = _CAPABILITIES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported frozen ALFWorld contract: {name!r}") from exc
    if base.configured_fanout:
        fanout = int(getattr(args, "n_samples_per_prompt", base.fanout) or base.fanout)
        if fanout <= 0:
            raise ValueError("configured frozen response fanout must be positive")
        return FrozenArmCapabilities(**{**base.__dict__, "fanout": fanout})
    return base
