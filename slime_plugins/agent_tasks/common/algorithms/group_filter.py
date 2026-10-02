from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slime.utils.types import Sample


@dataclass(frozen=True)
class ConstantGroupFilterResult:
    attempted: int
    accepted: int
    all_zero: int
    all_one: int
    rows: list[dict[str, Any]]

    @property
    def rejected(self) -> int:
        return self.attempted - self.accepted

    @property
    def active_fraction(self) -> float:
        return self.accepted / self.attempted


def filter_constant_binary_reward_groups(
    args: Any,
    groups: list[tuple[dict[str, Any], list[Sample]]],
    *,
    reason_key: str,
    active_key: str,
    require_action_match_metadata: bool = False,
) -> ConstantGroupFilterResult:
    """Mark all-zero/all-one response groups inactive without replacement.

    Task-specific callers own grouping and identity validation. This function
    owns the reusable binary-reward classification and mutation exactly once.
    """

    if not groups:
        raise ValueError("constant-group filter received no groups")
    accepted = all_zero = all_one = 0
    audit_rows: list[dict[str, Any]] = []
    for identity, rows in groups:
        if not rows:
            raise ValueError("constant-group filter received an empty group")
        rewards = [float(row.get_reward_value(args)) for row in rows]
        if any(reward not in {0.0, 1.0} for reward in rewards):
            raise ValueError("constant-group filter requires binary rewards")
        if require_action_match_metadata:
            matches = [float(bool((row.metadata or {}).get("action_match", False))) for row in rows]
            if rewards != matches:
                raise ValueError("reward/action_match metadata disagree")
        if all(reward == 0.0 for reward in rewards):
            reason = "all_zero"
            all_zero += 1
        elif all(reward == 1.0 for reward in rewards):
            reason = "all_one"
            all_one += 1
        else:
            reason = None
            accepted += 1
        for row in rows:
            row.remove_sample = reason is not None
            metadata = row.metadata if row.metadata is not None else {}
            row.metadata = metadata
            metadata[reason_key] = reason
            metadata[active_key] = reason is None
        audit_rows.append(
            {
                **identity,
                "reward_sum": int(sum(rewards)),
                "active": reason is None,
                "reason": reason,
            }
        )
    return ConstantGroupFilterResult(
        attempted=len(groups),
        accepted=accepted,
        all_zero=all_zero,
        all_one=all_one,
        rows=audit_rows,
    )
