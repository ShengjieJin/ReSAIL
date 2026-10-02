from __future__ import annotations

import re
from dataclasses import dataclass

from slime_plugins.agent_tasks.common.projection import (
    ACTION_CLOSE,
    ACTION_OPEN,
    CHINESE_RE,
    project_tagged_action_response,
)

TEXTCRAFT_ACTION_RE = re.compile(r"^(inventory|get [0-9]+ [a-z0-9 ]+|craft .+ using .+)$")


@dataclass(frozen=True)
class TextCraftProjectionResult:
    projected_action: str
    format_valid: bool
    missing_action_tag: bool
    contains_chinese: bool
    action_kind: str | None = None
    invalid_reason: str | None = None


def project_response(response: str) -> TextCraftProjectionResult:
    tagged = project_tagged_action_response(response, lowercase_action=False)
    action = sanitize_textcraft_action(tagged.projected_action)
    action_kind = _action_kind(action)
    invalid_reasons = [reason for reason in (tagged.invalid_reason or "").split(",") if reason]
    if action_kind is None:
        invalid_reasons.append("invalid_textcraft_action")
    return TextCraftProjectionResult(
        projected_action=action,
        format_valid=tagged.format_valid and action_kind is not None,
        missing_action_tag=tagged.missing_action_tag,
        contains_chinese=tagged.contains_chinese,
        action_kind=action_kind,
        invalid_reason=",".join(invalid_reasons) if invalid_reasons else None,
    )


def sanitize_textcraft_action(action: str) -> str:
    action = re.sub(r"[^A-Za-z0-9, ]+", "", action)
    return " ".join(action.split()).strip()


def _action_kind(action: str) -> str | None:
    if not action:
        return None
    if not TEXTCRAFT_ACTION_RE.match(action):
        return None
    return action.split(" ", 1)[0]
