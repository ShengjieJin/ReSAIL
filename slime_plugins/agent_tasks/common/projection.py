from __future__ import annotations

import re
from dataclasses import dataclass

ACTION_OPEN = "<action>"
ACTION_CLOSE = "</action>"
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class TaggedActionProjection:
    projected_action: str
    format_valid: bool
    missing_action_tag: bool
    contains_chinese: bool
    invalid_reason: str | None = None
    missing_thinking_tag: bool = False


def project_tagged_action_response(
    response: str,
    *,
    action_open: str = ACTION_OPEN,
    action_close: str = ACTION_CLOSE,
    require_thinking_tag: bool = True,
    fallback_chars: int = 30,
    lowercase_action: bool = True,
) -> TaggedActionProjection:
    lowered = response.lower()
    invalid_reasons: list[str] = []
    start = lowered.find(action_open)
    end = lowered.find(action_close, start + len(action_open))
    missing_action = start == -1 or end == -1 or end < start
    if missing_action:
        action = response[-fallback_chars:].strip()
        invalid_reasons.append("missing_action_tag")
    else:
        action = response[start + len(action_open) : end].strip()
    if lowercase_action:
        action = action.lower()

    missing_thinking_tag = not _has_thinking_tag(lowered)
    if require_thinking_tag and missing_thinking_tag:
        invalid_reasons.append("missing_thinking_tag")

    contains_chinese = bool(CHINESE_RE.search(response))
    if contains_chinese:
        invalid_reasons.append("contains_chinese")

    return TaggedActionProjection(
        projected_action=action,
        format_valid=not invalid_reasons,
        missing_action_tag=missing_action,
        contains_chinese=contains_chinese,
        invalid_reason=",".join(invalid_reasons) if invalid_reasons else None,
        missing_thinking_tag=missing_thinking_tag,
    )


def _has_thinking_tag(lowered_response: str) -> bool:
    return ("<think>" in lowered_response and "</think>" in lowered_response) or (
        "<thinking>" in lowered_response and "</thinking>" in lowered_response
    )
