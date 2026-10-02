from __future__ import annotations

from slime_plugins.agent_tasks.common.projection import (
    ACTION_CLOSE,
    ACTION_OPEN,
    CHINESE_RE,
    TaggedActionProjection,
    project_tagged_action_response,
)

ProjectionResult = TaggedActionProjection


def project_response(response: str) -> ProjectionResult:
    return project_tagged_action_response(response)
