from __future__ import annotations

from typing import Any

from slime_plugins.agent_tasks.alfworld.frozen import rewards as _alfworld_backend
from slime.utils.types import Sample


def post_process_action_match_rewards(args: Any, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """Use the established step-equal bundle reducer without changing its math."""

    return _alfworld_backend.post_process_action_match_rewards(args, samples)
