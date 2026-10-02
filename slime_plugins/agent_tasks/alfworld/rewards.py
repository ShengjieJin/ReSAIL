from __future__ import annotations

from typing import Any

from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.rewards import post_process_grouped_rewards

REWARD_GROUP_KEY = "uid"


def post_process_rewards(args: Any, samples: list[Sample]) -> tuple[list[float], list[float]]:
    return post_process_grouped_rewards(args, samples, group_key=REWARD_GROUP_KEY)
