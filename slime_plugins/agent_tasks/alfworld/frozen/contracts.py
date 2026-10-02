from __future__ import annotations

from typing import Any

from slime_plugins.agent_tasks.common.privilege import contains_privileged_student_input

PRIVILEGED_PROMPT_MARKERS = (
    "<expectation>",
    "</expectation>",
    "reference trajectory from a successful previous attempt",
    "reference trajectory from a failed attempt",
    "a successful trajectory for the current task:",
    "a failed trajectory for the current task:",
    "a trajectory for the current task:",
    "successful trajectory evidence:",
    "unsuccessful trajectory evidence:",
    "use this information as a reference and continue solving the original task.",
)
PRIVILEGED_METADATA_PREFIXES = (
    "teacher_",
    "sdpo_teacher_",
    "sdpo_privileged_",
    "privileged_",
)


def student_prompt_privilege_violation(prompt_text: str, *, metadata_keys: Any = ()) -> bool:
    return contains_privileged_student_input(
        prompt_text,
        metadata_keys=metadata_keys,
        prompt_markers=PRIVILEGED_PROMPT_MARKERS,
        metadata_prefixes=PRIVILEGED_METADATA_PREFIXES,
    )


def expected_trajectory_count(args: Any) -> int:
    """Return the configured frozen-corpus size."""

    configured = int(getattr(args, "alfworld_frozen_expected_trajectories", 0) or 0)
    return configured or 960
