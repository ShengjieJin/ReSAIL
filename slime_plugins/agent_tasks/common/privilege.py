from __future__ import annotations

from collections.abc import Iterable


def contains_privileged_student_input(
    prompt_text: str,
    *,
    metadata_keys: Iterable[object] = (),
    prompt_markers: tuple[str, ...] = (),
    metadata_prefixes: tuple[str, ...] = (),
) -> bool:
    """Return whether a student-visible input contains a forbidden surface."""

    lowered = str(prompt_text).lower()
    return any(marker.lower() in lowered for marker in prompt_markers) or any(
        str(key).lower().startswith(tuple(prefix.lower() for prefix in metadata_prefixes)) for key in metadata_keys
    )
