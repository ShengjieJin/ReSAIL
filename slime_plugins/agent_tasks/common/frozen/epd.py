"""Canonical TextCraft-capable facade for frozen EPD materialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from slime_plugins.agent_tasks.alfworld.frozen import epd as _backend

EPD_MANIFEST_SCHEMA_VERSION = _backend.EPD_MANIFEST_SCHEMA_VERSION
canonical_identity = _backend.canonical_identity
load_target_records = _backend.load_target_records
parse_generation_output = _backend.parse_generation_output


def canonical_steps_from_trajectories(
    trajectories: Iterable[dict[str, Any]],
    *,
    expected_trajectories: int | None = None,
    expected_steps: int | None = None,
) -> list[dict[str, Any]]:
    from .contracts import verify_trajectory

    values = list(trajectories)
    if expected_trajectories is not None and len(values) != int(expected_trajectories):
        raise ValueError(f"expected {expected_trajectories} trajectories, found {len(values)}")
    steps = []
    seen = set()
    for trajectory in values:
        verify_trajectory(trajectory)
        uid = str(trajectory["trajectory_uid"])
        if uid in seen:
            raise ValueError(f"duplicate trajectory_uid: {uid}")
        seen.add(uid)
        for turn in trajectory["turns"]:
            turn_idx = int(turn["turn_idx"])
            steps.append(
                {
                    "identity": canonical_identity(uid, turn_idx),
                    "trajectory_uid": uid,
                    "turn_idx": turn_idx,
                    "trajectory": trajectory,
                    "turn": turn,
                    "success": bool(trajectory["success"]),
                    "outcome": str(trajectory["outcome"]),
                    "split": str(trajectory["split"]),
                    "task_id": str(trajectory["task_id"]),
                }
            )
    steps.sort(key=lambda item: (item["trajectory_uid"], item["turn_idx"]))
    for canonical_index, step in enumerate(steps):
        step["canonical_index"] = canonical_index
    if expected_steps is not None and len(steps) != int(expected_steps):
        raise ValueError(f"expected {expected_steps} canonical turns, found {len(steps)}")
    return steps


def canonical_steps_from_corpus(
    corpus_dir: str | Path,
    *,
    expected_trajectories: int,
    expected_steps: int | None = None,
) -> list[dict[str, Any]]:
    from .contracts import load_shard

    root = Path(corpus_dir)
    paths = sorted(root.glob("batch_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no batch_*.pt shards found under {root}")
    trajectories = []
    locations = {}
    for shard_index, path in enumerate(paths):
        expected_name = f"batch_{shard_index:03d}.pt"
        if path.name != expected_name:
            raise ValueError(f"frozen corpus shard sequence has a gap: expected {expected_name}, got {path.name}")
        for trajectory_index, trajectory in enumerate(load_shard(path)):
            trajectories.append(trajectory)
            locations[str(trajectory["trajectory_uid"])] = (path.name, trajectory_index)
    steps = canonical_steps_from_trajectories(
        trajectories,
        expected_trajectories=expected_trajectories,
        expected_steps=expected_steps,
    )
    for step in steps:
        step["source_shard"], step["source_trajectory_index"] = locations[step["trajectory_uid"]]
    return steps


def build_sampling_params(
    seed: int = 42,
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    values = _backend.build_sampling_params(seed)
    values.update({"temperature": float(temperature), "max_new_tokens": int(max_new_tokens)})
    return values


def build_teacher_context(args: Any, trajectory: dict[str, Any], turn: dict[str, Any]) -> dict[str, Any]:
    return _backend.build_teacher_context(args, trajectory, turn)


def validate_teacher_output(
    response_text: str,
    finish_type: str,
    response_ids: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    from slime_plugins.agent_tasks.textcraft.projection import project_response

    text = str(response_text or "")
    token_ids = [int(token_id) for token_id in response_ids]
    projection = project_response(text)
    reasons = []
    if not text.strip():
        reasons.append("empty_response")
    if not token_ids:
        reasons.append("empty_response_ids")
    if str(finish_type) != "stop":
        reasons.append(f"finish_reason_{finish_type or 'missing'}")
    if not projection.format_valid:
        reasons.append(projection.invalid_reason or "invalid_action_format")
    if not projection.projected_action:
        reasons.append("empty_action")
    return {
        "valid": not reasons,
        "truncated": str(finish_type) == "length",
        "format_valid": bool(projection.format_valid),
        "projected_action": projection.projected_action,
        "invalid_reason": ",".join(reasons) if reasons else None,
        "response_token_count": len(token_ids),
    }


async def materialize_records(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    kwargs.setdefault("metadata_profile", "textcraft")
    kwargs.setdefault("teacher_output_validator", validate_teacher_output)
    kwargs.setdefault("sampling_params_overrides", {"max_new_tokens": 512})
    return await _backend.materialize_records(*args, **kwargs)

__all__ = [
    "EPD_MANIFEST_SCHEMA_VERSION",
    "build_sampling_params",
    "build_teacher_context",
    "canonical_identity",
    "canonical_steps_from_corpus",
    "canonical_steps_from_trajectories",
    "load_target_records",
    "materialize_records",
    "parse_generation_output",
    "validate_teacher_output",
]
