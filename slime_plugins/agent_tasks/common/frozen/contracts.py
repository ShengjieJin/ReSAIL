from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


ITERATIVE_FROZEN_SCHEMA_VERSION = 4


def verify_trajectory(trajectory: dict[str, Any]) -> None:
    required = {
        "trajectory_uid",
        "task_id",
        "task_description",
        "split",
        "success",
        "outcome",
        "episode_reward",
        "termination_reason",
        "truncated",
        "horizon_reached",
        "errors",
        "runtime_task_identity",
        "provenance",
        "turns",
    }
    if not isinstance(trajectory, dict) or required - trajectory.keys():
        raise ValueError(f"iterative frozen trajectory missing fields: {sorted(required - set(trajectory or {}))}")
    if not isinstance(trajectory["success"], bool):
        raise TypeError("iterative frozen success must be bool")
    if trajectory["outcome"] != ("success" if trajectory["success"] else "failure"):
        raise ValueError("iterative frozen outcome disagrees with success")
    runtime_identity = trajectory["runtime_task_identity"]
    if not isinstance(runtime_identity, dict) or not runtime_identity.get("task_description"):
        raise ValueError("iterative frozen trajectory requires runtime task identity")
    provenance = trajectory["provenance"]
    provenance_keys = {"stream_index", "task_seed", "source_group_index", "seed", "task_identity"}
    if not isinstance(provenance, dict):
        raise TypeError("iterative frozen provenance must be a mapping")
    if provenance_keys - provenance.keys():
        raise ValueError(f"iterative frozen provenance missing fields: {sorted(provenance_keys - provenance.keys())}")
    for key in ("stream_index", "task_seed", "source_group_index", "seed"):
        if not isinstance(provenance[key], int):
            raise TypeError(f"iterative frozen provenance {key} must be int")
    turns = trajectory["turns"]
    if not isinstance(turns, list) or not turns:
        raise ValueError("iterative frozen trajectory requires non-empty turns")
    for turn_idx, turn in enumerate(turns):
        _verify_turn(turn, turn_idx)
    if any(bool(turn["is_terminal"]) for turn in turns[:-1]):
        raise ValueError("only the final iterative frozen turn may be terminal")


def _verify_turn(turn: dict[str, Any], expected_turn_idx: int) -> None:
    required = {
        "turn_idx",
        "messages",
        "prompt_ids",
        "response_text",
        "response_ids",
        "deployment_behavior_log_probs",
        "current_observation",
        "next_observation",
        "frozen_action",
        "env_reward",
        "finish_reason",
        "prompt_overlength",
        "prompt_truncated",
        "is_terminal",
        "format_valid",
        "format_invalid_reason",
        "errors",
        "sampling_seed",
    }
    if not isinstance(turn, dict) or required - turn.keys():
        raise ValueError(f"iterative frozen turn missing fields: {sorted(required - set(turn or {}))}")
    if int(turn["turn_idx"]) != expected_turn_idx:
        raise ValueError("iterative frozen turns must be ordered and contiguous")
    prompt_ids, response_ids = turn["prompt_ids"], turn["response_ids"]
    if not prompt_ids or not all(isinstance(value, int) for value in prompt_ids):
        raise ValueError("iterative frozen prompt_ids must be non-empty list[int]")
    if not response_ids or not all(isinstance(value, int) for value in response_ids):
        raise ValueError("iterative frozen response_ids must be non-empty list[int]")
    log_probs = torch.as_tensor(turn["deployment_behavior_log_probs"], dtype=torch.float32)
    if log_probs.ndim != 1 or len(log_probs) != len(response_ids) or not torch.isfinite(log_probs).all():
        raise ValueError("iterative frozen behavior log-probs must be finite and response-aligned")
    if turn["finish_reason"] not in {"stop", "length"}:
        raise ValueError("iterative frozen finish_reason must be stop or length")
    if not isinstance(turn["sampling_seed"], int) or turn["sampling_seed"] <= 0:
        raise ValueError("iterative frozen turn requires a positive sampling_seed")


def write_shard(path: str | Path, trajectories: list[dict[str, Any]]) -> None:
    if not trajectories:
        raise ValueError("iterative frozen shard must not be empty")
    canonical = [_canonicalize(value) for value in trajectories]
    for trajectory in canonical:
        verify_trajectory(trajectory)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        torch.save(
            {
                "schema_version": ITERATIVE_FROZEN_SCHEMA_VERSION,
                "shard_size": len(canonical),
                "trajectories": canonical,
            },
            temporary_path,
        )
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_shard(path: str | Path) -> list[dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != ITERATIVE_FROZEN_SCHEMA_VERSION:
        raise ValueError(f"unsupported iterative frozen shard: {path}")
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list) or payload.get("shard_size") != len(trajectories):
        raise ValueError(f"invalid iterative frozen shard size: {path}")
    for trajectory in trajectories:
        verify_trajectory(trajectory)
    return trajectories


def _canonicalize(trajectory: dict[str, Any]) -> dict[str, Any]:
    value = dict(trajectory)
    value["episode_reward"] = float(value["episode_reward"])
    value["turns"] = []
    for raw_turn in trajectory["turns"]:
        turn = dict(raw_turn)
        turn["prompt_ids"] = [int(token_id) for token_id in turn["prompt_ids"]]
        turn["response_ids"] = [int(token_id) for token_id in turn["response_ids"]]
        turn["deployment_behavior_log_probs"] = torch.as_tensor(
            turn["deployment_behavior_log_probs"], dtype=torch.float32
        ).cpu()
        turn["env_reward"] = float(turn["env_reward"])
        value["turns"].append(turn)
    return value
