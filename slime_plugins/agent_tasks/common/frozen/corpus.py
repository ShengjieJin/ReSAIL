from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.artifacts import write_json_idempotent

from .contracts import ITERATIVE_FROZEN_SCHEMA_VERSION, load_shard, write_shard
from .sampling import stream_turn_seed


def process_iterative_corpus(args: Any, all_samples: list[Any], *, task: str) -> None:
    root_value = str(getattr(args, "agent_frozen_corpus_dir", "") or "").strip()
    if not root_value:
        raise ValueError("agent_frozen_corpus_dir is required for iterative collection")
    root = Path(root_value)
    rows = _flatten(all_samples)
    grouped: dict[int, list[Sample]] = defaultdict(list)
    for row in rows:
        stream_index = (row.metadata or {}).get("sample_group_index")
        if stream_index is None:
            raise ValueError("iterative collection row is missing sample_group_index")
        grouped[int(stream_index)].append(row)
    shard_size = int(getattr(args, "agent_frozen_shard_size", 64) or 64)
    stream_indices = sorted(grouped)
    if (
        shard_size <= 0
        or not stream_indices
        or len(stream_indices) % shard_size
        or stream_indices != list(range(stream_indices[0], stream_indices[0] + len(stream_indices)))
        or stream_indices[0] % shard_size
    ):
        raise ValueError("iterative collection must be an aligned contiguous multiple of shard_size")
    manifest = _manifest(args, task=task)
    manifest_path = root / "manifest.json"
    write_json_idempotent(manifest_path, manifest, indent=2)
    existing = sorted(root.glob("batch_*.pt"))
    for index, path in enumerate(existing):
        if path.name != f"batch_{index:03d}.pt":
            raise ValueError("iterative corpus shard sequence has a gap")
    first_shard = stream_indices[0] // shard_size
    if first_shard != len(existing):
        raise ValueError("iterative stream position differs from existing shard count")
    for offset in range(0, len(stream_indices), shard_size):
        shard_index = first_shard + offset // shard_size
        trajectories = [
            _trajectory(grouped[index], task=task, stream_index=index, task_seed=int(manifest["seed"]))
            for index in stream_indices[offset : offset + shard_size]
        ]
        write_shard(root / f"batch_{shard_index:03d}.pt", trajectories)


def verify_iterative_corpus(
    corpus_dir: str | Path,
    *,
    expected_trajectories: int,
    expected_task: str | None = None,
    expected_cycle_seed: int | None = None,
) -> dict[str, Any]:
    root = Path(corpus_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "iterative_frozen_corpus":
        raise ValueError("unsupported iterative corpus manifest")
    if expected_task is not None and manifest.get("task") != expected_task:
        raise ValueError("iterative corpus task differs from expectation")
    if expected_cycle_seed is not None and manifest.get("seed") != int(expected_cycle_seed):
        raise ValueError("iterative corpus seed differs from expectation")
    trajectories = []
    paths = sorted(root.glob("batch_*.pt"))
    for index, path in enumerate(paths):
        if path.name != f"batch_{index:03d}.pt":
            raise ValueError("iterative corpus shard sequence has a gap")
        trajectories.extend(load_shard(path))
    if len(trajectories) != int(expected_trajectories):
        raise ValueError(f"expected {expected_trajectories} trajectories, found {len(trajectories)}")
    for stream_index, trajectory in enumerate(trajectories):
        provenance = trajectory["provenance"]
        if provenance["stream_index"] != stream_index:
            raise ValueError("iterative corpus stream indices are not contiguous")
        for turn in trajectory["turns"]:
            expected_seed = stream_turn_seed(
                task=str(manifest["task"]),
                cycle_seed=int(manifest["seed"]),
                stream_index=stream_index,
                turn_idx=int(turn["turn_idx"]),
            )
            if turn["sampling_seed"] != expected_seed:
                raise ValueError("iterative corpus turn seed is not stream-bound")
    return {
        "trajectory_count": len(trajectories),
        "turn_count": sum(len(value["turns"]) for value in trajectories),
        "success_count": sum(int(value["success"]) for value in trajectories),
        "task_identity_rows": [value["provenance"] for value in trajectories],
    }


def _manifest(args: Any, *, task: str) -> dict[str, Any]:
    behavior_model = str(getattr(args, "agent_frozen_behavior_model", "") or "").strip()
    if not behavior_model:
        raise ValueError("agent_frozen_behavior_model must be a stable non-empty identifier")
    prompt_name = str(getattr(args, "agent_frozen_prompt_name", f"{task}-ordinary-v1") or "").strip()
    return {
        "schema_version": 1,
        "kind": "iterative_frozen_corpus",
        "corpus_schema_version": ITERATIVE_FROZEN_SCHEMA_VERSION,
        "task": task,
        "behavior_model": behavior_model,
        "seed": int(getattr(args, "rollout_seed", 42)),
        "split": str(getattr(args, f"{task}_train_split", "train")),
        "prompt_contract": {
            "name": prompt_name,
            "response_max_tokens": int(getattr(args, "rollout_max_response_len", 1024)),
        },
        "sampling_params": {
            "temperature": getattr(args, "rollout_temperature", None),
            "top_p": getattr(args, "rollout_top_p", None),
            "top_k": getattr(args, "rollout_top_k", None),
            "max_new_tokens": getattr(args, "rollout_max_response_len", None),
        },
    }


def _trajectory(rows: list[Sample], *, task: str, stream_index: int, task_seed: int) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int((row.metadata or {}).get("turn_idx", -1)))
    first_metadata = ordered[0].metadata or {}
    for turn_idx, row in enumerate(ordered):
        if int((row.metadata or {}).get("turn_idx", -1)) != turn_idx:
            raise ValueError("iterative trajectory turns are not contiguous")
    successes = {bool((row.metadata or {}).get("success")) for row in ordered}
    rewards = {float((row.metadata or {}).get("episode_reward", 0.0)) for row in ordered}
    if len(successes) != 1 or len(rewards) != 1:
        raise ValueError("iterative trajectory outcome metadata is inconsistent")
    success = successes.pop()
    runtime_identity = first_metadata.get("runtime_task_identity")
    if not isinstance(runtime_identity, dict) or not runtime_identity.get("task_description"):
        raise ValueError("iterative collection requires runtime_task_identity")
    turns = []
    for row in ordered:
        metadata = row.metadata or {}
        response_length = int(row.response_length)
        messages = metadata.get("messages")
        if not isinstance(messages, list):
            raise ValueError("iterative collection requires full messages; enable prompt metadata")
        if response_length <= 0 or row.rollout_log_probs is None or len(row.rollout_log_probs) != response_length:
            raise ValueError("iterative collection requires response-aligned behavior log-probs")
        turns.append(
            {
                "turn_idx": int(metadata["turn_idx"]),
                "messages": messages,
                "prompt_ids": [int(value) for value in row.tokens[:-response_length]],
                "response_text": str(row.response),
                "response_ids": [int(value) for value in row.tokens[-response_length:]],
                "deployment_behavior_log_probs": [float(value) for value in row.rollout_log_probs],
                "current_observation": str(metadata.get("current_observation", "")),
                "next_observation": str(metadata.get("next_observation", "")),
                "current_agent_coord": metadata.get("prompt_agent_coord"),
                "current_box_coords": metadata.get("prompt_box_coords"),
                "current_target_coords": metadata.get("prompt_target_coords"),
                "next_agent_coord": metadata.get("next_agent_coord"),
                "next_box_coords": metadata.get("next_box_coords"),
                "next_target_coords": metadata.get("next_target_coords"),
                "frozen_action": str(metadata.get("projected_action", "")),
                "action_effective": metadata.get("action_is_effective"),
                "env_reward": float(metadata.get("score", 0.0)),
                "finish_reason": str(metadata.get("finish_reason", "")),
                "prompt_overlength": bool(metadata.get("prompt_overlength", False)),
                "prompt_truncated": bool(metadata.get("history_auto_truncated", False)),
                "is_terminal": bool(metadata.get("is_terminal", False)),
                "format_valid": bool(metadata.get("format_valid", False)),
                "format_invalid_reason": metadata.get("invalid_reason"),
                "errors": _errors(metadata),
                "sampling_seed": int(metadata.get("sampling_seed", -1)),
            }
        )
    final = ordered[-1].metadata or {}
    horizon = bool(final.get("env_horizon_reached", False))
    termination = (
        "success"
        if success
        else (
            "error"
            if _errors(final)
            else "horizon" if horizon else "environment_terminal" if final.get("is_terminal") else "incomplete"
        )
    )
    task_id = str(first_metadata.get("task_id", f"{task}:{first_metadata.get('source_group_index', stream_index)}"))
    source_group_index = int(first_metadata.get("source_group_index", -1))
    seed = int(first_metadata.get("seed", -1))
    if source_group_index < 0 or seed < 0:
        raise ValueError("iterative collection requires deterministic source identity")
    return {
        "trajectory_uid": f"{task}-iterative-{stream_index:08d}",
        "task_id": task_id,
        "task_description": str(first_metadata.get("task_description") or runtime_identity["task_description"]),
        "source_metadata": {
            key: first_metadata[key]
            for key in ("uid", "traj_uid", "seed", "source_group_index", "split")
            if key in first_metadata
        },
        "split": str(first_metadata.get("split", "train")),
        "success": success,
        "outcome": "success" if success else "failure",
        "episode_reward": rewards.pop(),
        "termination_reason": termination,
        "truncated": horizon or any(turn["finish_reason"] == "length" for turn in turns),
        "horizon_reached": horizon,
        "errors": [_errors(row.metadata or {}) for row in ordered if _errors(row.metadata or {})],
        "runtime_task_identity": dict(runtime_identity),
        "provenance": {
            "stream_index": stream_index,
            "task_seed": task_seed,
            "source_group_index": source_group_index,
            "seed": seed,
            "task_identity": dict(runtime_identity),
        },
        "turns": turns,
    }


def _errors(metadata: dict[str, Any]) -> dict[str, Any]:
    if not metadata.get("episode_error") and not metadata.get("env_step_failed"):
        return {}
    return {
        key: metadata[key]
        for key in ("episode_error_type", "episode_error_stage", "episode_error_message", "env_step_failed")
        if key in metadata
    }


def _flatten(value: Any) -> list[Sample]:
    if isinstance(value, Sample):
        return [value]
    rows = []
    for item in value:
        rows.extend(_flatten(item))
    return rows
