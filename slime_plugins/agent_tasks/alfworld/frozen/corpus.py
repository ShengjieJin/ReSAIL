from __future__ import annotations

import json
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path, PureWindowsPath
from typing import Any

from slime.utils.types import Sample

from .data_source import (
    FROZEN_SHARD_SIZE,
    ITERATIVE_FROZEN_SCHEMA_VERSION,
    load_frozen_trajectory_shard,
    write_frozen_trajectory_shard,
)
from .sampling import collection_turn_seed

ITERATIVE_FROZEN_MANIFEST_SCHEMA_VERSION = 3
FROZEN_MANIFEST_FILENAME = "manifest.json"


def process(args: Any, all_samples: list[Any], data_source: Any) -> None:
    """Order one deployment batch by stream index and write atomic fixed-size shards."""

    # ``generate_rollout_async`` passes the bound ``get_samples`` method here,
    # not the DataSource instance.  Resume identity is already encoded in the
    # samples' global stream indices, so the corpus itself is the authority for
    # the next writable shard.
    del data_source

    configured_root = str(getattr(args, "alfworld_frozen_corpus_dir", "") or "").strip()
    if not configured_root:
        raise ValueError("alfworld_frozen_corpus_dir is required for frozen corpus collection")
    corpus_root = Path(configured_root)
    rows = _flatten_samples(all_samples)
    grouped: dict[int, list[Sample]] = defaultdict(list)
    for row in rows:
        stream_index = (row.metadata or {}).get("sample_group_index")
        if stream_index is None:
            raise ValueError("deployment sample is missing sample_group_index")
        grouped[int(stream_index)].append(row)
    shard_size = int(getattr(args, "alfworld_frozen_shard_size", FROZEN_SHARD_SIZE) or FROZEN_SHARD_SIZE)
    if shard_size <= 0:
        raise ValueError("alfworld_frozen_shard_size must be positive")
    if not grouped or len(grouped) % shard_size != 0:
        raise ValueError(
            "one frozen corpus collection batch must contain a positive multiple of "
            f"{shard_size} trajectories, got {len(grouped)}"
        )
    stream_indices = sorted(grouped)
    if stream_indices[0] % shard_size != 0 or stream_indices != list(
        range(stream_indices[0], stream_indices[0] + len(stream_indices))
    ):
        raise ValueError("frozen corpus batch must contain one aligned contiguous trajectory stream range")
    first_shard_idx = stream_indices[0] // shard_size
    manifest = _manifest_from_args(args)
    manifest_path = corpus_root / FROZEN_MANIFEST_FILENAME
    if manifest_path.exists():
        if _manifest_contract(load_frozen_corpus_manifest(corpus_root)) != _manifest_contract(manifest):
            raise ValueError("frozen corpus collection config does not match the existing manifest")
    else:
        write_frozen_corpus_manifest(corpus_root, **_manifest_write_kwargs(manifest))
    expected_shard_idx = _next_available_shard_index(corpus_root)
    if first_shard_idx != expected_shard_idx:
        raise ValueError(
            f"corpus stream implies shard {first_shard_idx}, but the on-disk corpus expects shard {expected_shard_idx}"
        )
    for offset in range(0, len(stream_indices), shard_size):
        shard_stream_indices = stream_indices[offset : offset + shard_size]
        shard_idx = first_shard_idx + offset // shard_size
        destination = corpus_root / f"batch_{shard_idx:03d}.pt"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing frozen corpus shard: {destination}")
        trajectories = [
            _trajectory_from_rows(grouped[stream_index], stream_index=stream_index, manifest=manifest)
            for stream_index in shard_stream_indices
        ]
        write_frozen_trajectory_shard(
            destination,
            trajectories,
            schema_version=int(manifest["corpus_schema_version"]),
            require_full_shard=False,
        )


def write_frozen_corpus_manifest(
    corpus_root: str | Path,
    *,
    behavior_model: str,
    prompt_contract: Mapping[str, Any],
    sampling_params: Mapping[str, Any],
    seed: int,
    split: str,
    collection_schema: str | None = None,
) -> None:
    manifest = _build_manifest(
        behavior_model=behavior_model,
        prompt_contract=prompt_contract,
        sampling_params=sampling_params,
        seed=seed,
        split=split,
        collection_schema=collection_schema,
    )
    root = Path(corpus_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / FROZEN_MANIFEST_FILENAME
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing frozen corpus manifest: {destination}")
    fd, temporary_name = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=root)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_frozen_corpus_manifest(corpus_root: str | Path) -> dict[str, Any]:
    path = Path(corpus_root) / FROZEN_MANIFEST_FILENAME
    with path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    _validate_manifest(manifest)
    return manifest


def verify_frozen_corpus(
    corpus_root: str | Path,
    *,
    expected_trajectories: int | None = None,
    require_manifest: bool = True,
    enforce_clean: bool = True,
    response_length_finish_max_rate: float = 0.0,
    collect_task_identities: bool = False,
    require_stream_bound_sampling: bool = False,
) -> dict[str, Any]:
    max_length_finish_rate = Fraction(str(response_length_finish_max_rate))
    if not 0 <= max_length_finish_rate <= 1:
        raise ValueError("response_length_finish_max_rate must be in [0, 1]")
    root = Path(corpus_root)
    manifest = load_frozen_corpus_manifest(root) if require_manifest else None
    response_token_cap = int(manifest["prompt_contract"]["response_max_tokens"]) if manifest is not None else None
    shard_paths = sorted(root.glob("batch_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"no batch_*.pt frozen shards found under {corpus_root}")
    trajectory_uids: set[str] = set()
    successful = 0
    turn_count = 0
    trajectory_error_count = 0
    turn_error_count = 0
    prompt_overlength_count = 0
    prompt_truncated_count = 0
    response_length_finish_count = 0
    format_invalid_count = 0
    truncated_trajectory_count = 0
    horizon_trajectory_count = 0
    termination_reason_counts: dict[str, int] = defaultdict(int)
    episode_rewards: list[float] = []
    environment_rewards: list[float] = []
    trajectory_lengths: list[int] = []
    prompt_token_lengths: list[int] = []
    response_token_lengths: list[int] = []
    total_token_lengths: list[int] = []
    shard_stats: list[dict[str, Any]] = []
    task_identity_rows: list[dict[str, Any]] = []
    for expected_shard_idx, shard_path in enumerate(shard_paths):
        expected_name = f"batch_{expected_shard_idx:03d}.pt"
        if shard_path.name != expected_name:
            raise ValueError(
                f"frozen corpus shard sequence has a gap: expected {expected_name}, got {shard_path.name}"
            )
        trajectories = load_frozen_trajectory_shard(shard_path)
        shard_successful = 0
        shard_turns = 0
        shard_prompt_tokens = 0
        shard_response_tokens = 0
        shard_response_length_finish_count = 0
        for trajectory in trajectories:
            if require_stream_bound_sampling:
                provenance = trajectory.get("provenance") or {}
                cycle_seed = int(provenance.get("task_seed", -1))
                stream_index = int(provenance.get("stream_index", -1))
                source_group_index = int(provenance.get("source_group_index", -1))
                expected_source_group_index = random.Random(cycle_seed + stream_index).randrange(2**31)
                if (
                    manifest is None
                    or cycle_seed != int(manifest["seed"])
                    or source_group_index != expected_source_group_index
                    or int(provenance.get("seed", -1)) != cycle_seed + source_group_index
                ):
                    raise ValueError("iterative frozen task identity is not bound to cycle seed and stream index")
                for turn in trajectory.get("turns", []):
                    expected_seed = collection_turn_seed(
                        cycle_seed=cycle_seed,
                        stream_index=stream_index,
                        turn_idx=int(turn.get("turn_idx", -1)),
                    )
                    if turn.get("sampling_seed") != expected_seed:
                        raise ValueError("iterative frozen sampling seed is not bound to cycle/stream/turn")
            if collect_task_identities:
                provenance = trajectory.get("provenance") or {}
                identity = provenance.get("task_identity") or {}
                task_identity_rows.append(
                    {
                        "stream_index": int(provenance.get("stream_index", -1)),
                        "task_seed": int(provenance.get("task_seed", -1)),
                        "source_group_index": int(provenance.get("source_group_index", -1)),
                        "seed": int(provenance.get("seed", -1)),
                        "task_id": str(trajectory.get("task_id", "")),
                        "task_description": str(trajectory.get("task_description", "")),
                        "gamefile": str(identity.get("gamefile", "")),
                    }
                )
            trajectory_uid = trajectory["trajectory_uid"]
            if trajectory_uid in trajectory_uids:
                raise ValueError(f"duplicate trajectory_uid across frozen corpus: {trajectory_uid}")
            trajectory_uids.add(trajectory_uid)
            successful += int(trajectory["success"])
            shard_successful += int(trajectory["success"])
            trajectory_error_count += int(bool(trajectory["errors"]))
            truncated_trajectory_count += int(trajectory["truncated"])
            horizon_trajectory_count += int(trajectory["horizon_reached"])
            termination_reason_counts[trajectory["termination_reason"]] += 1
            episode_rewards.append(float(trajectory["episode_reward"]))
            trajectory_length = len(trajectory["turns"])
            trajectory_lengths.append(trajectory_length)
            turn_count += trajectory_length
            shard_turns += trajectory_length
            for turn in trajectory["turns"]:
                turn_error_count += int(bool(turn["errors"]))
                prompt_overlength_count += int(turn["prompt_overlength"])
                prompt_truncated_count += int(turn["prompt_truncated"])
                response_length_finish_count += int(turn["finish_reason"] == "length")
                shard_response_length_finish_count += int(turn["finish_reason"] == "length")
                format_invalid_count += int(not turn["format_valid"])
                environment_rewards.append(float(turn["env_reward"]))
                prompt_length = len(turn["prompt_ids"])
                response_length = len(turn["response_ids"])
                if response_token_cap is not None and response_length > response_token_cap:
                    raise ValueError(
                        f"frozen response length {response_length} exceeds manifest cap {response_token_cap}"
                    )
                if turn["finish_reason"] == "length" and response_token_cap is not None:
                    if response_length != response_token_cap:
                        raise ValueError("length-finished frozen response must reach the manifest response cap")
                prompt_token_lengths.append(prompt_length)
                response_token_lengths.append(response_length)
                total_token_lengths.append(prompt_length + response_length)
                shard_prompt_tokens += prompt_length
                shard_response_tokens += response_length
        shard_stats.append(
            {
                "shard_index": expected_shard_idx,
                "trajectory_count": len(trajectories),
                "trajectory_uid_first": trajectories[0]["trajectory_uid"],
                "trajectory_uid_last": trajectories[-1]["trajectory_uid"],
                "successful_trajectory_count": shard_successful,
                "failed_trajectory_count": len(trajectories) - shard_successful,
                "turn_count": shard_turns,
                "prompt_token_count": shard_prompt_tokens,
                "response_token_count": shard_response_tokens,
                "response_length_finish_count": shard_response_length_finish_count,
                "response_length_finish_fraction": (
                    shard_response_length_finish_count / shard_turns if shard_turns else 0.0
                ),
            }
        )
    if expected_trajectories is not None and len(trajectory_uids) != int(expected_trajectories):
        raise ValueError(f"expected {expected_trajectories} trajectories, got {len(trajectory_uids)}")
    if enforce_clean:
        violations = {
            "trajectory_errors": trajectory_error_count,
            "turn_errors": turn_error_count,
            "prompt_overlength": prompt_overlength_count,
            "prompt_truncated": prompt_truncated_count,
            "incomplete_termination": termination_reason_counts.get("incomplete", 0),
        }
        if response_length_finish_count * max_length_finish_rate.denominator > (
            max_length_finish_rate.numerator * turn_count
        ):
            violations["response_length_finish_global"] = {
                "count": response_length_finish_count,
                "total": turn_count,
                "max_rate": float(max_length_finish_rate),
            }
        excessive_shards = [
            {
                "shard_index": row["shard_index"],
                "count": row["response_length_finish_count"],
                "total": row["turn_count"],
                "max_rate": float(max_length_finish_rate),
            }
            for row in shard_stats
            if row["response_length_finish_count"] * max_length_finish_rate.denominator
            > max_length_finish_rate.numerator * row["turn_count"]
        ]
        if excessive_shards:
            violations["response_length_finish_shards"] = excessive_shards
        violations = {key: value for key, value in violations.items() if value}
        if violations:
            raise ValueError(f"frozen corpus clean-data gate failed: {violations}")
    stats = {
        "manifest": manifest,
        "shard_count": len(shard_paths),
        "trajectory_count": len(trajectory_uids),
        "successful_trajectory_count": successful,
        "failed_trajectory_count": len(trajectory_uids) - successful,
        "turn_count": turn_count,
        "trajectory_error_count": trajectory_error_count,
        "turn_error_count": turn_error_count,
        "prompt_overlength_count": prompt_overlength_count,
        "prompt_truncated_count": prompt_truncated_count,
        "response_length_finish_count": response_length_finish_count,
        "response_length_finish_fraction": response_length_finish_count / turn_count if turn_count else 0.0,
        "response_length_finish_max_rate": float(max_length_finish_rate),
        "format_invalid_count": format_invalid_count,
        "format_invalid_fraction": format_invalid_count / turn_count if turn_count else 0.0,
        "truncated_trajectory_count": truncated_trajectory_count,
        "horizon_trajectory_count": horizon_trajectory_count,
        "termination_reason_counts": dict(sorted(termination_reason_counts.items())),
        "trajectory_length_distribution": _distribution(trajectory_lengths),
        "prompt_token_length_distribution": _distribution(prompt_token_lengths),
        "response_token_length_distribution": _distribution(response_token_lengths),
        "turn_token_length_distribution": _distribution(total_token_lengths),
        "episode_reward_distribution": _numeric_distribution(episode_rewards),
        "environment_reward_distribution": _numeric_distribution(environment_rewards),
        "shard_stats": shard_stats,
    }
    if collect_task_identities:
        stats["task_identity_rows"] = task_identity_rows
    return stats


def _trajectory_from_rows(
    rows: list[Sample],
    *,
    stream_index: int,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int((row.metadata or {}).get("turn_idx", -1)))
    first = ordered[0]
    first_metadata = first.metadata or {}
    trajectory_uid = f"alfworld-iterative-{stream_index:08d}"
    success_values = {bool((row.metadata or {}).get("success")) for row in ordered}
    if len(success_values) != 1:
        raise ValueError(f"inconsistent success labels in deployment trajectory {trajectory_uid}")
    success = success_values.pop()
    turns = []
    episode_reward_values = {float((row.metadata or {}).get("episode_reward", 0.0)) for row in ordered}
    if len(episode_reward_values) != 1:
        raise ValueError(f"inconsistent episode rewards in deployment trajectory {trajectory_uid}")
    for expected_turn_idx, row in enumerate(ordered):
        metadata = row.metadata or {}
        train_metadata = row.train_metadata if isinstance(row.train_metadata, dict) else {}
        sdpo = train_metadata.get("sdpo") if isinstance(train_metadata.get("sdpo"), dict) else {}
        turn_idx = int(metadata.get("turn_idx", -1))
        if turn_idx != expected_turn_idx:
            raise ValueError(f"deployment trajectory {trajectory_uid} turns are not ordered and contiguous")
        messages = metadata.get("messages")
        if not isinstance(messages, list):
            raise ValueError("frozen corpus collection requires full messages; set AGENT_TASK_KEEP_PROMPT_METADATA=1")
        response_length = int(row.response_length)
        if response_length <= 0 or response_length > len(row.tokens):
            raise ValueError(f"invalid response_length in deployment trajectory {trajectory_uid} turn {turn_idx}")
        if row.rollout_log_probs is None or len(row.rollout_log_probs) != response_length:
            raise ValueError(f"missing or misaligned behavior log-probs in {trajectory_uid} turn {turn_idx}")
        required_metadata = ("format_valid", "finish_reason", "projected_action")
        missing_metadata = [key for key in required_metadata if key not in metadata]
        required_sdpo = ()
        missing_sdpo = [key for key in required_sdpo if key not in sdpo]
        if missing_metadata or missing_sdpo:
            raise ValueError(
                f"deployment trajectory {trajectory_uid} turn {turn_idx} is missing canonical metadata: "
                f"sample={missing_metadata}, sdpo={missing_sdpo}"
            )
        is_terminal = bool(metadata.get("is_terminal", False))
        turn_value = {
            "turn_idx": turn_idx,
            "messages": messages,
            "prompt_ids": [int(token_id) for token_id in row.tokens[:-response_length]],
            "response_text": row.response,
            "response_ids": [int(token_id) for token_id in row.tokens[-response_length:]],
            "deployment_behavior_log_probs": list(row.rollout_log_probs),
            "current_observation": str(metadata.get("current_observation", sdpo.get("anchor_obs", ""))),
            "next_observation": str(metadata.get("next_observation", sdpo.get("next_anchor_obs", ""))),
            "frozen_action": str(metadata.get("projected_action", "")),
            "env_reward": float(metadata.get("score", 0.0)),
            "finish_reason": str(metadata.get("finish_reason", "unknown")),
            "prompt_overlength": bool(metadata.get("prompt_overlength", False)),
            "prompt_truncated": bool(metadata.get("history_auto_truncated", False)),
            "is_terminal": is_terminal,
            "format_valid": bool(metadata.get("format_valid", False)),
            "format_invalid_reason": metadata.get("invalid_reason"),
            "errors": _row_errors(metadata),
        }
        expected_sampling_seed = collection_turn_seed(
            cycle_seed=int(manifest["seed"]),
            stream_index=stream_index,
            turn_idx=turn_idx,
        )
        if int(metadata.get("sampling_seed", -1)) != expected_sampling_seed:
            raise ValueError(
                f"iterative trajectory {trajectory_uid} turn {turn_idx} sampling seed is not stream-bound"
            )
        turn_value["sampling_seed"] = expected_sampling_seed
        turns.append(turn_value)
    task_id = (
        first_metadata.get("task_id")
        or first_metadata.get("source_task_id")
        or f"{first_metadata.get('split', 'train')}:{first_metadata.get('source_group_index', stream_index)}"
    )
    final_metadata = ordered[-1].metadata or {}
    horizon_reached = bool(final_metadata.get("env_horizon_reached", False))
    termination_reason = _episode_termination_reason(final_metadata, horizon_reached=horizon_reached)
    value = {
        "trajectory_uid": trajectory_uid,
        "task_id": str(task_id),
        "task_description": str(
            first_metadata.get("task_description")
            or (first.train_metadata or {}).get("sdpo", {}).get("task_text", task_id)
        ),
        "source_metadata": {
            key: first_metadata[key]
            for key in ("uid", "traj_uid", "seed", "source_group_index", "split")
            if key in first_metadata
        },
        "split": str(first_metadata.get("split", "train")),
        "success": success,
        "outcome": "success" if success else "failure",
        "episode_reward": episode_reward_values.pop(),
        "termination_reason": termination_reason,
        "truncated": bool(horizon_reached or any(turn["finish_reason"] == "length" for turn in turns)),
        "horizon_reached": horizon_reached,
        "errors": [_row_errors(row.metadata or {}) for row in ordered if _row_errors(row.metadata or {})],
        "turns": turns,
    }
    runtime_task_identity = first_metadata.get("runtime_task_identity")
    if not isinstance(runtime_task_identity, dict) or not runtime_task_identity.get("task_description"):
        raise ValueError(f"iterative deployment trajectory {trajectory_uid} lacks runtime task identity")
    source_group_index = int(first_metadata.get("source_group_index", -1))
    seed = int(first_metadata.get("seed", -1))
    if source_group_index < 0 or seed < 0:
        raise ValueError(f"iterative deployment trajectory {trajectory_uid} lacks deterministic source identity")
    value.update(
        {
            "provenance": {
                "stream_index": int(stream_index),
                "task_seed": int(manifest["seed"]),
                "source_group_index": source_group_index,
                "seed": seed,
                "task_identity": {
                    "task_id": str(task_id),
                    "task_description": value["task_description"],
                    "split": value["split"],
                    "gamefile": str(runtime_task_identity.get("gamefile", "")),
                },
            },
            "runtime_task_identity": runtime_task_identity,
        }
    )
    return value


def _row_errors(metadata: dict[str, Any]) -> dict[str, Any]:
    if not bool(metadata.get("episode_error")) and not bool(metadata.get("env_step_failed")):
        return {}
    return {
        key: metadata[key]
        for key in ("episode_error_type", "episode_error_stage", "episode_error_message", "env_step_failed")
        if key in metadata
    }


def _episode_termination_reason(metadata: dict[str, Any], *, horizon_reached: bool) -> str:
    if bool(metadata.get("episode_error")) or bool(metadata.get("env_step_failed")):
        return "error"
    if bool(metadata.get("success")):
        return "success"
    if horizon_reached:
        return "horizon"
    if bool(metadata.get("is_terminal")):
        return "environment_terminal"
    return "incomplete"


def _flatten_samples(value: Any) -> list[Sample]:
    if isinstance(value, Sample):
        return [value]
    rows: list[Sample] = []
    for item in value:
        rows.extend(_flatten_samples(item))
    return rows


def _next_available_shard_index(corpus_root: Path) -> int:
    shard_paths = sorted(corpus_root.glob("batch_*.pt"))
    for expected_shard_idx, shard_path in enumerate(shard_paths):
        expected_name = f"batch_{expected_shard_idx:03d}.pt"
        if shard_path.name != expected_name:
            raise ValueError(
                f"frozen corpus shard sequence has a gap: expected {expected_name}, got {shard_path.name}"
            )
    return len(shard_paths)


def _manifest_from_args(args: Any) -> dict[str, Any]:
    behavior_model = str(
        getattr(args, "alfworld_frozen_behavior_model", "") or os.environ.get("ALFWORLD_FROZEN_BEHAVIOR_MODEL", "")
    ).strip()
    prompt_contract = getattr(args, "alfworld_frozen_prompt_contract", None)
    sampling_params = getattr(args, "alfworld_frozen_sampling_params", None)
    collection_schema = str(getattr(args, "alfworld_frozen_collection_schema", "") or "").strip() or None
    if prompt_contract is None:
        prompt_contract = {
            "name": os.environ.get("ALFWORLD_FROZEN_PROMPT_NAME", ""),
            "response_max_tokens": getattr(args, "rollout_max_response_len", None),
        }
    if sampling_params is None:
        sampling_params = {
            "temperature": getattr(args, "rollout_temperature", None),
            "top_p": getattr(args, "rollout_top_p", None),
            "top_k": getattr(args, "rollout_top_k", None),
            "max_new_tokens": getattr(args, "rollout_max_response_len", None),
            "stop": getattr(args, "rollout_stop", None),
            "stop_token_ids": getattr(args, "rollout_stop_token_ids", None),
            "skip_special_tokens": getattr(args, "rollout_skip_special_tokens", None),
        }
    if not behavior_model:
        raise ValueError("collection requires stable ALFWORLD_FROZEN_BEHAVIOR_MODEL (not a machine-local path)")
    return _build_manifest(
        behavior_model=behavior_model,
        prompt_contract=prompt_contract,
        sampling_params=sampling_params,
        seed=int(getattr(args, "rollout_seed", 42)),
        split=str(getattr(args, "alfworld_train_split", "train")),
        collection_schema=collection_schema,
    )


def _build_manifest(
    *,
    behavior_model: str,
    prompt_contract: Mapping[str, Any],
    sampling_params: Mapping[str, Any],
    seed: int,
    split: str,
    collection_schema: str | None = None,
) -> dict[str, Any]:
    if collection_schema != "iterative":
        raise ValueError("frozen corpus collection requires the iterative schema")
    if any("expectation" in str(key).lower() for key in prompt_contract):
        raise ValueError("frozen prompt contract must not contain Expectation fields")
    manifest = {
        "manifest_schema_version": ITERATIVE_FROZEN_MANIFEST_SCHEMA_VERSION,
        "corpus_schema_version": ITERATIVE_FROZEN_SCHEMA_VERSION,
        "behavior_model": str(behavior_model).strip(),
        "prompt_contract": dict(prompt_contract),
        "sampling_params": dict(sampling_params),
        "seed": int(seed),
        "split": str(split),
    }
    try:
        manifest = json.loads(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise TypeError("frozen corpus manifest must be JSON serializable") from exc
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: Any) -> None:
    required = {
        "manifest_schema_version",
        "corpus_schema_version",
        "behavior_model",
        "prompt_contract",
        "sampling_params",
        "seed",
        "split",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError(f"frozen corpus manifest fields must be {sorted(required)}")
    if manifest["manifest_schema_version"] != ITERATIVE_FROZEN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported frozen corpus manifest schema version")
    if manifest["corpus_schema_version"] != ITERATIVE_FROZEN_SCHEMA_VERSION:
        raise ValueError("frozen corpus manifest does not match the corpus schema version")
    if not isinstance(manifest["behavior_model"], str) or not manifest["behavior_model"]:
        raise ValueError("manifest behavior_model must be a non-empty stable identifier")
    if not isinstance(manifest["prompt_contract"], dict) or not manifest["prompt_contract"]:
        raise ValueError("manifest prompt_contract must be a non-empty object")
    required_prompt_contract = {"name", "response_max_tokens"}
    missing_prompt_contract = required_prompt_contract - manifest["prompt_contract"].keys()
    if missing_prompt_contract:
        raise ValueError(f"manifest prompt_contract missing fields: {sorted(missing_prompt_contract)}")
    if not isinstance(manifest["prompt_contract"]["name"], str) or not manifest["prompt_contract"]["name"]:
        raise ValueError("manifest prompt_contract name must be non-empty")
    if manifest["prompt_contract"]["response_max_tokens"] != 1024:
        raise ValueError("frozen corpus prompt contract requires response_max_tokens=1024")
    expectation_keys = [key for key in manifest["prompt_contract"] if "expectation" in str(key).lower()]
    if expectation_keys:
        raise ValueError(f"prompt contract contains Expectation fields: {expectation_keys}")
    if not isinstance(manifest["sampling_params"], dict) or not manifest["sampling_params"]:
        raise ValueError("manifest sampling_params must be a non-empty object")
    required_sampling_params = {"temperature", "top_p", "top_k", "max_new_tokens"}
    missing_sampling_params = required_sampling_params - manifest["sampling_params"].keys()
    if missing_sampling_params:
        raise ValueError(f"manifest sampling_params missing fields: {sorted(missing_sampling_params)}")
    if manifest["sampling_params"]["max_new_tokens"] != 1024:
        raise ValueError("manifest sampling_params max_new_tokens must be 1024")
    if not isinstance(manifest["seed"], int):
        raise TypeError("manifest seed must be int")
    if not isinstance(manifest["split"], str) or not manifest["split"]:
        raise ValueError("manifest split must be a non-empty string")
    _reject_manifest_forbidden(manifest, location="manifest")
    try:
        json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("frozen corpus manifest must be JSON serializable") from exc


def _reject_manifest_forbidden(value: Any, *, location: str) -> None:
    if isinstance(value, dict):
        forbidden = {
            key for key in value if any(fragment in str(key).lower() for fragment in ("hash", "digest", "checksum"))
        }
        if forbidden:
            raise ValueError(f"{location} contains forbidden fields: {sorted(forbidden)}")
        for key, item in value.items():
            _reject_manifest_forbidden(item, location=f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_manifest_forbidden(item, location=f"{location}[{index}]")
        return
    if isinstance(value, str) and (
        Path(value).is_absolute() or PureWindowsPath(value).is_absolute() or value.startswith("~")
    ):
        raise ValueError(f"{location} must not contain a machine-local path")


def _manifest_write_kwargs(manifest: dict[str, Any]) -> dict[str, Any]:
    values = {
        "behavior_model": manifest["behavior_model"],
        "prompt_contract": manifest["prompt_contract"],
        "sampling_params": manifest["sampling_params"],
        "seed": manifest["seed"],
        "split": manifest["split"],
    }
    values["collection_schema"] = "iterative"
    return values


def _manifest_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return dict(manifest)


def _distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "p50": 0, "p95": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": _nearest_rank(ordered, 0.50),
        "p95": _nearest_rank(ordered, 0.95),
    }


def _numeric_distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _nearest_rank(ordered: list[int], quantile: float) -> int:
    index = max(0, min(len(ordered) - 1, int((len(ordered) * quantile) + 0.999999) - 1))
    return ordered[index]


def _environment_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")
