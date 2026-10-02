from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.algorithms.sgs import sgs_filtering_enabled
from slime_plugins.agent_tasks.common.frozen.guidance import guidance_metadata
from slime_plugins.agent_tasks.common.segments import build_step_segment_sample

from ..generate import (
    _build_action_content_token_metadata,
    _finish_type,
    _get_tokenizer,
    _response_tokens_and_log_probs,
    _validate_finish_type,
)
from ..projection import project_response
from .capabilities import capabilities_for_args
from .contracts import expected_trajectory_count, student_prompt_privilege_violation
from .sampling import sampling_seed, sampling_seed_namespace
from ..runtime import assert_alfworld_offline_environment_disabled


async def generate(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> list[Sample]:
    if evaluation:
        raise ValueError("frozen ALFWorld generation is train-only; use the normal environment-backed eval path")
    assert_alfworld_offline_environment_disabled(args)
    trajectory = (sample.metadata or {}).get("frozen_trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError("frozen ALFWorld sample metadata must contain frozen_trajectory")
    arm = str((sample.metadata or {}).get("frozen_arm", ""))
    capabilities = capabilities_for_args(args, arm)
    for turn in trajectory.get("turns", []):
        prompt_text = "\n".join(str(message.get("content", "")) for message in turn.get("messages", []))
        if student_prompt_privilege_violation(prompt_text):
            raise RuntimeError("frozen student prefix contains privileged teacher text")
    if capabilities.response_source == "epd":
        return _replay_epd_targets(args, sample=sample, trajectory=trajectory)
    if capabilities.response_source == "deployment":
        return _replay_trajectory(args, sample=sample, trajectory=trajectory, arm=arm)
    if capabilities.response_source == "current_actor":
        return await _branch_trajectory(
            args,
            sample=sample,
            trajectory=trajectory,
            arm=arm,
            sampling_params=sampling_params,
        )
    raise ValueError(f"unsupported frozen ALFWorld arm: {arm!r}")


def _replay_trajectory(args: Any, *, sample: Sample, trajectory: dict[str, Any], arm: str) -> list[Sample]:
    capabilities = capabilities_for_args(args, arm)
    offline_sdpo = capabilities.response_source == "deployment" and capabilities.objective == "sdpo"
    tokenizer = _get_tokenizer(args)
    rows: list[Sample] = []
    turns = trajectory["turns"]
    source_turn_idx = (sample.metadata or {}).get("source_turn_idx")
    if source_turn_idx is not None:
        source_turn_idx = int(source_turn_idx)
        if source_turn_idx < 0 or source_turn_idx >= len(turns):
            raise ValueError(f"source_turn_idx is outside frozen trajectory: {source_turn_idx}")
        turns = [turns[source_turn_idx]]
    for turn in turns:
        response_ids = list(turn["response_ids"])
        metadata = _row_metadata(sample, trajectory=trajectory, turn=turn, arm=arm)
        extra_train_metadata = None
        loss_mask = [1] * len(response_ids)
        if capabilities.objective == "sdpo":
            sdpo = _sdpo_metadata(trajectory, turn, metadata)
            if sgs_filtering_enabled(args):
                projection = project_response(turn["response_text"])
                action_match = bool(
                    projection.format_valid
                    and bool(turn.get("format_valid", False))
                    and projection.projected_action == turn["frozen_action"]
                )
                metadata["action_match"] = action_match
                sdpo.update(
                    {
                        **_build_action_content_token_metadata(
                            tokenizer,
                            response_text=turn["response_text"],
                            response_ids=response_ids,
                            format_valid=projection.format_valid,
                        ),
                        "sgs_action_match": action_match,
                        "sgs_online_action": projection.projected_action,
                        "sgs_online_format_valid": projection.format_valid,
                        "sgs_frozen_action": turn["frozen_action"],
                        "sgs_source_draw_id": int(metadata["source_draw_id"]),
                        "sgs_task_id": str(trajectory["task_id"]),
                        "sgs_split": str(trajectory["split"]),
                    }
                )
            extra_train_metadata = {"sdpo": sdpo}
            if getattr(args, "agent_frozen_guidance_summary_dir", None):
                extra_train_metadata.update(guidance_metadata(args, trajectory))
        row = build_step_segment_sample(
            base_sample=sample,
            tokenizer=tokenizer,
            prompt_ids=turn["prompt_ids"],
            response_ids=response_ids,
            rollout_log_probs=turn["deployment_behavior_log_probs"].tolist(),
            reward=0.0 if offline_sdpo else float(trajectory["success"]),
            sample_index=_turn_sample_index(sample, turn["turn_idx"]),
            rollout_id=_bundle_rollout_id(sample),
            metadata=metadata,
            response_text=turn["response_text"],
            session_id=str(trajectory["trajectory_uid"]),
            extra_train_metadata=extra_train_metadata,
        )
        row.metadata.pop("frozen_trajectory", None)
        row.loss_mask = loss_mask
        row.status = (
            Sample.Status.TRUNCATED
            if offline_sdpo and turn["finish_reason"] == "length"
            else Sample.Status.COMPLETED
        )
        rows.append(row)
    return rows


_EPD_TARGET_CACHE: dict[tuple[Any, ...], tuple[dict[str, Any], dict[str, dict[str, Any]]]] = {}


def _load_epd_target_index(args: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    target_root = str(getattr(args, "alfworld_epd_target_dir", "") or "").strip()
    if not target_root:
        raise ValueError("EPD training requires alfworld_epd_target_dir")
    subset_target_contract = bool(getattr(args, "alfworld_epd_subset_target_contract", False))
    expected_trajectories = (
        int(getattr(args, "alfworld_frozen_expected_subset_trajectories", 0) or 0)
        if subset_target_contract
        else 0
    ) or expected_trajectory_count(args)
    direct_binding = getattr(args, "alfworld_epd_direct_binding", None)
    if not isinstance(direct_binding, dict):
        raise ValueError("EPD training requires alfworld_epd_direct_binding")
    binding_key = ("direct_v1", json.dumps(direct_binding, sort_keys=True))
    cache_key = (target_root, expected_trajectories, *binding_key)
    cached = _EPD_TARGET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from .epd import (
        EPD_MANIFEST_SCHEMA_VERSION,
        canonical_identity,
        load_target_records,
    )

    manifest_path = Path(target_root) / "manifest.json"
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"EPD target manifest is missing: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ValueError("EPD target manifest must have status=complete")
    if int(manifest.get("schema_version", -1)) != EPD_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported EPD target schema: {manifest.get('schema_version')}")
    verification_path = Path(target_root) / "verification.json"
    try:
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("EPD target directory must have a valid verification.json") from exc
    if (
        manifest.get("binding_mode") != "direct_v1"
        or manifest.get("direct_binding") != direct_binding
        or verification.get("status") != "verified"
        or verification.get("binding_mode") != "direct_v1"
        or verification.get("direct_binding") != direct_binding
        or verification.get("corpus_bound") is not True
        or verification.get("trajectory_count") != expected_trajectories
        or verification.get("canonical_turn_count") != manifest.get("canonical_turn_count")
        or int(getattr(args, "alfworld_epd_target_count", -1)) != manifest.get("target_count")
        or verification.get("target_count") != manifest.get("target_count")
    ):
        raise ValueError("EPD direct target verification differs from the configured binding")
    if (
        int(manifest.get("target_count", -1)) <= 0
        or int(manifest.get("target_count", -1)) != int(manifest.get("canonical_turn_count", -2))
        or int(manifest.get("trajectory_count", -1)) != expected_trajectories
    ):
        raise ValueError(
            "EPD target manifest must contain every canonical step in all " f"{expected_trajectories:,} trajectories"
        )
    records = load_target_records(target_root)
    expected_count = int(manifest.get("target_count", -1))
    if expected_count <= 0 or len(records) != expected_count:
        raise ValueError(f"EPD target count mismatch: manifest={expected_count}, records={len(records)}")
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("EPD target records must be objects")
        identity = str(record.get("identity", ""))
        trajectory_uid = str(record.get("trajectory_uid", ""))
        turn_idx = int(record.get("turn_idx", -1))
        if identity != canonical_identity(trajectory_uid, turn_idx):
            raise ValueError(f"EPD target identity mismatch: {identity!r}")
        if identity in index:
            raise ValueError(f"duplicate EPD target identity: {identity}")
        response_ids = record.get("response_ids")
        if not isinstance(response_ids, list) or not response_ids:
            raise ValueError(f"EPD target has no response_ids: {identity}")
        validation = record.get("validation", {})
        if not isinstance(validation.get("valid"), bool):
            raise ValueError(f"EPD target validation status is missing: {identity}")
        indexed_record = {
            "canonical_index": int(record.get("canonical_index", -1)),
            "original_prompt_ids": [int(token_id) for token_id in record.get("original_prompt_ids", [])],
            "response_ids": [int(token_id) for token_id in response_ids],
            "response_text": str(record.get("response_text", "")),
            "response_log_probs": record.get("response_log_probs"),
            "target_finish_reason": str(record.get("finish_reason", "")),
            "target_retry_count": int(record.get("retry_count", 0)),
            "target_primary_seed": int(record.get("primary_seed", 42)),
            "target_teacher_signal_type": record.get("teacher_signal_type"),
            "target_teacher_reference_scope": record.get("teacher_reference_scope"),
            "target_format_valid": bool(validation.get("format_valid", False)),
            "target_validation_valid": bool(validation["valid"]),
            "target_invalid_reason": validation.get("invalid_reason"),
            "target_selected_attempt": int(record.get("selected_attempt", record.get("attempt_count", 1))),
        }
        index[identity] = indexed_record
    expected_indices = list(range(expected_count))
    actual_indices = sorted(item["canonical_index"] for item in index.values())
    if actual_indices != expected_indices:
        raise ValueError("EPD target canonical indices are not a complete permutation")
    cached = (manifest, index)
    _EPD_TARGET_CACHE[cache_key] = cached
    return cached


def _replay_epd_targets(args: Any, *, sample: Sample, trajectory: dict[str, Any]) -> list[Sample]:
    tokenizer = _get_tokenizer(args)
    manifest, targets = _load_epd_target_index(args)
    source_turn_idx = (sample.metadata or {}).get("source_turn_idx")
    turns = trajectory["turns"]
    if source_turn_idx is not None:
        source_turn_idx = int(source_turn_idx)
        if source_turn_idx < 0 or source_turn_idx >= len(turns):
            raise ValueError(f"source_turn_idx is outside frozen trajectory: {source_turn_idx}")
        turns = [turns[source_turn_idx]]

    rows: list[Sample] = []
    from .epd import canonical_identity

    for turn in turns:
        identity = canonical_identity(str(trajectory["trajectory_uid"]), int(turn["turn_idx"]))
        target = targets.get(identity)
        if target is None:
            raise KeyError(f"EPD target is missing for {identity}")
        original_prompt_ids = [int(token_id) for token_id in turn["prompt_ids"]]
        if target["original_prompt_ids"] != original_prompt_ids:
            raise ValueError(f"EPD target original prompt mismatch for {identity}")
        response_ids = list(target["response_ids"])
        raw_log_probs = target.get("response_log_probs")
        if isinstance(raw_log_probs, list) and len(raw_log_probs) == len(response_ids):
            response_log_probs = [float(value) for value in raw_log_probs]
            logs_available = True
        else:
            response_log_probs = [0.0] * len(response_ids)
            logs_available = False
        metadata = _row_metadata(sample, trajectory=trajectory, turn=turn, arm=str(sample.metadata["frozen_arm"]))
        target_metadata = {
                "epd_target_identity": identity,
                "epd_target_canonical_index": target["canonical_index"],
                "epd_target_manifest_schema_version": int(manifest["schema_version"]),
                "epd_target_finish_reason": target["target_finish_reason"],
                "epd_target_retry_count": target["target_retry_count"],
                "epd_target_primary_seed": target["target_primary_seed"],
                "epd_target_teacher_signal_type": target["target_teacher_signal_type"],
                "epd_target_teacher_reference_scope": target["target_teacher_reference_scope"],
                "epd_target_log_probs_available": logs_available,
                "epd_target_format_valid": target.get("target_format_valid", True),
                "epd_target_validation_valid": target.get("target_validation_valid", True),
                "epd_target_invalid_reason": target.get("target_invalid_reason"),
                "epd_target_selected_attempt": target.get("target_selected_attempt", 1),
            }
        target_metadata.update(
            {
                "epd_target_binding_mode": "direct_v1",
                "epd_target_materialization_identity": manifest["direct_binding"]["materialization_identity"],
            }
        )
        metadata.update(target_metadata)
        row = build_step_segment_sample(
            base_sample=sample,
            tokenizer=tokenizer,
            prompt_ids=original_prompt_ids,
            response_ids=response_ids,
            rollout_log_probs=response_log_probs,
            reward=0.0,
            sample_index=_turn_sample_index(sample, turn["turn_idx"]),
            rollout_id=_bundle_rollout_id(sample),
            metadata=metadata,
            response_text=target["response_text"],
            session_id=None,
        )
        row.metadata.pop("frozen_trajectory", None)
        row.loss_mask = [1] * len(response_ids)
        row.reward = 0.0
        row.status = Sample.Status.COMPLETED
        rows.append(row)
    return rows


async def _branch_trajectory(
    args: Any,
    *,
    sample: Sample,
    trajectory: dict[str, Any],
    arm: str,
    sampling_params: dict[str, Any],
) -> list[Sample]:
    capabilities = capabilities_for_args(args, arm)
    tokenizer = _get_tokenizer(args)
    session_id = (
        f"{trajectory['trajectory_uid']}-draw-{int(sample.metadata['source_draw_id']):08d}"
        f"-branch-{sample.metadata['branch_idx']:02d}"
    )
    source_turn_idx = (sample.metadata or {}).get("source_turn_idx")
    turns = trajectory["turns"]
    if source_turn_idx is not None:
        source_turn_idx = int(source_turn_idx)
        if source_turn_idx < 0 or source_turn_idx >= len(turns):
            raise ValueError(f"source_turn_idx is outside frozen trajectory: {source_turn_idx}")
        turns = [turns[source_turn_idx]]
    turn_sampling_params = [
        _branch_sampling_params(
            args,
            sample=sample,
            trajectory_uid=str(trajectory["trajectory_uid"]),
            turn_idx=int(turn["turn_idx"]),
            arm=arm,
            base=sampling_params,
        )
        for turn in turns
    ]
    outputs = await asyncio.gather(
        *(
            generate_branch_response(
                args,
                prompt_ids=list(turn["prompt_ids"]),
                sampling_params=turn_params,
                session_id=session_id,
            )
            for turn, turn_params in zip(turns, turn_sampling_params, strict=True)
        )
    )
    rows: list[Sample] = []
    for turn, output, turn_params in zip(turns, outputs, turn_sampling_params, strict=True):
        response_text = str(output.get("text", ""))
        meta_info = dict(output.get("meta_info") or {})
        response_ids, response_log_probs = _response_tokens_and_log_probs(meta_info)
        if not response_ids:
            raise RuntimeError("frozen ALFWorld branch generation returned no response tokens")
        finish_type = _finish_type(meta_info)
        _validate_finish_type(finish_type)
        projection = project_response(response_text)
        metadata = _row_metadata(sample, trajectory=trajectory, turn=turn, arm=arm)
        metadata.update(
            {
                "student_projected_action": projection.projected_action,
                "student_format_valid": projection.format_valid,
                "student_invalid_reason": projection.invalid_reason,
                "finish_reason": finish_type,
                "sampling_seed": turn_params.get("sampling_seed"),
            }
        )
        reward = 0.0
        extra_train_metadata = None
        loss_mask = [1] * len(response_ids)
        if capabilities.objective == "grpo":
            if capabilities.strict_action_format_match:
                reward = float(
                    projection.format_valid
                    and bool(turn.get("format_valid", False))
                    and projection.projected_action == turn["frozen_action"]
                )
            else:
                reward = float(projection.projected_action == turn["frozen_action"])
            metadata["action_match"] = bool(reward)
            metadata["uid"] = f"{trajectory['trajectory_uid']}:turn:{turn['turn_idx']:03d}"
        else:
            action_match = bool(
                projection.format_valid
                and bool(turn.get("format_valid", False))
                and projection.projected_action == turn["frozen_action"]
            )
            metadata["action_match"] = action_match
            sdpo = _sdpo_metadata(trajectory, turn, metadata)
            if sgs_filtering_enabled(args):
                sdpo.update(
                    {
                        **_build_action_content_token_metadata(
                            tokenizer,
                            response_text=response_text,
                            response_ids=response_ids,
                            format_valid=projection.format_valid,
                        ),
                        "sgs_action_match": action_match,
                        "sgs_online_action": projection.projected_action,
                        "sgs_online_format_valid": projection.format_valid,
                        "sgs_frozen_action": turn["frozen_action"],
                        "sgs_source_draw_id": int(metadata["source_draw_id"]),
                        "sgs_task_id": str(trajectory["task_id"]),
                        "sgs_split": str(trajectory["split"]),
                    }
                )
            sdpo["algorithm_active_mask"] = True
            extra_train_metadata = {"sdpo": sdpo}
            if getattr(args, "agent_frozen_guidance_summary_dir", None):
                extra_train_metadata.update(guidance_metadata(args, trajectory))
        row = build_step_segment_sample(
            base_sample=sample,
            tokenizer=tokenizer,
            prompt_ids=turn["prompt_ids"],
            response_ids=response_ids,
            rollout_log_probs=response_log_probs,
            reward=reward,
            sample_index=_turn_sample_index(sample, turn["turn_idx"]),
            rollout_id=_bundle_rollout_id(sample),
            metadata=metadata,
            response_text=response_text,
            session_id=session_id,
            extra_train_metadata=extra_train_metadata,
        )
        row.metadata.pop("frozen_trajectory", None)
        row.loss_mask = loss_mask
        row.status = Sample.Status.TRUNCATED if finish_type == "length" else Sample.Status.COMPLETED
        rows.append(row)
    return rows


def _branch_sampling_params(
    args: Any,
    *,
    sample: Sample,
    trajectory_uid: str,
    turn_idx: int,
    arm: str,
    base: dict[str, Any],
) -> dict[str, Any]:
    """Return an isolated deterministic seed for one frozen source step.

    Methods with the same sampling namespace use common random numbers at the
    same update and source step.
    """

    values = dict(base)
    update = int((sample.metadata or {}).get("training_update", 0))
    branch_idx = int((sample.metadata or {}).get("branch_idx", 0))
    if sampling_seed_namespace(args, arm) is None:
        return values
    base_seed = int(getattr(args, "rollout_seed", 42))
    values["sampling_seed"] = sampling_seed(
        args,
        arm=arm,
        base_seed=base_seed,
        update=update,
        trajectory_uid=trajectory_uid,
        turn_idx=turn_idx,
        branch_idx=branch_idx,
    )
    return values


async def generate_branch_response(
    args: Any,
    *,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    """Generate one branch from exact frozen prompt IDs without an environment call."""

    generator = getattr(args, "alfworld_frozen_generator", None)
    if generator is not None:
        return await generator(
            args=args,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            session_id=session_id,
        )
    from slime.rollout.sglang_rollout import get_model_url
    from slime.utils.http_utils import post

    headers = None
    if session_id and getattr(args, "router_policy", None) == "consistent_hashing":
        headers = {"X-SMG-Routing-Key": session_id}
    return await post(
        get_model_url(args, str(getattr(args, "alfworld_sglang_model_name", "default")), "/generate"),
        {"input_ids": prompt_ids, "sampling_params": sampling_params, "return_logprob": True},
        headers=headers,
    )


def _row_metadata(
    sample: Sample,
    *,
    trajectory: dict[str, Any],
    turn: dict[str, Any],
    arm: str,
) -> dict[str, Any]:
    return {
        "uid": trajectory["trajectory_uid"],
        "traj_uid": sample.metadata["traj_uid"],
        "source_trajectory_uid": trajectory["trajectory_uid"],
        "source_draw_id": int(sample.metadata["source_draw_id"]),
        "source_turn_idx": int(turn["turn_idx"]),
        "turn_idx": int(turn["turn_idx"]),
        "branch_idx": int(sample.metadata["branch_idx"]),
        "bundle_rollout_id": _bundle_rollout_id(sample),
        "frozen_arm": arm,
        "frozen_action": turn["frozen_action"],
        "current_observation": turn["current_observation"],
        "next_observation": turn["next_observation"],
        "outcome": trajectory["outcome"],
        "success": trajectory["success"],
        "task_id": trajectory["task_id"],
        "split": trajectory["split"],
    }


def _sdpo_metadata(
    trajectory: dict[str, Any],
    turn: dict[str, Any],
    row_metadata: dict[str, Any],
) -> dict[str, Any]:
    messages = turn["messages"]
    current_prompt_text = str(messages[-1]["content"]) if messages else ""
    metadata = {
        "uid": trajectory["trajectory_uid"],
        "traj_uid": row_metadata["traj_uid"],
        "source_trajectory_uid": trajectory["trajectory_uid"],
        "turn_idx": int(turn["turn_idx"]),
        "task_text": str(trajectory.get("task_description") or trajectory["task_id"]),
        "anchor_obs": turn["current_observation"],
        "next_anchor_obs": turn["next_observation"],
        "projected_action": turn["frozen_action"],
        "is_action_valid": not bool(turn["errors"]),
        "is_terminal": int(turn["turn_idx"]) == len(trajectory["turns"]) - 1,
        "episode_rewards": float(trajectory["success"]),
        "episode_lengths": len(trajectory["turns"]),
        "sdpo_current_prompt_text": current_prompt_text,
        "sdpo_current_raw_prompt": messages,
        "sdpo_metadata_profile": "alfworld",
        "frozen_outcome_label": trajectory["outcome"],
        "frozen_trajectory_uid": trajectory["trajectory_uid"],
        "frozen_trajectory_length": len(trajectory["turns"]),
    }
    if capabilities_for_args(None, row_metadata["frozen_arm"]).privileged_teacher_trajectory:
        metadata["sdpo_privileged_trajectory"] = [
            {
                "turn_idx": int(frozen_turn["turn_idx"]),
                "projected_action": frozen_turn["frozen_action"],
                "next_anchor_obs": frozen_turn["next_observation"],
            }
            for frozen_turn in trajectory["turns"]
        ]
    return metadata


def _bundle_rollout_id(sample: Sample) -> int:
    value = sample.rollout_id if sample.rollout_id is not None else sample.index
    if value is None:
        raise ValueError("frozen ALFWorld sample requires an integer rollout_id or index")
    return int(value)


def _turn_sample_index(sample: Sample, turn_idx: int) -> int:
    return _bundle_rollout_id(sample) * 1_000_000 + int(turn_idx)
