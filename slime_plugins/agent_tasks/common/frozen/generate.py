from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.algorithms.sgs import sgs_filtering_enabled
from slime_plugins.agent_tasks.common.segments import build_step_segment_sample

from .capabilities import capabilities_for_arm
from .epd import EPD_MANIFEST_SCHEMA_VERSION, canonical_identity, load_target_records
from .sampling import branch_turn_seed
from .guidance import guidance_metadata


async def generate(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> list[Sample]:
    if evaluation:
        raise ValueError("common frozen generation is train-only")
    task = str(getattr(args, "agent_frozen_task", "") or "").strip()
    if task != "textcraft":
        raise ValueError(f"unsupported common frozen task adapter: {task!r}")
    trajectory = (sample.metadata or {}).get("frozen_trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError("common frozen sample requires frozen_trajectory")
    _reject_privileged_student_prefix(trajectory)
    arm = str((sample.metadata or {}).get("frozen_arm", ""))
    capabilities = capabilities_for_arm(arm)
    if capabilities.response_source == "deployment":
        return _replay_deployment(args, sample=sample, trajectory=trajectory, arm=arm)
    if capabilities.response_source == "epd":
        return _replay_epd(args, sample=sample, trajectory=trajectory, arm=arm)
    return await _branch_current_actor(
        args,
        sample=sample,
        trajectory=trajectory,
        arm=arm,
        sampling_params=sampling_params,
    )


def _replay_deployment(args: Any, *, sample: Sample, trajectory: dict[str, Any], arm: str) -> list[Sample]:
    capabilities = capabilities_for_arm(arm)
    tokenizer = _tokenizer(args)
    rows = []
    for turn in trajectory["turns"]:
        response_ids = list(turn["response_ids"])
        metadata = _row_metadata(sample, trajectory, turn, arm)
        train_metadata = None
        if capabilities.objective == "sdpo":
            sdpo = _sdpo_metadata(args, trajectory, turn, metadata, capabilities)
            if sgs_filtering_enabled(args):
                response_text = str(turn["response_text"])
                projected_action, format_valid, _ = _project_response(response_text)
                action_match = bool(
                    format_valid
                    and bool(turn.get("format_valid", False))
                    and projected_action == turn["frozen_action"]
                )
                metadata["action_match"] = action_match
                sdpo.update(
                    {
                        **_action_token_metadata(
                            tokenizer,
                            response_text=response_text,
                            response_ids=response_ids,
                            format_valid=format_valid,
                        ),
                        "sgs_action_match": action_match,
                        "sgs_online_action": projected_action,
                        "sgs_online_format_valid": format_valid,
                        "sgs_frozen_action": turn["frozen_action"],
                        "sgs_source_draw_id": int(metadata["source_draw_id"]),
                        "sgs_task_id": str(trajectory["task_id"]),
                        "sgs_split": str(trajectory["split"]),
                    }
                )
            train_metadata = {"sdpo": sdpo}
            if getattr(args, "agent_frozen_guidance_summary_dir", None):
                train_metadata.update(guidance_metadata(args, trajectory))
        row = build_step_segment_sample(
            base_sample=sample,
            tokenizer=tokenizer,
            prompt_ids=turn["prompt_ids"],
            response_ids=response_ids,
            rollout_log_probs=turn["deployment_behavior_log_probs"].tolist(),
            reward=float(trajectory["success"]),
            sample_index=_sample_index(sample, turn["turn_idx"]),
            rollout_id=_rollout_id(sample),
            metadata=metadata,
            response_text=turn["response_text"],
            session_id=str(trajectory["trajectory_uid"]),
            extra_train_metadata=train_metadata,
        )
        row.metadata.pop("frozen_trajectory", None)
        rows.append(row)
    return rows


async def _branch_current_actor(
    args: Any,
    *,
    sample: Sample,
    trajectory: dict[str, Any],
    arm: str,
    sampling_params: dict[str, Any],
) -> list[Sample]:
    capabilities = capabilities_for_arm(arm)
    tokenizer = _tokenizer(args)
    branch_idx = int((sample.metadata or {}).get("branch_idx", 0))
    update = int((sample.metadata or {}).get("training_update", 0))
    namespace = str(getattr(args, "agent_frozen_sampling_seed_namespace", "") or capabilities.sampling_namespace or "")
    if not namespace:
        raise ValueError(f"{arm} requires a deterministic sampling namespace")
    params_by_turn = []
    for turn in trajectory["turns"]:
        params = dict(sampling_params)
        params["sampling_seed"] = branch_turn_seed(
            namespace=namespace,
            base_seed=int(getattr(args, "rollout_seed", 42)),
            update=update,
            trajectory_uid=str(trajectory["trajectory_uid"]),
            turn_idx=int(turn["turn_idx"]),
            branch_idx=branch_idx,
            include_branch=capabilities.objective == "grpo",
        )
        params_by_turn.append(params)
    session_id = (
        f"{trajectory['trajectory_uid']}-draw-{int(sample.metadata['source_draw_id']):08d}-branch-{branch_idx:02d}"
    )
    outputs = await asyncio.gather(
        *(
            _generate_response(args, prompt_ids=turn["prompt_ids"], sampling_params=params, session_id=session_id)
            for turn, params in zip(trajectory["turns"], params_by_turn, strict=True)
        )
    )
    rows = []
    for turn, params, output in zip(trajectory["turns"], params_by_turn, outputs, strict=True):
        text = str(output.get("text", ""))
        response_ids, response_log_probs = _response_tokens(output)
        if not response_ids:
            raise RuntimeError(f"{_task(args)} frozen generation returned no response tokens")
        finish = _finish_type(output)
        if finish not in {"stop", "length"}:
            raise RuntimeError(f"unsupported {_task(args)} frozen finish reason: {finish!r}")
        projected_action, format_valid, invalid_reason = _project_response(text)
        metadata = _row_metadata(sample, trajectory, turn, arm)
        action_match = bool(format_valid and bool(turn["format_valid"]) and projected_action == turn["frozen_action"])
        metadata.update(
            {
                "student_projected_action": projected_action,
                "student_format_valid": format_valid,
                "student_invalid_reason": invalid_reason,
                "finish_reason": finish,
                "sampling_seed": params["sampling_seed"],
                "action_match": action_match,
            }
        )
        reward = float(action_match) if capabilities.objective == "grpo" else 0.0
        if capabilities.objective == "grpo":
            metadata["uid"] = f"{trajectory['trajectory_uid']}:turn:{int(turn['turn_idx']):03d}"
        train_metadata = None
        if capabilities.objective == "sdpo":
            sdpo = _sdpo_metadata(args, trajectory, turn, metadata, capabilities)
            if sgs_filtering_enabled(args):
                sdpo.update(
                    {
                        **_action_token_metadata(
                            tokenizer,
                            response_text=text,
                            response_ids=response_ids,
                            format_valid=format_valid,
                        ),
                        "sgs_action_match": action_match,
                        "sgs_online_action": projected_action,
                        "sgs_online_format_valid": format_valid,
                        "sgs_frozen_action": turn["frozen_action"],
                        "sgs_source_draw_id": int(metadata["source_draw_id"]),
                        "sgs_task_id": str(trajectory["task_id"]),
                        "sgs_split": str(trajectory["split"]),
                    }
                )
            train_metadata = {"sdpo": sdpo}
            if getattr(args, "agent_frozen_guidance_summary_dir", None):
                train_metadata.update(guidance_metadata(args, trajectory))
        row = build_step_segment_sample(
            base_sample=sample,
            tokenizer=tokenizer,
            prompt_ids=turn["prompt_ids"],
            response_ids=response_ids,
            rollout_log_probs=response_log_probs,
            reward=reward,
            sample_index=_sample_index(sample, turn["turn_idx"]),
            rollout_id=_rollout_id(sample),
            metadata=metadata,
            response_text=text,
            session_id=session_id,
            extra_train_metadata=train_metadata,
        )
        row.metadata.pop("frozen_trajectory", None)
        row.status = Sample.Status.TRUNCATED if finish == "length" else Sample.Status.COMPLETED
        rows.append(row)
    return rows


def _replay_epd(args: Any, *, sample: Sample, trajectory: dict[str, Any], arm: str) -> list[Sample]:
    manifest, targets = _epd_targets(args)
    tokenizer = _tokenizer(args)
    rows = []
    for turn in trajectory["turns"]:
        identity = canonical_identity(str(trajectory["trajectory_uid"]), int(turn["turn_idx"]))
        target = targets.get(identity)
        if target is None or [int(value) for value in target.get("original_prompt_ids", [])] != list(
            turn["prompt_ids"]
        ):
            raise ValueError(f"EPD target is missing or prompt-misaligned: {identity}")
        response_ids = [int(value) for value in target["response_ids"]]
        log_probs = target.get("response_log_probs")
        response_log_probs = (
            [float(value) for value in log_probs]
            if isinstance(log_probs, list) and len(log_probs) == len(response_ids)
            else [0.0] * len(response_ids)
        )
        metadata = _row_metadata(sample, trajectory, turn, arm)
        metadata.update(
            {
                "epd_target_identity": identity,
                "epd_target_canonical_index": int(target["canonical_index"]),
                "epd_target_binding_mode": "direct_v1",
                "epd_target_materialization_identity": manifest["direct_binding"]["materialization_identity"],
            }
        )
        row = build_step_segment_sample(
            base_sample=sample,
            tokenizer=tokenizer,
            prompt_ids=turn["prompt_ids"],
            response_ids=response_ids,
            rollout_log_probs=response_log_probs,
            reward=0.0,
            sample_index=_sample_index(sample, turn["turn_idx"]),
            rollout_id=_rollout_id(sample),
            metadata=metadata,
            response_text=str(target.get("response_text", "")),
            extra_train_metadata=None,
        )
        row.metadata.pop("frozen_trajectory", None)
        rows.append(row)
    return rows


def _epd_targets(args: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(str(getattr(args, "agent_frozen_epd_target_dir", "") or ""))
    binding = getattr(args, "agent_frozen_epd_direct_binding", None)
    if not root.is_dir() or not isinstance(binding, dict):
        raise ValueError("EPD direct_v1 requires target directory and direct binding")
    expected_trajectories = int(getattr(args, "agent_frozen_expected_trajectories", 0) or 0)
    cache_key = (
        str(root.resolve()),
        json.dumps(binding, sort_keys=True, separators=(",", ":")),
        expected_trajectories,
    )
    cached = getattr(args, "_agent_frozen_epd_target_cache", None)
    if isinstance(cached, tuple) and len(cached) == 3 and cached[0] == cache_key:
        return cached[1], cached[2]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    empty_count = 0
    if str(getattr(args, "agent_frozen_empty_guideline_policy", "") or "") == "skip_trajectory":
        guidance_root = Path(str(getattr(args, "agent_frozen_guidance_summary_dir", "") or ""))
        guidance_manifest = json.loads((guidance_root / "manifest.json").read_text(encoding="utf-8"))
        empty_count = int(guidance_manifest.get("empty_guideline_count", 0))
    effective_trajectories = expected_trajectories - empty_count
    if (
        manifest.get("schema_version") != EPD_MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("binding_mode") != "direct_v1"
        or manifest.get("direct_binding") != binding
        or int(manifest.get("trajectory_count", -1)) != effective_trajectories
        or int(verification.get("trajectory_count", -1)) != effective_trajectories
        or int(verification.get("empty_guideline_count", 0)) != empty_count
        or verification.get("status") != "verified"
        or verification.get("direct_binding") != binding
        or verification.get("corpus_bound") is not True
    ):
        raise ValueError("EPD target verification differs from configured direct binding")
    records = load_target_records(root)
    if len(records) != int(manifest.get("target_count", -1)):
        raise ValueError("EPD target count differs from manifest")
    index = {str(record["identity"]): record for record in records}
    if len(index) != len(records):
        raise ValueError("EPD target identities are not unique")
    setattr(args, "_agent_frozen_epd_target_cache", (cache_key, manifest, index))
    return manifest, index


def _sdpo_metadata(args, trajectory, turn, row_metadata, capabilities) -> dict[str, Any]:
    messages = turn["messages"]
    default_profile = "textcraft"
    profile = str(getattr(args, "agent_task_sdpo_metadata_profile", default_profile) or default_profile)
    value = {
        "uid": trajectory["trajectory_uid"],
        "traj_uid": row_metadata["traj_uid"],
        "source_trajectory_uid": trajectory["trajectory_uid"],
        "turn_idx": int(turn["turn_idx"]),
        "task_text": str(trajectory["task_description"]),
        "anchor_obs": turn["current_observation"],
        "next_anchor_obs": turn["next_observation"],
        "projected_action": turn["frozen_action"],
        "is_action_valid": not bool(turn["errors"]),
        "is_terminal": int(turn["turn_idx"]) == len(trajectory["turns"]) - 1,
        "episode_rewards": float(trajectory["success"]),
        "episode_lengths": len(trajectory["turns"]),
        "sdpo_current_prompt_text": str(messages[-1]["content"]) if messages else "",
        "sdpo_current_raw_prompt": messages,
        "sdpo_metadata_profile": profile,
        "frozen_outcome_label": trajectory["outcome"],
        "frozen_trajectory_uid": trajectory["trajectory_uid"],
        "frozen_trajectory_length": len(trajectory["turns"]),
        "format_valid": bool(turn.get("format_valid", False)),
    }
    if capabilities.privileged_teacher_trajectory:
        privileged = []
        for item in trajectory["turns"]:
            record = {
                "turn_idx": int(item["turn_idx"]),
                "task_text": str(trajectory["task_description"]),
                "sdpo_metadata_profile": profile,
                "projected_action": item["frozen_action"],
                "next_anchor_obs": item["next_observation"],
                "is_terminal": int(item["turn_idx"]) == len(trajectory["turns"]) - 1,
                "episode_rewards": float(trajectory["success"]),
            }
            privileged.append(record)
        value["sdpo_privileged_trajectory"] = privileged
    return value


def _row_metadata(sample, trajectory, turn, arm) -> dict[str, Any]:
    return {
        "uid": trajectory["trajectory_uid"],
        "traj_uid": sample.metadata["traj_uid"],
        "source_trajectory_uid": trajectory["trajectory_uid"],
        "source_draw_id": int(sample.metadata["source_draw_id"]),
        "source_turn_idx": int(turn["turn_idx"]),
        "turn_idx": int(turn["turn_idx"]),
        "branch_idx": int(sample.metadata["branch_idx"]),
        "bundle_rollout_id": _rollout_id(sample),
        "frozen_arm": arm,
        "agent_frozen_task": str((sample.metadata or {}).get("agent_frozen_task") or ""),
        "frozen_action": turn["frozen_action"],
        "current_observation": turn["current_observation"],
        "next_observation": turn["next_observation"],
        "outcome": trajectory["outcome"],
        "success": trajectory["success"],
        "task_id": trajectory["task_id"],
        "split": trajectory["split"],
        "training_update": int(sample.metadata["training_update"]),
    }


async def _generate_response(args, *, prompt_ids, sampling_params, session_id):
    task = _task(args)
    generator = getattr(args, f"{task}_frozen_generator", None)
    if generator is not None:
        return await generator(
            args=args, prompt_ids=prompt_ids, sampling_params=sampling_params, session_id=session_id
        )
    from slime.rollout.sglang_rollout import get_model_url
    from slime.utils.http_utils import post

    headers = (
        {"X-SMG-Routing-Key": session_id}
        if session_id and getattr(args, "router_policy", None) == "consistent_hashing"
        else None
    )
    model_name = str(getattr(args, f"{task}_sglang_model_name", "default"))
    return await post(
        get_model_url(args, model_name, "/generate"),
        {"input_ids": prompt_ids, "sampling_params": sampling_params, "return_logprob": True},
        headers=headers,
    )


def _response_tokens(output: dict[str, Any]) -> tuple[list[int], list[float]]:
    values = (output.get("meta_info") or {}).get("output_token_logprobs") or []
    return [int(item[1]) for item in values], [float(item[0]) for item in values]


def _finish_type(output: dict[str, Any]) -> str:
    finish = (output.get("meta_info") or {}).get("finish_reason") or {}
    return str(finish.get("type", "")) if isinstance(finish, dict) else str(finish)


def _tokenizer(args: Any):
    tokenizer = getattr(args, f"{_task(args)}_tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    from slime.rollout.sglang_rollout import GenerateState

    return GenerateState(args).tokenizer


def _task(args: Any) -> str:
    return str(getattr(args, "agent_frozen_task", "") or "").strip()


def _project_response(text: str) -> tuple[str, bool, str | None]:
    from slime_plugins.agent_tasks.textcraft.projection import project_response

    projection = project_response(text)
    return str(projection.projected_action), projection.format_valid, projection.invalid_reason


def _rollout_id(sample: Sample) -> int:
    value = sample.rollout_id if sample.rollout_id is not None else sample.index
    if value is None:
        raise ValueError("common frozen sample requires rollout_id")
    return int(value)


def _sample_index(sample: Sample, turn_idx: int) -> int:
    return _rollout_id(sample) * 1_000_000 + int(turn_idx)


def _reject_privileged_student_prefix(trajectory: dict[str, Any]) -> None:
    markers = (
        "a successful trajectory for the current task:",
        "a failed trajectory for the current task:",
        "reference trajectory from",
        "one-step hindsight:",
    )
    for turn in trajectory.get("turns", []):
        prompt = "\n".join(str(message.get("content", "")) for message in turn.get("messages", [])).lower()
        if any(marker in prompt for marker in markers):
            raise RuntimeError("common frozen student prefix contains privileged teacher text")


def _action_token_metadata(
    tokenizer, *, response_text: str, response_ids: list[int], format_valid: bool
) -> dict[str, Any]:
    empty = [0] * len(response_ids)
    if not format_valid:
        return {
            "sgs_action_token_mask": empty,
            "sgs_action_alignment_valid": False,
            "sgs_action_alignment_reason": "response_format_invalid",
        }
    lowered = response_text.lower()
    start = lowered.find("<action>")
    end = lowered.find("</action>", start + 8)
    if start < 0 or end < 0:
        return {
            "sgs_action_token_mask": empty,
            "sgs_action_alignment_valid": False,
            "sgs_action_alignment_reason": "action_content_missing",
        }
    start += 8
    try:
        encoded = tokenizer(response_text, add_special_tokens=False, return_offsets_mapping=True)
        if [int(value) for value in encoded["input_ids"]] != response_ids:
            raise ValueError
        offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    except (KeyError, TypeError, ValueError, AttributeError):
        return {
            "sgs_action_token_mask": empty,
            "sgs_action_alignment_valid": False,
            "sgs_action_alignment_reason": "response_token_alignment_failed",
        }
    mask = [int(left >= start and right <= end and right > left) for left, right in offsets]
    return {
        "sgs_action_token_mask": mask,
        "sgs_action_alignment_valid": bool(any(mask)),
        "sgs_action_alignment_reason": None if any(mask) else "no_contained_action_content_tokens",
    }
