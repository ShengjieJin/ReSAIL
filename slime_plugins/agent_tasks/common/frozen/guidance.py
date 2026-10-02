from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from slime_plugins.agent_tasks.common.algorithms.sdpo_context import (
    _build_self_trajectory_summary_context,
    finalize_sdpo_context_plan,
)

SUMMARY_SCHEMA_VERSION = 1


def summary_prompt_args(
    *, metadata_profile: str = "textcraft", summary_schema: str = "own_outcome_detailed"
) -> SimpleNamespace:
    return SimpleNamespace(
        sdpo_context_prompt_style="controlled",
        sdpo_teacher_context_mode="own_outcome",
        sdpo_solution_context_format="guidance_plan",
        sdpo_guidance_summary_source="self_trajectory",
        sdpo_guidance_summary_schema=str(summary_schema),
        sdpo_guidance_generation_mode="model_summary",
        sdpo_no_success_context_mode="failed_negative",
        sdpo_own_outcome_label_mode="explicit",
        sdpo_success_reward_threshold=1.0,
        sdpo_multi_turn_weighting="inverse_length",
        agent_task_sdpo_metadata_profile=str(metadata_profile),
    )


def trajectory_rows(trajectory: dict[str, Any], *, metadata_profile: str = "textcraft") -> list[dict[str, Any]]:
    uid = str(trajectory["trajectory_uid"])
    rows = []
    for turn in trajectory["turns"]:
        messages = [dict(message) for message in turn["messages"]]
        row = {
            "uid": uid,
            "traj_uid": uid,
            "turn_idx": int(turn["turn_idx"]),
            "task_text": str(trajectory.get("task_description") or trajectory["task_id"]),
            "anchor_obs": str(turn.get("current_observation", "")),
            "next_anchor_obs": str(turn.get("next_observation", "")),
            "projected_action": str(turn.get("frozen_action", "")),
            "is_action_valid": not bool(turn.get("errors")),
            "is_terminal": int(turn["turn_idx"]) == len(trajectory["turns"]) - 1,
            "episode_rewards": float(trajectory["success"]),
            "episode_lengths": len(trajectory["turns"]),
            "sdpo_current_prompt_text": str(messages[-1]["content"]) if messages else "",
            "sdpo_current_raw_prompt": messages,
            "algorithm_active_mask": True,
            "sdpo_metadata_profile": str(metadata_profile),
        }
        rows.append(row)
    return rows


def build_summary_request(
    trajectory: dict[str, Any],
    *,
    metadata_profile: str = "textcraft",
    summary_schema: str = "own_outcome_detailed",
) -> dict[str, Any]:
    args = summary_prompt_args(metadata_profile=metadata_profile, summary_schema=summary_schema)
    rows = trajectory_rows(trajectory, metadata_profile=metadata_profile)
    current = rows[0]
    plan = _build_self_trajectory_summary_context(args, current, {(current["uid"], current["traj_uid"]): rows})
    success = bool(trajectory["success"])
    prompt_key = "sdpo_success_guidance_prompt" if success else "sdpo_failure_guidance_prompt"
    prompt = str(plan[prompt_key])
    request = {
        "trajectory_uid": str(trajectory["trajectory_uid"]),
        "task_id": str(trajectory["task_id"]),
        "success": success,
        "outcome": str(trajectory["outcome"]),
        "kind": "success" if success else "failure",
        "prompt": prompt,
    }
    return request


def build_summary_teacher_context(
    trajectory: dict[str, Any],
    turn: dict[str, Any],
    record: dict[str, Any],
    *,
    metadata_profile: str = "textcraft",
    summary_schema: str = "own_outcome_detailed",
) -> dict[str, Any]:
    request = build_summary_request(
        trajectory,
        metadata_profile=metadata_profile,
        summary_schema=summary_schema,
    )
    validate_summary_record(record, request=request)
    args = summary_prompt_args(metadata_profile=metadata_profile, summary_schema=summary_schema)
    rows = trajectory_rows(trajectory, metadata_profile=metadata_profile)
    current = next(row for row in rows if int(row["turn_idx"]) == int(turn["turn_idx"]))
    plan = _build_self_trajectory_summary_context(args, current, {(current["uid"], current["traj_uid"]): rows})
    finalized = finalize_sdpo_context_plan([plan], {str(record["prompt"]): str(record["summary"])})[0]
    return {
        "teacher_messages": finalized["sdpo_teacher_messages"],
        "teacher_prompt_text": finalized["sdpo_teacher_prompt_text"],
        "teacher_signal_type": finalized["sdpo_teacher_signal_type"],
        "teacher_reference_scope": finalized["sdpo_context_reference_scope"],
        "outcome": str(trajectory["outcome"]),
        "success": bool(trajectory["success"]),
        "trajectory_uid": str(trajectory["trajectory_uid"]),
        "turn_idx": int(turn["turn_idx"]),
        "prompt_builder": "sdpo_context.own_outcome_guidance_plan",
        "context_mode": "own_outcome",
        "context_prompt_style": "controlled",
    }


def summary_has_schema(text: str, kind: str, *, summary_schema: str = "own_outcome_detailed") -> bool:
    value = str(text or "")
    required = (
        ("Guidance summary:", "- Minimal plan:", "- Critical actions:", "- Checks:", "- Avoid:")
        if kind == "success"
        else ("Failure analysis:", "- Failure diagnosis:", "- Useful evidence:", "- Corrected plan:", "- Avoid:")
    )
    forbidden_header = "Failure analysis:" if kind == "success" else "Guidance summary:"
    forbidden_transcript = ("Successful trajectory evidence:", "Unsuccessful trajectory evidence:")
    if forbidden_header in value or any(marker in value for marker in forbidden_transcript):
        return False
    lines = [line.strip() for line in value.strip().splitlines() if line.strip()]
    if len(lines) != len(required) or lines[0] != required[0]:
        return False
    return all(
        re.fullmatch(rf"{re.escape(marker)}[ \t]+\S.*", line) is not None
        for line, marker in zip(lines[1:], required[1:], strict=True)
    )




def validate_summary_record(record: dict[str, Any], *, request: dict[str, Any] | None = None) -> None:
    if int(record.get("schema_version", -1)) != SUMMARY_SCHEMA_VERSION:
        raise ValueError("guidance summary record schema mismatch")
    if request is not None and any(record.get(key) != request[key] for key in request):
        raise ValueError("guidance summary record differs from trajectory-derived request")
    summary_schema = str(record.get("summary_schema") or "own_outcome_detailed")
    if summary_schema != "own_outcome_detailed":
        raise ValueError("guidance summary record has an unsupported summary schema")
    if not summary_has_schema(
        str(record.get("summary", "")),
        str(record.get("kind", "")),
        summary_schema=summary_schema,
    ):
        raise ValueError("guidance summary record fails the required outcome-specific schema")
    if int(record.get("attempt_count", 0)) not in {1, 2, 3}:
        raise ValueError("guidance summary attempt_count must be in 1..3")


def load_summary_records(args: Any) -> dict[str, dict[str, Any]]:
    root = Path(str(getattr(args, "agent_frozen_guidance_summary_dir", "") or ""))
    if not root.is_dir():
        raise FileNotFoundError(f"guidance summary directory does not exist: {root}")
    cache = getattr(args, "_agent_frozen_guidance_summary_cache", None)
    if isinstance(cache, tuple) and cache[0] == str(root.resolve()):
        return cache[1]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line) for line in (root / "summaries.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    expected = int(
        getattr(args, "agent_frozen_expected_guidance_trajectories", 0)
        or getattr(args, "agent_frozen_expected_trajectories", 0)
        or 0
    )
    empty_uids_raw = manifest.get("empty_guideline_uids", [])
    empty_uids = {str(uid) for uid in empty_uids_raw} if isinstance(empty_uids_raw, list) else set()
    empty_policy = str(getattr(args, "agent_frozen_empty_guideline_policy", "") or "")
    corpus_dir = str(
        getattr(args, "agent_frozen_corpus_dir", "") or getattr(args, "alfworld_frozen_corpus_dir", "") or ""
    )
    model_path = str(getattr(args, "agent_frozen_guidance_summary_model_path", "") or "")
    if (
        manifest.get("schema_version") != SUMMARY_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or int(manifest.get("trajectory_count", -1)) != expected
        or manifest.get("corpus_dir") != corpus_dir
        or (model_path and manifest.get("model_path") != model_path)
        or verification.get("status") != "verified"
        or verification.get("manifest") != manifest
        or not isinstance(empty_uids_raw, list)
        or len(empty_uids) != len(empty_uids_raw)
        or (empty_uids and empty_policy != "skip_trajectory")
        or int(manifest.get("valid_guideline_count", len(records))) != len(records)
        or int(manifest.get("empty_guideline_count", len(empty_uids))) != len(empty_uids)
        or len(records) + len(empty_uids) != expected
    ):
        raise ValueError("guidance summary artifact binding is invalid")
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        validate_summary_record(record)
        uid = str(record["trajectory_uid"])
        if uid in index:
            raise ValueError(f"duplicate guidance summary trajectory_uid: {uid}")
        index[uid] = record
    if set(index) & empty_uids:
        raise ValueError("empty-guideline identities overlap valid guidance records")
    setattr(args, "_agent_frozen_guidance_summary_cache", (str(root.resolve()), index))
    setattr(args, "_agent_frozen_empty_guideline_uids", empty_uids)
    return index


def empty_guideline_uids(args: Any) -> set[str]:
    load_summary_records(args)
    return set(getattr(args, "_agent_frozen_empty_guideline_uids", set()))


def guidance_metadata(args: Any, trajectory: dict[str, Any]) -> dict[str, Any]:
    uid = str(trajectory["trajectory_uid"])
    record = load_summary_records(args)[uid]
    validated = getattr(args, "_agent_frozen_guidance_summary_validated", None)
    if not isinstance(validated, set):
        validated = set()
        setattr(args, "_agent_frozen_guidance_summary_validated", validated)
    if uid not in validated:
        profile = str(getattr(args, "agent_task_sdpo_metadata_profile", "textcraft") or "textcraft")
        summary_schema = str(
            getattr(args, "agent_frozen_guidance_summary_schema", "")
            or getattr(args, "sdpo_guidance_summary_schema", "")
            or "own_outcome_detailed"
        )
        validate_summary_record(
            record,
            request=build_summary_request(
                trajectory,
                metadata_profile=profile,
                summary_schema=summary_schema,
            ),
        )
        validated.add(uid)
    return {"sdpo_guidance_summary_outputs": {str(record["prompt"]): str(record["summary"])}}
