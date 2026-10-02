from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slime_plugins.agent_tasks.common.artifacts import write_json_idempotent, write_json_once
from .capabilities import capabilities_for_args
from .contracts import student_prompt_privilege_violation


def summarize_action_match(
    samples: list[Any],
    *,
    outcome_split: bool = False,
) -> dict[str, dict[str, Any]]:
    tracked = []
    for sample in samples:
        arm = str((sample.metadata or {}).get("frozen_arm", ""))
        if arm and capabilities_for_args(None, arm).action_match_audit:
            tracked.append(sample)
    if not tracked:
        return {"rates": {}, "counts": {}}
    arms = {str((sample.metadata or {})["frozen_arm"]) for sample in tracked}
    if len(arms) != 1:
        raise ValueError(f"one rollout action-match audit cannot mix frozen contracts: {sorted(arms)}")
    arm = arms.pop()
    match_count = sum(int(bool((sample.metadata or {}).get("action_match", False))) for sample in tracked)
    response_length_finish_count = sum(
        int(str((sample.metadata or {}).get("finish_reason", "")) == "length") for sample in tracked
    )
    total = len(tracked)
    rates: dict[str, float | int] = {"action_match_rate": match_count / total}
    counts: dict[str, Any] = {
        "arm": arm,
        "action_match_count": match_count,
        "action_match_total": total,
        "response_length_finish_count": response_length_finish_count,
        "response_length_finish_total": total,
    }
    if outcome_split:
        success_rows: list[Any] = []
        failure_rows: list[Any] = []
        for sample in tracked:
            success = (sample.metadata or {}).get("success")
            if not isinstance(success, bool):
                raise ValueError("outcome-split action-match audit requires a frozen boolean outcome")
            (success_rows if success else failure_rows).append(sample)
        success_match_count = sum(
            int(bool((sample.metadata or {}).get("action_match", False))) for sample in success_rows
        )
        failure_match_count = sum(
            int(bool((sample.metadata or {}).get("action_match", False))) for sample in failure_rows
        )
        counts.update(
            {
                "success_trajectory_action_match_count": success_match_count,
                "success_trajectory_action_match_total": len(success_rows),
                "failure_trajectory_action_match_count": failure_match_count,
                "failure_trajectory_action_match_total": len(failure_rows),
            }
        )
        if success_rows:
            rates["action_match_rate_success_trajectories"] = success_match_count / len(success_rows)
        if failure_rows:
            rates["action_match_rate_failure_trajectories"] = failure_match_count / len(failure_rows)

    seed_rows = []
    response_token_count = 0
    for sample in tracked:
        metadata = sample.metadata or {}
        sampling_seed = metadata.get("sampling_seed")
        if not isinstance(sampling_seed, int) or sampling_seed <= 0:
            raise ValueError("current-actor rollout is missing a positive deterministic sampling seed")
        identity = {
            "trajectory_uid": str(metadata["source_trajectory_uid"]),
            "turn_idx": int(metadata["source_turn_idx"]),
            "branch_idx": int(metadata["branch_idx"]),
        }
        seed_rows.append({**identity, "sampling_seed": sampling_seed})
        response_length = int(getattr(sample, "response_length", 0) or 0)
        response_tokens = list(getattr(sample, "tokens", [])[-response_length:]) if response_length else []
        if len(response_tokens) != response_length:
            raise ValueError("current-actor response token payload is incomplete")
        response_token_count += response_length
    seed_rows.sort(key=lambda row: (row["trajectory_uid"], row["turn_idx"], row["branch_idx"]))
    counts.update(
        {
            "sampling_seed_namespace": capabilities_for_args(None, arm).sampling_namespace,
            "sampling_seed_count": len(seed_rows),
            "sampling_seeds": seed_rows,
            "response_count": len(tracked),
            "response_token_count": response_token_count,
        }
    )
    if capabilities_for_args(None, arm).step_equal_policy_loss:
        groups: dict[tuple[int, int], list[Any]] = {}
        for sample in tracked:
            metadata = sample.metadata or {}
            key = (int(metadata["source_draw_id"]), int(metadata["source_turn_idx"]))
            groups.setdefault(key, []).append(sample)
        for key, rows in groups.items():
            if len(rows) != 8 or {int((row.metadata or {})["branch_idx"]) for row in rows} != set(range(8)):
                raise ValueError(f"GRPO action group {key} must contain branch_idx 0..7")
        non_constant = sum(
            int(len({bool((row.metadata or {}).get("action_match", False)) for row in rows}) > 1)
            for rows in groups.values()
        )
        rates["grpo/non_constant_group_rate"] = non_constant / len(groups)
        counts.update({"non_constant_group_count": non_constant, "non_constant_group_total": len(groups)})
    return {"rates": rates, "counts": counts}


def write_action_match_audit(args: Any, rollout_id: int, counts: dict[str, Any]) -> None:
    root_value = str(getattr(args, "alfworld_action_match_audit_dir", "") or "").strip()
    if not root_value:
        return
    payload = {
        "schema_version": 2 if "success_trajectory_action_match_total" in counts else 1,
        "rollout_id": int(rollout_id),
        **counts,
    }
    write_json_once(Path(root_value) / f"rollout_{int(rollout_id):03d}.json", payload)


def write_sglang_request_audit(
    args: Any,
    rollout_id: int,
    rollout_extra_metrics: dict[str, Any] | None,
) -> None:
    root_value = str(getattr(args, "alfworld_sglang_request_audit_dir", "") or "").strip()
    if not root_value:
        return
    prefix = "rollout/sglang_request/"
    metrics = {
        str(key).removeprefix(prefix): value
        for key, value in (rollout_extra_metrics or {}).items()
        if str(key).startswith(prefix)
    }
    required = {
        "configured_concurrency",
        "submitted",
        "completed",
        "errors",
        "cancelled",
        "observed_max_inflight",
        "queue_wait_p50_seconds",
        "queue_wait_p95_seconds",
        "queue_wait_p99_seconds",
        "service_p50_seconds",
        "service_p95_seconds",
        "service_p99_seconds",
        "end_to_end_p50_seconds",
        "end_to_end_p95_seconds",
        "end_to_end_p99_seconds",
    }
    if set(metrics) != required:
        raise ValueError(f"incomplete SGLang request-level metrics: {sorted(set(metrics) ^ required)}")
    payload = {
        "schema_version": 1,
        "kind": "alfworld_sglang_request_level_schedule",
        "rollout_id": int(rollout_id),
        **metrics,
    }
    write_json_once(Path(root_value) / f"rollout_{int(rollout_id):03d}.json", payload)


def write_eval_result_audit(
    args: Any,
    rollout_id: int,
    samples_by_dataset: dict[str, list[Any]],
    metrics: dict[str, Any],
) -> None:
    root_value = str(getattr(args, "alfworld_eval_result_audit_dir", "") or "").strip()
    if not root_value:
        return
    datasets: dict[str, Any] = {}
    for dataset_name, samples in sorted(samples_by_dataset.items()):
        episode_rows = [sample for sample in samples if bool((sample.metadata or {}).get("is_terminal", False))]
        privilege_violations = [student_input_privilege_violation(sample) for sample in samples]
        datasets[dataset_name] = {
            "episode_count": len(episode_rows),
            "step_count": len(samples),
            "success_count": sum(int(bool((sample.metadata or {}).get("success", False))) for sample in episode_rows),
            "response_length_finish_count": sum(
                int(str((sample.metadata or {}).get("finish_reason", "")) == "length") for sample in samples
            ),
            "sampling_seed_count": sum(
                int((sample.metadata or {}).get("sampling_seed") is not None) for sample in samples
            ),
            "student_privileged_field_count": sum(int(value) for value in privilege_violations),
        }
    payload = {
        "schema_version": 1,
        "rollout_id": int(rollout_id),
        "eval_sampling_seed": int(getattr(args, "eval_sampling_seed", 314159)),
        "eval_identity_seed": int(
            getattr(args, "alfworld_eval_identity_seed", getattr(args, "eval_sampling_seed", 314159))
        ),
        "eval_replicate_seeds": [
            int(value)
            for value in (
                getattr(args, "alfworld_eval_replicate_seeds", None)
                or [getattr(args, "eval_sampling_seed", 314159)]
            )
        ],
        "datasets": datasets,
        "metrics": {
            str(key): float(value) for key, value in sorted(metrics.items()) if isinstance(value, (int, float))
        },
    }
    write_json_idempotent(
        Path(root_value) / f"eval_{int(rollout_id):03d}.json",
        payload,
        indent=2,
    )


def student_input_privilege_violation(sample: Any) -> bool:
    """Audit student-visible prompt fields, excluding post-action outcome metadata."""

    metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    train_metadata = sample.train_metadata if isinstance(getattr(sample, "train_metadata", None), dict) else {}
    neutral_train_metadata = {
        "uid",
        "traj_uid",
        "turn_idx",
        "agent_task",
        "agent_task_trace_dir",
        "sample_rollout_id",
        "sample_index",
        "group_index",
        "task_id",
        "eval_dataset_name",
        "split",
    }
    if set(train_metadata) - neutral_train_metadata:
        return True
    prompt_values = [
        metadata.get("prompt_text_preview"),
        metadata.get("raw_prompt_preview"),
        metadata.get("raw_prompt"),
        metadata.get("rendered_prompt"),
        metadata.get("prompt_text"),
    ]
    messages = metadata.get("messages")
    if messages is not None:
        prompt_values.append(json.dumps(messages, ensure_ascii=False, sort_keys=True))
    return student_prompt_privilege_violation(
        "\n".join(str(value) for value in prompt_values if value is not None).lower(),
        metadata_keys=metadata,
    )
