from __future__ import annotations

import copy
import hashlib
import math
import re
from collections import defaultdict
from typing import Any

BASE_SDPO_METADATA_KEYS = (
    "uid",
    "traj_uid",
    "turn_idx",
    "task_text",
    "anchor_obs",
    "next_anchor_obs",
    "projected_action",
    "is_action_valid",
    "is_terminal",
    "sdpo_current_prompt_text",
    "sdpo_current_raw_prompt",
)

SDPO_METADATA_PROFILE_KEYS = {
    "agent": BASE_SDPO_METADATA_KEYS,
    "alfworld": BASE_SDPO_METADATA_KEYS,
    "textcraft": BASE_SDPO_METADATA_KEYS,
}

_REPROMPT_TEMPLATE = "{prompt}{solution}{failure}{feedback}\n\nCorrectly solve the original question."
_ROW_PRECOMPUTED_GUIDANCE_OUTPUT_KEY = "_sdpo_row_precomputed_guidance_output"
_SOLUTION_TEMPLATE = (
    "\n\nHelpful guidance from a previous successful attempt on the same task:\n\n"
    "{successful_previous_attempt}\n\n"
    "Use this guidance only as a reference. Make the next decision based on the current observation "
    "and the admissible actions of the current state."
)
_TRAJECTORY_DEMO_SOLUTION_TEMPLATE = (
    "\n\nReference trajectory from a successful previous attempt on the same task:\n\n"
    "{successful_previous_attempt}\n\n"
    "Use this reference only as context for the same task. Make the next decision from the current observation "
    "and the current admissible actions. If the current state differs from the reference trajectory, do not "
    "blindly copy an action."
)
_TRAJECTORY_DEMO_FAILURE_TEMPLATE = (
    "\n\nReference trajectory from a failed attempt on the same task:\n\n"
    "{unsuccessful_previous_attempt}\n\n"
    "Use this reference only as negative evidence. It is an attempt that did not solve the task.\n"
    "Compare it with the current observation and admissible actions before deciding.\n"
    "Avoid repeating actions from the failed trajectory when they caused no progress, invalid transitions, "
    "loops, or terminal failure.\n"
    "Do not blindly reject every action in the failed trajectory: an early action may still be useful if it "
    "is valid and matches the current state.\n"
    "Continue solving the original task and follow the original response format."
)
_FAILURE_TEMPLATE = (
    "\n\nLessons from a previous unsuccessful attempt on the same task:\n\n"
    "{unsuccessful_previous_attempt}\n\n"
    "Use these lessons only as guidance. Make the next decision based on the current observation "
    "and the admissible actions of the current state."
)
_FEEDBACK_TEMPLATE = "\n\nRelevant environment transition:\n\n{feedback_raw}"
_CONTROLLED_CONTEXT_SUFFIX = "Use this information as a reference and continue solving the original task."
_CONTROLLED_OUTCOME_HEADINGS = {
    "success": "A successful trajectory for the current task:",
    "failure": "A failed trajectory for the current task:",
    "omitted": "A trajectory for the current task:",
}
_GUIDANCE_SUMMARY_SOURCES = {"success_priority", "self_trajectory"}
_SUCCESS_GUIDANCE_SUMMARY_INSTRUCTIONS = """You are writing a reusable guidance summary.
This summary will be inserted into a future prompt for the same interactive task.

Output ONLY this format:
<thinking>
...
</thinking>

Guidance summary:
- Minimal plan: ...
- Critical actions: ...
- Checks / avoid: ...

Rules:
- Do not copy the full task, observations, or command list.
- Keep only facts needed to choose future actions.
- Use exact action strings in backticks when they are critical.
- Keep it concise.
- Think briefly inside <thinking> before writing the guidance summary."""
_FAILURE_GUIDANCE_SUMMARY_INSTRUCTIONS = """You are writing a reusable failure analysis.
This analysis will be inserted into a future prompt for the same interactive task.

Output ONLY this format:
<thinking>
...
</thinking>

Failure analysis:
- Likely mistake: ...
- Avoid: ...

Rules:
- Do not copy the full task, observations, or command list.
- Focus on the failed action pattern and the corrected next decision.
- Use exact action strings in backticks when they are critical.
- Keep it concise.
- Think briefly inside <thinking> before writing the failure analysis."""
_OWN_OUTCOME_SUCCESS_GUIDANCE_SUMMARY_INSTRUCTIONS = """You are writing concise guidance from one completed trajectory.
This guidance will be inserted into a future prompt for the same interactive task.

Output ONLY this format:
<thinking>
...
</thinking>

Guidance summary:
- Minimal plan: ...
- Critical actions: ...
- Checks: ...
- Avoid: ...

Rules:
- Use the complete trajectory as evidence, but do not copy the full task, observations, or command list.
- Use exact action strings in backticks when they are critical.
- Keep it concise and do not invent facts absent from the evidence.
- Think briefly inside <thinking> before writing the guidance summary."""
_OWN_OUTCOME_FAILURE_GUIDANCE_SUMMARY_INSTRUCTIONS = """You are writing concise guidance from one failed trajectory.
This guidance will be inserted into a future prompt for the same interactive task.

Output ONLY this format:
<thinking>
...
</thinking>

Failure analysis:
- Failure diagnosis: ...
- Useful evidence: ...
- Corrected plan: ...
- Avoid: ...

Rules:
- Use the complete failed trajectory as evidence, but do not copy the full task, observations, or command list.
- Distinguish useful partial progress from the action pattern that caused failure.
- Use exact action strings in backticks when they are critical.
- Keep it concise and do not invent facts absent from the evidence.
- Think briefly inside <thinking> before writing the failure analysis."""
_IMAGE_PLACEHOLDER = "<image>"


def build_sdpo_train_metadata(sample: Any, *, profile: str | None = None) -> dict[str, Any]:
    return canonicalize_sdpo_metadata(_sample_sdpo_source(sample), profile=profile)


def extract_sdpo_metadata(sample: Any, *, profile: str | None = None) -> dict[str, Any]:
    return build_sdpo_train_metadata(sample, profile=profile)


def build_alfworld_sdpo_train_metadata(sample: Any) -> dict[str, Any]:
    return build_sdpo_train_metadata(sample, profile="alfworld")


def build_textcraft_sdpo_train_metadata(sample: Any) -> dict[str, Any]:
    return build_sdpo_train_metadata(sample, profile="textcraft")




def extract_alfworld_sdpo_metadata(sample: Any) -> dict[str, Any]:
    return build_alfworld_sdpo_train_metadata(sample)


def extract_textcraft_sdpo_metadata(sample: Any) -> dict[str, Any]:
    return build_textcraft_sdpo_train_metadata(sample)




def canonicalize_alfworld_sdpo_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return canonicalize_sdpo_metadata(metadata, profile="alfworld")


def canonicalize_textcraft_sdpo_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return canonicalize_sdpo_metadata(metadata, profile="textcraft")




def canonicalize_sdpo_metadata(metadata: dict[str, Any], *, profile: str | None = None) -> dict[str, Any]:
    value = copy.deepcopy(dict(metadata))
    profile = _resolve_profile(profile, value)
    for singular, plural in (("episode_reward", "episode_rewards"), ("episode_length", "episode_lengths")):
        if plural not in value:
            if singular not in value:
                raise KeyError(f"SDPO metadata missing {plural} or {singular}")
            value[plural] = value[singular]

    for key in SDPO_METADATA_PROFILE_KEYS[profile]:
        if key not in value:
            raise KeyError(f"{profile} SDPO metadata missing {key}")

    value["uid"] = str(value["uid"])
    value["traj_uid"] = str(value["traj_uid"])
    value["turn_idx"] = int(value["turn_idx"])
    value["episode_rewards"] = float(value["episode_rewards"])
    value["episode_lengths"] = int(value["episode_lengths"])
    if value["episode_lengths"] <= 0:
        raise ValueError(f"{profile} SDPO episode_lengths must be positive, got {value['episode_lengths']}")
    value["is_action_valid"] = bool(value["is_action_valid"])
    value["is_terminal"] = bool(value["is_terminal"])
    value["algorithm_active_mask"] = bool(value.get("algorithm_active_mask", True))
    value["sdpo_metadata_profile"] = profile
    return value


def prepare_sdpo_context_plan(args: Any, samples: list[Any]) -> list[dict[str, Any]]:
    profile = getattr(args, "agent_task_sdpo_metadata_profile", None)
    _validate_context_prompt_config(args)
    _guidance_summary_source(args)
    no_success_context_mode = _no_success_context_mode(args)
    if str(getattr(args, "sdpo_teacher_context_mode", "original")) == "own_outcome":
        solution_format = str(getattr(args, "sdpo_solution_context_format", "guidance_plan"))
        if solution_format not in {"trajectory_demo", "guidance_plan"}:
            raise ValueError("sdpo_teacher_context_mode=own_outcome requires trajectory_demo or guidance_plan.")
        if solution_format == "guidance_plan" and _guidance_summary_source(args) != "self_trajectory":
            raise ValueError("own_outcome guidance_plan requires sdpo_guidance_summary_source=self_trajectory.")
        if no_success_context_mode != "failed_negative":
            raise ValueError(
                "sdpo_teacher_context_mode=own_outcome requires sdpo_no_success_context_mode=failed_negative."
            )
    rows = []
    for sample in samples:
        row = build_sdpo_train_metadata(sample, profile=profile)
        if getattr(sample, "remove_sample", False):
            row["algorithm_active_mask"] = False
        rows.append(row)
    trajectories = _group_trajectories(rows)
    successful = _collect_successful_trajectories(args, trajectories)
    failed = _collect_failed_trajectories(args, trajectories)
    plan = [_build_plan_row(args, row, successful, failed, trajectories) for row in rows]
    _normalize_active_weights(plan)
    return plan


build_sdpo_context_plan = prepare_sdpo_context_plan


def prepare_grpo_token_weight_context_plan(args: Any, samples: list[Any]) -> list[dict[str, Any]]:
    """Build row-aligned trajectory contrasts without applying SDPO sample semantics."""
    rows = []
    for sample in samples:
        train_metadata = getattr(sample, "train_metadata", None) or {}
        source = train_metadata.get("grpo_token_weight_context") if isinstance(train_metadata, dict) else None
        if isinstance(source, dict):
            row = canonicalize_sdpo_metadata(
                source,
                profile=getattr(args, "agent_task_sdpo_metadata_profile", None),
            )
        else:
            row = build_sdpo_train_metadata(
                sample,
                profile=getattr(args, "agent_task_sdpo_metadata_profile", None),
            )
        if getattr(sample, "remove_sample", False):
            row["algorithm_active_mask"] = False
        rows.append(row)

    trajectories = _group_trajectories(rows)
    threshold = float(getattr(args, "grpo_token_weight_success_reward_threshold", 1.0))
    successful = _collect_trajectories_by_outcome(trajectories, threshold=threshold, success=True)
    failed = _collect_trajectories_by_outcome(trajectories, threshold=threshold, success=False)
    max_demo_steps = getattr(args, "grpo_token_weight_max_demo_steps", None)

    plan = []
    for row in rows:
        uid = row["uid"]
        positive_candidates = successful.get(uid, [])
        if not positive_candidates:
            plan.append(_empty_grpo_token_weight_context(uid))
            continue

        positive_text = _build_trajectory_demo(positive_candidates[0], max_demo_steps)
        positive_prompt, positive_messages = _build_grpo_positive_prompt(row, positive_text)
        failed_candidates = failed.get(uid, [])
        if failed_candidates:
            candidate = failed_candidates[_random_failed_candidate_index(args, row, len(failed_candidates))]
            failure_text = _build_trajectory_demo(candidate, max_demo_steps)
            contrast_prompt, contrast_messages = _build_grpo_failed_prompt(row, failure_text)
            contrast_source = "failed"
        else:
            contrast_prompt = str(row["sdpo_current_prompt_text"])
            contrast_messages = _build_teacher_messages(row, contrast_prompt)
            contrast_source = "no_extra_context"
        plan.append(
            {
                "grpo_token_weight_positive_prompt_text": positive_prompt,
                "grpo_token_weight_positive_messages": positive_messages,
                "grpo_token_weight_contrast_prompt_text": contrast_prompt,
                "grpo_token_weight_contrast_messages": contrast_messages,
                "grpo_token_weight_contrast_source": contrast_source,
                "grpo_token_weight_uid": uid,
            }
        )
    return plan


def _empty_grpo_token_weight_context(uid: str) -> dict[str, Any]:
    return {
        "grpo_token_weight_positive_prompt_text": None,
        "grpo_token_weight_positive_messages": None,
        "grpo_token_weight_contrast_prompt_text": None,
        "grpo_token_weight_contrast_messages": None,
        "grpo_token_weight_contrast_source": "",
        "grpo_token_weight_uid": uid,
    }


def _collect_trajectories_by_outcome(
    trajectories: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    threshold: float,
    success: bool,
) -> dict[str, list[list[dict[str, Any]]]]:
    by_uid: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for (uid, _traj_uid), rows in trajectories.items():
        active_rows = _active_rows(rows)
        if len(active_rows) != len(rows):
            continue
        reward = max(float(row["episode_rewards"]) for row in active_rows)
        if (reward >= threshold) == success:
            by_uid[uid].append(active_rows)
    for candidates in by_uid.values():
        candidates.sort(
            key=lambda rows: (
                (
                    -max(float(row["episode_rewards"]) for row in rows)
                    if success
                    else max(float(row["episode_rewards"]) for row in rows)
                ),
                max(int(row["episode_lengths"]) for row in rows),
                str(rows[-1]["traj_uid"]),
            )
        )
    return by_uid


def _build_grpo_positive_prompt(
    row: dict[str, Any],
    trajectory_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    solution = _TRAJECTORY_DEMO_SOLUTION_TEMPLATE.format(successful_previous_attempt=trajectory_text)
    prompt = _REPROMPT_TEMPLATE.format(
        prompt=row["sdpo_current_prompt_text"],
        solution=solution,
        failure="",
        feedback="",
    )
    return prompt, _build_teacher_messages(row, prompt)


def _build_grpo_failed_prompt(
    row: dict[str, Any],
    trajectory_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    failure = _TRAJECTORY_DEMO_FAILURE_TEMPLATE.format(unsuccessful_previous_attempt=trajectory_text)
    prompt = f"{row['sdpo_current_prompt_text']}{failure}"
    return prompt, _build_teacher_messages(row, prompt)


def finalize_sdpo_context_plan(
    plan: list[dict[str, Any]],
    guidance_summary_outputs: dict[str, str] | list[str] | None,
) -> list[dict[str, Any]]:
    finalized = [_copy_plan_row(row) for row in plan]
    prompt_to_output = _guidance_prompt_output_map(finalized, guidance_summary_outputs)

    for row in finalized:
        args = row.pop("_sdpo_args", None)
        row_precomputed_output = row.pop(_ROW_PRECOMPUTED_GUIDANCE_OUTPUT_KEY, None)
        metadata = row["sdpo_metadata"]
        solution_text = str(row.get("sdpo_solution_text") or "")
        failure_text = str(row.get("sdpo_failure_text") or "")
        success_prompt = str(row.get("sdpo_success_guidance_prompt") or "")
        failure_prompt = str(row.get("sdpo_failure_guidance_prompt") or "")
        teacher_signal = str(row.get("sdpo_teacher_signal_type") or "")
        summary_schema = str(getattr(args, "sdpo_guidance_summary_schema", "")) if args is not None else ""
        is_failed_negative_demo = teacher_signal == "failed_negative_demo"
        if row.get("sdpo_filter_reason"):
            row["sdpo_solution_text"] = ""
            row["sdpo_failure_text"] = ""
            row["sdpo_feedback_text"] = ""
            _deactivate(row)
            continue
        if success_prompt:
            solution_text, fallback_used = _guidance_output_or_fallback(
                (
                    {success_prompt: clean_guidance_summary(row_precomputed_output)}
                    if row_precomputed_output is not None
                    else prompt_to_output
                ),
                success_prompt,
                kind="success",
                summary_schema=summary_schema,
            )
            row["sdpo_success_guidance_text"] = solution_text
            row["sdpo_success_guidance_fallback_used"] = fallback_used
        if failure_prompt:
            failure_text, fallback_used = _guidance_output_or_fallback(
                (
                    {failure_prompt: clean_guidance_summary(row_precomputed_output)}
                    if row_precomputed_output is not None
                    else prompt_to_output
                ),
                failure_prompt,
                kind="failure",
                summary_schema=summary_schema,
            )
            row["sdpo_failure_guidance_text"] = failure_text
            row["sdpo_failure_guidance_fallback_used"] = fallback_used

        feedback_text = str(row.get("sdpo_feedback_text") or "")
        if args is not None and _context_prompt_style(args) == "controlled":
            if teacher_signal == "feedback":
                feedback_text = _controlled_feedback_text(metadata)
                prompt, messages = _build_controlled_feedback_teacher_prompt(metadata, feedback_text)
                row["sdpo_teacher_prompt_text"] = prompt
                row["sdpo_teacher_messages"] = messages
            elif teacher_signal in {
                "solution_demo",
                "failed_negative_demo",
                "solution_guidance_prompt",
                "failure_guidance_prompt",
            }:
                prompt, messages = _build_controlled_trajectory_teacher_prompt(
                    args,
                    metadata,
                    solution_text or failure_text,
                    is_success=teacher_signal in {"solution_demo", "solution_guidance_prompt"},
                )
                row["sdpo_teacher_prompt_text"] = prompt
                row["sdpo_teacher_messages"] = messages
        elif args is not None and not is_failed_negative_demo:
            feedback_text = _feedback_text(
                args,
                metadata,
                has_solution=bool(solution_text),
                has_failure=bool(failure_text),
            )
            prompt, messages = _build_teacher_prompt(
                args,
                metadata,
                solution_text=solution_text,
                failure_text=failure_text,
                feedback_text=feedback_text,
            )
            row["sdpo_teacher_prompt_text"] = prompt
            row["sdpo_teacher_messages"] = messages

        row["sdpo_solution_text"] = solution_text
        row["sdpo_failure_text"] = failure_text
        row["sdpo_feedback_text"] = feedback_text
        if not bool(metadata.get("algorithm_active_mask", True)):
            _deactivate(row)
        elif solution_text and success_prompt:
            row["sdpo_teacher_signal_type"] = "solution_guidance"
            row["self_distillation_mask"] = 1.0
        elif failure_text and failure_prompt:
            row["sdpo_teacher_signal_type"] = "failure_guidance"
            row["self_distillation_mask"] = 1.0
        elif failure_text and is_failed_negative_demo:
            row["sdpo_teacher_signal_type"] = "failed_negative_demo"
            row["self_distillation_mask"] = 1.0
        elif solution_text:
            row["sdpo_teacher_signal_type"] = "solution_demo"
            row["self_distillation_mask"] = 1.0
        elif feedback_text:
            row["sdpo_teacher_signal_type"] = "feedback"
            row["self_distillation_mask"] = 1.0
        else:
            _deactivate(row)
    return finalized


def compute_sdpo_context_metrics(
    plan: list[dict[str, Any]], *, success_reward_threshold: float = 1.0
) -> dict[str, float]:
    if not plan:
        return {
            "self_distillation/sample_count": 0.0,
            "self_distillation/mask_fraction": 0.0,
            "self_distillation/success_sample_fraction": 0.0,
            "self_distillation/solution_used_fraction": 0.0,
            "self_distillation/guidance_summary_used_fraction": 0.0,
            "self_distillation/failure_summary_used_fraction": 0.0,
            "self_distillation/guidance_summary_fallback_fraction": 0.0,
            "self_distillation/trajectory_demo_used_fraction": 0.0,
            "self_distillation/failed_negative_used_fraction": 0.0,
            "self_distillation/own_failed_negative_used_fraction": 0.0,
            "self_distillation/own_success_demo_used_fraction": 0.0,
            "self_distillation/group_success_demo_used_fraction": 0.0,
            "self_distillation/feedback_used_fraction": 0.0,
            "self_distillation/no_success_filtered_fraction": 0.0,
            "self_distillation/all_success_filtered_fraction": 0.0,
            "self_distillation/active_sample_fraction": 0.0,
            "self_distillation/teacher_prompt_avg_char_len": 0.0,
            "self_distillation/teacher_prompt_active_avg_char_len": 0.0,
            "self_distillation/teacher_prompt_max_char_len": 0.0,
            "self_distillation/representation_success_count_mean": 0.0,
            "self_distillation/representation_success_count_max": 0.0,
            "self_distillation/representation_mean_success_fraction": 0.0,
            "self_distillation/representation_multi_success_fraction": 0.0,
            "self_distillation/token_weight_contrast_failed_fraction": 0.0,
            "self_distillation/token_weight_contrast_no_context_fraction": 0.0,
            "self_distillation/trajectory_balance_raw_weight_mean": 0.0,
            "self_distillation/trajectory_balance_raw_weight_min": 0.0,
            "self_distillation/trajectory_balance_raw_weight_max": 0.0,
            "self_distillation/trajectory_balance_normalized_weight_min": 0.0,
            "self_distillation/trajectory_balance_normalized_weight_mean": 0.0,
            "self_distillation/trajectory_balance_normalized_weight_max": 0.0,
            "self_distillation/trajectory_balance_success_weight_mass": 0.0,
            "self_distillation/trajectory_balance_failure_weight_mass": 0.0,
            "self_distillation/trajectory_balance_effective_sample_fraction": 0.0,
        }

    active = 0
    solution = 0
    success = 0
    guidance_summary = 0
    failure_summary = 0
    guidance_fallback = 0
    trajectory_demo = 0
    failed_negative = 0
    own_failed_negative = 0
    own_success_demo = 0
    group_success_demo = 0
    feedback = 0
    no_success_filtered = 0
    all_success_filtered = 0
    representation_success_counts: list[int] = []
    representation_mean_success = 0
    representation_multi_success = 0
    token_weight_contrast_failed = 0
    token_weight_contrast_no_context = 0
    prompt_lengths: list[int] = []
    active_prompt_lengths: list[int] = []
    active_raw_weights: list[float] = []
    active_normalized_weights: list[float] = []
    success_weight = 0.0
    for row in plan:
        signal = str(row.get("sdpo_teacher_signal_type") or "")
        reference_scope = str(row.get("sdpo_context_reference_scope") or "")
        token_weight_contrast_source = str(row.get("sdpo_token_weight_contrast_source") or "")
        is_active = float(row.get("self_distillation_mask", 0.0)) > 0.0 and signal != "none"
        active += int(is_active)
        solution += int(is_active and signal in {"solution_demo", "solution_guidance"})
        success += int(float((row.get("sdpo_metadata") or {}).get("episode_rewards", 0.0)) >= success_reward_threshold)
        guidance_summary += int(
            is_active and (signal == "solution_guidance" or bool(row.get("sdpo_success_guidance_text")))
        )
        failure_summary += int(
            is_active and (signal == "failure_guidance" or bool(row.get("sdpo_failure_guidance_text")))
        )
        guidance_fallback += int(
            is_active
            and (
                bool(row.get("sdpo_success_guidance_fallback_used"))
                or bool(row.get("sdpo_failure_guidance_fallback_used"))
            )
        )
        trajectory_demo += int(is_active and signal == "solution_demo")
        failed_negative += int(is_active and signal == "failed_negative_demo")
        own_failed_negative += int(is_active and reference_scope == "own_failed_negative")
        own_success_demo += int(is_active and reference_scope == "own_success_demo")
        group_success_demo += int(is_active and reference_scope == "group_success_demo")
        feedback += int(is_active and signal == "feedback")
        no_success_filtered += int(str(row.get("sdpo_filter_reason") or "") == "no_success_trajectory")
        all_success_filtered += int(str(row.get("sdpo_filter_reason") or "") == "all_success_trajectory")
        success_count = int(row.get("sdpo_representation_success_count") or 0)
        if is_active and signal == "solution_demo" and success_count > 0:
            representation_success_counts.append(success_count)
            representation_mean_success += int(row.get("sdpo_representation_success_aggregation") == "mean")
            representation_multi_success += int(success_count > 1)
        token_weight_contrast_failed += int(is_active and token_weight_contrast_source == "failed")
        token_weight_contrast_no_context += int(is_active and token_weight_contrast_source == "no_extra_context")
        prompt_len = len(str(row.get("sdpo_teacher_prompt_text") or ""))
        prompt_lengths.append(prompt_len)
        if is_active:
            active_prompt_lengths.append(prompt_len)
            raw_weight = float(row.get("sdpo_raw_loss_weight", 0.0))
            normalized_weight = float(row.get("sdpo_loss_weights", 0.0))
            active_raw_weights.append(raw_weight)
            active_normalized_weights.append(normalized_weight)
            if float((row.get("sdpo_metadata") or {}).get("episode_rewards", 0.0)) >= success_reward_threshold:
                success_weight += normalized_weight

    count = float(len(plan))
    representation_count = float(len(representation_success_counts))
    normalized_weight_sum = sum(active_normalized_weights)
    effective_sample_fraction = (
        normalized_weight_sum**2
        / (len(active_normalized_weights) * sum(weight**2 for weight in active_normalized_weights))
        if active_normalized_weights and any(active_normalized_weights)
        else 0.0
    )
    return {
        "self_distillation/sample_count": count,
        "self_distillation/mask_fraction": active / count,
        "self_distillation/success_sample_fraction": success / count,
        "self_distillation/solution_used_fraction": solution / count,
        "self_distillation/guidance_summary_used_fraction": guidance_summary / count,
        "self_distillation/failure_summary_used_fraction": failure_summary / count,
        "self_distillation/guidance_summary_fallback_fraction": guidance_fallback / count,
        "self_distillation/trajectory_demo_used_fraction": trajectory_demo / count,
        "self_distillation/failed_negative_used_fraction": failed_negative / count,
        "self_distillation/own_failed_negative_used_fraction": own_failed_negative / count,
        "self_distillation/own_success_demo_used_fraction": own_success_demo / count,
        "self_distillation/group_success_demo_used_fraction": group_success_demo / count,
        "self_distillation/feedback_used_fraction": feedback / count,
        "self_distillation/no_success_filtered_fraction": no_success_filtered / count,
        "self_distillation/all_success_filtered_fraction": all_success_filtered / count,
        "self_distillation/active_sample_fraction": active / count,
        "self_distillation/teacher_prompt_avg_char_len": sum(prompt_lengths) / count,
        "self_distillation/teacher_prompt_active_avg_char_len": (
            sum(active_prompt_lengths) / len(active_prompt_lengths) if active_prompt_lengths else 0.0
        ),
        "self_distillation/teacher_prompt_max_char_len": float(max(prompt_lengths)),
        "self_distillation/representation_success_count_mean": (
            sum(representation_success_counts) / representation_count if representation_success_counts else 0.0
        ),
        "self_distillation/representation_success_count_max": float(
            max(representation_success_counts) if representation_success_counts else 0
        ),
        "self_distillation/representation_mean_success_fraction": (
            representation_mean_success / representation_count if representation_success_counts else 0.0
        ),
        "self_distillation/representation_multi_success_fraction": (
            representation_multi_success / representation_count if representation_success_counts else 0.0
        ),
        "self_distillation/token_weight_contrast_failed_fraction": token_weight_contrast_failed / count,
        "self_distillation/token_weight_contrast_no_context_fraction": token_weight_contrast_no_context / count,
        "self_distillation/trajectory_balance_raw_weight_mean": (
            sum(active_raw_weights) / len(active_raw_weights) if active_raw_weights else 0.0
        ),
        "self_distillation/trajectory_balance_raw_weight_min": min(active_raw_weights) if active_raw_weights else 0.0,
        "self_distillation/trajectory_balance_raw_weight_max": max(active_raw_weights) if active_raw_weights else 0.0,
        "self_distillation/trajectory_balance_normalized_weight_min": (
            min(active_normalized_weights) if active_normalized_weights else 0.0
        ),
        "self_distillation/trajectory_balance_normalized_weight_mean": (
            normalized_weight_sum / len(active_normalized_weights) if active_normalized_weights else 0.0
        ),
        "self_distillation/trajectory_balance_normalized_weight_max": (
            max(active_normalized_weights) if active_normalized_weights else 0.0
        ),
        "self_distillation/trajectory_balance_success_weight_mass": (
            success_weight / normalized_weight_sum if normalized_weight_sum else 0.0
        ),
        "self_distillation/trajectory_balance_failure_weight_mass": (
            (normalized_weight_sum - success_weight) / normalized_weight_sum if normalized_weight_sum else 0.0
        ),
        "self_distillation/trajectory_balance_effective_sample_fraction": effective_sample_fraction,
    }


def collect_guidance_summary_prompts(plan: list[dict[str, Any]]) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for row in plan:
        for key in ("sdpo_success_guidance_prompt", "sdpo_failure_guidance_prompt"):
            prompt = str(row.get(key) or "").strip()
            if prompt and prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
    return prompts


def collect_guidance_summary_prompt_payloads(plan: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for row in plan:
        for prompt_key, payload_key in (
            ("sdpo_success_guidance_prompt", "sdpo_success_guidance_payload"),
            ("sdpo_failure_guidance_prompt", "sdpo_failure_guidance_payload"),
        ):
            prompt = str(row.get(prompt_key) or "").strip()
            if not prompt or prompt in payloads:
                continue
            payload = row.get(payload_key)
            if isinstance(payload, dict):
                payloads[prompt] = _copy_guidance_payload(payload)
    return payloads


def clean_guidance_summary(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"<thinking\b[^>]*>.*?</thinking>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    cleaned = re.sub(r"<thinking\b[^>]*>.*\Z", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    section = re.search(r"(?im)^\s*#{0,3}\s*(Guidance summary:|Failure analysis:)", cleaned)
    if section:
        cleaned = cleaned[section.start() :].strip()
    return re.sub(r"^\s*#{0,3}\s*Goal:\s*.*?(?=\n\S|\Z)", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()


def _sample_sdpo_source(sample: Any) -> dict[str, Any]:
    train_metadata = getattr(sample, "train_metadata", None) or {}
    if isinstance(train_metadata, dict) and isinstance(train_metadata.get("sdpo"), dict):
        return train_metadata["sdpo"]
    metadata = getattr(sample, "metadata", None) or {}
    source = dict(metadata)
    if "sdpo_current_prompt_text" not in source:
        source["sdpo_current_prompt_text"] = source.get("raw_prompt", getattr(sample, "prompt", ""))
    if "sdpo_current_raw_prompt" not in source:
        source["sdpo_current_raw_prompt"] = source.get("messages", source["sdpo_current_prompt_text"])
    return source


def _resolve_profile(profile: str | None, metadata: dict[str, Any]) -> str:
    profile = profile or metadata.get("sdpo_metadata_profile") or "agent"
    profile = str(profile).strip().lower()
    if profile not in SDPO_METADATA_PROFILE_KEYS:
        expected = ", ".join(sorted(SDPO_METADATA_PROFILE_KEYS))
        raise ValueError(f"Unsupported agent-task SDPO metadata profile {profile!r}; expected {expected}")
    return profile


def _group_trajectories(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    trajectories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        trajectories[(row["uid"], row["traj_uid"])].append(row)
    for trajectory_rows in trajectories.values():
        trajectory_rows.sort(key=lambda item: item["turn_idx"])
    return trajectories


def _collect_successful_trajectories(
    args: Any,
    trajectories: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, list[list[dict[str, Any]]]]:
    threshold = float(getattr(args, "sdpo_success_reward_threshold", 1.0))
    by_uid: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for (uid, _traj_uid), rows in trajectories.items():
        active_rows = _active_rows(rows)
        if len(active_rows) != len(rows):
            continue
        if max(float(row["episode_rewards"]) for row in active_rows) >= threshold:
            by_uid[uid].append(active_rows)
    for candidates in by_uid.values():
        candidates.sort(
            key=lambda rows: (
                -max(float(row["episode_rewards"]) for row in rows),
                max(int(row["episode_lengths"]) for row in rows),
                str(rows[-1]["traj_uid"]),
            )
        )
    return by_uid


def _collect_failed_trajectories(
    args: Any,
    trajectories: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, list[list[dict[str, Any]]]]:
    threshold = float(getattr(args, "sdpo_success_reward_threshold", 1.0))
    by_uid: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for (uid, _traj_uid), rows in trajectories.items():
        active_rows = _active_rows(rows)
        if len(active_rows) != len(rows):
            continue
        if max(float(row["episode_rewards"]) for row in active_rows) < threshold:
            by_uid[uid].append(active_rows)
    for candidates in by_uid.values():
        candidates.sort(
            key=lambda rows: (
                max(float(row["episode_rewards"]) for row in rows),
                max(int(row["episode_lengths"]) for row in rows),
                str(rows[-1]["traj_uid"]),
            )
        )
    return by_uid


def _build_plan_row(
    args: Any,
    row: dict[str, Any],
    successful: dict[str, list[list[dict[str, Any]]]],
    failed: dict[str, list[list[dict[str, Any]]]],
    trajectories: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    if not bool(row.get("algorithm_active_mask", True)):
        prompt, messages = _build_teacher_prompt(args, row)
        return _plan_row(args, row, prompt, messages, signal="none", active=False)

    uid_has_success = bool(successful.get(row["uid"]))
    failed_candidates = failed.get(row["uid"], [])
    teacher_context_mode = str(getattr(args, "sdpo_teacher_context_mode", "original"))
    no_success_context_mode = _no_success_context_mode(args)
    if not uid_has_success and no_success_context_mode == "filter":
        prompt, messages = _build_teacher_prompt(args, row)
        return _plan_row(
            args,
            row,
            prompt,
            messages,
            signal="none",
            active=False,
            filter_reason="no_success_trajectory",
        )
    if uid_has_success and not failed_candidates and bool(getattr(args, "sdpo_filter_all_success_groups", False)):
        prompt, messages = _build_teacher_prompt(args, row)
        return _plan_row(
            args,
            row,
            prompt,
            messages,
            signal="none",
            active=False,
            filter_reason="all_success_trajectory",
        )

    if teacher_context_mode == "feedback_only":
        if _context_prompt_style(args) == "controlled":
            feedback_text = _controlled_feedback_text(row)
            prompt, messages = _build_controlled_feedback_teacher_prompt(row, feedback_text)
            return _plan_row(
                args,
                row,
                prompt,
                messages,
                signal="feedback",
                active=True,
                feedback_text=feedback_text,
            )
        feedback_text = _feedback_text(args, row, has_solution=False, has_failure=False)
        prompt, messages = _build_teacher_prompt(args, row, feedback_text=feedback_text)
        return _plan_row(
            args,
            row,
            prompt,
            messages,
            signal="feedback" if feedback_text else "none",
            active=bool(feedback_text),
            feedback_text=feedback_text,
        )

    if teacher_context_mode == "own_outcome":
        if str(getattr(args, "sdpo_solution_context_format", "guidance_plan")) == "guidance_plan":
            return _build_self_trajectory_summary_context(args, row, trajectories)
        return _build_own_outcome_trajectory_context(args, row, trajectories)

    if (
        str(getattr(args, "sdpo_solution_context_format", "guidance_plan")) == "guidance_plan"
        and _guidance_summary_source(args) == "self_trajectory"
    ):
        return _build_self_trajectory_summary_context(args, row, trajectories)

    candidates = _success_candidates(args, row, successful.get(row["uid"], []))
    if not candidates:
        return _build_no_success_teacher_context(
            args,
            row,
            trajectories,
            allow_failure_guidance=not uid_has_success,
            allow_failed_negative=not uid_has_success,
        )

    candidate = candidates[0]
    is_self = candidate[-1]["traj_uid"] == row["traj_uid"]
    aggregation = _representation_success_aggregation(args)
    if str(getattr(args, "sdpo_solution_context_format", "guidance_plan")) == "guidance_plan":
        solution_text = _solution_text(args, candidate)
        prompt, messages = _build_teacher_prompt(args, row)
        return _plan_row(
            args,
            row,
            prompt,
            messages,
            signal="solution_guidance_prompt",
            active=False,
            success_guidance_prompt=solution_text,
            success_guidance_payload=_guidance_payload_for_records(solution_text, candidate),
            selected_success_traj_uid=candidate[-1]["traj_uid"],
            success_is_self=is_self,
            representation_success_aggregation=aggregation,
            representation_success_count=1,
            **_token_weight_contrast_context(args, row, failed_candidates),
        )
    if aggregation == "mean":
        teacher_prompt_texts: list[str] = []
        teacher_messages_list: list[list[dict[str, Any]]] = []
        success_traj_uids: list[str] = []
        solution_text = ""
        feedback_text = _feedback_text(args, row, has_solution=True, has_failure=False)
        for success_candidate in candidates:
            candidate_solution_text = _solution_text(args, success_candidate)
            if not solution_text:
                solution_text = candidate_solution_text
            candidate_prompt, candidate_messages = _build_teacher_prompt(
                args,
                row,
                solution_text=candidate_solution_text,
                feedback_text=feedback_text,
            )
            teacher_prompt_texts.append(candidate_prompt)
            teacher_messages_list.append(candidate_messages)
            success_traj_uids.append(str(success_candidate[-1]["traj_uid"]))
        return _plan_row(
            args,
            row,
            teacher_prompt_texts[0],
            teacher_messages_list[0],
            signal="solution_demo",
            active=True,
            solution_text=solution_text,
            feedback_text=feedback_text,
            selected_success_traj_uid=",".join(success_traj_uids),
            success_is_self=is_self,
            reference_scope="group_success_demo",
            representation_success_aggregation="mean",
            representation_success_count=len(teacher_prompt_texts),
            teacher_prompt_texts=teacher_prompt_texts,
            teacher_messages_list=teacher_messages_list,
            **_token_weight_contrast_context(args, row, failed_candidates),
        )
    solution_text = _solution_text(args, candidate)
    feedback_text = _feedback_text(args, row, has_solution=True, has_failure=False)
    prompt, messages = _build_teacher_prompt(args, row, solution_text=solution_text, feedback_text=feedback_text)
    return _plan_row(
        args,
        row,
        prompt,
        messages,
        signal="solution_demo",
        active=True,
        solution_text=solution_text,
        feedback_text=feedback_text,
        selected_success_traj_uid=candidate[-1]["traj_uid"],
        success_is_self=is_self,
        reference_scope="group_success_demo",
        representation_success_aggregation=aggregation,
        representation_success_count=1,
        **_token_weight_contrast_context(args, row, failed_candidates),
    )


def _build_self_trajectory_summary_context(
    args: Any,
    row: dict[str, Any],
    trajectories: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    current_traj = _active_rows(trajectories.get((row["uid"], row["traj_uid"]), [row])) or [row]
    privileged_trajectory = _validated_privileged_trajectory(row)
    reference_trajectory = (
        privileged_trajectory
        if privileged_trajectory is not None and _task_text(privileged_trajectory)
        else current_traj
    )
    reward = max(float(record["episode_rewards"]) for record in current_traj)
    prompt, messages = _build_teacher_prompt(args, row)
    if reward >= float(getattr(args, "sdpo_success_reward_threshold", 1.0)):
        success_prompt_text = _solution_text(args, reference_trajectory)
        return _plan_row(
            args,
            row,
            prompt,
            messages,
            signal="solution_guidance_prompt",
            active=False,
            success_guidance_prompt=success_prompt_text,
            success_guidance_payload=_guidance_payload_for_records(success_prompt_text, reference_trajectory),
            selected_success_traj_uid=row["traj_uid"],
            success_is_self=True,
        )
    failure_prompt_text = _failure_text(args, reference_trajectory)
    return _plan_row(
        args,
        row,
        prompt,
        messages,
        signal="failure_guidance_prompt",
        active=False,
        failure_guidance_prompt=failure_prompt_text,
        failure_guidance_payload=_guidance_payload_for_records(failure_prompt_text, reference_trajectory),
    )


def _build_own_outcome_trajectory_context(
    args: Any,
    row: dict[str, Any],
    trajectories: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    grouped_rows = _active_rows(trajectories.get((row["uid"], row["traj_uid"]), [row])) or [row]
    privileged_trajectory = _validated_privileged_trajectory(row)
    reference_trajectory = privileged_trajectory or grouped_rows
    reward = float(row["episode_rewards"])
    is_success = reward >= float(getattr(args, "sdpo_success_reward_threshold", 1.0))
    if _context_prompt_style(args) == "controlled":
        trajectory_text = _build_trajectory_demo(reference_trajectory, max_demo_steps=None)
        prompt, messages = _build_controlled_trajectory_teacher_prompt(
            args,
            row,
            trajectory_text,
            is_success=is_success,
        )
        if is_success:
            return _plan_row(
                args,
                row,
                prompt,
                messages,
                signal="solution_demo",
                active=True,
                solution_text=trajectory_text,
                selected_success_traj_uid=row["traj_uid"],
                success_is_self=True,
                reference_scope="own_success_demo",
            )
        return _plan_row(
            args,
            row,
            prompt,
            messages,
            signal="failed_negative_demo",
            active=True,
            failure_text=trajectory_text,
            reference_scope="own_failed_negative",
        )
    if is_success:
        solution_text = _solution_text(args, reference_trajectory)
        prompt, messages = _build_teacher_prompt(args, row, solution_text=solution_text)
        return _plan_row(
            args,
            row,
            prompt,
            messages,
            signal="solution_demo",
            active=True,
            solution_text=solution_text,
            selected_success_traj_uid=row["traj_uid"],
            success_is_self=True,
            reference_scope="own_success_demo",
        )
    return _build_failed_negative_trajectory_context(args, row, reference_trajectory)


def _validated_privileged_trajectory(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    raw = row.get("sdpo_privileged_trajectory")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("sdpo_privileged_trajectory must be a non-empty list")
    expected_length = int(row.get("frozen_trajectory_length", len(raw)))
    if len(raw) != expected_length:
        raise ValueError(
            "sdpo_privileged_trajectory length does not match frozen_trajectory_length: "
            f"{len(raw)} != {expected_length}"
        )
    trajectory: list[dict[str, Any]] = []
    for expected_turn_idx, record in enumerate(raw):
        if not isinstance(record, dict):
            raise ValueError("sdpo_privileged_trajectory records must be dictionaries")
        turn_idx = int(record.get("turn_idx", -1))
        if turn_idx != expected_turn_idx:
            raise ValueError(
                "sdpo_privileged_trajectory turn indices must be contiguous from zero: "
                f"expected {expected_turn_idx}, got {turn_idx}"
            )
        if "projected_action" not in record:
            raise ValueError(f"sdpo_privileged_trajectory turn {turn_idx} is missing projected_action")
        if "next_anchor_obs" not in record:
            raise ValueError(f"sdpo_privileged_trajectory turn {turn_idx} is missing next_anchor_obs")
        trajectory.append(dict(record))
    return trajectory


def _build_no_success_teacher_context(
    args: Any,
    row: dict[str, Any],
    trajectories: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    allow_failure_guidance: bool,
    allow_failed_negative: bool,
) -> dict[str, Any]:
    current_traj = _active_rows(trajectories.get((row["uid"], row["traj_uid"]), [row])) or [row]
    if (
        allow_failed_negative
        and _no_success_context_mode(args) == "failed_negative"
        and str(getattr(args, "sdpo_solution_context_format", "guidance_plan")) == "trajectory_demo"
    ):
        return _build_failed_negative_trajectory_context(args, row, current_traj)
    failure_text = ""
    if (
        allow_failure_guidance
        and str(getattr(args, "sdpo_solution_context_format", "guidance_plan")) == "guidance_plan"
    ):
        failure_text = _failure_text(args, current_traj)
    feedback_text = _feedback_text(args, row, has_solution=False, has_failure=bool(failure_text))
    if failure_text:
        prompt, messages = _build_teacher_prompt(args, row)
        return _plan_row(
            args,
            row,
            prompt,
            messages,
            signal="failure_guidance_prompt",
            active=False,
            failure_guidance_prompt=failure_text,
            failure_guidance_payload=_guidance_payload_for_records(failure_text, current_traj),
        )
    prompt, messages = _build_teacher_prompt(args, row, feedback_text=feedback_text)
    return _plan_row(
        args,
        row,
        prompt,
        messages,
        signal="feedback" if feedback_text else "none",
        active=bool(feedback_text),
        feedback_text=feedback_text,
    )


def _build_failed_negative_trajectory_context(
    args: Any,
    row: dict[str, Any],
    trajectory: list[dict[str, Any]],
) -> dict[str, Any]:
    failure_text = _build_trajectory_demo(trajectory, max_demo_steps=None)
    prompt, messages = _build_failed_negative_teacher_prompt(args, row, failure_text)
    return _plan_row(
        args,
        row,
        prompt,
        messages,
        signal="failed_negative_demo",
        active=bool(failure_text),
        failure_text=failure_text,
        reference_scope="own_failed_negative",
    )


def _plan_row(
    args: Any,
    row: dict[str, Any],
    prompt: str,
    messages: list[dict[str, Any]],
    *,
    signal: str,
    active: bool,
    solution_text: str = "",
    failure_text: str = "",
    feedback_text: str = "",
    success_guidance_prompt: str = "",
    success_guidance_text: str = "",
    failure_guidance_prompt: str = "",
    failure_guidance_text: str = "",
    selected_success_traj_uid: str | None = None,
    success_is_self: bool | None = None,
    success_guidance_payload: dict[str, Any] | None = None,
    failure_guidance_payload: dict[str, Any] | None = None,
    reference_scope: str = "",
    filter_reason: str = "",
    representation_success_aggregation: str = "sample",
    representation_success_count: int = 0,
    teacher_prompt_texts: list[str] | None = None,
    teacher_messages_list: list[list[dict[str, Any]]] | None = None,
    token_weight_contrast_prompt_text: str | None = None,
    token_weight_contrast_messages: list[dict[str, Any]] | None = None,
    token_weight_contrast_source: str = "",
    token_weight_contrast_traj_uid: str | None = None,
) -> dict[str, Any]:
    raw_loss_weight = _raw_loss_weight(args, row)
    result = {
        "_sdpo_args": args,
        "sdpo_metadata": row,
        "sdpo_teacher_prompt_text": prompt,
        "sdpo_teacher_messages": messages,
        "sdpo_teacher_signal_type": signal,
        "self_distillation_mask": 1.0 if active else 0.0,
        "sdpo_raw_loss_weight": raw_loss_weight,
        "sdpo_loss_weights": raw_loss_weight,
        "sdpo_solution_text": solution_text,
        "sdpo_failure_text": failure_text,
        "sdpo_feedback_text": feedback_text,
        "sdpo_success_guidance_prompt": success_guidance_prompt,
        "sdpo_success_guidance_text": success_guidance_text,
        "sdpo_failure_guidance_prompt": failure_guidance_prompt,
        "sdpo_failure_guidance_text": failure_guidance_text,
        "sdpo_context_reference_scope": reference_scope,
        "sdpo_filter_reason": filter_reason,
        "sdpo_representation_success_aggregation": representation_success_aggregation,
        "sdpo_representation_success_count": int(representation_success_count),
        "sdpo_teacher_prompt_texts": list(teacher_prompt_texts) if teacher_prompt_texts else None,
        "sdpo_teacher_messages_list": (
            [[dict(message) for message in messages] for messages in teacher_messages_list]
            if teacher_messages_list
            else None
        ),
        "sdpo_token_weight_contrast_prompt_text": token_weight_contrast_prompt_text,
        "sdpo_token_weight_contrast_messages": (
            [dict(message) for message in token_weight_contrast_messages] if token_weight_contrast_messages else None
        ),
        "sdpo_token_weight_contrast_source": token_weight_contrast_source,
        "sdpo_token_weight_contrast_traj_uid": token_weight_contrast_traj_uid,
    }
    if row.get(_ROW_PRECOMPUTED_GUIDANCE_OUTPUT_KEY) is not None:
        result[_ROW_PRECOMPUTED_GUIDANCE_OUTPUT_KEY] = str(row[_ROW_PRECOMPUTED_GUIDANCE_OUTPUT_KEY])
    if success_guidance_payload:
        result["sdpo_success_guidance_payload"] = _copy_guidance_payload(success_guidance_payload)
    if failure_guidance_payload:
        result["sdpo_failure_guidance_payload"] = _copy_guidance_payload(failure_guidance_payload)
    if selected_success_traj_uid is not None:
        result["sdpo_selected_success_traj_uid"] = selected_success_traj_uid
    if success_is_self is not None:
        result["sdpo_success_is_self"] = success_is_self
    return result


def _success_candidates(args: Any, row: dict[str, Any], candidates: list[list[dict[str, Any]]]):
    if bool(getattr(args, "sdpo_dont_reprompt_on_self_success", False)):
        candidates = [candidate for candidate in candidates if candidate[-1]["traj_uid"] != row["traj_uid"]]
    aggregation = _representation_success_aggregation(args)
    if aggregation == "mean":
        return list(candidates)
    if aggregation == "random" and candidates:
        return [candidates[_random_success_candidate_index(args, row, len(candidates))]]
    return candidates[:1]


def _representation_success_aggregation(args: Any) -> str:
    aggregation = str(getattr(args, "sdpo_representation_success_aggregation", "sample")).strip().lower()
    if aggregation == "random":
        return "random"
    if (
        aggregation == "mean"
        and str(getattr(args, "sdpo_distillation_mode", "")) == "representation"
        and str(getattr(args, "sdpo_solution_context_format", "guidance_plan")) == "trajectory_demo"
    ):
        return "mean"
    return "sample"


def _random_success_candidate_index(args: Any, row: dict[str, Any], candidate_count: int) -> int:
    seed = getattr(args, "rollout_seed", getattr(args, "seed", 0))
    key = f"{seed}|{row['uid']}|{row['traj_uid']}|{row['turn_idx']}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % candidate_count


def _token_weight_contrast_context(
    args: Any,
    row: dict[str, Any],
    failed_candidates: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    if not bool(getattr(args, "sdpo_token_weights", False)):
        return {}
    if str(getattr(args, "sdpo_token_weight_source", "teacher_contrast")) == "student":
        return {}
    if failed_candidates:
        candidate = failed_candidates[_random_failed_candidate_index(args, row, len(failed_candidates))]
        failure_text = _build_trajectory_demo(candidate, _max_demo_steps(args))
        prompt, messages = _build_failed_negative_teacher_prompt(args, row, failure_text)
        return {
            "token_weight_contrast_prompt_text": prompt,
            "token_weight_contrast_messages": messages,
            "token_weight_contrast_source": "failed",
            "token_weight_contrast_traj_uid": str(candidate[-1]["traj_uid"]),
        }
    prompt, messages = _build_teacher_prompt(args, row)
    return {
        "token_weight_contrast_prompt_text": prompt,
        "token_weight_contrast_messages": messages,
        "token_weight_contrast_source": "no_extra_context",
    }


def _random_failed_candidate_index(args: Any, row: dict[str, Any], candidate_count: int) -> int:
    seed = getattr(args, "rollout_seed", getattr(args, "seed", 0))
    key = f"{seed}|failed|{row['uid']}|{row['traj_uid']}|{row['turn_idx']}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % candidate_count


def _build_teacher_prompt(
    args: Any,
    row: dict[str, Any],
    *,
    solution_text: str = "",
    failure_text: str = "",
    feedback_text: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    mode = str(getattr(args, "sdpo_teacher_context_mode", "original"))
    if mode not in {"original", "feedback_only", "solution_only", "own_outcome"}:
        raise ValueError(f"Unknown sdpo_teacher_context_mode: {mode}")
    if mode == "feedback_only":
        solution_text = ""
        failure_text = ""
    elif mode == "solution_only":
        feedback_text = ""
    solution_template_attr = "sdpo_solution_template"
    solution_template_default = _SOLUTION_TEMPLATE
    if (
        solution_text
        and str(getattr(args, "sdpo_solution_context_format", "guidance_plan")) == "trajectory_demo"
        and not hasattr(args, "sdpo_solution_template")
    ):
        solution_template_attr = "sdpo_trajectory_solution_template"
        solution_template_default = _TRAJECTORY_DEMO_SOLUTION_TEMPLATE
    solution = _format_template(args, solution_template_attr, solution_template_default, solution_text)
    failure = _format_template(args, "sdpo_failure_template", _FAILURE_TEMPLATE, failure_text)
    feedback = _format_template(args, "sdpo_feedback_template", _FEEDBACK_TEMPLATE, feedback_text)
    base_prompt = str(row["sdpo_current_prompt_text"])
    if solution or failure or feedback:
        prompt = str(getattr(args, "sdpo_reprompt_template", _REPROMPT_TEMPLATE)).format(
            prompt=base_prompt,
            solution=solution,
            failure=failure,
            feedback=feedback,
        )
    else:
        prompt = base_prompt
    return prompt, _build_teacher_messages(row, prompt)




def _build_failed_negative_teacher_prompt(
    args: Any,
    row: dict[str, Any],
    failure_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    failure = _format_template(
        args,
        "sdpo_trajectory_failure_template",
        _TRAJECTORY_DEMO_FAILURE_TEMPLATE,
        failure_text,
    )
    prompt = f"{row['sdpo_current_prompt_text']}{failure}" if failure else str(row["sdpo_current_prompt_text"])
    return prompt, _build_teacher_messages(row, prompt)


def _build_controlled_feedback_teacher_prompt(
    row: dict[str, Any],
    feedback_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    prompt = f"{row['sdpo_current_prompt_text']}\n\n{feedback_text}\n\n{_CONTROLLED_CONTEXT_SUFFIX}"
    return prompt, _build_teacher_messages(row, prompt)


def build_controlled_outcome_guidance_prompt(
    prompt_text: str,
    messages: list[dict[str, Any]],
    guidance_text: str,
    *,
    is_success: bool,
    label_mode: str = "explicit",
) -> tuple[str, list[dict[str, Any]]]:
    """Append canonical controlled outcome guidance to the current user turn."""
    guidance = str(guidance_text).strip()
    if not guidance:
        raise ValueError("controlled outcome guidance must be non-empty")
    if label_mode not in {"explicit", "omitted"}:
        raise ValueError(f"unsupported controlled outcome label mode: {label_mode!r}")
    heading_key = "omitted" if label_mode == "omitted" else ("success" if is_success else "failure")
    heading = _CONTROLLED_OUTCOME_HEADINGS[heading_key]
    prompt = f"{prompt_text}\n\n{heading}\n\n{guidance}\n\n{_CONTROLLED_CONTEXT_SUFFIX}"
    if not messages:
        raise ValueError("controlled outcome guidance requires non-empty messages")
    updated = [dict(message) for message in messages]
    last = dict(updated[-1])
    if str(last.get("role") or "") != "user":
        raise ValueError("controlled outcome guidance requires a final user message")
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = f"{content}\n\n{heading}\n\n{guidance}\n\n{_CONTROLLED_CONTEXT_SUFFIX}"
    elif isinstance(content, list):
        image_items = _image_items_from_messages([last])
        last["content"] = _content_with_images(prompt, image_items)
    else:
        raise ValueError("controlled outcome guidance requires text or multimodal-list final user content")
    updated[-1] = last
    return prompt, updated


def _build_controlled_trajectory_teacher_prompt(
    args: Any,
    row: dict[str, Any],
    trajectory_text: str,
    *,
    is_success: bool,
) -> tuple[str, list[dict[str, Any]]]:
    override = row.get("sdpo_controlled_teacher_prompt_override")
    if override is not None:
        if not isinstance(override, dict):
            raise ValueError("sdpo controlled teacher prompt override must be a mapping")
        prompt = str(override.get("prompt_text") or "")
        messages = override.get("messages")
        guidance = str(override.get("guidance_text") or "").strip()
        if not prompt or not isinstance(messages, list) or not messages:
            raise ValueError("sdpo controlled teacher prompt override is incomplete")
        if guidance != str(trajectory_text).strip():
            raise ValueError("sdpo controlled teacher prompt override guidance does not match trajectory text")
        return prompt, [dict(message) for message in messages]
    prompt, messages = build_controlled_outcome_guidance_prompt(
        str(row["sdpo_current_prompt_text"]),
        _build_teacher_messages(row, str(row["sdpo_current_prompt_text"])),
        trajectory_text,
        is_success=is_success,
        label_mode=_own_outcome_label_mode(args),
    )
    return prompt, messages


def _build_teacher_messages(row: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    raw_chat = row.get("sdpo_current_raw_prompt")
    if isinstance(raw_chat, list):
        if _IMAGE_PLACEHOLDER in prompt:
            image_items = _image_items_from_messages([raw_chat[-1]]) or _image_items_from_messages(raw_chat)
            return [dict(message) for message in raw_chat[:-1]] + [
                {"role": "user", "content": _content_with_images(prompt, image_items)}
            ]
        return [dict(message) for message in raw_chat[:-1]] + [{"role": "user", "content": prompt}]
    return [{"role": "user", "content": prompt}]


def _solution_text(args: Any, candidate: list[dict[str, Any]]) -> str:
    if str(getattr(args, "sdpo_solution_context_format", "guidance_plan")) == "guidance_plan":
        evidence = _build_trajectory_evidence(candidate, _max_demo_steps(args))
        if str(getattr(args, "sdpo_guidance_summary_schema", "legacy")) == "own_outcome_detailed":
            return _build_own_outcome_success_guidance_summary_prompt(_task_text(candidate), evidence)
        return _build_success_guidance_summary_prompt(_task_text(candidate), evidence)
    return _build_trajectory_demo(candidate, _max_demo_steps(args))


def _failure_text(args: Any, trajectory: list[dict[str, Any]]) -> str:
    evidence = _build_trajectory_evidence(trajectory, _max_demo_steps(args))
    if str(getattr(args, "sdpo_guidance_summary_schema", "legacy")) == "own_outcome_detailed":
        return _build_own_outcome_failure_guidance_summary_prompt(_task_text(trajectory), evidence)
    return _build_failure_guidance_summary_prompt(_task_text(trajectory), evidence)


def _guidance_payload_for_records(prompt: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    image_data = _guidance_image_data(records)
    placeholder_count = str(prompt).count(_IMAGE_PLACEHOLDER)
    if placeholder_count >= 0:
        image_data = image_data[:placeholder_count]
    payload: dict[str, Any] = {
        "prompt": prompt,
        "messages": _guidance_messages(prompt, image_data),
        "image_data": image_data,
        "image_hashes": [_image_hash(image) for image in image_data],
    }
    return payload


def _build_success_guidance_summary_prompt(task_text: str, trajectory_demo_text: str) -> str:
    return (
        f"{_SUCCESS_GUIDANCE_SUMMARY_INSTRUCTIONS}\n\n"
        f"Task snapshot:\n{_guidance_task_snapshot(task_text)}\n\n"
        f"Successful trajectory evidence:\n{trajectory_demo_text}"
    ).strip()


def _build_failure_guidance_summary_prompt(task_text: str, trajectory_demo_text: str) -> str:
    return (
        f"{_FAILURE_GUIDANCE_SUMMARY_INSTRUCTIONS}\n\n"
        f"Task snapshot:\n{_guidance_task_snapshot(task_text)}\n\n"
        f"Unsuccessful trajectory evidence:\n{trajectory_demo_text}"
    ).strip()


def _build_own_outcome_success_guidance_summary_prompt(task_text: str, trajectory_demo_text: str) -> str:
    return (
        f"{_OWN_OUTCOME_SUCCESS_GUIDANCE_SUMMARY_INSTRUCTIONS}\n\n"
        f"Task snapshot:\n{_guidance_task_snapshot(task_text)}\n\n"
        f"Successful trajectory evidence:\n{trajectory_demo_text}"
    ).strip()


def _build_own_outcome_failure_guidance_summary_prompt(task_text: str, trajectory_demo_text: str) -> str:
    return (
        f"{_OWN_OUTCOME_FAILURE_GUIDANCE_SUMMARY_INSTRUCTIONS}\n\n"
        f"Task snapshot:\n{_guidance_task_snapshot(task_text)}\n\n"
        f"Unsuccessful trajectory evidence:\n{trajectory_demo_text}"
    ).strip()




















def _build_trajectory_demo(records: list[dict[str, Any]], max_demo_steps: int | None) -> str:
    selected = records if max_demo_steps is None else records[:max_demo_steps]
    lines: list[str] = []
    has_action = False
    step_no = 0
    for record in selected:
        action = _compact_text(record.get("projected_action"))
        result = _compact_text(record.get("next_anchor_obs"))
        if not action:
            continue
        step_no += 1
        has_action = True
        lines.append(f"Step {step_no}")
        lines.append(f"Action: `{action}`")
        if result:
            lines.append(f"Observation: {result}")
        lines.append("")
    if not has_action:
        lines.append("No explicit action sequence was available.")
    return "\n".join(lines).strip()


def _build_trajectory_evidence(records: list[dict[str, Any]], max_demo_steps: int | None) -> str:
    selected = records if max_demo_steps is None else records[:max_demo_steps]
    lines: list[str] = []
    action_count = 0
    for record in selected:
        action = _limit_text(_compact_text(record.get("projected_action")), 240)
        if not action:
            continue
        action_count += 1
        result = _limit_text(_compact_text(record.get("next_anchor_obs")), 480)
        lines.append(f"{action_count}. Action: `{action}`")
        if result:
            lines.append(f"   Result: {result}")
    if not lines:
        lines.append("No explicit action sequence was available.")
    return "\n".join(lines).strip()






def _guidance_task_snapshot(task_text: str) -> str:
    text = _compact_text(task_text)
    if not text:
        return "Goal: unavailable"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("goal:"):
            return _limit_text(stripped, 240)
    compact_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(_limit_text(line, 240) for line in compact_lines[:4]).strip()








def _guidance_image_data(records: list[dict[str, Any]]) -> list[str]:
    images: list[str] = []
    selected_records = [record for record in records if record.get("projected_action")]
    for record in selected_records:
        image = record.get("sdpo_guidance_image_data")
        if image:
            images.append(str(image))
    return images


def _guidance_messages(prompt: str, image_data: list[str]) -> list[dict[str, Any]]:
    if image_data:
        return [{"role": "user", "content": _content_with_images(prompt, _image_items_from_data(image_data))}]
    return [{"role": "user", "content": prompt}]


def _image_items_from_data(image_data: list[str]) -> list[dict[str, Any]]:
    return [{"type": "image", "image": image} for image in image_data]


def _image_items_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    image_items: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                image_items.append(dict(item))
    return image_items


def _content_with_images(text: str, image_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not image_items:
        return [{"type": "text", "text": text}]
    parts = str(text).split(_IMAGE_PLACEHOLDER)
    content: list[dict[str, Any]] = []
    for idx, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if idx < len(parts) - 1:
            image_item = image_items[min(idx, len(image_items) - 1)]
            content.append(dict(image_item))
    if len(parts) == 1:
        content.append(dict(image_items[0]))
    return content


def _copy_guidance_payload(payload: dict[str, Any]) -> dict[str, Any]:
    copied = dict(payload)
    if isinstance(copied.get("messages"), list):
        copied["messages"] = copy.deepcopy(copied["messages"])
    if isinstance(copied.get("image_data"), list):
        copied["image_data"] = list(copied["image_data"])
    if isinstance(copied.get("image_hashes"), list):
        copied["image_hashes"] = list(copied["image_hashes"])
    return copied


def _image_hash(image: str) -> str:
    return hashlib.sha256(str(image).encode("utf-8")).hexdigest()[:16]








def _feedback_text(args: Any, row: dict[str, Any], *, has_solution: bool, has_failure: bool) -> str:
    mode = str(getattr(args, "sdpo_teacher_context_mode", "original"))
    include_feedback = bool(getattr(args, "sdpo_include_environment_feedback", True))
    feedback_only_without_solution = bool(getattr(args, "sdpo_environment_feedback_only_without_solution", True))
    if mode not in {"original", "feedback_only"} or not include_feedback:
        return ""
    if mode == "original" and feedback_only_without_solution and (has_solution or has_failure):
        return ""
    action = _compact_text(row.get("projected_action"))
    next_obs = _compact_text(row.get("next_anchor_obs"))
    if action and next_obs:
        return f"If you take action `{action}`, the next state is:\n{next_obs}"
    if next_obs:
        return f"The next state is:\n{next_obs}"
    return ""


def _controlled_feedback_text(row: dict[str, Any]) -> str:
    action = _compact_text(row.get("projected_action"))
    next_obs = _compact_text(row.get("next_anchor_obs"))
    if not action or not next_obs:
        raise ValueError("Controlled feedback_only prompt requires projected_action and next_anchor_obs.")
    return f"If the current action is:\n`{action}`\n\nObserved next state:\n{next_obs}"


def _context_prompt_style(args: Any) -> str:
    return str(getattr(args, "sdpo_context_prompt_style", "legacy")).strip().lower()


def _own_outcome_label_mode(args: Any) -> str:
    return str(getattr(args, "sdpo_own_outcome_label_mode", "explicit")).strip().lower()


def _validate_context_prompt_config(args: Any) -> None:
    style = _context_prompt_style(args)
    label_mode = _own_outcome_label_mode(args)
    teacher_mode = str(getattr(args, "sdpo_teacher_context_mode", "original")).strip().lower()
    if style not in {"legacy", "controlled"}:
        raise ValueError("sdpo_context_prompt_style must be legacy or controlled.")
    if label_mode not in {"explicit", "omitted"}:
        raise ValueError("sdpo_own_outcome_label_mode must be explicit or omitted.")
    if style == "controlled" and teacher_mode not in {"feedback_only", "own_outcome"}:
        raise ValueError(
            "sdpo_context_prompt_style=controlled requires "
            "sdpo_teacher_context_mode=feedback_only or own_outcome."
        )
    if label_mode == "omitted" and not (style == "controlled" and teacher_mode == "own_outcome"):
        raise ValueError(
            "sdpo_own_outcome_label_mode=omitted requires "
            "sdpo_context_prompt_style=controlled and sdpo_teacher_context_mode=own_outcome."
        )


def _format_template(args: Any, attr: str, default: str, value: str) -> str:
    if not value:
        return ""
    template = str(getattr(args, attr, default))
    if attr in {"sdpo_solution_template", "sdpo_trajectory_solution_template"}:
        return template.format(successful_previous_attempt=value, value=value)
    if attr in {"sdpo_failure_template", "sdpo_trajectory_failure_template"}:
        return template.format(unsuccessful_previous_attempt=value)
    if attr == "sdpo_feedback_template":
        return template.format(feedback_raw=value)
    return template.format(value=value)


def _guidance_summary_source(args: Any) -> str:
    source = str(getattr(args, "sdpo_guidance_summary_source", "success_priority")).strip().lower()
    if source not in _GUIDANCE_SUMMARY_SOURCES:
        expected = ", ".join(sorted(_GUIDANCE_SUMMARY_SOURCES))
        raise ValueError(f"Unknown sdpo_guidance_summary_source: {source!r}; expected {expected}")
    return source


def _no_success_context_mode(args: Any) -> str:
    mode = str(getattr(args, "sdpo_no_success_context_mode", "feedback")).strip().lower()
    if mode not in {"feedback", "filter", "failed_negative"}:
        raise ValueError("sdpo_no_success_context_mode must be feedback, filter, or failed_negative.")
    return mode


def _guidance_prompt_output_map(
    plan: list[dict[str, Any]], outputs: dict[str, str] | list[str] | None
) -> dict[str, str]:
    if outputs is None:
        outputs = {}
    if isinstance(outputs, list):
        prompts = collect_guidance_summary_prompts(plan)
        if len(outputs) != len(prompts):
            raise ValueError(f"guidance_summary_outputs size mismatch: expected {len(prompts)}, got {len(outputs)}")
        return {prompt: clean_guidance_summary(output) for prompt, output in zip(prompts, outputs)}
    return {str(prompt): clean_guidance_summary(output) for prompt, output in outputs.items()}


def _guidance_output_or_fallback(
    prompt_to_output: dict[str, str],
    prompt: str,
    *,
    kind: str,
    summary_schema: str = "",
) -> tuple[str, bool]:
    if prompt not in prompt_to_output:
        raise ValueError(f"Missing SDPO {kind} guidance summary output for prompt.")
    output = str(prompt_to_output[prompt] or "").strip()
    if _guidance_output_has_schema(output, kind):
        return output, False
    if output:
        return _repair_guidance_summary_schema(output, prompt, kind=kind), True
    return _fallback_guidance_summary(prompt, kind=kind), True


def _guidance_output_has_schema(text: str, kind: str) -> bool:
    header_kinds = _guidance_output_header_kinds(text)
    if not header_kinds:
        return False
    if kind == "success":
        return header_kinds[0] == "success" and all(header == "success" for header in header_kinds)
    if kind == "failure":
        return header_kinds[0] == "failure" and all(header == "failure" for header in header_kinds)
    return False


def _guidance_output_header_kinds(text: str) -> list[str]:
    header_kinds: list[str] = []
    for line in str(text or "").splitlines():
        normalized = line.strip().lower().lstrip("#").strip()
        if normalized.startswith("guidance summary:"):
            header_kinds.append("success")
        elif normalized.startswith("failure analysis:"):
            header_kinds.append("failure")
    return header_kinds


def _repair_guidance_summary_schema(text: str, prompt: str, *, kind: str) -> str:
    if any(
        header == ("failure" if kind == "success" else "success") for header in _guidance_output_header_kinds(text)
    ):
        return _fallback_guidance_summary(prompt, kind=kind)
    header = "Failure analysis:" if kind == "failure" else "Guidance summary:"
    return f"{header}\n{text.strip()}"


def _fallback_guidance_summary(prompt: str, *, kind: str) -> str:
    actions = _extract_guidance_actions(prompt)
    action_text = ", ".join(f"`{action}`" for action in actions) if actions else "the evidence action sequence"
    is_visual = "observation image" in prompt.lower() or _IMAGE_PLACEHOLDER in prompt
    if kind == "failure":
        if is_visual:
            return (
                "Failure analysis:\n"
                "- Likely mistake: the attempted visual action sequence did not solve the board.\n"
                "- Visual / state evidence: re-check the player, box, and target positions before acting.\n"
                f"- Avoid: repeating {action_text} without confirming that it changes the state toward the target."
            )
        return (
            "Failure analysis:\n"
            "- Likely mistake: the attempted action sequence did not complete the task.\n"
            f"- Avoid: repeating {action_text} without checking the resulting state."
        )
    if is_visual:
        return (
            "Guidance summary:\n"
            "- Minimal plan: follow the successful visual action pattern from the evidence.\n"
            f"- Critical actions: {action_text}.\n"
            "- Visual / state checks: compare the player, box, and target positions before each move.\n"
            "- Avoid: moves that leave the state unchanged or push the box away from the target."
        )
    return (
        "Guidance summary:\n"
        "- Minimal plan: follow the successful action pattern from the evidence.\n"
        f"- Critical actions: {action_text}.\n"
        "- Checks / avoid: verify each action changes the state toward the goal before continuing."
    )


def _extract_guidance_actions(prompt: str) -> list[str]:
    actions: list[str] = []
    seen: set[str] = set()
    for action in re.findall(r"Action:\s*`([^`]+)`", prompt):
        action = _compact_text(action)
        if not action or action in seen:
            continue
        seen.add(action)
        actions.append(_limit_text(action, 80))
        if len(actions) >= 6:
            break
    return actions


def _copy_plan_row(row: dict[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    if isinstance(copied.get("sdpo_metadata"), dict):
        copied["sdpo_metadata"] = dict(copied["sdpo_metadata"])
    if isinstance(copied.get("sdpo_teacher_messages"), list):
        copied["sdpo_teacher_messages"] = [dict(message) for message in copied["sdpo_teacher_messages"]]
    if isinstance(copied.get("sdpo_token_weight_contrast_messages"), list):
        copied["sdpo_token_weight_contrast_messages"] = [
            dict(message) for message in copied["sdpo_token_weight_contrast_messages"]
        ]
    if isinstance(copied.get("sdpo_teacher_prompt_texts"), list):
        copied["sdpo_teacher_prompt_texts"] = list(copied["sdpo_teacher_prompt_texts"])
    if isinstance(copied.get("sdpo_teacher_messages_list"), list):
        copied["sdpo_teacher_messages_list"] = [
            [dict(message) for message in messages] for messages in copied["sdpo_teacher_messages_list"]
        ]
    for key in ("sdpo_success_guidance_payload", "sdpo_failure_guidance_payload"):
        if isinstance(copied.get(key), dict):
            copied[key] = _copy_guidance_payload(copied[key])
    return copied


def _deactivate(row: dict[str, Any]) -> None:
    row["sdpo_teacher_signal_type"] = "none"
    row["self_distillation_mask"] = 0.0
    row["sdpo_loss_weights"] = 0.0


def _raw_loss_weight(args: Any, row: dict[str, Any]) -> float:
    if not bool(row.get("algorithm_active_mask", True)):
        return 0.0
    length = max(int(row["episode_lengths"]), 1)
    weighting = str(getattr(args, "sdpo_multi_turn_weighting", "traj_equal"))
    if weighting == "step_equal":
        return float(length)
    if weighting == "traj_equal":
        return 1.0
    if weighting == "hybrid":
        return math.sqrt(length)
    if weighting == "inverse_length":
        return 1.0 / float(length)
    raise ValueError(f"Unknown sdpo_multi_turn_weighting: {weighting}")


def _normalize_active_weights(plan: list[dict[str, Any]]) -> None:
    positive = [row for row in plan if float(row["sdpo_loss_weights"]) > 0.0]
    total = sum(float(row["sdpo_loss_weights"]) for row in positive)
    if total <= 0:
        return
    scale = len(positive) / total
    for row in positive:
        row["sdpo_loss_weights"] = float(row["sdpo_loss_weights"]) * scale


def _task_text(records: list[dict[str, Any]]) -> str:
    for record in records:
        text = _compact_text(record.get("task_text"))
        if text:
            return text
    return ""


def _active_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if bool(row.get("algorithm_active_mask", True))]


def _max_demo_steps(args: Any) -> int | None:
    value = getattr(args, "sdpo_max_demo_steps", None)
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"}):
        return None
    steps = int(value)
    return steps if steps > 0 else None


def _compact_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _limit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 16)].rstrip() + " ... [truncated]"
