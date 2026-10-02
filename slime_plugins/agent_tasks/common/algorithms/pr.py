"""Privileged Retention (PR): construct privileged-view targets and loss rows."""

from __future__ import annotations

import math
from typing import Any

from .sgs import PRECOMPUTED_FIELDS

PR_COMPONENT_FIELD = "pr_component"
PR_BASE_WEIGHT_FIELD = "pr_base_weight"


def expand_pr_rows(
    args: Any,
    train_data: dict[str, Any],
    *,
    retention_weight: float,
    selected_indices: list[int] | None = None,
    retention_base_weights: list[float] | None = None,
    retention_precomputed: list[dict[str, Any]] | None = None,
) -> None:
    """Build selected ordinary rows plus the configured retention support.

    Component-specific scales compensate for the joint row-count normalizer, so
    selected-only and all-step retention both compute ``L_sel + lambda * L_ret``.
    """
    if not math.isfinite(retention_weight) or retention_weight <= 0.0:
        raise ValueError("privileged retention weight must be finite and positive")
    source_row_count = len(train_data["tokens"])
    if source_row_count == 0:
        raise ValueError("privileged retention requires selected rows")
    present_precomputed = [field in train_data for field in PRECOMPUTED_FIELDS]
    if any(present_precomputed) and not all(present_precomputed):
        raise ValueError("retention received a partial set of precomputed selected-step distillation tensors")
    if not any(present_precomputed):
        for field in PRECOMPUTED_FIELDS:
            train_data[field] = [None] * source_row_count

    from slime.algorithms.sdpo.teacher_alignment import build_sdpo_teacher_rollout_data
    from slime.rollout.sglang_rollout import GenerateState

    selected_indices = list(range(source_row_count)) if selected_indices is None else list(selected_indices)
    if (
        not selected_indices
        or len(set(selected_indices)) != len(selected_indices)
        or any(index < 0 or index >= source_row_count for index in selected_indices)
    ):
        raise ValueError("privileged retention selected indices are invalid")
    support = str(getattr(args, "pr_support", "selected"))
    retention_indices = list(range(source_row_count)) if support == "all" else selected_indices
    source_data = {
        key: (
            [value[index] for index in retention_indices]
            if isinstance(value, list) and len(value) == source_row_count
            else value
        )
        for key, value in train_data.items()
    }
    state = GenerateState(args)
    retention_view = str(getattr(args, "pr_view", "privileged"))
    if retention_view == "privileged":
        retention_student = build_sdpo_teacher_rollout_data(
            source_data,
            state.tokenizer,
            processor=state.processor,
            apply_chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
            max_prompt_tokens=getattr(args, "sdpo_max_reprompt_tokens", None),
            truncation_side=getattr(args, "sdpo_reprompt_truncation_side", "right"),
        )
    elif retention_view == "ordinary":
        required = ("sgs_plain_prompt_text", "sgs_plain_messages")
        missing = [field for field in required if field not in source_data]
        if missing:
            raise ValueError(f"ordinary retention is missing plain-view prompt fields: {missing}")
        retention_student = {
            "tokens": list(source_data["tokens"]),
            "total_lengths": [len(tokens) for tokens in source_data["tokens"]],
            "response_lengths": list(source_data["response_lengths"]),
            "loss_masks": list(source_data["loss_masks"]),
        }
    else:
        raise ValueError(f"unsupported retention view: {retention_view}")
    source_weights = [float(value) for value in train_data["sdpo_loss_weights"]]
    if len(source_weights) != source_row_count:
        raise ValueError("privileged retention base weights do not match selected rows")
    selection_base_weights = [source_weights[index] for index in selected_indices]
    if retention_base_weights is None:
        retention_base_weights = list(selection_base_weights)
    if len(retention_base_weights) != len(retention_indices):
        raise ValueError("privileged retention weights do not match retention support")
    if retention_precomputed is not None:
        if len(retention_precomputed) != len(retention_indices) or any(
            any(field not in payload for field in PRECOMPUTED_FIELDS) for payload in retention_precomputed
        ):
            raise ValueError("cached privileged retention targets do not match retention support")

    original = {key: list(value) for key, value in train_data.items() if isinstance(value, list)}
    for key, rows in original.items():
        if len(rows) != source_row_count:
            continue
        selected_rows = [rows[index] for index in selected_indices]
        if key in {"tokens", "total_lengths", "response_lengths", "loss_masks"}:
            appended = list(retention_student[key])
        elif retention_view == "ordinary" and key == "sdpo_teacher_prompt_text":
            appended = list(source_data["sgs_plain_prompt_text"])
        elif retention_view == "ordinary" and key == "sdpo_teacher_messages":
            appended = list(source_data["sgs_plain_messages"])
        elif key in PRECOMPUTED_FIELDS:
            appended = (
                [payload[key] for payload in retention_precomputed]
                if retention_precomputed is not None
                else [None] * len(retention_indices)
            )
        elif key == "sample_indices":
            appended = [-(int(rows[index]) + 1) for index in retention_indices]
        elif key == "sdpo_metadata":
            appended = [
                {
                    **rows[index],
                    "sdpo_objective_component": (
                        "privileged_retention" if retention_view == "privileged" else "ordinary_retention"
                    ),
                    **({"pr_view": retention_view} if retention_view != "privileged" else {}),
                }
                for index in retention_indices
            ]
        else:
            appended = [rows[index] for index in retention_indices]
        train_data[key] = selected_rows + appended

    selected_count = len(selected_indices)
    retention_count = len(retention_indices)
    joint_count = selected_count + retention_count
    selection_scale = joint_count / selected_count
    retention_scale = joint_count / retention_count
    train_data["sdpo_loss_weights"] = [selection_scale * weight for weight in selection_base_weights] + [
        retention_scale * retention_weight * weight for weight in retention_base_weights
    ]
    train_data[PR_BASE_WEIGHT_FIELD] = selection_base_weights + list(retention_base_weights)
    train_data["pr_normalization_scale"] = [selection_scale] * selected_count + [retention_scale] * retention_count
    train_data[PR_COMPONENT_FIELD] = [0.0] * selected_count + [1.0] * retention_count
    train_data["self_distillation_mask"] = [1.0] * joint_count
