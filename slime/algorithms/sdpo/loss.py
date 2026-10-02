from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def compute_sdpo_token_logprob_loss(
    *,
    student_log_probs: Any,
    sdpo_teacher_log_probs: Any | None = None,
    loss_masks: Any,
    self_distillation_mask: Any,
    sdpo_loss_weights: Any,
    response_mask: Any | None = None,
    clip_ratio: float | None = None,
    old_log_probs: Any | None = None,
    deployment_behavior_log_probs: Any | None = None,
    deployment_tis_clip: float | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Compute SDPO's token-logprob reverse-KL surrogate on aligned response tokens."""
    if "teacher_log_probs" in kwargs:
        raise ValueError("SDPO loss requires sdpo_teacher_log_probs; OPD teacher_log_probs is not accepted.")
    if sdpo_teacher_log_probs is None:
        raise ValueError("SDPO loss requires sdpo_teacher_log_probs; do not use OPD teacher_log_probs.")

    student_rows = _as_row_tensors(student_log_probs, name="student_log_probs")
    if not student_rows:
        raise ValueError("student_log_probs must contain at least one row.")
    teacher_rows = _as_row_tensors(sdpo_teacher_log_probs, name="sdpo_teacher_log_probs", like_rows=student_rows)
    loss_mask_rows = _as_row_tensors(loss_masks, name="loss_masks", like_rows=student_rows)
    response_mask_rows = (
        [torch.ones_like(row) for row in student_rows]
        if response_mask is None
        else _as_row_tensors(response_mask, name="response_mask", like_rows=student_rows)
    )

    if not (len(student_rows) == len(teacher_rows) == len(loss_mask_rows) == len(response_mask_rows)):
        raise ValueError("SDPO loss row count mismatch.")

    sample_mask = _as_vector_tensor(self_distillation_mask, name="self_distillation_mask", like=student_rows)
    loss_weights = _as_vector_tensor(sdpo_loss_weights, name="sdpo_loss_weights", like=student_rows)
    if sample_mask.numel() != len(student_rows):
        raise ValueError("self_distillation_mask length must match student_log_probs rows.")
    if loss_weights.numel() != len(student_rows):
        raise ValueError("sdpo_loss_weights length must match student_log_probs rows.")
    old_rows = None
    deployment_rows = None
    if clip_ratio is not None or deployment_tis_clip is not None:
        if old_log_probs is None:
            raise ValueError("SDPO policy clipping/TIS requires old_log_probs.")
        old_rows = _as_row_tensors(old_log_probs, name="old_log_probs", like_rows=student_rows)
    if deployment_tis_clip is not None:
        if deployment_behavior_log_probs is None:
            raise ValueError("SDPO deployment TIS requires deployment_behavior_log_probs.")
        deployment_rows = _as_row_tensors(
            deployment_behavior_log_probs,
            name="deployment_behavior_log_probs",
            like_rows=student_rows,
        )
    if old_rows is not None and len(old_rows) != len(student_rows):
        raise ValueError("old_log_probs row count must match student_log_probs rows.")
    if deployment_rows is not None and len(deployment_rows) != len(student_rows):
        raise ValueError("deployment_behavior_log_probs row count must match student_log_probs rows.")

    loss_terms: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    for idx, (student, teacher, loss_mask, resp_mask) in enumerate(
        zip(student_rows, teacher_rows, loss_mask_rows, response_mask_rows, strict=True)
    ):
        if not (student.shape == teacher.shape == loss_mask.shape == resp_mask.shape):
            raise ValueError(f"SDPO loss token shape mismatch at row {idx}.")
        log_ratio = student - teacher
        correction_weight = torch.ones_like(student)
        if clip_ratio is not None:
            correction_weight = correction_weight * torch.exp((student - old_rows[idx]).detach()).clamp(
                max=float(clip_ratio)
            )
        if deployment_tis_clip is not None:
            correction_weight = correction_weight * torch.exp((old_rows[idx] - deployment_rows[idx]).detach()).clamp(
                max=float(deployment_tis_clip)
            )
        valid_mask = resp_mask * loss_mask * sample_mask[idx]
        per_token_loss = log_ratio.detach() * student * correction_weight * loss_weights[idx]
        loss_terms.append(per_token_loss * valid_mask)
        valid_masks.append(valid_mask)

    total_valid = torch.stack([mask.sum() for mask in valid_masks]).sum()
    if torch.isclose(total_valid, torch.zeros((), dtype=total_valid.dtype, device=total_valid.device)):
        return sum(
            (row.sum() * 0.0 for row in student_rows),
            torch.zeros((), dtype=student_rows[0].dtype, device=student_rows[0].device),
        )
    return torch.stack([term.sum() for term in loss_terms]).sum() / total_valid


sdpo_token_logprob_loss = compute_sdpo_token_logprob_loss


def _as_row_tensors(value: Any, *, name: str, like_rows: list[torch.Tensor] | None = None) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 1:
            rows = [value]
        elif value.ndim == 2:
            rows = [row for row in value]
        else:
            raise ValueError(f"{name} must be 1D or 2D.")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
        if _is_flat_sequence(items):
            rows = [torch.as_tensor(items, dtype=_dtype_from_like(like_rows))]
        else:
            rows = [
                item if isinstance(item, torch.Tensor) else torch.as_tensor(item, dtype=_dtype_from_like(like_rows))
                for item in items
            ]
    else:
        raise TypeError(f"{name} must be a torch.Tensor or a sequence.")

    if like_rows is not None:
        if len(rows) == len(like_rows):
            rows = [row.to(device=like.device, dtype=like.dtype) for row, like in zip(rows, like_rows, strict=True)]
        else:
            rows = [row.to(device=like_rows[0].device, dtype=like_rows[0].dtype) for row in rows]
    return [row.reshape(-1) for row in rows]


def _as_vector_tensor(value: Any, *, name: str, like: list[torch.Tensor]) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        vector = value
    else:
        vector = torch.as_tensor(value, dtype=like[0].dtype, device=like[0].device)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    return vector.to(device=like[0].device, dtype=like[0].dtype)


def _is_flat_sequence(items: list[Any]) -> bool:
    return all(not _is_row_like(item) for item in items)


def _is_row_like(item: Any) -> bool:
    if isinstance(item, torch.Tensor):
        return item.ndim > 0
    return isinstance(item, Sequence) and not isinstance(item, (str, bytes))


def _dtype_from_like(like_rows: list[torch.Tensor] | None) -> torch.dtype:
    if like_rows:
        return like_rows[0].dtype
    return torch.float32
