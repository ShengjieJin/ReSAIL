"""Self-distillation policy optimization helpers."""

from .loss import compute_sdpo_token_logprob_loss, sdpo_token_logprob_loss
from .teacher_alignment import (
    align_sdpo_teacher_log_probs,
    build_sdpo_teacher_rollout_data,
    slice_sdpo_teacher_response_log_probs,
)

__all__ = [
    "align_sdpo_teacher_log_probs",
    "build_sdpo_teacher_rollout_data",
    "compute_sdpo_token_logprob_loss",
    "sdpo_token_logprob_loss",
    "slice_sdpo_teacher_response_log_probs",
]
