"""Compressed distributions for comparing SGS teacher context views."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

@torch.no_grad()
def compress_action_view_logits(
    logits: torch.Tensor,
    support_ids: torch.Tensor,
    realized_token_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compress aligned action-token logits on a caller-owned top-k support.

    The caller supplies a shared support so both context views use the same
    compressed distribution. Full-vocabulary logits remain forward-local.
    """
    if logits.ndim != 2 or support_ids.ndim != 2:
        raise ValueError("logits and support_ids must have shapes [tokens, vocabulary] and [tokens, topk].")
    if logits.size(0) != support_ids.size(0):
        raise ValueError("support_ids must contain one support row per logits row.")
    if realized_token_ids.ndim != 1 or realized_token_ids.numel() != logits.size(0):
        raise ValueError("realized_token_ids must contain one token id per logits row.")
    if support_ids.numel() and (int(support_ids.min()) < 0 or int(support_ids.max()) >= logits.size(1)):
        raise ValueError("support token id is outside the vocabulary.")
    if realized_token_ids.numel() and (
        int(realized_token_ids.min()) < 0 or int(realized_token_ids.max()) >= logits.size(1)
    ):
        raise ValueError("realized token id is outside the vocabulary.")

    log_probs = F.log_softmax(logits.float(), dim=-1)
    support = support_ids.to(device=log_probs.device, dtype=torch.long)
    realized = realized_token_ids.to(device=log_probs.device, dtype=torch.long)
    topk_log_probs = log_probs.gather(-1, support)
    # This is the same top-k + one residual bucket contract used by the SDPO
    # loss. Clamping protects round-off when the retained mass rounds to one.
    retained_log_mass = torch.logsumexp(topk_log_probs, dim=-1).clamp(max=-1e-7)
    tail_log_probs = torch.log(-torch.expm1(retained_log_mass))
    realized_log_probs = log_probs.gather(-1, realized.unsqueeze(-1)).squeeze(-1)
    return {
        "topk_log_probs": topk_log_probs.detach().cpu(),
        "tail_log_probs": tail_log_probs.detach().cpu(),
        "realized_log_probs": realized_log_probs.detach().cpu(),
    }


def _compressed_log_probs(view: dict[str, torch.Tensor]) -> torch.Tensor:
    topk = torch.as_tensor(view["topk_log_probs"], dtype=torch.float32)
    tail = torch.as_tensor(view["tail_log_probs"], dtype=torch.float32).reshape(-1, 1)
    if topk.ndim != 2 or tail.size(0) != topk.size(0):
        raise ValueError("compressed view tensors must have shapes [tokens, topk] and [tokens].")
    return torch.cat((topk, tail), dim=-1)


@torch.no_grad()
def paired_compressed_view_metrics(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compare two views on one caller-owned top-k+tail support."""
    left_logp = _compressed_log_probs(left)
    right_logp = _compressed_log_probs(right)
    if left_logp.shape != right_logp.shape:
        raise ValueError("compressed view tensors must have identical shapes")
    left_p, right_p = left_logp.exp(), right_logp.exp()
    log_mix = torch.logaddexp(left_logp, right_logp) - math.log(2.0)
    return {
        "kl_left_right": (left_p * (left_logp - right_logp)).sum(dim=-1).float().clamp_min(0.0),
        "kl_right_left": (right_p * (right_logp - left_logp)).sum(dim=-1).float().clamp_min(0.0),
        "js": (
            0.5
            * (
                (left_p * (left_logp - log_mix)).sum(dim=-1)
                + (right_p * (right_logp - log_mix)).sum(dim=-1)
            )
        )
        .float()
        .clamp(min=0.0, max=math.log(2.0)),
        "total_variation": (0.5 * (left_p - right_p).abs().sum(dim=-1)).float().clamp(0.0, 1.0),
    }
