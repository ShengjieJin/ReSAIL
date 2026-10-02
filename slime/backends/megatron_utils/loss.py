# Modified for ReSAIL. See NOTICE and LICENSE for attribution and terms.
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from torch.utils.checkpoint import checkpoint

from slime.algorithms.sdpo.loss import compute_sdpo_token_logprob_loss as compute_sdpo_token_logprob_loss
from slime.algorithms.sdpo.loss import sdpo_token_logprob_loss as sdpo_token_logprob_loss
from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.misc import load_function
from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from slime.utils.types import RolloutBatch

from .cp_utils import (
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
)


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
    apply_temperature: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and optionally applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for optional scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.
        apply_temperature: Whether to divide outputs by `rollout_temperature`.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    qkv_format = args.qkv_format

    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    if apply_temperature and args.rollout_temperature != 1.0:
        logits = logits.div(args.rollout_temperature)

    cp_size = mpu.get_context_parallel_world_size()
    end = 0
    seq_start = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
            else:
                end += total_length
                start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            logits_local_len = logits.size(0)
            cp_rank = mpu.get_context_parallel_rank()
            chunk_start = cp_rank * logits_local_len
            chunk_end = chunk_start + logits_local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                logits_chunk = logits[0:0]
                tokens_chunk = tokens[0:0]
            else:
                logits_chunk = logits[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        seq_start += total_length

        yield logits_chunk, tokens_chunk


def _allgather_cp_redistribute(
    res: dict[str, list[torch.Tensor]],
    *,
    logits_local_len: int,
    args: Namespace,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> None:
    """Redistribute response tensors from allgather-CP layout to zigzag ring-attn layout.

    After allgather context parallelism, each rank holds a contiguous chunk of
    the global sequence.  This helper reconstructs per-sample full response
    tensors via a differentiable all-reduce and re-slices them into the zigzag
    CP pattern expected by downstream code.

    The *res* dict is modified **in-place**.

    Args:
        res: Dict mapping metric names to lists of per-sample tensors.
        logits_local_len: Local sequence length on this rank.
        args: Configuration (needs ``qkv_format``).
        total_lengths: Total sequence lengths (prompt + response) per sample.
        response_lengths: Response segment lengths per sample.
        max_seq_lens: Optional padded max sequence lengths per sample.
    """
    cp_group = mpu.get_context_parallel_group()
    cp_rank = mpu.get_context_parallel_rank()
    chunk_start = cp_rank * logits_local_len
    chunk_end = chunk_start + logits_local_len

    for key, values in res.items():
        # Skip keys where all values are None (e.g. entropy when not computed)
        if all(v is None for v in values):
            continue

        # Determine reference dtype/device from first non-None value
        ref_value = next(v for v in values if v is not None)
        ref_dtype = ref_value.dtype
        ref_device = ref_value.device

        # Reconstruct full response tensors with each rank's contiguous contribution
        full_resps = []
        seq_start = 0
        for value, total_length, response_length in zip(values, total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)

            if value is None or e <= s:
                # This rank has no response logprobs for this sample
                full_resp = torch.zeros(
                    response_length,
                    dtype=ref_dtype,
                    device=ref_device,
                    requires_grad=True,
                )
            else:
                resp_start = s - logit_global_start
                resp_end = e - logit_global_start
                full_resp = F.pad(value, (resp_start, response_length - resp_end))

            assert full_resp.size(0) == response_length, f"Expected {response_length}, got {full_resp.size(0)}"
            full_resps.append(full_resp)
            seq_start += total_length

        # Single differentiable all-reduce to gather full response from all CP ranks
        all_cat = torch.cat(full_resps, dim=0)
        all_cat = dist.nn.all_reduce(all_cat, group=cp_group)

        # Re-slice each sample into zigzag CP pattern
        new_values = []
        for idx, (full_resp, total_length, response_length) in enumerate(
            zip(all_cat.split(response_lengths, dim=0), total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
            new_values.append(
                slice_log_prob_with_cp(full_resp, total_length, response_length, args.qkv_format, max_seq_len)
            )

        res[key] = new_values


def _build_shifted_tokens(
    T: int,
    device: torch.device,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    allgather_cp: bool,
) -> torch.Tensor:
    """Build shifted target tokens for the full packed/padded logits."""
    cp_size = mpu.get_context_parallel_world_size()

    # --- zigzag CP: completely different layout ---
    if cp_size > 1 and not allgather_cp:
        full_tokens = torch.zeros(T, dtype=torch.long, device=device)
        end = 0
        for i, (tokens, total_length, response_length) in enumerate(
            zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            chunk_size_cp, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            for half, base in ((0, end), (1, end + chunk_size_cp)):
                lo = logits_offset[half][0] - chunks_offset[half][0]
                hi = logits_offset[half][1] - chunks_offset[half][0]
                full_tokens[base + lo : base + hi] = tokens[tokens_offset[half][0] : tokens_offset[half][1]]
            end += 2 * chunk_size_cp
        return full_tokens

    # --- cp1 and allgather-CP both build global shifted tokens the same way ---
    T_global = sum(total_lengths) if allgather_cp else T
    full_tokens = torch.zeros(T_global, dtype=torch.long, device=device)

    if qkv_format == "thd" or allgather_cp:
        offset = 0
        for tokens, total_length in zip(unconcat_tokens, total_lengths, strict=False):
            full_tokens[offset : offset + total_length - 1] = tokens[1:total_length]
            offset += total_length
    else:  # bshd, cp1
        for i, (tokens, total_length) in enumerate(zip(unconcat_tokens, total_lengths, strict=False)):
            seq_start = max_seq_lens[i] * i
            full_tokens[seq_start : seq_start + total_length - 1] = tokens[1:total_length]

    # allgather-CP: slice to local chunk
    if allgather_cp:
        cp_rank = mpu.get_context_parallel_rank()
        chunk_start = cp_rank * T
        chunk_end = chunk_start + T
        if chunk_end <= T_global:
            return full_tokens[chunk_start:chunk_end].contiguous()
        local = torch.zeros(T, dtype=torch.long, device=device)
        valid = T_global - chunk_start
        if valid > 0:
            local[:valid] = full_tokens[chunk_start:]
        return local

    return full_tokens


def _extract_per_sample(
    log_prob_full: torch.Tensor,
    entropy_full: torch.Tensor | None,
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    allgather_cp: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Slice per-sample response log-probs/entropy from full-length 1-D tensors."""
    cp_size = mpu.get_context_parallel_world_size()
    log_probs_list: list[torch.Tensor] = []
    entropy_list: list[torch.Tensor] = []

    if cp_size > 1 and not allgather_cp:
        # zigzag CP
        pos = 0
        for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=False)):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            chunk_size_cp, chunks_offset, logits_offset, _tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            lo0 = logits_offset[0][0] - chunks_offset[0][0]
            hi0 = logits_offset[0][1] - chunks_offset[0][0]
            lo1 = logits_offset[1][0] - chunks_offset[1][0]
            hi1 = logits_offset[1][1] - chunks_offset[1][0]

            lp = torch.cat(
                [
                    log_prob_full[pos + lo0 : pos + hi0],
                    log_prob_full[pos + chunk_size_cp + lo1 : pos + chunk_size_cp + hi1],
                ],
                dim=0,
            )
            log_probs_list.append(lp)
            if entropy_full is not None:
                ent = torch.cat(
                    [
                        entropy_full[pos + lo0 : pos + hi0],
                        entropy_full[pos + chunk_size_cp + lo1 : pos + chunk_size_cp + hi1],
                    ],
                    dim=0,
                )
                entropy_list.append(ent)
            pos += 2 * chunk_size_cp

    elif allgather_cp:
        cp_rank = mpu.get_context_parallel_rank()
        local_len = log_prob_full.size(0)
        chunk_start = cp_rank * local_len
        chunk_end = chunk_start + local_len

        seq_start = 0
        for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                log_probs_list.append(torch.zeros((0,), dtype=log_prob_full.dtype, device=log_prob_full.device))
                if entropy_full is not None:
                    entropy_list.append(torch.zeros((0,), dtype=entropy_full.dtype, device=entropy_full.device))
            else:
                log_probs_list.append(log_prob_full[s - chunk_start : e - chunk_start])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[s - chunk_start : e - chunk_start])
            seq_start += total_length

    else:
        # cp1
        if qkv_format == "thd":
            offset = 0
            for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
                end = offset + total_length
                start = end - response_length
                log_probs_list.append(log_prob_full[start - 1 : end - 1])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[start - 1 : end - 1])
                offset += total_length
        else:  # bshd
            for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=False)):
                end = max_seq_lens[i] * i + total_length
                start = end - response_length
                log_probs_list.append(log_prob_full[start - 1 : end - 1])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[start - 1 : end - 1])

    return log_probs_list, entropy_list


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    temperature: float | None = None,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    ``with_entropy=True`` uses the fused log-probability/entropy helper.
    ``with_entropy=False`` keeps the log-probability-only helper, which is
    used by SFT and SDPO paths that do not need actor entropy.
    """
    assert non_loss_data
    qkv_format = args.qkv_format

    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    # Apply sampling temperature scaling to logits to match rollout/eval policy distribution.
    effective_temperature = getattr(args, "rollout_temperature", 1.0) if temperature is None else temperature
    if effective_temperature != 1.0:
        logits = logits / effective_temperature
    logits = logits.contiguous()
    T = logits.size(0)
    device = logits.device
    tp_group = mpu.get_tensor_model_parallel_group()
    chunk_size = args.log_probs_chunk_size

    # --- build full shifted-token target tensor ---
    full_tokens = _build_shifted_tokens(
        T, device, unconcat_tokens, total_lengths, response_lengths, qkv_format, max_seq_lens, args.allgather_cp
    )

    # --- compute on full [T,V] logits at once via calculate_log_probs_and_entropy ---
    log_prob_full, entropy_full = calculate_log_probs_and_entropy(
        logits,
        full_tokens,
        tp_group,
        with_entropy=with_entropy,
        chunk_size=chunk_size,
    )
    log_prob_full = log_prob_full.squeeze(-1)  # [T, 1] -> [T]

    # --- extract per-sample response portions ---
    log_probs_list, entropy_list = _extract_per_sample(
        log_prob_full,
        entropy_full,
        total_lengths,
        response_lengths,
        qkv_format,
        max_seq_lens,
        args.allgather_cp,
    )

    res = {"log_probs": log_probs_list}
    if with_entropy:
        res["entropy"] = entropy_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits_local_len=T,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return torch.empty((0,), device=device), res


def get_log_probs_entropy_and_sdpo_student_representations(
    logits: torch.Tensor,
    *,
    hidden_states: torch.Tensor,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    temperature: float | None = None,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Collect actor log-probs and detached response representations in one forward."""
    empty, output = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=with_entropy,
        temperature=temperature,
        non_loss_data=non_loss_data,
        max_seq_lens=max_seq_lens,
    )
    rows = get_response_hidden_representations(
        hidden_states,
        args=args,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )
    output["sdpo_student_representations"] = [row.detach().to(device="cpu", dtype=torch.bfloat16) for row in rows]
    return empty, output


def get_sdpo_distillation_tensors(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    topk_indices: list[torch.Tensor] | None = None,
    include_metric_topk: bool | None = None,
    response_log_probs: list[torch.Tensor] | None = None,
    temperature: float | None = None,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Collect response-aligned tensors needed by SDPO distillation.

    With ``topk_indices=None`` this emits the student/teacher top-k support
    itself. With ``topk_indices`` it evaluates logits on that exact support so
    the teacher distribution and current-student distribution are aligned.
    """
    del with_entropy, non_loss_data, include_metric_topk
    if response_log_probs is None:
        _, base = get_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=False,
            max_seq_lens=max_seq_lens,
            temperature=temperature,
        )
    else:
        if len(response_log_probs) != len(response_lengths):
            raise ValueError("response_log_probs length must match response_lengths.")
        base = {"log_probs": response_log_probs}

    if not getattr(args, "sdpo_full_logit_distillation", True):
        return torch.empty((0,), device=logits.device), base

    if mpu.get_tensor_model_parallel_world_size() != 1:
        raise NotImplementedError(
            "SDPO top-k/full-logit distillation currently requires tensor_model_parallel_size=1. "
            "Use --sdpo-distillation-mode sample_token or --no-sdpo-full-logit-distillation for token-logprob SDPO."
        )
    if getattr(args, "allgather_cp", False) and mpu.get_context_parallel_world_size() > 1:
        raise NotImplementedError("SDPO top-k distillation is not supported with --allgather-cp yet.")

    topk = _normalize_sdpo_topk(getattr(args, "sdpo_distillation_topk", 20))
    if topk_indices is not None:
        topk_indices = _as_sdpo_index_rows(topk_indices, like_rows=base["log_probs"])

    topk_log_probs: list[torch.Tensor] = []
    new_topk_indices: list[torch.Tensor] = []
    all_log_probs: list[torch.Tensor] = []

    effective_temperature = getattr(args, "rollout_temperature", 1.0) if temperature is None else temperature
    old_temperature = getattr(args, "rollout_temperature", 1.0)
    args.rollout_temperature = effective_temperature
    try:
        response_iter = get_responses(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )
        for idx, (logits_chunk, _tokens_chunk) in enumerate(response_iter):
            log_probs = F.log_softmax(logits_chunk, dim=-1)
            if topk is None:
                all_log_probs.append(log_probs)
                continue

            k = min(topk, log_probs.size(-1))
            if topk_indices is None:
                values, indices = torch.topk(log_probs, k=k, dim=-1)
            else:
                indices = topk_indices[idx].to(device=log_probs.device, dtype=torch.long)
                values = torch.gather(log_probs, dim=-1, index=indices)
            topk_log_probs.append(values)
            new_topk_indices.append(indices)
    finally:
        args.rollout_temperature = old_temperature

    if topk is None:
        base["sdpo_all_log_probs"] = all_log_probs
    else:
        base["sdpo_topk_log_probs"] = topk_log_probs
        base["sdpo_topk_indices"] = new_topk_indices
    return torch.empty((0,), device=logits.device), base


def get_sdpo_compressed_action_view_tensors(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    response_masks: list[torch.Tensor | list[int]],
    topk_indices: list[torch.Tensor],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    temperature: float | None = None,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Extract compressed action-view distributions on a shared top-k support."""
    del with_entropy, non_loss_data, temperature
    from slime.algorithms.sdpo.sgs_metrics import compress_action_view_logits

    if len(response_masks) != len(response_lengths):
        raise ValueError("response_masks length must match response_lengths.")
    if len(topk_indices) != len(response_lengths):
        raise ValueError("topk_indices length must match response_lengths.")
    selected_topk_log_probs: list[torch.Tensor] = []
    selected_tail_log_probs: list[torch.Tensor] = []
    selected_realized_log_probs: list[torch.Tensor] = []
    selected_support_ids: list[torch.Tensor] = []
    selected_token_ids: list[torch.Tensor] = []
    for row_idx, ((logits_chunk, tokens_chunk), raw_mask, raw_support) in enumerate(
        zip(
            get_responses(
                logits,
                args=args,
                unconcat_tokens=unconcat_tokens,
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                max_seq_lens=max_seq_lens,
                apply_temperature=False,
            ),
            response_masks,
            topk_indices,
            strict=True,
        )
    ):
        mask = torch.as_tensor(raw_mask, dtype=torch.bool, device=logits_chunk.device).reshape(-1)
        if mask.numel() != logits_chunk.size(0):
            raise ValueError(
                f"response_masks[{row_idx}] length must equal response length: "
                f"{mask.numel()} != {logits_chunk.size(0)}."
            )
        support = torch.as_tensor(raw_support, dtype=torch.long, device=logits_chunk.device)
        if support.ndim != 2 or support.size(0) != logits_chunk.size(0):
            raise ValueError(f"topk_indices[{row_idx}] must have one row per response token.")
        action_logits = logits_chunk[mask]
        action_support = support[mask]
        action_tokens = tokens_chunk[mask]
        compressed = compress_action_view_logits(action_logits, action_support, action_tokens)
        selected_topk_log_probs.append(compressed["topk_log_probs"])
        selected_tail_log_probs.append(compressed["tail_log_probs"])
        selected_realized_log_probs.append(compressed["realized_log_probs"])
        selected_support_ids.append(action_support.detach().cpu())
        selected_token_ids.append(action_tokens.detach().to(device="cpu", dtype=torch.long))
    return torch.empty((0,), device=logits.device), {
        "sdpo_action_view_topk_log_probs": selected_topk_log_probs,
        "sdpo_action_view_tail_log_probs": selected_tail_log_probs,
        "sdpo_action_view_realized_log_probs": selected_realized_log_probs,
        "sdpo_action_view_support_ids": selected_support_ids,
        "sdpo_action_view_token_ids": selected_token_ids,
    }


def get_sdpo_distillation_and_compressed_action_view_tensors(
    logits: torch.Tensor,
    *,
    response_masks: list[torch.Tensor | list[int]],
    topk_indices: list[torch.Tensor] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Collect SDPO tensors and compressed action views in one forward."""
    empty, output = get_sdpo_distillation_tensors(logits, topk_indices=topk_indices, **kwargs)
    _, action_output = get_sdpo_compressed_action_view_tensors(
        logits,
        response_masks=response_masks,
        topk_indices=output["sdpo_topk_indices"],
        **kwargs,
    )
    output.update(action_output)
    return empty, output


def get_sdpo_dual_support_distillation_and_compressed_action_view_tensors(
    logits: torch.Tensor,
    *,
    response_masks: list[torch.Tensor | list[int]],
    topk_indices: list[torch.Tensor],
    secondary_topk_indices: list[torch.Tensor] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Collect selected-step distillation tensors and a second KL support from one model forward.

    ``topk_indices`` remains the exact ordinary-student support used by
    selected-step distillation.  The secondary support is either supplied by the privileged
    student (reverse retention) or selected from these logits themselves
    (forward retention).  Only the post-forward log-softmax/top-k work is
    shared; token support and both KL definitions remain unchanged.
    """
    empty, output = get_sdpo_distillation_tensors(logits, topk_indices=topk_indices, **kwargs)
    _, secondary = get_sdpo_distillation_tensors(
        logits,
        topk_indices=secondary_topk_indices,
        response_log_probs=output["log_probs"],
        **kwargs,
    )
    _, action_output = get_sdpo_compressed_action_view_tensors(
        logits,
        response_masks=response_masks,
        topk_indices=output["sdpo_topk_indices"],
        **kwargs,
    )
    output.update(action_output)
    output["sdpo_retention_topk_indices"] = secondary["sdpo_topk_indices"]
    output["sdpo_retention_topk_log_probs"] = secondary["sdpo_topk_log_probs"]
    return empty, output


def get_sdpo_representation_tensors(
    hidden_states: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    temperature: float | None = None,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Collect response-token hidden states for OPRD-Vanilla style distillation."""
    del unconcat_tokens, with_entropy, temperature, non_loss_data
    rows = get_response_hidden_representations(
        hidden_states,
        args=args,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )
    rows = [row.detach().to(device="cpu", dtype=torch.bfloat16) for row in rows]
    return torch.empty((0,), device=hidden_states.device), {"representations": rows}


def get_sdpo_representation_and_prompt_last_tensors(
    model_output: torch.Tensor,
    *,
    hidden_states: torch.Tensor,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    temperature: float | None = None,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Collect response states and the one prompt-last state needed for decision attribution."""
    del model_output, unconcat_tokens, with_entropy, temperature, non_loss_data
    rows, prompt_last_rows = get_response_and_prompt_last_hidden_representations(
        hidden_states,
        args=args,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )
    return torch.empty((0,), device=hidden_states.device), {
        "representations": [row.detach().to(device="cpu", dtype=torch.bfloat16) for row in rows],
        "prompt_last_representations": [
            row.detach().to(device="cpu", dtype=torch.bfloat16) for row in prompt_last_rows
        ],
    }


def get_sdpo_distillation_and_representation_tensors(
    logits: torch.Tensor,
    *,
    hidden_states: torch.Tensor,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    temperature: float | None = None,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    topk_indices: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Collect SDPO logit distillation tensors and response hidden states from one forward."""
    _, distillation = get_sdpo_distillation_tensors(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=with_entropy,
        temperature=temperature,
        non_loss_data=non_loss_data,
        max_seq_lens=max_seq_lens,
        topk_indices=topk_indices,
    )
    _, representations = get_sdpo_representation_tensors(
        hidden_states,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=with_entropy,
        temperature=temperature,
        non_loss_data=non_loss_data,
        max_seq_lens=max_seq_lens,
    )
    distillation.update(representations)
    return torch.empty((0,), device=logits.device), distillation


def get_response_hidden_representations(
    hidden_states: torch.Tensor,
    *,
    args: Namespace,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> list[torch.Tensor]:
    """Slice last-layer hidden states for response tokens, without next-token shift."""
    if mpu.get_tensor_model_parallel_world_size() != 1:
        raise NotImplementedError("SDPO representation distillation currently requires tensor_model_parallel_size=1.")
    if mpu.get_context_parallel_world_size() != 1:
        raise NotImplementedError("SDPO representation distillation currently requires context_parallel_size=1.")
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must be 3D, got {tuple(hidden_states.shape)}.")

    qkv_format = args.qkv_format
    rows: list[torch.Tensor] = []
    if qkv_format == "thd":
        if hidden_states.size(1) == 1:
            flat_hidden = hidden_states[:, 0, :]
        elif hidden_states.size(0) == 1:
            flat_hidden = hidden_states[0, :, :]
        else:
            raise ValueError(f"Cannot infer THD hidden layout from shape {tuple(hidden_states.shape)}.")
        offset = 0
        for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
            end = offset + total_length
            start = end - response_length
            rows.append(flat_hidden[start:end])
            offset += total_length
    elif qkv_format == "bshd":
        if max_seq_lens is None:
            raise ValueError("max_seq_lens is required for qkv_format=bshd.")
        batch_size = len(total_lengths)
        if hidden_states.size(1) == batch_size:
            batch_hidden = hidden_states.transpose(0, 1).contiguous()
        elif hidden_states.size(0) == batch_size:
            batch_hidden = hidden_states
        else:
            raise ValueError(f"Cannot infer BSHD hidden layout from shape {tuple(hidden_states.shape)}.")
        for idx, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=False)):
            end = total_length
            start = end - response_length
            rows.append(batch_hidden[idx, start:end])
    else:
        raise ValueError(f"Unsupported qkv_format: {qkv_format}")

    for idx, (row, response_length) in enumerate(zip(rows, response_lengths, strict=False)):
        if row.size(0) != response_length:
            raise ValueError(f"SDPO representation row {idx} length mismatch: {row.size(0)} != {response_length}.")
    return rows


def get_response_and_prompt_last_hidden_representations(
    hidden_states: torch.Tensor,
    *,
    args: Namespace,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Slice response states and prompt-last states directly from THD or BSHD hidden tensors."""
    rows = get_response_hidden_representations(
        hidden_states,
        args=args,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )
    prompt_last_rows: list[torch.Tensor] = []
    if args.qkv_format == "thd":
        flat_hidden = hidden_states[:, 0, :] if hidden_states.size(1) == 1 else hidden_states[0, :, :]
        offset = 0
        for total_length, response_length in zip(total_lengths, response_lengths, strict=True):
            prompt_last_index = offset + total_length - response_length - 1
            if prompt_last_index < offset:
                raise ValueError("Decision-state attribution requires at least one prompt token per row.")
            prompt_last_rows.append(flat_hidden[prompt_last_index])
            offset += total_length
    else:
        batch_size = len(total_lengths)
        batch_hidden = hidden_states.transpose(0, 1) if hidden_states.size(1) == batch_size else hidden_states
        for idx, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=True)):
            prompt_last_index = total_length - response_length - 1
            if prompt_last_index < 0:
                raise ValueError("Decision-state attribution requires at least one prompt token per row.")
            prompt_last_rows.append(batch_hidden[idx, prompt_last_index])
    return rows, prompt_last_rows


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration passed to `get_responses`; temperature scaling is
            disabled for value outputs.
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        apply_temperature=False,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits_local_len=logits.size(1),
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return torch.empty((0,), device=logits.device), res


def apply_opd_kl_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
) -> None:
    """Apply on-policy distillation KL penalty to advantages.

    Computes reverse KL (student_logp - teacher_logp) and adds weighted penalty
    to advantages in-place. This is orthogonal to the base advantage estimator.

    Args:
        args: Configuration containing `use_opd` and `opd_kl_coef`.
        rollout_data: Dict containing "teacher_log_probs".
        advantages: List of advantage tensors to modify in-place.
        student_log_probs: List of student log-probability tensors.

    References:
        https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py
    """

    if student_log_probs is None:
        return

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError(f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing.")

    device = student_log_probs[0].device
    teacher_log_probs = [t.to(device=device) for t in teacher_log_probs]

    reverse_kls = []
    for i, adv in enumerate(advantages):
        reverse_kl = student_log_probs[i] - teacher_log_probs[i]
        advantages[i] = adv - args.opd_kl_coef * reverse_kl
        reverse_kls.append(reverse_kl)

    # Store reverse KL for logging
    rollout_data["opd_reverse_kl"] = reverse_kls


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    and "reinforce_plus_plus_baseline". When `args.normalize_advantages` is
    True, advantages are whitened across the data-parallel group using masked
    statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    If ``args.custom_advantage_function_path`` is set, it is called after KL computation
    and must populate ``rollout_data["advantages"]`` and
    ``rollout_data["returns"]``.

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    rollout_log_probs: list[torch.Tensor] | None = rollout_data.get("rollout_log_probs")
    log_probs: list[torch.Tensor] | None = (
        rollout_log_probs if args.use_rollout_logprobs else rollout_data.get("log_probs")
    )
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # return when not the last pp stage.
    if not mpu.is_pipeline_last_stage():
        return

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs or rollout_log_probs or values or loss_masks
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]
    rollout_data["kl"] = kl

    if args.custom_advantage_function_path is not None:
        custom_adv_fn = load_function(args.custom_advantage_function_path)
        custom_adv_fn(args, rollout_data)
        advantages, returns = rollout_data["advantages"], rollout_data["returns"]

    elif args.advantage_estimator in ["grpo", "gspo"]:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # TODO: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "ppo":
        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = mpu.get_context_parallel_rank()
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, rewards, args.gamma, args.lambd
        )

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    # Apply on-policy distillation KL penalty to advantages (orthogonal to advantage estimator)
    if args.use_opd:
        apply_opd_kl_to_advantages(
            args=args,
            rollout_data=rollout_data,
            advantages=advantages,
            student_log_probs=log_probs,
        )

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                    total_len, response_len, args.qkv_format, max_seq_len
                )

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
            dp_group = mpu.get_data_parallel_group()

            whitened_advs_flat = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=dp_group,
                shift_mean=True,
            )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def _apply_grpo_policy_token_weights(
    args: Namespace,
    batch: RolloutBatch,
    pg_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not getattr(args, "grpo_token_weights", False):
        return pg_loss, pg_loss

    rows = batch.get("grpo_policy_token_weights")
    if not isinstance(rows, list) or not rows:
        raise KeyError("--grpo-token-weights requires batch key 'grpo_policy_token_weights'.")
    response_lengths = batch.get("response_lengths")
    if not isinstance(response_lengths, list) or len(rows) != len(response_lengths):
        raise ValueError("GRPO policy token weight rows must align with response_lengths.")
    for row_idx, (row, response_length) in enumerate(zip(rows, response_lengths, strict=True)):
        if torch.as_tensor(row).numel() != int(response_length):
            raise ValueError(
                f"GRPO policy token weight row {row_idx} length mismatch: "
                f"{torch.as_tensor(row).numel()} != {int(response_length)}."
            )
    weights = torch.cat(
        [torch.as_tensor(row, device=pg_loss.device, dtype=pg_loss.dtype).reshape(-1) for row in rows],
        dim=0,
    )
    if weights.numel() != pg_loss.numel():
        raise ValueError(
            "GRPO policy token weight token count mismatch: "
            f"{weights.numel()} weights for {pg_loss.numel()} policy-loss tokens."
        )
    if not torch.isfinite(weights).all():
        raise ValueError("GRPO policy token weights must be finite.")
    if (weights < 0).any():
        raise ValueError("GRPO policy token weights must be non-negative.")
    return pg_loss * weights, pg_loss


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch.get("log_probs")

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    if not args.use_rollout_logprobs and not old_log_probs:
        old_log_probs = [log_prob.detach() for log_prob in log_probs]
    train_log_probs_for_tis = batch.get("log_probs")
    if not train_log_probs_for_tis:
        train_log_probs_for_tis = [log_prob.detach() for log_prob in log_probs]

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": train_log_probs_for_tis,
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # [decouple IS and rejection] Rebuild sum_of_sample_mean with
        # modified_response_masks for numerator correction (rejected tokens
        # zeroed in pg_loss). Denominators stay the precomputed per-rollout
        # totals from ``rollout_mask_sums`` (based on original loss_masks) —
        # same normalizer as the outer reducer, so pg_loss and the rest of the
        # reported metrics live in the same per-rollout-mean space.
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            batch["rollout_mask_sums"],
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    weighted_pg_loss, unweighted_pg_loss = _apply_grpo_policy_token_weights(args, batch, pg_loss)
    pg_loss = pg_loss_reducer(weighted_pg_loss)
    grpo_unweighted_pg_loss = None
    if getattr(args, "grpo_token_weights", False):
        grpo_unweighted_pg_loss = pg_loss_reducer(unweighted_pg_loss.detach())
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # entropy loss
    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_logprob_abs_diff = sum_of_sample_mean((old_log_probs - rollout_log_probs).abs())

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()

    if grpo_unweighted_pg_loss is not None:
        reported_loss["grpo_unweighted_pg_loss"] = grpo_unweighted_pg_loss.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    # Add OPD metrics if available
    if "opd_reverse_kl" in batch:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    _, values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def sdpo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute SDPO distillation loss from aligned teacher signals."""
    del sum_of_sample_mean
    if getattr(args, "sdpo_distillation_mode", None) == "representation":
        return _sdpo_representation_loss_function(args, batch, logits)

    required_keys = ("sdpo_teacher_log_probs", "self_distillation_mask", "sdpo_loss_weights")
    for key in required_keys:
        if key not in batch:
            raise KeyError(f"SDPO loss requires batch key {key!r}.")

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=max_seq_lens,
    )
    log_probs = log_probs_and_entropy["log_probs"]
    teacher_log_probs = _as_sdpo_response_rows(
        batch["sdpo_teacher_log_probs"],
        name="sdpo_teacher_log_probs",
        like_rows=log_probs,
    )
    self_distillation_mask = _as_sdpo_vector(
        batch["self_distillation_mask"],
        name="self_distillation_mask",
        like_rows=log_probs,
    )
    sdpo_loss_weights = _as_sdpo_vector(
        batch["sdpo_loss_weights"],
        name="sdpo_loss_weights",
        like_rows=log_probs,
    )
    retention_components = None
    retention_base_weights = None
    retention_normalization_scales = None
    raw_retention_components = batch.get("pr_component")
    raw_retention_base_weights = batch.get("pr_base_weight")
    if (raw_retention_components is None) != (raw_retention_base_weights is None):
        raise KeyError("PR component rows and base loss weights must either both be present or both be absent")
    if raw_retention_components is not None:
        retention_components = _as_sdpo_vector(
            raw_retention_components,
            name="pr_component",
            like_rows=log_probs,
        )
        retention_base_weights = _as_sdpo_vector(
            raw_retention_base_weights,
            name="pr_base_weight",
            like_rows=log_probs,
        )
        raw_retention_normalization_scales = batch.get("pr_normalization_scale")
        if raw_retention_normalization_scales is None:
            retention_normalization_scales = torch.full_like(retention_components, 2.0)
        else:
            retention_normalization_scales = _as_sdpo_vector(
                raw_retention_normalization_scales,
                name="pr_normalization_scale",
                like_rows=log_probs,
            )
    sdpo_token_weights = _as_optional_sdpo_response_rows(
        batch.get("sdpo_token_weights"),
        name="sdpo_token_weights",
        like_rows=log_probs,
        device=logits.device,
    )
    if self_distillation_mask.numel() != len(log_probs):
        raise ValueError("self_distillation_mask length must match student_log_probs rows.")
    if sdpo_loss_weights.numel() != len(log_probs):
        raise ValueError("sdpo_loss_weights length must match student_log_probs rows.")
    if retention_components is not None:
        if (
            retention_components.numel() != len(log_probs)
            or retention_base_weights.numel() != len(log_probs)
            or retention_normalization_scales.numel() != len(log_probs)
        ):
            raise ValueError("PR component/base-weight rows must match student rows")
        if not torch.all((retention_components == 0) | (retention_components == 1)):
            raise ValueError("PR component mask must contain only zero or one")
        if not torch.all(torch.isfinite(retention_normalization_scales) & (retention_normalization_scales > 0)):
            raise ValueError("PR normalization scales must be finite and positive")

    full_logit_distillation = bool(getattr(args, "sdpo_full_logit_distillation", True))
    distillation_topk = _normalize_sdpo_topk(getattr(args, "sdpo_distillation_topk", 20))
    topk_student_log_probs = None
    topk_teacher_log_probs = None
    all_student_log_probs = None
    all_teacher_log_probs = None
    if full_logit_distillation:
        if distillation_topk is None:
            if "sdpo_teacher_all_log_probs" not in batch:
                raise KeyError("SDPO dense full-vocab loss requires batch key 'sdpo_teacher_all_log_probs'.")
        else:
            for key in ("sdpo_topk_indices", "sdpo_teacher_topk_log_probs"):
                if key not in batch:
                    raise KeyError(f"SDPO top-k loss requires batch key {key!r}.")
        _, sdpo_student = get_sdpo_distillation_tensors(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=False,
            max_seq_lens=max_seq_lens,
            topk_indices=batch.get("sdpo_topk_indices"),
            response_log_probs=log_probs,
        )
        if distillation_topk is None:
            all_student_log_probs = _as_sdpo_matrix_rows(
                sdpo_student["sdpo_all_log_probs"],
                name="sdpo_all_log_probs",
                like_rows=log_probs,
            )
            all_teacher_log_probs = _as_sdpo_matrix_rows(
                batch["sdpo_teacher_all_log_probs"],
                name="sdpo_teacher_all_log_probs",
                like_rows=log_probs,
            )
        else:
            topk_student_log_probs = _as_sdpo_matrix_rows(
                sdpo_student["sdpo_topk_log_probs"],
                name="sdpo_topk_log_probs",
                like_rows=log_probs,
            )
            topk_teacher_log_probs = _as_sdpo_matrix_rows(
                batch["sdpo_teacher_topk_log_probs"],
                name="sdpo_teacher_topk_log_probs",
                like_rows=log_probs,
            )

    clip_ratio = getattr(args, "sdpo_clip_ratio", None)
    deployment_tis_clip = getattr(args, "sdpo_deployment_tis_clip", None)
    old_log_probs = None
    deployment_log_probs = None
    if clip_ratio is not None or deployment_tis_clip is not None:
        if "log_probs" not in batch or not batch["log_probs"]:
            raise KeyError("SDPO policy clipping/TIS requires batch key 'log_probs' from the frozen old actor.")
        old_log_probs = _as_sdpo_response_rows(batch["log_probs"], name="log_probs", like_rows=log_probs)
    if deployment_tis_clip is not None:
        if "rollout_log_probs" not in batch or not batch["rollout_log_probs"]:
            raise KeyError("SDPO deployment TIS requires batch key 'rollout_log_probs'.")
        deployment_log_probs = _as_sdpo_response_rows(
            batch["rollout_log_probs"], name="rollout_log_probs", like_rows=log_probs
        )
    active_loss_masks: list[torch.Tensor] = []
    per_token_losses: list[torch.Tensor] = []
    retention_base_token_losses: list[torch.Tensor] = []
    policy_ratio_rows: list[torch.Tensor] = []
    deployment_ratio_rows: list[torch.Tensor] = []
    combined_weight_rows: list[torch.Tensor] = []
    for idx, (student, teacher) in enumerate(zip(log_probs, teacher_log_probs, strict=True)):
        if student.shape != teacher.shape:
            raise ValueError(f"SDPO loss token shape mismatch at row {idx}.")
        if full_logit_distillation:
            if distillation_topk is None:
                per_token_loss = _compute_sdpo_kl_loss(
                    all_student_log_probs[idx],
                    all_teacher_log_probs[idx],
                    alpha=float(getattr(args, "sdpo_alpha", 1.0)),
                )
            else:
                student_distill = topk_student_log_probs[idx]
                teacher_distill = topk_teacher_log_probs[idx]
                if getattr(args, "sdpo_distillation_add_tail", True):
                    student_distill = _add_sdpo_tail_bucket(student_distill)
                    teacher_distill = _add_sdpo_tail_bucket(teacher_distill)
                else:
                    student_distill = _renorm_sdpo_log_probs(student_distill)
                    teacher_distill = _renorm_sdpo_log_probs(teacher_distill)
                alpha = float(getattr(args, "sdpo_alpha", 1.0))
                if (
                    retention_components is not None
                    and float(retention_components[idx].item()) == 1.0
                    and str(getattr(args, "pr_kl_direction", "reverse")) == "forward"
                ):
                    alpha = 0.0
                per_token_loss = _compute_sdpo_kl_loss(
                    student_distill,
                    teacher_distill,
                    alpha=alpha,
                )
        else:
            log_ratio = student - teacher
            per_token_loss = log_ratio.detach() * student

        active_loss_masks.append(batch["loss_masks"][idx] * self_distillation_mask[idx])
        if retention_components is not None:
            retention_base_token_losses.append(
                per_token_loss * retention_base_weights[idx] * sdpo_token_weights[idx]
            )
        if clip_ratio is None and deployment_tis_clip is None:
            # The paired no-ratio arm is intentionally a literal no-correction
            # path: no ratio tensors, no unity correction multiply, no metrics.
            per_token_losses.append(
                per_token_loss * sdpo_loss_weights[idx] * sdpo_token_weights[idx]
            )
        else:
            correction_weight = torch.ones_like(student)
            if clip_ratio is not None:
                policy_ratio = torch.exp((student - old_log_probs[idx]).detach())
                policy_ratio_rows.append(policy_ratio)
                correction_weight = correction_weight * torch.clamp(policy_ratio, max=float(clip_ratio))
            if deployment_tis_clip is not None:
                deployment_ratio = torch.exp((old_log_probs[idx] - deployment_log_probs[idx]).detach())
                deployment_ratio_rows.append(deployment_ratio)
                correction_weight = correction_weight * torch.clamp(deployment_ratio, max=float(deployment_tis_clip))
            combined_weight_rows.append(correction_weight)
            per_token_losses.append(
                per_token_loss
                * correction_weight
                * sdpo_loss_weights[idx]
                * sdpo_token_weights[idx]
            )

    loss_agg_mode = _normalize_sdpo_loss_agg_mode(getattr(args, "sdpo_loss_agg_mode", "turn_mean"))
    step_equal = loss_agg_mode == "step_equal"
    # step_equal uses the per-microbatch sum of response-row means. loss_function
    # returns the active-row count as Megatron's normalizer so microbatches,
    # DP ranks, and CP ranks compose into one equal-source-step mean.
    active_reducer = get_sum_of_sample_mean(
        total_lengths,
        response_lengths,
        active_loss_masks,
        None if step_equal else batch.get("rollout_mask_sums"),
        loss_agg_mode == "token_mean",
        args.qkv_format,
        max_seq_lens,
    )
    loss = active_reducer(torch.cat(per_token_losses, dim=0))
    loss = loss + 0 * logits.sum()

    active_sample_count = self_distillation_mask.to(dtype=torch.float32).sum()
    active_token_count = torch.stack(
        [(mask.to(device=logits.device, dtype=torch.float32)).sum() for mask in active_loss_masks]
    ).sum()
    metrics = {
        "loss": loss.clone().detach(),
        "sdpo_loss": loss.clone().detach(),
        "sdpo_active_samples": active_sample_count.clone().detach(),
        "sdpo_active_tokens": active_token_count.clone().detach(),
    }
    if retention_components is not None:
        component_rows = [
            torch.full_like(row, retention_components[idx])
            for idx, row in enumerate(retention_base_token_losses)
        ]
        base_rows = torch.cat(retention_base_token_losses, dim=0)
        component_mask = torch.cat(component_rows, dim=0)
        # The joint batch has S selected ordinary rows and R configured
        # privileged-retention rows. Per-component scales compensate for the
        # shared S+R active-row normalizer and recover each original mean.
        normalization_rows = torch.cat(
            [
                torch.full_like(row, retention_normalization_scales[idx])
                for idx, row in enumerate(retention_base_token_losses)
            ],
            dim=0,
        )
        selection_loss = active_reducer(normalization_rows * base_rows * (1.0 - component_mask))
        retention_loss = active_reducer(normalization_rows * base_rows * component_mask)
        retention_weight = float(getattr(args, "pr_weight", 0.0) or 0.0)
        metrics.update(
            {
                "sgs_loss": selection_loss.clone().detach(),
                "pr_loss": retention_loss.clone().detach(),
                "pr_weighted_loss": (retention_weight * retention_loss).clone().detach(),
            }
        )
    if step_equal:
        active_steps = _sdpo_active_prefix_count_from_masks(active_loss_masks).detach()
        metric_name = (
            "sdpo_active_prefixes"
            if str(getattr(args, "sdpo_loss_agg_mode", "")) == "oel_prefix_mean"
            else "sdpo_active_steps"
        )
        metrics[metric_name] = active_steps
    if policy_ratio_rows:
        metrics.update(
            _sdpo_ratio_metrics("sdpo_policy_ratio", policy_ratio_rows, active_loss_masks, cap=float(clip_ratio))
        )
    if deployment_ratio_rows:
        metrics.update(
            _sdpo_ratio_metrics(
                "sdpo_deployment_tis_ratio",
                deployment_ratio_rows,
                active_loss_masks,
                cap=float(deployment_tis_clip),
            )
        )
    if policy_ratio_rows or deployment_ratio_rows:
        metrics.update(_sdpo_weight_metrics("sdpo_combined_correction", combined_weight_rows, active_loss_masks))
    return loss, metrics


def _sdpo_representation_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    required_keys = ("sdpo_teacher_representations", "self_distillation_mask", "sdpo_loss_weights")
    for key in required_keys:
        if key not in batch:
            raise KeyError(f"SDPO representation loss requires batch key {key!r}.")

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    student_representations = get_response_hidden_representations(
        hidden_states,
        args=args,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )
    teacher_representations = _as_sdpo_representation_rows(
        batch["sdpo_teacher_representations"],
        name="sdpo_teacher_representations",
        like_rows=student_representations,
    )
    self_distillation_mask = _as_sdpo_vector(
        batch["self_distillation_mask"],
        name="self_distillation_mask",
        like_rows=student_representations,
    )
    sdpo_loss_weights = _as_sdpo_vector(
        batch["sdpo_loss_weights"],
        name="sdpo_loss_weights",
        like_rows=student_representations,
    )
    sdpo_token_weights = _as_optional_sdpo_response_rows(
        batch.get("sdpo_token_weights"),
        name="sdpo_token_weights",
        like_rows=student_representations,
        device=hidden_states.device,
    )
    if self_distillation_mask.numel() != len(student_representations):
        raise ValueError("self_distillation_mask length must match student representation rows.")
    if sdpo_loss_weights.numel() != len(student_representations):
        raise ValueError("sdpo_loss_weights length must match student representation rows.")

    active_loss_masks: list[torch.Tensor] = []
    per_token_losses: list[torch.Tensor] = []
    per_token_mses: list[torch.Tensor] = []
    per_token_mean_mses: list[torch.Tensor] = []
    per_token_sum_mses: list[torch.Tensor] = []
    per_token_cosines: list[torch.Tensor] = []
    representation_reduction = getattr(args, "sdpo_representation_reduction", "mean")
    if representation_reduction not in {"mean", "sum"}:
        raise ValueError("sdpo_representation_reduction must be mean or sum.")
    for idx, (student, teacher) in enumerate(zip(student_representations, teacher_representations, strict=True)):
        if student.shape != teacher.shape:
            raise ValueError(f"SDPO representation shape mismatch at row {idx}: {student.shape} != {teacher.shape}.")
        mean_mse, sum_mse, cosine = _normalized_representation_mse_and_cosine(student, teacher)
        mse = mean_mse if representation_reduction == "mean" else sum_mse
        active_loss_masks.append(batch["loss_masks"][idx] * self_distillation_mask[idx])
        per_token_losses.append(mse * sdpo_loss_weights[idx] * sdpo_token_weights[idx])
        per_token_mses.append(mse)
        per_token_mean_mses.append(mean_mse)
        per_token_sum_mses.append(sum_mse)
        per_token_cosines.append(cosine)

    loss_agg_mode = _normalize_sdpo_loss_agg_mode(getattr(args, "sdpo_loss_agg_mode", "turn_mean"))
    step_equal = loss_agg_mode == "step_equal"
    active_reducer = get_sum_of_sample_mean(
        total_lengths,
        response_lengths,
        active_loss_masks,
        None if step_equal else batch.get("rollout_mask_sums"),
        loss_agg_mode == "token_mean",
        args.qkv_format,
        max_seq_lens,
    )
    representation_loss = active_reducer(torch.cat(per_token_losses, dim=0))
    representation_mse = active_reducer(torch.cat(per_token_mses, dim=0))
    representation_mse_mean = active_reducer(torch.cat(per_token_mean_mses, dim=0))
    representation_mse_sum = active_reducer(torch.cat(per_token_sum_mses, dim=0))
    representation_cosine = active_reducer(torch.cat(per_token_cosines, dim=0))
    representation_coef = float(getattr(args, "sdpo_representation_coef", 1.0))
    loss = representation_loss * representation_coef
    loss = loss + 0 * hidden_states.sum()

    active_sample_count = self_distillation_mask.to(dtype=torch.float32).sum()
    active_token_count = torch.stack(
        [mask.to(device=hidden_states.device, dtype=torch.float32).sum() for mask in active_loss_masks]
    ).sum()
    return (
        loss,
        {
            "loss": loss.clone().detach(),
            "sdpo_loss": loss.clone().detach(),
            "sdpo_representation_loss": representation_loss.clone().detach(),
            "sdpo_representation_weighted_loss": loss.clone().detach(),
            "sdpo_representation_mse": representation_mse.clone().detach(),
            "sdpo_representation_mse_mean": representation_mse_mean.clone().detach(),
            "sdpo_representation_mse_sum": representation_mse_sum.clone().detach(),
            "sdpo_representation_cosine": representation_cosine.clone().detach(),
            "sdpo_active_samples": active_sample_count.clone().detach(),
            "sdpo_active_tokens": active_token_count.clone().detach(),
            **(
                {
                    (
                        "sdpo_active_prefixes"
                        if str(getattr(args, "sdpo_loss_agg_mode", "")) == "oel_prefix_mean"
                        else "sdpo_active_steps"
                    ): _sdpo_active_prefix_count_from_masks(active_loss_masks).detach()
                }
                if step_equal
                else {}
            ),
        },
    )


def _normalized_representation_mse_and_cosine(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student_norm = F.normalize(student.float(), p=2, dim=-1)
    teacher_norm = F.normalize(teacher.detach().float(), p=2, dim=-1)
    squared_distance = (student_norm - teacher_norm) ** 2
    mean_mse = squared_distance.mean(dim=-1)
    sum_mse = squared_distance.sum(dim=-1)
    cosine = (student_norm * teacher_norm).sum(dim=-1)
    return mean_mse, sum_mse, cosine


def _as_sdpo_response_rows(value: Any, *, name: str, like_rows: list[torch.Tensor]) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 1 and len(like_rows) == 1:
            rows = [value]
        elif value.ndim == 2:
            rows = [row for row in value]
        else:
            raise ValueError(f"{name} must be a 1D tensor for one row or a 2D tensor.")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = [row if isinstance(row, torch.Tensor) else torch.as_tensor(row) for row in value]
    else:
        raise TypeError(f"{name} must be a torch.Tensor or a sequence.")

    if len(rows) != len(like_rows):
        raise ValueError(f"{name} row count must match student_log_probs rows.")
    return [
        row.reshape(-1).to(device=like.device, dtype=like.dtype) for row, like in zip(rows, like_rows, strict=True)
    ]


def _as_optional_sdpo_response_rows(
    value: Any,
    *,
    name: str,
    like_rows: list[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    if value is None:
        return [torch.ones(row.size(0), device=device, dtype=row.dtype) for row in like_rows]
    rows = _as_sdpo_response_rows(value, name=name, like_rows=like_rows)
    ans = []
    for idx, (row, like) in enumerate(zip(rows, like_rows, strict=True)):
        if row.ndim != 1 or row.numel() != like.size(0):
            raise ValueError(f"{name} row {idx} length mismatch: {tuple(row.shape)} != ({like.size(0)},).")
        ans.append(row.to(device=like.device, dtype=like.dtype))
    return ans


def _as_sdpo_representation_rows(value: Any, *, name: str, like_rows: list[torch.Tensor]) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 2 and len(like_rows) == 1:
            rows = [value]
        elif value.ndim == 3:
            rows = [row for row in value]
        else:
            raise ValueError(f"{name} must be a 2D tensor for one row or a 3D tensor.")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = [row if isinstance(row, torch.Tensor) else torch.as_tensor(row) for row in value]
    else:
        raise TypeError(f"{name} must be a torch.Tensor or a sequence.")

    if len(rows) != len(like_rows):
        raise ValueError(f"{name} row count must match student representation rows.")
    ans = []
    for row, like in zip(rows, like_rows, strict=True):
        if row.ndim != 2:
            raise ValueError(f"{name} rows must be 2D [response, hidden].")
        if row.shape != like.shape:
            raise ValueError(f"{name} row shape mismatch: {tuple(row.shape)} != {tuple(like.shape)}.")
        ans.append(row.to(device=like.device, dtype=like.dtype))
    return ans


def _as_sdpo_vector(value: Any, *, name: str, like_rows: list[torch.Tensor]) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        vector = value
    else:
        vector = torch.as_tensor(value, dtype=like_rows[0].dtype, device=like_rows[0].device)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    return vector.to(device=like_rows[0].device, dtype=like_rows[0].dtype)


def _as_sdpo_matrix_rows(value: Any, *, name: str, like_rows: list[torch.Tensor]) -> list[torch.Tensor]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of tensors.")
    rows = [row if isinstance(row, torch.Tensor) else torch.as_tensor(row) for row in value]
    if len(rows) != len(like_rows):
        raise ValueError(f"{name} row count must match student_log_probs rows.")
    ans = []
    for row, like in zip(rows, like_rows, strict=True):
        if row.ndim != 2:
            raise ValueError(f"{name} rows must be 2D [response, support].")
        if row.size(0) != like.numel():
            raise ValueError(f"{name} response length mismatch.")
        ans.append(row.to(device=like.device, dtype=like.dtype))
    return ans


def _as_sdpo_index_rows(value: Any, *, like_rows: list[torch.Tensor]) -> list[torch.Tensor]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("sdpo_topk_indices must be a sequence of tensors.")
    rows = [row if isinstance(row, torch.Tensor) else torch.as_tensor(row) for row in value]
    if len(rows) != len(like_rows):
        raise ValueError("sdpo_topk_indices row count must match student_log_probs rows.")
    ans = []
    for row, like in zip(rows, like_rows, strict=True):
        if row.ndim != 2:
            raise ValueError("sdpo_topk_indices rows must be 2D [response, topk].")
        if row.size(0) != like.numel():
            raise ValueError("sdpo_topk_indices response length mismatch.")
        ans.append(row.to(device=like.device, dtype=torch.long))
    return ans


def _normalize_sdpo_topk(topk: int | None) -> int | None:
    if topk is None or int(topk) < 0:
        return None
    if int(topk) <= 0:
        raise ValueError("sdpo_distillation_topk must be positive, or -1 for dense full-vocab distillation.")
    return int(topk)


def _add_sdpo_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def _renorm_sdpo_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


def _compute_sdpo_kl_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("SDPO distillation tensor shape mismatch.")
    if alpha == 0.0:
        kl_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif alpha == 1.0:
        kl_loss = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        alpha_t = torch.tensor(alpha, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_log_probs + torch.log1p(-alpha_t),
                    teacher_log_probs + torch.log(alpha_t),
                ]
            ),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        kl_loss = torch.lerp(kl_student, kl_teacher, alpha_t)
    return kl_loss.sum(dim=-1)


def compute_sdpo_topk_token_kl(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    alpha: float,
    add_tail: bool,
) -> torch.Tensor:
    """Compute response-aligned SDPO KL values on a shared top-k support."""
    if add_tail:
        student_log_probs = _add_sdpo_tail_bucket(student_log_probs)
        teacher_log_probs = _add_sdpo_tail_bucket(teacher_log_probs)
    else:
        student_log_probs = _renorm_sdpo_log_probs(student_log_probs)
        teacher_log_probs = _renorm_sdpo_log_probs(teacher_log_probs)
    return _compute_sdpo_kl_loss(student_log_probs, teacher_log_probs, alpha=alpha)


def _normalize_sdpo_loss_agg_mode(mode: str) -> str:
    mode = str(mode).replace("-", "_")
    if mode == "oel_prefix_mean":
        mode = "step_equal"
    if mode not in {"token_mean", "turn_mean", "step_equal"}:
        raise ValueError("sdpo_loss_agg_mode must be one of: token_mean, turn_mean, step_equal")
    return mode


def _sdpo_ratio_metrics(
    prefix: str,
    rows: list[torch.Tensor],
    masks: list[torch.Tensor],
    *,
    cap: float,
) -> dict[str, torch.Tensor]:
    # The train-step reducer sums microbatch contributions and divides by the
    # step global batch size. Return sums of per-sample statistics so dynamic
    # microbatching cannot scale telemetry by num_microbatches / batch_size.
    active_rows = [
        row[mask.to(device=row.device, dtype=torch.bool)].to(dtype=torch.float32)
        for row, mask in zip(rows, masks, strict=True)
    ]
    active_rows = [row for row in active_rows if row.numel()]
    if not active_rows:
        # Dynamic batching may produce a padding-only local microbatch even
        # when the global training step has active SDPO tokens. Metrics are
        # sample sums, so the correct local contribution is zero.
        zero = torch.zeros((), device=rows[0].device, dtype=torch.float32)
        return {
            f"{prefix}_mean": zero,
            f"{prefix}_p50": zero,
            f"{prefix}_p90": zero,
            f"{prefix}_p95": zero,
            f"{prefix}_p99": zero,
            f"{prefix}_max": zero,
            f"{prefix}_clip_fraction": zero,
            f"{prefix}_ess_fraction": zero,
        }
    quantile_levels = torch.tensor([0.5, 0.9, 0.95, 0.99], device=active_rows[0].device)
    quantiles = torch.stack([torch.quantile(row, quantile_levels) for row in active_rows]).sum(dim=0)
    clipped_rows = [torch.clamp(row, max=cap) for row in active_rows]
    return {
        f"{prefix}_mean": torch.stack([row.mean() for row in active_rows]).sum().detach(),
        f"{prefix}_p50": quantiles[0].detach(),
        f"{prefix}_p90": quantiles[1].detach(),
        f"{prefix}_p95": quantiles[2].detach(),
        f"{prefix}_p99": quantiles[3].detach(),
        f"{prefix}_max": torch.stack([row.max() for row in active_rows]).sum().detach(),
        f"{prefix}_clip_fraction": torch.stack([(row > cap).to(dtype=torch.float32).mean() for row in active_rows])
        .sum()
        .detach(),
        f"{prefix}_ess_fraction": torch.stack(
            [row.sum().square() / (row.numel() * row.square().sum().clamp_min(1e-12)) for row in clipped_rows]
        )
        .sum()
        .detach(),
    }


def _sdpo_weight_metrics(
    prefix: str,
    rows: list[torch.Tensor],
    masks: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    active_rows = [
        row[mask.to(device=row.device, dtype=torch.bool)].to(dtype=torch.float32)
        for row, mask in zip(rows, masks, strict=True)
    ]
    active_rows = [row for row in active_rows if row.numel()]
    if not active_rows:
        # See _sdpo_ratio_metrics: an empty local shard contributes zero to
        # the train-step sum and is not a malformed global SDPO batch.
        zero = torch.zeros((), device=rows[0].device, dtype=torch.float32)
        return {
            f"{prefix}_mean": zero,
            f"{prefix}_p50": zero,
            f"{prefix}_p90": zero,
            f"{prefix}_p95": zero,
            f"{prefix}_p99": zero,
            f"{prefix}_max": zero,
            f"{prefix}_ess_fraction": zero,
        }
    quantile_levels = torch.tensor([0.5, 0.9, 0.95, 0.99], device=active_rows[0].device)
    quantiles = torch.stack([torch.quantile(row, quantile_levels) for row in active_rows]).sum(dim=0)
    return {
        f"{prefix}_mean": torch.stack([row.mean() for row in active_rows]).sum().detach(),
        f"{prefix}_p50": quantiles[0].detach(),
        f"{prefix}_p90": quantiles[1].detach(),
        f"{prefix}_p95": quantiles[2].detach(),
        f"{prefix}_p99": quantiles[3].detach(),
        f"{prefix}_max": torch.stack([row.max() for row in active_rows]).sum().detach(),
        f"{prefix}_ess_fraction": torch.stack(
            [row.sum().square() / (row.numel() * row.square().sum().clamp_min(1e-12)) for row in active_rows]
        )
        .sum()
        .detach(),
    }


def _sdpo_active_token_count(batch: RolloutBatch) -> torch.Tensor:
    required_keys = ("self_distillation_mask", "loss_masks")
    for key in required_keys:
        if key not in batch:
            raise KeyError(f"SDPO token normalizer requires batch key {key!r}.")
    loss_masks = batch["loss_masks"]
    device = loss_masks[0].device
    dtype = torch.float32
    self_distillation_mask = batch["self_distillation_mask"]
    sample_mask = (
        self_distillation_mask.to(device=device, dtype=dtype)
        if isinstance(self_distillation_mask, torch.Tensor)
        else torch.as_tensor(self_distillation_mask, device=device, dtype=dtype)
    )
    if sample_mask.numel() != len(loss_masks):
        raise ValueError("self_distillation_mask length must match loss_masks rows.")
    return torch.stack(
        [loss_mask.to(dtype=dtype).sum() * sample_mask[idx] for idx, loss_mask in enumerate(loss_masks)]
    ).sum()


def _sdpo_active_prefix_count_from_masks(loss_masks: list[torch.Tensor]) -> torch.Tensor:
    """Count response rows with at least one active loss token."""
    if not loss_masks:
        return torch.tensor(0, dtype=torch.long)
    return torch.stack(
        # Megatron's per-token schedule accumulates this normalizer into an
        # integer ``total_num_tokens``.  Keep it integral here; using float32
        # makes the schedule fail before the first optimizer update.
        [(loss_mask.to(dtype=torch.float32).sum() > 0).to(dtype=torch.long) for loss_mask in loss_masks]
    ).sum()


def _sdpo_active_prefix_count(batch: RolloutBatch) -> torch.Tensor:
    required_keys = ("self_distillation_mask", "loss_masks")
    for key in required_keys:
        if key not in batch:
            raise KeyError(f"SDPO prefix normalizer requires batch key {key!r}.")
    loss_masks = batch["loss_masks"]
    if not loss_masks:
        return torch.tensor(0.0)
    device = loss_masks[0].device
    sample_mask = batch["self_distillation_mask"]
    sample_mask = (
        sample_mask.to(device=device)
        if isinstance(sample_mask, torch.Tensor)
        else torch.as_tensor(sample_mask, device=device)
    )
    if sample_mask.numel() != len(loss_masks):
        raise ValueError("self_distillation_mask length must match loss_masks rows.")
    return _sdpo_active_prefix_count_from_masks(
        [loss_mask * sample_mask[idx] for idx, loss_mask in enumerate(loss_masks)]
    )


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    step_global_batch_size: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", "sdpo_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        step_global_batch_size: Sample count for the current training step
            (total across DP). Replaces the legacy ``args.global_batch_size``
            fallback so the train side stops depending on "every DP rank holds
            the same N samples".
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is the active token/prefix count (scalar tensor) when
          `args.calculate_per_token_loss` is True or SDPO uses token/prefix
          aggregation; otherwise it is `1` (int). OEL prefix aggregation
          requires the per-token schedule so this count is global across the
          optimizer step rather than divided independently per microbatch.
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    sdpo_loss_selected = getattr(args, "loss_type", None) == "sdpo_loss"
    policy_step_equal = (
        getattr(args, "loss_type", None) == "policy_loss"
        and getattr(args, "policy_loss_agg_mode", "rollout_mean") == "step_equal"
    )
    raw_sdpo_loss_agg_mode = getattr(args, "sdpo_loss_agg_mode", "turn_mean")
    sdpo_loss_agg_mode = _normalize_sdpo_loss_agg_mode(raw_sdpo_loss_agg_mode) if sdpo_loss_selected else None
    sdpo_token_mean = sdpo_loss_agg_mode == "token_mean"
    sdpo_step_equal = sdpo_loss_agg_mode == "step_equal"
    if sdpo_step_equal and not getattr(args, "calculate_per_token_loss", False):
        raise ValueError(f"sdpo_loss_agg_mode={raw_sdpo_loss_agg_mode} requires calculate_per_token_loss=true")
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])
    if sdpo_token_mean:
        num_tokens = torch.clamp_min(_sdpo_active_token_count(batch), 1)
    elif sdpo_step_equal:
        # Keep zero for empty microbatches. Megatron's per-token finalizer
        # aggregates the raw counts across microbatches/ranks and scales only
        # when the global count is nonzero.
        num_tokens = _sdpo_active_prefix_count(batch)
    elif policy_step_equal:
        num_tokens = _sdpo_active_prefix_count_from_masks(batch["loss_masks"])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        None if sdpo_step_equal or policy_step_equal else batch["rollout_mask_sums"],
        args.calculate_per_token_loss and not policy_step_equal,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )

    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "sdpo_loss":
            func = sdpo_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean, use_reentrant=False)
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # With allgather-CP, some CP ranks may have no loss-contributing tokens (e.g., all
    # padding). Without this, gradient doesn't flow through their attention path, so
    # the CP gather's backward (reduce-scatter) is not called, deadlocking other CP
    # ranks that call it. Adding this zero loss forces autograd to traverse the full
    # graph on every rank without changing gradient values.
    if args.allgather_cp and mpu.get_context_parallel_world_size() > 1:
        loss = loss + 0 * logits.sum()

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    if sdpo_token_mean or sdpo_step_equal or policy_step_equal:
        loss = loss * mpu.get_context_parallel_world_size()
    elif not args.calculate_per_token_loss:
        loss = (
            loss
            * num_microbatches
            / step_global_batch_size
            * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    else:
        loss = loss * mpu.get_context_parallel_world_size()

    return (
        loss,
        (
            num_tokens
            if args.calculate_per_token_loss or sdpo_token_mean or sdpo_step_equal or policy_step_equal
            else torch.tensor(1, device=logits.device)
        ),
        {
            "keys": list(log.keys()),
            # values[0] is the consumer's reporting denominator after
            # all-reduce. For per-token-loss it must equal step total tokens
            # (only known by summing per-mb num_tokens across mbs / DP). For
            # per-rollout-mean it is a constant — ``step_global_batch_size`` —
            # so we leave a 0 placeholder here and let ``train_one_step``
            # substitute the constant directly, instead of routing it through
            # per-mb fractions.
            "values": torch.tensor(
                [
                    num_tokens
                    if args.calculate_per_token_loss or sdpo_token_mean or sdpo_step_equal or policy_step_equal
                    else 0,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )
