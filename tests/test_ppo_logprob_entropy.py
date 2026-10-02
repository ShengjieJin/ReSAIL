from __future__ import annotations

import os
import pickle
import socket
from datetime import timedelta

import pytest
import torch
import torch.nn.functional as F

from slime.utils import ppo_utils


NUM_GPUS = 0


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _reference_log_prob_entropy(logits: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=tokens.view(-1, 1))
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return token_log_probs, entropy


@pytest.mark.unit
def test_fused_logprob_entropy_matches_torch_reference_forward_and_backward_single_process():
    torch.manual_seed(1234)
    tokens = torch.tensor([0, 3, 5, 2, 6], dtype=torch.long)
    logits = torch.randn(tokens.numel(), 7, dtype=torch.float32, requires_grad=True)
    logits_before = logits.detach().clone()

    log_prob, entropy = ppo_utils.calculate_log_probs_and_entropy(
        logits,
        tokens,
        tp_group=None,
        with_entropy=True,
    )
    ref_logits = logits.detach().clone().requires_grad_(True)
    ref_log_prob, ref_entropy = _reference_log_prob_entropy(ref_logits, tokens)

    assert log_prob.shape == (tokens.numel(), 1)
    assert entropy.shape == (tokens.numel(),)
    torch.testing.assert_close(log_prob, ref_log_prob, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(entropy, ref_entropy, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(logits.detach(), logits_before)

    grad_log_prob = torch.linspace(-0.4, 0.6, tokens.numel()).view(-1, 1)
    grad_entropy = torch.linspace(0.7, -0.3, tokens.numel())
    loss = (log_prob * grad_log_prob).sum() + (entropy * grad_entropy).sum()
    ref_loss = (ref_log_prob * grad_log_prob).sum() + (ref_entropy * grad_entropy).sum()
    loss.backward()
    ref_loss.backward()

    torch.testing.assert_close(logits.grad, ref_logits.grad, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_fused_entropy_path_does_not_call_compute_log_probs(monkeypatch):
    def fail_compute_log_probs(*_args, **_kwargs):
        raise AssertionError("with_entropy=True must use the fused logprob/entropy path")

    monkeypatch.setattr(ppo_utils, "compute_log_probs", fail_compute_log_probs)
    logits = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
    tokens = torch.tensor([0, 4, 2, 1], dtype=torch.long)

    log_prob, entropy = ppo_utils.calculate_log_probs_and_entropy(
        logits,
        tokens,
        tp_group=None,
        with_entropy=True,
    )

    assert log_prob.shape == (4, 1)
    assert entropy.shape == (4,)


@pytest.mark.unit
def test_chunked_fused_matches_unchunked_forward_and_backward():
    torch.manual_seed(5678)
    tokens = torch.tensor([1, 4, 0, 5, 2, 3], dtype=torch.long)
    logits = torch.randn(tokens.numel(), 6, dtype=torch.float32)
    grad_log_prob = torch.linspace(0.2, 1.1, tokens.numel()).view(-1, 1)
    grad_entropy = torch.linspace(-0.5, 0.4, tokens.numel())

    unchunked_logits = logits.clone().requires_grad_(True)
    unchunked_log_prob, unchunked_entropy = ppo_utils.calculate_log_probs_and_entropy(
        unchunked_logits,
        tokens,
        tp_group=None,
        with_entropy=True,
        chunk_size=-1,
    )
    unchunked_loss = (unchunked_log_prob * grad_log_prob).sum() + (unchunked_entropy * grad_entropy).sum()
    unchunked_loss.backward()

    chunked_logits = logits.clone().requires_grad_(True)
    chunked_log_prob, chunked_entropy = ppo_utils.calculate_log_probs_and_entropy(
        chunked_logits,
        tokens,
        tp_group=None,
        with_entropy=True,
        chunk_size=2,
    )
    chunked_loss = (chunked_log_prob * grad_log_prob).sum() + (chunked_entropy * grad_entropy).sum()
    chunked_loss.backward()

    torch.testing.assert_close(chunked_log_prob, unchunked_log_prob, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(chunked_entropy, unchunked_entropy, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(chunked_logits.grad, unchunked_logits.grad, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_logprob_only_path_preserves_compute_log_probs_chunked_and_unchunked(monkeypatch):
    calls: list[tuple[tuple[int, ...], torch.Tensor]] = []

    def fake_compute_log_probs(logits: torch.Tensor, tokens: torch.Tensor, _process_group):
        calls.append((tuple(logits.shape), tokens.detach().clone()))
        logits.add_(1000.0)
        return tokens.to(dtype=logits.dtype).view(-1, 1)

    monkeypatch.setattr(ppo_utils, "compute_log_probs", fake_compute_log_probs)
    tokens = torch.tensor([4, 1, 3, 0, 2], dtype=torch.long)
    logits = torch.randn(tokens.numel(), 5, dtype=torch.float32)
    logits_before = logits.clone()

    log_prob, entropy = ppo_utils.calculate_log_probs_and_entropy(
        logits,
        tokens,
        tp_group=None,
        with_entropy=False,
        chunk_size=-1,
    )

    assert entropy is None
    assert log_prob.shape == (tokens.numel(), 1)
    assert len(calls) == 1
    assert calls[0][0] == (tokens.numel(), 5)
    torch.testing.assert_close(calls[0][1], tokens)
    torch.testing.assert_close(logits, logits_before)

    calls.clear()
    chunked_logits = logits.clone()
    chunked_before = chunked_logits.clone()
    chunked_log_prob, chunked_entropy = ppo_utils.calculate_log_probs_and_entropy(
        chunked_logits,
        tokens,
        tp_group=None,
        with_entropy=False,
        chunk_size=2,
    )

    assert chunked_entropy is None
    assert chunked_log_prob.shape == (tokens.numel(), 1)
    assert len(calls) == 3
    assert [shape for shape, _tokens in calls] == [(2, 5), (2, 5), (1, 5)]
    torch.testing.assert_close(torch.cat([call_tokens for _shape, call_tokens in calls]), tokens)
    torch.testing.assert_close(chunked_logits, chunked_before)


@pytest.mark.unit
def test_empty_inputs_preserve_existing_shapes(monkeypatch):
    def fail_compute_log_probs(*_args, **_kwargs):
        raise AssertionError("empty inputs should not call compute_log_probs")

    monkeypatch.setattr(ppo_utils, "compute_log_probs", fail_compute_log_probs)
    logits = torch.empty(0, 5, dtype=torch.float32)
    tokens = torch.empty(0, dtype=torch.long)

    log_prob, entropy = ppo_utils.calculate_log_probs_and_entropy(
        logits,
        tokens,
        tp_group=None,
        with_entropy=False,
    )
    assert log_prob.shape == (0,)
    assert entropy is None

    log_prob, entropy = ppo_utils.calculate_log_probs_and_entropy(
        logits,
        tokens,
        tp_group=None,
        with_entropy=True,
    )
    assert log_prob.shape == (0,)
    assert entropy.shape == (0,)


@pytest.mark.unit
def test_removed_entropy_helper_surface():
    assert not hasattr(ppo_utils, "compute_entropy_from_logits")
    assert not hasattr(ppo_utils, "_VocabParallelEntropy")


def _tp_fused_worker(rank: int, world_size: int, master_port: int, result_path: str) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        tokens = torch.tensor([0, 4, 2, 5, 1], dtype=torch.long)
        full_logits_data = torch.tensor(
            [
                [0.3, -0.2, 1.1, -0.7, 0.5, 0.9],
                [1.0, 0.1, -0.4, 0.8, 1.5, -1.2],
                [-0.6, 0.4, 0.7, 1.2, -0.3, 0.2],
                [0.9, -1.1, 0.0, 0.6, 0.3, 1.4],
                [-0.8, 1.3, 0.2, -0.5, 0.7, 0.4],
            ],
            dtype=torch.float32,
        )
        grad_log_prob = torch.tensor([[0.5], [-0.25], [0.75], [-0.5], [0.2]], dtype=torch.float32)
        grad_entropy = torch.tensor([0.1, -0.3, 0.4, -0.2, 0.6], dtype=torch.float32)

        local_vocab = full_logits_data.size(-1) // world_size
        vocab_start = rank * local_vocab
        vocab_end = vocab_start + local_vocab
        local_logits = full_logits_data[:, vocab_start:vocab_end].clone().requires_grad_(True)

        log_prob, entropy = ppo_utils.calculate_log_probs_and_entropy(
            local_logits,
            tokens,
            tp_group=dist.group.WORLD,
            with_entropy=True,
        )
        loss = (log_prob * grad_log_prob).sum() + (entropy * grad_entropy).sum()
        loss.backward()

        ref_logits = full_logits_data.clone().requires_grad_(True)
        ref_log_prob, ref_entropy = _reference_log_prob_entropy(ref_logits, tokens)
        ref_loss = (ref_log_prob * grad_log_prob).sum() + (ref_entropy * grad_entropy).sum()
        ref_loss.backward()

        gathered_grads = [torch.empty_like(local_logits.grad) for _ in range(world_size)]
        dist.all_gather(gathered_grads, local_logits.grad)

        if rank == 0:
            sharded_grad = torch.cat(gathered_grads, dim=-1)
            torch.testing.assert_close(log_prob, ref_log_prob, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(entropy, ref_entropy, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(sharded_grad, ref_logits.grad, rtol=1e-6, atol=1e-6)
            with open(result_path, "wb") as f:
                pickle.dump("ok", f)
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_vocab_parallel_fused_matches_full_vocab_reference_two_rank_gloo(tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    if not dist.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is not available")

    result_path = str(tmp_path / "tp_fused_result.pkl")
    mp.spawn(
        _tp_fused_worker,
        args=(2, _free_port(), result_path),
        nprocs=2,
        join=True,
    )

    with open(result_path, "rb") as f:
        assert pickle.load(f) == "ok"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
