import torch

from slime.algorithms.sdpo.sgs_metrics import (
    compress_action_view_logits,
    paired_compressed_view_metrics,
)
from slime.backends.megatron_utils import loss as megatron_loss


def test_sgs_compressed_teacher_views_share_support():
    support = torch.tensor([[0, 1], [1, 2]])
    realized = torch.tensor([0, 2])
    ordinary = compress_action_view_logits(
        torch.tensor([[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]]), support, realized
    )
    privileged = compress_action_view_logits(
        torch.tensor([[1.0, 3.0, 0.0], [0.0, 1.0, 3.0]]), support, realized
    )
    metrics = paired_compressed_view_metrics(ordinary, privileged)
    assert torch.all(metrics["js"] > 0)
    assert torch.allclose(
        metrics["js"], paired_compressed_view_metrics(privileged, ordinary)["js"]
    )


def test_compressed_callback_uses_support_emitted_by_student_distillation(monkeypatch):
    support = [torch.tensor([[4, 2]])]
    seen = {}

    def fake_distillation(logits, *, topk_indices, **kwargs):
        assert topk_indices is None
        return torch.empty(0), {"sdpo_topk_indices": support}

    def fake_compressed(logits, *, topk_indices, response_masks, **kwargs):
        seen["support"] = topk_indices
        seen["mask"] = response_masks
        return torch.empty(0), {"sdpo_action_view_support_ids": support}

    monkeypatch.setattr(megatron_loss, "get_sdpo_distillation_tensors", fake_distillation)
    monkeypatch.setattr(megatron_loss, "get_sdpo_compressed_action_view_tensors", fake_compressed)
    _, output = megatron_loss.get_sdpo_distillation_and_compressed_action_view_tensors(
        torch.empty(1, 1, 7),
        response_masks=[[1]],
        topk_indices=None,
    )

    assert seen["support"] is support
    assert seen["mask"] == [[1]]
    assert output["sdpo_action_view_support_ids"] is support
