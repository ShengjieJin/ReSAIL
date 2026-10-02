from __future__ import annotations

import math
import random
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch

NUM_GPUS = 0


def _import_actor_module():
    try:
        from slime.backends.megatron_utils import actor as actor_module
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(f"actor module requires full Megatron training package: {exc}")
    return actor_module


def _set_mpu_attr(monkeypatch, mpu, name: str, value) -> None:
    monkeypatch.setattr(mpu, name, value, raising=False)


@pytest.mark.unit
def test_dynamic_six_dp_uses_rounding_only_for_unused_megatron_bootstrap_calculator():
    from slime.backends.megatron_utils.initialize import _decrease_dummy_batch_size_if_needed, _dummy_global_batch_size

    common = {
        "rank": 0,
        "global_batch_size": 512,
        "micro_batch_size": 1,
        "decrease_batch_size_if_needed": False,
    }
    assert _decrease_dummy_batch_size_if_needed(
        SimpleNamespace(**common, data_parallel_size=6, use_dynamic_batch_size=True)
    )
    assert not _decrease_dummy_batch_size_if_needed(
        SimpleNamespace(**common, data_parallel_size=4, use_dynamic_batch_size=True)
    )
    assert not _decrease_dummy_batch_size_if_needed(
        SimpleNamespace(**common, data_parallel_size=6, use_dynamic_batch_size=False)
    )
    assert not _decrease_dummy_batch_size_if_needed(SimpleNamespace(**common, data_parallel_size=6))
    tiny = SimpleNamespace(**{**common, "global_batch_size": 1}, data_parallel_size=8, use_dynamic_batch_size=True)
    assert not _decrease_dummy_batch_size_if_needed(tiny)
    assert _dummy_global_batch_size(tiny) == 8
    assert tiny.global_batch_size == 1


@pytest.mark.unit
def test_tensor_backuper_save_load_roundtrip(tmp_path):
    from slime.utils.tensor_backper import TensorBackuper

    params = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
    backuper = TensorBackuper.create(lambda: params.items(), single_tag=None)

    backuper.backup("actor")
    path = tmp_path / "rank_00000.pt"
    backuper.save("actor", path)

    params["w"].add_(10.0)
    params["b"].add_(10.0)
    backuper.load("sdpo_teacher", path)
    backuper.restore("sdpo_teacher")

    torch.testing.assert_close(params["w"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(params["b"], torch.tensor([3.0]))


@pytest.mark.unit
def test_tensor_backuper_discards_temporary_tag_without_touching_live_tensors():
    from slime.utils.tensor_backper import TensorBackuper

    params = {"w": torch.tensor([1.0])}
    backuper = TensorBackuper.create(lambda: params.items(), single_tag=None)
    backuper.backup("temporary")
    params["w"].fill_(7.0)

    backuper.discard("temporary")

    assert backuper.backup_tags == []
    torch.testing.assert_close(params["w"], torch.tensor([7.0]))
    with pytest.raises(KeyError, match="unknown"):
        backuper.discard("temporary")


@pytest.mark.unit
def test_online_old_actor_queue_skips_initial_redundant_backup_then_advances():
    actor_module = _import_actor_module()

    class Backuper:
        backup_tags = ["actor", "old_actor", "rollout_actor"]

        def __init__(self):
            self.calls = []

        def copy(self, **kwargs):
            self.calls.append(("copy", kwargs))

        def backup(self, tag):
            self.calls.append(("backup", tag))

    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.weights_backuper = Backuper()
    actor._online_old_actor_weight_sync_initialized = False

    assert actor._advance_online_old_actor_after_weight_sync() is False
    assert actor.weights_backuper.calls == []
    assert actor._advance_online_old_actor_after_weight_sync() is True
    assert actor.weights_backuper.calls == [
        ("copy", {"src_tag": "rollout_actor", "dst_tag": "old_actor"}),
        ("copy", {"src_tag": "actor", "dst_tag": "rollout_actor"}),
    ]


@pytest.mark.unit
def test_get_sdpo_distillation_tensors_scores_student_topk_support(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)

    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        sdpo_full_logit_distillation=True,
        sdpo_distillation_topk=2,
        allgather_cp=False,
    )
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    topk_indices = [torch.tensor([[2, 0], [1, 3]])]

    _, out = loss_module.get_sdpo_distillation_tensors(
        logits,
        args=args,
        unconcat_tokens=[torch.tensor([10, 11, 12])],
        total_lengths=[3],
        response_lengths=[2],
        response_log_probs=[torch.tensor([-0.1, -0.2])],
        topk_indices=topk_indices,
    )

    expected = torch.log_softmax(logits[0, :2], dim=-1).gather(dim=-1, index=topk_indices[0])
    torch.testing.assert_close(out["sdpo_topk_log_probs"][0], expected)
    torch.testing.assert_close(out["sdpo_topk_indices"][0], topk_indices[0])
    torch.testing.assert_close(out["log_probs"][0], torch.tensor([-0.1, -0.2]))


@pytest.mark.unit
@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
def test_decision_hidden_slicing_is_exact_for_thd_and_bshd(monkeypatch, qkv_format):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    flat = torch.arange(7 * 2, dtype=torch.float32).reshape(7, 2)
    hidden = flat[:, None, :] if qkv_format == "thd" else flat.reshape(1, 7, 2)

    response, prompt_last = loss_module.get_response_and_prompt_last_hidden_representations(
        hidden,
        args=SimpleNamespace(qkv_format=qkv_format),
        total_lengths=[7],
        response_lengths=[3],
        max_seq_lens=[7] if qkv_format == "bshd" else None,
    )

    torch.testing.assert_close(prompt_last[0], flat[3])
    torch.testing.assert_close(response[0], flat[4:7])


@pytest.mark.unit
def test_decision_hidden_slicing_r1_keeps_prompt_and_single_response(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    hidden = torch.arange(3 * 2, dtype=torch.float32).reshape(3, 1, 2)
    response, prompt_last = loss_module.get_response_and_prompt_last_hidden_representations(
        hidden,
        args=SimpleNamespace(qkv_format="thd"),
        total_lengths=[3],
        response_lengths=[1],
    )
    torch.testing.assert_close(prompt_last[0], hidden[1, 0])
    torch.testing.assert_close(response[0], hidden[2:3, 0])


@pytest.mark.unit
def test_get_sdpo_distillation_tensors_requests_logprob_without_entropy_when_needed(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    calls: list[bool] = []

    def fake_get_log_probs_and_entropy(*_args, **kwargs):
        calls.append(kwargs["with_entropy"])
        return torch.empty((0,)), {"log_probs": [torch.tensor([-0.1, -0.2])]}

    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", fake_get_log_probs_and_entropy)
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        sdpo_full_logit_distillation=False,
        allgather_cp=False,
    )

    _, out = loss_module.get_sdpo_distillation_tensors(
        torch.zeros(1, 3, 4),
        args=args,
        unconcat_tokens=[torch.tensor([10, 11, 12])],
        total_lengths=[3],
        response_lengths=[2],
        with_entropy=True,
        response_log_probs=None,
    )

    assert calls == [False]
    torch.testing.assert_close(out["log_probs"][0], torch.tensor([-0.1, -0.2]))


@pytest.mark.unit
def test_get_sdpo_representation_tensors_slices_response_suffix_without_logit_shift(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)

    args = SimpleNamespace(qkv_format="thd")
    hidden_states = torch.arange(5 * 2, dtype=torch.bfloat16).reshape(5, 1, 2)

    _, out = loss_module.get_sdpo_representation_tensors(
        hidden_states,
        args=args,
        unconcat_tokens=[torch.tensor([10, 11, 12]), torch.tensor([20, 21])],
        total_lengths=[3, 2],
        response_lengths=[2, 1],
    )

    assert out["representations"][0].device.type == "cpu"
    assert out["representations"][0].dtype == torch.bfloat16
    torch.testing.assert_close(out["representations"][0], hidden_states[1:3, 0, :])
    torch.testing.assert_close(out["representations"][1], hidden_states[4:5, 0, :])


@pytest.mark.unit
def test_sdpo_token_loss_requests_logprob_without_entropy(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_rank", lambda: 0)
    calls: list[bool] = []

    def fake_get_log_probs_and_entropy(*_args, **kwargs):
        calls.append(kwargs["with_entropy"])
        return torch.empty((0,)), {"log_probs": [torch.tensor([-0.25, -0.75])]}

    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", fake_get_log_probs_and_entropy)
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        sdpo_full_logit_distillation=False,
        sdpo_loss_agg_mode="turn_mean",
        sdpo_clip_ratio=None,
        allgather_cp=False,
    )
    batch = {
        "unconcat_tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "rollout_mask_sums": torch.tensor([2.0]),
        "sdpo_teacher_log_probs": [torch.tensor([-0.5, -0.5])],
        "self_distillation_mask": torch.tensor([1.0]),
        "sdpo_loss_weights": torch.tensor([1.0]),
        # get_batch keeps requested optional fields with a None value.  A
        # selected-step distillation-only batch must not be interpreted as PR.
        "pr_component": None,
        "pr_base_weight": None,
    }

    loss, log = loss_module.sdpo_loss_function(
        args,
        batch,
        torch.zeros(1, 3, 4, requires_grad=True),
        lambda x: x.sum(),
    )

    assert calls == [False]
    assert torch.isfinite(loss)
    assert torch.isfinite(log["sdpo_loss"])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("component", "base_weight"),
    [([0.0], None), (None, [1.0])],
)
def test_sdpo_token_loss_rejects_half_present_method_b_vectors(monkeypatch, component, base_weight):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (torch.empty((0,)), {"log_probs": [torch.tensor([-0.2])]}),
    )
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        sdpo_full_logit_distillation=False,
        sdpo_loss_agg_mode="step_equal",
        sdpo_clip_ratio=None,
        allgather_cp=False,
    )
    batch = {
        "unconcat_tokens": [torch.tensor([1, 2])],
        "total_lengths": [2],
        "response_lengths": [1],
        "loss_masks": [torch.ones(1)],
        "sdpo_teacher_log_probs": [torch.tensor([-0.5])],
        "self_distillation_mask": torch.tensor([1.0]),
        "sdpo_loss_weights": torch.tensor([1.0]),
        "pr_component": component,
        "pr_base_weight": base_weight,
    }
    with pytest.raises(KeyError, match="both be present or both be absent"):
        loss_module.sdpo_loss_function(
            args,
            batch,
            torch.zeros(1, 2, 4, requires_grad=True),
            lambda x: x.sum(),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("components", "expected_selection", "expected_retention", "expected_joint"),
    [([0.0, 1.0], -0.12, -0.08, -0.144), ([0.0, 0.0], -0.20, 0.0, -0.20)],
)
def test_sdpo_token_loss_reports_exact_method_b_components(
    monkeypatch, components, expected_selection, expected_retention, expected_joint
):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (
            torch.empty((0,)),
            {"log_probs": [torch.tensor([-0.2]), torch.tensor([-0.4])]},
        ),
    )
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        sdpo_full_logit_distillation=False,
        sdpo_loss_agg_mode="step_equal",
        sdpo_clip_ratio=None,
        allgather_cp=False,
        pr_weight=0.3,
    )
    batch = {
        "unconcat_tokens": [torch.tensor([1, 2]), torch.tensor([3, 4])],
        "total_lengths": [2, 2],
        "response_lengths": [1, 1],
        "loss_masks": [torch.ones(1), torch.ones(1)],
        "sdpo_teacher_log_probs": [torch.tensor([-0.5]), torch.tensor([-0.5])],
        "self_distillation_mask": torch.tensor([1.0, 1.0]),
        "sdpo_loss_weights": torch.tensor([2.0, 0.6] if components == [0.0, 1.0] else [2.0, 2.0]),
        "pr_component": components,
        "pr_base_weight": [1.0, 1.0],
    }
    loss, metrics = loss_module.sdpo_loss_function(
        args,
        batch,
        torch.zeros(1, 4, 4, requires_grad=True),
        lambda x: x.sum(),
    )
    assert loss.item() == pytest.approx(expected_joint)
    assert metrics["sgs_loss"].item() == pytest.approx(expected_selection)
    assert metrics["pr_loss"].item() == pytest.approx(expected_retention)
    assert metrics["pr_weighted_loss"].item() == pytest.approx(0.3 * expected_retention)


@pytest.mark.unit
def test_sdpo_token_loss_applies_per_token_sdpo_weights(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_rank", lambda: 0)

    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (torch.empty((0,)), {"log_probs": [torch.tensor([-0.25, -0.75])]}),
    )
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        sdpo_full_logit_distillation=False,
        sdpo_loss_agg_mode="token_mean",
        sdpo_clip_ratio=None,
        allgather_cp=False,
    )
    batch = {
        "unconcat_tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "rollout_mask_sums": torch.tensor([2.0]),
        "sdpo_teacher_log_probs": [torch.tensor([-0.5, -0.5])],
        "self_distillation_mask": torch.tensor([1.0]),
        "sdpo_loss_weights": torch.tensor([1.0]),
        "sdpo_token_weights": [torch.tensor([2.0, 0.5])],
    }

    loss, _log = loss_module.sdpo_loss_function(
        args,
        batch,
        torch.zeros(1, 3, 4, requires_grad=True),
        lambda x: x.sum(),
    )

    expected = ((-0.25 - -0.5) * -0.25 * 2.0) + (((-0.75 - -0.5) * -0.75) * 0.5)
    assert loss.item() == pytest.approx(expected)


@pytest.mark.unit
def test_sdpo_oel_is_sum_of_equal_active_prefix_means(monkeypatch):
    """OEL sums row means for Megatron, with the row count as its normalizer."""
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (
            torch.empty((0,)),
            {
                "log_probs": [
                    torch.tensor([1.0, 3.0]),
                    torch.tensor([2.0, 4.0, 6.0, 8.0]),
                    torch.tensor([100.0]),
                    torch.tensor([200.0, 300.0]),
                ]
            },
        ),
    )
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        sdpo_full_logit_distillation=False,
        sdpo_loss_agg_mode="oel_prefix_mean",
        sdpo_clip_ratio=None,
        sdpo_deployment_tis_clip=None,
        allgather_cp=False,
    )
    batch = {
        "unconcat_tokens": [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6, 7, 8]),
            torch.tensor([9, 10]),
            torch.tensor([11, 12, 13]),
        ],
        "total_lengths": [3, 5, 2, 3],
        "response_lengths": [2, 4, 1, 2],
        "loss_masks": [
            torch.tensor([1.0, 1.0]),
            torch.tensor([1.0, 1.0, 1.0, 1.0]),
            torch.tensor([1.0]),
            torch.tensor([0.0, 0.0]),
        ],
        # Deliberately unrelated to per-prefix lengths: OEL must not use it.
        "rollout_mask_sums": torch.tensor([100.0, 100.0, 100.0, 100.0]),
        "sdpo_teacher_log_probs": [
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0, 0.0]),
        ],
        "self_distillation_mask": torch.tensor([1.0, 1.0, 0.0, 1.0]),
        "sdpo_loss_weights": torch.ones(4),
    }

    loss, metrics = loss_module.sdpo_loss_function(
        args,
        batch,
        torch.zeros(1, 13, 4, requires_grad=True),
        lambda x: x.sum(),
    )

    # Per-token losses are student_log_prob * (student - teacher):
    # row 0 mean=(1+9)/2=5; row 1 mean=(4+16+36+64)/4=30.
    # Rows 2 and 3 are inactive (self mask / zero active tokens).
    assert loss.item() == pytest.approx(35.0)
    assert metrics["sdpo_active_prefixes"].item() == pytest.approx(2.0)


@pytest.mark.unit
def test_sdpo_oel_is_invariant_to_active_prefix_row_permutation(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_rank", lambda: 0)

    def run(rows):
        monkeypatch.setattr(
            loss_module,
            "get_log_probs_and_entropy",
            lambda *_args, **_kwargs: (torch.empty((0,)), {"log_probs": [row[0] for row in rows]}),
        )
        args = SimpleNamespace(
            qkv_format="thd",
            rollout_temperature=1.0,
            sdpo_full_logit_distillation=False,
            sdpo_loss_agg_mode="oel_prefix_mean",
            sdpo_clip_ratio=None,
            sdpo_deployment_tis_clip=None,
            allgather_cp=False,
        )
        batch = {
            "unconcat_tokens": [torch.arange(row[0].numel() + 1) for row in rows],
            "total_lengths": [int(row[0].numel() + 1) for row in rows],
            "response_lengths": [int(row[0].numel()) for row in rows],
            "loss_masks": [row[1] for row in rows],
            "rollout_mask_sums": torch.tensor([999.0] * len(rows)),
            "sdpo_teacher_log_probs": [torch.zeros_like(row[0]) for row in rows],
            "self_distillation_mask": torch.ones(len(rows)),
            "sdpo_loss_weights": torch.ones(len(rows)),
        }
        return loss_module.sdpo_loss_function(
            args,
            batch,
            torch.zeros(1, sum(int(row[0].numel() + 1) for row in rows), 4, requires_grad=True),
            lambda x: x.sum(),
        )[0]

    rows = [
        (torch.tensor([1.0, 3.0]), torch.tensor([1.0, 1.0])),
        (torch.tensor([2.0, 4.0, 6.0]), torch.tensor([1.0, 1.0, 1.0])),
        (torch.tensor([5.0]), torch.tensor([1.0])),
    ]
    first = run(rows)
    second = run([rows[2], rows[0], rows[1]])
    assert first.item() == pytest.approx(second.item())


@pytest.mark.unit
def test_loss_function_oel_uses_active_prefix_count_normalizer_and_token_scaling(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_data_parallel_world_size", lambda with_context_parallel=True: 1)
    monkeypatch.setattr(
        loss_module,
        "sdpo_loss_function",
        lambda *args: (torch.tensor(35.0), {"sdpo_loss": torch.tensor(35.0)}),
    )
    args = SimpleNamespace(
        loss_type="sdpo_loss",
        sdpo_loss_agg_mode="oel_prefix_mean",
        calculate_per_token_loss=True,
        qkv_format="thd",
        allgather_cp=False,
        recompute_loss_function=False,
    )
    batch = {
        "loss_masks": [torch.ones(2), torch.ones(4), torch.ones(1), torch.zeros(2)],
        "total_lengths": [3, 5, 2, 3],
        "response_lengths": [2, 4, 1, 2],
        "self_distillation_mask": torch.tensor([1.0, 1.0, 0.0, 1.0]),
    }

    loss, normalizer, log = loss_module.loss_function(
        args,
        batch,
        num_microbatches=2,
        step_global_batch_size=32,
        logits=torch.zeros(1, 1, 1),
    )

    assert loss.item() == pytest.approx(35.0)
    assert normalizer.item() == pytest.approx(2.0)
    assert normalizer.dtype == torch.long
    assert log["values"][0].item() == pytest.approx(2.0)


@pytest.mark.unit
def test_loss_function_oel_preserves_zero_active_prefix_count(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        loss_module,
        "sdpo_loss_function",
        lambda _args, _batch, logits, _sum_of_sample_mean: (
            logits.sum() * 0,
            {"sdpo_loss": torch.tensor(0.0, device=logits.device)},
        ),
    )
    args = SimpleNamespace(
        loss_type="sdpo_loss",
        sdpo_loss_agg_mode="oel_prefix_mean",
        calculate_per_token_loss=True,
        qkv_format="thd",
        allgather_cp=False,
        recompute_loss_function=False,
    )
    batch = {
        "loss_masks": [torch.zeros(2), torch.zeros(4)],
        "total_lengths": [3, 5],
        "response_lengths": [2, 4],
        "self_distillation_mask": torch.ones(2),
    }

    _, normalizer, log = loss_module.loss_function(
        args,
        batch,
        num_microbatches=2,
        step_global_batch_size=32,
        logits=torch.zeros(1, 1, 1, requires_grad=True),
    )

    assert normalizer.item() == pytest.approx(0.0)
    assert normalizer.dtype == torch.long
    assert log["values"][0].item() == pytest.approx(0.0)


@pytest.mark.unit
def test_sdpo_oel_report_averages_prefix_sums_across_microbatches_and_cp(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module
    from slime.backends.megatron_utils.cp_utils import reduce_train_step_metrics

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_data_parallel_world_size", lambda with_context_parallel=True: 1)
    monkeypatch.setattr(
        loss_module,
        "sdpo_loss_function",
        lambda *args: (torch.tensor(float(args[1]["loss_masks"][0].sum())), {"sdpo_loss": torch.tensor(1.0)}),
    )
    args = SimpleNamespace(
        loss_type="sdpo_loss",
        sdpo_loss_agg_mode="oel_prefix_mean",
        calculate_per_token_loss=True,
        qkv_format="thd",
        allgather_cp=False,
        recompute_loss_function=False,
    )

    def one_microbatch(token_count):
        return loss_module.loss_function(
            args,
            {
                "loss_masks": [torch.ones(token_count)],
                "total_lengths": [token_count + 1],
                "response_lengths": [token_count],
                "self_distillation_mask": torch.tensor([1.0]),
            },
            num_microbatches=2,
            step_global_batch_size=32,
            logits=torch.zeros(1, 1, 1),
        )

    first_loss, first_count, _ = one_microbatch(5)
    second_loss, second_count, _ = one_microbatch(30)
    assert first_count.item() == pytest.approx(1.0)
    assert second_count.item() == pytest.approx(1.0)

    # Two microbatch prefix sums are [5, 30], so the global OEL report is
    # (5 + 30) / (1 + 1) = 17.5. CP duplicates both slots; cp_factor cancels.
    values = torch.tensor([first_count.item() + second_count.item(), first_loss.item() + second_loss.item()])
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, group=None, op=None: None)
    report = reduce_train_step_metrics(
        [{"keys": ["sdpo_loss"], "values": values}],
        calculate_per_token_loss=True,
        step_global_batch_size=32,
        cp_size=1,
        dp_with_cp_group=object(),
    )
    assert report["sdpo_loss"] == pytest.approx(17.5)
    cp_report = reduce_train_step_metrics(
        [{"keys": ["sdpo_loss"], "values": torch.tensor([4.0, 35.0])}],
        calculate_per_token_loss=True,
        step_global_batch_size=32,
        cp_size=2,
        dp_with_cp_group=object(),
    )
    assert cp_report["sdpo_loss"] == pytest.approx(17.5)


@pytest.mark.unit
def test_sdpo_oel_report_all_empty_step_uses_safe_display_denominator(monkeypatch):
    from slime.backends.megatron_utils.cp_utils import reduce_train_step_metrics

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, group=None, op=None: None)
    report = reduce_train_step_metrics(
        [{"keys": ["sdpo_loss"], "values": torch.tensor([0.0, 0.0])}],
        calculate_per_token_loss=True,
        step_global_batch_size=32,
        cp_size=2,
        dp_with_cp_group=object(),
    )

    assert report["sdpo_loss"] == pytest.approx(0.0)


@pytest.mark.unit
def test_sdpo_oel_global_schedule_normalizes_uneven_microbatches(monkeypatch):
    """The global-count schedule is required for an OEL gradient mean."""
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_data_parallel_world_size", lambda with_context_parallel=True: 1)

    def fake_sdpo(_args, batch, logits, _sum_of_sample_mean):
        value = torch.as_tensor(batch["mock_row_mean_sum"], dtype=logits.dtype, device=logits.device)
        return logits.sum() * value, {"sdpo_loss": value}

    monkeypatch.setattr(loss_module, "sdpo_loss_function", fake_sdpo)

    def run(calculate_per_token_loss: bool) -> float:
        args = SimpleNamespace(
            loss_type="sdpo_loss",
            sdpo_loss_agg_mode="oel_prefix_mean",
            calculate_per_token_loss=calculate_per_token_loss,
            qkv_format="thd",
            allgather_cp=False,
            recompute_loss_function=False,
        )
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        total_prefixes = 0.0
        for row_mean_sum, row_count in ((10.0, 1), (3.0, 3)):
            batch = {
                "loss_masks": [torch.ones(1) for _ in range(row_count)],
                "total_lengths": [2] * row_count,
                "response_lengths": [1] * row_count,
                "self_distillation_mask": torch.ones(row_count),
                "mock_row_mean_sum": row_mean_sum,
            }
            loss, normalizer, _ = loss_module.loss_function(
                args,
                batch,
                num_microbatches=2,
                step_global_batch_size=4,
                logits=parameter.reshape(1, 1, 1),
            )
            assert normalizer.item() == pytest.approx(row_count)
            total_prefixes += normalizer.item()
            loss.backward()

        # The global path leaves each microbatch sum intact; finalization
        # divides the accumulated gradient by the global prefix count.
        return parameter.grad.item() / total_prefixes

    assert run(calculate_per_token_loss=True) == pytest.approx(13.0 / 4.0)
    with pytest.raises(ValueError, match="oel_prefix_mean.*calculate_per_token_loss"):
        run(calculate_per_token_loss=False)


@pytest.mark.unit
def test_sdpo_token_loss_combines_official_policy_clip_and_deployment_tis(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (torch.empty((0,)), {"log_probs": [torch.tensor([-2.0])]}),
    )
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        sdpo_full_logit_distillation=False,
        sdpo_loss_agg_mode="token_mean",
        sdpo_clip_ratio=2.0,
        sdpo_deployment_tis_clip=2.0,
        allgather_cp=False,
    )
    batch = {
        "unconcat_tokens": [torch.tensor([1, 2])],
        "total_lengths": [2],
        "response_lengths": [1],
        "loss_masks": [torch.ones(1)],
        "rollout_mask_sums": torch.tensor([1.0]),
        "sdpo_teacher_log_probs": [torch.tensor([-1.0])],
        "self_distillation_mask": torch.tensor([1.0]),
        "sdpo_loss_weights": torch.tensor([1.0]),
        "log_probs": [torch.tensor([-3.0])],
        "rollout_log_probs": [torch.tensor([-5.0])],
    }

    loss, metrics = loss_module.sdpo_loss_function(
        args,
        batch,
        torch.zeros(1, 2, 4, requires_grad=True),
        lambda x: x.sum(),
    )

    assert loss.item() == pytest.approx(((-2.0 - -1.0) * -2.0) * 2.0 * 2.0)
    assert metrics["sdpo_policy_ratio_clip_fraction"].item() == pytest.approx(1.0)
    assert metrics["sdpo_deployment_tis_ratio_clip_fraction"].item() == pytest.approx(1.0)
    assert metrics["sdpo_combined_correction_mean"].item() == pytest.approx(4.0)


@pytest.mark.unit
def test_sdpo_no_ratio_omits_all_correction_metrics_even_with_deployment_log_probs(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (torch.empty((0,)), {"log_probs": [torch.tensor([-2.0])]}),
    )
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        sdpo_full_logit_distillation=False,
        sdpo_loss_agg_mode="token_mean",
        sdpo_clip_ratio=None,
        sdpo_deployment_tis_clip=None,
        allgather_cp=False,
    )
    batch = {
        "unconcat_tokens": [torch.tensor([1, 2])],
        "total_lengths": [2],
        "response_lengths": [1],
        "loss_masks": [torch.ones(1)],
        "rollout_mask_sums": torch.tensor([1.0]),
        "sdpo_teacher_log_probs": [torch.tensor([-1.0])],
        "self_distillation_mask": torch.tensor([1.0]),
        "sdpo_loss_weights": torch.tensor([1.0]),
        "log_probs": [torch.tensor([999.0])],
        "rollout_log_probs": [torch.tensor([-999.0])],
    }

    loss, metrics = loss_module.sdpo_loss_function(
        args,
        batch,
        torch.zeros(1, 2, 4, requires_grad=True),
        lambda x: x.sum(),
    )

    assert torch.isfinite(loss)
    assert not any(
        key.startswith(("sdpo_policy_ratio_", "sdpo_deployment_tis_ratio_", "sdpo_combined_correction_"))
        for key in metrics
    )


@pytest.mark.unit
def test_offline_corrected_refresh_captures_live_actor_before_old_copy_and_boundary_sync_skips_old():
    actor_module = _import_actor_module()
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        offline_async_eval=True,
        offline_train_eval_colocate=False,
        keep_old_actor=True,
        loss_type="sdpo_loss",
        sdpo_clip_ratio=2.0,
        sdpo_deployment_tis_clip=2.0,
    )
    events = []
    actor.weights_backuper = SimpleNamespace(
        backup=lambda tag: events.append(("backup", tag)),
        copy=lambda *, src_tag, dst_tag: events.append(("copy", src_tag, dst_tag)),
    )
    actor._update_sdpo_ema_teacher = lambda: events.append(("ema",))

    assert actor._refresh_actor_backups_after_update()
    assert events == [("backup", "actor"), ("copy", "actor", "old_actor"), ("ema",)]
    assert not actor._should_update_old_actor_during_weight_sync()


@pytest.mark.unit
def test_offline_no_ratio_never_requests_old_actor_refresh():
    actor_module = _import_actor_module()
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        offline_async_eval=True,
        offline_train_eval_colocate=False,
        keep_old_actor=False,
        loss_type="sdpo_loss",
        sdpo_clip_ratio=None,
        sdpo_deployment_tis_clip=None,
    )
    assert not actor._uses_per_update_sdpo_old_actor()
    assert not actor._should_update_old_actor_during_weight_sync()
    assert not actor._needs_sdpo_correction_log_probs()


@pytest.mark.unit
def test_online_sdpo_without_ratio_or_auxiliary_consumer_skips_log_prob_forward():
    actor_module = _import_actor_module()
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        offline_async_eval=False,
        loss_type="sdpo_loss",
        sdpo_clip_ratio=None,
        sdpo_deployment_tis_clip=None,
        kl_coef=0.0,
        use_opd=False,
        use_tis=False,
        use_opsm=False,
        log_correct_samples=False,
        custom_advantage_function_path=None,
    )

    assert not actor._needs_sdpo_correction_log_probs()

    actor.args.sdpo_clip_ratio = 2.0
    assert actor._needs_sdpo_correction_log_probs()


@pytest.mark.unit
def test_zero_kl_grpo_builds_returns_from_loss_masks_without_actor_log_probs(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    monkeypatch.setattr(loss_module.mpu, "is_pipeline_last_stage", lambda: True)
    args = SimpleNamespace(
        use_rollout_logprobs=False,
        kl_coef=0.0,
        custom_advantage_function_path=None,
        advantage_estimator="grpo",
        use_opd=False,
        normalize_advantages=False,
    )
    rollout_data = {
        "rewards": [1.0, 0.0],
        "values": None,
        "response_lengths": [2, 1],
        "loss_masks": [torch.tensor([1.0, 1.0]), torch.tensor([1.0])],
        "total_lengths": [3, 2],
    }

    loss_module.compute_advantages_and_returns(args, rollout_data)

    assert [row.shape for row in rollout_data["kl"]] == [torch.Size([2]), torch.Size([1])]
    assert all(torch.count_nonzero(row) == 0 for row in rollout_data["kl"])
    assert len(rollout_data["advantages"]) == len(rollout_data["returns"]) == 2


@pytest.mark.unit
def test_dual_support_teacher_extraction_matches_two_independent_support_gathers(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    calls = []

    def fake_distillation(logits, *, topk_indices=None, response_log_probs=None, **_kwargs):
        calls.append((id(logits), topk_indices, response_log_probs))
        support = [torch.tensor([[9, 8]])] if topk_indices is None else topk_indices
        log_probs = [torch.tensor([-0.2])] if response_log_probs is None else response_log_probs
        return torch.empty(0), {
            "log_probs": log_probs,
            "sdpo_topk_indices": support,
            "sdpo_topk_log_probs": [support[0].to(dtype=torch.float32) / 10],
        }

    monkeypatch.setattr(loss_module, "get_sdpo_distillation_tensors", fake_distillation)
    monkeypatch.setattr(
        loss_module,
        "get_sdpo_compressed_action_view_tensors",
        lambda logits, *, topk_indices, **_kwargs: (
            torch.empty(0),
            {"sdpo_action_view_support_ids": topk_indices},
        ),
    )
    logits = torch.zeros(1, 2, 4)
    primary = [torch.tensor([[3, 1]])]

    _, output = loss_module.get_sdpo_dual_support_distillation_and_compressed_action_view_tensors(
        logits,
        args=SimpleNamespace(),
        unconcat_tokens=[torch.tensor([1, 2])],
        total_lengths=[2],
        response_lengths=[1],
        response_masks=[[True]],
        topk_indices=primary,
        secondary_topk_indices=None,
    )

    assert len(calls) == 2
    assert calls[0][0] == calls[1][0] == id(logits)
    assert calls[0][1] is primary
    assert calls[1][1] is None
    assert calls[1][2] is output["log_probs"]
    torch.testing.assert_close(output["sdpo_topk_indices"][0], primary[0])
    torch.testing.assert_close(output["sdpo_retention_topk_indices"][0], torch.tensor([[9, 8]]))


@pytest.mark.unit
def test_dual_support_actor_forward_routes_secondary_support_through_microbatch_data(monkeypatch):
    actor_module = _import_actor_module()
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace()
    actor.model = object()
    iterator = SimpleNamespace(rollout_data={})
    secondary = [torch.tensor([[7, 6]])]

    def fake_forward_only(callback, args, model, data_iterator, num_microbatches, *, store_prefix):
        assert callback is actor_module.get_sdpo_dual_support_distillation_and_compressed_action_view_tensors
        assert args is actor.args
        assert model is actor.model
        assert data_iterator == [iterator]
        assert num_microbatches == [1]
        assert store_prefix == "teacher_"
        assert iterator.rollout_data["sdpo_secondary_topk_indices"] is secondary
        return {"ok": []}

    monkeypatch.setattr(actor_module, "forward_only", fake_forward_only)

    assert actor.compute_sdpo_dual_support_compressed_action_view_data(
        [iterator], [1], "teacher_", secondary_topk_indices=secondary
    ) == {"ok": []}


@pytest.mark.unit
def test_first_weight_sync_connects_each_actor_rank_after_discovery_marker_is_consumed(monkeypatch):
    actor_module = _import_actor_module()
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)

    class RemoteMethod:
        def __init__(self, fn):
            self.fn = fn

        def remote(self):
            return self.fn()

    actor.args = SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        use_fault_tolerance=False,
        offload_train=False,
        use_critic=False,
        colocate=False,
        ci_test=False,
        update_weights_interval=1,
    )
    engines = [object() for _ in range(4)]
    actor.rollout_manager = SimpleNamespace(
        get_updatable_engines_and_lock=RemoteMethod(lambda: (engines, object(), 0, [1, 1, 1, 1], [0, 1, 2, 3]))
    )
    calls = []
    actor.weight_updater = SimpleNamespace(
        connect_rollout_engines=lambda *args, **kwargs: (
            calls.append((args, kwargs)),
            setattr(actor.weight_updater, "rollout_engines", args[0]),
        ),
        update_weights=lambda: calls.append("update"),
        weight_version=0,
    )
    actor.weights_backuper = SimpleNamespace()
    actor._should_update_old_actor_during_weight_sync = lambda: False
    monkeypatch.setattr(actor_module.ray, "get", lambda value: value)
    monkeypatch.setattr(actor_module.dist, "barrier", lambda **_kwargs: None)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: None)
    monkeypatch.setattr(actor_module, "print_memory", lambda *_args, **_kwargs: None)

    actor.update_weights.__wrapped__(actor)

    assert len(calls) == 2
    assert calls[0][0][0] == engines
    assert calls[1] == "update"


@pytest.mark.unit
def test_sdpo_correction_metrics_are_sample_sums_for_train_step_reduction():
    from slime.backends.megatron_utils import loss as loss_module

    rows = [torch.ones(2), torch.ones(4)]
    masks = [torch.ones(2), torch.ones(4)]

    ratio_metrics = loss_module._sdpo_ratio_metrics("ratio", rows, masks, cap=2.0)
    weight_metrics = loss_module._sdpo_weight_metrics("weight", rows, masks)

    # The generic train-step reducer divides by global batch size. Two active
    # samples must therefore contribute 2.0, not one microbatch mean of 1.0.
    assert ratio_metrics["ratio_mean"].item() == pytest.approx(2.0)
    assert ratio_metrics["ratio_p50"].item() == pytest.approx(2.0)
    assert ratio_metrics["ratio_max"].item() == pytest.approx(2.0)
    assert ratio_metrics["ratio_clip_fraction"].item() == pytest.approx(0.0)
    assert ratio_metrics["ratio_ess_fraction"].item() == pytest.approx(2.0)
    assert weight_metrics["weight_mean"].item() == pytest.approx(2.0)
    assert weight_metrics["weight_ess_fraction"].item() == pytest.approx(2.0)


@pytest.mark.unit
def test_sdpo_correction_metrics_contribute_zero_for_padding_only_microbatch():
    from slime.backends.megatron_utils import loss as loss_module

    rows = [torch.tensor([1.5, 3.0])]
    masks = [torch.zeros(2)]

    ratio_metrics = loss_module._sdpo_ratio_metrics("ratio", rows, masks, cap=2.0)
    weight_metrics = loss_module._sdpo_weight_metrics("weight", rows, masks)

    assert set(ratio_metrics) == {
        "ratio_mean",
        "ratio_p50",
        "ratio_p90",
        "ratio_p95",
        "ratio_p99",
        "ratio_max",
        "ratio_clip_fraction",
        "ratio_ess_fraction",
    }
    assert set(weight_metrics) == {
        "weight_mean",
        "weight_p50",
        "weight_p90",
        "weight_p95",
        "weight_p99",
        "weight_max",
        "weight_ess_fraction",
    }
    assert all(metric.item() == pytest.approx(0.0) for metric in ratio_metrics.values())
    assert all(metric.item() == pytest.approx(0.0) for metric in weight_metrics.values())


@pytest.mark.unit
def test_main_divergence_score_uses_only_active_main_loss_tokens():
    actor_module = _import_actor_module()
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(sdpo_alpha=1.0, sdpo_distillation_add_tail=True)
    student = torch.log(torch.tensor([[0.7, 0.2], [0.4, 0.4], [0.8, 0.1]], dtype=torch.float32))
    teacher = torch.log(torch.tensor([[0.6, 0.3], [0.2, 0.6], [0.5, 0.4]], dtype=torch.float32))

    scores, counts = actor._compute_sdpo_main_divergence_scores(
        {"loss_masks": [torch.tensor([1, 0, 1])], "self_distillation_mask": [1]},
        {"sdpo_sdpo_topk_log_probs": [student]},
        {"sdpo_teacher_sdpo_topk_log_probs": [teacher]},
    )

    expected = actor_module.compute_sdpo_topk_token_kl(student[[0, 2]], teacher[[0, 2]], alpha=1.0, add_tail=True)
    assert counts == [2]
    assert scores == pytest.approx([float(expected.mean())])


@pytest.mark.unit
def test_sdpo_representation_loss_does_not_request_log_probs(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("representation SDPO must not request log_probs")
        ),
    )

    args = SimpleNamespace(
        qkv_format="thd",
        sdpo_loss_agg_mode="turn_mean",
        sdpo_distillation_mode="representation",
        sdpo_representation_coef=1.0,
        sdpo_representation_reduction="mean",
    )
    hidden_states = torch.tensor(
        [
            [[9.0, 9.0]],
            [[8.0, 8.0]],
            [[2.0, 0.0]],
            [[0.0, 3.0]],
        ],
        requires_grad=True,
    )
    batch = {
        "unconcat_tokens": [torch.tensor([1, 2, 3, 4])],
        "total_lengths": [4],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1.0, 0.0])],
        "rollout_mask_sums": torch.tensor([2.0]),
        "sdpo_teacher_representations": [torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.bfloat16)],
        "self_distillation_mask": torch.tensor([1.0]),
        "sdpo_loss_weights": torch.tensor([1.0]),
    }

    loss, log = loss_module.sdpo_loss_function(args, batch, hidden_states, lambda x: x.sum())

    assert torch.isfinite(loss)
    assert torch.isfinite(log["sdpo_representation_loss"])


@pytest.mark.unit
def test_sdpo_representation_loss_uses_oprd_normalized_mse_and_masks(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)

    args = SimpleNamespace(
        qkv_format="thd",
        sdpo_loss_agg_mode="turn_mean",
        sdpo_distillation_mode="representation",
        sdpo_representation_coef=3000.0,
        sdpo_representation_reduction="mean",
    )
    hidden_states = torch.tensor(
        [
            [[9.0, 9.0]],
            [[8.0, 8.0]],
            [[2.0, 0.0]],
            [[0.0, 3.0]],
        ],
        requires_grad=True,
    )
    batch = {
        "unconcat_tokens": [torch.tensor([1, 2, 3, 4])],
        "total_lengths": [4],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1.0, 0.0])],
        "rollout_mask_sums": torch.tensor([2.0]),
        "sdpo_teacher_representations": [torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.bfloat16)],
        "self_distillation_mask": torch.tensor([1.0]),
        "sdpo_loss_weights": torch.tensor([2.0]),
    }

    loss, log = loss_module.sdpo_loss_function(args, batch, hidden_states, lambda x: x.sum())
    loss.backward()

    student_norm = torch.nn.functional.normalize(torch.tensor([[2.0, 0.0]]), p=2, dim=-1)
    teacher_norm = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0]]), p=2, dim=-1)
    expected_mse = ((student_norm - teacher_norm) ** 2).mean()
    expected_sum_mse = ((student_norm - teacher_norm) ** 2).sum()
    expected_rep_loss = expected_mse * 2.0 / 2.0
    assert loss.item() == pytest.approx((expected_rep_loss * 3000.0).item())
    assert log["sdpo_loss"].item() == pytest.approx((expected_rep_loss * 3000.0).item())
    assert log["sdpo_representation_loss"].item() == pytest.approx(expected_rep_loss.item())
    assert log["sdpo_representation_weighted_loss"].item() == pytest.approx((expected_rep_loss * 3000.0).item())
    assert "sdpo_representation_coef" not in log
    assert log["sdpo_representation_mse"].item() == pytest.approx((expected_mse / 2.0).item())
    assert log["sdpo_representation_mse_mean"].item() == pytest.approx((expected_mse / 2.0).item())
    assert log["sdpo_representation_mse_sum"].item() == pytest.approx((expected_sum_mse / 2.0).item())
    assert log["sdpo_active_samples"].item() == pytest.approx(1.0)
    assert log["sdpo_active_tokens"].item() == pytest.approx(1.0)
    assert hidden_states.grad is not None
    assert hidden_states.grad[2].abs().sum() > 0
    assert hidden_states.grad[3].abs().sum() == pytest.approx(0.0)


@pytest.mark.unit
def test_sdpo_representation_loss_applies_per_token_sdpo_weights(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)

    args = SimpleNamespace(
        qkv_format="thd",
        sdpo_loss_agg_mode="token_mean",
        sdpo_distillation_mode="representation",
        sdpo_representation_coef=1.0,
        sdpo_representation_reduction="sum",
    )
    hidden_states = torch.tensor(
        [
            [[9.0, 9.0]],
            [[8.0, 8.0]],
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ],
        requires_grad=True,
    )
    teacher = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.bfloat16)
    batch = {
        "unconcat_tokens": [torch.tensor([1, 2, 3, 4])],
        "total_lengths": [4],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1.0, 1.0])],
        "rollout_mask_sums": torch.tensor([2.0]),
        "sdpo_teacher_representations": [teacher],
        "self_distillation_mask": torch.tensor([1.0]),
        "sdpo_loss_weights": torch.tensor([1.0]),
        "sdpo_token_weights": [torch.tensor([0.25, 2.0])],
    }

    loss, log = loss_module.sdpo_loss_function(args, batch, hidden_states, lambda x: x.sum())

    first = ((torch.tensor([1.0, 0.0]) - torch.tensor([0.0, 1.0])) ** 2).sum()
    second = ((torch.tensor([0.0, 1.0]) - torch.tensor([1.0, 0.0])) ** 2).sum()
    expected = first * 0.25 + second * 2.0
    assert loss.item() == pytest.approx(expected.item())
    assert log["sdpo_representation_loss"].item() == pytest.approx(expected.item())


@pytest.mark.unit
def test_sdpo_representation_loss_can_use_hidden_dim_sum_reduction(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)

    args = SimpleNamespace(
        qkv_format="thd",
        sdpo_loss_agg_mode="turn_mean",
        sdpo_distillation_mode="representation",
        sdpo_representation_coef=1.0,
        sdpo_representation_reduction="sum",
    )
    hidden_states = torch.tensor(
        [
            [[9.0, 9.0]],
            [[8.0, 8.0]],
            [[2.0, 0.0]],
            [[0.0, 3.0]],
        ],
        requires_grad=True,
    )
    batch = {
        "unconcat_tokens": [torch.tensor([1, 2, 3, 4])],
        "total_lengths": [4],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1.0, 0.0])],
        "rollout_mask_sums": torch.tensor([2.0]),
        "sdpo_teacher_representations": [torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.bfloat16)],
        "self_distillation_mask": torch.tensor([1.0]),
        "sdpo_loss_weights": torch.tensor([2.0]),
    }

    loss, log = loss_module.sdpo_loss_function(args, batch, hidden_states, lambda x: x.sum())
    loss.backward()

    student_norm = torch.nn.functional.normalize(torch.tensor([[2.0, 0.0]]), p=2, dim=-1)
    teacher_norm = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0]]), p=2, dim=-1)
    expected_mean_mse = ((student_norm - teacher_norm) ** 2).mean()
    expected_sum_mse = ((student_norm - teacher_norm) ** 2).sum()
    expected_rep_loss = expected_sum_mse * 2.0 / 2.0
    assert loss.item() == pytest.approx(expected_rep_loss.item())
    assert log["sdpo_representation_loss"].item() == pytest.approx(expected_rep_loss.item())
    assert log["sdpo_representation_weighted_loss"].item() == pytest.approx(expected_rep_loss.item())
    assert log["sdpo_representation_mse"].item() == pytest.approx((expected_sum_mse / 2.0).item())
    assert log["sdpo_representation_mse_mean"].item() == pytest.approx((expected_mean_mse / 2.0).item())
    assert log["sdpo_representation_mse_sum"].item() == pytest.approx((expected_sum_mse / 2.0).item())
    assert hidden_states.grad is not None
    assert hidden_states.grad[2].abs().sum() > 0


@pytest.mark.unit
def test_loss_function_dispatches_sdpo_loss_and_uses_turn_mean_scaling(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_data_parallel_world_size", lambda with_context_parallel=True: 1)
    monkeypatch.setattr(
        loss_module,
        "sdpo_loss_function",
        lambda *args: (torch.tensor(3.0), {"sdpo_loss": torch.tensor(3.0)}),
    )

    args = SimpleNamespace(
        loss_type="sdpo_loss",
        sdpo_loss_agg_mode="turn_mean",
        calculate_per_token_loss=False,
        qkv_format="thd",
        allgather_cp=False,
        recompute_loss_function=False,
    )
    batch = {
        "loss_masks": [torch.ones(2)],
        "total_lengths": [3],
        "response_lengths": [2],
        "rollout_mask_sums": [torch.tensor(2.0)],
    }

    loss, normalizer, log = loss_module.loss_function(
        args,
        batch,
        num_microbatches=2,
        step_global_batch_size=4,
        logits=torch.zeros(1, 1, 1),
    )

    assert loss.item() == pytest.approx(1.5)
    assert normalizer.item() == pytest.approx(1.0)
    assert log["keys"] == ["sdpo_loss"]


@pytest.mark.unit
def test_actor_step_row_indices_follow_microbatch_schedule():
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    rollout_data = {
        "total_lengths": [1, 1, 1, 1],
        "num_microbatches": [2, 1],
        "micro_batch_indices": [[2], [0, 1], [3]],
    }

    assert actor._sdpo_train_step_row_indices(rollout_data, step_id=0, num_steps_per_rollout=2) == [2, 0, 1]
    assert actor._sdpo_train_step_row_indices(rollout_data, step_id=1, num_steps_per_rollout=2) == [3]
    assert actor._remap_step_micro_batch_indices([2, 0, 1], [[2], [0, 1]]) == [[0], [1, 2]]


@pytest.mark.unit
def test_actor_precompute_injects_teacher_topk_outputs_without_double_slice(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_pipeline_model_parallel_world_size", lambda: 1)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        sdpo_distillation_mode="topk",
        sdpo_full_logit_distillation=True,
        sdpo_distillation_topk=1,
        use_routing_replay=False,
    )
    actor._active_model_tag = "actor"
    actor.weights_backuper = SimpleNamespace(backup_tags=["actor"], backup=lambda tag: None)
    actor._switch_model = lambda tag: setattr(actor, "_active_model_tag", tag)
    actor._sdpo_teacher_model_tag = lambda: "actor"
    actor._build_sdpo_teacher_rollout_data = lambda step_rollout_data: dict(step_rollout_data)
    actor._align_sdpo_teacher_log_probs_to_student_cp = lambda teacher_log_probs, *_: teacher_log_probs

    calls: list[str] = []

    def fake_compute(_data_iterator, _num_microbatches, store_prefix=""):
        calls.append(store_prefix)
        if store_prefix == "sdpo_":
            return {"sdpo_sdpo_topk_indices": [torch.tensor([[1], [0]]), torch.tensor([[2]])]}
        return {
            "sdpo_teacher_log_probs": [torch.tensor([-0.1, -0.2]), torch.tensor([-0.3])],
            "sdpo_teacher_sdpo_topk_log_probs": [torch.tensor([[-1.0], [-2.0]]), torch.tensor([[-3.0]])],
        }

    actor.compute_sdpo_distillation_data = fake_compute
    rollout_data = {
        "tokens": [torch.tensor([1, 11]), torch.tensor([2, 22]), torch.tensor([3, 33, 34])],
        "total_lengths": [2, 2, 3],
        "response_lengths": [1, 1, 2],
        "loss_masks": [torch.tensor([1]), torch.tensor([1]), torch.tensor([1, 1])],
        "sdpo_teacher_prompt_text": ["a", "b", "c"],
        "num_microbatches": [2, 1],
        "global_batch_sizes": [2, 1],
        "micro_batch_indices": [[2], [0], [1]],
    }

    actor._precompute_sdpo_topk_for_train_step(rollout_data, step_id=0, num_steps_per_rollout=2)

    assert calls == ["sdpo_", "sdpo_teacher_"]
    torch.testing.assert_close(rollout_data["sdpo_topk_indices"][2], torch.tensor([[1], [0]]))
    torch.testing.assert_close(rollout_data["sdpo_topk_indices"][0], torch.tensor([[2]]))
    torch.testing.assert_close(rollout_data["sdpo_teacher_log_probs"][2], torch.tensor([-0.1, -0.2]))
    torch.testing.assert_close(rollout_data["sdpo_teacher_log_probs"][0], torch.tensor([-0.3]))
    torch.testing.assert_close(rollout_data["sdpo_teacher_topk_log_probs"][2], torch.tensor([[-1.0], [-2.0]]))


@pytest.mark.unit
def test_actor_sdpo_token_weights_normalize_by_uid_mean(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    monkeypatch.setattr(actor_module.dist, "is_available", lambda: False)
    monkeypatch.setattr(actor_module.dist, "is_initialized", lambda: False)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(sdpo_token_weights=True, sdpo_token_weights_power=1.0)
    rollout_data = {
        "response_lengths": [2, 1],
        "loss_masks": [torch.tensor([1.0, 1.0]), torch.tensor([1.0])],
        "self_distillation_mask": [1.0, 1.0],
        "sdpo_metadata": [{"uid": "task-a"}, {"uid": "task-a"}],
    }
    positive = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0]]),
    ]
    contrast = [
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[-1.0, 0.0]]),
    ]

    weights, metrics = actor._compute_sdpo_token_weights(rollout_data, positive, contrast)

    torch.testing.assert_close(weights[0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(weights[1], torch.tensor([2.0]))
    assert metrics["self_distillation/token_weights_active_mean"] == pytest.approx(1.0)
    assert metrics["self_distillation/token_weights_fallback_uid_count"] == pytest.approx(0.0)
    assert metrics["self_distillation/token_weights_power"] == pytest.approx(1.0)
    prefix = actor_module.SDPO_TOKEN_WEIGHT_MONITOR_PREFIX
    assert metrics[f"{prefix}/power"] == pytest.approx(1.0)
    assert metrics[f"{prefix}/fallback_uid_count"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/disagreement/active_token_count"] == pytest.approx(3.0)
    assert metrics[f"{prefix}/disagreement/active_mean"] == pytest.approx(1.0)
    assert metrics[f"{prefix}/disagreement/active_min"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/disagreement/active_max"] == pytest.approx(2.0)
    assert metrics[f"{prefix}/disagreement/active_p50"] == pytest.approx(1.0)
    assert metrics[f"{prefix}/disagreement/token_frac_lt_0_01"] == pytest.approx(1 / 3)
    assert metrics[f"{prefix}/disagreement/token_frac_ge_1"] == pytest.approx(2 / 3)
    assert metrics[f"{prefix}/weights/active_token_count"] == pytest.approx(3.0)
    assert metrics[f"{prefix}/weights/active_mean"] == pytest.approx(1.0)
    assert metrics[f"{prefix}/weights/active_min"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/weights/active_max"] == pytest.approx(2.0)
    assert metrics[f"{prefix}/weights/token_frac_lt_0_25"] == pytest.approx(1 / 3)
    assert metrics[f"{prefix}/weights/token_frac_0_5_to_1"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/weights/token_frac_1_to_2"] == pytest.approx(1 / 3)
    assert metrics[f"{prefix}/weights/token_frac_2_to_4"] == pytest.approx(1 / 3)
    assert metrics[f"{prefix}/weights/mass_frac_lt_0_25"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/weights/mass_frac_0_5_to_1"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/weights/mass_frac_1_to_2"] == pytest.approx(1 / 3)
    assert metrics[f"{prefix}/weights/mass_frac_2_to_4"] == pytest.approx(2 / 3)


@pytest.mark.unit
def test_actor_sdpo_token_weights_negative_power_inverts_disagreement(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    monkeypatch.setattr(actor_module.dist, "is_available", lambda: False)
    monkeypatch.setattr(actor_module.dist, "is_initialized", lambda: False)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(sdpo_token_weights=True, sdpo_token_weights_power=-1.0)
    rollout_data = {
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1.0, 1.0])],
        "self_distillation_mask": [1.0],
        "sdpo_metadata": [{"uid": "task-a"}],
    }
    positive = [torch.tensor([[1.0, 0.0], [1.0, 0.0]])]
    contrast = [torch.tensor([[1.0, 0.0], [-1.0, 0.0]])]

    weights, metrics = actor._compute_sdpo_token_weights(rollout_data, positive, contrast)

    assert torch.isfinite(weights[0]).all()
    assert weights[0][0] > weights[0][1]
    assert weights[0].mean().item() == pytest.approx(1.0)
    assert weights[0][0].item() == pytest.approx(2.0, rel=1e-5)
    assert weights[0][1].item() < 1e-5
    assert metrics["self_distillation/token_weights_power"] == pytest.approx(-1.0)
    assert metrics["self_distillation/token_weights_fallback_uid_count"] == pytest.approx(0.0)


@pytest.mark.unit
def test_actor_sdpo_token_weights_cap_preserves_uid_mean(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    monkeypatch.setattr(actor_module.dist, "is_available", lambda: False)
    monkeypatch.setattr(actor_module.dist, "is_initialized", lambda: False)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(sdpo_token_weights=True, sdpo_token_weights_power=1.0, sdpo_token_weights_max=2.0)
    rollout_data = {
        "response_lengths": [3],
        "loss_masks": [torch.ones(3)],
        "self_distillation_mask": [1.0],
        "sdpo_metadata": [{"uid": "task-a"}],
    }
    positive = [torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])]
    contrast = [
        torch.tensor(
            [
                [0.95, math.sqrt(1.0 - 0.95**2)],
                [0.95, math.sqrt(1.0 - 0.95**2)],
                [-0.4, math.sqrt(1.0 - (-0.4) ** 2)],
            ]
        )
    ]

    weights, metrics = actor._compute_sdpo_token_weights(rollout_data, positive, contrast)

    torch.testing.assert_close(weights[0], torch.tensor([0.5, 0.5, 2.0]), atol=1e-5, rtol=1e-5)
    assert metrics["self_distillation/token_weights_active_mean"] == pytest.approx(1.0)
    assert metrics["self_distillation/token_weights_active_max"] == pytest.approx(2.0)
    assert metrics["self_distillation/token_weights_max"] == pytest.approx(2.0)
    prefix = actor_module.SDPO_TOKEN_WEIGHT_MONITOR_PREFIX
    assert metrics[f"{prefix}/cap/enabled"] == pytest.approx(1.0)
    assert metrics[f"{prefix}/cap/max"] == pytest.approx(2.0)
    assert metrics[f"{prefix}/cap/hit_token_fraction"] == pytest.approx(1 / 3)
    assert metrics[f"{prefix}/cap/preclip_mass_fraction"] == pytest.approx(14 / 15, rel=1e-5)
    assert metrics[f"{prefix}/cap/postclip_mass_fraction"] == pytest.approx(2 / 3, rel=1e-5)
    assert metrics[f"{prefix}/cap/lambda_min"] == pytest.approx(5.0, rel=1e-5)
    assert metrics[f"{prefix}/cap/lambda_mean"] == pytest.approx(5.0, rel=1e-5)
    assert metrics[f"{prefix}/cap/lambda_max"] == pytest.approx(5.0, rel=1e-5)
    assert metrics[f"{prefix}/cap/infeasible_uid_count"] == pytest.approx(0.0)


@pytest.mark.unit
def test_actor_sdpo_token_weights_cap_infeasible_uid_falls_back_to_ones(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    monkeypatch.setattr(actor_module.dist, "is_available", lambda: False)
    monkeypatch.setattr(actor_module.dist, "is_initialized", lambda: False)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(sdpo_token_weights=True, sdpo_token_weights_power=1.0, sdpo_token_weights_max=2.0)
    rollout_data = {
        "response_lengths": [3],
        "loss_masks": [torch.ones(3)],
        "self_distillation_mask": [1.0],
        "sdpo_metadata": [{"uid": "task-a"}],
    }
    positive = [torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])]
    contrast = [torch.tensor([[1.0, 0.0], [1.0, 0.0], [-0.5, math.sqrt(1.0 - (-0.5) ** 2)]])]

    weights, metrics = actor._compute_sdpo_token_weights(rollout_data, positive, contrast)

    torch.testing.assert_close(weights[0], torch.ones(3))
    assert metrics["self_distillation/token_weights_active_mean"] == pytest.approx(1.0)
    assert metrics["self_distillation/token_weights_active_max"] == pytest.approx(1.0)
    prefix = actor_module.SDPO_TOKEN_WEIGHT_MONITOR_PREFIX
    assert metrics[f"{prefix}/cap/enabled"] == pytest.approx(1.0)
    assert metrics[f"{prefix}/cap/hit_token_fraction"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/cap/infeasible_uid_count"] == pytest.approx(1.0)


@pytest.mark.unit
def test_actor_sdpo_token_weight_distributed_monitor_payloads_are_small_stats(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    monkeypatch.setattr(actor_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(actor_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(actor_module.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: None)

    payloads = []

    def fake_all_gather_object(output, obj, group=None):
        del group
        payloads.append(obj)
        output[0] = obj

    monkeypatch.setattr(actor_module.dist, "all_gather_object", fake_all_gather_object)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(sdpo_token_weights=True, sdpo_token_weights_power=1.0, sdpo_token_weights_max=2.0)
    rollout_data = {
        "response_lengths": [3],
        "loss_masks": [torch.ones(3)],
        "self_distillation_mask": [1.0],
        "sdpo_metadata": [{"uid": "task-a"}],
    }
    positive = [torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])]
    contrast = [
        torch.tensor(
            [
                [0.95, math.sqrt(1.0 - 0.95**2)],
                [0.95, math.sqrt(1.0 - 0.95**2)],
                [-0.4, math.sqrt(1.0 - (-0.4) ** 2)],
            ]
        )
    ]

    actor._compute_sdpo_token_weights(rollout_data, positive, contrast)

    def contains_tensor(value):
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, dict):
            return any(contains_tensor(item) for item in value.values())
        if isinstance(value, (list, tuple, set)):
            return any(contains_tensor(item) for item in value)
        return False

    assert payloads
    assert not any(contains_tensor(payload) for payload in payloads)


@pytest.mark.unit
def test_actor_sdpo_token_weight_metric_scalars_copy_for_interleaved_logging():
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    prefix = actor_module.SDPO_TOKEN_WEIGHT_MONITOR_PREFIX
    source = {
        f"{prefix}/weights/active_mean": torch.tensor(1.25),
        "self_distillation/token_weights_active_mean": torch.tensor(1.25),
        "sdpo_token_weights": [torch.tensor([1.0])],
        "unrelated": torch.tensor(9.0),
    }
    target = {}

    MegatronTrainRayActor._copy_sdpo_token_weight_metric_scalars(source, target)

    assert set(target) == {f"{prefix}/weights/active_mean", "self_distillation/token_weights_active_mean"}
    assert target[f"{prefix}/weights/active_mean"].item() == pytest.approx(1.25)
    assert target["self_distillation/token_weights_active_mean"].item() == pytest.approx(1.25)


@pytest.mark.unit
def test_actor_precomputes_interleaved_token_weights_after_all_steps(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    calls: list[tuple[str, int | None]] = []

    def fake_precompute(rollout_data, *, step_id, num_steps_per_rollout, finalize_token_weights=True):
        assert num_steps_per_rollout == 2
        assert finalize_token_weights is False
        calls.append(("step", step_id))
        rows = rollout_data.setdefault("sdpo_teacher_representations", [None, None])
        rows[step_id] = torch.ones(1, 2) * float(step_id + 1)

    def fake_ensure(rollout_data):
        assert all(isinstance(row, torch.Tensor) for row in rollout_data["sdpo_teacher_representations"])
        calls.append(("ensure", None))
        rollout_data["sdpo_token_weights"] = [torch.ones(1), torch.ones(1)]

    monkeypatch.setattr(actor, "_precompute_sdpo_topk_for_train_step", fake_precompute)
    monkeypatch.setattr(actor, "_ensure_sdpo_token_weights", fake_ensure)
    rollout_data = {"total_lengths": [1, 1]}

    actor._precompute_sdpo_topk_for_all_train_steps(rollout_data, num_steps_per_rollout=2)

    assert calls == [("step", 0), ("step", 1), ("ensure", None)]
    assert actor._sdpo_token_weights_complete(rollout_data)


@pytest.mark.unit
def test_actor_sdpo_token_weights_power_zero_returns_ones(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    monkeypatch.setattr(actor_module.dist, "is_available", lambda: False)
    monkeypatch.setattr(actor_module.dist, "is_initialized", lambda: False)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(sdpo_token_weights=True, sdpo_token_weights_power=0.0)
    rollout_data = {
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1.0, 1.0])],
        "self_distillation_mask": [1.0],
        "sdpo_metadata": [{"uid": "task-a"}],
    }
    positive = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
    contrast = [torch.tensor([[1.0, 0.0], [-1.0, 0.0]])]

    weights, metrics = actor._compute_sdpo_token_weights(rollout_data, positive, contrast)

    torch.testing.assert_close(weights[0], torch.ones(2))
    assert metrics["self_distillation/token_weights_active_mean"] == pytest.approx(1.0)
    assert metrics["self_distillation/token_weights_power"] == pytest.approx(0.0)
    prefix = actor_module.SDPO_TOKEN_WEIGHT_MONITOR_PREFIX
    assert metrics[f"{prefix}/disagreement/token_frac_lt_0_01"] == pytest.approx(1 / 2)
    assert metrics[f"{prefix}/disagreement/token_frac_ge_1"] == pytest.approx(1 / 2)
    assert metrics[f"{prefix}/weights/token_frac_1_to_2"] == pytest.approx(1.0)
    assert metrics[f"{prefix}/weights/mass_frac_1_to_2"] == pytest.approx(1.0)


@pytest.mark.unit
def test_actor_grpo_policy_token_weights_use_generation_decision_states(monkeypatch):
    actor_module = _import_actor_module()
    monkeypatch.setattr(actor_module.dist, "is_available", lambda: False)
    monkeypatch.setattr(actor_module.dist, "is_initialized", lambda: False)
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(grpo_token_weights=True, grpo_token_weights_power=1.0, grpo_token_weights_max=8.0)
    rollout_data = {"loss_masks": [torch.ones(3)]}
    positive_response = [torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])]
    contrast_response = [torch.tensor([[0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]])]

    weights, _ = actor._compute_grpo_policy_token_weights(
        rollout_data,
        positive_response,
        [torch.tensor([1.0, 0.0])],
        contrast_response,
        [torch.tensor([0.8, 0.6])],
    )

    # Decision states are [prompt_last, response[:-1]], with disagreements [0.2, 1, 2].
    torch.testing.assert_close(weights[0], torch.tensor([0.1875, 0.9375, 1.8750]))


@pytest.mark.unit
def test_actor_grpo_policy_token_weights_normalize_cap_and_fallback(monkeypatch):
    actor_module = _import_actor_module()
    monkeypatch.setattr(actor_module.dist, "is_available", lambda: False)
    monkeypatch.setattr(actor_module.dist, "is_initialized", lambda: False)
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(grpo_token_weights=True, grpo_token_weights_power=1.0, grpo_token_weights_max=1.5)
    rollout_data = {"loss_masks": [torch.ones(3), torch.ones(2), torch.ones(2)]}
    positive = [torch.tensor([[1.0, 0.0]] * length) for length in (3, 2, 2)]
    contrast = [
        torch.tensor([[0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
    ]
    positive_prompt = [torch.tensor([1.0, 0.0])] * 3
    contrast_prompt = [torch.tensor([0.8, 0.6]), torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]

    weights, metrics = actor._compute_grpo_policy_token_weights(
        rollout_data, positive, positive_prompt, contrast, contrast_prompt
    )

    assert weights[0].mean().item() == pytest.approx(1.0, abs=1e-5)
    assert weights[0].max().item() == pytest.approx(1.5, abs=1e-5)
    torch.testing.assert_close(weights[1], torch.ones(2))  # zero disagreement
    torch.testing.assert_close(weights[2], torch.ones(2))  # one nonzero score cannot preserve mean under cap=1.5
    assert metrics["grpo_token_weights/fallback_zero_row_count"] == pytest.approx(1.0)
    assert metrics["grpo_token_weights/cap_infeasible_row_count"] == pytest.approx(1.0)


@pytest.mark.unit
def test_actor_grpo_policy_token_weights_compute_two_ref_forwards_once(monkeypatch):
    actor_module = _import_actor_module()
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(grpo_token_weights=True, use_routing_replay=False)
    actor._active_model_tag = "actor"
    actor.weights_backuper = SimpleNamespace(backup_tags=["actor", "ref"])
    actor._switch_model = lambda tag: setattr(actor, "_active_model_tag", tag)
    actor._build_grpo_token_weight_context_rollout_data = lambda data, field, rows: {
        "num_microbatches": [1],
        "field": field,
    }
    monkeypatch.setattr(actor_module, "get_data_iterator", lambda data: [data])
    calls = []

    def fake_compute(iterator, _microbatches, store_prefix="", collect_decision_states=False):
        assert collect_decision_states is True
        calls.append((iterator[0]["field"], store_prefix, actor._active_model_tag))
        return {
            f"{store_prefix}representations": [torch.tensor([[1.0, 0.0], [1.0, 0.0]])],
            f"{store_prefix}prompt_last_representations": [torch.tensor([1.0, 0.0])],
        }

    actor.compute_sdpo_representation_data = fake_compute
    rollout_data = {
        "total_lengths": [3, 3],
        "response_lengths": [2, 2],
        "loss_masks": [torch.ones(2), torch.ones(2)],
        "grpo_token_weight_positive_prompt_text": ["positive", None],
        "grpo_token_weight_contrast_prompt_text": ["contrast", None],
    }

    actor._ensure_grpo_policy_token_weights(rollout_data)
    actor._ensure_grpo_policy_token_weights(rollout_data)

    assert calls == [
        ("grpo_token_weight_positive_prompt_text", "grpo_token_weight_positive_", "ref"),
        ("grpo_token_weight_contrast_prompt_text", "grpo_token_weight_contrast_", "ref"),
    ]
    torch.testing.assert_close(rollout_data["grpo_policy_token_weights"][1], torch.ones(2))
    assert actor._active_model_tag == "actor"


@pytest.mark.unit
def test_actor_grpo_policy_token_weights_default_off_does_nothing():
    actor_module = _import_actor_module()
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(grpo_token_weights=False)
    actor.compute_sdpo_representation_data = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("default-off must not run a representation forward")
    )
    rollout_data = {"total_lengths": [2], "loss_masks": [torch.ones(1)]}
    actor._ensure_grpo_policy_token_weights(rollout_data)
    assert "grpo_policy_token_weights" not in rollout_data


@pytest.mark.unit
def test_actor_grpo_policy_token_weights_rejects_malformed_cached_rows():
    actor_module = _import_actor_module()
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(grpo_token_weights=True)
    rollout_data = {
        "total_lengths": [3, 3],
        "response_lengths": [2, 2],
        "grpo_policy_token_weights": [torch.ones(3), torch.ones(1)],
    }

    assert actor._grpo_policy_token_weights_complete(rollout_data) is False


@pytest.mark.unit
def test_actor_sdpo_token_weights_skip_zero_loss_contrast_rows():
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    rollout_data = {
        "sdpo_token_weight_contrast_prompt_text": ["contrast-a", "contrast-b"],
        "self_distillation_mask": [1.0, 1.0],
        "loss_masks": [torch.tensor([0.0]), torch.tensor([1.0])],
    }

    assert actor._sdpo_token_weight_contrast_row_indices(rollout_data) == [1]


@pytest.mark.unit
def test_actor_sdpo_token_weights_recomputes_partial_placeholder_list(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    monkeypatch.setattr(actor_module.dist, "is_available", lambda: False)
    monkeypatch.setattr(actor_module.dist, "is_initialized", lambda: False)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(sdpo_token_weights=True, sdpo_token_weights_power=1.0, use_routing_replay=False)
    actor._active_model_tag = "actor"
    actor._switch_model = lambda tag: setattr(actor, "_active_model_tag", tag)
    actor._sdpo_teacher_model_tag = lambda: "ref"
    actor._build_sdpo_token_weight_contrast_rollout_data = lambda rollout_data, row_indices: {
        "num_microbatches": [1],
    }
    calls: list[str] = []

    def fake_compute(_data_iterator, _num_microbatches, store_prefix=""):
        calls.append(store_prefix)
        return {"sdpo_token_weight_contrast_representations": [torch.tensor([[0.0, 1.0]])]}

    actor.compute_sdpo_representation_data = fake_compute
    monkeypatch.setattr(actor_module, "get_data_iterator", lambda _rollout_data: ["iterator"])
    rollout_data = {
        "total_lengths": [2],
        "response_lengths": [1],
        "loss_masks": [torch.tensor([1.0])],
        "self_distillation_mask": [1.0],
        "sdpo_metadata": [{"uid": "task-a"}],
        "sdpo_teacher_representations": [torch.tensor([[1.0, 0.0]])],
        "sdpo_token_weight_contrast_prompt_text": ["contrast"],
        "sdpo_token_weights": [None],
    }

    actor._ensure_sdpo_token_weights(rollout_data)

    assert calls == ["sdpo_token_weight_contrast_"]
    assert isinstance(rollout_data["sdpo_token_weights"][0], torch.Tensor)
    torch.testing.assert_close(rollout_data["sdpo_token_weights"][0], torch.ones(1))


@pytest.mark.unit
def test_actor_compute_log_prob_student_source_uses_single_combined_forward(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        sdpo_token_weights=True,
        sdpo_token_weight_source="student",
        use_rollout_entropy=False,
    )
    actor.model = ["model"]
    calls = []

    def fake_forward_only(callback, args, model, data_iterator, num_microbatches, **kwargs):
        calls.append((callback, args, model, data_iterator, num_microbatches, kwargs))
        return {
            "log_probs": [torch.tensor([-0.2])],
            "sdpo_student_representations": [torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16)],
        }

    monkeypatch.setattr(actor_module, "forward_only", fake_forward_only)
    monkeypatch.setattr(actor_module, "timer", lambda *_args, **_kwargs: nullcontext())

    result = actor.compute_log_prob(
        ["iterator"],
        [1],
        collect_sdpo_student_representations=True,
    )

    assert len(calls) == 1
    callback, args, model, data_iterator, num_microbatches, kwargs = calls[0]
    assert callback is actor_module.get_log_probs_entropy_and_sdpo_student_representations
    assert (args, model, data_iterator, num_microbatches) == (actor.args, actor.model, ["iterator"], [1])
    assert kwargs["capture_hidden_states"] is True
    assert set(result) == {"log_probs", "sdpo_student_representations"}


@pytest.mark.unit
def test_actor_compute_log_prob_teacher_contrast_keeps_log_prob_only_forward(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(use_rollout_entropy=False)
    actor.model = ["model"]
    calls = []

    def fake_forward_only(callback, *_args, **kwargs):
        calls.append((callback, kwargs))
        return {"log_probs": [torch.tensor([-0.2])]}

    monkeypatch.setattr(actor_module, "forward_only", fake_forward_only)
    monkeypatch.setattr(actor_module, "timer", lambda *_args, **_kwargs: nullcontext())

    actor.compute_log_prob(["iterator"], [1])

    assert calls == [
        (
            actor_module.get_log_probs_and_entropy,
            {
                "store_prefix": "",
                "with_entropy": None,
                "temperature": None,
                "capture_hidden_states": False,
            },
        )
    ]


@pytest.mark.unit
def test_student_representation_callback_returns_detached_cpu_rows(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (torch.empty(0), {"log_probs": [torch.tensor([-0.2])]}),
    )
    hidden_states = torch.ones(2, 1, 3, requires_grad=True)

    _, output = loss_module.get_log_probs_entropy_and_sdpo_student_representations(
        torch.zeros(1, 2, 4),
        hidden_states=hidden_states,
        args=SimpleNamespace(qkv_format="thd"),
        unconcat_tokens=[torch.tensor([1, 2])],
        total_lengths=[2],
        response_lengths=[1],
    )

    representation = output["sdpo_student_representations"][0]
    assert representation.device.type == "cpu"
    assert representation.dtype == torch.bfloat16
    assert representation.requires_grad is False


@pytest.mark.unit
def test_actor_student_token_weights_use_student_rows_without_teacher_contrast_forward(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        sdpo_token_weights=True,
        sdpo_token_weights_power=1.0,
        sdpo_token_weight_source="student",
    )
    student_rows = [torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16)]
    captured = {}
    actor._build_sdpo_token_weight_contrast_rollout_data = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("student source must not build contrast teacher data")
    )
    actor.compute_sdpo_representation_data = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("student source must not forward the contrast teacher")
    )

    def fake_compute(rollout_data, positive, contrast):
        captured["args"] = (rollout_data, positive, contrast)
        return [torch.ones(1)], {"self_distillation/token_weights_active_mean": 1.0}

    actor._compute_sdpo_token_weights = fake_compute
    rollout_data = {
        "total_lengths": [2],
        "sdpo_teacher_representations": [torch.tensor([[1.0, 0.0]])],
        "sdpo_student_representations": student_rows,
    }

    actor._ensure_sdpo_token_weights(rollout_data)

    assert captured["args"][2] is student_rows
    torch.testing.assert_close(rollout_data["sdpo_token_weights"][0], torch.ones(1))


@pytest.mark.unit
@pytest.mark.parametrize("student_rows", [None, [], [torch.ones(1, 2), torch.ones(1, 2)]])
def test_actor_student_token_weights_require_row_aligned_student_representations(student_rows):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(sdpo_token_weights=True, sdpo_token_weight_source="student")
    rollout_data = {
        "total_lengths": [2],
        "sdpo_teacher_representations": [torch.ones(1, 2)],
        "sdpo_student_representations": student_rows,
    }

    with pytest.raises(ValueError, match="requires row-aligned sdpo_student_representations"):
        actor._ensure_sdpo_token_weights(rollout_data)


@pytest.mark.unit
def test_actor_sdpo_token_weights_fallback_to_ones_when_uid_disagreement_is_zero(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    monkeypatch.setattr(actor_module.dist, "is_available", lambda: False)
    monkeypatch.setattr(actor_module.dist, "is_initialized", lambda: False)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(sdpo_token_weights=True, sdpo_token_weights_power=1.0)
    rollout_data = {
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1.0, 1.0])],
        "self_distillation_mask": [1.0],
        "sdpo_metadata": [{"uid": "task-a"}],
    }
    positive = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
    contrast = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]

    weights, metrics = actor._compute_sdpo_token_weights(rollout_data, positive, contrast)

    torch.testing.assert_close(weights[0], torch.ones(2))
    assert metrics["self_distillation/token_weights_active_mean"] == pytest.approx(1.0)
    assert metrics["self_distillation/token_weights_fallback_uid_count"] == pytest.approx(1.0)
    prefix = actor_module.SDPO_TOKEN_WEIGHT_MONITOR_PREFIX
    assert metrics[f"{prefix}/fallback_uid_count"] == pytest.approx(1.0)
    assert metrics[f"{prefix}/disagreement/token_frac_lt_0_01"] == pytest.approx(1.0)
    assert metrics[f"{prefix}/weights/token_frac_1_to_2"] == pytest.approx(1.0)
    assert metrics[f"{prefix}/weights/mass_frac_1_to_2"] == pytest.approx(1.0)


@pytest.mark.unit
def test_actor_ensure_sdpo_teacher_log_probs_uses_combined_forward_for_token_weights(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_pipeline_model_parallel_world_size", lambda: 1)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        sdpo_token_weights=True,
        sdpo_distillation_mode="topk",
        sdpo_full_logit_distillation=True,
        sdpo_distillation_topk=1,
        use_routing_replay=False,
    )
    actor._active_model_tag = "actor"
    actor._switch_model = lambda tag: setattr(actor, "_active_model_tag", tag)
    actor._sdpo_teacher_model_tag = lambda: "ref"
    actor._build_sdpo_teacher_rollout_data = lambda rollout_data: dict(rollout_data, num_microbatches=[1])
    actor._align_sdpo_teacher_log_probs_to_student_cp = lambda rows, _teacher, _rollout: rows
    actor.compute_sdpo_distillation_data = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("teacher C+ path must use combined distillation+representation forward")
    )

    calls: list[str] = []

    def fake_combined(_data_iterator, _num_microbatches, store_prefix=""):
        calls.append(store_prefix)
        return {
            "sdpo_teacher_log_probs": [torch.tensor([-0.1])],
            "sdpo_teacher_sdpo_topk_log_probs": [torch.tensor([[-1.0]])],
            "sdpo_teacher_representations": [torch.tensor([[1.0, 0.0]])],
        }

    actor.compute_sdpo_distillation_and_representation_data = fake_combined
    rollout_data = {
        "tokens": [torch.tensor([1, 2])],
        "total_lengths": [2],
        "response_lengths": [1],
        "loss_masks": [torch.tensor([1.0])],
        "sdpo_teacher_prompt_text": ["teacher"],
        "sdpo_topk_indices": [torch.tensor([[0]])],
        "num_microbatches": [1],
        "global_batch_sizes": [1],
        "micro_batch_indices": [[0]],
    }

    actor._ensure_sdpo_teacher_log_probs(rollout_data)

    assert calls == ["sdpo_teacher_"]
    torch.testing.assert_close(rollout_data["sdpo_teacher_representations"][0], torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(rollout_data["sdpo_teacher_topk_log_probs"][0], torch.tensor([[-1.0]]))


@pytest.mark.unit
def test_actor_combined_sdpo_forward_requests_hidden_capture(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace()
    actor.model = ["model"]
    captured = {}

    def fake_forward_only(callback, args, model, data_iterator, num_microbatches, **kwargs):
        captured["callback"] = callback
        captured["args"] = args
        captured["model"] = model
        captured["data_iterator"] = data_iterator
        captured["num_microbatches"] = num_microbatches
        captured["kwargs"] = kwargs
        return {"sdpo_teacher_log_probs": [torch.tensor([-0.1])]}

    monkeypatch.setattr(actor_module, "forward_only", fake_forward_only)

    result = actor.compute_sdpo_distillation_and_representation_data(["it"], [1], store_prefix="sdpo_teacher_")

    assert result["sdpo_teacher_log_probs"][0].item() == pytest.approx(-0.1)
    assert captured["callback"] is actor_module.get_sdpo_distillation_and_representation_tensors
    assert captured["kwargs"]["store_prefix"] == "sdpo_teacher_"
    assert captured["kwargs"]["capture_hidden_states"] is True


@pytest.mark.unit
def test_actor_ensure_sdpo_teacher_representations_skips_topk_and_stores_cpu_bf16(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: 1)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(sdpo_distillation_mode="representation", use_routing_replay=False)
    actor._active_model_tag = "actor"
    actor.weights_backuper = SimpleNamespace(backup_tags=["actor", "ref"])
    actor._switch_model = lambda tag: setattr(actor, "_active_model_tag", tag)
    actor._sdpo_teacher_model_tag = lambda: "ref"
    actor._build_sdpo_teacher_rollout_data = lambda rollout_data: dict(rollout_data, num_microbatches=[1])
    actor.compute_sdpo_distillation_data = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("representation mode must not compute top-k/logprob distillation data")
    )

    calls: list[str] = []

    def fake_compute(_data_iterator, _num_microbatches, store_prefix=""):
        calls.append(store_prefix)
        return {
            "sdpo_teacher_representations": [
                torch.ones(2, 4, dtype=torch.float32),
                torch.zeros(1, 4, dtype=torch.float32),
            ]
        }

    actor.compute_sdpo_representation_data = fake_compute
    rollout_data = {
        "tokens": [torch.tensor([1, 11, 12]), torch.tensor([2, 22])],
        "total_lengths": [3, 2],
        "response_lengths": [2, 1],
        "loss_masks": [torch.tensor([1, 1]), torch.tensor([1])],
        "sdpo_teacher_prompt_text": ["a", "b"],
        "num_microbatches": [1],
        "global_batch_sizes": [2],
        "micro_batch_indices": [[0, 1]],
    }

    actor._ensure_sdpo_teacher_representations(rollout_data)

    assert calls == ["sdpo_teacher_"]
    assert actor._active_model_tag == "actor"
    assert rollout_data["sdpo_teacher_representations"][0].device.type == "cpu"
    assert rollout_data["sdpo_teacher_representations"][0].dtype == torch.bfloat16
    torch.testing.assert_close(rollout_data["sdpo_teacher_representations"][0], torch.ones(2, 4).bfloat16())
    assert rollout_data["sdpo_teacher_representation_chunk_count_per_rank"].item() == pytest.approx(1.0)
    assert rollout_data["sdpo_teacher_representation_expanded_rows_per_rank"].item() == pytest.approx(2.0)
    assert rollout_data["sdpo_teacher_representation_tokens_per_rank"].item() == pytest.approx(5.0)
    assert rollout_data["sdpo_teacher_representation_microbatch_count_per_rank"].item() == pytest.approx(1.0)
    assert rollout_data["sdpo_teacher_representation_max_microbatch_tokens_per_rank"].item() == pytest.approx(5.0)


@pytest.mark.unit
def test_actor_ensure_sdpo_teacher_representations_averages_success_ensembles(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: 1)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(sdpo_distillation_mode="representation", use_routing_replay=False)
    actor._active_model_tag = "actor"
    actor.weights_backuper = SimpleNamespace(backup_tags=["actor", "ref"])
    actor._switch_model = lambda tag: setattr(actor, "_active_model_tag", tag)
    actor._sdpo_teacher_model_tag = lambda: "ref"
    actor.compute_sdpo_distillation_data = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("representation mode must not compute top-k/logprob distillation data")
    )

    captured_prompts: list[str] = []

    def fake_build_teacher_rollout_data(expanded_rollout_data):
        captured_prompts.extend(expanded_rollout_data["sdpo_teacher_prompt_text"])
        row_count = len(expanded_rollout_data["tokens"])
        return dict(
            expanded_rollout_data,
            num_microbatches=[1],
            global_batch_sizes=[row_count],
            micro_batch_indices=[list(range(row_count))],
            sdpo_teacher_prompt_token_lengths=[10, 20, 30],
            sdpo_teacher_prompt_token_lengths_raw=[11, 21, 31],
            sdpo_teacher_prompt_truncated=[0.0, 1.0, 0.0],
            sdpo_teacher_prompt_char_lengths=[100, 200, 300],
        )

    actor._build_sdpo_teacher_rollout_data = fake_build_teacher_rollout_data

    def fake_compute(_data_iterator, _num_microbatches, store_prefix=""):
        assert store_prefix == "sdpo_teacher_"
        return {
            "sdpo_teacher_representations": [
                torch.ones(2, 4, dtype=torch.float32),
                torch.full((2, 4), 3.0, dtype=torch.float32),
                torch.full((1, 4), 7.0, dtype=torch.float32),
            ],
        }

    actor.compute_sdpo_representation_data = fake_compute
    rollout_data = {
        "tokens": [torch.tensor([1, 11, 12]), torch.tensor([2, 22])],
        "total_lengths": [3, 2],
        "response_lengths": [2, 1],
        "loss_masks": [torch.tensor([1, 1]), torch.tensor([1])],
        "sdpo_teacher_prompt_text": ["a", "b"],
        "sdpo_teacher_prompt_texts": [["a-success-1", "a-success-2"], None],
        "sdpo_teacher_messages": [
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ],
        "sdpo_teacher_messages_list": [
            [[{"role": "user", "content": "a-success-1"}], [{"role": "user", "content": "a-success-2"}]],
            None,
        ],
        "num_microbatches": [1],
        "global_batch_sizes": [2],
        "micro_batch_indices": [[0, 1]],
    }

    actor._ensure_sdpo_teacher_representations(rollout_data)

    assert captured_prompts == ["a-success-1", "a-success-2", "b"]
    assert actor._active_model_tag == "actor"
    assert len(rollout_data["sdpo_teacher_representations"]) == 2
    torch.testing.assert_close(
        rollout_data["sdpo_teacher_representations"][0],
        torch.full((2, 4), 2.0, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        rollout_data["sdpo_teacher_representations"][1],
        torch.full((1, 4), 7.0, dtype=torch.bfloat16),
    )
    assert rollout_data["sdpo_teacher_prompt_token_lengths"] == [15.0, 30.0]
    assert rollout_data["sdpo_teacher_prompt_token_lengths_raw"] == [16.0, 31.0]
    assert rollout_data["sdpo_teacher_prompt_truncated"] == [1.0, 0.0]
    assert rollout_data["sdpo_teacher_prompt_char_lengths"] == [150.0, 300.0]
    assert rollout_data["sdpo_teacher_prompt_token_lengths_ensemble_total"] == [30.0, 30.0]
    assert rollout_data["sdpo_teacher_prompt_char_lengths_ensemble_total"] == [300.0, 300.0]
    assert rollout_data["sdpo_teacher_representation_chunk_count_per_rank"].item() == pytest.approx(1.0)
    assert rollout_data["sdpo_teacher_representation_expanded_rows_per_rank"].item() == pytest.approx(3.0)
    assert rollout_data["sdpo_teacher_representation_tokens_per_rank"].item() == pytest.approx(8.0)
    assert rollout_data["sdpo_teacher_representation_microbatch_count_per_rank"].item() == pytest.approx(1.0)
    assert rollout_data["sdpo_teacher_representation_max_microbatch_tokens_per_rank"].item() == pytest.approx(8.0)


@pytest.mark.unit
def test_actor_ensure_sdpo_teacher_representations_chunks_success_ensembles(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: 1)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        sdpo_distillation_mode="representation",
        use_routing_replay=False,
        sdpo_teacher_representation_forward_chunk_rows=2,
    )
    actor._active_model_tag = "actor"
    actor.weights_backuper = SimpleNamespace(backup_tags=["actor", "ref"])
    actor._switch_model = lambda tag: setattr(actor, "_active_model_tag", tag)
    actor._sdpo_teacher_model_tag = lambda: "ref"

    captured_chunks: list[list[str]] = []

    def prompt_value(prompt: str) -> float:
        return float(prompt.rsplit("-", 1)[1])

    def fake_build_teacher_rollout_data(expanded_rollout_data):
        prompts = list(expanded_rollout_data["sdpo_teacher_prompt_text"])
        captured_chunks.append(prompts)
        values = [prompt_value(prompt) for prompt in prompts]
        row_count = len(prompts)
        return dict(
            expanded_rollout_data,
            num_microbatches=[1],
            global_batch_sizes=[row_count],
            micro_batch_indices=[list(range(row_count))],
            sdpo_teacher_prompt_token_lengths=values,
            sdpo_teacher_prompt_token_lengths_raw=[value + 10.0 for value in values],
            sdpo_teacher_prompt_truncated=[1.0 if value > 5.0 else 0.0 for value in values],
            sdpo_teacher_prompt_char_lengths=[value * 10.0 for value in values],
        )

    actor._build_sdpo_teacher_rollout_data = fake_build_teacher_rollout_data

    def fake_compute(_data_iterator, _num_microbatches, store_prefix=""):
        assert store_prefix == "sdpo_teacher_"
        return {
            "sdpo_teacher_representations": [
                torch.full((1, 2), prompt_value(prompt), dtype=torch.float32) for prompt in captured_chunks[-1]
            ]
        }

    actor.compute_sdpo_representation_data = fake_compute
    rollout_data = {
        "tokens": [torch.tensor([1, 11]), torch.tensor([2, 22])],
        "total_lengths": [2, 2],
        "response_lengths": [1, 1],
        "loss_masks": [torch.tensor([1]), torch.tensor([1])],
        "sdpo_teacher_prompt_text": ["a", "b"],
        "sdpo_teacher_prompt_texts": [["a-1", "a-3", "a-5"], ["b-7"]],
        "num_microbatches": [1],
        "global_batch_sizes": [2],
        "micro_batch_indices": [[0, 1]],
    }

    actor._ensure_sdpo_teacher_representations(rollout_data)

    assert captured_chunks == [["a-1", "a-3"], ["a-5", "b-7"]]
    assert actor._active_model_tag == "actor"
    torch.testing.assert_close(
        rollout_data["sdpo_teacher_representations"][0],
        torch.full((1, 2), 3.0, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        rollout_data["sdpo_teacher_representations"][1],
        torch.full((1, 2), 7.0, dtype=torch.bfloat16),
    )
    assert rollout_data["sdpo_teacher_prompt_token_lengths"] == [3.0, 7.0]
    assert rollout_data["sdpo_teacher_prompt_token_lengths_raw"] == [13.0, 17.0]
    assert rollout_data["sdpo_teacher_prompt_truncated"] == [0.0, 1.0]
    assert rollout_data["sdpo_teacher_prompt_char_lengths"] == [30.0, 70.0]
    assert rollout_data["sdpo_teacher_prompt_token_lengths_ensemble_total"] == [9.0, 7.0]
    assert rollout_data["sdpo_teacher_prompt_char_lengths_ensemble_total"] == [90.0, 70.0]
    assert rollout_data["sdpo_teacher_representation_chunk_count_per_rank"].item() == pytest.approx(2.0)
    assert rollout_data["sdpo_teacher_representation_expanded_rows_per_rank"].item() == pytest.approx(4.0)
    assert rollout_data["sdpo_teacher_representation_tokens_per_rank"].item() == pytest.approx(8.0)
    assert rollout_data["sdpo_teacher_representation_microbatch_count_per_rank"].item() == pytest.approx(2.0)
    assert rollout_data["sdpo_teacher_representation_max_microbatch_tokens_per_rank"].item() == pytest.approx(4.0)


@pytest.mark.unit
def test_actor_ensure_sdpo_teacher_representations_chunks_keep_forward_only_order(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: 1)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        sdpo_distillation_mode="representation",
        use_routing_replay=False,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=10,
        sdpo_teacher_representation_max_tokens_per_gpu=None,
        sdpo_teacher_representation_forward_chunk_rows=3,
    )
    actor.train_parallel_config = {"cp_size": 1}
    actor._active_model_tag = "actor"
    actor.weights_backuper = SimpleNamespace(backup_tags=["actor", "ref"])
    actor._switch_model = lambda tag: setattr(actor, "_active_model_tag", tag)
    actor._sdpo_teacher_model_tag = lambda: "ref"

    def fake_build_teacher_rollout_data(expanded_rollout_data):
        teacher_rollout_data = dict(expanded_rollout_data)
        actor._repack_sdpo_teacher_microbatches(teacher_rollout_data)
        return teacher_rollout_data

    actor._build_sdpo_teacher_rollout_data = fake_build_teacher_rollout_data

    def prompt_value(prompt: str) -> float:
        return float(prompt.rsplit("-", 1)[1])

    def fake_compute(_data_iterator, _num_microbatches, store_prefix=""):
        assert store_prefix == "sdpo_teacher_"
        prompts = _data_iterator[0].rollout_data["sdpo_teacher_prompt_text"]
        return {
            "sdpo_teacher_representations": [
                torch.full((1, 1), prompt_value(prompt), dtype=torch.float32) for prompt in prompts
            ]
        }

    actor.compute_sdpo_representation_data = fake_compute
    rollout_data = {
        "tokens": [torch.tensor([1] * 6), torch.tensor([2] * 4)],
        "total_lengths": [6, 4],
        "response_lengths": [1, 1],
        "loss_masks": [torch.tensor([1]), torch.tensor([1])],
        "sdpo_teacher_prompt_text": ["a", "b"],
        "sdpo_teacher_prompt_texts": [["a-1", "a-5", "a-9"], ["b-3"]],
        "num_microbatches": [1],
        "global_batch_sizes": [2],
        "micro_batch_indices": [[0, 1]],
    }

    actor._ensure_sdpo_teacher_representations(rollout_data)

    assert [float(row.item()) for row in rollout_data["sdpo_teacher_representations"]] == [5.0, 3.0]
    assert rollout_data["sdpo_teacher_representation_chunk_count_per_rank"].item() == pytest.approx(2.0)
    assert rollout_data["sdpo_teacher_representation_microbatch_count_per_rank"].item() == pytest.approx(4.0)


@pytest.mark.unit
def test_actor_ensure_sdpo_teacher_representations_keeps_forward_only_output_order(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: 1)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        sdpo_distillation_mode="representation",
        use_routing_replay=False,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=10,
        sdpo_teacher_representation_max_tokens_per_gpu=None,
    )
    actor.train_parallel_config = {"cp_size": 1}
    actor._active_model_tag = "actor"
    actor.weights_backuper = SimpleNamespace(backup_tags=["actor", "ref"])
    actor._switch_model = lambda tag: setattr(actor, "_active_model_tag", tag)
    actor._sdpo_teacher_model_tag = lambda: "ref"

    def fake_build_teacher_rollout_data(expanded_rollout_data):
        teacher_rollout_data = dict(expanded_rollout_data)
        actor._repack_sdpo_teacher_microbatches(teacher_rollout_data)
        return teacher_rollout_data

    actor._build_sdpo_teacher_rollout_data = fake_build_teacher_rollout_data

    def fake_compute(_data_iterator, _num_microbatches, store_prefix=""):
        assert store_prefix == "sdpo_teacher_"
        assert _data_iterator[0].rollout_data["micro_batch_indices"] == [[0, 2], [1]]
        return {
            "sdpo_teacher_representations": [
                torch.full((1, 1), float(row_idx), dtype=torch.float32)
                for row_idx in range(len(_data_iterator[0].rollout_data["total_lengths"]))
            ]
        }

    actor.compute_sdpo_representation_data = fake_compute
    rollout_data = {
        "tokens": [torch.tensor([1] * 6), torch.tensor([2] * 6), torch.tensor([3] * 4)],
        "total_lengths": [6, 6, 4],
        "response_lengths": [1, 1, 1],
        "loss_masks": [torch.tensor([1]), torch.tensor([1]), torch.tensor([1])],
        "sdpo_teacher_prompt_text": ["a", "b", "c"],
        "num_microbatches": [1],
        "global_batch_sizes": [3],
        "micro_batch_indices": [[0, 1, 2]],
    }

    actor._ensure_sdpo_teacher_representations(rollout_data)

    assert [float(row.item()) for row in rollout_data["sdpo_teacher_representations"]] == [0.0, 1.0, 2.0]
    assert rollout_data["sdpo_teacher_representation_microbatch_count_per_rank"].item() == pytest.approx(2.0)


@pytest.mark.unit
def test_actor_repacks_logprob_forward_with_logprob_token_budget(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: 1)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=4,
        log_probs_max_tokens_per_gpu=5,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
    )
    actor.train_parallel_config = {"cp_size": 2}
    rollout_data = {
        "tokens": [
            torch.tensor([1] * 9),
            torch.tensor([2] * 8),
            torch.tensor([3] * 7),
            torch.tensor([4]),
            torch.tensor([5]),
        ],
        "total_lengths": [9, 8, 7, 1, 1],
        "response_lengths": [1, 1, 1, 1, 1],
        "loss_masks": [torch.tensor([1]), torch.tensor([1]), torch.tensor([1]), torch.tensor([1]), torch.tensor([1])],
        "num_microbatches": [5],
        "global_batch_sizes": [5],
        "micro_batch_indices": [[0], [1], [2], [3], [4]],
    }
    data_iterator = actor_module.get_data_iterator(rollout_data)

    logprob_iterator, logprob_num_microbatches = actor._get_log_prob_data_iterator(
        rollout_data,
        data_iterator,
        rollout_data["num_microbatches"],
    )

    assert logprob_num_microbatches == [3]
    assert logprob_iterator[0].rollout_data is not rollout_data
    assert logprob_iterator[0].rollout_data["micro_batch_indices"] == [[0, 3], [1, 4], [2]]
    assert rollout_data["micro_batch_indices"] == [[0], [1], [2], [3], [4]]
    assert rollout_data["num_microbatches"] == [5]


@pytest.mark.unit
def test_actor_keeps_original_logprob_schedule_when_budget_matches_train(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=10,
        log_probs_max_tokens_per_gpu=10,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
    )
    actor.train_parallel_config = {"cp_size": 1}
    rollout_data = {
        "tokens": [torch.tensor([1] * 6), torch.tensor([2] * 6), torch.tensor([3] * 4)],
        "total_lengths": [6, 6, 4],
        "response_lengths": [1, 1, 1],
        "loss_masks": [torch.tensor([1]), torch.tensor([1]), torch.tensor([1])],
        "num_microbatches": [3],
        "global_batch_sizes": [3],
        "micro_batch_indices": [[0], [1], [2]],
    }
    data_iterator = actor_module.get_data_iterator(rollout_data)

    logprob_iterator, logprob_num_microbatches = actor._get_log_prob_data_iterator(
        rollout_data,
        data_iterator,
        rollout_data["num_microbatches"],
    )

    assert logprob_iterator is data_iterator
    assert logprob_num_microbatches == [3]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("use_routing_replay", "use_rollout_routing_replay"),
    [(True, False), (False, True)],
)
def test_actor_keeps_original_logprob_schedule_for_routing_replay(
    monkeypatch,
    use_routing_replay,
    use_rollout_routing_replay,
):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=10,
        log_probs_max_tokens_per_gpu=16,
        use_routing_replay=use_routing_replay,
        use_rollout_routing_replay=use_rollout_routing_replay,
    )
    actor.train_parallel_config = {"cp_size": 1}
    rollout_data = {
        "tokens": [torch.tensor([1] * 6), torch.tensor([2] * 6), torch.tensor([3] * 4)],
        "total_lengths": [6, 6, 4],
        "response_lengths": [1, 1, 1],
        "loss_masks": [torch.tensor([1]), torch.tensor([1]), torch.tensor([1])],
        "num_microbatches": [3],
        "global_batch_sizes": [3],
        "micro_batch_indices": [[0], [1], [2]],
    }
    data_iterator = actor_module.get_data_iterator(rollout_data)

    logprob_iterator, logprob_num_microbatches = actor._get_log_prob_data_iterator(
        rollout_data,
        data_iterator,
        rollout_data["num_microbatches"],
    )

    assert logprob_iterator is data_iterator
    assert logprob_num_microbatches == [3]


@pytest.mark.unit
def test_actor_keeps_original_logprob_schedule_without_dynamic_batch(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=10,
        log_probs_max_tokens_per_gpu=16,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
    )
    actor.train_parallel_config = {"cp_size": 1}
    rollout_data = {
        "tokens": [torch.tensor([1] * 6), torch.tensor([2] * 6), torch.tensor([3] * 4)],
        "total_lengths": [6, 6, 4],
        "response_lengths": [1, 1, 1],
        "loss_masks": [torch.tensor([1]), torch.tensor([1]), torch.tensor([1])],
        "num_microbatches": [3],
        "global_batch_sizes": [3],
        "micro_batch_indices": [[0], [1], [2]],
    }
    data_iterator = actor_module.get_data_iterator(rollout_data)

    logprob_iterator, logprob_num_microbatches = actor._get_log_prob_data_iterator(
        rollout_data,
        data_iterator,
        rollout_data["num_microbatches"],
    )

    assert logprob_iterator is data_iterator
    assert logprob_num_microbatches == [3]


@pytest.mark.unit
def test_actor_logprob_repack_preserves_rollout_step_boundaries(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: 1)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=4,
        log_probs_max_tokens_per_gpu=6,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
    )
    actor.train_parallel_config = {"cp_size": 1}
    rollout_data = {
        "tokens": [torch.tensor([1] * 4), torch.tensor([2] * 2), torch.tensor([3] * 4), torch.tensor([4] * 2)],
        "total_lengths": [4, 2, 4, 2],
        "response_lengths": [1, 1, 1, 1],
        "loss_masks": [torch.tensor([1]), torch.tensor([1]), torch.tensor([1]), torch.tensor([1])],
        "num_microbatches": [2, 2],
        "global_batch_sizes": [1, 1],
        "micro_batch_indices": [[0], [1], [2], [3]],
    }
    data_iterator = actor_module.get_data_iterator(rollout_data)

    logprob_iterator, logprob_num_microbatches = actor._get_log_prob_data_iterator(
        rollout_data,
        data_iterator,
        rollout_data["num_microbatches"],
    )

    assert logprob_num_microbatches == [1, 1]
    assert logprob_iterator[0].rollout_data["micro_batch_indices"] == [[0, 1], [2, 3]]
    assert logprob_iterator[0].rollout_data["global_batch_sizes"] == [1, 1]
    assert rollout_data["micro_batch_indices"] == [[0], [1], [2], [3]]


@pytest.mark.unit
def test_actor_logprob_repack_aligns_virtual_pipeline_microbatch_group(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: 2)
    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: 1)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=4,
        log_probs_max_tokens_per_gpu=10,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
    )
    actor.train_parallel_config = {"cp_size": 1, "vpp_size": 2, "microbatch_group_size_per_vp_stage": 2}
    rollout_data = {
        "tokens": [
            torch.tensor([1] * 9),
            torch.tensor([2] * 8),
            torch.tensor([3] * 7),
            torch.tensor([4]),
            torch.tensor([5]),
        ],
        "total_lengths": [9, 8, 7, 1, 1],
        "response_lengths": [1, 1, 1, 1, 1],
        "loss_masks": [torch.tensor([1]), torch.tensor([1]), torch.tensor([1]), torch.tensor([1]), torch.tensor([1])],
        "num_microbatches": [5],
        "global_batch_sizes": [5],
        "micro_batch_indices": [[0], [1], [2], [3], [4]],
    }
    data_iterator = actor_module.get_data_iterator(rollout_data)

    logprob_iterator, logprob_num_microbatches = actor._get_log_prob_data_iterator(
        rollout_data,
        data_iterator,
        rollout_data["num_microbatches"],
    )

    assert logprob_num_microbatches == [4]
    assert logprob_iterator[0].rollout_data["micro_batch_indices"] == [[0], [1, 4], [2], [3]]


@pytest.mark.unit
def test_actor_logprob_repack_falls_back_when_vpp_alignment_cannot_split(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: 2)
    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: 1)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=4,
        log_probs_max_tokens_per_gpu=10,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
    )
    actor.train_parallel_config = {"cp_size": 1, "vpp_size": 2, "microbatch_group_size_per_vp_stage": 4}
    rollout_data = {
        "tokens": [torch.tensor([1] * 9), torch.tensor([2] * 8)],
        "total_lengths": [9, 8],
        "response_lengths": [1, 1],
        "loss_masks": [torch.tensor([1]), torch.tensor([1])],
        "num_microbatches": [2],
        "global_batch_sizes": [2],
        "micro_batch_indices": [[0], [1]],
    }
    data_iterator = actor_module.get_data_iterator(rollout_data)

    logprob_iterator, logprob_num_microbatches = actor._get_log_prob_data_iterator(
        rollout_data,
        data_iterator,
        rollout_data["num_microbatches"],
    )

    assert logprob_iterator is data_iterator
    assert logprob_num_microbatches == [2]


@pytest.mark.unit
def test_actor_accumulates_teacher_representation_dynamic_fill_ratio(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: 1)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    actor.train_parallel_config = {"cp_size": 1}
    stats = actor._new_sdpo_teacher_representation_stats()
    actor._accumulate_sdpo_teacher_representation_stats(
        stats,
        {
            "total_lengths": [4, 5, 12],
            "micro_batch_indices": [[0, 1], [2]],
        },
    )
    rollout_data = {}

    actor._set_sdpo_teacher_representation_stats(rollout_data, stats)

    assert rollout_data["sdpo_teacher_representation_chunk_count_per_rank"].item() == pytest.approx(1.0)
    assert rollout_data["sdpo_teacher_representation_expanded_rows_per_rank"].item() == pytest.approx(3.0)
    assert rollout_data["sdpo_teacher_representation_tokens_per_rank"].item() == pytest.approx(21.0)
    assert rollout_data["sdpo_teacher_representation_microbatch_count_per_rank"].item() == pytest.approx(2.0)
    assert rollout_data["sdpo_teacher_representation_max_microbatch_tokens_per_rank"].item() == pytest.approx(12.0)
    assert rollout_data["sdpo_teacher_representation_avg_fill_ratio_per_rank"].item() == pytest.approx(1.05)


@pytest.mark.unit
@pytest.mark.parametrize("entropy_enabled", [False, True])
def test_actor_train_skips_sdpo_representation_advantage_preforward(monkeypatch, entropy_enabled):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        use_rollout_routing_replay=False,
        compute_advantages_and_returns=True,
        loss_type="sdpo_loss",
        sdpo_distillation_mode="representation",
        use_routing_replay=False,
        kl_coef=0.0,
        use_critic=False,
        use_opd=False,
        get_mismatch_metrics=False,
        use_rollout_logprobs=False,
        use_tis=False,
        use_opsm=False,
        custom_advantage_function_path=None,
        advantage_estimator="grpo",
        keep_old_actor=False,
        rollout_temperature=1.0,
        use_rollout_entropy=False,
        agent_task_diversity_entropy_enabled=entropy_enabled,
        log_correct_samples=False,
        rollout_data_postprocess_path=actor_module.SDPO_ENTROPY_ROLLOUT_DATA_POSTPROCESS_PATH,
        ref_update_interval=None,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=4,
        log_probs_max_tokens_per_gpu=6,
    )
    actor.train_parallel_config = {"cp_size": 1}
    actor.model = object()
    actor.optimizer = object()
    actor.opt_param_scheduler = object()
    actor._active_model_tag = "actor"
    actor.weights_backuper = SimpleNamespace(
        backup_tags=["actor", "ref"],
        backup=lambda tag: calls.append(("backup", tag)),
    )
    actor.weight_updater = SimpleNamespace(pop_metrics=lambda: {})
    actor.prof = SimpleNamespace(step=lambda rollout_id: calls.append(("prof", rollout_id)))
    actor._update_sdpo_ema_teacher = lambda: calls.append("ema")
    actor._switch_model = lambda tag: (calls.append(("switch", tag)), setattr(actor, "_active_model_tag", tag))

    def fake_ensure_sdpo_teacher_representations(rollout_data):
        calls.append("ensure_teacher_representations")
        rollout_data.setdefault("sdpo_teacher_representations", [torch.ones(1, 2)])

    actor._ensure_sdpo_teacher_representations = fake_ensure_sdpo_teacher_representations

    calls = []

    def fake_compute_log_prob(_data_iterator, _num_microbatches, store_prefix="", with_entropy=None, **_kwargs):
        calls.append(("compute_log_prob", store_prefix, with_entropy, actor._active_model_tag))
        if not entropy_enabled:
            raise AssertionError("representation-only train must not precompute actor/ref log_probs")
        assert store_prefix == ""
        assert with_entropy is True
        assert actor._active_model_tag == "actor"
        assert _data_iterator[0].rollout_data is not rollout_data
        assert _data_iterator[0].rollout_data["micro_batch_indices"] == [[0, 2], [1]]
        assert _num_microbatches == [2]
        return {"log_probs": [torch.tensor([9.0])], "actor_entropy": [torch.tensor([0.5])]}

    actor.compute_log_prob = fake_compute_log_prob

    def fake_compute_advantages(_args, _rollout_data):
        raise AssertionError("representation-only train must not compute advantages")

    monkeypatch.setattr(actor_module, "compute_advantages_and_returns", fake_compute_advantages)

    def fake_postprocess(_args, _rollout_id, rollout_data):
        calls.append("postprocess")
        assert ("actor_entropy" in rollout_data) is entropy_enabled
        assert "log_probs" not in rollout_data

    actor.rollout_data_postprocess = fake_postprocess

    def fake_train(
        _rollout_id,
        _model,
        _optimizer,
        _scheduler,
        _data_iterator,
        _num_microbatches,
        _global_batch_sizes,
        **kwargs,
    ):
        calls.append(("train", kwargs))
        assert "before_train_step" not in kwargs
        assert "ensure_teacher_representations" in calls
        assert "sdpo_teacher_representations" in rollout_data
        assert "advantages" not in rollout_data
        assert "log_probs" not in rollout_data
        assert _data_iterator[0].rollout_data is rollout_data
        assert _data_iterator[0].rollout_data["micro_batch_indices"] == [[0], [1], [2]]
        assert _num_microbatches == [3]

    monkeypatch.setattr(actor_module, "train", fake_train)
    monkeypatch.setattr(actor_module, "timer", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(actor_module, "inverse_timer", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(actor_module, "log_rollout_data", lambda *_args, **_kwargs: calls.append("log_rollout"))
    monkeypatch.setattr(actor_module, "log_perf_data", lambda *_args, **_kwargs: calls.append("log_perf"))
    monkeypatch.setattr(
        actor_module.train_dump_utils,
        "save_debug_train_data",
        lambda *_args, **_kwargs: calls.append("dump"),
    )

    rollout_data = {
        "tokens": [torch.tensor([1] * 4), torch.tensor([2] * 4), torch.tensor([3] * 2)],
        "total_lengths": [4, 4, 2],
        "response_lengths": [1, 1, 1],
        "loss_masks": [torch.tensor([1]), torch.tensor([1]), torch.tensor([1])],
        "num_microbatches": [3],
        "global_batch_sizes": [3],
        "micro_batch_indices": [[0], [1], [2]],
    }

    actor.train_actor(rollout_id=7, rollout_data=rollout_data)

    assert ("backup", "actor") in calls
    assert "ema" in calls
    if entropy_enabled:
        assert ("compute_log_prob", "", True, "actor") in calls
        assert "actor_entropy" in rollout_data
    else:
        assert not any(call[0] == "compute_log_prob" for call in calls if isinstance(call, tuple))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "expects_ref"),
    [
        ({}, False),
        ({"kl_coef": 0.1}, True),
        ({"use_kl_loss": True}, True),
        ({"custom_advantage_function_path": "custom.module:advantage"}, True),
    ],
)
def test_actor_train_only_preforwards_ref_log_probs_when_kl_needs_ref(monkeypatch, overrides, expects_ref):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    values = dict(
        use_rollout_routing_replay=False,
        compute_advantages_and_returns=True,
        loss_type="sdpo_loss",
        sdpo_distillation_mode="topk",
        sdpo_full_logit_distillation=True,
        sdpo_distillation_topk=20,
        use_routing_replay=False,
        kl_coef=0.0,
        use_kl_loss=False,
        custom_advantage_function_path=None,
        use_rollout_logprobs=False,
        get_mismatch_metrics=False,
        use_critic=False,
        keep_old_actor=False,
        use_opd=False,
        advantage_estimator="grpo",
        agent_task_diversity_entropy_enabled=False,
        rollout_temperature=1.0,
        ref_update_interval=None,
        use_dynamic_batch_size=False,
    )
    values.update(overrides)
    actor.args = SimpleNamespace(**values)
    actor.model = object()
    actor.optimizer = object()
    actor.opt_param_scheduler = object()
    actor._active_model_tag = "actor"
    actor.rollout_data_postprocess = None

    calls = []
    actor.weights_backuper = SimpleNamespace(
        backup_tags=["actor", "ref"],
        backup=lambda tag: calls.append(("backup", tag)),
    )
    actor.weight_updater = SimpleNamespace(pop_metrics=lambda: {})
    actor.prof = SimpleNamespace(step=lambda rollout_id: calls.append(("prof", rollout_id)))
    actor._update_sdpo_ema_teacher = lambda: calls.append("ema")
    actor._switch_model = lambda tag: (calls.append(("switch", tag)), setattr(actor, "_active_model_tag", tag))

    def fake_compute_log_prob(_data_iterator, _num_microbatches, store_prefix="", with_entropy=None, **_kwargs):
        calls.append(("compute_log_prob", store_prefix, actor._active_model_tag))
        assert with_entropy is None
        if store_prefix == "ref_":
            assert actor._active_model_tag == "ref"
            return {"ref_log_probs": [torch.tensor([-0.1])]}
        assert store_prefix == ""
        assert actor._active_model_tag == "actor"
        return {"log_probs": [torch.tensor([-0.2])]}

    actor.compute_log_prob = fake_compute_log_prob

    def fake_compute_advantages(_args, rollout_data):
        calls.append("advantages")
        assert ("log_probs" in rollout_data) is expects_ref
        assert ("ref_log_probs" in rollout_data) is expects_ref
        rollout_data["advantages"] = [torch.tensor([1.0])]
        rollout_data["returns"] = [torch.tensor([1.0])]

    monkeypatch.setattr(actor_module, "compute_advantages_and_returns", fake_compute_advantages)

    def fake_ensure_sdpo_teacher_log_probs(rollout_data):
        calls.append("ensure_teacher_log_probs")
        rollout_data["sdpo_teacher_log_probs"] = [torch.tensor([-0.3])]
        rollout_data["sdpo_topk_indices"] = [torch.tensor([[0]])]
        rollout_data["sdpo_teacher_topk_log_probs"] = [torch.tensor([[-0.4]])]

    actor._ensure_sdpo_teacher_log_probs = fake_ensure_sdpo_teacher_log_probs

    def fake_train(
        _rollout_id,
        _model,
        _optimizer,
        _scheduler,
        _data_iterator,
        _num_microbatches,
        _global_batch_sizes,
        **kwargs,
    ):
        calls.append(("train", kwargs))
        assert "before_train_step" not in kwargs
        assert "ensure_teacher_log_probs" in calls

    monkeypatch.setattr(actor_module, "train", fake_train)
    monkeypatch.setattr(actor_module, "timer", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(actor_module, "inverse_timer", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(actor_module, "log_rollout_data", lambda *_args, **_kwargs: calls.append("log_rollout"))
    monkeypatch.setattr(actor_module, "log_perf_data", lambda *_args, **_kwargs: calls.append("log_perf"))
    monkeypatch.setattr(
        actor_module.train_dump_utils,
        "save_debug_train_data",
        lambda *_args, **_kwargs: calls.append("dump"),
    )

    rollout_data = {
        "tokens": [torch.tensor([1, 2])],
        "total_lengths": [2],
        "response_lengths": [1],
        "loss_masks": [torch.tensor([1])],
        "num_microbatches": [1],
        "global_batch_sizes": [1],
        "micro_batch_indices": [[0]],
    }

    actor.train_actor(rollout_id=3, rollout_data=rollout_data)

    ref_preforwards = [call for call in calls if call == ("compute_log_prob", "ref_", "ref")]
    assert bool(ref_preforwards) is expects_ref
    assert (("compute_log_prob", "", "actor") in calls) is expects_ref
    assert ("backup", "actor") in calls
    assert "ema" in calls


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "postprocess_path", "has_postprocess"),
    [
        ({"log_correct_samples": True}, None, False),
        ({"use_routing_replay": True}, None, False),
        ({}, "custom.hook", True),
    ],
)
def test_actor_keeps_sdpo_representation_preforward_for_side_effects(
    overrides,
    postprocess_path,
    has_postprocess,
):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    values = dict(
        loss_type="sdpo_loss",
        sdpo_distillation_mode="representation",
        kl_coef=0.0,
        use_critic=False,
        use_opd=False,
        get_mismatch_metrics=False,
        use_rollout_logprobs=False,
        log_correct_samples=False,
        use_routing_replay=False,
        use_tis=False,
        use_opsm=False,
        custom_advantage_function_path=None,
        advantage_estimator="grpo",
        rollout_data_postprocess_path=postprocess_path,
    )
    values.update(overrides)
    actor.args = SimpleNamespace(**values)
    actor.rollout_data_postprocess = (lambda *_args, **_kwargs: None) if has_postprocess else None

    assert actor._sdpo_representation_should_skip_advantage_precompute() is False


@pytest.mark.unit
def test_actor_allows_sdpo_representation_preforward_skip_with_entropy_postprocess():
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        loss_type="sdpo_loss",
        sdpo_distillation_mode="representation",
        kl_coef=0.0,
        use_critic=False,
        use_opd=False,
        get_mismatch_metrics=False,
        use_rollout_logprobs=False,
        log_correct_samples=False,
        use_routing_replay=False,
        use_tis=False,
        use_opsm=False,
        custom_advantage_function_path=None,
        advantage_estimator="grpo",
        rollout_data_postprocess_path=actor_module.SDPO_ENTROPY_ROLLOUT_DATA_POSTPROCESS_PATH,
    )
    actor.rollout_data_postprocess = lambda *_args, **_kwargs: None

    assert actor._sdpo_representation_should_skip_advantage_precompute() is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("tp_size", "cp_size", "message"),
    [(2, 1, "tensor_model_parallel_size=1"), (1, 2, "context_parallel_size=1")],
)
def test_sdpo_representation_runtime_parallelism_guards(monkeypatch, tp_size, cp_size, message):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_tensor_model_parallel_world_size", lambda: tp_size)
    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: cp_size)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(sdpo_distillation_mode="representation")

    with pytest.raises(NotImplementedError, match=message):
        actor._validate_sdpo_distillation_parallelism()


@pytest.mark.unit
def test_sdpo_token_weights_runtime_rejects_pipeline_parallelism(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_pipeline_model_parallel_world_size", lambda: 2)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(sdpo_token_weights=True, sdpo_distillation_mode="topk")

    with pytest.raises(NotImplementedError, match="pipeline_model_parallel_size=1"):
        actor._validate_sdpo_distillation_parallelism()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("tp_size", "cp_size", "message"),
    [(2, 1, "tensor_model_parallel_size=1"), (1, 2, "context_parallel_size=1")],
)
def test_sdpo_representation_hidden_slice_parallelism_guards(monkeypatch, tp_size, cp_size, message):
    from slime.backends.megatron_utils import loss as loss_module

    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_tensor_model_parallel_world_size", lambda: tp_size)
    _set_mpu_attr(monkeypatch, loss_module.mpu, "get_context_parallel_world_size", lambda: cp_size)

    with pytest.raises(NotImplementedError, match=message):
        loss_module.get_response_hidden_representations(
            torch.zeros(1, 1, 4),
            args=SimpleNamespace(qkv_format="thd"),
            total_lengths=[1],
            response_lengths=[1],
        )


@pytest.mark.unit
def test_actor_repacks_teacher_microbatches_by_teacher_lengths(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: 1)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    actor.train_parallel_config = {"cp_size": 1}
    teacher_rollout_data = {
        "total_lengths": [8, 8, 3, 15, 2],
        "micro_batch_indices": [[0, 1, 2, 3, 4]],
        "num_microbatches": [1],
        "global_batch_sizes": [5],
    }

    actor._repack_sdpo_teacher_microbatches(teacher_rollout_data)

    assert teacher_rollout_data["micro_batch_indices"] == [[0], [1], [2], [3], [4]]
    assert teacher_rollout_data["num_microbatches"] == [5]
    assert teacher_rollout_data["global_batch_sizes"] == [5]


@pytest.mark.unit
def test_actor_repacks_teacher_microbatches_with_teacher_specific_token_budget(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: 1)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        sdpo_distillation_mode="representation",
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=10,
        sdpo_teacher_representation_max_tokens_per_gpu=16,
    )
    actor.train_parallel_config = {"cp_size": 1}
    teacher_rollout_data = {
        "total_lengths": [6, 6, 4],
        "micro_batch_indices": [[0], [1], [2]],
        "num_microbatches": [3],
        "global_batch_sizes": [3],
    }

    actor._repack_sdpo_teacher_microbatches(teacher_rollout_data)

    assert teacher_rollout_data["micro_batch_indices"] == [[0, 1, 2]]
    assert teacher_rollout_data["num_microbatches"] == [1]
    assert teacher_rollout_data["global_batch_sizes"] == [3]
    stats = actor._new_sdpo_teacher_representation_stats()
    actor._accumulate_sdpo_teacher_representation_stats(stats, teacher_rollout_data)
    rollout_data = {}
    actor._set_sdpo_teacher_representation_stats(rollout_data, stats)
    assert rollout_data["sdpo_teacher_representation_avg_fill_ratio_per_rank"].item() == pytest.approx(1.0)


@pytest.mark.unit
def test_actor_packs_teacher_representation_microbatches_by_length(monkeypatch):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: 1)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        sdpo_distillation_mode="representation",
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=10,
        sdpo_teacher_representation_max_tokens_per_gpu=None,
    )
    actor.train_parallel_config = {"cp_size": 1}
    teacher_rollout_data = {"total_lengths": [6, 4, 6, 4]}

    actor._repack_sdpo_teacher_microbatches(teacher_rollout_data)

    assert teacher_rollout_data["micro_batch_indices"] == [[0, 1], [2, 3]]
    stats = actor._new_sdpo_teacher_representation_stats()
    actor._accumulate_sdpo_teacher_representation_stats(stats, teacher_rollout_data)
    rollout_data = {}
    actor._set_sdpo_teacher_representation_stats(rollout_data, stats)
    assert rollout_data["sdpo_teacher_representation_microbatch_count_per_rank"].item() == pytest.approx(2.0)
    assert rollout_data["sdpo_teacher_representation_avg_fill_ratio_per_rank"].item() == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.parametrize("distillation_mode", ["topk", "sample_token"])
def test_actor_ignores_representation_teacher_budget_for_non_representation_teacher_repack(
    monkeypatch,
    distillation_mode,
):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    _set_mpu_attr(monkeypatch, actor_module.mpu, "get_context_parallel_world_size", lambda: 1)

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        sdpo_distillation_mode=distillation_mode,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=10,
        sdpo_teacher_representation_max_tokens_per_gpu=16,
    )
    actor.train_parallel_config = {"cp_size": 1}
    teacher_rollout_data = {"total_lengths": [6, 6, 4]}

    actor._repack_sdpo_teacher_microbatches(teacher_rollout_data)

    assert teacher_rollout_data["micro_batch_indices"] == [[0], [1, 2]]
    assert teacher_rollout_data["num_microbatches"] == [2]
    assert teacher_rollout_data["global_batch_sizes"] == [3]


@pytest.mark.unit
def test_actor_loads_sdpo_ema_teacher_iteration_zero_sidecar(monkeypatch, tmp_path):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor
    monkeypatch.setattr(actor_module, "is_megatron_main_rank", lambda: True)

    path = tmp_path / "sdpo_ema_teacher" / "iter_0000000" / "rank_00000.pt"
    path.parent.mkdir(parents=True)
    path.write_text("placeholder")

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(load=str(tmp_path))
    actor._use_sdpo_ema_teacher = lambda: True
    actor._rank_for_sidecar = lambda: 0
    calls: list[tuple[str, Path]] = []
    actor.weights_backuper = SimpleNamespace(load=lambda tag, load_path: calls.append((tag, Path(load_path))))

    actor._maybe_load_sdpo_ema_teacher(0)

    assert calls == [("sdpo_teacher", path)]


@pytest.mark.unit
def test_actor_rejects_missing_iteration_zero_sidecar_for_run_checkpoint(tmp_path):
    actor_module = _import_actor_module()
    MegatronTrainRayActor = actor_module.MegatronTrainRayActor

    checkpoint_root = tmp_path / "run" / "checkpoints"
    (checkpoint_root / "iter_0000000").mkdir(parents=True)
    (tmp_path / "run" / "alfworld_config.yaml").write_text("agent_task_sdpo_enabled: true\n")

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(load=str(checkpoint_root))
    actor._use_sdpo_ema_teacher = lambda: True
    actor._rank_for_sidecar = lambda: 0
    actor.weights_backuper = SimpleNamespace(load=lambda tag, load_path: None)

    with pytest.raises(FileNotFoundError, match="iteration 0"):
        actor._maybe_load_sdpo_ema_teacher(0)


@pytest.mark.unit
def test_sgs_prepass_keeps_actor_awake_and_residency_is_one_shot(monkeypatch):
    actor_module = _import_actor_module()
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(offload_train=True, sgs_selection_fraction=0.25)
    calls = []
    actor.wake_up = lambda: calls.append("wake")
    actor.sleep = lambda: calls.append("sleep")
    actor._get_rollout_data = lambda ref: {"ref": ref}
    actor._score_sgs_rows = lambda data: [{"data": data}]
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 3)

    result = actor.score_sgs(7, "box")

    assert calls == ["wake"]
    assert result["rollout_id"] == 7
    assert result["rank"] == 3
    assert actor._consume_sgs_prepass_residency() is True
    assert actor._consume_sgs_prepass_residency() is False


@pytest.mark.unit
def test_sgs_prepass_failure_restores_offloaded_state(monkeypatch):
    actor_module = _import_actor_module()
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(offload_train=True, sgs_selection_fraction=0.25)
    calls = []
    actor.wake_up = lambda: calls.append("wake")
    actor.sleep = lambda: calls.append("sleep")
    actor._get_rollout_data = lambda _ref: {}
    actor._score_sgs_rows = lambda _data: (_ for _ in ()).throw(RuntimeError("score failed"))
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)

    with pytest.raises(RuntimeError, match="score failed"):
        actor.score_sgs(0, "box")

    assert calls == ["wake", "sleep"]
    assert actor._consume_sgs_prepass_residency() is False


@pytest.mark.unit
def test_sgs_prepass_does_not_double_wake_runner_managed_offline_actor(monkeypatch):
    actor_module = _import_actor_module()
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        offload_train=True,
        offline_train_eval_colocate=True,
        sgs_selection_fraction=0.05,
        debug_rollout_only=False,
    )
    calls = []
    actor.wake_up = lambda: calls.append("wake")
    actor.sleep = lambda: calls.append("sleep")
    actor._get_rollout_data = lambda ref: {"ref": ref}
    actor._score_sgs_rows = lambda data: [{"data": data}]
    actor.role = "actor"
    actor.train_actor = lambda rollout_id, data, external_data=None: {
        "rollout_id": rollout_id,
        "data": data,
        "external_data": external_data,
    }
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)

    result = actor.score_sgs(0, "box")
    train_result = actor.train(0, "box")

    assert calls == []
    assert result["rollout_id"] == 0
    assert train_result["rollout_id"] == 0
    assert actor._consume_sgs_prepass_residency() is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("prepass_kept_awake", "expected_calls"),
    [(True, ["sleep"]), (False, ["wake", "sleep"])],
)
def test_train_consumes_sgs_prepass_residency(prepass_kept_awake, expected_calls):
    actor_module = _import_actor_module()
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        debug_rollout_only=False,
        offline_train_eval_colocate=False,
        offload_train=True,
    )
    actor.role = "actor"
    actor._sgs_prepass_kept_awake = prepass_kept_awake
    calls = []
    actor.wake_up = lambda: calls.append("wake")
    actor.sleep = lambda: calls.append("sleep")
    actor._get_rollout_data = lambda ref: {"ref": ref}
    actor.train_actor = lambda rollout_id, data, external_data=None: (rollout_id, data, external_data)

    result = actor.train(7, "box", external_data="extra")

    assert result == (7, {"ref": "box"}, "extra")
    assert calls == expected_calls
    assert actor._sgs_prepass_kept_awake is False


@pytest.mark.unit
def test_method_b_retention_target_may_remain_missing_until_student_support_is_computed(monkeypatch):
    actor_module = _import_actor_module()
    monkeypatch.setattr(
        actor_module,
        "slice_log_prob_with_cp",
        lambda row, total, response, qkv, maximum: (row, total, response, qkv, maximum),
    )

    assert actor_module._slice_train_log_prob_with_cp(
        None, 9, 2, "thd", None, allow_missing_retention_target=True
    ) is None
    assert actor_module._slice_train_log_prob_with_cp(
        [0.1, 0.2], 9, 2, "thd", None, allow_missing_retention_target=False
    ) == ([0.1, 0.2], 9, 2, "thd", None)
    with pytest.raises(ValueError, match="outside a PR"):
        actor_module._slice_train_log_prob_with_cp(
            None, 9, 2, "thd", None, allow_missing_retention_target=False
        )


@pytest.mark.unit
def test_method_b_retention_targets_use_privileged_student_topk_then_frozen_teacher(monkeypatch):
    actor_module = _import_actor_module()
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(pr_weight=0.3)
    actor._active_model_tag = "actor"
    actor._switch_model = lambda tag: setattr(actor, "_active_model_tag", tag)
    actor._sdpo_teacher_model_tag = lambda: "sdpo_teacher"
    actor._routing_replay_stage = lambda _stage: nullcontext()
    actor._repack_sdpo_teacher_microbatches = lambda data: data.update(
        num_microbatches=[1], micro_batch_indices=[[0]], global_batch_sizes=[1]
    )
    actor._align_sdpo_teacher_log_probs_to_student_cp = lambda rows, *_args: rows

    calls = []

    def fake_compute(_iterator, _num_microbatches, store_prefix=""):
        calls.append((actor._active_model_tag, store_prefix))
        if store_prefix == "sdpo_retention_student_":
            return {"sdpo_retention_student_sdpo_topk_indices": [torch.tensor([[7, 3]])]}
        assert store_prefix == "sdpo_retention_teacher_"
        return {
            "sdpo_retention_teacher_log_probs": [torch.tensor([-0.2])],
            "sdpo_retention_teacher_sdpo_topk_log_probs": [torch.tensor([[-0.4, -1.1]])],
        }

    actor.compute_sdpo_distillation_data = fake_compute
    rollout_data = {
        "tokens": [torch.tensor([1, 11]), torch.tensor([9, 11])],
        "total_lengths": [2, 2],
        "response_lengths": [1, 1],
        "loss_masks": [torch.tensor([1]), torch.tensor([1])],
        "num_microbatches": [1],
        "pr_component": [0.0, 1.0],
        "sdpo_topk_indices": [torch.tensor([[2, 1]]), None],
        "sdpo_teacher_log_probs": [torch.tensor([-0.1]), None],
        "sdpo_teacher_topk_log_probs": [torch.tensor([[-0.3, -1.2]]), None],
    }

    actor._ensure_pr_targets(rollout_data)

    assert calls == [
        ("actor", "sdpo_retention_student_"),
        ("sdpo_teacher", "sdpo_retention_teacher_"),
    ]
    assert actor._active_model_tag == "actor"
    torch.testing.assert_close(rollout_data["sdpo_topk_indices"][1], torch.tensor([[7, 3]]))
    torch.testing.assert_close(rollout_data["sdpo_teacher_log_probs"][1], torch.tensor([-0.2]))
    torch.testing.assert_close(
        rollout_data["sdpo_teacher_topk_log_probs"][1], torch.tensor([[-0.4, -1.1]])
    )
    assert float(rollout_data["self_distillation/retention/rows_per_rank"]) == 1.0


@pytest.mark.unit
def test_method_b_allows_a_data_parallel_rank_without_local_retention_rows():
    actor_module = _import_actor_module()
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(pr_weight=0.5)
    actor.compute_sdpo_distillation_data = lambda *_args, **_kwargs: pytest.fail(
        "a rank without retention rows must not run retention forwards"
    )
    rollout_data = {
        "tokens": [torch.tensor([1, 11]), torch.tensor([2, 11])],
        "pr_component": [0.0, 0.0],
    }

    actor._ensure_pr_targets(rollout_data)

    assert float(rollout_data["self_distillation/retention/rows_per_rank"]) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
