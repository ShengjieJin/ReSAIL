from __future__ import annotations

from slime_plugins.agent_tasks.common.algorithms.sgs import sgs_audit_record

from contextlib import nullcontext
import json
import subprocess
from types import SimpleNamespace

import pytest
import torch

from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.algorithms import resail


NUM_GPUS = 0


def _value(cli: list[str], option: str) -> str:
    return str(cli[cli.index(option) + 1])


































def _sample(draw: int, turn: int, *, selected: bool) -> Sample:
    sample = Sample(
        response_length=1,
        metadata={"source_draw_id": draw, "source_turn_idx": turn},
    )
    sample.sgs_finalized = True
    sample.sgs_scored = True
    sample.sgs_selected = selected
    sample.sgs_score_record = {"action_token_count": 1}
    sample.sgs_batch_metrics = {}
    if selected:
        sample.sgs_precomputed = {
            field: f"{field}-{draw}-{turn}" for field in resail.PRECOMPUTED_FIELDS
        }
    return sample


@pytest.mark.unit
def test_dense_retention_keeps_selected_ordinary_rows_and_all_privileged_rows(monkeypatch):
    from slime.algorithms.sdpo import teacher_alignment
    from slime.rollout import sglang_rollout

    samples = [
        _sample(0, 0, selected=True),
        _sample(0, 1, selected=False),
        _sample(1, 0, selected=True),
        _sample(1, 1, selected=False),
    ]
    monkeypatch.setattr(
        resail,
        "default_sdpo_converter",
        lambda _args, _samples: {
            "tokens": [[index, 9] for index in range(4)],
            "total_lengths": [2] * 4,
            "response_lengths": [1] * 4,
            "loss_masks": [[1]] * 4,
            "sample_indices": list(range(4)),
            "sdpo_metadata": [{"row": index} for index in range(4)],
            "sdpo_loss_weights": [99.0] * 4,
            "self_distillation_mask": [1.0] * 4,
        },
    )

    class FakeGenerateState:
        tokenizer = object()
        processor = None

        def __init__(self, _args):
            pass

    monkeypatch.setattr(sglang_rollout, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(
        teacher_alignment,
        "build_sdpo_teacher_rollout_data",
        lambda data, *_args, **_kwargs: {
            "tokens": [[100 + row[0], 9] for row in data["tokens"]],
            "total_lengths": list(data["total_lengths"]),
            "response_lengths": list(data["response_lengths"]),
            "loss_masks": list(data["loss_masks"]),
        },
    )
    args = SimpleNamespace(
        sgs_selection_fraction=0.05,
        tlb_loss_aggregation="trajectory_balanced",
        rollout_batch_size=2,
        pr_weight=0.3,
        pr_support="all",
    )

    data = resail.convert_samples_to_train_data(args, samples)

    assert data["tokens"] == [[0, 9], [2, 9], [100, 9], [101, 9], [102, 9], [103, 9]]
    assert data["pr_component"] == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert data["pr_base_weight"] == pytest.approx([1.0] * 6)
    assert data["pr_normalization_scale"] == pytest.approx(
        [3.0, 3.0, 1.5, 1.5, 1.5, 1.5]
    )
    assert data["sdpo_loss_weights"] == pytest.approx([3.0, 3.0, 0.45, 0.45, 0.45, 0.45])
    assert sum(data["sdpo_loss_weights"][:2]) / len(data["sdpo_loss_weights"]) == pytest.approx(1.0)
    assert sum(data["sdpo_loss_weights"][2:]) / len(data["sdpo_loss_weights"]) == pytest.approx(0.3)
    assert data["self_distillation_mask"] == [1.0] * 6
    for field in resail.PRECOMPUTED_FIELDS:
        assert data[field][:2] == [f"{field}-0-0", f"{field}-1-0"]
        assert data[field][2:] == [None] * 4


@pytest.mark.unit
def test_dense_retention_cached_targets_preserve_rows_masks_weights_and_order(monkeypatch):
    from slime.algorithms.sdpo import teacher_alignment
    from slime.rollout import sglang_rollout

    samples = [_sample(0, 0, selected=True), _sample(0, 1, selected=False)]
    for index, sample in enumerate(samples):
        sample.pr_precomputed = {
            field: f"cached-{field}-{index}" for field in resail.PRECOMPUTED_FIELDS
        }
    monkeypatch.setattr(
        resail,
        "default_sdpo_converter",
        lambda _args, _samples: {
            "tokens": [[1, 9], [2, 9]],
            "total_lengths": [2, 2],
            "response_lengths": [1, 1],
            "loss_masks": [[1], [1]],
            "sample_indices": [0, 1],
            "sdpo_metadata": [{"row": 0}, {"row": 1}],
            "sdpo_loss_weights": [99.0, 99.0],
            "self_distillation_mask": [1.0, 1.0],
        },
    )

    class FakeGenerateState:
        tokenizer = object()
        processor = None

        def __init__(self, _args):
            pass

    monkeypatch.setattr(sglang_rollout, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(
        teacher_alignment,
        "build_sdpo_teacher_rollout_data",
        lambda data, *_args, **_kwargs: {
            "tokens": [[101, 9], [102, 9]],
            "total_lengths": list(data["total_lengths"]),
            "response_lengths": list(data["response_lengths"]),
            "loss_masks": list(data["loss_masks"]),
        },
    )
    args = SimpleNamespace(
        sgs_selection_fraction=0.05,
        tlb_loss_aggregation="trajectory_balanced",
        rollout_batch_size=1,
        pr_weight=0.5,
        pr_support="all",
    )

    data = resail.convert_samples_to_train_data(args, samples)

    assert data["tokens"] == [[1, 9], [101, 9], [102, 9]]
    assert data["loss_masks"] == [[1], [1], [1]]
    assert data["pr_component"] == [0.0, 1.0, 1.0]
    assert data["sdpo_loss_weights"] == pytest.approx([3.0, 0.75, 0.75])
    for field in resail.PRECOMPUTED_FIELDS:
        assert data[field] == [
            f"{field}-0-0",
            f"cached-{field}-0",
            f"cached-{field}-1",
        ]


@pytest.mark.unit
def test_dense_retention_requires_all_nominal_trajectory_groups(monkeypatch):
    samples = [_sample(0, 0, selected=True), _sample(0, 1, selected=False)]
    monkeypatch.setattr(
        resail,
        "default_sdpo_converter",
        lambda _args, _samples: {"sdpo_loss_weights": [1.0, 1.0]},
    )
    args = SimpleNamespace(
        sgs_selection_fraction=0.05,
        tlb_loss_aggregation="trajectory_balanced",
        rollout_batch_size=2,
        pr_weight=0.3,
        pr_support="all",
    )
    with pytest.raises(ValueError, match="one non-empty group per rollout source"):
        resail.convert_samples_to_train_data(args, samples)


@pytest.mark.unit
def test_dense_retention_unequal_trajectories_matches_direct_objective_and_gradient(monkeypatch):
    from slime.algorithms.sdpo import teacher_alignment
    from slime.rollout import sglang_rollout

    samples = [
        _sample(0, 0, selected=True),
        _sample(1, 0, selected=True),
        _sample(1, 1, selected=False),
        _sample(2, 0, selected=False),
        _sample(2, 1, selected=False),
        _sample(2, 2, selected=False),
    ]
    monkeypatch.setattr(
        resail,
        "default_sdpo_converter",
        lambda _args, _samples: {
            "tokens": [[index, 9] for index in range(6)],
            "total_lengths": [2] * 6,
            "response_lengths": [1] * 6,
            "loss_masks": [[1]] * 6,
            "sample_indices": list(range(6)),
            "sdpo_metadata": [{"row": index} for index in range(6)],
            "sdpo_loss_weights": [99.0] * 6,
            "self_distillation_mask": [1.0] * 6,
        },
    )

    class FakeGenerateState:
        tokenizer = object()
        processor = None

        def __init__(self, _args):
            pass

    monkeypatch.setattr(sglang_rollout, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(
        teacher_alignment,
        "build_sdpo_teacher_rollout_data",
        lambda data, *_args, **_kwargs: {
            "tokens": [[100 + row[0], 9] for row in data["tokens"]],
            "total_lengths": list(data["total_lengths"]),
            "response_lengths": list(data["response_lengths"]),
            "loss_masks": list(data["loss_masks"]),
        },
    )
    args = SimpleNamespace(
        sgs_selection_fraction=0.05,
        tlb_loss_aggregation="trajectory_balanced",
        rollout_batch_size=3,
        pr_weight=0.3,
        pr_support="all",
    )
    data = resail.convert_samples_to_train_data(args, samples)

    assert data["pr_component"] == [0.0, 0.0] + [1.0] * 6
    assert data["pr_base_weight"] == pytest.approx(
        [2 / 3, 2 / 3, 2.0, 1.0, 1.0, 2 / 3, 2 / 3, 2 / 3]
    )
    assert data["pr_normalization_scale"] == pytest.approx([4.0, 4.0] + [4 / 3] * 6)

    theta_joint = torch.tensor(0.4, requires_grad=True)
    targets = torch.arange(1.0, 9.0)
    joint_losses = (theta_joint - targets).square()
    joint = (torch.tensor(data["sdpo_loss_weights"]) * joint_losses).sum() / 8
    joint.backward()

    theta_direct = torch.tensor(0.4, requires_grad=True)
    selection = ((theta_direct - targets[:2]).square()).sum() / 3
    retention_by_trajectory = torch.stack(
        [
            (theta_direct - targets[2:3]).square().mean(),
            (theta_direct - targets[3:5]).square().mean(),
            (theta_direct - targets[5:8]).square().mean(),
        ]
    ).mean()
    direct = selection + 0.3 * retention_by_trajectory
    direct.backward()

    torch.testing.assert_close(joint, direct)
    torch.testing.assert_close(theta_joint.grad, theta_direct.grad)

    # An arbitrary two-way row split must preserve the same global numerator
    # and denominator used by the distributed active-row reducer.
    left = (torch.tensor(data["sdpo_loss_weights"][:3]) * joint_losses.detach()[:3]).sum()
    right = (torch.tensor(data["sdpo_loss_weights"][3:]) * joint_losses.detach()[3:]).sum()
    torch.testing.assert_close((left + right) / 8, direct.detach())


@pytest.mark.unit
def test_retention_normalization_scale_survives_data_iterator_get_batch(monkeypatch):
    try:
        from slime.backends.megatron_utils import data as data_module
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(f"data module requires full Megatron training package: {exc}")

    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    rollout_data = {
        "tokens": [torch.tensor([1, 2]), torch.tensor([3, 4])],
        "total_lengths": [2, 2],
        "response_lengths": [1, 1],
        "loss_masks": [torch.ones(1), torch.ones(1)],
        "pr_normalization_scale": [4.0, 4 / 3],
    }
    iterator = data_module.DataIterator(rollout_data, [[0, 1]])
    batch = data_module.get_batch(
        iterator,
        [
            "tokens",
            "total_lengths",
            "response_lengths",
            "loss_masks",
            "pr_normalization_scale",
        ],
        pad_multiplier=1,
        qkv_format="thd",
    )
    assert batch["pr_normalization_scale"] == pytest.approx([4.0, 4 / 3])






@pytest.mark.unit
def test_dense_retention_tensor_cache_does_not_change_persisted_audit_schema():
    record = {
        "source_draw_id": 1,
        "turn_idx": 2,
        "teacher_js": 0.25,
        "precomputed": {"sdpo_topk_indices": torch.tensor([[1]])},
        "retention_precomputed": {"sdpo_topk_indices": torch.tensor([[2]])},
    }

    persisted = sgs_audit_record(record)

    assert persisted == {"source_draw_id": 1, "turn_idx": 2, "teacher_js": 0.25}


@pytest.mark.unit
def test_method_b_forward_targets_use_frozen_teacher_topk_support(monkeypatch):
    try:
        from slime.backends.megatron_utils import actor as actor_module
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(f"actor module requires full Megatron training package: {exc}")

    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        pr_weight=0.5,
        pr_kl_direction="forward",
    )
    actor._active_model_tag = "actor"
    actor._switch_model = lambda tag: setattr(actor, "_active_model_tag", tag)
    actor._sdpo_teacher_model_tag = lambda: "ref"
    actor._routing_replay_stage = lambda _stage: nullcontext()
    actor._repack_sdpo_teacher_microbatches = lambda data: data.update(
        num_microbatches=[1], micro_batch_indices=[[0]], global_batch_sizes=[1]
    )
    actor._align_sdpo_teacher_log_probs_to_student_cp = lambda rows, *_args: rows
    calls = []

    def fake_compute(_iterator, _num_microbatches, store_prefix=""):
        calls.append((actor._active_model_tag, store_prefix))
        assert store_prefix == "sdpo_retention_teacher_"
        return {
            "sdpo_retention_teacher_sdpo_topk_indices": [torch.tensor([[8, 4]])],
            "sdpo_retention_teacher_log_probs": [torch.tensor([-0.2])],
            "sdpo_retention_teacher_sdpo_topk_log_probs": [torch.tensor([[-0.3, -1.4]])],
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

    assert calls == [("ref", "sdpo_retention_teacher_")]
    assert actor._active_model_tag == "actor"
    torch.testing.assert_close(rollout_data["sdpo_topk_indices"][1], torch.tensor([[8, 4]]))
    torch.testing.assert_close(rollout_data["sdpo_teacher_log_probs"][1], torch.tensor([-0.2]))


@pytest.mark.unit
@pytest.mark.parametrize("direction", ["reverse", "forward"])
def test_method_b_cached_targets_skip_redundant_retention_forwards(direction):
    try:
        from slime.backends.megatron_utils import actor as actor_module
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(f"actor module requires full Megatron training package: {exc}")

    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        pr_weight=0.5,
        pr_kl_direction=direction,
    )
    actor.compute_sdpo_distillation_data = lambda *_args, **_kwargs: pytest.fail(
        "fully cached retention targets must not run another model forward"
    )
    rollout_data = {
        "tokens": [torch.tensor([1, 11]), torch.tensor([9, 11])],
        "pr_component": [0.0, 1.0],
        "sdpo_topk_indices": [torch.tensor([[2, 1]]), torch.tensor([[8, 4]])],
        "sdpo_teacher_log_probs": [torch.tensor([-0.1]), torch.tensor([-0.2])],
        "sdpo_teacher_topk_log_probs": [
            torch.tensor([[-0.3, -1.2]]),
            torch.tensor([[-0.4, -1.1]]),
        ],
    }

    actor._ensure_pr_targets(rollout_data)

    assert float(rollout_data["self_distillation/retention/rows_per_rank"]) == 1.0


@pytest.mark.unit
def test_method_b_forward_changes_only_retention_component_kl_direction(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (
            torch.empty((0,)),
            {"log_probs": [torch.tensor([-0.2]), torch.tensor([-0.4])]},
        ),
    )
    monkeypatch.setattr(
        loss_module,
        "get_sdpo_distillation_tensors",
        lambda *_args, **_kwargs: (
            torch.empty((0,)),
            {"sdpo_topk_log_probs": [torch.tensor([[-0.2]]), torch.tensor([[-0.4]])]},
        ),
    )
    seen_alpha = []

    def fake_kl(student, _teacher, *, alpha):
        seen_alpha.append(alpha)
        return torch.ones(student.shape[0], dtype=student.dtype)

    monkeypatch.setattr(loss_module, "_compute_sdpo_kl_loss", fake_kl)
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        sdpo_full_logit_distillation=True,
        sdpo_distillation_topk=1,
        sdpo_distillation_add_tail=True,
        sdpo_loss_agg_mode="step_equal",
        sdpo_clip_ratio=None,
        sdpo_deployment_tis_clip=None,
        allgather_cp=False,
        sdpo_alpha=1.0,
        pr_weight=0.3,
        pr_kl_direction="forward",
    )
    batch = {
        "unconcat_tokens": [torch.tensor([1, 2]), torch.tensor([3, 4])],
        "total_lengths": [2, 2],
        "response_lengths": [1, 1],
        "loss_masks": [torch.ones(1), torch.ones(1)],
        "sdpo_teacher_log_probs": [torch.tensor([-0.5]), torch.tensor([-0.5])],
        "sdpo_topk_indices": [torch.tensor([[0]]), torch.tensor([[0]])],
        "sdpo_teacher_topk_log_probs": [torch.tensor([[-0.4]]), torch.tensor([[-0.4]])],
        "self_distillation_mask": torch.tensor([1.0, 1.0]),
        "sdpo_loss_weights": torch.tensor([2.0, 0.6]),
        "pr_component": [0.0, 1.0],
        "pr_base_weight": [1.0, 1.0],
        "pr_normalization_scale": [2.0, 2.0],
    }

    loss_module.sdpo_loss_function(
        args,
        batch,
        torch.zeros(1, 4, 4, requires_grad=True),
        lambda values: values.sum(),
    )

    assert seen_alpha == [1.0, 0.0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
