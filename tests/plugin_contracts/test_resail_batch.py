from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.algorithms import resail


NUM_GPUS = 0


def _sample(draw: int, turn: int, *, selected: bool) -> Sample:
    sample = Sample(
        index=draw * 100 + turn,
        rollout_id=0,
        tokens=[draw, turn, 9],
        response_length=1,
        loss_mask=[1],
        metadata={"source_draw_id": draw, "source_turn_idx": turn},
    )
    sample.sgs_finalized = True
    sample.sgs_scored = False
    sample.sgs_selected = selected
    sample.sgs_score_record = {
        "source_draw_id": draw,
        "turn_idx": turn,
        "action_token_count": 1,
        "response_token_count": 1,
    }
    sample.sgs_batch_metrics = {"attempted_steps": 4.0, "selected_steps": 2.0}
    return sample


@pytest.mark.unit
def test_unscored_random_keeps_selected_rows_and_all_step_retention(monkeypatch):
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
        lambda _args, rows: {
            "tokens": [list(row.tokens) for row in rows],
            "total_lengths": [len(row.tokens) for row in rows],
            "response_lengths": [row.response_length for row in rows],
            "loss_masks": [list(row.loss_mask) for row in rows],
            "sample_indices": list(range(len(rows))),
            "sdpo_metadata": [{"row": index} for index in range(len(rows))],
            "sdpo_loss_weights": [99.0] * len(rows),
            "self_distillation_mask": [1.0] * len(rows),
            "sdpo_teacher_prompt_text": [f"privileged-{index}" for index in range(len(rows))],
            "sdpo_teacher_messages": [[{"role": "user", "content": f"p-{index}"}] for index in range(len(rows))],
            "sgs_plain_prompt_text": [f"plain-{index}" for index in range(len(rows))],
            "sgs_plain_messages": [[{"role": "user", "content": f"o-{index}"}] for index in range(len(rows))],
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
            "tokens": [[100 + row[0], *row[1:]] for row in data["tokens"]],
            "total_lengths": list(data["total_lengths"]),
            "response_lengths": list(data["response_lengths"]),
            "loss_masks": list(data["loss_masks"]),
        },
    )
    args = SimpleNamespace(
        sgs_selection_fraction=0.05,
        tlb_loss_aggregation="trajectory_balanced",
        rollout_batch_size=2,
        pr_weight=0.5,
        pr_support="all",
        pr_view="privileged",
    )

    data = resail.convert_samples_to_train_data(args, samples)

    assert data["pr_component"] == [0.0, 0.0] + [1.0] * 4
    assert data["sdpo_loss_weights"] == pytest.approx([3.0, 3.0] + [0.75] * 4)
    assert data["pr_base_weight"] == pytest.approx([1.0] * 6)
    assert data["pr_normalization_scale"] == pytest.approx(
        [3.0, 3.0] + [1.5] * 4
    )
    for field in resail.PRECOMPUTED_FIELDS:
        assert data[field] == [None] * 6


@pytest.mark.unit
def test_ordinary_retention_uses_plain_tokens_and_plain_teacher_prompt(monkeypatch):
    from slime.algorithms.sdpo import teacher_alignment
    from slime.rollout import sglang_rollout

    class FakeGenerateState:
        tokenizer = object()
        processor = None

        def __init__(self, _args):
            pass

    monkeypatch.setattr(sglang_rollout, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(
        teacher_alignment,
        "build_sdpo_teacher_rollout_data",
        lambda *_args, **_kwargs: pytest.fail("ordinary retention must not build privileged student rows"),
    )
    train_data = {
        "tokens": [[1, 11], [2, 22]],
        "response_lengths": [1, 1],
        "loss_masks": [[1], [1]],
        "sample_indices": [0, 1],
        "sdpo_metadata": [{"row": 0}, {"row": 1}],
        "sdpo_loss_weights": [1.0, 1.0],
        "self_distillation_mask": [1.0, 1.0],
        "sdpo_teacher_prompt_text": ["privileged-0", "privileged-1"],
        "sdpo_teacher_messages": [[{"role": "user", "content": "p0"}], [{"role": "user", "content": "p1"}]],
        "sgs_plain_prompt_text": ["plain-0", "plain-1"],
        "sgs_plain_messages": [[{"role": "user", "content": "o0"}], [{"role": "user", "content": "o1"}]],
        **{field: [f"{field}-0", f"{field}-1"] for field in resail.PRECOMPUTED_FIELDS},
    }

    resail.expand_pr_rows(
        SimpleNamespace(pr_view="ordinary", pr_support="all"),
        train_data,
        retention_weight=0.5,
    )

    assert train_data["tokens"] == [[1, 11], [2, 22], [1, 11], [2, 22]]
    assert train_data["sdpo_teacher_prompt_text"] == [
        "privileged-0",
        "privileged-1",
        "plain-0",
        "plain-1",
    ]
    assert train_data["sdpo_teacher_messages"][2:] == train_data["sgs_plain_messages"][:2]
    assert [row["sdpo_objective_component"] for row in train_data["sdpo_metadata"][2:]] == [
        "ordinary_retention",
        "ordinary_retention",
    ]
    assert all(row["pr_view"] == "ordinary" for row in train_data["sdpo_metadata"][2:])


@pytest.mark.unit
def test_unscored_random_computes_selected_targets_without_full_scoring(monkeypatch):
    from slime.backends.megatron_utils import actor as actor_module

    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(sgs_selection_mode="random", sgs_skip_full_scoring=True)
    actor._active_model_tag = "actor"
    actor._switch_model = lambda tag: setattr(actor, "_active_model_tag", tag)
    actor._sdpo_teacher_model_tag = lambda: "ref"
    actor._routing_replay_stage = lambda _stage: nullcontext()
    actor._repack_sdpo_teacher_microbatches = lambda data: data.update(num_microbatches=[1])
    actor._align_sdpo_teacher_log_probs_to_student_cp = lambda rows, *_args: rows
    actor._build_sdpo_teacher_rollout_data = lambda data: {**data, "num_microbatches": [1]}
    monkeypatch.setattr(actor_module, "get_data_iterator", lambda data: data)
    calls = []

    def compute(_iterator, _num_microbatches, *, store_prefix):
        calls.append((actor._active_model_tag, store_prefix))
        if actor._active_model_tag == "actor":
            return {f"{store_prefix}sdpo_topk_indices": [torch.tensor([[7, 3]])]}
        return {
            f"{store_prefix}log_probs": [torch.tensor([-0.2])],
            f"{store_prefix}sdpo_topk_log_probs": [torch.tensor([[-0.4, -1.1]])],
        }

    actor.compute_sdpo_distillation_data = compute
    rollout_data = {
        "tokens": [torch.tensor([1, 11]), torch.tensor([9, 11])],
        "total_lengths": [2, 2],
        "response_lengths": [1, 1],
        "loss_masks": [torch.tensor([1]), torch.tensor([1])],
        "sdpo_teacher_prompt_text": ["privileged-selected", "privileged-retention"],
        "pr_component": [0.0, 1.0],
        "sdpo_topk_indices": [None, None],
        "sdpo_teacher_log_probs": [None, None],
        "sdpo_teacher_topk_log_probs": [None, None],
    }

    actor._ensure_unscored_random_selection_targets(rollout_data)

    assert calls == [
        ("actor", "sdpo_random_selection_student_"),
        ("ref", "sdpo_random_selection_teacher_"),
    ]
    torch.testing.assert_close(rollout_data["sdpo_topk_indices"][0], torch.tensor([[7, 3]]))
    torch.testing.assert_close(rollout_data["sdpo_teacher_log_probs"][0], torch.tensor([-0.2]))
    assert rollout_data["sdpo_topk_indices"][1] is None


@pytest.mark.unit
def test_distill_kl_scoring_skips_plain_teacher_forward(monkeypatch):
    from slime.backends.megatron_utils import actor as actor_module

    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        sgs_ranking_score="distillation_kl",
        sgs_score_scope="response",
        pr_weight=0.0,
        pr_support="all",
        pr_kl_direction="reverse",
        pr_view="privileged",
    )
    actor._active_model_tag = "actor"
    actor._switch_model = lambda tag: setattr(actor, "_active_model_tag", tag)
    actor._sdpo_teacher_model_tag = lambda: "ref"
    actor._validate_sdpo_distillation_parallelism = lambda: None
    actor._build_sdpo_teacher_rollout_data = lambda data: {**data, "num_microbatches": [1]}
    actor._build_sgs_plain_view_data = lambda _data: pytest.fail(
        "Distill-KL ranking must not build the plain-teacher view"
    )
    actor._align_sdpo_teacher_log_probs_to_student_cp = lambda rows, *_args: rows
    actor._compute_sdpo_main_divergence_scores = lambda *_args: ([0.7], [2])
    monkeypatch.setattr(actor_module, "get_data_iterator", lambda data: data)
    calls = []

    def compute(_iterator, _num_microbatches, *, store_prefix, include_distillation):
        calls.append((actor._active_model_tag, store_prefix, include_distillation))
        if actor._active_model_tag == "actor":
            return {"sdpo_sdpo_topk_indices": [torch.tensor([[1, 2]])]}
        return {
            "sdpo_teacher_log_probs": [torch.tensor([-0.2, -0.3])],
            "sdpo_teacher_sdpo_topk_log_probs": [torch.tensor([[-0.4, -1.1], [-0.5, -1.0]])],
        }

    actor.compute_sdpo_compressed_action_view_data = compute
    rollout_data = {
        "tokens": [[1, 2, 3]],
        "response_lengths": [2],
        "loss_masks": [[1, 1]],
        "num_microbatches": [1],
        "sdpo_teacher_prompt_text": ["privileged"],
        "sgs_action_token_mask": [[1, 1]],
        "sgs_action_alignment_valid": [True],
        "sgs_action_alignment_reason": [None],
        "sdpo_metadata": [
            {
                "sgs_source_draw_id": 0,
                "sgs_task_id": "task",
                "source_trajectory_uid": "trajectory",
                "turn_idx": 0,
                "frozen_trajectory_length": 1,
            }
        ],
    }

    rows = actor._score_sgs_rows(rollout_data)

    assert calls == [("actor", "sdpo_", True), ("ref", "sdpo_teacher_", True)]
    assert rows[0]["distillation_kl"] == pytest.approx(0.7)
    assert rows[0]["teacher_js"] is None
    assert rows[0]["score_unavailable_reason"] is None
