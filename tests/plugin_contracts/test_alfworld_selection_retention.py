from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slime.utils.types import Sample
from slime.algorithms.sdpo.sgs_metrics import paired_compressed_view_metrics
from slime_plugins.agent_tasks.common.eval import EvalTarget
from slime_plugins.agent_tasks.alfworld import eval as alfworld_eval
from slime_plugins.agent_tasks.alfworld import log as alfworld_log
from slime_plugins.agent_tasks.alfworld.frozen.contracts import expected_trajectory_count
from slime_plugins.agent_tasks.alfworld.frozen import data_source as frozen_data_source
from slime_plugins.agent_tasks.common.algorithms.resail import convert_samples_to_train_data
from slime_plugins.agent_tasks.common.algorithms.sgs import score_masks_for_scope, select_sgs_steps, sgs_score_available, sgs_option, sgs_ranking_score_field, sgs_scoring_required, sgs_selection_fraction, teacher_js_statistics
from slime_plugins.agent_tasks.common.algorithms import resail as resail

NUM_GPUS = 0


















@pytest.mark.unit
def test_response_score_scope_uses_every_active_response_token_and_not_action_parse():
    loss_masks = [[1, 1, 0, 1], [1, 1]]
    action_masks = [[0, 1, 0, 0], [0, 0]]
    assert score_masks_for_scope(loss_masks, action_masks, score_scope="response") == [
        [True, True, False, True],
        [True, True],
    ]
    assert score_masks_for_scope(loss_masks, action_masks, score_scope="action") == [
        [False, True, False, False],
        [False, False],
    ]
    assert sgs_score_available(score_scope="response", alignment_valid=False, score_token_count=3)
    assert not sgs_score_available(score_scope="action", alignment_valid=False, score_token_count=1)
    mean, total = teacher_js_statistics([0.9, 0.8, 0.01], expected_token_count=3)
    assert mean == pytest.approx(0.57)
    assert total == pytest.approx(1.71)
    response_rank = select_sgs_steps(
        [
            {"source_draw_id": 0, "turn_idx": 0, "traj_uid": "thinking-high", "teacher_js": mean},
            {"source_draw_id": 1, "turn_idx": 0, "traj_uid": "action-high", "teacher_js": 0.2},
        ],
        fraction=0.5,
        minimum_selected=1,
    )
    assert response_rank.selected_keys == ((0, 0),)


@pytest.mark.unit
def test_trajectory_topx_keeps_at_least_one_step_per_trajectory_and_fails_on_zero_scoreable():
    rows = [
        {"source_draw_id": draw, "turn_idx": turn, "traj_uid": f"t{draw}", "teacher_js": score}
        for draw, scores in ((0, [0.1, 0.9, 0.3]), (1, [0.2, 0.8]))
        for turn, score in enumerate(scores)
    ]
    selected = select_sgs_steps(
        rows, fraction=0.01, minimum_selected=8, selection_scope="trajectory"
    )
    assert set(selected.selected_keys) == {(0, 1), (1, 1)}
    broken = [dict(row, teacher_js=None) if row["source_draw_id"] == 1 else row for row in rows]
    with pytest.raises(RuntimeError, match="zero scoreable"):
        select_sgs_steps(broken, fraction=0.1, minimum_selected=8, selection_scope="trajectory")
    conflicting = [*rows, {"source_draw_id": 0, "turn_idx": 9, "traj_uid": "wrong", "teacher_js": 0.4}]
    with pytest.raises(RuntimeError, match="multiple trajectory UIDs"):
        select_sgs_steps(conflicting, fraction=0.1, minimum_selected=8, selection_scope="trajectory")


@pytest.mark.unit
def test_filtered_converter_preserves_source_weights(monkeypatch):
    samples = []
    for draw, weight in ((0, 2.0), (1, 0.5)):
        sample = Sample(response_length=2, metadata={"source_draw_id": draw, "source_turn_idx": 0})
        sample.sgs_scored = True
        sample.sgs_selected = True
        sample.sgs_precomputed = {field: object() for field in resail.PRECOMPUTED_FIELDS}
        sample.sgs_score_record = {"action_token_count": 1}
        sample.sgs_batch_metrics = {}
        samples.append(sample)
    monkeypatch.setattr(
        resail,
        "default_sdpo_converter",
        lambda _args, _samples: {"sdpo_loss_weights": [2.0, 0.5]},
    )
    data = convert_samples_to_train_data(
        SimpleNamespace(
            sgs_selection_fraction=0.1,
            tlb_loss_aggregation="donor",
            rollout_batch_size=32,
        ),
        samples,
    )
    assert data["sdpo_loss_weights"] == [2.0, 0.5]


@pytest.mark.unit
def test_method_b_expands_selected_support_with_privileged_rows_and_exact_joint_weights(monkeypatch):
    from slime.algorithms.sdpo import teacher_alignment
    from slime.rollout import sglang_rollout

    class FakeGenerateState:
        def __init__(self, _args):
            self.tokenizer = object()
            self.processor = None

    monkeypatch.setattr(sglang_rollout, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(
        teacher_alignment,
        "build_sdpo_teacher_rollout_data",
        lambda *_args, **_kwargs: {
            "tokens": [[91, 11], [92, 21, 22]],
            "total_lengths": [2, 3],
            "response_lengths": [1, 2],
            "loss_masks": [[1], [1, 1]],
        },
    )
    train_data = {
        "tokens": [[1, 11], [2, 21, 22]],
        "total_lengths": [2, 3],
        "response_lengths": [1, 2],
        "loss_masks": [[1], [1, 1]],
        "sample_indices": [4, 9],
        "sdpo_metadata": [{"row": 0}, {"row": 1}],
        "sdpo_loss_weights": [0.25, 1.75],
        "self_distillation_mask": [1.0, 1.0],
        "sdpo_teacher_prompt_text": ["privileged a", "privileged b"],
        **{field: [f"{field}-0", f"{field}-1"] for field in resail.PRECOMPUTED_FIELDS},
    }

    resail.expand_pr_rows(
        SimpleNamespace(), train_data, retention_weight=0.3
    )

    assert train_data["tokens"] == [[1, 11], [2, 21, 22], [91, 11], [92, 21, 22]]
    assert train_data["total_lengths"] == [2, 3, 2, 3]
    assert train_data["sample_indices"] == [4, 9, -5, -10]
    assert train_data["sdpo_loss_weights"] == pytest.approx([0.5, 3.5, 0.15, 1.05])
    assert train_data["pr_base_weight"] == pytest.approx([0.25, 1.75, 0.25, 1.75])
    assert train_data["pr_component"] == [0.0, 0.0, 1.0, 1.0]
    assert train_data["self_distillation_mask"] == [1.0] * 4
    for field in resail.PRECOMPUTED_FIELDS:
        assert train_data[field][2:] == [None, None]
    assert train_data["sdpo_metadata"][2:] == [
        {"row": 0, "sdpo_objective_component": "privileged_retention"},
        {"row": 1, "sdpo_objective_component": "privileged_retention"},
    ]


@pytest.mark.unit
def test_sgs_selector_ranks_each_kl_direction_independently():
    rows = [
        {
            "source_draw_id": 0,
            "turn_idx": 0,
            "traj_uid": "a",
            "teacher_js": 0.4,
            "teacher_kl_plain_privileged": 0.9,
            "teacher_kl_privileged_plain": 0.1,
            "distillation_kl": 0.3,
        },
        {
            "source_draw_id": 1,
            "turn_idx": 0,
            "traj_uid": "b",
            "teacher_js": 0.8,
            "teacher_kl_plain_privileged": 0.2,
            "teacher_kl_privileged_plain": 0.95,
            "distillation_kl": 0.1,
        },
        {
            "source_draw_id": 2,
            "turn_idx": 0,
            "traj_uid": "c",
            "teacher_js": 0.2,
            "teacher_kl_plain_privileged": 0.1,
            "teacher_kl_privileged_plain": 0.3,
            "distillation_kl": 0.99,
        },
    ]
    expected = {
        "teacher_js": ((1, 0),),
        "teacher_kl_plain_privileged": ((0, 0),),
        "teacher_kl_privileged_plain": ((1, 0),),
        "distillation_kl": ((2, 0),),
    }
    for field, keys in expected.items():
        selected = select_sgs_steps(
            rows, fraction=0.01, minimum_selected=1, score_field=field
        )
        assert selected.selected_keys == keys
    broken = [dict(rows[0], distillation_kl=None), *rows[1:]]
    assert select_sgs_steps(
        broken, fraction=0.5, minimum_selected=1, score_field="distillation_kl"
    ).scoreable_count == 2


@pytest.mark.unit
def test_sgs_teacher_kl_fields_preserve_direction():
    plain_probabilities = torch.tensor([[0.8, 0.1, 0.1]])
    privileged_probabilities = torch.tensor([[0.5, 0.25, 0.25]])
    plain = {
        "topk_log_probs": plain_probabilities[:, :2].log(),
        "tail_log_probs": plain_probabilities[:, 2].log(),
    }
    privileged = {
        "topk_log_probs": privileged_probabilities[:, :2].log(),
        "tail_log_probs": privileged_probabilities[:, 2].log(),
    }
    metrics = paired_compressed_view_metrics(plain, privileged)
    expected_plain_privileged = sum(
        float(p) * math.log(float(p / q))
        for p, q in zip(plain_probabilities[0], privileged_probabilities[0], strict=True)
    )
    expected_privileged_plain = sum(
        float(q) * math.log(float(q / p))
        for p, q in zip(plain_probabilities[0], privileged_probabilities[0], strict=True)
    )
    assert float(metrics["kl_left_right"][0]) == pytest.approx(expected_plain_privileged)
    assert float(metrics["kl_right_left"][0]) == pytest.approx(expected_privileged_plain)
    assert expected_plain_privileged != pytest.approx(expected_privileged_plain)


@pytest.mark.unit
def test_random_selection_skips_scores_and_uses_selected_row_converter(monkeypatch):
    args = SimpleNamespace(
        sgs_selection_fraction=0.05,
        sgs_skip_full_scoring=True,
        tlb_loss_aggregation="trajectory_balanced",
        sgs_selection_mode="random",
        rollout_batch_size=32,
    )
    assert not sgs_scoring_required(args)
    assert sgs_ranking_score_field(SimpleNamespace(sgs_ranking_score=None)) == "teacher_js"
    samples = []
    for draw in range(8):
        sample = Sample(response_length=2, metadata={"source_draw_id": draw, "source_turn_idx": 0})
        sample.sgs_finalized = True
        sample.sgs_scored = False
        sample.sgs_selected = True
        sample.sgs_score_record = {"response_token_count": 2}
        sample.sgs_batch_metrics = {"scoring_wall_seconds": 0.0}
        samples.append(sample)
    monkeypatch.setattr(resail, "default_sdpo_converter", lambda _args, _samples: {})
    data = convert_samples_to_train_data(args, samples)
    assert all(weight == pytest.approx(0.25) for weight in data["sdpo_loss_weights"])
    assert not any(field in data for field in resail.PRECOMPUTED_FIELDS)
    attempted = [
        {"source_draw_id": index, "turn_idx": 0, "traj_uid": f"t{index}"}
        for index in range(100)
    ]
    selected = select_sgs_steps(
        attempted,
        fraction=0.05,
        minimum_selected=8,
        selection_mode="random",
        selection_seed=42,
        rollout_id=0,
        score_field=None,
    )
    assert selected.requested_count == 5
    assert selected.selected_count == 8
    assert selected.dp_floor_applied




@pytest.mark.unit
def test_random_selection_persists_complete_unscored_audit(
    tmp_path: Path, monkeypatch
):
    from slime.ray.rollout import RolloutManager

    manager_class = RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_class)
    manager.args = SimpleNamespace(
        sgs_selection_fraction=0.05,
        sgs_audit_dir=str(tmp_path),
        tlb_loss_aggregation="trajectory_balanced",
        sgs_selection_seed=42,
        rollout_batch_size=32,
    )
    manager.train_parallel_config = {"dp_size": 8}
    monkeypatch.setattr(
        resail,
        "default_sdpo_converter",
        lambda _args, samples: {"row_count": len(samples)},
    )
    manager._convert_samples_to_train_data = lambda samples: convert_samples_to_train_data(
        manager.args, samples
    )
    manager._split_train_data_by_dp = lambda data: data
    samples = []
    for index in range(20):
        sample = Sample(
            response_length=3,
            metadata={
                "source_draw_id": index,
                "source_turn_idx": 0,
                "source_trajectory_uid": f"trajectory-{index}",
                "task_id": f"task-{index}",
                "outcome": "success" if index % 2 else "failure",
            },
        )
        sample.train_metadata = {
            "sdpo": {
                "frozen_trajectory_length": index + 1,
                "sgs_action_token_mask": [0, 1, 1],
            }
        }
        samples.append(sample)
    data = manager._finalize_unscored_random_topx(7, samples)
    audit = json.loads((tmp_path / "rollout_007.json").read_text())
    assert data["row_count"] == 8
    assert audit["scoring_performed"] is False
    assert audit["scoring_wall_seconds"] == 0.0
    assert audit["selected_steps"] == 8
    assert audit["selected_action_tokens"] == 16
    assert audit["selected_response_tokens"] == 24
    assert all(row["outcome"] in {"success", "failure"} for row in audit["rows"])
    assert all(row["trajectory_length"] > 0 for row in audit["rows"])
    assert all(row.get("teacher_js") is None for row in audit["rows"])


def test_frozen_sdpo_metadata_preserves_canonical_source_trajectory_uid():
    from slime_plugins.agent_tasks.alfworld.frozen.generate import _sdpo_metadata

    trajectory = {
        "trajectory_uid": "canonical-source-trajectory",
        "task_id": "task-1",
        "task_description": "test task",
        "outcome": "success",
        "success": True,
        "turns": [{"turn_idx": 0, "frozen_action": "finish", "next_observation": "done"}],
    }
    turn = {
        "turn_idx": 0,
        "messages": [{"role": "user", "content": "observation"}],
        "current_observation": "observation",
        "next_observation": "done",
        "frozen_action": "finish",
        "errors": [],
    }
    row_metadata = {"traj_uid": "runtime-branch-trajectory", "frozen_arm": "iterative_trajectory_distillation"}

    metadata = _sdpo_metadata(trajectory, turn, row_metadata)

    assert metadata["source_trajectory_uid"] == "canonical-source-trajectory"
    assert metadata["frozen_trajectory_uid"] == "canonical-source-trajectory"
    assert metadata["traj_uid"] == "runtime-branch-trajectory"








