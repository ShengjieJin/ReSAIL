from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
from slime.backends.megatron_utils import loss as loss_module
from slime.backends.megatron_utils.loss import _sdpo_active_prefix_count_from_masks
from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss
from slime_plugins.agent_tasks.common.algorithms.sdpo_context import (
    _normalize_active_weights,
    _raw_loss_weight,
)


NUM_GPUS = 0


def _step_equal_shard(response_losses: list[list[float]]) -> tuple[float, int]:
    response_lengths = [len(row) for row in response_losses]
    masks = [torch.ones(length) for length in response_lengths]
    reducer = get_sum_of_sample_mean(
        response_lengths,
        response_lengths,
        masks,
        sample_denoms=None,
        calculate_per_token_loss=False,
        qkv_format="thd",
    )
    values = torch.tensor([value for row in response_losses for value in row], dtype=torch.float32)
    numerator = float(reducer(values))
    denominator = int(_sdpo_active_prefix_count_from_masks(masks))
    return numerator, denominator


@pytest.mark.unit
def test_step_equal_uses_one_global_denominator_across_uneven_dp_and_microbatches(monkeypatch):
    from slime.backends.megatron_utils import cp_utils

    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: 1)
    # Unequal response lengths, unequal rank row counts, and unequal microbatches.
    shards = [
        [[1.0], [2.0, 2.0, 2.0]],
        [[4.0, 4.0]],
        [[8.0, 8.0], [16.0], [32.0, 32.0, 32.0, 32.0]],
    ]
    parts = [_step_equal_shard(shard) for shard in shards]
    actual = sum(value for value, _ in parts) / sum(count for _, count in parts)
    expected = (1.0 + 2.0 + 4.0 + 8.0 + 16.0 + 32.0) / 6.0
    assert actual == pytest.approx(expected)
    # This is deliberately not the mean of rank-local means.
    assert actual != pytest.approx(sum(value / count for value, count in parts) / len(parts))


@pytest.mark.unit
def test_b3_and_b4_share_unweighted_step_equal_contract():
    rows = [
        {"episode_lengths": length, "algorithm_active_mask": True}
        for length in (1, 3, 3, 5, 5, 5)
    ]
    args = SimpleNamespace(sdpo_multi_turn_weighting="traj_equal")
    for row in rows:
        row["sdpo_loss_weights"] = _raw_loss_weight(args, row)
    _normalize_active_weights(rows)
    assert [row["sdpo_loss_weights"] for row in rows] == pytest.approx([1.0] * len(rows))


@pytest.mark.unit
def test_b6_response_then_trajectory_equal_is_invariant_to_step_and_response_lengths():
    trajectory_losses = {
        "short": [[2.0]],
        "medium": [[3.0, 3.0, 3.0], [5.0]],
        "long": [[7.0], [11.0, 11.0], [13.0], [17.0, 17.0, 17.0, 17.0]],
    }
    rows = []
    for trajectory, response_rows in trajectory_losses.items():
        for response in response_rows:
            rows.append(
                {
                    "trajectory": trajectory,
                    "episode_lengths": len(response_rows),
                    "response_mean": sum(response) / len(response),
                    "algorithm_active_mask": True,
                }
            )
    args = SimpleNamespace(sdpo_multi_turn_weighting="inverse_length")
    for row in rows:
        row["sdpo_loss_weights"] = _raw_loss_weight(args, row)
    _normalize_active_weights(rows)
    global_step_denominator = len(rows)
    actual = sum(row["response_mean"] * row["sdpo_loss_weights"] for row in rows) / global_step_denominator
    expected = sum(
        sum(sum(response) / len(response) for response in responses) / len(responses)
        for responses in trajectory_losses.values()
    ) / len(trajectory_losses)
    assert actual == pytest.approx(expected)


@pytest.mark.unit
def test_step_equal_empty_microbatch_keeps_zero_raw_denominator(monkeypatch):
    from slime.backends.megatron_utils import cp_utils

    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: 1)
    assert int(_sdpo_active_prefix_count_from_masks([])) == 0
    assert int(_sdpo_active_prefix_count_from_masks([torch.zeros(3), torch.ones(1)])) == 1


@pytest.mark.unit
def test_b2_production_policy_and_aux_kl_share_global_step_denominator(monkeypatch):
    """Exercise the production policy-loss dispatch, including constant groups.

    The 24 responses are three source steps with eight branches each.  The
    first two groups have zero advantage (the all-zero/all-one GRPO case), but
    remain in the denominator and in auxiliary KL.  Rows are split 5/9/10 to
    model uneven DP/microbatch ownership, with deliberately unequal response
    lengths.
    """

    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)

    def fake_log_probs_and_entropy(logits, *, response_lengths, **_kwargs):
        rows = list(logits.reshape(-1).split(response_lengths))
        return None, {
            "log_probs": rows,
            "entropy": [torch.zeros_like(row) for row in rows],
        }

    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", fake_log_probs_and_entropy)

    rows = []
    for step in range(3):
        for branch in range(8):
            length = 1 + ((step * 3 + branch) % 4)
            # Constant-reward groups naturally carry zero advantages.  The
            # non-constant group retains all eight branches.
            advantage = 0.0 if step < 2 else (-0.5 if branch % 2 == 0 else 0.5)
            current = torch.tensor(
                [0.015 * (branch - 3.5) + 0.005 * token for token in range(length)],
                dtype=torch.float32,
            )
            reference = current - torch.tensor(
                [0.03 * (1 + ((step + branch + token) % 3)) for token in range(length)],
                dtype=torch.float32,
            )
            rows.append(
                {
                    "length": length,
                    "advantage": advantage,
                    "current": current,
                    "old": torch.zeros(length),
                    "reference": reference,
                }
            )

    args = SimpleNamespace(
        loss_type="policy_loss",
        policy_loss_agg_mode="step_equal",
        sdpo_loss_agg_mode="turn_mean",
        # This flag makes Megatron aggregate the returned response counts
        # across all microbatches/DP ranks before scaling backward.
        calculate_per_token_loss=True,
        qkv_format="thd",
        recompute_loss_function=False,
        allgather_cp=False,
        use_rollout_logprobs=False,
        use_opsm=False,
        advantage_estimator="grpo",
        eps_clip=0.2,
        eps_clip_high=0.28,
        get_mismatch_metrics=False,
        use_tis=False,
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.0,
        use_kl_loss=True,
        use_unbiased_kl=False,
        kl_loss_type="low_var_kl",
        kl_loss_coef=0.01,
        grpo_token_weights=False,
    )

    shard_results = []
    shard_logits = []
    for shard in (rows[:5], rows[5:14], rows[14:]):
        response_lengths = [row["length"] for row in shard]
        logits = torch.cat([row["current"] for row in shard]).clone().requires_grad_(True)
        batch = {
            "advantages": [
                torch.full((row["length"],), row["advantage"], dtype=torch.float32) for row in shard
            ],
            "log_probs": [row["old"] for row in shard],
            "ref_log_probs": [row["reference"] for row in shard],
            "unconcat_tokens": [torch.zeros(row["length"] + 1, dtype=torch.long) for row in shard],
            "response_lengths": response_lengths,
            "total_lengths": [length + 1 for length in response_lengths],
            "loss_masks": [torch.ones(length) for length in response_lengths],
            "rollout_mask_sums": response_lengths,
        }
        scaled_loss, active_responses, log = loss_module.loss_function(
            args,
            batch,
            num_microbatches=3,
            step_global_batch_size=24,
            logits=logits,
        )
        shard_results.append((scaled_loss, int(active_responses), log))
        shard_logits.append(logits)

    denominator = sum(count for _, count, _ in shard_results)
    assert denominator == 24
    actual_loss = sum(loss for loss, _, _ in shard_results) / denominator
    rank_local_mean_then_mean = torch.stack(
        [loss / count for loss, count, _ in shard_results]
    ).mean()
    actual_logs = {}
    for key_index, key in enumerate(shard_results[0][2]["keys"], start=1):
        actual_logs[key] = sum(result[2]["values"][key_index] for result in shard_results) / denominator

    response_pg = []
    response_kl = []
    for row in rows:
        ppo_kl = row["old"] - row["current"]
        advantages = torch.full_like(row["current"], row["advantage"])
        pg, _ = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)
        response_pg.append(pg.mean())
        response_kl.append(
            compute_approx_kl(row["current"], row["reference"], args.kl_loss_type).mean()
        )
    expected_pg = torch.stack(response_pg).mean()
    expected_kl = torch.stack(response_kl).mean()
    expected_loss = expected_pg + args.kl_loss_coef * expected_kl

    torch.testing.assert_close(actual_logs["pg_loss"], expected_pg)
    torch.testing.assert_close(actual_logs["kl_loss"], expected_kl)
    torch.testing.assert_close(actual_loss, expected_loss)
    assert not torch.isclose(actual_loss.detach(), rank_local_mean_then_mean.detach())
    torch.testing.assert_close(actual_logs["loss"], expected_loss)
    # The zero-advantage groups still contribute 16/24 of the auxiliary-KL
    # denominator; dropping them produces a measurably different answer.
    nonconstant_only_kl = torch.stack(response_kl[16:]).mean()
    assert actual_logs["kl_loss"] != pytest.approx(float(nonconstant_only_kl))

    actual_loss.backward()
    assert all(logits.grad is not None and torch.isfinite(logits.grad).all() for logits in shard_logits)


@pytest.mark.unit
def test_b1_b5_production_sft_uses_global_active_token_denominator(monkeypatch):
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)

    def fake_log_probs_and_entropy(logits, *, response_lengths, **_kwargs):
        return None, {"log_probs": list(logits.reshape(-1).split(response_lengths))}

    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", fake_log_probs_and_entropy)
    token_nll_rows = [
        [1.0],
        [2.0, 4.0, 6.0],
        [3.0, 9.0],
        [5.0, 5.0, 5.0, 5.0],
        [7.0],
        [8.0, 10.0, 12.0],
    ]
    args = SimpleNamespace(
        loss_type="sft_loss",
        calculate_per_token_loss=True,
        qkv_format="thd",
        recompute_loss_function=False,
        allgather_cp=False,
        sdpo_loss_agg_mode="turn_mean",
        policy_loss_agg_mode="rollout_mean",
    )
    results = []
    shard_logits = []
    for shard in (token_nll_rows[:1], token_nll_rows[1:4], token_nll_rows[4:]):
        response_lengths = [len(row) for row in shard]
        logits = -torch.tensor([value for row in shard for value in row], dtype=torch.float32).requires_grad_(True)
        logits.retain_grad()
        batch = {
            "unconcat_tokens": [torch.zeros(length + 1, dtype=torch.long) for length in response_lengths],
            "response_lengths": response_lengths,
            "total_lengths": [length + 1 for length in response_lengths],
            "loss_masks": [torch.ones(length) for length in response_lengths],
            "rollout_mask_sums": response_lengths,
        }
        loss, active_tokens, _ = loss_module.loss_function(
            args,
            batch,
            num_microbatches=3,
            step_global_batch_size=len(token_nll_rows),
            logits=logits,
        )
        results.append((loss, int(active_tokens)))
        shard_logits.append(logits)

    denominator = sum(count for _, count in results)
    actual = sum(loss for loss, _ in results) / denominator
    expected = torch.tensor([value for row in token_nll_rows for value in row]).mean()
    assert denominator == sum(len(row) for row in token_nll_rows)
    torch.testing.assert_close(actual, expected)
    assert float(actual.detach()) != pytest.approx(
        sum(sum(row) / len(row) for row in token_nll_rows) / len(token_nll_rows)
    )
    actual.backward()
    assert all(logits.grad is not None and torch.isfinite(logits.grad).all() for logits in shard_logits)


@pytest.mark.unit
@pytest.mark.parametrize("weighting", ["traj_equal", "inverse_length"])
def test_b3_b4_b6_production_sdpo_global_reducer_across_uneven_shards(monkeypatch, weighting):
    """Use the production SDPO dispatcher for step-equal and trajectory-equal loss."""

    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)

    def fake_log_probs_and_entropy(logits, *, response_lengths, **_kwargs):
        rows = list(logits.reshape(-1).split(response_lengths))
        return None, {"log_probs": rows}

    def fake_distillation_tensors(logits, *, response_lengths, **_kwargs):
        rows = list(logits.reshape(-1).split(response_lengths))
        topk_rows = []
        for row in rows:
            probability = torch.sigmoid(row)
            topk_rows.append(
                torch.stack(
                    [torch.log(0.55 * probability), torch.log(0.25 * (1.0 - probability))], dim=-1
                )
            )
        return None, {"sdpo_topk_log_probs": topk_rows}

    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", fake_log_probs_and_entropy)
    monkeypatch.setattr(loss_module, "get_sdpo_distillation_tensors", fake_distillation_tensors)

    rows = []
    for trajectory_index in range(32):
        trajectory_steps = 1 + (trajectory_index % 4)
        for step_index in range(trajectory_steps):
            length = 1 + ((trajectory_index + step_index) % 4)
            student = torch.tensor(
                [0.02 * (trajectory_index - 15.5) + 0.03 * (step_index + token) for token in range(length)],
                dtype=torch.float32,
            )
            teacher_first = 0.18 + 0.003 * ((trajectory_index + step_index) % 7)
            teacher_second = 0.12 + 0.002 * ((trajectory_index + 2 * step_index) % 5)
            teacher = torch.log(
                torch.tensor([[teacher_first, teacher_second]] * length, dtype=torch.float32)
            )
            rows.append(
                {
                    "trajectory_index": trajectory_index,
                    "trajectory_steps": trajectory_steps,
                    "student": student,
                    "teacher_topk": teacher,
                }
            )

    weight_rows = [
        {"episode_lengths": row["trajectory_steps"], "algorithm_active_mask": True} for row in rows
    ]
    weight_args = SimpleNamespace(sdpo_multi_turn_weighting=weighting)
    for row in weight_rows:
        row["sdpo_loss_weights"] = _raw_loss_weight(weight_args, row)
    _normalize_active_weights(weight_rows)
    for row, weight_row in zip(rows, weight_rows, strict=True):
        row["weight"] = weight_row["sdpo_loss_weights"]

    args = SimpleNamespace(
        loss_type="sdpo_loss",
        calculate_per_token_loss=True,
        qkv_format="thd",
        recompute_loss_function=False,
        allgather_cp=False,
        sdpo_loss_agg_mode="step_equal",
        policy_loss_agg_mode="rollout_mean",
        sdpo_distillation_mode="topk",
        sdpo_full_logit_distillation=True,
        sdpo_distillation_topk=2,
        sdpo_distillation_add_tail=True,
        sdpo_alpha=1.0,
        sdpo_clip_ratio=None,
        sdpo_deployment_tis_clip=None,
    )
    shard_results = []
    shard_logits = []
    for shard in (rows[:17], rows[17:46], rows[46:]):
        response_lengths = [len(row["student"]) for row in shard]
        logits = torch.cat([row["student"] for row in shard]).clone().requires_grad_(True)
        batch = {
            "unconcat_tokens": [torch.zeros(length + 1, dtype=torch.long) for length in response_lengths],
            "response_lengths": response_lengths,
            "total_lengths": [length + 1 for length in response_lengths],
            "loss_masks": [torch.ones(length) for length in response_lengths],
            "rollout_mask_sums": response_lengths,
            "sdpo_teacher_log_probs": [torch.zeros(length) for length in response_lengths],
            "sdpo_topk_indices": [torch.zeros((length, 2), dtype=torch.long) for length in response_lengths],
            "sdpo_teacher_topk_log_probs": [row["teacher_topk"] for row in shard],
            "self_distillation_mask": torch.ones(len(shard)),
            "sdpo_loss_weights": torch.tensor([row["weight"] for row in shard]),
        }
        loss, active_steps, _ = loss_module.loss_function(
            args,
            batch,
            num_microbatches=3,
            step_global_batch_size=len(rows),
            logits=logits,
        )
        shard_results.append((loss, int(active_steps)))
        shard_logits.append(logits)

    denominator = sum(count for _, count in shard_results)
    actual = sum(loss for loss, _ in shard_results) / denominator
    response_means = []
    for row in rows:
        probability = torch.sigmoid(row["student"])
        student_topk = torch.stack(
            [torch.log(0.55 * probability), torch.log(0.25 * (1.0 - probability))], dim=-1
        )
        student_with_tail = loss_module._add_sdpo_tail_bucket(student_topk)
        teacher_with_tail = loss_module._add_sdpo_tail_bucket(row["teacher_topk"])
        response_means.append(
            loss_module._compute_sdpo_kl_loss(student_with_tail, teacher_with_tail, alpha=1.0).mean()
        )
    if weighting == "traj_equal":
        expected = torch.stack(response_means).mean()
    else:
        trajectory_means = []
        for trajectory_index in range(32):
            selected = [
                value
                for row, value in zip(rows, response_means, strict=True)
                if row["trajectory_index"] == trajectory_index
            ]
            trajectory_means.append(torch.stack(selected).mean())
        expected = torch.stack(trajectory_means).mean()
    assert denominator == len(rows)
    torch.testing.assert_close(actual, expected)
    rank_local_mean = sum(loss / count for loss, count in shard_results) / len(shard_results)
    assert float(actual.detach()) != pytest.approx(float(rank_local_mean.detach()))
    actual.backward()
    assert all(logits.grad is not None and torch.isfinite(logits.grad).all() for logits in shard_logits)


@pytest.mark.unit
def test_b13_production_sdpo_masks_mismatches_with_fixed_t32_outer_mean(monkeypatch):
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)

    def fake_log_probs_and_entropy(logits, *, response_lengths, **_kwargs):
        return None, {"log_probs": list(logits.reshape(-1).split(response_lengths))}

    def fake_distillation_tensors(logits, *, response_lengths, **_kwargs):
        rows = list(logits.reshape(-1).split(response_lengths))
        return None, {
            "sdpo_topk_log_probs": [
                torch.stack(
                    [torch.log(0.55 * torch.sigmoid(row)), torch.log(0.25 * (1 - torch.sigmoid(row)))],
                    dim=-1,
                )
                for row in rows
            ]
        }

    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", fake_log_probs_and_entropy)
    monkeypatch.setattr(loss_module, "get_sdpo_distillation_tensors", fake_distillation_tensors)

    rows = []
    for trajectory in range(32):
        step_count = 1 + trajectory % 3
        for step in range(step_count):
            length = 1 + (trajectory + step) % 4
            matched = trajectory != 0 and step % 2 == 0
            student = torch.tensor(
                [0.01 * (trajectory - 15) + 0.02 * (step + token) for token in range(length)]
            )
            teacher = torch.log(torch.tensor([[0.19, 0.13]] * length))
            rows.append(
                {
                    "trajectory": trajectory,
                    "matched": matched,
                    "student": student,
                    "teacher": teacher,
                }
            )
    match_counts = {
        trajectory: sum(int(row["matched"]) for row in rows if row["trajectory"] == trajectory)
        for trajectory in range(32)
    }
    active_count = sum(match_counts.values())
    for row in rows:
        matches = match_counts[row["trajectory"]]
        row["weight"] = active_count / (32 * matches) if row["matched"] else 0.0

    args = SimpleNamespace(
        loss_type="sdpo_loss",
        calculate_per_token_loss=True,
        qkv_format="thd",
        recompute_loss_function=False,
        allgather_cp=False,
        sdpo_loss_agg_mode="step_equal",
        policy_loss_agg_mode="rollout_mean",
        sdpo_distillation_mode="topk",
        sdpo_full_logit_distillation=True,
        sdpo_distillation_topk=2,
        sdpo_distillation_add_tail=True,
        sdpo_alpha=1.0,
        sdpo_clip_ratio=None,
        sdpo_deployment_tis_clip=None,
    )
    shard_results = []
    shard_logits = []
    shard_rows = (rows[:19], rows[19:47], rows[47:])
    for shard in shard_rows:
        lengths = [len(row["student"]) for row in shard]
        logits = torch.cat([row["student"] for row in shard]).clone().requires_grad_(True)
        batch = {
            "unconcat_tokens": [torch.zeros(length + 1, dtype=torch.long) for length in lengths],
            "response_lengths": lengths,
            "total_lengths": [length + 1 for length in lengths],
            "loss_masks": [torch.ones(length) for length in lengths],
            "rollout_mask_sums": lengths,
            "sdpo_teacher_log_probs": [torch.zeros(length) for length in lengths],
            "sdpo_topk_indices": [torch.zeros((length, 2), dtype=torch.long) for length in lengths],
            "sdpo_teacher_topk_log_probs": [row["teacher"] for row in shard],
            "self_distillation_mask": torch.tensor([float(row["matched"]) for row in shard]),
            "sdpo_loss_weights": torch.tensor([row["weight"] for row in shard]),
        }
        loss, active_steps, _ = loss_module.loss_function(
            args, batch, num_microbatches=3, step_global_batch_size=len(rows), logits=logits
        )
        shard_results.append((loss, int(active_steps)))
        shard_logits.append(logits)

    denominator = sum(count for _, count in shard_results)
    assert denominator == active_count
    actual = sum(loss for loss, _ in shard_results) / denominator
    response_means = []
    for row in rows:
        probability = torch.sigmoid(row["student"])
        student = torch.stack(
            [torch.log(0.55 * probability), torch.log(0.25 * (1 - probability))], dim=-1
        )
        response_means.append(
            loss_module._compute_sdpo_kl_loss(
                loss_module._add_sdpo_tail_bucket(student),
                loss_module._add_sdpo_tail_bucket(row["teacher"]),
                alpha=1.0,
            ).mean()
        )
    expected_trajectories = []
    for trajectory in range(32):
        selected = [
            value
            for row, value in zip(rows, response_means, strict=True)
            if row["trajectory"] == trajectory and row["matched"]
        ]
        expected_trajectories.append(torch.stack(selected).mean() if selected else torch.tensor(0.0))
    expected = torch.stack(expected_trajectories).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    for shard, logits in zip(shard_rows, shard_logits, strict=True):
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        offset = 0
        for row in shard:
            length = len(row["student"])
            gradient = logits.grad[offset : offset + length]
            if not row["matched"]:
                assert torch.count_nonzero(gradient) == 0
            offset += length


@pytest.mark.unit
def test_b16_production_sdpo_averages_matched_branches_then_steps_then_fixed_t32(monkeypatch):
    """Exercise the real reducer across uneven shards for the nested trajectory mean."""

    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)

    def fake_log_probs_and_entropy(logits, *, response_lengths, **_kwargs):
        return None, {"log_probs": list(logits.reshape(-1).split(response_lengths))}

    def fake_distillation_tensors(logits, *, response_lengths, **_kwargs):
        rows = list(logits.reshape(-1).split(response_lengths))
        return None, {
            "sdpo_topk_log_probs": [
                torch.stack(
                    [torch.log(0.55 * torch.sigmoid(row)), torch.log(0.25 * (1 - torch.sigmoid(row)))],
                    dim=-1,
                )
                for row in rows
            ]
        }

    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", fake_log_probs_and_entropy)
    monkeypatch.setattr(loss_module, "get_sdpo_distillation_tensors", fake_distillation_tensors)
    rows = []
    for trajectory in range(32):
        for step in range(1 + trajectory % 3):
            matched_branches = 0 if trajectory == 0 or (trajectory + step) % 5 == 0 else 1 + (trajectory + step) % 8
            for branch in range(8):
                length = 1 + (trajectory + step + branch) % 4
                rows.append(
                    {
                        "trajectory": trajectory,
                        "step": step,
                        "branch": branch,
                        "matched": branch < matched_branches,
                        "student": torch.tensor(
                            [0.006 * (trajectory - 15) + 0.011 * (step + branch + token) for token in range(length)]
                        ),
                        "teacher": torch.log(torch.tensor([[0.19, 0.13]] * length)),
                    }
                )
    active_steps = {
        trajectory: len(
            {
                row["step"] for row in rows
                if row["trajectory"] == trajectory and row["matched"]
            }
        )
        for trajectory in range(32)
    }
    matched_by_step = {
        (trajectory, step): sum(
            int(row["matched"]) for row in rows
            if row["trajectory"] == trajectory and row["step"] == step
        )
        for trajectory in range(32)
        for step in range(1 + trajectory % 3)
    }
    active_responses = sum(int(row["matched"]) for row in rows)
    for row in rows:
        matched = matched_by_step[(row["trajectory"], row["step"])]
        row["weight"] = (
            active_responses / (32 * active_steps[row["trajectory"]] * matched)
            if row["matched"] else 0.0
        )

    args = SimpleNamespace(
        loss_type="sdpo_loss", calculate_per_token_loss=True, qkv_format="thd",
        recompute_loss_function=False, allgather_cp=False, sdpo_loss_agg_mode="step_equal",
        policy_loss_agg_mode="rollout_mean", sdpo_distillation_mode="topk",
        sdpo_full_logit_distillation=True, sdpo_distillation_topk=2,
        sdpo_distillation_add_tail=True, sdpo_alpha=1.0, sdpo_clip_ratio=None,
        sdpo_deployment_tis_clip=None,
    )
    shard_rows = (rows[:137], rows[137:389], rows[389:])
    parts = []
    logits_parts = []
    for shard in shard_rows:
        lengths = [len(row["student"]) for row in shard]
        logits = torch.cat([row["student"] for row in shard]).clone().requires_grad_(True)
        batch = {
            "unconcat_tokens": [torch.zeros(length + 1, dtype=torch.long) for length in lengths],
            "response_lengths": lengths,
            "total_lengths": [length + 1 for length in lengths],
            "loss_masks": [torch.ones(length) for length in lengths],
            "rollout_mask_sums": lengths,
            "sdpo_teacher_log_probs": [torch.zeros(length) for length in lengths],
            "sdpo_topk_indices": [torch.zeros((length, 2), dtype=torch.long) for length in lengths],
            "sdpo_teacher_topk_log_probs": [row["teacher"] for row in shard],
            "self_distillation_mask": torch.tensor([float(row["matched"]) for row in shard]),
            "sdpo_loss_weights": torch.tensor([row["weight"] for row in shard]),
        }
        loss, denominator, _ = loss_module.loss_function(
            args, batch, num_microbatches=3, step_global_batch_size=len(rows), logits=logits
        )
        parts.append((loss, int(denominator)))
        logits_parts.append(logits)
    assert sum(count for _, count in parts) == active_responses
    actual = sum(loss for loss, _ in parts) / active_responses
    response_means = []
    for row in rows:
        probability = torch.sigmoid(row["student"])
        student = torch.stack(
            [torch.log(0.55 * probability), torch.log(0.25 * (1 - probability))], dim=-1
        )
        response_means.append(
            loss_module._compute_sdpo_kl_loss(
                loss_module._add_sdpo_tail_bucket(student),
                loss_module._add_sdpo_tail_bucket(row["teacher"]), alpha=1.0,
            ).mean()
        )
    trajectory_means = []
    for trajectory in range(32):
        step_means = []
        for step in range(1 + trajectory % 3):
            selected = [
                value for row, value in zip(rows, response_means, strict=True)
                if row["trajectory"] == trajectory and row["step"] == step and row["matched"]
            ]
            if selected:
                step_means.append(torch.stack(selected).mean())
        trajectory_means.append(torch.stack(step_means).mean() if step_means else torch.tensor(0.0))
    expected = torch.stack(trajectory_means).mean()
    torch.testing.assert_close(actual, expected)
    rank_local_mean = sum(loss / count for loss, count in parts) / len(parts)
    assert float(actual.detach()) != pytest.approx(float(rank_local_mean.detach()))
    actual.backward()
    for shard, logits in zip(shard_rows, logits_parts, strict=True):
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        offset = 0
        for row in shard:
            length = len(row["student"])
            if not row["matched"]:
                assert torch.count_nonzero(logits.grad[offset : offset + length]) == 0
            offset += length


@pytest.mark.unit
def test_b14_production_policy_and_aux_kl_use_only_nonconstant_groups(monkeypatch):
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)

    def fake_log_probs_and_entropy(logits, *, response_lengths, **_kwargs):
        rows = list(logits.reshape(-1).split(response_lengths))
        return None, {"log_probs": rows, "entropy": [torch.zeros_like(row) for row in rows]}

    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", fake_log_probs_and_entropy)
    rows = []
    for group in range(4):
        for branch in range(8):
            length = 1 + (group + branch) % 3
            active = group >= 2
            advantage = (-0.5 if branch % 2 == 0 else 0.5) if active else 0.0
            current = torch.tensor([0.02 * (branch - 3) + 0.01 * token for token in range(length)])
            rows.append(
                {
                    "active": active,
                    "length": length,
                    "advantage": advantage,
                    "current": current,
                    "old": torch.zeros(length),
                    "reference": current - 0.04,
                }
            )
    args = SimpleNamespace(
        loss_type="policy_loss",
        policy_loss_agg_mode="step_equal",
        sdpo_loss_agg_mode="turn_mean",
        calculate_per_token_loss=True,
        qkv_format="thd",
        recompute_loss_function=False,
        allgather_cp=False,
        use_rollout_logprobs=False,
        use_opsm=False,
        advantage_estimator="grpo",
        eps_clip=0.2,
        eps_clip_high=0.28,
        get_mismatch_metrics=False,
        use_tis=False,
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.0,
        use_kl_loss=True,
        use_unbiased_kl=False,
        kl_loss_type="low_var_kl",
        kl_loss_coef=0.01,
        grpo_token_weights=False,
    )
    shard_rows = (rows[:7], rows[7:18], rows[18:])
    parts = []
    logits_parts = []
    for shard in shard_rows:
        lengths = [row["length"] for row in shard]
        logits = torch.cat([row["current"] for row in shard]).clone().requires_grad_(True)
        batch = {
            "advantages": [torch.full((row["length"],), row["advantage"]) for row in shard],
            "log_probs": [row["old"] for row in shard],
            "ref_log_probs": [row["reference"] for row in shard],
            "unconcat_tokens": [torch.zeros(length + 1, dtype=torch.long) for length in lengths],
            "response_lengths": lengths,
            "total_lengths": [length + 1 for length in lengths],
            "loss_masks": [torch.ones(length) if row["active"] else torch.zeros(length) for row, length in zip(shard, lengths, strict=True)],
            "rollout_mask_sums": lengths,
        }
        loss, active_responses, _ = loss_module.loss_function(
            args, batch, num_microbatches=3, step_global_batch_size=len(rows), logits=logits
        )
        parts.append((loss, int(active_responses)))
        logits_parts.append(logits)
    denominator = sum(count for _, count in parts)
    assert denominator == 16
    actual = sum(loss for loss, _ in parts) / denominator
    expected_rows = []
    for row in rows:
        if not row["active"]:
            continue
        ppo_kl = row["old"] - row["current"]
        advantage = torch.full_like(row["current"], row["advantage"])
        pg, _ = compute_policy_loss(ppo_kl, advantage, args.eps_clip, args.eps_clip_high)
        kl = compute_approx_kl(row["current"], row["reference"], args.kl_loss_type)
        expected_rows.append(pg.mean() + args.kl_loss_coef * kl.mean())
    torch.testing.assert_close(actual, torch.stack(expected_rows).mean())
    actual.backward()
    for shard, logits in zip(shard_rows, logits_parts, strict=True):
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        offset = 0
        for row in shard:
            gradient = logits.grad[offset : offset + row["length"]]
            if not row["active"]:
                assert torch.count_nonzero(gradient) == 0
            offset += row["length"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
