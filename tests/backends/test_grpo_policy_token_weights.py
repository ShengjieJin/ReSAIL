from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils import loss as loss_module
from slime.backends.megatron_utils.loss import _apply_grpo_policy_token_weights


NUM_GPUS = 0


@pytest.mark.unit
def test_grpo_policy_token_weights_disabled_preserves_pg_loss():
    pg_loss = torch.tensor([1.0, 2.0], requires_grad=True)

    weighted, unweighted = _apply_grpo_policy_token_weights(
        SimpleNamespace(grpo_token_weights=False),
        {},
        pg_loss,
    )

    assert weighted is pg_loss
    assert unweighted is pg_loss


@pytest.mark.unit
def test_grpo_policy_token_weights_apply_row_aligned_multipliers():
    pg_loss = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    batch = {
        "response_lengths": [2, 1],
        "grpo_policy_token_weights": [
            torch.tensor([0.5, 1.5]),
            torch.tensor([2.0]),
        ],
    }

    weighted, unweighted = _apply_grpo_policy_token_weights(
        SimpleNamespace(grpo_token_weights=True),
        batch,
        pg_loss,
    )

    torch.testing.assert_close(weighted, torch.tensor([0.5, 3.0, 6.0]))
    assert unweighted is pg_loss
    weighted.sum().backward()
    torch.testing.assert_close(pg_loss.grad, torch.tensor([0.5, 1.5, 2.0]))


@pytest.mark.unit
def test_grpo_policy_token_weights_all_ones_match_vanilla_gradient():
    vanilla = torch.tensor([1.0, -2.0], requires_grad=True)
    weighted_input = vanilla.detach().clone().requires_grad_(True)

    weighted, _ = _apply_grpo_policy_token_weights(
        SimpleNamespace(grpo_token_weights=True),
        {"response_lengths": [2], "grpo_policy_token_weights": [torch.ones(2)]},
        weighted_input,
    )
    weighted.sum().backward()
    vanilla.sum().backward()

    torch.testing.assert_close(weighted, vanilla.detach())
    torch.testing.assert_close(weighted_input.grad, vanilla.grad)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("batch", "message"),
    [
        ({"response_lengths": [2]}, "requires batch key"),
        ({"response_lengths": [2], "grpo_policy_token_weights": [torch.ones(1)]}, "row 0 length mismatch"),
        (
            {"response_lengths": [1, 1], "grpo_policy_token_weights": [torch.ones(2), torch.empty(0)]},
            "row 0 length mismatch",
        ),
        (
            {"response_lengths": [2], "grpo_policy_token_weights": [torch.tensor([float("nan"), 1.0])]},
            "finite",
        ),
        ({"response_lengths": [2], "grpo_policy_token_weights": [torch.tensor([-1.0, 1.0])]}, "non-negative"),
    ],
)
def test_grpo_policy_token_weights_reject_invalid_rows(batch, message):
    with pytest.raises((KeyError, ValueError), match=message):
        _apply_grpo_policy_token_weights(
            SimpleNamespace(grpo_token_weights=True),
            batch,
            torch.ones(2),
        )


@pytest.mark.unit
def test_policy_loss_weights_post_tis_pg_only_and_preserves_other_terms(monkeypatch):
    monkeypatch.setattr(loss_module, "all_gather_with_cp", lambda row, *_args: row)
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (
            None,
            {"log_probs": [torch.tensor([0.2, 0.4])], "entropy": [torch.tensor([4.0, 6.0])]},
        ),
    )
    monkeypatch.setattr(
        loss_module,
        "compute_policy_loss",
        lambda *_args, **_kwargs: (torch.tensor([2.0, 4.0]), torch.tensor([0.25, 0.5])),
    )
    monkeypatch.setattr(
        loss_module,
        "compute_opsm_mask",
        lambda **_kwargs: (torch.tensor([1.0, 0.5]), torch.tensor(0.125)),
    )

    def fake_tis(**kwargs):
        torch.testing.assert_close(kwargs["pg_loss"], torch.tensor([2.0, 2.0]))
        return torch.tensor([3.0, 0.0]), [torch.tensor([1.0, 0.0])], {"tis": torch.tensor([9.0, 9.0])}

    monkeypatch.setattr(loss_module, "vanilla_tis_function", fake_tis)
    monkeypatch.setattr(
        loss_module,
        "get_sum_of_sample_mean",
        lambda *_args, **_kwargs: lambda values: values.reshape(-1)[0] / 2.0,
    )
    monkeypatch.setattr(loss_module, "compute_approx_kl", lambda *_args, **_kwargs: torch.tensor([7.0, 11.0]))

    base_args = dict(
        use_rollout_logprobs=False,
        use_opsm=True,
        advantage_estimator="grpo",
        eps_clip=0.2,
        eps_clip_high=0.28,
        get_mismatch_metrics=False,
        use_tis=True,
        custom_tis_function_path=None,
        calculate_per_token_loss=False,
        qkv_format="thd",
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.5,
        use_kl_loss=True,
        use_unbiased_kl=False,
        kl_loss_type="low_var_kl",
        kl_loss_coef=0.1,
    )
    batch = {
        "advantages": [torch.tensor([1.0, 1.0])],
        "log_probs": [torch.tensor([0.1, 0.1])],
        "rollout_log_probs": [torch.tensor([0.1, 0.1])],
        "ref_log_probs": [torch.tensor([0.0, 0.0])],
        "unconcat_tokens": [torch.tensor([1, 2, 3])],
        "response_lengths": [2],
        "total_lengths": [3],
        "loss_masks": [torch.ones(2)],
        "rollout_mask_sums": [2],
        "grpo_policy_token_weights": [torch.tensor([2.0, 0.5])],
    }
    original_reducer = lambda values: values.mean()

    weighted_loss, weighted_metrics = loss_module.policy_loss_function(
        SimpleNamespace(**base_args, grpo_token_weights=True),
        batch,
        torch.empty(0),
        original_reducer,
    )
    vanilla_loss, vanilla_metrics = loss_module.policy_loss_function(
        SimpleNamespace(**base_args, grpo_token_weights=False),
        batch,
        torch.empty(0),
        original_reducer,
    )

    assert weighted_metrics["pg_loss"] == pytest.approx(3.0)
    assert weighted_metrics["grpo_unweighted_pg_loss"] == pytest.approx(1.5)
    assert vanilla_metrics["pg_loss"] == pytest.approx(1.5)
    assert weighted_loss - vanilla_loss == pytest.approx(1.5)
    for key in ("entropy_loss", "pg_clipfrac", "ppo_kl", "kl_loss", "opsm_clipfrac"):
        torch.testing.assert_close(weighted_metrics[key], vanilla_metrics[key])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
