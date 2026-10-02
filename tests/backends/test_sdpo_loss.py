from __future__ import annotations

import importlib
import math

import pytest
import torch


def _find_loss_function():
    candidates = (
        ("slime.algorithms.sdpo.loss", ("compute_sdpo_token_logprob_loss", "sdpo_token_logprob_loss")),
        ("slime.backends.megatron_utils.loss", ("compute_sdpo_token_logprob_loss",)),
    )
    attempted: list[str] = []
    for module_name, names in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            attempted.append(f"{module_name}: {exc}")
            continue
        for name in names:
            fn = getattr(module, name, None)
            if callable(fn):
                return fn
        attempted.append(f"{module_name}: missing {names}")
    pytest.fail(f"SDPO token-logprob loss implementation missing. Tried: {attempted}")


def _loss_value(result):
    if isinstance(result, tuple):
        return float(result[0])
    if isinstance(result, dict):
        return float(result["loss"])
    return float(result)


def _loss_tensor(result):
    if isinstance(result, tuple):
        return result[0]
    if isinstance(result, dict):
        return result["loss"]
    return result


def _call_loss(**overrides):
    kwargs = {
        "student_log_probs": [[-1.0, -2.0], [-0.5, -0.25]],
        "sdpo_teacher_log_probs": [[-2.0, -1.0], [-0.25, -0.75]],
        "loss_masks": [[1.0, 1.0], [1.0, 0.0]],
        "self_distillation_mask": [1.0, 1.0],
        "sdpo_loss_weights": [1.0, 2.0],
        "clip_ratio": None,
    }
    kwargs.update(overrides)
    return _find_loss_function()(**kwargs)


@pytest.mark.unit
def test_token_logprob_reverse_kl_surrogate_matches_plan_formula():
    result = _call_loss(
        student_log_probs=[[-1.0, -2.0]],
        sdpo_teacher_log_probs=[[-2.0, -1.0]],
        loss_masks=[[1.0, 1.0]],
        self_distillation_mask=[1.0],
        sdpo_loss_weights=[1.0],
    )

    expected = ((-1.0 - -2.0) * -1.0 + (-2.0 - -1.0) * -2.0) / 2.0
    assert _loss_value(result) == pytest.approx(expected)


@pytest.mark.unit
def test_token_logprob_mask_and_sample_weights_apply_before_reduction():
    result = _call_loss()

    expected_terms = [
        (-1.0 - -2.0) * -1.0 * 1.0,
        (-2.0 - -1.0) * -2.0 * 1.0,
        (-0.5 - -0.25) * -0.5 * 2.0,
    ]
    assert _loss_value(result) == pytest.approx(sum(expected_terms) / 3.0)


@pytest.mark.unit
def test_token_logprob_clip_ratio_applies_official_current_over_old_upper_cap():
    result = _call_loss(
        student_log_probs=[[-3.0, -0.1]],
        sdpo_teacher_log_probs=[[-1.0, -2.1]],
        loss_masks=[[1.0, 1.0]],
        self_distillation_mask=[1.0],
        sdpo_loss_weights=[1.0],
        old_log_probs=[[-4.0, -0.1]],
        clip_ratio=2.0,
    )

    expected = (((-3.0 - -1.0) * -3.0) * 2.0 + ((-0.1 - -2.1) * -0.1) * 1.0) / 2.0
    assert _loss_value(result) == pytest.approx(expected)


@pytest.mark.unit
def test_token_logprob_combines_policy_clip_and_old_over_deployment_tis():
    result = _call_loss(
        student_log_probs=[[-2.0]],
        sdpo_teacher_log_probs=[[-1.0]],
        loss_masks=[[1.0]],
        self_distillation_mask=[1.0],
        sdpo_loss_weights=[1.0],
        old_log_probs=[[-3.0]],
        deployment_behavior_log_probs=[[-5.0]],
        clip_ratio=2.0,
        deployment_tis_clip=2.0,
    )

    assert _loss_value(result) == pytest.approx(((-2.0 - -1.0) * -2.0) * 2.0 * 2.0)


@pytest.mark.unit
def test_token_logprob_clip_requires_frozen_old_log_probs():
    with pytest.raises(ValueError, match="old_log_probs"):
        _call_loss(clip_ratio=2.0)


@pytest.mark.unit
def test_token_logprob_surrogate_gradient_uses_detached_log_ratio():
    student = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    teacher = torch.tensor([[-2.0, -1.0]])

    loss = _loss_tensor(
        _call_loss(
            student_log_probs=student,
            sdpo_teacher_log_probs=teacher,
            loss_masks=torch.tensor([[1.0, 1.0]]),
            self_distillation_mask=torch.tensor([1.0]),
            sdpo_loss_weights=torch.tensor([1.0]),
            clip_ratio=None,
        )
    )
    loss.backward()

    torch.testing.assert_close(student.grad, torch.tensor([[0.5, -0.5]]))


@pytest.mark.unit
def test_token_logprob_all_zero_self_distillation_mask_returns_finite_zero():
    result = _call_loss(
        loss_masks=[[1.0, 1.0], [1.0, 1.0]],
        self_distillation_mask=[0.0, 0.0],
        sdpo_loss_weights=[1.0, 1.0],
    )
    loss = _loss_value(result)

    assert math.isfinite(loss)
    assert loss == pytest.approx(0.0)


@pytest.mark.unit
def test_sdpo_loss_requires_sdpo_teacher_log_probs_not_opd_teacher_key():
    fn = _find_loss_function()

    with pytest.raises(
        (KeyError, ValueError, AssertionError, TypeError), match="sdpo_teacher_log_probs|teacher_log_probs"
    ):
        fn(
            student_log_probs=[[-1.0]],
            teacher_log_probs=[[-2.0]],
            loss_masks=[[1.0]],
            self_distillation_mask=[1.0],
            sdpo_loss_weights=[1.0],
        )
