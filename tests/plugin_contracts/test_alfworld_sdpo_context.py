from __future__ import annotations

import gzip
import importlib
import json
from types import SimpleNamespace

import pytest

from slime.utils.types import Sample


NUM_GPUS = 0


def _find_function(names: tuple[str, ...], description: str):
    module_name = "slime_plugins.agent_tasks.common.algorithms.sdpo_context"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"SDPO {description} implementation missing. Tried {module_name}: {exc}")
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    pytest.fail(f"SDPO {description} implementation missing. Tried {module_name}: missing {names}")


def _find_sdpo_convert():
    module_name = "slime_plugins.agent_tasks.common.algorithms.sdpo"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"SDPO train-data converter implementation missing. Tried {module_name}: {exc}")
    fn = getattr(module, "convert_samples_to_train_data", None)
    if not callable(fn):
        pytest.fail(f"SDPO train-data converter implementation missing. Tried {module_name}: missing convert")
    return fn


def _find_grpo_token_weight_context_plan():
    return _find_function(
        ("prepare_grpo_token_weight_context_plan",),
        "GRPO token-weight context plan",
    )


def _find_validated_privileged_trajectory():
    return _find_function(("_validated_privileged_trajectory",), "privileged trajectory validation")


@pytest.mark.unit
def test_privileged_trajectory_keeps_present_empty_projected_action_but_rejects_missing_field():
    validate = _find_validated_privileged_trajectory()
    row = {
        "frozen_trajectory_length": 2,
        "sdpo_privileged_trajectory": [
            {"turn_idx": 0, "projected_action": "look", "next_anchor_obs": "room"},
            {"turn_idx": 1, "projected_action": "", "next_anchor_obs": "room"},
        ],
    }

    assert validate(row) == row["sdpo_privileged_trajectory"]

    del row["sdpo_privileged_trajectory"][1]["projected_action"]
    with pytest.raises(ValueError, match="turn 1 is missing projected_action"):
        validate(row)


def _metadata(
    *,
    uid: str = "task-a",
    traj_uid: str = "traj-a",
    turn_idx: int = 0,
    reward: float = 0.0,
    length: int = 1,
    action: str = "look",
    next_obs: str = "next observation",
    raw_prompt: list[dict[str, str]] | None = None,
) -> dict:
    return {
        "uid": uid,
        "traj_uid": traj_uid,
        "turn_idx": turn_idx,
        "task_text": f"Task text for {uid}",
        "anchor_obs": f"anchor obs {turn_idx}",
        "next_anchor_obs": next_obs,
        "projected_action": action,
        "is_action_valid": True,
        "is_terminal": turn_idx + 1 >= length,
        "episode_reward": reward,
        "episode_length": length,
        "sdpo_current_prompt_text": "CURRENT TASK CONTEXT",
        "sdpo_current_raw_prompt": raw_prompt
        or [
            {"role": "system", "content": "SYSTEM RULES"},
            {"role": "user", "content": "CURRENT TASK CONTEXT"},
        ],
    }


def _sample(index: int, metadata: dict) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        tokens=[100 + index, 200 + index],
        response_length=1,
        loss_mask=[1],
        train_metadata={"sdpo": metadata},
        metadata=metadata,
    )


def _train_sample(index: int, metadata: dict) -> Sample:
    sample = Sample(
        index=index,
        rollout_id=index,
        tokens=[100 + index, 200 + index],
        response_length=1,
        reward=float(metadata.get("episode_reward", metadata.get("episode_rewards", 0.0))),
        loss_mask=[1],
        rollout_log_probs=[-0.1],
        train_metadata={
            "uid": metadata["uid"],
            "traj_uid": metadata["traj_uid"],
            "turn_idx": metadata["turn_idx"],
            "agent_task": "alfworld",
            "sample_rollout_id": index,
            "sample_index": index,
            "sdpo": metadata,
        },
        metadata={"raw_reward": float(metadata.get("episode_reward", metadata.get("episode_rewards", 0.0)))},
    )
    sample.status = Sample.Status.COMPLETED
    return sample


def _build_batch() -> list[SimpleNamespace]:
    return [
        _sample(0, _metadata(traj_uid="success-short", turn_idx=0, reward=1.0, length=1, action="open fridge")),
        _sample(1, _metadata(traj_uid="success-long", turn_idx=0, reward=1.0, length=2, action="open cabinet")),
        _sample(2, _metadata(traj_uid="success-long", turn_idx=1, reward=1.0, length=2, action="take mug")),
        _sample(3, _metadata(traj_uid="failure", turn_idx=0, reward=0.0, length=1, action="look around")),
    ]


def _args(**overrides) -> SimpleNamespace:
    defaults = {
        "sdpo_success_reward_threshold": 1.0,
        "sdpo_dont_reprompt_on_self_success": False,
        "sdpo_teacher_context_mode": "original",
        "sdpo_context_prompt_style": "legacy",
        "sdpo_own_outcome_label_mode": "explicit",
        "sdpo_no_success_context_mode": "feedback",
        "sdpo_solution_context_format": "guidance_plan",
        "sdpo_multi_turn_weighting": "traj_equal",
        "sdpo_include_environment_feedback": True,
        "sdpo_environment_feedback_only_without_solution": True,
        "sdpo_max_demo_steps": None,
        "sdpo_max_reprompt_tokens": 10240,
        "sdpo_filter_all_success_groups": False,
        "agent_task_sdpo_metadata_profile": "alfworld",
        "reward_key": None,
        "advantage_estimator": "grpo",
        "rewards_normalization": False,
        "n_samples_per_prompt": 2,
        "rollout_batch_size": 1,
        "grpo_std_normalization": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _mock_guidance_outputs(plan: list[dict]) -> dict[str, str]:
    outputs = {}
    for row in plan:
        for key in ("sdpo_success_guidance_prompt", "sdpo_failure_guidance_prompt"):
            prompt = row.get(key) or ""
            if not prompt or prompt in outputs:
                continue
            if "Successful trajectory evidence:" in prompt:
                action = "open fridge" if "open fridge" in prompt else "successful action"
                outputs[prompt] = (
                    "<thinking>summarize the successful action.</thinking>\n\n"
                    f"Guidance summary:\n- Minimal plan: use `{action}`."
                )
            else:
                action = "look around" if "look around" in prompt else "failed action"
                outputs[prompt] = (
                    "<thinking>identify the failed action.</thinking>\n\n"
                    "Failure analysis:\n"
                    f"- Likely mistake: `{action}` did not complete the task.\n"
                    "- Avoid: repeating the same failed action."
                )
    return outputs


def _plan(samples: list[SimpleNamespace], **overrides) -> list[dict]:
    prepare = _find_function(("prepare_sdpo_context_plan", "build_sdpo_context_plan"), "context plan")
    finalize = _find_function(("finalize_sdpo_context_plan",), "context plan finalizer")
    plan = prepare(_args(**overrides), samples)
    return finalize(plan, _mock_guidance_outputs(plan))


@pytest.mark.unit
def test_alfworld_sdpo_train_metadata_contract_is_complete():
    build_metadata = _find_function(
        ("build_alfworld_sdpo_train_metadata", "extract_alfworld_sdpo_metadata"),
        "AlfWorld train metadata extraction",
    )
    sample = _sample(0, _metadata(action="take apple", next_obs="apple taken"))

    sdpo = build_metadata(sample)

    required = {
        "uid",
        "traj_uid",
        "turn_idx",
        "task_text",
        "anchor_obs",
        "next_anchor_obs",
        "projected_action",
        "is_action_valid",
        "is_terminal",
        "episode_rewards",
        "episode_lengths",
        "sdpo_current_prompt_text",
        "sdpo_current_raw_prompt",
    }
    assert required <= set(sdpo)
    assert sdpo["projected_action"] == "take apple"
    assert sdpo["next_anchor_obs"] == "apple taken"


@pytest.mark.unit
def test_canonical_schema_accepts_singular_and_plural_episode_fields():
    canonicalize = _find_function(
        ("canonicalize_sdpo_metadata", "canonicalize_alfworld_sdpo_metadata"),
        "metadata canonicalizer",
    )
    singular = _metadata(reward=0.5, length=7)
    plural = {**_metadata(reward=0.0, length=1), "episode_rewards": 0.75, "episode_lengths": 4}

    singular_out = canonicalize(singular)
    plural_out = canonicalize(plural)

    assert singular_out["episode_rewards"] == pytest.approx(0.5)
    assert singular_out["episode_lengths"] == 7
    assert plural_out["episode_rewards"] == pytest.approx(0.75)
    assert plural_out["episode_lengths"] == 4


@pytest.mark.unit
def test_success_candidate_priority_matches_reward_length_traj_uid_sort():
    plan = _plan(_build_batch())

    failed_prompt = plan[3]["sdpo_teacher_prompt_text"]

    assert "open fridge" in failed_prompt
    assert "open cabinet" not in failed_prompt
    assert plan[3]["sdpo_teacher_signal_type"] == "solution_guidance"
    assert plan[3]["self_distillation_mask"] == pytest.approx(1.0)


@pytest.mark.unit
def test_dont_reprompt_on_only_self_success_falls_back_to_feedback():
    samples = [
        _sample(0, _metadata(traj_uid="success", turn_idx=0, reward=1.0, length=1, action="open fridge")),
    ]

    plan = _plan(
        samples,
        sdpo_dont_reprompt_on_self_success=True,
        sdpo_solution_context_format="guidance_plan",
    )

    assert plan[0]["self_distillation_mask"] == pytest.approx(1.0)
    assert plan[0]["sdpo_teacher_signal_type"] == "feedback"
    assert "Relevant environment transition" in plan[0]["sdpo_teacher_prompt_text"]
    assert "Lessons from a previous unsuccessful attempt" not in plan[0]["sdpo_teacher_prompt_text"]


@pytest.mark.unit
def test_guidance_summary_source_self_trajectory_uses_each_sample_own_experience():
    plan = _plan(
        _build_batch(),
        sdpo_guidance_summary_source="self_trajectory",
        sdpo_dont_reprompt_on_self_success=True,
    )

    assert plan[0]["sdpo_teacher_signal_type"] == "solution_guidance"
    assert plan[0]["sdpo_success_is_self"] is True
    assert "open fridge" in plan[0]["sdpo_teacher_prompt_text"]
    assert "open cabinet" not in plan[0]["sdpo_teacher_prompt_text"]
    assert plan[3]["sdpo_teacher_signal_type"] == "failure_guidance"
    assert "look around" in plan[3]["sdpo_teacher_prompt_text"]
    assert "Failure analysis:" in plan[3]["sdpo_teacher_prompt_text"]
    assert "Guidance summary:" not in plan[3]["sdpo_teacher_prompt_text"]


@pytest.mark.unit
def test_sdpo_converter_builds_full_batch_context_for_same_uid_success_failure():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(
                uid="task-a",
                traj_uid="success",
                turn_idx=1,
                reward=1.0,
                length=2,
                action="take mug",
                next_obs="mug acquired",
            ),
        ),
        _train_sample(
            1,
            _metadata(
                uid="task-a",
                traj_uid="success",
                turn_idx=0,
                reward=1.0,
                length=2,
                action="open fridge",
                next_obs="fridge is open",
            ),
        ),
        _train_sample(
            2,
            _metadata(
                uid="task-a",
                traj_uid="failure",
                reward=0.0,
                length=1,
                action="look around",
                next_obs="nothing changed",
            ),
        ),
    ]

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=3,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
        ),
        samples,
    )

    task_a_failure_prompt = train_data["sdpo_teacher_prompt_text"][2]
    assert train_data["sdpo_teacher_signal_type"][2] == "solution_demo"
    assert train_data["self_distillation_mask"][2] == pytest.approx(1.0)
    assert train_data["sdpo_loss_weights"][2] > 0.0
    assert train_data["raw_reward"] == pytest.approx([1.0, 1.0, 0.0])
    assert "open fridge" in task_a_failure_prompt
    assert "take mug" in task_a_failure_prompt
    assert "Reference trajectory from a successful previous attempt on the same task:" in task_a_failure_prompt
    assert "Step 1\nAction: `open fridge`\nObservation: fridge is open" in task_a_failure_prompt
    assert "Step 2\nAction: `take mug`\nObservation: mug acquired" in task_a_failure_prompt
    assert "Use this reference only as context for the same task." in task_a_failure_prompt
    assert "If the current state differs from the reference trajectory, do not blindly copy an action." in (
        task_a_failure_prompt
    )
    reference_block = task_a_failure_prompt.split(
        "Reference trajectory from a successful previous attempt on the same task:", 1
    )[1].split("Use this reference only as context for the same task.", 1)[0]
    assert "Task:" not in reference_block
    assert "Initial state:" not in reference_block
    assert "Initial observation:" not in reference_block
    assert "Observation before action:" not in reference_block
    assert "New observation:" not in reference_block
    assert "Likely action plan:" not in reference_block
    assert "Trajectory result:" not in reference_block
    assert "episode_reward" not in reference_block
    assert "episode_length" not in reference_block
    assert "Guidance from a successful previous trajectory" not in reference_block
    assert "Guidance summary:" not in reference_block
    assert "Failure analysis:" not in reference_block
    assert "look around" not in reference_block
    assert train_data["sdpo_teacher_signal_type"] == ["solution_demo", "solution_demo", "solution_demo"]
    assert train_data["self_distillation/trajectory_demo_used_fraction"] == pytest.approx([1.0] * 3)
    assert train_data["self_distillation/guidance_summary_prompt_count"] == pytest.approx([0.0] * 3)


@pytest.mark.unit
def test_random_success_aggregation_selects_one_reproducible_demo_per_step():
    success_actions = {
        "success-alpha": "open alpha cabinet",
        "success-beta": "open beta cabinet",
        "success-gamma": "open gamma cabinet",
    }
    samples = [
        _sample(
            idx,
            _metadata(
                uid="task-random",
                traj_uid=traj_uid,
                reward=1.0,
                length=1,
                action=action,
                next_obs=f"{traj_uid} complete",
            ),
        )
        for idx, (traj_uid, action) in enumerate(success_actions.items())
    ]
    samples.extend(
        _sample(
            10 + turn_idx,
            _metadata(
                uid="task-random",
                traj_uid="failure-rollout",
                turn_idx=turn_idx,
                reward=0.0,
                length=16,
                action=f"inspect room {turn_idx}",
                next_obs=f"room {turn_idx} unchanged",
            ),
        )
        for turn_idx in range(16)
    )

    args = dict(
        rollout_seed=31415,
        sdpo_distillation_mode="representation",
        sdpo_representation_success_aggregation="random",
        sdpo_solution_context_format="trajectory_demo",
        sdpo_guidance_generation_mode="disabled",
    )
    first_plan = _plan(samples, **args)
    second_plan = _plan(samples, **args)

    first_failure_rows = [row for row in first_plan if row["sdpo_metadata"]["traj_uid"] == "failure-rollout"]
    second_selected = [
        row["sdpo_selected_success_traj_uid"]
        for row in second_plan
        if row["sdpo_metadata"]["traj_uid"] == "failure-rollout"
    ]
    selected = [row["sdpo_selected_success_traj_uid"] for row in first_failure_rows]

    assert selected == second_selected
    assert len(set(selected)) > 1
    assert set(selected) <= set(success_actions)
    for row, selected_traj_uid in zip(first_failure_rows, selected, strict=True):
        assert row["sdpo_teacher_signal_type"] == "solution_demo"
        assert row["sdpo_representation_success_aggregation"] == "random"
        assert row["sdpo_representation_success_count"] == 1
        assert row["sdpo_teacher_prompt_texts"] is None
        assert row["sdpo_teacher_messages_list"] is None
        prompt = row["sdpo_teacher_prompt_text"]
        assert success_actions[selected_traj_uid] in prompt
        for traj_uid, action in success_actions.items():
            if traj_uid != selected_traj_uid:
                assert action not in prompt


@pytest.mark.unit
def test_sample_success_aggregation_still_uses_first_sorted_success_demo():
    samples = [
        _sample(
            0,
            _metadata(
                uid="task-sample",
                traj_uid="success-beta",
                reward=1.0,
                length=1,
                action="open beta cabinet",
            ),
        ),
        _sample(
            1,
            _metadata(
                uid="task-sample",
                traj_uid="success-alpha",
                reward=1.0,
                length=1,
                action="open alpha cabinet",
            ),
        ),
        _sample(
            2,
            _metadata(
                uid="task-sample",
                traj_uid="failure",
                reward=0.0,
                length=1,
                action="inspect room",
            ),
        ),
    ]

    plan = _plan(
        samples,
        sdpo_distillation_mode="representation",
        sdpo_representation_success_aggregation="sample",
        sdpo_solution_context_format="trajectory_demo",
        sdpo_guidance_generation_mode="disabled",
    )

    failure_row = next(row for row in plan if row["sdpo_metadata"]["traj_uid"] == "failure")
    assert failure_row["sdpo_selected_success_traj_uid"] == "success-alpha"
    assert "open alpha cabinet" in failure_row["sdpo_teacher_prompt_text"]
    assert "open beta cabinet" not in failure_row["sdpo_teacher_prompt_text"]


@pytest.mark.unit
def test_trajectory_demo_all_failed_keeps_one_step_feedback_context():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(
                uid="task-a",
                traj_uid="failure-a",
                reward=0.0,
                length=2,
                action="look around",
                next_obs="still in the kitchen",
            ),
        ),
        _train_sample(
            1,
            _metadata(
                uid="task-a",
                traj_uid="failure-a",
                turn_idx=1,
                reward=0.0,
                length=2,
                action="open fridge",
                next_obs="fridge opens but task is incomplete",
            ),
        ),
    ]

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
        ),
        samples,
    )

    assert train_data["sdpo_teacher_signal_type"] == ["feedback", "feedback"]
    assert train_data["self_distillation_mask"] == pytest.approx([1.0, 1.0])
    assert train_data["self_distillation/trajectory_demo_used_fraction"] == pytest.approx([0.0, 0.0])
    assert train_data["self_distillation/guidance_summary_prompt_count"] == pytest.approx([0.0, 0.0])
    prompt0, prompt1 = train_data["sdpo_teacher_prompt_text"]
    feedback0 = prompt0.split("Relevant environment transition:", 1)[1].split(
        "Correctly solve the original question.", 1
    )[0]
    feedback1 = prompt1.split("Relevant environment transition:", 1)[1].split(
        "Correctly solve the original question.", 1
    )[0]
    assert "If you take action `look around`, the next state is:\nstill in the kitchen" in feedback0
    assert "open fridge" not in feedback0
    assert "fridge opens but task is incomplete" not in feedback0
    assert "If you take action `open fridge`, the next state is:\nfridge opens but task is incomplete" in feedback1
    assert "look around" not in feedback1
    assert "still in the kitchen" not in feedback1
    for prompt in train_data["sdpo_teacher_prompt_text"]:
        assert "Relevant environment transition:" in prompt
        assert "If you take action `" in prompt
        assert "Reference trajectory from a successful previous attempt" not in prompt
        assert "Reference trajectory from an unsuccessful previous attempt" not in prompt
        assert "Lessons from a previous unsuccessful attempt" not in prompt
        assert "Failure analysis:" not in prompt


@pytest.mark.unit
def test_trajectory_demo_no_success_failed_negative_uses_complete_own_episode():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(
                uid="task-a",
                traj_uid="failure-a",
                reward=0.0,
                length=2,
                action="look around",
                next_obs="still in the kitchen",
            ),
        ),
        _train_sample(
            1,
            _metadata(
                uid="task-a",
                traj_uid="failure-a",
                turn_idx=1,
                reward=0.0,
                length=2,
                action="open fridge",
                next_obs="fridge opens but task is incomplete",
            ),
        ),
    ]

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_no_success_context_mode="failed_negative",
        ),
        samples,
    )

    assert train_data["sdpo_teacher_signal_type"] == ["failed_negative_demo", "failed_negative_demo"]
    assert train_data["self_distillation_mask"] == pytest.approx([1.0, 1.0])
    assert train_data["self_distillation/trajectory_demo_used_fraction"] == pytest.approx([0.0, 0.0])
    assert train_data["self_distillation/failed_negative_used_fraction"] == pytest.approx([1.0, 1.0])
    assert train_data["self_distillation/own_failed_negative_used_fraction"] == pytest.approx([1.0, 1.0])
    assert train_data["self_distillation/feedback_used_fraction"] == pytest.approx([0.0, 0.0])
    assert train_data["self_distillation/no_success_filtered_fraction"] == pytest.approx([0.0, 0.0])
    assert train_data["self_distillation/guidance_summary_prompt_count"] == pytest.approx([0.0, 0.0])
    expected_prompt = (
        "CURRENT TASK CONTEXT\n\n"
        "Reference trajectory from a failed attempt on the same task:\n\n"
        "Step 1\n"
        "Action: `look around`\n"
        "Observation: still in the kitchen\n\n"
        "Step 2\n"
        "Action: `open fridge`\n"
        "Observation: fridge opens but task is incomplete\n\n"
        "Use this reference only as negative evidence. It is an attempt that did not solve the task.\n"
        "Compare it with the current observation and admissible actions before deciding.\n"
        "Avoid repeating actions from the failed trajectory when they caused no progress, invalid transitions, "
        "loops, or terminal failure.\n"
        "Do not blindly reject every action in the failed trajectory: an early action may still be useful if it "
        "is valid and matches the current state.\n"
        "Continue solving the original task and follow the original response format."
    )
    assert train_data["sdpo_teacher_prompt_text"][0] == expected_prompt
    assert train_data["sdpo_teacher_prompt_text"][1] == expected_prompt
    for prompt in train_data["sdpo_teacher_prompt_text"]:
        assert "Reference trajectory from a failed attempt on the same task:" in prompt
        assert "Step 1\nAction: `look around`\nObservation: still in the kitchen" in prompt
        assert "Step 2\nAction: `open fridge`\nObservation: fridge opens but task is incomplete" in prompt
        assert "Use this reference only as negative evidence. It is an attempt that did not solve the task." in prompt
        assert "Continue solving the original task and follow the original response format." in prompt
        assert "Relevant environment transition:" not in prompt
        assert "Correctly solve the original question." not in prompt


@pytest.mark.unit
def test_trajectory_demo_failed_negative_does_not_label_self_success_fallback_as_failed():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(
                uid="task-a",
                traj_uid="success-a",
                reward=1.0,
                length=1,
                action="open fridge",
                next_obs="fridge is open",
            ),
        ),
    ]

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_no_success_context_mode="failed_negative",
            sdpo_dont_reprompt_on_self_success=True,
        ),
        samples,
    )

    assert train_data["sdpo_teacher_signal_type"] == ["feedback"]
    assert train_data["self_distillation/failed_negative_used_fraction"] == pytest.approx([0.0])
    assert "Relevant environment transition:" in train_data["sdpo_teacher_prompt_text"][0]
    assert "Reference trajectory from a failed attempt" not in train_data["sdpo_teacher_prompt_text"][0]


@pytest.mark.unit
def test_trajectory_demo_failed_negative_only_replaces_no_success_groups():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="success-a", reward=1.0, length=1, action="open fridge"),
        ),
        _train_sample(
            1,
            _metadata(uid="task-a", traj_uid="failure-a", reward=0.0, length=1, action="look around"),
        ),
        _train_sample(
            2,
            _metadata(uid="task-b", traj_uid="failure-b", reward=0.0, length=2, action="look under table"),
        ),
        _train_sample(
            3,
            _metadata(
                uid="task-b",
                traj_uid="failure-b",
                turn_idx=1,
                reward=0.0,
                length=2,
                action="open drawer",
                next_obs="drawer is empty",
            ),
        ),
    ]

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_no_success_context_mode="failed_negative",
        ),
        samples,
    )

    assert train_data["sdpo_teacher_signal_type"] == [
        "solution_demo",
        "solution_demo",
        "failed_negative_demo",
        "failed_negative_demo",
    ]
    assert train_data["self_distillation/trajectory_demo_used_fraction"] == pytest.approx([0.5] * 4)
    assert train_data["self_distillation/group_success_demo_used_fraction"] == pytest.approx([0.5] * 4)
    assert train_data["self_distillation/failed_negative_used_fraction"] == pytest.approx([0.5] * 4)
    assert train_data["self_distillation/own_failed_negative_used_fraction"] == pytest.approx([0.5] * 4)
    assert train_data["self_distillation/feedback_used_fraction"] == pytest.approx([0.0] * 4)
    assert "Reference trajectory from a successful previous attempt" in train_data["sdpo_teacher_prompt_text"][1]
    assert "Reference trajectory from a failed attempt" not in train_data["sdpo_teacher_prompt_text"][1]
    assert "Reference trajectory from a failed attempt" in train_data["sdpo_teacher_prompt_text"][2]
    assert "open fridge" not in train_data["sdpo_teacher_prompt_text"][2]


@pytest.mark.unit
def test_trajectory_demo_own_outcome_rejects_filter_no_success_mode():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="failure-a", reward=0.0, length=1, action="look around"),
        ),
    ]

    with pytest.raises(ValueError, match="own_outcome requires sdpo_no_success_context_mode=failed_negative"):
        convert(
            _args(
                rollout_batch_size=1,
                n_samples_per_prompt=1,
                sdpo_guidance_generation_mode="disabled",
                sdpo_solution_context_format="trajectory_demo",
                sdpo_teacher_context_mode="own_outcome",
                sdpo_no_success_context_mode="filter",
            ),
            samples,
        )


@pytest.mark.unit
def test_trajectory_demo_own_outcome_uses_each_episode_as_reference():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(
                uid="task-a",
                traj_uid="success-a",
                reward=1.0,
                length=2,
                action="open fridge",
                next_obs="fridge is open",
            ),
        ),
        _train_sample(
            1,
            _metadata(
                uid="task-a",
                traj_uid="success-a",
                turn_idx=1,
                reward=1.0,
                length=2,
                action="take mug",
                next_obs="mug acquired",
            ),
        ),
        _train_sample(
            2,
            _metadata(
                uid="task-a",
                traj_uid="failure-a",
                reward=0.0,
                length=2,
                action="look around",
                next_obs="nothing changed",
            ),
        ),
        _train_sample(
            3,
            _metadata(
                uid="task-a",
                traj_uid="failure-a",
                turn_idx=1,
                reward=0.0,
                length=2,
                action="open cabinet",
                next_obs="cabinet is empty",
            ),
        ),
    ]

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=4,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_teacher_context_mode="own_outcome",
            sdpo_no_success_context_mode="failed_negative",
        ),
        samples,
    )

    assert train_data["sdpo_teacher_signal_type"] == [
        "solution_demo",
        "solution_demo",
        "failed_negative_demo",
        "failed_negative_demo",
    ]
    assert train_data["self_distillation/trajectory_demo_used_fraction"] == pytest.approx([0.5] * 4)
    assert train_data["self_distillation/own_success_demo_used_fraction"] == pytest.approx([0.5] * 4)
    assert train_data["self_distillation/group_success_demo_used_fraction"] == pytest.approx([0.0] * 4)
    assert train_data["self_distillation/failed_negative_used_fraction"] == pytest.approx([0.5] * 4)
    assert train_data["self_distillation/own_failed_negative_used_fraction"] == pytest.approx([0.5] * 4)
    assert "Reference trajectory from a successful previous attempt" in train_data["sdpo_teacher_prompt_text"][0]
    assert "Step 1\nAction: `open fridge`\nObservation: fridge is open" in train_data["sdpo_teacher_prompt_text"][0]
    assert "Step 2\nAction: `take mug`\nObservation: mug acquired" in train_data["sdpo_teacher_prompt_text"][0]
    assert "look around" not in train_data["sdpo_teacher_prompt_text"][0]
    assert "Reference trajectory from a failed attempt" in train_data["sdpo_teacher_prompt_text"][2]
    assert "Step 1\nAction: `look around`\nObservation: nothing changed" in train_data["sdpo_teacher_prompt_text"][2]
    assert "Step 2\nAction: `open cabinet`\nObservation: cabinet is empty" in train_data["sdpo_teacher_prompt_text"][2]
    assert "open fridge" not in train_data["sdpo_teacher_prompt_text"][2]


def _convert_controlled_own_outcome(*, reward: float, label_mode: str) -> dict:
    convert = _find_sdpo_convert()
    sample = _train_sample(
        0,
        _metadata(
            uid="task-single",
            traj_uid="trajectory-single",
            reward=reward,
            length=1,
            action="open fridge",
            next_obs="fridge is open",
        ),
    )
    return convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_teacher_context_mode="own_outcome",
            sdpo_no_success_context_mode="failed_negative",
            sdpo_context_prompt_style="controlled",
            sdpo_own_outcome_label_mode=label_mode,
        ),
        [sample],
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reward", "heading", "signal", "scope"),
    [
        (1.0, "A successful trajectory for the current task:", "solution_demo", "own_success_demo"),
        (0.0, "A failed trajectory for the current task:", "failed_negative_demo", "own_failed_negative"),
    ],
)
def test_controlled_own_outcome_n1_uses_exact_explicit_prompt(reward, heading, signal, scope):
    train_data = _convert_controlled_own_outcome(reward=reward, label_mode="explicit")

    expected = (
        f"CURRENT TASK CONTEXT\n\n{heading}\n\n"
        "Step 1\nAction: `open fridge`\nObservation: fridge is open\n\n"
        "Use this information as a reference and continue solving the original task."
    )
    assert train_data["sdpo_teacher_prompt_text"] == [expected]
    assert train_data["sdpo_teacher_signal_type"] == [signal]
    assert train_data["self_distillation_mask"] == pytest.approx([1.0])
    assert train_data[f"self_distillation/{scope}_used_fraction"] == pytest.approx([1.0])


@pytest.mark.unit
@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_controlled_own_outcome_explicit_and_omitted_differ_only_in_heading(reward):
    explicit = _convert_controlled_own_outcome(reward=reward, label_mode="explicit")["sdpo_teacher_prompt_text"][0]
    omitted = _convert_controlled_own_outcome(reward=reward, label_mode="omitted")["sdpo_teacher_prompt_text"][0]

    explicit_heading = (
        "A successful trajectory for the current task:"
        if reward == 1.0
        else "A failed trajectory for the current task:"
    )
    assert omitted == explicit.replace(explicit_heading, "A trajectory for the current task:")


@pytest.mark.unit
def test_controlled_own_outcome_keeps_full_trajectory_for_every_turn():
    prepare = _find_function(("prepare_sdpo_context_plan",), "context planning")
    samples = [
        _sample(
            0,
            _metadata(
                uid="task-single",
                traj_uid="trajectory-single",
                turn_idx=0,
                reward=1.0,
                length=2,
                action="open fridge",
                next_obs="fridge is open",
            ),
        ),
        _sample(
            1,
            _metadata(
                uid="task-single",
                traj_uid="trajectory-single",
                turn_idx=1,
                reward=1.0,
                length=2,
                action="take mug",
                next_obs="mug acquired",
            ),
        ),
    ]

    plan = prepare(
        _args(
            sdpo_teacher_context_mode="own_outcome",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_no_success_context_mode="failed_negative",
            sdpo_context_prompt_style="controlled",
            sdpo_max_demo_steps=1,
        ),
        samples,
    )

    for row in plan:
        assert "Step 1\nAction: `open fridge`\nObservation: fridge is open" in row["sdpo_teacher_prompt_text"]
        assert "Step 2\nAction: `take mug`\nObservation: mug acquired" in row["sdpo_teacher_prompt_text"]


@pytest.mark.unit
def test_controlled_feedback_only_uses_exact_prompt():
    convert = _find_sdpo_convert()
    sample = _train_sample(
        0,
        _metadata(
            uid="task-single",
            traj_uid="trajectory-single",
            action="open fridge",
            next_obs="fridge is open",
        ),
    )

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            sdpo_teacher_context_mode="feedback_only",
            sdpo_context_prompt_style="controlled",
        ),
        [sample],
    )

    assert train_data["sdpo_teacher_prompt_text"] == [
        "CURRENT TASK CONTEXT\n\n"
        "If the current action is:\n`open fridge`\n\n"
        "Observed next state:\nfridge is open\n\n"
        "Use this information as a reference and continue solving the original task."
    ]
    assert train_data["sdpo_teacher_signal_type"] == ["feedback"]
    assert train_data["self_distillation_mask"] == pytest.approx([1.0])


@pytest.mark.unit
def test_explicit_legacy_style_preserves_default_own_outcome_prompt():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-single", traj_uid="trajectory-single", reward=1.0, action="open fridge"),
        )
    ]
    common = {
        "rollout_batch_size": 1,
        "n_samples_per_prompt": 1,
        "sdpo_guidance_generation_mode": "disabled",
        "sdpo_solution_context_format": "trajectory_demo",
        "sdpo_teacher_context_mode": "own_outcome",
        "sdpo_no_success_context_mode": "failed_negative",
    }

    default_prompt = convert(_args(**common), samples)["sdpo_teacher_prompt_text"]
    legacy_prompt = convert(_args(**common, sdpo_context_prompt_style="legacy"), samples)["sdpo_teacher_prompt_text"]

    assert legacy_prompt == default_prompt
    assert "Reference trajectory from a successful previous attempt" in legacy_prompt[0]
    assert legacy_prompt[0].endswith("Correctly solve the original question.")


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            {"sdpo_context_prompt_style": "unknown"},
            "sdpo_context_prompt_style must be legacy or controlled",
        ),
        (
            {"sdpo_own_outcome_label_mode": "unknown"},
            "sdpo_own_outcome_label_mode must be explicit or omitted",
        ),
        (
            {"sdpo_context_prompt_style": "controlled", "sdpo_teacher_context_mode": "original"},
            "controlled requires sdpo_teacher_context_mode=feedback_only or own_outcome",
        ),
        (
            {"sdpo_own_outcome_label_mode": "omitted"},
            "omitted requires sdpo_context_prompt_style=controlled",
        ),
    ],
)
def test_controlled_prompt_config_rejects_invalid_combinations(overrides, match):
    prepare = _find_function(("prepare_sdpo_context_plan",), "context planning")
    samples = [_sample(0, _metadata())]

    with pytest.raises(ValueError, match=match):
        prepare(_args(**overrides), samples)


@pytest.mark.unit
def test_controlled_feedback_requires_action_and_next_observation():
    prepare = _find_function(("prepare_sdpo_context_plan",), "context planning")
    samples = [_sample(0, _metadata(action="", next_obs="fridge is open"))]

    with pytest.raises(ValueError, match="requires projected_action and next_anchor_obs"):
        prepare(
            _args(sdpo_teacher_context_mode="feedback_only", sdpo_context_prompt_style="controlled"),
            samples,
        )


@pytest.mark.unit
def test_trajectory_demo_no_success_filter_compacts_to_zero_loss_placeholders():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="success-a", reward=1.0, length=1, action="open fridge"),
        ),
        _train_sample(
            1,
            _metadata(uid="task-a", traj_uid="failure-a", reward=0.0, length=1, action="look around"),
        ),
        _train_sample(
            2,
            _metadata(
                uid="task-b",
                traj_uid="failure-b",
                reward=0.0,
                length=2,
                action="look around",
                next_obs="still searching",
            ),
        ),
        _train_sample(
            3,
            _metadata(
                uid="task-b",
                traj_uid="failure-b",
                turn_idx=1,
                reward=0.0,
                length=2,
                action="open fridge",
                next_obs="fridge is empty",
            ),
        ),
        _train_sample(
            4,
            _metadata(uid="task-b", traj_uid="failure-c", reward=0.0, length=1, action="open cabinet"),
        ),
    ]
    samples[0].rollout_id = 10
    samples[1].rollout_id = 11
    samples[2].rollout_id = 20
    samples[3].rollout_id = 20
    samples[4].rollout_id = 21
    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_no_success_context_mode="filter",
        ),
        samples,
    )

    assert len(train_data["tokens"]) == 4
    assert train_data["rollout_ids"] == [10, 11, 20, 21]
    assert train_data["sdpo_teacher_signal_type"] == ["solution_demo", "solution_demo", "none", "none"]
    assert train_data["self_distillation_mask"] == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert train_data["loss_masks"] == [[1], [1], [0], [0]]
    assert train_data["self_distillation/full_sample_count"] == pytest.approx([5.0] * 4)
    assert train_data["self_distillation/train_sample_count"] == pytest.approx([4.0] * 4)
    assert train_data["self_distillation/no_success_filtered_fraction"] == pytest.approx([3 / 5] * 4)
    assert train_data["self_distillation/filtered_sample_fraction"] == pytest.approx([3 / 5] * 4)
    assert train_data["self_distillation/compacted_sample_fraction"] == pytest.approx([1 / 5] * 4)
    assert train_data["self_distillation/zero_loss_placeholder_fraction"] == pytest.approx([2 / 5] * 4)
    assert train_data["self_distillation/feedback_used_fraction"] == pytest.approx([0.0] * 4)
    assert train_data["self_distillation/trajectory_demo_used_fraction"] == pytest.approx([2 / 5] * 4)
    assert train_data["self_distillation/compaction_empty_batch_fallback"] == pytest.approx([0.0] * 4)
    assert "Reference trajectory from a successful previous attempt" in train_data["sdpo_teacher_prompt_text"][1]
    assert "Relevant environment transition:" not in train_data["sdpo_teacher_prompt_text"][2]
    assert "Relevant environment transition:" not in train_data["sdpo_teacher_prompt_text"][3]


@pytest.mark.unit
def test_trajectory_demo_no_success_filter_keeps_one_placeholder_for_all_failed_batch():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="failure-a", reward=0.0, length=2, action="look around"),
        ),
        _train_sample(
            1,
            _metadata(
                uid="task-a",
                traj_uid="failure-a",
                turn_idx=1,
                reward=0.0,
                length=2,
                action="open fridge",
            ),
        ),
    ]
    samples[0].rollout_id = 100
    samples[1].rollout_id = 100

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_no_success_context_mode="filter",
        ),
        samples,
    )

    assert len(train_data["tokens"]) == 1
    assert train_data["rollout_ids"] == [100]
    assert train_data["sdpo_teacher_signal_type"] == ["none"]
    assert train_data["self_distillation_mask"] == pytest.approx([0.0])
    assert train_data["sdpo_loss_weights"] == pytest.approx([0.0])
    assert train_data["loss_masks"] == [[0]]
    assert train_data["self_distillation/full_sample_count"] == pytest.approx([2.0])
    assert train_data["self_distillation/train_sample_count"] == pytest.approx([1.0])
    assert train_data["self_distillation/no_success_filtered_fraction"] == pytest.approx([1.0])
    assert train_data["self_distillation/compacted_sample_fraction"] == pytest.approx([0.5])


@pytest.mark.unit
def test_sdpo_token_weights_use_failed_contrast_when_success_and_failure_exist():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(0, _metadata(uid="task-a", traj_uid="success-a", reward=1.0, action="open fridge")),
        _train_sample(1, _metadata(uid="task-a", traj_uid="failure-a", reward=0.0, action="look around")),
    ]

    train_data = convert(
        _args(
            sdpo_token_weights=True,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_no_success_context_mode="filter",
        ),
        samples,
    )

    assert train_data["self_distillation_mask"] == pytest.approx([1.0, 1.0])
    assert train_data["sdpo_token_weight_contrast_source"] == ["failed", "failed"]
    for prompt in train_data["sdpo_token_weight_contrast_prompt_text"]:
        assert "Reference trajectory from a failed attempt" in prompt
        assert "Action: `look around`" in prompt
        assert "Action: `open fridge`" not in prompt
    assert train_data["self_distillation/token_weight_contrast_failed_fraction"] == pytest.approx([1.0, 1.0])


@pytest.mark.unit
def test_sdpo_student_token_weights_skip_contrast_context():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(0, _metadata(uid="task-a", traj_uid="success-a", reward=1.0, action="open fridge")),
        _train_sample(1, _metadata(uid="task-a", traj_uid="failure-a", reward=0.0, action="look around")),
    ]

    train_data = convert(
        _args(
            sdpo_token_weights=True,
            sdpo_token_weight_source="student",
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_no_success_context_mode="filter",
        ),
        samples,
    )

    assert train_data["sdpo_token_weight_contrast_prompt_text"] == [None, None]
    assert train_data["sdpo_token_weight_contrast_messages"] == [None, None]
    assert train_data["sdpo_token_weight_contrast_source"] == ["", ""]
    assert train_data["sdpo_token_weight_contrast_traj_uid"] == [None, None]
    assert train_data["self_distillation/token_weight_contrast_failed_fraction"] == pytest.approx([0.0, 0.0])


@pytest.mark.unit
def test_sdpo_token_weights_failed_contrast_selection_is_stable_with_multiple_failures():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(0, _metadata(uid="task-a", traj_uid="success-a", reward=1.0, action="open fridge")),
        _train_sample(1, _metadata(uid="task-a", traj_uid="failure-a", reward=0.0, action="look around")),
        _train_sample(2, _metadata(uid="task-a", traj_uid="failure-b", reward=0.0, action="open cabinet")),
        _train_sample(3, _metadata(uid="task-a", traj_uid="failure-c", reward=0.0, action="take mug")),
    ]
    args = _args(
        sdpo_token_weights=True,
        sdpo_guidance_generation_mode="disabled",
        sdpo_solution_context_format="trajectory_demo",
        sdpo_no_success_context_mode="filter",
    )

    first = convert(args, samples)
    second = convert(args, samples)

    assert first["sdpo_token_weight_contrast_traj_uid"] == second["sdpo_token_weight_contrast_traj_uid"]
    assert set(first["sdpo_token_weight_contrast_traj_uid"]) <= {"failure-a", "failure-b", "failure-c"}
    assert first["sdpo_token_weight_contrast_source"] == ["failed"] * 4


@pytest.mark.unit
def test_sdpo_token_weights_use_no_extra_context_for_all_success_group():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(0, _metadata(uid="task-a", traj_uid="success-a", reward=1.0, action="open fridge")),
        _train_sample(1, _metadata(uid="task-a", traj_uid="success-b", reward=1.0, action="take mug")),
    ]

    train_data = convert(
        _args(
            sdpo_token_weights=True,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_no_success_context_mode="filter",
        ),
        samples,
    )

    assert train_data["self_distillation_mask"] == pytest.approx([1.0, 1.0])
    assert train_data["sdpo_token_weight_contrast_source"] == ["no_extra_context", "no_extra_context"]
    assert train_data["sdpo_token_weight_contrast_prompt_text"] == ["CURRENT TASK CONTEXT", "CURRENT TASK CONTEXT"]
    assert "Reference trajectory from a successful previous attempt" in train_data["sdpo_teacher_prompt_text"][0]
    assert (
        "Reference trajectory from a successful previous attempt"
        not in train_data["sdpo_token_weight_contrast_prompt_text"][0]
    )
    assert train_data["self_distillation/token_weight_contrast_no_context_fraction"] == pytest.approx([1.0, 1.0])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("second_traj_uid", "second_reward"),
    [("failure-a", 0.0), ("success-b", 1.0)],
    ids=["mixed", "all-success"],
)
def test_sdpo_token_weights_propagate_selected_success_traj_uid_per_row(second_traj_uid, second_reward):
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            index,
            _metadata(
                uid="task-a",
                traj_uid=traj_uid,
                reward=reward,
                action=action,
            ),
        )
        for index, (traj_uid, reward, action) in enumerate(
            zip(
                ("success-a", second_traj_uid),
                (1.0, second_reward),
                ("open fridge", "take mug"),
                strict=True,
            )
        )
    ]

    train_data = convert(
        _args(
            sdpo_token_weights=True,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_no_success_context_mode="filter",
        ),
        samples,
    )

    assert train_data["sdpo_selected_success_traj_uid"] == ["success-a", "success-a"]


@pytest.mark.unit
def test_sdpo_token_weights_filter_all_success_group_when_enabled():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(0, _metadata(uid="task-a", traj_uid="success-a", reward=1.0, action="open fridge")),
        _train_sample(1, _metadata(uid="task-a", traj_uid="success-b", reward=1.0, action="take mug")),
    ]

    train_data = convert(
        _args(
            sdpo_token_weights=True,
            sdpo_filter_all_success_groups=True,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_no_success_context_mode="filter",
        ),
        samples,
    )

    assert train_data["self_distillation_mask"] == pytest.approx([0.0, 0.0])
    assert train_data["sdpo_loss_weights"] == pytest.approx([0.0, 0.0])
    assert train_data["loss_masks"] == [[0], [0]]
    assert train_data["sdpo_token_weight_contrast_prompt_text"] == [None, None]
    assert train_data["sdpo_token_weight_contrast_source"] == ["", ""]
    assert train_data["self_distillation/all_success_filtered_fraction"] == pytest.approx([1.0, 1.0])
    assert train_data["self_distillation/no_success_filtered_fraction"] == pytest.approx([0.0, 0.0])
    assert train_data["self_distillation/token_weight_contrast_no_context_fraction"] == pytest.approx([0.0, 0.0])


@pytest.mark.unit
def test_sdpo_token_weights_filter_all_success_keeps_mixed_groups_failed_contrast():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(0, _metadata(uid="task-a", traj_uid="success-a", reward=1.0, action="open fridge")),
        _train_sample(1, _metadata(uid="task-a", traj_uid="failure-a", reward=0.0, action="look around")),
    ]

    train_data = convert(
        _args(
            sdpo_token_weights=True,
            sdpo_filter_all_success_groups=True,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_no_success_context_mode="filter",
        ),
        samples,
    )

    assert train_data["self_distillation_mask"] == pytest.approx([1.0, 1.0])
    assert train_data["sdpo_token_weight_contrast_source"] == ["failed", "failed"]
    assert train_data["self_distillation/all_success_filtered_fraction"] == pytest.approx([0.0, 0.0])
    assert train_data["self_distillation/token_weight_contrast_failed_fraction"] == pytest.approx([1.0, 1.0])
    assert train_data["self_distillation/token_weight_contrast_no_context_fraction"] == pytest.approx([0.0, 0.0])


@pytest.mark.unit
def test_filter_all_success_groups_compacts_alongside_no_success_filter():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(0, _metadata(uid="task-a", traj_uid="success-a", reward=1.0, action="open fridge")),
        _train_sample(1, _metadata(uid="task-a", traj_uid="failure-a", reward=0.0, action="look around")),
        _train_sample(2, _metadata(uid="task-b", traj_uid="success-b", reward=1.0, action="open cabinet")),
        _train_sample(3, _metadata(uid="task-b", traj_uid="success-c", reward=1.0, action="take mug")),
    ]
    samples[2].rollout_id = 30
    samples[3].rollout_id = 30

    train_data = convert(
        _args(
            sdpo_token_weights=True,
            sdpo_filter_all_success_groups=True,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_no_success_context_mode="filter",
        ),
        samples,
    )

    assert len(train_data["tokens"]) == 3
    assert train_data["rollout_ids"] == [0, 1, 30]
    assert train_data["sdpo_teacher_signal_type"] == ["solution_demo", "solution_demo", "none"]
    assert train_data["self_distillation_mask"] == pytest.approx([1.0, 1.0, 0.0])
    assert train_data["self_distillation/all_success_filtered_fraction"] == pytest.approx([0.5] * 3)
    assert train_data["self_distillation/no_success_filtered_fraction"] == pytest.approx([0.0] * 3)
    assert train_data["self_distillation/filtered_sample_fraction"] == pytest.approx([0.5] * 3)
    assert train_data["self_distillation/compacted_sample_fraction"] == pytest.approx([0.25] * 3)
    assert train_data["self_distillation/zero_loss_placeholder_fraction"] == pytest.approx([0.25] * 3)


@pytest.mark.unit
def test_sdpo_token_weights_keep_all_failed_filter_without_contrast():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(0, _metadata(uid="task-a", traj_uid="failure-a", reward=0.0, action="look around")),
    ]

    train_data = convert(
        _args(
            sdpo_token_weights=True,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_no_success_context_mode="filter",
        ),
        samples,
    )

    assert train_data["self_distillation_mask"] == pytest.approx([0.0])
    assert train_data["sdpo_loss_weights"] == pytest.approx([0.0])
    assert train_data["sdpo_token_weight_contrast_prompt_text"] == [None]
    assert train_data["sdpo_token_weight_contrast_source"] == [""]


@pytest.mark.unit
def test_grpo_token_weight_context_plan_uses_success_and_failed_trajectory_demos_without_thinking():
    prepare = _find_grpo_token_weight_context_plan()
    samples = [
        _train_sample(0, _metadata(uid="task-a", traj_uid="success-a", reward=1.0, action="open fridge")),
        _train_sample(1, _metadata(uid="task-a", traj_uid="failure-a", reward=0.0, action="look around")),
    ]

    plan = prepare(_args(grpo_token_weights=True), samples)

    assert [row["grpo_token_weight_uid"] for row in plan] == ["task-a", "task-a"]
    assert [row["grpo_token_weight_contrast_source"] for row in plan] == ["failed", "failed"]
    for row in plan:
        assert "Reference trajectory from a successful previous attempt" in row[
            "grpo_token_weight_positive_prompt_text"
        ]
        assert "Action: `open fridge`" in row["grpo_token_weight_positive_prompt_text"]
        assert "Reference trajectory from a failed attempt" in row[
            "grpo_token_weight_contrast_prompt_text"
        ]
        assert "Action: `look around`" in row["grpo_token_weight_contrast_prompt_text"]
        assert "<thinking>" not in row["grpo_token_weight_positive_prompt_text"]
        assert "<thinking>" not in row["grpo_token_weight_contrast_prompt_text"]


@pytest.mark.unit
def test_grpo_token_weight_context_plan_uses_no_extra_context_for_all_success_and_none_for_all_failed():
    prepare = _find_grpo_token_weight_context_plan()
    all_success = [
        _train_sample(0, _metadata(uid="task-a", traj_uid="success-a", reward=1.0, action="open fridge")),
        _train_sample(1, _metadata(uid="task-a", traj_uid="success-b", reward=1.0, action="take mug")),
    ]
    all_failed = [
        _train_sample(2, _metadata(uid="task-b", traj_uid="failure-a", reward=0.0, action="look around")),
        _train_sample(3, _metadata(uid="task-b", traj_uid="failure-b", reward=0.0, action="open cabinet")),
    ]

    success_plan = prepare(_args(grpo_token_weights=True), all_success)
    failed_plan = prepare(_args(grpo_token_weights=True), all_failed)

    assert [row["grpo_token_weight_contrast_source"] for row in success_plan] == [
        "no_extra_context",
        "no_extra_context",
    ]
    assert [row["grpo_token_weight_contrast_prompt_text"] for row in success_plan] == [
        "CURRENT TASK CONTEXT",
        "CURRENT TASK CONTEXT",
    ]
    for row in success_plan:
        assert "Reference trajectory from a successful previous attempt" in row[
            "grpo_token_weight_positive_prompt_text"
        ]
    for row in failed_plan:
        assert row == {
            "grpo_token_weight_positive_prompt_text": None,
            "grpo_token_weight_positive_messages": None,
            "grpo_token_weight_contrast_prompt_text": None,
            "grpo_token_weight_contrast_messages": None,
            "grpo_token_weight_contrast_source": "",
            "grpo_token_weight_uid": "task-b",
        }


@pytest.mark.unit
def test_grpo_token_weight_context_plan_selection_is_stable_across_input_order():
    prepare = _find_grpo_token_weight_context_plan()
    samples = [
        _train_sample(0, _metadata(uid="task-a", traj_uid="success-b", reward=1.0, action="take mug")),
        _train_sample(1, _metadata(uid="task-a", traj_uid="success-a", reward=1.0, action="open fridge")),
        _train_sample(2, _metadata(uid="task-a", traj_uid="failure-b", reward=0.0, action="open cabinet")),
        _train_sample(3, _metadata(uid="task-a", traj_uid="failure-a", reward=0.0, action="look around")),
    ]

    first = prepare(_args(grpo_token_weights=True, rollout_seed=42), samples)
    reversed_plan = prepare(_args(grpo_token_weights=True, rollout_seed=42), list(reversed(samples)))
    second = {sample.index: row for sample, row in zip(reversed(samples), reversed_plan, strict=True)}

    for sample, row in zip(samples, first, strict=True):
        assert row == second[sample.index]
        assert "Action: `open fridge`" in row["grpo_token_weight_positive_prompt_text"]


@pytest.mark.unit
def test_guidance_plan_self_trajectory_no_success_filter_compacts_failure_guidance():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="failure-a", reward=0.0, length=2, action="look around"),
        ),
        _train_sample(
            1,
            _metadata(
                uid="task-a",
                traj_uid="failure-a",
                turn_idx=1,
                reward=0.0,
                length=2,
                action="open fridge",
            ),
        ),
    ]
    samples[0].rollout_id = 100
    samples[1].rollout_id = 100

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="guidance_plan",
            sdpo_guidance_summary_source="self_trajectory",
            sdpo_no_success_context_mode="filter",
        ),
        samples,
    )

    assert len(train_data["tokens"]) == 1
    assert train_data["sdpo_teacher_signal_type"] == ["none"]
    assert train_data["self_distillation_mask"] == pytest.approx([0.0])
    assert train_data["loss_masks"] == [[0]]
    assert train_data["self_distillation/no_success_filtered_fraction"] == pytest.approx([1.0])
    assert train_data["self_distillation/guidance_summary_prompt_count"] == pytest.approx([0.0])
    assert "Failure analysis:" not in train_data["sdpo_teacher_prompt_text"][0]
    assert "Relevant environment transition:" not in train_data["sdpo_teacher_prompt_text"][0]


@pytest.mark.unit
def test_no_success_filter_keeps_schedule_aligned_zero_loss_placeholders():
    from slime.utils.dp_schedule import build_dp_schedule

    convert = _find_sdpo_convert()
    samples = []
    for rollout_id in range(8):
        for turn_idx in range(2):
            sample = _train_sample(
                rollout_id * 2 + turn_idx,
                _metadata(
                    uid="task-a",
                    traj_uid=f"failure-{rollout_id}",
                    turn_idx=turn_idx,
                    reward=0.0,
                    length=2,
                    action=f"action {turn_idx}",
                ),
            )
            sample.rollout_id = rollout_id
            samples.append(sample)

    args = _args(
        rollout_batch_size=1,
        n_samples_per_prompt=8,
        global_batch_size=8,
        micro_batch_size=2,
        use_dynamic_batch_size=False,
        balance_data=False,
        train_parallel_config={
            "dp_size": 8,
            "cp_size": 1,
            "vpp_size": 1,
            "microbatch_group_size_per_vp_stage": 1,
        },
        sdpo_guidance_generation_mode="disabled",
        sdpo_solution_context_format="trajectory_demo",
        sdpo_no_success_context_mode="filter",
    )
    train_data = convert(args, samples)

    assert len(train_data["tokens"]) == 16
    assert train_data["self_distillation_mask"] == pytest.approx([0.0] * 16)
    assert train_data["loss_masks"] == [[0]] * 16
    assert train_data["self_distillation/schedule_alignment_placeholder_fraction"] == pytest.approx([0.5] * 16)
    assert train_data["self_distillation/schedule_alignment_shortfall_count"] == pytest.approx([0.0] * 16)
    partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
        args,
        args.train_parallel_config,
        [len(tokens) for tokens in train_data["tokens"]],
        global_batch_size=args.global_batch_size,
        rollout_indices=train_data["rollout_ids"],
    )
    assert global_batch_sizes == [8]
    assert num_microbatches == [1]
    assert [len(partition) for partition in partitions] == [2] * 8
    assert all(len(rank_batches) == 1 for rank_batches in micro_batch_indices)


@pytest.mark.unit
def test_no_success_filter_keeps_vpp_schedule_aligned_zero_loss_placeholders():
    from slime.utils.dp_schedule import build_dp_schedule

    convert = _find_sdpo_convert()
    samples = []
    for rollout_id in range(8):
        for turn_idx in range(2):
            sample = _train_sample(
                rollout_id * 2 + turn_idx,
                _metadata(
                    uid="task-a",
                    traj_uid=f"failure-{rollout_id}",
                    turn_idx=turn_idx,
                    reward=0.0,
                    length=2,
                    action=f"action {turn_idx}",
                ),
            )
            sample.rollout_id = rollout_id
            samples.append(sample)

    args = _args(
        rollout_batch_size=1,
        n_samples_per_prompt=8,
        global_batch_size=8,
        micro_batch_size=1,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=1,
        balance_data=False,
        train_parallel_config={
            "dp_size": 8,
            "cp_size": 1,
            "vpp_size": 2,
            "microbatch_group_size_per_vp_stage": 2,
        },
        sdpo_guidance_generation_mode="disabled",
        sdpo_solution_context_format="trajectory_demo",
        sdpo_no_success_context_mode="filter",
    )
    train_data = convert(args, samples)

    assert len(train_data["tokens"]) == 16
    assert train_data["self_distillation_mask"] == pytest.approx([0.0] * 16)
    assert train_data["loss_masks"] == [[0]] * 16
    assert train_data["self_distillation/schedule_alignment_placeholder_fraction"] == pytest.approx([0.5] * 16)
    assert train_data["self_distillation/schedule_alignment_shortfall_count"] == pytest.approx([0.0] * 16)
    partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
        args,
        args.train_parallel_config,
        [len(tokens) for tokens in train_data["tokens"]],
        global_batch_size=args.global_batch_size,
        rollout_indices=train_data["rollout_ids"],
    )
    assert global_batch_sizes == [8]
    assert num_microbatches == [2]
    assert [len(partition) for partition in partitions] == [2] * 8
    assert all(len(rank_batches) == 2 for rank_batches in micro_batch_indices)


@pytest.mark.unit
def test_guidance_plan_no_success_filter_keeps_self_success_fallback_feedback():
    plan = _plan(
        [
            _sample(
                0,
                _metadata(uid="task-a", traj_uid="success", reward=1.0, length=1, action="open fridge"),
            ),
        ],
        sdpo_dont_reprompt_on_self_success=True,
        sdpo_solution_context_format="guidance_plan",
        sdpo_no_success_context_mode="filter",
    )

    assert plan[0]["sdpo_teacher_signal_type"] == "feedback"
    assert plan[0]["self_distillation_mask"] == pytest.approx(1.0)
    assert plan[0]["sdpo_filter_reason"] == ""
    assert "Relevant environment transition" in plan[0]["sdpo_teacher_prompt_text"]
    assert "Failure analysis:" not in plan[0]["sdpo_teacher_prompt_text"]


@pytest.mark.unit
def test_trajectory_demo_feedback_only_ignores_available_success_demo():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="success-a", reward=1.0, length=1, action="open fridge"),
        ),
        _train_sample(
            1,
            _metadata(uid="task-a", traj_uid="failure-a", reward=0.0, length=1, action="look around"),
        ),
    ]

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_teacher_context_mode="feedback_only",
        ),
        samples,
    )

    assert train_data["sdpo_teacher_signal_type"] == ["feedback", "feedback"]
    assert train_data["self_distillation/trajectory_demo_used_fraction"] == pytest.approx([0.0, 0.0])
    assert train_data["self_distillation/feedback_used_fraction"] == pytest.approx([1.0, 1.0])
    for prompt in train_data["sdpo_teacher_prompt_text"]:
        assert "Relevant environment transition:" in prompt
        assert "Reference trajectory from a successful previous attempt" not in prompt


@pytest.mark.unit
def test_feedback_only_no_success_filter_takes_filter_precedence():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="failure-a", reward=0.0, length=1, action="look around"),
        ),
    ]
    samples[0].rollout_id = 100

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            sdpo_guidance_generation_mode="disabled",
            sdpo_teacher_context_mode="feedback_only",
            sdpo_solution_context_format="trajectory_demo",
            sdpo_no_success_context_mode="filter",
        ),
        samples,
    )

    assert train_data["sdpo_teacher_signal_type"] == ["none"]
    assert train_data["self_distillation_mask"] == pytest.approx([0.0])
    assert train_data["sdpo_loss_weights"] == pytest.approx([0.0])
    assert train_data["loss_masks"] == [[0]]
    assert train_data["self_distillation/no_success_filtered_fraction"] == pytest.approx([1.0])
    assert train_data["self_distillation/feedback_used_fraction"] == pytest.approx([0.0])
    assert "Relevant environment transition:" not in train_data["sdpo_teacher_prompt_text"][0]
    assert "Reference trajectory from a successful previous attempt" not in train_data["sdpo_teacher_prompt_text"][0]


@pytest.mark.unit
def test_sdpo_converter_uses_guidance_summary_generator_for_guidance_plan():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="success", reward=1.0, length=1, action="open fridge"),
        ),
        _train_sample(
            1,
            _metadata(uid="task-a", traj_uid="failure", reward=0.0, length=1, action="look around"),
        ),
    ]

    def generator(prompts: list[str]) -> dict[str, str]:
        return {
            prompt: (
                "<thinking>extract the reusable action sequence.</thinking>\n\n"
                "Guidance summary:\n"
                "- Minimal plan: use `open fridge`.\n"
                "- Checks / avoid: track state changes."
            )
            for prompt in prompts
        }

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            sdpo_guidance_summary_generator=generator,
            sdpo_solution_context_format="guidance_plan",
        ),
        samples,
    )

    assert train_data["sdpo_teacher_signal_type"][1] == "solution_guidance"
    assert train_data["self_distillation_mask"][1] == pytest.approx(1.0)
    assert "Guidance summary:" in train_data["sdpo_teacher_prompt_text"][1]
    assert "<thinking>" not in train_data["sdpo_teacher_prompt_text"][1]
    assert "- Objective:" not in train_data["sdpo_teacher_prompt_text"][1]
    assert "Successful trajectory evidence:" not in train_data["sdpo_teacher_prompt_text"][1]
    assert train_data["self_distillation/guidance_summary_prompt_count"] == pytest.approx([1.0, 1.0])
    assert train_data["self_distillation/guidance_summary_used_fraction"] == pytest.approx([1.0, 1.0])
    assert train_data["self_distillation/guidance_summary_schema_valid_fraction"] == pytest.approx([1.0, 1.0])


@pytest.mark.unit
def test_guidance_summary_generation_prompts_are_explicit_and_compact():
    prepare = _find_function(("prepare_sdpo_context_plan", "build_sdpo_context_plan"), "context plan")
    success_plan = prepare(_args(), _build_batch())
    success_prompts = [
        row["sdpo_success_guidance_prompt"] for row in success_plan if row["sdpo_success_guidance_prompt"]
    ]
    assert success_prompts
    success_prompt = success_prompts[0]

    assert "Output ONLY this format:" in success_prompt
    assert "<thinking>" in success_prompt
    assert "Guidance summary:" in success_prompt
    assert "- Objective:" not in success_prompt
    assert "- Minimal plan:" in success_prompt
    assert "- Critical actions:" in success_prompt
    assert "- Checks / avoid:" in success_prompt
    assert "Task snapshot:" in success_prompt
    assert "Successful trajectory evidence:" in success_prompt
    assert "Action: `open fridge`" in success_prompt
    assert "Likely action plan:" not in success_prompt
    assert "Initial state:" not in success_prompt
    assert "Guidance from a successful previous trajectory" not in success_prompt

    failure_plan = prepare(
        _args(sdpo_guidance_summary_source="self_trajectory", sdpo_dont_reprompt_on_self_success=True),
        _build_batch(),
    )
    failure_prompts = [
        row["sdpo_failure_guidance_prompt"] for row in failure_plan if row["sdpo_failure_guidance_prompt"]
    ]
    assert failure_prompts
    failure_prompt = failure_prompts[0]

    assert "Output ONLY this format:" in failure_prompt
    assert "<thinking>" in failure_prompt
    assert "Failure analysis:" in failure_prompt
    assert "- Correction:" not in failure_prompt
    assert "- Likely mistake:" in failure_prompt
    assert "- Avoid:" in failure_prompt
    assert "Unsuccessful trajectory evidence:" in failure_prompt
    assert "Action: `look around`" in failure_prompt
    assert "Likely action plan:" not in failure_prompt


@pytest.mark.unit
def test_clean_guidance_summary_strips_thinking_and_keeps_guidance_section():
    clean = _find_function(("clean_guidance_summary",), "guidance summary cleaner")

    success = clean(
        "<thinking>hidden reasoning</thinking>\n"
        "Extra preface that should not reach the teacher.\n\n"
        "Guidance summary:\n"
        "- Minimal plan: use `open fridge`."
    )
    failure = clean(
        "Goal: duplicate task text\n"
        "<thinking>hidden failure reasoning</thinking>\n"
        "Failure analysis:\n"
        "- Likely mistake: repeated `look around`.\n"
        "- Avoid: repeat attempts."
    )

    assert success == "Guidance summary:\n- Minimal plan: use `open fridge`."
    assert failure == "Failure analysis:\n- Likely mistake: repeated `look around`.\n- Avoid: repeat attempts."
    assert clean("<thinking>truncated hidden reasoning") == ""


@pytest.mark.unit
def test_empty_cleaned_guidance_summary_uses_fallback_and_keeps_debug_record(tmp_path):
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="success", reward=1.0, length=1, action="open fridge"),
        ),
        _train_sample(
            1,
            _metadata(uid="task-a", traj_uid="failure", reward=0.0, length=1, action="look around"),
        ),
    ]

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            sdpo_guidance_summary_generator=lambda prompts: {
                prompt: "<thinking>the model forgot to write the final summary</thinking>" for prompt in prompts
            },
            sdpo_solution_context_format="guidance_plan",
            sdpo_guidance_debug_enabled=True,
            sdpo_guidance_debug_dir=str(tmp_path / "sdpo_guidance"),
        ),
        samples,
    )

    assert train_data["sdpo_teacher_signal_type"][1] == "solution_guidance"
    assert "Guidance summary:" in train_data["sdpo_teacher_prompt_text"][1]
    assert "Critical actions:" in train_data["sdpo_teacher_prompt_text"][1]
    assert "<thinking>" not in train_data["sdpo_teacher_prompt_text"][1]
    assert train_data["self_distillation/guidance_summary_fallback_fraction"] == pytest.approx([1.0, 1.0])
    assert train_data["self_distillation/guidance_summary_schema_valid_fraction"] == pytest.approx([0.0, 0.0])

    files = sorted((tmp_path / "sdpo_guidance").glob("rollout_*.jsonl.gz"))
    assert files
    with gzip.open(files[0], "rt", encoding="utf-8") as handle:
        record = json.loads(handle.readline())
    assert record["fallback_used"] is True
    assert record["model_clean_output_text"] == ""
    assert "<thinking>" in record["raw_output_text"]
    assert "Guidance summary:" in record["clean_output_text"]
    assert record["schema_valid"] is True


@pytest.mark.unit
def test_schema_invalid_guidance_summary_is_repaired_for_teacher_and_debug(tmp_path):
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="success", reward=1.0, length=1, action="open fridge"),
        ),
        _train_sample(
            1,
            _metadata(uid="task-a", traj_uid="failure", reward=0.0, length=1, action="look around"),
        ),
    ]

    raw_summary = "- Minimal plan: use `open fridge`.\n- Critical actions: `open fridge`."
    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            sdpo_guidance_summary_generator=lambda prompts: {prompt: raw_summary for prompt in prompts},
            sdpo_solution_context_format="guidance_plan",
            sdpo_guidance_debug_enabled=True,
            sdpo_guidance_debug_dir=str(tmp_path / "sdpo_guidance"),
        ),
        samples,
    )

    assert "Guidance summary:\n- Minimal plan: use `open fridge`." in train_data["sdpo_teacher_prompt_text"][1]
    assert train_data["self_distillation/guidance_summary_fallback_fraction"] == pytest.approx([1.0, 1.0])
    assert train_data["self_distillation/guidance_summary_schema_valid_fraction"] == pytest.approx([0.0, 0.0])

    files = sorted((tmp_path / "sdpo_guidance").glob("rollout_*.jsonl.gz"))
    assert files
    with gzip.open(files[0], "rt", encoding="utf-8") as handle:
        record = json.loads(handle.readline())
    assert record["fallback_used"] is True
    assert record["model_clean_output_text"] == raw_summary
    assert record["clean_output_text"].startswith("Guidance summary:\n")
    assert record["schema_valid"] is True


@pytest.mark.unit
def test_mixed_schema_guidance_summary_uses_fallback_without_opposite_header(tmp_path):
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="success", reward=1.0, length=1, action="open fridge"),
        ),
        _train_sample(
            1,
            _metadata(uid="task-a", traj_uid="failure", reward=0.0, length=1, action="look around"),
        ),
    ]

    mixed_summary = (
        "Failure analysis:\n"
        "- Likely mistake: this is the wrong schema for a success guidance prompt.\n"
        "Guidance summary:\n"
        "- Minimal plan: do not accept mixed headers."
    )
    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            sdpo_guidance_summary_generator=lambda prompts: {prompt: mixed_summary for prompt in prompts},
            sdpo_solution_context_format="guidance_plan",
            sdpo_guidance_debug_enabled=True,
            sdpo_guidance_debug_dir=str(tmp_path / "sdpo_guidance"),
        ),
        samples,
    )

    guidance_block = train_data["sdpo_teacher_prompt_text"][1].split(
        "Helpful guidance from a previous successful attempt", 1
    )[-1]
    assert "Guidance summary:" in guidance_block
    assert "Failure analysis:" not in guidance_block
    assert train_data["self_distillation/guidance_summary_fallback_fraction"] == pytest.approx([1.0, 1.0])
    assert train_data["self_distillation/guidance_summary_schema_valid_fraction"] == pytest.approx([0.0, 0.0])

    files = sorted((tmp_path / "sdpo_guidance").glob("rollout_*.jsonl.gz"))
    assert files
    with gzip.open(files[0], "rt", encoding="utf-8") as handle:
        record = json.loads(handle.readline())
    assert record["fallback_used"] is True
    assert record["model_clean_output_text"] == mixed_summary
    assert "Guidance summary:" in record["clean_output_text"]
    assert "Failure analysis:" not in record["clean_output_text"]
    assert record["schema_valid"] is True


@pytest.mark.unit
def test_mixed_schema_failure_guidance_uses_fallback_without_opposite_header():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="success", reward=1.0, length=1, action="open fridge"),
        ),
        _train_sample(
            1,
            _metadata(uid="task-a", traj_uid="failure", reward=0.0, length=1, action="look around"),
        ),
    ]

    mixed_summary = (
        "Guidance summary:\n"
        "- Minimal plan: this is the wrong schema for a failure guidance prompt.\n"
        "Failure analysis:\n"
        "- Likely mistake: do not accept mixed headers."
    )
    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            sdpo_guidance_summary_generator=lambda prompts: {prompt: mixed_summary for prompt in prompts},
            sdpo_dont_reprompt_on_self_success=True,
            sdpo_guidance_summary_source="self_trajectory",
            sdpo_solution_context_format="guidance_plan",
        ),
        samples,
    )

    assert train_data["sdpo_teacher_signal_type"][1] == "failure_guidance"
    guidance_block = train_data["sdpo_teacher_prompt_text"][1].split("Lessons from this failed attempt", 1)[-1]
    assert "Failure analysis:" in guidance_block
    assert "Guidance summary:" not in guidance_block
    assert train_data["self_distillation/guidance_summary_fallback_fraction"] == pytest.approx([1.0, 1.0])
    assert train_data["self_distillation/guidance_summary_schema_valid_fraction"] == pytest.approx([0.0, 0.0])


@pytest.mark.unit
def test_sdpo_guidance_generator_dict_must_cover_all_prompts():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="success", reward=1.0, length=1, action="open fridge"),
        ),
        _train_sample(
            1,
            _metadata(uid="task-a", traj_uid="failure", reward=0.0, length=1, action="look around"),
        ),
    ]

    with pytest.raises(ValueError, match="returned no output"):
        convert(
            _args(
                rollout_batch_size=1,
                n_samples_per_prompt=2,
                sdpo_guidance_summary_generator=lambda prompts: {},
                sdpo_solution_context_format="guidance_plan",
            ),
            samples,
        )


@pytest.mark.unit
def test_sdpo_converter_guidance_plan_requires_generation_when_prompts_exist():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="success", reward=1.0, length=1, action="open fridge"),
        ),
        _train_sample(
            1,
            _metadata(uid="task-a", traj_uid="failure", reward=0.0, length=1, action="look around"),
        ),
    ]

    with pytest.raises(ValueError, match="sdpo_guidance_generation_mode is disabled"):
        convert(
            _args(
                rollout_batch_size=1,
                n_samples_per_prompt=2,
                sdpo_guidance_generation_mode="disabled",
                sdpo_solution_context_format="guidance_plan",
            ),
            samples,
        )


@pytest.mark.unit
def test_sdpo_converter_binds_frozen_self_trajectory_summary_after_sparse_selection():
    convert = _find_sdpo_convert()
    sample = _train_sample(
        0,
        _metadata(uid="task-a", traj_uid="failure", reward=0.0, length=3, action="look around"),
    )
    sample.train_metadata["sdpo_guidance_summary_outputs"] = {
        "prompt-generated-from-the-complete-frozen-trajectory": (
            "<thinking>use the verified frozen outcome.</thinking>\n\n"
            "Failure analysis:\n"
            "- Likely mistake: the frozen attempt repeated `look around`.\n"
            "- Avoid: repeating `look around`."
        )
    }

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            sdpo_teacher_context_mode="own_outcome",
            sdpo_guidance_summary_source="self_trajectory",
            sdpo_no_success_context_mode="failed_negative",
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="guidance_plan",
        ),
        [sample],
    )

    assert train_data["sdpo_teacher_signal_type"] == ["failure_guidance"]
    assert "frozen attempt repeated `look around`" in train_data["sdpo_teacher_prompt_text"][0]
    assert train_data["self_distillation/guidance_summary_prompt_count"] == pytest.approx([0.0])


@pytest.mark.unit
def test_sparse_self_trajectory_output_binding_keeps_trajectory_outputs_separate():
    module = importlib.import_module("slime_plugins.agent_tasks.common.algorithms.sdpo")
    context = importlib.import_module("slime_plugins.agent_tasks.common.algorithms.sdpo_context")
    bind = module._bind_precomputed_self_trajectory_outputs
    first = _train_sample(0, _metadata(uid="task-a", traj_uid="first", reward=0.0))
    second = _train_sample(1, _metadata(uid="task-b", traj_uid="second", reward=0.0))
    first.train_metadata["sdpo_guidance_summary_outputs"] = {"frozen prompt first": "first summary"}
    second.train_metadata["sdpo_guidance_summary_outputs"] = {"frozen prompt second": "second summary"}
    sparse_plan = [
        {"sdpo_failure_guidance_prompt": "same sparse prompt"},
        {"sdpo_failure_guidance_prompt": "same sparse prompt"},
    ]

    bound = bind([first, second], sparse_plan)

    key = context._ROW_PRECOMPUTED_GUIDANCE_OUTPUT_KEY
    assert [row[key] for row in bound] == ["first summary", "second summary"]
    assert [row["sdpo_failure_guidance_prompt"] for row in bound] == ["same sparse prompt"] * 2
    assert sparse_plan == [
        {"sdpo_failure_guidance_prompt": "same sparse prompt"},
        {"sdpo_failure_guidance_prompt": "same sparse prompt"},
    ]


@pytest.mark.unit
def test_converter_keeps_distinct_summaries_when_frozen_prompt_text_collides(tmp_path):
    convert = _find_sdpo_convert()
    first = _train_sample(0, _metadata(uid="task-a", traj_uid="first", reward=0.0, action="look"))
    second = _train_sample(1, _metadata(uid="task-b", traj_uid="second", reward=0.0, action="look"))
    shared_prompt = "same complete frozen trajectory prompt"
    first.train_metadata["sdpo_guidance_summary_outputs"] = {
        shared_prompt: "Failure analysis:\n- Likely mistake: marker A.\n- Avoid: action A."
    }
    second.train_metadata["sdpo_guidance_summary_outputs"] = {
        shared_prompt: "Failure analysis:\n- Likely mistake: marker B.\n- Avoid: action B."
    }

    train_data = convert(
        _args(
            rollout_batch_size=2,
            n_samples_per_prompt=1,
            sdpo_teacher_context_mode="own_outcome",
            sdpo_guidance_summary_source="self_trajectory",
            sdpo_no_success_context_mode="failed_negative",
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="guidance_plan",
            sdpo_guidance_debug_enabled=True,
            sdpo_guidance_debug_dir=str(tmp_path / "sdpo_guidance"),
        ),
        [first, second],
    )

    assert "marker A" in train_data["sdpo_teacher_prompt_text"][0]
    assert "marker B" in train_data["sdpo_teacher_prompt_text"][1]
    assert train_data["self_distillation/guidance_summary_output_count"] == pytest.approx([2.0, 2.0])
    assert all(value > 0 for value in train_data["self_distillation/guidance_summary_output_chars_mean"])
    assert train_data["self_distillation/guidance_summary_schema_valid_fraction"] == pytest.approx([1.0, 1.0])

    files = sorted((tmp_path / "sdpo_guidance").glob("rollout_*.jsonl.gz"))
    assert len(files) == 1
    with gzip.open(files[0], "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    assert len(records) == 2
    assert {record["output_binding"] for record in records} == {"row_precomputed"}
    assert {record["frozen_source_prompt_text"] for record in records} == {shared_prompt}
    assert {tuple(record["target_traj_uids"]) for record in records} == {("first",), ("second",)}
    outputs_by_traj = {record["target_traj_uids"][0]: record["raw_output_text"] for record in records}
    assert "marker A" in outputs_by_traj["first"]
    assert "marker B" in outputs_by_traj["second"]
    assert "marker B" not in outputs_by_traj["first"]
    assert "marker A" not in outputs_by_traj["second"]
    assert all(record["schema_valid"] for record in records)


@pytest.mark.unit
def test_sdpo_converter_falls_back_for_empty_cleaned_guidance_summary():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="success", reward=1.0, length=1, action="open fridge"),
        ),
        _train_sample(
            1,
            _metadata(uid="task-a", traj_uid="failure", reward=0.0, length=1, action="look around"),
        ),
    ]

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            sdpo_guidance_summary_generator=lambda prompts: {prompt: "Goal: duplicate task" for prompt in prompts},
            sdpo_solution_context_format="guidance_plan",
        ),
        samples,
    )

    assert "Guidance summary:" in train_data["sdpo_teacher_prompt_text"][1]
    assert train_data["self_distillation/guidance_summary_fallback_fraction"] == pytest.approx([1.0, 1.0])
    assert train_data["self_distillation/guidance_summary_schema_valid_fraction"] == pytest.approx([0.0, 0.0])


@pytest.mark.unit
def test_sdpo_converter_falls_back_for_none_guidance_summary_output():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="success", reward=1.0, length=1, action="open fridge"),
        ),
        _train_sample(
            1,
            _metadata(uid="task-a", traj_uid="failure", reward=0.0, length=1, action="look around"),
        ),
    ]

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            sdpo_guidance_summary_generator=lambda prompts: {prompt: None for prompt in prompts},
            sdpo_solution_context_format="guidance_plan",
        ),
        samples,
    )

    assert "Guidance summary:" in train_data["sdpo_teacher_prompt_text"][1]
    assert train_data["self_distillation/guidance_summary_fallback_fraction"] == pytest.approx([1.0, 1.0])
    assert train_data["self_distillation/guidance_summary_schema_valid_fraction"] == pytest.approx([0.0, 0.0])


@pytest.mark.unit
def test_sdpo_converter_deactivates_removed_samples_and_zeroes_loss_mask():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="success", reward=1.0, length=1, action="open fridge"),
        ),
        _train_sample(
            1,
            _metadata(uid="task-a", traj_uid="failure", reward=0.0, length=1, action="look around"),
        ),
    ]
    samples[1].remove_sample = True

    train_data = convert(
        _args(rollout_batch_size=1, n_samples_per_prompt=2, sdpo_solution_context_format="trajectory_demo"),
        samples,
    )

    assert train_data["loss_masks"][1] == [0]
    assert train_data["self_distillation_mask"][1] == pytest.approx(0.0)
    assert train_data["sdpo_loss_weights"][1] == pytest.approx(0.0)
    assert train_data["sdpo_teacher_signal_type"][1] == "none"
    assert train_data["self_distillation/mask_fraction"] == pytest.approx([0.5, 0.5])


@pytest.mark.unit
def test_sdpo_converter_ignores_removed_success_as_teacher_candidate():
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="removed-success", reward=1.0, length=1, action="open fridge"),
        ),
        _train_sample(
            1,
            _metadata(uid="task-a", traj_uid="active-failure", reward=0.0, length=1, action="look around"),
        ),
    ]
    samples[0].remove_sample = True

    def generator(prompts: list[str]) -> dict[str, str]:
        return {
            prompt: (
                "<thinking>avoid repeating the failed action.</thinking>\n\n"
                "Failure analysis:\n"
                "- Likely mistake: repeated `look around`.\n"
                "- Avoid: repeating `look around`."
            )
            for prompt in prompts
        }

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            sdpo_guidance_summary_generator=generator,
            sdpo_solution_context_format="guidance_plan",
        ),
        samples,
    )

    assert train_data["self_distillation_mask"][0] == pytest.approx(0.0)
    assert train_data["sdpo_teacher_signal_type"][0] == "none"
    assert train_data["sdpo_teacher_signal_type"][1] == "failure_guidance"
    assert "Lessons from a previous unsuccessful attempt" in train_data["sdpo_teacher_prompt_text"][1]
    assert "Helpful guidance from a previous successful attempt" not in train_data["sdpo_teacher_prompt_text"][1]


@pytest.mark.unit
def test_sdpo_converter_does_not_generate_guidance_for_removed_only_samples():
    convert = _find_sdpo_convert()
    sample = _train_sample(
        0,
        _metadata(uid="task-a", traj_uid="removed-success", reward=1.0, length=1, action="open fridge"),
    )
    sample.remove_sample = True

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            sdpo_guidance_generation_mode="disabled",
            sdpo_solution_context_format="guidance_plan",
        ),
        [sample],
    )

    assert train_data["self_distillation_mask"] == pytest.approx([0.0])
    assert train_data["sdpo_teacher_signal_type"] == ["none"]
    assert train_data["loss_masks"] == [[0]]


@pytest.mark.unit
def test_sdpo_weighting_compensates_for_rollout_denominator_semantics():
    prepare = _find_function(("prepare_sdpo_context_plan", "build_sdpo_context_plan"), "context plan")
    samples = [
        _sample(0, _metadata(traj_uid="long", turn_idx=0, reward=1.0, length=2, action="open cabinet")),
        _sample(1, _metadata(traj_uid="long", turn_idx=1, reward=1.0, length=2, action="take mug")),
        _sample(2, _metadata(traj_uid="short", turn_idx=0, reward=1.0, length=1, action="open fridge")),
    ]

    traj_equal = prepare(
        _args(sdpo_solution_context_format="trajectory_demo", sdpo_multi_turn_weighting="traj_equal"), samples
    )
    step_equal = prepare(
        _args(sdpo_solution_context_format="trajectory_demo", sdpo_multi_turn_weighting="step_equal"), samples
    )
    hybrid = prepare(
        _args(sdpo_solution_context_format="trajectory_demo", sdpo_multi_turn_weighting="hybrid"), samples
    )

    assert [row["sdpo_loss_weights"] for row in traj_equal] == pytest.approx([1.0, 1.0, 1.0])
    assert step_equal[0]["sdpo_loss_weights"] == pytest.approx(step_equal[1]["sdpo_loss_weights"])
    assert step_equal[0]["sdpo_loss_weights"] > step_equal[2]["sdpo_loss_weights"]
    assert hybrid[0]["sdpo_loss_weights"] == pytest.approx(hybrid[1]["sdpo_loss_weights"])
    assert step_equal[0]["sdpo_loss_weights"] > hybrid[0]["sdpo_loss_weights"] > traj_equal[0]["sdpo_loss_weights"]


@pytest.mark.unit
def test_sdpo_converter_writes_guidance_debug_jsonl(tmp_path):
    convert = _find_sdpo_convert()
    samples = [
        _train_sample(
            0,
            _metadata(uid="task-a", traj_uid="success", reward=1.0, length=1, action="open fridge"),
        ),
        _train_sample(
            1,
            _metadata(uid="task-a", traj_uid="failure", reward=0.0, length=1, action="look around"),
        ),
    ]

    train_data = convert(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            sdpo_guidance_summary_generator=lambda prompts: {
                prompt: (
                    "<thinking>summarize the successful route.</thinking>\n\n"
                    "Guidance summary:\n"
                    "- Minimal plan: use `open fridge`."
                )
                for prompt in prompts
            },
            sdpo_solution_context_format="guidance_plan",
            sdpo_guidance_debug_enabled=True,
            sdpo_guidance_debug_dir=str(tmp_path / "sdpo_guidance"),
        ),
        samples,
    )

    files = sorted((tmp_path / "sdpo_guidance").glob("rollout_*.jsonl.gz"))
    assert files
    with gzip.open(files[0], "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    assert records
    record = records[0]
    assert record["schema_name"] == "sdpo_guidance_debug"
    assert record["task_profile"] == "alfworld"
    assert record["prompt_kind"] == "success"
    assert "Output ONLY this format:" in record["prompt_text"]
    assert "<thinking>" in record["raw_output_text"]
    assert "Guidance summary:" in record["clean_output_text"]
    assert "<thinking>" not in record["clean_output_text"]
    assert "- Objective:" not in record["clean_output_text"]
    assert record["schema_valid"] is True
    assert record["used_count"] > 0
    assert train_data["self_distillation/guidance_summary_debug_logged_count"] == pytest.approx([1.0, 1.0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
