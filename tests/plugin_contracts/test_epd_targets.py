from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch


def _turn(turn_idx: int, *, prompt: str, action: str) -> dict:
    return {
        "turn_idx": turn_idx,
        "messages": [{"role": "user", "content": prompt}],
        "prompt_ids": [100 + turn_idx, 200 + turn_idx],
        "response_text": f"<thinking>do {action}</thinking><action>{action}</action>",
        "response_ids": [300 + turn_idx],
        "deployment_behavior_log_probs": torch.tensor([-0.1], dtype=torch.float32),
        "current_observation": f"observation-{turn_idx}",
        "next_observation": f"next-observation-{turn_idx}",
        "frozen_action": action,
        "env_reward": 1.0,
        "finish_reason": "stop",
        "prompt_overlength": False,
        "prompt_truncated": False,
        "is_terminal": turn_idx == 1,
        "format_valid": True,
        "format_invalid_reason": None,
        "errors": {},
    }


def _trajectory(*, success: bool = True) -> dict:
    return {
        "trajectory_uid": "alfworld-iterative-00000001",
        "task_id": "task-1",
        "task_description": "put the object away",
        "runtime_task_identity": {"task_description": "put the object away", "gamefile": "game-1"},
        "provenance": {
            "stream_index": 0,
            "task_seed": 123,
            "source_group_index": 456,
            "seed": 579,
            "task_identity": {"task_id": "task-1"},
        },
        "split": "train",
        "success": success,
        "outcome": "success" if success else "failure",
        "episode_reward": float(success),
        "termination_reason": "success" if success else "incomplete",
        "truncated": False,
        "horizon_reached": False,
        "errors": {},
        "turns": [
            _turn(0, prompt="original-prefix-0", action="look"),
            _turn(1, prompt="original-prefix-1", action="open"),
        ],
    }


def _direct_binding() -> dict:
    return {
        "model_path": "/models/epd-teacher",
        "corpus_dir": "/corpora/alfworld-iterative",
        "teacher_iteration": 29,
        "materialization_identity": "alfworld-epd-test",
    }


@pytest.mark.unit
def test_epd_prompt_reuses_controlled_own_outcome_trajectory_context():
    from slime_plugins.agent_tasks.alfworld.frozen.epd import build_teacher_context, build_teacher_prompt_args

    trajectory = _trajectory()
    row = build_teacher_context(build_teacher_prompt_args(), trajectory, trajectory["turns"][1])

    assert row["teacher_messages"][-1]["role"] == "user"
    teacher_prompt = row["teacher_messages"][-1]["content"]
    assert "original-prefix-1" in teacher_prompt
    assert "A successful trajectory for the current task:" in teacher_prompt
    assert "Action: `look`" in teacher_prompt
    assert "Action: `open`" in teacher_prompt
    assert "Use this information as a reference and continue solving the original task." in teacher_prompt
    assert row["teacher_signal_type"] == "solution_demo"
    assert row["outcome"] == "success"


@pytest.mark.unit
def test_epd_prompt_preserves_explicit_failure_outcome():
    from slime_plugins.agent_tasks.alfworld.frozen.epd import build_teacher_context, build_teacher_prompt_args

    trajectory = _trajectory(success=False)
    row = build_teacher_context(build_teacher_prompt_args(), trajectory, trajectory["turns"][0])

    teacher_prompt = row["teacher_messages"][-1]["content"]
    assert "A failed trajectory for the current task:" in teacher_prompt
    assert row["teacher_signal_type"] == "failed_negative_demo"
    assert row["outcome"] == "failure"


@pytest.mark.unit
def test_epd_canonical_expansion_keeps_success_and_failure_turns():
    from slime_plugins.agent_tasks.alfworld.frozen.epd import canonical_steps_from_trajectories

    successful = _trajectory(success=True)
    failed = _trajectory(success=False)
    failed["trajectory_uid"] = "alfworld-iterative-00000000"
    steps = canonical_steps_from_trajectories(
        [successful, failed],
        expected_trajectories=2,
        expected_steps=4,
    )
    assert [step["identity"] for step in steps] == [
        "alfworld-iterative-00000000/turn/0000",
        "alfworld-iterative-00000000/turn/0001",
        "alfworld-iterative-00000001/turn/0000",
        "alfworld-iterative-00000001/turn/0001",
    ]
    assert [step["outcome"] for step in steps] == ["failure", "failure", "success", "success"]


@pytest.mark.unit
def test_epd_retry_seed_and_sampling_contract_are_identity_derived():
    from slime_plugins.agent_tasks.alfworld.frozen.epd import (
        EPD_PRIMARY_SEED,
        build_sampling_params,
        retry_seed,
    )

    assert build_sampling_params(EPD_PRIMARY_SEED) == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": 1024,
        "sampling_seed": 42,
    }
    assert retry_seed("uid-a", 2, 1) == retry_seed("uid-a", 2, 1)
    assert retry_seed("uid-a", 2, 1) != retry_seed("uid-a", 2, 2)
    assert retry_seed("uid-a", 2, 1) != retry_seed("uid-b", 2, 1)
    assert 0 < retry_seed("uid-a", 2, 1) < 2**31


@pytest.mark.unit
def test_epd_teacher_tokenization_uses_m14_chat_template_without_truncation():
    from slime_plugins.agent_tasks.alfworld.frozen.epd import _teacher_prompt_ids

    class Tokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return [11, 12, 13]

    tokenizer = Tokenizer()
    assert _teacher_prompt_ids(tokenizer, [{"role": "user", "content": "teacher"}], [99]) == [11, 12, 13]
    assert tokenizer.calls[0][1] == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


@pytest.mark.unit
def test_epd_teacher_tokenization_can_use_online_sdpo_prompt_cap():
    from slime_plugins.agent_tasks.alfworld.frozen.epd import _teacher_prompt_ids

    class Tokenizer:
        def __init__(self):
            self.truncation_side = "left"
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs, self.truncation_side))
            token_ids = list(range(12))
            return token_ids[: kwargs["max_length"]] if kwargs.get("truncation") else token_ids

    tokenizer = Tokenizer()
    assert _teacher_prompt_ids(
        tokenizer,
        [{"role": "user", "content": "teacher"}],
        [99],
        max_prompt_tokens=10,
        truncation_side="right",
    ) == list(range(10))
    assert tokenizer.calls[0][1]["max_length"] == 10
    assert tokenizer.calls[0][1]["truncation"] is True
    assert tokenizer.calls[0][2] == "right"
    assert tokenizer.truncation_side == "left"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "finish_type", "response_ids", "valid"),
    [
        ("<thinking>ok</thinking><action>look</action>", "stop", [1, 2], True),
        ("<thinking>ok</thinking><action>look</action>", "length", [1, 2], False),
        ("<thinking>ok</thinking>look", "stop", [1, 2], False),
        ("", "stop", [1], False),
    ],
)
def test_epd_teacher_output_validation(text, finish_type, response_ids, valid):
    from slime_plugins.agent_tasks.alfworld.frozen.epd import validate_teacher_output

    result = validate_teacher_output(text, finish_type, response_ids)
    assert result["valid"] is valid
    assert result["truncated"] is (finish_type == "length")
    if valid:
        assert result["projected_action"] == "look"
    else:
        assert result["invalid_reason"]


@pytest.mark.unit
def test_epd_materializer_uses_completion_order_independent_canonical_records(tmp_path):
    from slime_plugins.agent_tasks.alfworld.frozen.epd import materialize_records

    steps = []
    for uid, turn_idx in (("uid-b", 0), ("uid-a", 1), ("uid-a", 0)):
        trajectory = _trajectory()
        trajectory["trajectory_uid"] = uid
        turn = _turn(turn_idx, prompt=f"{uid}-prefix-{turn_idx}", action="look")
        steps.append({"trajectory": trajectory, "turn": turn, "trajectory_uid": uid, "turn_idx": turn_idx})

    async def generate(step, _endpoint):
        await asyncio.sleep(0.001 * (3 - step["turn_idx"]))
        return {
            "text": "<thinking>ok</thinking><action>look</action>",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.2, 501]],
            },
        }

    records = asyncio.run(
        materialize_records(
            steps,
            output_dir=tmp_path,
            endpoints=["fake://0", "fake://1"],
            tokenizer=None,
            generate_request=generate,
            expected_count=3,
            worker_count=2,
            shard_size=2,
            direct_binding=_direct_binding(),
        )
    )

    assert [(row["trajectory_uid"], row["turn_idx"]) for row in records] == [
        ("uid-a", 0),
        ("uid-a", 1),
        ("uid-b", 0),
    ]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["target_count"] == 3
    assert [shard["count"] for shard in manifest["shards"]] == [2, 1]


@pytest.mark.unit
def test_epd_materializer_retries_semantic_failures_with_identity_seeds(tmp_path):
    from slime_plugins.agent_tasks.alfworld.frozen.epd import materialize_records, retry_seed

    trajectory = _trajectory()
    step = {
        "trajectory": trajectory,
        "turn": trajectory["turns"][0],
        "trajectory_uid": trajectory["trajectory_uid"],
        "turn_idx": 0,
    }
    calls = 0

    async def generate(_request, _endpoint):
        nonlocal calls
        calls += 1
        if calls < 3:
            return {
                "text": "<thinking>unfinished</thinking>",
                "meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.2, 501]]},
            }
        return {
            "text": "<thinking>ok</thinking><action>look</action>",
            "meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.2, 501]]},
        }

    records = asyncio.run(
        materialize_records(
            [step],
            output_dir=tmp_path,
            endpoints=["fake://0"],
            tokenizer=None,
            generate_request=generate,
            expected_count=1,
            worker_count=1,
            direct_binding=_direct_binding(),
        )
    )
    assert records[0]["attempt_count"] == 3
    assert [attempt["seed"] for attempt in records[0]["attempts"]] == [
        42,
        retry_seed(trajectory["trajectory_uid"], 0, 1),
        retry_seed(trajectory["trajectory_uid"], 0, 2),
    ]
    assert [attempt["failure_class"] for attempt in records[0]["attempts"]] == ["semantic", "semantic", None]


@pytest.mark.unit
def test_epd_materializer_global_concurrency_is_independent_of_engine_count(tmp_path):
    from slime_plugins.agent_tasks.alfworld.frozen.epd import materialize_records

    steps = []
    for index in range(16):
        trajectory = _trajectory()
        trajectory["trajectory_uid"] = f"uid-{index:02d}"
        steps.append(
            {
                "trajectory": trajectory,
                "turn": trajectory["turns"][0],
                "trajectory_uid": trajectory["trajectory_uid"],
                "turn_idx": 0,
            }
        )
    active = 0
    max_active = 0

    async def generate(_request, _endpoint):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {
            "text": "<thinking>ok</thinking><action>look</action>",
            "meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.2, 501]]},
        }

    asyncio.run(
        materialize_records(
            steps,
            output_dir=tmp_path,
            endpoints=["fake://0", "fake://1"],
            tokenizer=None,
            generate_request=generate,
            expected_count=16,
            worker_count=2,
            global_concurrency=8,
            direct_binding=_direct_binding(),
        )
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert max_active > 2
    assert manifest["observed_max_inflight"] == max_active
    assert set(manifest["request_count_by_endpoint"]) == {"fake://0", "fake://1"}
    assert all(value > 0 for value in manifest["valid_target_count_by_endpoint"].values())


@pytest.mark.unit
def test_epd_materializer_fails_promptly_when_worker_fails(tmp_path):
    from slime_plugins.agent_tasks.alfworld.frozen.epd import materialize_records

    trajectory = _trajectory()
    step = {
        "trajectory": trajectory,
        "turn": trajectory["turns"][0],
        "trajectory_uid": trajectory["trajectory_uid"],
        "turn_idx": 0,
    }

    async def generate(_request, _endpoint):
        raise RuntimeError("engine unavailable")

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        asyncio.run(
            asyncio.wait_for(
                materialize_records(
                    [step],
                    output_dir=tmp_path,
                    endpoints=["fake://0"],
                    tokenizer=None,
                    generate_request=generate,
                    expected_count=1,
                    worker_count=1,
                    direct_binding=_direct_binding(),
                ),
                timeout=5,
            )
        )
    assert (tmp_path / "failure.json").is_file()


@pytest.mark.unit
def test_epd_materializer_resume_reloads_endpoint_accounting(tmp_path):
    from slime_plugins.agent_tasks.alfworld.frozen.epd import materialize_records

    steps = []
    for index in range(4):
        trajectory = _trajectory()
        trajectory["trajectory_uid"] = f"resume-{index}"
        steps.append(
            {
                "trajectory": trajectory,
                "turn": trajectory["turns"][0],
                "trajectory_uid": trajectory["trajectory_uid"],
                "turn_idx": 0,
            }
        )

    async def generate(_request, _endpoint):
        return {
            "text": "<thinking>ok</thinking><action>look</action>",
            "meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.2, 501]]},
        }

    kwargs = {
        "output_dir": tmp_path,
        "endpoints": ["fake://0", "fake://1"],
        "tokenizer": None,
        "expected_count": 4,
        "worker_count": 2,
        "global_concurrency": 4,
        "direct_binding": _direct_binding(),
    }
    asyncio.run(materialize_records(steps, generate_request=generate, **kwargs))

    async def no_request(_request, _endpoint):
        raise AssertionError("complete EPD output must be reused without requests")

    asyncio.run(materialize_records(steps, generate_request=no_request, **kwargs))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert all(value >= 1 for value in manifest["observed_max_inflight_by_endpoint"].values())


@pytest.mark.unit
def test_epd_materializer_repairs_only_a_trailing_interrupted_progress_line(tmp_path):
    from slime_plugins.agent_tasks.alfworld.frozen.epd import materialize_records

    trajectory = _trajectory()
    step = {
        "trajectory": trajectory,
        "turn": trajectory["turns"][0],
        "trajectory_uid": trajectory["trajectory_uid"],
        "turn_idx": 0,
    }

    async def generate(_request, _endpoint):
        return {
            "text": "<thinking>ok</thinking><action>look</action>",
            "meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.2, 501]]},
        }

    kwargs = {
        "output_dir": tmp_path,
        "endpoints": ["fake://0"],
        "tokenizer": None,
        "expected_count": 1,
        "worker_count": 1,
        "direct_binding": _direct_binding(),
    }
    asyncio.run(materialize_records([step], generate_request=generate, **kwargs))
    progress = tmp_path / "progress.jsonl"
    with progress.open("ab") as handle:
        handle.write(b'{"identity":"interrupted')

    asyncio.run(materialize_records([step], generate_request=generate, **kwargs))
    assert progress.read_bytes().endswith(b"\n")
    repair = json.loads((tmp_path / "progress_repair.json").read_text())
    assert repair["removed_bytes"] > 0


@pytest.mark.unit
def test_epd_materializer_attempt_seed_offset_changes_recovery_wave_seed(tmp_path):
    from slime_plugins.agent_tasks.alfworld.frozen.epd import materialize_records, retry_seed

    trajectory = _trajectory()
    step = {
        "trajectory": trajectory,
        "turn": trajectory["turns"][0],
        "trajectory_uid": trajectory["trajectory_uid"],
        "turn_idx": 0,
    }
    observed = []

    async def generate(request, _endpoint):
        observed.append(request["sampling_params"]["sampling_seed"])
        return {
            "text": "<thinking>ok</thinking><action>look</action>",
            "meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.2, 501]]},
        }

    asyncio.run(
        materialize_records(
            [step],
            output_dir=tmp_path,
            endpoints=["fake://0"],
            tokenizer=None,
            generate_request=generate,
            expected_count=1,
            worker_count=1,
            attempt_seed_offset=3,
            direct_binding=_direct_binding(),
        )
    )
    assert observed == [retry_seed(trajectory["trajectory_uid"], 0, 3)]


@pytest.mark.unit
def test_epd_materializer_rejects_a_malformed_middle_progress_line(tmp_path):
    from slime_plugins.agent_tasks.alfworld.frozen.epd import materialize_records

    trajectories = []
    steps = []
    for index in range(2):
        trajectory = _trajectory()
        trajectory["trajectory_uid"] = f"middle-{index}"
        trajectories.append(trajectory)
        steps.append(
            {
                "trajectory": trajectory,
                "turn": trajectory["turns"][0],
                "trajectory_uid": trajectory["trajectory_uid"],
                "turn_idx": 0,
            }
        )

    async def generate(_request, _endpoint):
        return {
            "text": "<thinking>ok</thinking><action>look</action>",
            "meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.2, 501]]},
        }

    asyncio.run(
        materialize_records(
            steps,
            output_dir=tmp_path,
            endpoints=["fake://0"],
            tokenizer=None,
            generate_request=generate,
            expected_count=2,
            worker_count=1,
            direct_binding=_direct_binding(),
        )
    )
    progress = tmp_path / "progress.jsonl"
    lines = progress.read_bytes().splitlines(keepends=True)
    progress.write_bytes(lines[0] + b"not-json\n" + lines[1])

    with pytest.raises(ValueError, match="malformed middle"):
        asyncio.run(
            materialize_records(
                steps,
                output_dir=tmp_path,
                endpoints=["fake://0"],
                tokenizer=None,
                generate_request=generate,
                expected_count=2,
                worker_count=1,
                direct_binding=_direct_binding(),
            )
        )
