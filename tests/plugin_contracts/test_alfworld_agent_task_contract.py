from __future__ import annotations

import asyncio
import gzip
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

try:
    from ._shared import install_paths, install_stubs
except ImportError:
    try:
        from plugin_contracts._shared import install_paths, install_stubs
    except ImportError:
        from _shared import install_paths, install_stubs

install_paths()
install_stubs(with_ray=False)

NUM_GPUS = 0

from slime.utils.types import Sample
from slime_plugins.agent_tasks.alfworld import envs as alfworld_envs
from slime_plugins.agent_tasks.alfworld import eval as alfworld_eval
from slime_plugins.agent_tasks.alfworld import projection as alfworld_projection
from slime_plugins.agent_tasks.alfworld import rewards as alfworld_rewards
from slime_plugins.agent_tasks.alfworld.config import get_alfworld_config
from slime_plugins.agent_tasks.alfworld.data_source import AlfWorldDataSource
from slime_plugins.agent_tasks.alfworld.envs import ensure_alfworld_cache
from slime_plugins.agent_tasks.alfworld.eval import generate_eval_rollout
from slime_plugins.agent_tasks.alfworld.generate import generate
from slime_plugins.agent_tasks.alfworld.log import log_eval_samples, log_rollout_samples
from slime_plugins.agent_tasks.alfworld.projection import project_response
from slime_plugins.agent_tasks.alfworld.prompts import build_prompt
from slime_plugins.agent_tasks.alfworld.rewards import post_process_rewards
from slime_plugins.agent_tasks.common.data_source import GroupedPlaceholderDataSource
from slime_plugins.agent_tasks.common.entropy_postprocess import postprocess_rollout_entropy
from slime_plugins.agent_tasks.common.eval import build_eval_dataset_output, build_eval_placeholder_sample
from slime_plugins.agent_tasks.common.history import TextHistory
from slime_plugins.agent_tasks.common.logging import summarize_alfworld_samples, summarize_diversity_samples
from slime_plugins.agent_tasks.common.projection import project_tagged_action_response
from slime_plugins.agent_tasks.common.trace import (
    add_sample_trace_record,
    add_trace_record,
    build_trace_record,
    compact_prompt_metadata,
    clear_trace_records,
    drain_trace_records_for_samples,
    get_agent_trace_config,
    write_trace_sidecar,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        rendered = "".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages)
        if kwargs.get("enable_thinking") is False:
            rendered += "<think></think>"
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 251 for char in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(65 + (token_id % 26)) for token_id in token_ids)


class ExactCharacterTokenizer(FakeTokenizer):
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        return {
            "input_ids": self.encode(text, add_special_tokens=add_special_tokens),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(token_id) for token_id in token_ids)


class BoundaryStraddlingTokenizer(FakeTokenizer):
    def __init__(self):
        self.last_encoded_text = ""

    def encode(self, text, add_special_tokens=False):
        self.last_encoded_text = text
        return [1]

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        self.last_encoded_text = text
        return {"input_ids": [1], "offset_mapping": [(0, len(text))]}

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return self.last_encoded_text if token_ids else ""


class BoundaryWithInteriorTokenizer(ExactCharacterTokenizer):
    def __init__(self, target_text: str, target_spans: list[tuple[int, int]]):
        self.target_text = target_text
        self.target_spans = target_spans
        self.target_ids = [10_000 + index for index in range(len(target_spans))]

    def encode(self, text, add_special_tokens=False):
        if text == self.target_text:
            return list(self.target_ids)
        return super().encode(text, add_special_tokens=add_special_tokens)

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        if text == self.target_text:
            return {"input_ids": list(self.target_ids), "offset_mapping": list(self.target_spans)}
        return super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
        )


class FakeResetResult:
    def __init__(self, observation: str, info: dict, worker_id: str):
        self.observation = observation
        self.info = info
        self.reset_seconds = 0.01
        self.worker_id = worker_id
        self.reused_worker = False
        self.worker_created = True


class FakeAlfWorldEnv:
    def __init__(self, *, seed: int, split: str, max_episode_steps: int, plan: list[tuple[str, bool, float]]):
        self.seed = seed
        self.split = split
        self.max_episode_steps = max_episode_steps
        self.plan = plan
        self.step_index = 0
        self.worker_id = f"fake-{split}-{seed}"
        self.actions: list[str] = []

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        return FakeResetResult(
            "You are in a kitchen. Your task is to: put a clean mug on the counter",
            {"admissible_commands": ["look", "open fridge", "take mug", "help"], "won": 0},
            self.worker_id,
        )

    def step(self, action: str):
        self.actions.append(action)
        observation, done, won = self.plan[min(self.step_index, len(self.plan) - 1)]
        self.step_index += 1
        return observation, won, done, {"admissible_commands": ["look", "take mug"], "won": won}

    def close(self):
        pass


class FakeEnvFactory:
    def __init__(self, plan: list[tuple[str, bool, float]] | None = None):
        self.plan = plan or [("middle obs", False, 0.0), ("final obs", True, 1.0)]
        self.created: list[FakeAlfWorldEnv] = []

    def __call__(self, *, seed: int, split: str, max_episode_steps: int):
        env = FakeAlfWorldEnv(seed=seed, split=split, max_episode_steps=max_episode_steps, plan=self.plan)
        self.created.append(env)
        return env


class FailingStepEnv(FakeAlfWorldEnv):
    def step(self, action: str):
        self.actions.append(action)
        raise IndexError("Cannot choose from an empty sequence")


class FailingStepEnvFactory(FakeEnvFactory):
    def __call__(self, *, seed: int, split: str, max_episode_steps: int):
        env = FailingStepEnv(seed=seed, split=split, max_episode_steps=max_episode_steps, plan=self.plan)
        self.created.append(env)
        return env


class FakeGenerator:
    def __init__(self, responses: list[str] | None = None, finish_reasons: list[str] | None = None):
        self.responses = responses or [
            "<thinking>reason</thinking><action>open fridge</action>",
            "<thinking>reason</thinking><action>take mug</action>",
        ]
        self.finish_reasons = finish_reasons or ["stop"] * len(self.responses)
        self.calls = 0

    async def __call__(self, *, args, prompt_ids, rendered_prompt, sampling_params, session_id):
        call_idx = self.calls
        response = self.responses[min(call_idx, len(self.responses) - 1)]
        finish_reason = self.finish_reasons[min(call_idx, len(self.finish_reasons) - 1)]
        self.calls += 1
        token_ids = [900 + self.calls, 1000 + self.calls]
        return {
            "text": response,
            "meta_info": {
                "finish_reason": {"type": finish_reason},
                "output_token_logprobs": [[-0.1, token_id] for token_id in token_ids],
                "prompt_tokens": len(prompt_ids),
                "cached_tokens": 3,
                "completion_tokens": len(token_ids),
            },
        }


class ExactCharacterGenerator(FakeGenerator):
    async def __call__(self, *, args, prompt_ids, rendered_prompt, sampling_params, session_id):
        call_idx = self.calls
        response = self.responses[min(call_idx, len(self.responses) - 1)]
        finish_reason = self.finish_reasons[min(call_idx, len(self.finish_reasons) - 1)]
        self.calls += 1
        token_ids = args.alfworld_tokenizer.encode(response, add_special_tokens=False)
        return {
            "text": response,
            "meta_info": {
                "finish_reason": {"type": finish_reason},
                "output_token_logprobs": [[-0.1, token_id] for token_id in token_ids],
                "prompt_tokens": len(prompt_ids),
                "cached_tokens": 3,
                "completion_tokens": len(token_ids),
            },
        }


def make_args(tmp_path: Path, **overrides):
    args = SimpleNamespace(
        rollout_seed=7,
        rollout_batch_size=16,
        n_samples_per_prompt=2,
        wandb_always_use_train_step=True,
        apply_chat_template=True,
        apply_chat_template_kwargs={"enable_thinking": False},
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=16,
        rollout_max_prompt_len=4096,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        eval_temperature=0.4,
        eval_top_p=1.0,
        eval_top_k=-1,
        eval_max_response_len=16,
        reward_key=None,
        grpo_std_normalization=True,
        global_batch_size=4,
        alfworld_tokenizer=FakeTokenizer(),
        alfworld_generator=FakeGenerator(),
        alfworld_env_factory=FakeEnvFactory(),
        alfworld_max_episode_steps=30,
        alfworld_invalid_action_penalty=0.1,
        alfworld_eval_episodes=3,
        alfworld_eval_concurrency=2,
        alfworld_sample_log_dir=str(tmp_path / "samples"),
        alfworld_sample_log_limit=100,
        alfworld_cache_dir=str(tmp_path / "alfworld_cache"),
        agent_task_keep_prompt_metadata=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def get_default_rollout_manager_cls(monkeypatch: pytest.MonkeyPatch):
    # Preserve the real module for restoration after this test's import stubs.
    importlib.import_module("slime.ray.rollout")
    ray_module = ModuleType("ray")
    ray_util_module = ModuleType("ray.util")
    scheduling_module = ModuleType("ray.util.scheduling_strategies")

    def remote(target=None, **kwargs):
        if target is None:
            return lambda decorated: decorated
        return target

    class PlacementGroupSchedulingStrategy:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    scheduling_module.PlacementGroupSchedulingStrategy = PlacementGroupSchedulingStrategy
    ray_util_module.scheduling_strategies = scheduling_module
    ray_module.remote = remote
    ray_module.util = ray_util_module
    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util_module)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", scheduling_module)

    for module_name in list(sys.modules):
        if module_name == "sglang" or module_name.startswith("sglang."):
            monkeypatch.delitem(sys.modules, module_name)

    sglang_module = ModuleType("sglang")
    srt_module = ModuleType("sglang.srt")
    constants_module = ModuleType("sglang.srt.constants")
    constants_module.GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"
    constants_module.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
    constants_module.GPU_MEMORY_TYPE_WEIGHTS = "weights"
    server_args_module = ModuleType("sglang.srt.server_args")
    server_args_module.ServerArgs = type("ServerArgs", (), {})
    utils_module = ModuleType("sglang.srt.utils")
    utils_module.kill_process_tree = lambda *args, **kwargs: None
    srt_module.constants = constants_module
    srt_module.server_args = server_args_module
    srt_module.utils = utils_module
    sglang_module.srt = srt_module
    monkeypatch.setitem(sys.modules, "sglang", sglang_module)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.constants", constants_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils_module)

    monkeypatch.delitem(sys.modules, "slime.ray.rollout", raising=False)
    return importlib.import_module("slime.ray.rollout").RolloutManager


@pytest.mark.unit
def test_config_defaults_repo_cache_and_max_step(monkeypatch):
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    monkeypatch.delenv("ALFWORLD_MAX_EPISODE_STEPS", raising=False)

    config = get_alfworld_config(SimpleNamespace())

    assert config.cache_dir == Path(".cache/alfworld")
    assert config.max_episode_steps == 30
    assert config.history_format == "inline"
    assert config.history_max_steps is None
    assert config.history_assistant_content == "full_response"
    assert config.env_pool_size == 128
    assert config.prewarm_env_pool is True
    assert config.eval_concurrency == 128
    assert config.max_episode_errors == 8
    assert config.max_episode_error_rate == 0.05
    assert config.eval_dataset_name == "alfworld_eval"
    assert config.eval_split == "eval_in_distribution"
    assert config.eval_out_of_distribution_dataset_name is None
    assert config.eval_out_of_distribution_split == "eval_out_of_distribution"


@pytest.mark.unit
def test_config_accepts_legacy_history_length_alias(tmp_path: Path):
    config = get_alfworld_config(
        SimpleNamespace(alfworld_cache_dir=str(tmp_path), alfworld_history_format="inline", alfworld_history_length=2)
    )

    assert config.history_format == "inline"
    assert config.history_max_steps == 2


@pytest.mark.unit
def test_config_legacy_history_length_only_keeps_inline_behavior(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("ALFWORLD_HISTORY_FORMAT", raising=False)
    monkeypatch.delenv("ALFWORLD_HISTORY_MAX_STEPS", raising=False)
    monkeypatch.setenv("ALFWORLD_HISTORY_LENGTH", "2")

    config = get_alfworld_config(SimpleNamespace(alfworld_cache_dir=str(tmp_path)))

    assert config.history_format == "inline"
    assert config.history_max_steps == 2


@pytest.mark.unit
def test_config_history_max_steps_wins_over_legacy_length(tmp_path: Path):
    config = get_alfworld_config(
        SimpleNamespace(
            alfworld_cache_dir=str(tmp_path),
            alfworld_history_max_steps=3,
            alfworld_history_length=2,
        )
    )

    assert config.history_format == "inline"
    assert config.history_max_steps == 3


@pytest.mark.unit
def test_default_textworld_config_uses_dqn_without_expert_plan(tmp_path: Path):
    config = alfworld_envs.default_textworld_config(tmp_path, max_episode_steps=32)

    assert config["general"]["training_method"] == "dqn"
    assert config["rl"]["training"]["max_nb_steps_per_episode"] == 32


@pytest.mark.unit
def test_compact_info_keeps_only_rollout_required_fields():
    info = {
        "admissible_commands": [["look", "open fridge"]],
        "won": [False],
        "extra.gamefile": ["game/path"],
        "extra.expert_plan": [["look"]],
        "facts": [{"large": "unused"}],
    }

    compact = alfworld_envs._compact_info(info)

    assert compact == {
        "admissible_commands": ["look", "open fridge"],
        "won": False,
        "extra.gamefile": "game/path",
    }


@pytest.mark.unit
def test_env_pool_prewarm_creates_ready_idle_workers(monkeypatch, tmp_path: Path):
    created = []

    class FakeProcessWorker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            self.pinged = False
            created.append(self)

        @property
        def dirty(self):
            return self.closed

        def ping(self):
            self.pinged = True
            return {"worker_id": f"fake-{self.kwargs['worker_index']}"}

        def close(self):
            self.closed = True

    monkeypatch.setattr(alfworld_envs, "ProcessAlfWorldEnvWorker", FakeProcessWorker)
    pool = alfworld_envs.AlfWorldEnvPool(
        pool_size=3,
        split="train",
        cache_dir=tmp_path,
        max_episode_steps=32,
        suppress_output=True,
    )

    asyncio.run(pool.prewarm_async(seed=123))
    worker = asyncio.run(pool.acquire_async(seed=999))

    assert len(created) == 3
    assert all(item.pinged for item in created)
    assert worker in created
    assert [item.kwargs["seed"] for item in created] == [123, 123, 123]
    assert [item.kwargs["worker_index"] for item in created] == [0, 1, 2]
    pool.close()


@pytest.mark.unit
def test_env_pool_prewarm_closes_workers_on_ping_failure(monkeypatch, tmp_path: Path):
    created = []

    class FakeProcessWorker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            created.append(self)

        @property
        def dirty(self):
            return self.closed

        def ping(self):
            if self.kwargs["worker_index"] == 1:
                raise RuntimeError("prewarm failed")
            return {"worker_id": f"fake-{self.kwargs['worker_index']}"}

        def close(self):
            self.closed = True

    monkeypatch.setattr(alfworld_envs, "ProcessAlfWorldEnvWorker", FakeProcessWorker)
    pool = alfworld_envs.AlfWorldEnvPool(
        pool_size=3,
        split="train",
        cache_dir=tmp_path,
        max_episode_steps=32,
        suppress_output=True,
    )

    with pytest.raises(RuntimeError, match="prewarm failed"):
        asyncio.run(pool.prewarm_async(seed=123))

    assert len(created) == 3
    assert all(item.closed for item in created)
























@pytest.mark.unit
def test_projection_extracts_first_action_and_reports_tag_diagnostics():
    custom_thinking = project_response("<thinking>ok</thinking><action>Open Fridge</action><action>look</action>")
    native_think = project_response("<think>ok</think><action>look</action>")
    missing_thinking = project_response("<action>look</action>")
    missing_action = project_response("<thinking>ok</thinking>look")
    chinese = project_response("<thinking>打开</thinking><action>look</action>")

    assert custom_thinking.projected_action == "open fridge"
    assert custom_thinking.format_valid is True
    assert custom_thinking.missing_thinking_tag is False
    assert native_think.format_valid is True
    assert native_think.missing_thinking_tag is False
    assert missing_thinking.format_valid is False
    assert missing_thinking.missing_thinking_tag is True
    assert missing_thinking.invalid_reason == "missing_thinking_tag"
    assert missing_action.format_valid is False
    assert missing_action.missing_action_tag is True
    assert missing_action.missing_thinking_tag is False
    assert "missing_action_tag" in missing_action.invalid_reason
    assert chinese.contains_chinese is True
    assert "contains_chinese" in chinese.invalid_reason


@pytest.mark.unit
def test_projection_reports_missing_thinking_without_requiring_it():
    projection = project_tagged_action_response("<action>look</action>", require_thinking_tag=False)

    assert projection.format_valid is True
    assert projection.missing_thinking_tag is True
    assert projection.invalid_reason is None


@pytest.mark.unit
def test_data_source_groups_share_uid_and_reject_buffer_recycle(tmp_path: Path):
    args = make_args(tmp_path, n_samples_per_prompt=3)
    data_source = AlfWorldDataSource(args)

    assert isinstance(data_source, GroupedPlaceholderDataSource)
    groups = data_source.get_samples(2)

    assert len(groups) == 2
    assert len(groups[0]) == 3
    assert {sample.metadata["uid"] for sample in groups[0]} == {"alfworld-train-00000000"}
    assert [sample.metadata["repeat_idx"] for sample in groups[0]] == [0, 1, 2]
    assert [sample.index for sample in groups[0] + groups[1]] == list(range(6))
    with pytest.raises(RuntimeError, match="variable step-row"):
        data_source.add_samples(groups)


@pytest.mark.unit
def test_data_source_shuffle_uses_rollout_seed_and_preserves_unique_uids(tmp_path: Path):
    args = make_args(tmp_path, n_samples_per_prompt=2, rollout_shuffle=True, alfworld_num_groups=8)

    groups = AlfWorldDataSource(args).get_samples(3)

    assert [group[0].metadata["source_group_index"] for group in groups] == [6, 7, 2]
    assert [group[0].metadata["sample_group_index"] for group in groups] == [0, 1, 2]
    assert [group[0].metadata["seed"] for group in groups] == [13, 14, 9]
    assert [group[0].metadata["uid"] for group in groups] == [
        "alfworld-train-00000000",
        "alfworld-train-00000001",
        "alfworld-train-00000002",
    ]
    assert all(sample.metadata["train_shuffle"] is True for group in groups for sample in group)


@pytest.mark.unit
def test_common_eval_helpers_match_alfworld_output_shape():
    sample = build_eval_placeholder_sample(
        rollout_id=4,
        dataset_idx=1,
        dataset_name="eval_out_of_distribution",
        episode_idx=2,
        seed=44,
        split="eval_out_of_distribution",
    )
    sample.reward = 3.0
    sample.status = Sample.Status.TRUNCATED

    output = build_eval_dataset_output([[sample]])

    assert sample.index == 41_000_002
    assert sample.metadata["uid"] == "eval_out_of_distribution-0004-000002"
    assert sample.metadata["traj_uid"] == "eval_out_of_distribution-0004-000002-traj-00"
    assert sample.metadata["seed"] == 44
    assert output == {
        "rewards": [3.0],
        "truncated": [True],
        "samples": [sample],
        "step_samples": [sample],
    }


@pytest.mark.unit
def test_custom_generate_returns_step_segments_with_shared_rollout_id(tmp_path: Path):
    args = make_args(tmp_path)
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert len(rows) == 2
    assert {row.rollout_id for row in rows} == {placeholder.index}
    assert [row.index for row in rows] == [0, 1]
    assert all(row.loss_mask == [1, 1] for row in rows)
    assert all(row.rollout_log_probs == [-0.1, -0.1] for row in rows)
    assert all(row.metadata["uid"] == "alfworld-train-00000000" for row in rows)
    assert all(row.metadata["episode_reward"] == 1.0 for row in rows)
    assert all(row.reward == 1.0 for row in rows)
    assert all(row.metadata["episode_error"] == 0.0 for row in rows)
    assert "<user>" in rows[0].metadata["rendered_prompt"]
    assert "put a clean mug on the counter" in rows[0].metadata["raw_prompt"]
    assert [message["role"] for message in rows[1].metadata["messages"]] == ["user"]
    assert "Action 1: 'open fridge'" in rows[1].metadata["messages"][0]["content"]
    assert rows[1].metadata["history_assistant_content"] == "full_response"
    assert rows[-1].metadata["is_terminal"] is True
    assert rows[0].metadata["admissible_member"] is True
    assert "expectation_enabled" not in rows[0].metadata


@pytest.mark.unit
def test_custom_generate_propagates_missing_thinking_diagnostic(tmp_path: Path):
    args = make_args(
        tmp_path,
        alfworld_generator=FakeGenerator(
            responses=[
                "<action>open fridge</action>",
                "<think>reason</think><action>take mug</action>",
            ]
        ),
    )
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert [row.metadata["missing_thinking_tag"] for row in rows] == [True, False]
    assert [row.metadata["format_valid"] for row in rows] == [False, True]
    assert rows[0].metadata["invalid_reason"] == "missing_thinking_tag"


@pytest.mark.unit
def test_chat_history_can_keep_only_recent_n_steps(tmp_path: Path):
    args = make_args(
        tmp_path,
        alfworld_history_format="chat",
        alfworld_history_max_steps=1,
        alfworld_env_factory=FakeEnvFactory(
            plan=[
                ("obs after first action", False, 0.0),
                ("obs after second action", False, 0.0),
                ("final obs", True, 1.0),
            ]
        ),
        alfworld_generator=FakeGenerator(
            responses=[
                "<thinking>first</thinking><action>open fridge</action>",
                "<thinking>second</thinking><action>look</action>",
                "<thinking>third</thinking><action>take mug</action>",
            ]
        ),
    )
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    turn_two_messages = rows[2].metadata["messages"]
    assert [message["role"] for message in turn_two_messages] == ["user", "user", "assistant", "user"]
    assert "obs after first action" in turn_two_messages[1]["content"]
    assert "Your admissible actions of the current situation are" in turn_two_messages[1]["content"]
    assert "'take mug'" in turn_two_messages[1]["content"]
    assert turn_two_messages[2]["content"] == "<thinking>second</thinking><action>look</action>"
    assert "open fridge" not in rows[2].metadata["raw_prompt"]


@pytest.mark.unit
def test_chat_history_can_preserve_full_assistant_response_when_configured(tmp_path: Path):
    args = make_args(tmp_path, alfworld_history_format="chat", alfworld_history_assistant_content="full_response")
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert rows[1].metadata["messages"][2]["content"] == "<thinking>reason</thinking><action>open fridge</action>"
    assert rows[1].metadata["history_assistant_content"] == "full_response"


@pytest.mark.unit
def test_full_chat_auto_truncates_to_recent_history_when_prompt_budget_is_hit(tmp_path: Path):
    long_reason = " ".join(["very-long-reasoning"] * 120)
    args = make_args(
        tmp_path,
        alfworld_history_format="chat",
        rollout_max_prompt_len=1200,
        alfworld_history_assistant_content="full_response",
        alfworld_env_factory=FakeEnvFactory(
            plan=[
                ("obs after first action", False, 0.0),
                ("obs after second action", False, 0.0),
                ("final obs", True, 1.0),
            ]
        ),
        alfworld_generator=FakeGenerator(
            responses=[
                f"<thinking>{long_reason}</thinking><action>open fridge</action>",
                "<thinking>short</thinking><action>look</action>",
                "<thinking>third</thinking><action>take mug</action>",
            ]
        ),
    )
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert rows[1].metadata["history_auto_truncated"] is True
    assert rows[1].metadata["history_steps_total"] == 1
    assert rows[1].metadata["history_steps_kept"] == 0
    assert rows[1].metadata["history_auto_truncated_dropped"] == 1
    assert rows[1].metadata["full_prompt_tokens_before_truncation"] > rows[1].metadata["prompt_tokens"]
    assert rows[1].metadata["prompt_overlength"] is False
    assert "very-long-reasoning" not in rows[1].metadata["raw_prompt"]


@pytest.mark.unit
def test_chat_history_zero_steps_keeps_task_instruction_and_current_observation(tmp_path: Path):
    args = make_args(tmp_path, alfworld_history_format="chat", alfworld_history_max_steps=0)
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert [message["role"] for message in rows[1].metadata["messages"]] == ["user", "user"]
    assert "Your task is to: put a clean mug on the counter" in rows[1].metadata["messages"][0]["content"]
    assert "middle obs" in rows[1].metadata["messages"][1]["content"]
    assert "'take mug'" in rows[1].metadata["messages"][1]["content"]
    assert "<action>open fridge</action>" not in rows[1].metadata["raw_prompt"]


@pytest.mark.unit
def test_inline_history_format_keeps_single_user_message(tmp_path: Path):
    args = make_args(tmp_path, alfworld_history_format="inline", alfworld_history_max_steps=2)
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert [message["role"] for message in rows[1].metadata["messages"]] == ["user"]
    assert "most recent 1 observations" in rows[1].metadata["messages"][0]["content"]
    assert "'take mug'" in rows[1].metadata["messages"][0]["content"]
    assert "Action 1: 'open fridge'" in rows[1].metadata["messages"][0]["content"]


@pytest.mark.unit
def test_custom_generate_recovers_env_step_error_as_terminal_error_sample(tmp_path: Path):
    env_factory = FailingStepEnvFactory()
    args = make_args(tmp_path, alfworld_env_factory=env_factory)
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert len(rows) == 1
    assert env_factory.created[0].actions == ["open fridge"]
    row = rows[0]
    assert row.metadata["is_terminal"] is True
    assert row.metadata["episode_error"] == 1.0
    assert row.metadata["episode_error_count"] == 1.0
    assert row.metadata["episode_error_type"] == "IndexError"
    assert row.metadata["episode_error_stage"] == "env_step"
    assert "Cannot choose from an empty sequence" in row.metadata["episode_error_message"]
    assert row.metadata["env_step_failed"] is True
    assert row.metadata["success"] is False
    assert row.metadata["episode_reward"] == 0.0
    assert row.reward == 0.0
    assert row.rollout_log_probs == [-0.1, -0.1]


@pytest.mark.unit
def test_step_segments_survive_default_converter_with_rollout_mask_sums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    args = make_args(tmp_path)
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]
    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))
    manager = object.__new__(get_default_rollout_manager_cls(monkeypatch))
    manager.args = args
    manager.custom_reward_post_process_func = post_process_rewards
    manager.custom_convert_samples_to_train_data_func = None

    train_data = manager._convert_samples_to_train_data(rows)

    expected_mask_sum = sum(sum(row.loss_mask) for row in rows)
    assert train_data["rollout_ids"] == [placeholder.index, placeholder.index]
    assert train_data["rollout_mask_sums"] == [expected_mask_sum, expected_mask_sum]
    assert train_data["loss_masks"] == [[1, 1], [1, 1]]
    assert train_data["rollout_log_probs"] == [[-0.1, -0.1], [-0.1, -0.1]]
    assert [(row["uid"], row["traj_uid"], row["turn_idx"]) for row in train_data["metadata"]] == [
        ("alfworld-train-00000000", "alfworld-train-00000000-traj-00", 0),
        ("alfworld-train-00000000", "alfworld-train-00000000-traj-00", 1),
    ]
    assert all(row["agent_task"] == "alfworld" for row in train_data["metadata"])
    assert [row["sample_index"] for row in train_data["metadata"]] == [0, 1]


@pytest.mark.unit
def test_custom_generate_populates_sdpo_train_metadata_and_preserves_trace_keys(tmp_path: Path):
    args = make_args(
        tmp_path,
        agent_task_keep_prompt_metadata=False,
        custom_convert_samples_to_train_data_path=(
            "slime_plugins.agent_tasks.common.algorithms.sdpo.convert_samples_to_train_data"
        ),
    )
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    first = rows[0]
    assert first.train_metadata["uid"] == "alfworld-train-00000000"
    assert first.train_metadata["agent_task"] == "alfworld"
    assert first.train_metadata["agent_task_trace_dir"] is not None
    assert first.train_metadata["sample_rollout_id"] == placeholder.index
    assert first.train_metadata["sample_index"] == first.index

    sdpo = first.train_metadata.get("sdpo")
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
    assert isinstance(sdpo, dict)
    assert required <= set(sdpo)
    assert sdpo["uid"] == first.metadata["uid"]
    assert sdpo["traj_uid"] == first.metadata["traj_uid"]
    assert sdpo["turn_idx"] == 0
    assert sdpo["task_text"] == "put a clean mug on the counter"
    assert sdpo["anchor_obs"].startswith("You are in a kitchen")
    assert sdpo["next_anchor_obs"] == "middle obs"
    assert sdpo["projected_action"] == "open fridge"
    assert sdpo["is_action_valid"] is True
    assert sdpo["is_terminal"] == first.metadata["is_terminal"]
    assert sdpo["episode_rewards"] == pytest.approx(1.0)
    assert sdpo["episode_lengths"] == 2
    assert "open fridge" in sdpo["sdpo_current_prompt_text"]
    assert "<assistant>" not in sdpo["sdpo_current_prompt_text"]
    assert "USER:" not in sdpo["sdpo_current_prompt_text"]
    assert isinstance(sdpo["sdpo_current_raw_prompt"], list)
    assert sdpo["sdpo_current_raw_prompt"][0]["role"] == "user"
    assert "raw_prompt" not in first.metadata
    assert "messages" not in first.metadata
    assert "is_action_valid" not in first.metadata


@pytest.mark.unit
def test_custom_generate_populates_context_metadata_for_grpo_token_weights(tmp_path: Path):
    args = make_args(
        tmp_path,
        agent_task_keep_prompt_metadata=False,
        grpo_token_weights=True,
    )
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert all("sdpo" not in row.train_metadata for row in rows)
    assert all(isinstance(row.train_metadata.get("grpo_token_weight_context"), dict) for row in rows)
    assert [row.train_metadata["grpo_token_weight_context"]["episode_rewards"] for row in rows] == pytest.approx(
        [1.0, 1.0]
    )
    assert [row.train_metadata["grpo_token_weight_context"]["episode_lengths"] for row in rows] == [2, 2]
    assert "raw_prompt" not in rows[0].metadata
    assert "messages" not in rows[0].metadata


@pytest.mark.unit
def test_alfworld_sdpo_representation_mean_success_aggregation_uses_all_success_demos(tmp_path: Path):
    from slime_plugins.agent_tasks.common.algorithms.sdpo import convert_samples_to_train_data

    def make_sdpo_sample(index: int, traj_uid: str, reward: float, action: str, next_obs: str) -> Sample:
        metadata = {
            "uid": "task-a",
            "traj_uid": traj_uid,
            "turn_idx": 0,
            "task_text": "put a clean mug on the counter",
            "anchor_obs": "You are in a kitchen.",
            "next_anchor_obs": next_obs,
            "projected_action": action,
            "is_action_valid": True,
            "is_terminal": True,
            "episode_rewards": reward,
            "episode_lengths": 1,
            "sdpo_current_prompt_text": "Current observation: You are in a kitchen.\nAdmissible actions: look",
            "sdpo_current_raw_prompt": [
                {
                    "role": "user",
                    "content": "Current observation: You are in a kitchen.\nAdmissible actions: look",
                }
            ],
        }
        return Sample(
            index=index,
            rollout_id=index,
            tokens=[index, index + 100],
            response_length=1,
            reward=reward,
            loss_mask=[1],
            status=Sample.Status.COMPLETED,
            metadata={"uid": "task-a", "traj_uid": traj_uid},
            train_metadata={"sdpo": metadata},
        )

    success_a = make_sdpo_sample(0, "success-a", 1.0, "open fridge", "The fridge is open.")
    success_b = make_sdpo_sample(1, "success-b", 1.0, "take mug", "You take the mug.")
    failure = make_sdpo_sample(2, "failure", 0.0, "look", "You see a fridge.")
    args = make_args(
        tmp_path,
        agent_task_sdpo_metadata_profile="alfworld",
        sdpo_success_reward_threshold=1.0,
        sdpo_dont_reprompt_on_self_success=False,
        sdpo_teacher_context_mode="original",
        sdpo_no_success_context_mode="filter",
        sdpo_solution_context_format="trajectory_demo",
        sdpo_guidance_generation_mode="disabled",
        sdpo_guidance_summary_source="success_priority",
        sdpo_multi_turn_weighting="traj_equal",
        sdpo_max_demo_steps=None,
        sdpo_include_environment_feedback=True,
        sdpo_environment_feedback_only_without_solution=True,
        sdpo_distillation_mode="representation",
        sdpo_representation_success_aggregation="mean",
    )

    train_data = convert_samples_to_train_data(args, [success_a, success_b, failure])

    assert train_data["self_distillation_mask"] == pytest.approx([1.0, 1.0, 1.0])
    assert train_data["sdpo_teacher_signal_type"] == ["solution_demo", "solution_demo", "solution_demo"]
    assert train_data["sdpo_representation_success_aggregation"] == ["mean", "mean", "mean"]
    assert train_data["sdpo_representation_success_count"] == [2, 2, 2]
    assert train_data["self_distillation/representation_success_count_mean"] == pytest.approx([2.0, 2.0, 2.0])
    assert train_data["self_distillation/representation_success_count_max"] == pytest.approx([2.0, 2.0, 2.0])
    assert train_data["self_distillation/representation_mean_success_fraction"] == pytest.approx([1.0, 1.0, 1.0])
    assert train_data["self_distillation/representation_multi_success_fraction"] == pytest.approx([1.0, 1.0, 1.0])

    prompt_ensemble = train_data["sdpo_teacher_prompt_texts"][-1]
    assert len(prompt_ensemble) == 2
    assert "Action: `open fridge`" in prompt_ensemble[0]
    assert "Action: `take mug`" in prompt_ensemble[1]
    assert train_data["sdpo_teacher_prompt_text"][-1] == prompt_ensemble[0]
    assert len(train_data["sdpo_teacher_messages_list"][-1]) == 2


@pytest.mark.unit
def test_custom_generate_without_sdpo_keeps_prompts_out_of_train_metadata_and_reward_semantics(tmp_path: Path):
    args = make_args(
        tmp_path,
        agent_task_keep_prompt_metadata=False,
        alfworld_generator=FakeGenerator(
            responses=[
                "<thinking>reason</thinking><action>dance</action>",
                "<thinking>reason</thinking><action>take mug</action>",
            ]
        ),
    )
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    first = rows[0]
    assert "sdpo" not in first.train_metadata
    assert "raw_prompt" not in first.metadata
    assert "messages" not in first.metadata
    assert "is_action_valid" not in first.metadata
    assert first.metadata["format_valid"] is True
    assert first.metadata["admissible_member"] is False
    assert first.reward == pytest.approx(1.0)


@pytest.mark.unit
def test_custom_generate_rejects_abort_finish_reason_before_env_step(tmp_path: Path):
    env_factory = FakeEnvFactory()
    args = make_args(
        tmp_path,
        alfworld_env_factory=env_factory,
        alfworld_generator=FakeGenerator(
            responses=["<thinking>reason</thinking><action>open fridge</action>"],
            finish_reasons=["abort"],
        ),
    )
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]

    with pytest.raises(RuntimeError, match="generation aborted before env step"):
        asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert len(env_factory.created) == 1
    assert env_factory.created[0].actions == []


@pytest.mark.unit
def test_invalid_format_penalty_is_per_step_but_raw_reward_is_terminal(tmp_path: Path):
    args = make_args(
        tmp_path,
        alfworld_generator=FakeGenerator(
            responses=[
                "<thinking>reason</thinking>look",
                "<thinking>reason</thinking><action>take mug</action>",
            ]
        ),
    )
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert [row.metadata["raw_reward"] for row in rows] == [1.0, 1.0]
    assert [row.metadata["episode_reward"] for row in rows] == [1.0, 1.0]
    assert rows[0].reward == pytest.approx(0.9)
    assert rows[1].reward == pytest.approx(1.0)


@pytest.mark.unit
def test_grpo_reward_post_process_normalizes_by_uid_with_trajectory_scores(tmp_path: Path):
    args = make_args(tmp_path)
    samples = [
        Sample(reward=1.0, metadata={"uid": "task-a", "traj_uid": "task-a-traj-0"}),
        Sample(reward=1.0, metadata={"uid": "task-a", "traj_uid": "task-a-traj-0"}),
        Sample(reward=1.0, metadata={"uid": "task-a", "traj_uid": "task-a-traj-0"}),
        Sample(reward=0.0, metadata={"uid": "task-a", "traj_uid": "task-a-traj-1"}),
        Sample(reward=1.0, metadata={"uid": "task-b", "traj_uid": "task-b-traj-0"}),
    ]

    raw_rewards, rewards = post_process_rewards(args, samples)

    assert raw_rewards == [1.0, 1.0, 1.0, 0.0, 1.0]
    assert rewards[:4] == pytest.approx([0.707106, 0.707106, 0.707106, -0.707106], abs=2e-6)
    assert rewards[4] == pytest.approx(1.0)


@pytest.mark.unit
def test_eval_episode_count_is_not_multiplied_by_train_fanout(tmp_path: Path):
    args = make_args(tmp_path, n_samples_per_prompt=8, alfworld_eval_episodes=3)

    output = generate_eval_rollout(args, 4, AlfWorldDataSource(args), evaluation=True)

    assert set(output.data) == {"alfworld_eval"}
    data = output.data["alfworld_eval"]
    assert len(data["samples"]) == 3
    assert len(data["step_samples"]) == 6
    assert len(data["rewards"]) == 3
    assert len(args.alfworld_env_factory.created) == 3
    assert output.metrics == {}


@pytest.mark.unit
def test_eval_can_run_in_distribution_and_out_of_distribution_datasets(tmp_path: Path):
    args = make_args(
        tmp_path,
        n_samples_per_prompt=8,
        alfworld_eval_episodes=3,
        alfworld_eval_out_of_distribution_dataset_name="eval_out_of_distribution",
    )

    output = generate_eval_rollout(args, 4, AlfWorldDataSource(args), evaluation=True)

    assert list(output.data) == ["alfworld_eval", "eval_out_of_distribution"]
    for dataset_name in ["alfworld_eval", "eval_out_of_distribution"]:
        data = output.data[dataset_name]
        assert len(data["samples"]) == 3
        assert len(data["step_samples"]) == 6
        assert len(data["rewards"]) == 3
    created_splits = [env.split for env in args.alfworld_env_factory.created]
    assert created_splits.count("eval_in_distribution") == 3
    assert created_splits.count("eval_out_of_distribution") == 3
    assert [sample.metadata["split"] for sample in output.data["alfworld_eval"]["samples"]] == [
        "eval_in_distribution"
    ] * 3
    assert [sample.metadata["split"] for sample in output.data["eval_out_of_distribution"]["samples"]] == [
        "eval_out_of_distribution"
    ] * 3
    in_distribution_indices = {sample.index for sample in output.data["alfworld_eval"]["samples"]}
    out_of_distribution_indices = {sample.index for sample in output.data["eval_out_of_distribution"]["samples"]}
    assert in_distribution_indices.isdisjoint(out_of_distribution_indices)


@pytest.mark.unit
def test_dual_eval_datasets_share_one_concurrency_queue(monkeypatch, tmp_path: Path):
    active_count = 0
    max_active_count = 0

    async def fake_run_alfworld_episode(args, *, sample, sampling_params, config, split, evaluation=False):
        nonlocal active_count, max_active_count
        active_count += 1
        max_active_count = max(max_active_count, active_count)
        await asyncio.sleep(0)
        active_count -= 1
        return [
            Sample(
                index=sample.index,
                group_index=sample.group_index,
                prompt=sample.prompt,
                reward=1.0,
                status=Sample.Status.COMPLETED,
                metadata={**sample.metadata, "episode_reward": 1.0},
            )
        ]

    monkeypatch.setattr(alfworld_eval, "run_alfworld_episode", fake_run_alfworld_episode)
    args = make_args(
        tmp_path,
        alfworld_eval_episodes=1,
        alfworld_eval_concurrency=2,
        alfworld_eval_out_of_distribution_dataset_name="eval_out_of_distribution",
    )

    output = generate_eval_rollout(args, 4, AlfWorldDataSource(args), evaluation=True)

    assert list(output.data) == ["alfworld_eval", "eval_out_of_distribution"]
    assert max_active_count == 2


@pytest.mark.unit
def test_rollout_log_hook_writes_jsonl_and_readable_metrics(tmp_path: Path):
    args = make_args(tmp_path, global_batch_size=2)
    samples = []
    for rollout_id in [10, 10, 11, 12]:
        samples.append(
            Sample(
                index=rollout_id,
                rollout_id=rollout_id,
                response="ok",
                response_length=1,
                reward=1.0,
                loss_mask=[1],
                rollout_log_probs=[-0.1],
                status=Sample.Status.COMPLETED,
                metadata={
                    "uid": f"uid-{rollout_id}",
                    "traj_uid": f"traj-{rollout_id}",
                    "turn_idx": 0,
                    "rollout_id": rollout_id,
                    "format_valid": True,
                    "missing_action_tag": False,
                    "missing_thinking_tag": False,
                    "episode_reward": 1.0,
                    "success": True,
                    "episode_length": 1,
                    "episode_seconds": 0.5,
                    "env_horizon_reached": False,
                    "episode_error": 0.0,
                    "episode_error_count": 0.0,
                    "env_step_failed": False,
                    "contains_chinese": False,
                    "prompt_tokens": 10,
                    "raw_prompt_tokens": 10,
                    "response_tokens": 1,
                    "cached_tokens": 5,
                    "prompt_overlength": False,
                    "generation_seconds": 0.2,
                    "generation_request_seconds": 0.2,
                    "env_step_seconds": 0.1,
                    "reset_seconds": 0.05,
                    "raw_prompt": "Your current observation is: kitchen",
                    "rendered_prompt": "<user>Your current observation is: kitchen</user><assistant>",
                    "messages": [{"role": "user", "content": "Your current observation is: kitchen"}],
                },
            )
        )
    metrics = {}

    assert log_rollout_samples(0, args, samples, metrics, 1.0) is False

    assert metrics["rollout/alfworld/episode/count"] == 3.0
    assert metrics["rollout/alfworld/episode/success_rate"] == 1.0
    assert metrics["rollout/alfworld/action/format_valid_rate"] == 1.0
    assert metrics["rollout/alfworld/action/missing_thinking_tag_rate"] == 0.0
    assert metrics["rollout/alfworld/perf/prefix_cache_hit_rate"] == 0.5
    assert "rollout/alfworld/success" not in metrics
    assert "rollout/alfworld/episode_length_mean" not in metrics
    rollout_rows = [
        json.loads(line)
        for line in (tmp_path / "samples" / "rollout_0.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rollout_rows[0]["model_input"] == "<user>Your current observation is: kitchen</user><assistant>"
    assert rollout_rows[0]["response"] == "ok"
    assert rollout_rows[0]["missing_thinking_tag"] is False
    assert "model_input_raw" not in rollout_rows[0]
    assert "model_input_messages" not in rollout_rows[0]
    assert "prompt_preview" not in rollout_rows[0]
    assert "prompt_tail" not in rollout_rows[0]


@pytest.mark.unit
def test_rollout_log_hook_records_diversity_and_train_trace_sidecar(tmp_path: Path):
    clear_trace_records()
    args = make_args(
        tmp_path,
        agent_task_trace_enabled=True,
        agent_task_trace_dir=str(tmp_path / "traces"),
        agent_task_keep_prompt_metadata=False,
        agent_task_diversity_metrics_enabled=True,
    )
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]
    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))
    metrics = {}

    assert log_rollout_samples(3, args, [rows], metrics, 1.0) is False

    assert metrics["diversity/rollout/alfworld/sampled_nll/token_mean"] == pytest.approx(0.1)
    assert metrics["diversity/rollout/alfworld/sampled_logprob/token_mean"] == pytest.approx(-0.1)
    assert "diversity/rollout/alfworld/actor_entropy/coverage" not in metrics
    assert "rollout/diversity/rollout/alfworld/sampled_nll/token_mean" not in metrics
    path = tmp_path / "traces" / "alfworld" / "rollout_3.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as reader:
        trace_rows = [json.loads(line) for line in reader]
    assert len(trace_rows) == len(rows)
    assert trace_rows[0]["phase"] == "train"
    assert trace_rows[0]["outer_rollout_id"] == 3
    assert trace_rows[0]["sample_index"] == rows[0].index
    assert trace_rows[0]["rollout_log_probs"] == [-0.1, -0.1]
    assert trace_rows[0]["current_observation"].startswith("You are in a kitchen")
    assert trace_rows[0]["next_observation"] == "middle obs"
    assert trace_rows[0]["admissible_actions"] == ["look", "open fridge", "take mug", "help"]
    assert trace_rows[0]["actor_entropy"]["reason"] == "pending_scoring"
    assert "raw_prompt" not in trace_rows[0]
    assert "messages" not in trace_rows[0]
    assert "raw_prompt" not in rows[0].metadata
    assert "rendered_prompt" not in rows[0].metadata
    assert "messages" not in rows[0].metadata
    assert "current_observation" not in rows[0].metadata
    clear_trace_records()


@pytest.mark.unit
def test_log_hooks_skip_diversity_metrics_when_disabled(tmp_path: Path):
    args = make_args(
        tmp_path,
        use_wandb=False,
        use_tensorboard=False,
        agent_task_diversity_metrics_enabled=False,
        alfworld_eval_out_of_distribution_dataset_name="eval_out_of_distribution",
    )
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]
    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))
    rollout_metrics = {}
    assert log_rollout_samples(0, args, [rows], rollout_metrics, 1.0) is False
    assert not any(key.startswith("diversity/") for key in rollout_metrics)

    output = generate_eval_rollout(args, 0, AlfWorldDataSource(args), evaluation=True)
    eval_metrics = {}
    assert log_eval_samples(0, args, output.data, eval_metrics) is True
    assert not any(key.startswith("diversity/") for key in eval_metrics)


@pytest.mark.unit
def test_eval_log_hook_uses_readable_metric_namespace_and_skips_default_eval_keys(tmp_path: Path):
    args = make_args(
        tmp_path,
        use_wandb=False,
        use_tensorboard=False,
        alfworld_sample_log_limit=4,
        alfworld_eval_out_of_distribution_dataset_name="eval_out_of_distribution",
        agent_task_trace_enabled=True,
        agent_task_trace_dir=str(tmp_path / "traces"),
        agent_task_diversity_metrics_enabled=True,
    )
    clear_trace_records()
    output = generate_eval_rollout(args, 4, AlfWorldDataSource(args), evaluation=True)
    metrics = {}

    assert log_eval_samples(4, args, output.data, metrics) is True

    assert metrics["eval/alfworld_eval/episode/count"] == 3.0
    assert metrics["eval/alfworld_eval/episode/success_rate"] == 1.0
    assert metrics["eval/alfworld_eval/error/episode_rate"] == 0.0
    assert metrics["eval_out_of_distribution/alfworld_eval/episode/count"] == 3.0
    assert metrics["eval_out_of_distribution/alfworld_eval/episode/success_rate"] == 1.0
    assert metrics["eval_out_of_distribution/alfworld_eval/error/episode_rate"] == 0.0
    id_prefix = "eval/alfworld_eval/"
    ood_prefix = "eval_out_of_distribution/alfworld_eval/"
    id_suffixes = {key[len(id_prefix) :] for key in metrics if key.startswith(id_prefix)}
    ood_suffixes = {key[len(ood_prefix) :] for key in metrics if key.startswith(ood_prefix)}
    assert id_suffixes == ood_suffixes
    assert "eval/alfworld_eval/alfworld/success" not in metrics
    assert "eval/alfworld_eval/alfworld/episodes" not in metrics
    assert "eval/eval_out_of_distribution/episode/success_rate" not in metrics
    assert metrics["diversity/eval/alfworld_eval/sampled_nll/token_mean"] == pytest.approx(0.1)
    assert metrics["diversity/eval/eval_out_of_distribution/sampled_nll/token_mean"] == pytest.approx(0.1)
    eval_rows = [
        json.loads(line) for line in (tmp_path / "samples" / "eval_4.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(eval_rows) == 4
    assert {row["eval_dataset_name"] for row in eval_rows} == {"alfworld_eval", "eval_out_of_distribution"}
    assert {row["split"] for row in eval_rows} == {"eval_in_distribution", "eval_out_of_distribution"}
    assert all("model_input" in row for row in eval_rows)
    assert all("put a clean mug on the counter" in row["model_input"] for row in eval_rows)
    with gzip.open(tmp_path / "traces" / "alfworld" / "eval_4.jsonl.gz", "rt", encoding="utf-8") as reader:
        trace_rows = [json.loads(line) for line in reader]
    assert len(trace_rows) == 12
    assert {row["phase"] for row in trace_rows} == {"eval"}
    assert {row["eval_dataset_name"] for row in trace_rows} == {"alfworld_eval", "eval_out_of_distribution"}
    clear_trace_records()


@pytest.mark.unit
def test_summarize_alfworld_samples_reports_error_budget_without_old_flat_keys(tmp_path: Path):
    args = make_args(tmp_path, alfworld_env_factory=FailingStepEnvFactory())
    placeholder = AlfWorldDataSource(args).get_samples(1)[0][0]
    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    summary = summarize_alfworld_samples(
        rows,
        metric_root="alfworld",
        max_episode_errors=0,
        max_episode_error_rate=0.0,
    )

    assert summary["alfworld/episode/count"] == 1.0
    assert summary["alfworld/error/episode_count"] == 1.0
    assert summary["alfworld/error/env_step_count"] == 1.0
    assert summary["alfworld/error/budget_exceeded"] == 1.0
    assert "alfworld/success" not in summary


@pytest.mark.unit
def test_diversity_summary_uses_sampled_names_and_actor_entropy_summary():
    samples = [
        Sample(
            index=1,
            rollout_id=1,
            response="a",
            response_length=2,
            reward=1.0,
            loss_mask=[1, 1],
            rollout_log_probs=[-1.0, -3.0],
            status=Sample.Status.COMPLETED,
            metadata={
                "traj_uid": "traj-1",
                "turn_idx": 0,
                "success": True,
                "format_valid": True,
                "is_action_valid": True,
                "actor_entropy": {
                    "enabled": True,
                    "token_count": 2,
                    "token_mean": 0.5,
                    "token_std": 0.1,
                },
            },
        ),
        Sample(
            index=2,
            rollout_id=2,
            response="b",
            response_length=1,
            reward=0.0,
            loss_mask=[1],
            rollout_log_probs=[-6.0],
            status=Sample.Status.COMPLETED,
            metadata={
                "traj_uid": "traj-2",
                "turn_idx": 1,
                "success": False,
                "format_valid": False,
                "is_action_valid": False,
                "env_horizon_reached": True,
                "actor_entropy": {"enabled": False, "reason": "pending_scoring"},
            },
        ),
        Sample(
            index=3,
            rollout_id=3,
            response="c",
            response_length=1,
            reward=0.0,
            loss_mask=[1],
            rollout_log_probs=[-2.0],
            status=Sample.Status.COMPLETED,
            metadata={
                "traj_uid": "traj-3",
                "turn_idx": 2,
                "format_valid": False,
                "actor_entropy": {"enabled": False, "reason": "pending_scoring"},
            },
        ),
    ]

    metrics = summarize_diversity_samples(
        samples,
        metric_root="diversity/rollout/alfworld",
        low_conf_nll_threshold=5.0,
    )

    assert metrics["diversity/rollout/alfworld/sampled_logprob/token_mean"] == pytest.approx(-12.0 / 4.0)
    assert metrics["diversity/rollout/alfworld/sampled_nll/token_mean"] == pytest.approx(12.0 / 4.0)
    assert metrics["diversity/rollout/alfworld/sampled_nll/low_conf_token_rate"] == pytest.approx(1.0 / 4.0)
    assert metrics["diversity/rollout/alfworld/sampled_nll/by_success/success_token_mean"] == pytest.approx(2.0)
    assert metrics["diversity/rollout/alfworld/sampled_nll/by_action_valid/invalid_token_mean"] == pytest.approx(6.0)
    assert metrics["diversity/rollout/alfworld/sampled_nll/by_format_valid/invalid_token_mean"] == pytest.approx(4.0)
    assert metrics["diversity/rollout/alfworld/sampled_nll/by_success/failure_token_mean"] == pytest.approx(6.0)
    assert metrics["diversity/rollout/alfworld/actor_entropy/coverage"] == pytest.approx(1.0 / 3.0)
    assert metrics["diversity/rollout/alfworld/actor_entropy/token_mean"] == pytest.approx(0.5)
    assert metrics["diversity/rollout/alfworld/actor_entropy/token_std"] == pytest.approx(0.1)
    assert not any(key.startswith("rollout/") for key in metrics)
    assert "diversity/rollout/alfworld/entropy/token_mean" not in metrics


@pytest.mark.unit
def test_agent_trace_config_disabled_does_not_write(tmp_path: Path):
    args = make_args(
        tmp_path,
        agent_task_trace_enabled=False,
        agent_task_trace_dir=str(tmp_path / "traces"),
    )
    config = get_agent_trace_config(args, task="alfworld", sample_log_dir=args.alfworld_sample_log_dir)

    result = write_trace_sidecar(
        config,
        task="alfworld",
        phase="train",
        outer_rollout_id=0,
        records=[{"schema_name": "should_not_write"}],
    )

    assert result is None
    assert not (tmp_path / "traces").exists()


@pytest.mark.unit
def test_agent_trace_writer_outputs_gzip_schema_and_manifest(tmp_path: Path):
    args = make_args(
        tmp_path,
        agent_task_trace_enabled=True,
        agent_task_trace_dir=str(tmp_path / "traces"),
        agent_task_trace_compression="gzip",
    )
    config = get_agent_trace_config(args, task="alfworld", sample_log_dir=args.alfworld_sample_log_dir)
    record = build_trace_record(
        task="alfworld",
        phase="train",
        metadata={
            "uid": "uid-1",
            "traj_uid": "traj-1",
            "turn_idx": 0,
            "rollout_id": 99,
            "sample_index": 42,
            "group_index": 0,
            "task_id": "task-1",
            "seed": 7,
            "split": "train",
            "prompt_tokens": 12,
            "format_valid": True,
            "is_action_valid": True,
            "success": True,
            "actor_entropy": {
                "enabled": True,
                "source": "old_actor_forward",
                "token_count": 2,
                "token_mean": 0.4,
                "token_std": 0.05,
                "actor_entropy_tokens": [0.35, 0.45],
                "nan_value": float("nan"),
            },
        },
        response_text="<action>look</action>",
        response_token_ids=[101, 102],
        rollout_log_probs=[-0.2, float("inf")],
        prompt_text="hidden prompt",
        messages=[{"role": "user", "content": "hidden prompt"}],
        include_prompts=False,
        extra={"path_value": tmp_path / "x"},
    )

    result = write_trace_sidecar(
        config,
        task="alfworld",
        phase="train",
        outer_rollout_id=3,
        records=[record],
    )

    assert result is not None
    with gzip.open(result.path, "rt", encoding="utf-8") as reader:
        rows = [json.loads(line) for line in reader]
    assert rows[0]["schema_name"] == "agent_task_step_trace"
    assert rows[0]["schema_version"] == 1
    assert rows[0]["outer_rollout_id"] == 3
    assert rows[0]["sample_rollout_id"] == 99
    assert "prompt_hash" not in rows[0]
    assert "raw_prompt" not in rows[0]
    assert "messages" not in rows[0]
    assert rows[0]["rollout_log_probs"] == [-0.2, None]
    assert rows[0]["actor_entropy"]["enabled"] is True
    assert rows[0]["actor_entropy"]["token_mean"] == 0.4
    assert rows[0]["actor_entropy"]["nan_value"] is None
    assert "actor_entropy_tokens" not in rows[0]["actor_entropy"]
    assert rows[0]["path_value"] == str(tmp_path / "x")
    manifest_rows = [json.loads(line) for line in result.manifest_path.read_text(encoding="utf-8").splitlines()]
    assert manifest_rows[-1]["row_count"] == 1
    assert manifest_rows[-1]["phase"] == "train"
    assert manifest_rows[-1]["compression"] == "gzip"


@pytest.mark.unit
def test_reference_trace_keeps_prompts_and_tokens_without_integrity_hashes(monkeypatch, tmp_path: Path):
    args = make_args(
        tmp_path,
        agent_task_trace_enabled=True,
        agent_task_trace_phases="eval",
        agent_task_trace_include_prompts=True,
        agent_task_trace_dir=str(tmp_path / "traces"),
        agent_task_prompt_hash_enabled=True,
    )
    config = get_agent_trace_config(args, task="alfworld", sample_log_dir=args.alfworld_sample_log_dir)
    messages = [{"role": "user", "content": "Your task is to put the apple away."}]
    sample = Sample(
        index=0, rollout_id=0, response="<action>open fridge</action>", response_length=2,
        reward=0.0, loss_mask=[1, 1], rollout_log_probs=[-0.1, -0.2],
        status=Sample.Status.COMPLETED,
        metadata={"traj_uid": "reference-0", "turn_idx": 0, "seed": 1314159,
                  "prompt_tokens": 3, "response_tokens": 2, "split": "eval_out_of_distribution"},
    )

    def forbid_hash(*_args, **_kwargs):
        raise AssertionError("trace serialization must not calculate a content hash")

    clear_trace_records()
    monkeypatch.setattr(hashlib, "sha256", forbid_hash)
    metadata = compact_prompt_metadata(
        args, raw_prompt="raw task prompt", rendered_prompt="rendered task prompt", messages=messages,
    )
    assert metadata["raw_prompt"] == "raw task prompt"
    assert metadata["rendered_prompt"] == "rendered task prompt"
    assert metadata["messages"] == messages
    assert metadata["prompt_text_preview"] == "rendered task prompt"
    assert "prompt_hash" not in metadata and "raw_prompt_hash" not in metadata
    sample.metadata.update(metadata)
    record = add_sample_trace_record(
        config=config, task="alfworld", phase="eval", sample=sample,
        response_token_ids=[17, 18], rollout_log_probs=[-0.1, -0.2],
        prompt_text="rendered task prompt", messages=messages,
        extra={"prompt_ids": [11, 12, 13], "task_description": "put the apple away"},
    )
    assert record is not None
    selected = drain_trace_records_for_samples("alfworld", "eval", [sample])
    result = write_trace_sidecar(
        config, task="alfworld", phase="eval", outer_rollout_id=0, records=selected,
    )
    assert result is not None
    with gzip.open(result.path, "rt", encoding="utf-8") as reader:
        rows = [json.loads(line) for line in reader if line.strip()]
    assert len(rows) == 1
    assert rows[0]["raw_prompt"] == "rendered task prompt"
    assert rows[0]["messages"] == messages
    assert rows[0]["prompt_ids"] == [11, 12, 13]
    assert rows[0]["response_token_ids"] == [17, 18]
    assert rows[0]["rollout_log_probs"] == [-0.1, -0.2]
    assert rows[0]["prompt_token_count"] == 3
    assert "prompt_hash" not in rows[0] and "raw_prompt_hash" not in rows[0]
    clear_trace_records()


@pytest.mark.unit
def test_agent_trace_store_drains_by_phase_and_sample_key(tmp_path: Path):
    clear_trace_records()
    sample = Sample(
        index=42,
        rollout_id=99,
        response="ok",
        response_length=1,
        reward=1.0,
        loss_mask=[1],
        rollout_log_probs=[-0.1],
        status=Sample.Status.COMPLETED,
        metadata={"traj_uid": "traj-1", "turn_idx": 0},
    )
    add_trace_record(
        "alfworld",
        {"phase": "train", "sample_rollout_id": 99, "traj_uid": "traj-1", "turn_idx": "0", "sample_index": "42"},
    )
    add_trace_record(
        "alfworld",
        {"phase": "eval", "sample_rollout_id": 99, "traj_uid": "traj-1", "turn_idx": 0, "sample_index": 42},
    )
    add_trace_record(
        "alfworld",
        {"phase": "train", "sample_rollout_id": 99, "traj_uid": "other", "turn_idx": 0, "sample_index": 42},
    )

    selected = drain_trace_records_for_samples("alfworld", "train", [sample])
    remaining_eval = drain_trace_records_for_samples("alfworld", "eval", [sample])
    remaining_train = drain_trace_records_for_samples("alfworld", "train", [sample])

    assert len(selected) == 1
    assert selected[0]["phase"] == "train"
    assert len(remaining_eval) == 1
    assert remaining_train == []
    clear_trace_records()


@pytest.mark.unit
def test_rollout_entropy_postprocess_patches_trace_and_logs_diversity(monkeypatch, tmp_path: Path):
    args = make_args(
        tmp_path,
        agent_task_trace_enabled=True,
        agent_task_trace_dir=str(tmp_path / "traces"),
        agent_task_diversity_entropy_enabled=True,
        agent_task_diversity_entropy_required=True,
        keep_old_actor=True,
    )
    config = get_agent_trace_config(args, task="alfworld", sample_log_dir=args.alfworld_sample_log_dir)
    record = build_trace_record(
        task="alfworld",
        phase="train",
        metadata={
            "uid": "uid-1",
            "traj_uid": "traj-1",
            "turn_idx": 0,
            "rollout_id": 99,
            "sample_rollout_id": 99,
            "sample_index": 42,
            "agent_task": "alfworld",
        },
        response_text="ok",
        response_token_ids=[101, 102],
        rollout_log_probs=[-0.1, -0.2],
    )
    write_trace_sidecar(config, task="alfworld", phase="train", outer_rollout_id=3, records=[record])
    logged = {}

    def fake_log(_args, metrics, step_key=None):
        logged.update(metrics)
        logged["step_key"] = step_key

    monkeypatch.setattr("slime_plugins.agent_tasks.common.entropy_postprocess.logging_utils.log", fake_log)
    rollout_data = {
        "actor_entropy": [[0.4, 0.6]],
        "loss_masks": [[1, 1]],
        "response_lengths": [2],
        "metadata": [
            {
                "sample_rollout_id": 99,
                "traj_uid": "traj-1",
                "turn_idx": 0,
                "sample_index": 42,
                "agent_task": "alfworld",
                "agent_task_trace_dir": str(tmp_path / "traces"),
            }
        ],
    }

    postprocess_rollout_entropy(args, 3, rollout_data)

    assert "actor_entropy" not in rollout_data
    assert logged["diversity/rollout/alfworld/actor_entropy/coverage"] == 1.0
    assert logged["diversity/rollout/alfworld/actor_entropy/token_mean"] == pytest.approx(0.5)
    assert logged["step_key"] == "rollout/step"
    with gzip.open(tmp_path / "traces" / "alfworld" / "rollout_3.jsonl.gz", "rt", encoding="utf-8") as reader:
        rows = [json.loads(line) for line in reader]
    assert rows[0]["actor_entropy"]["enabled"] is True
    assert rows[0]["actor_entropy"]["source"] == "old_actor_forward"
    assert rows[0]["actor_entropy"]["token_count"] == 2
    assert rows[0]["actor_entropy"]["token_mean"] == pytest.approx(0.5)


@pytest.mark.unit
def test_rollout_entropy_postprocess_low_coverage_is_best_effort(monkeypatch, tmp_path: Path):
    args = make_args(
        tmp_path,
        agent_task_trace_enabled=True,
        agent_task_trace_dir=str(tmp_path / "traces"),
        agent_task_diversity_entropy_enabled=True,
        agent_task_diversity_entropy_required=True,
        agent_task_diversity_entropy_coverage_threshold=0.9,
    )
    config = get_agent_trace_config(args, task="alfworld", sample_log_dir=args.alfworld_sample_log_dir)
    record = build_trace_record(
        task="alfworld",
        phase="train",
        metadata={
            "uid": "uid-1",
            "traj_uid": "traj-1",
            "turn_idx": 0,
            "rollout_id": 99,
            "sample_rollout_id": 99,
            "sample_index": 42,
            "agent_task": "alfworld",
        },
        response_text="ok",
        response_token_ids=[101, 102],
        rollout_log_probs=[-0.1, -0.2],
    )
    write_trace_sidecar(config, task="alfworld", phase="train", outer_rollout_id=3, records=[record])
    logged = {}

    def fake_log(_args, metrics, step_key=None):
        logged.update(metrics)
        logged["step_key"] = step_key

    monkeypatch.setattr("slime_plugins.agent_tasks.common.entropy_postprocess.logging_utils.log", fake_log)
    rollout_data = {
        "actor_entropy": [[0.4, 0.6]],
        "loss_masks": [[1, 1]],
        "response_lengths": [2, 2],
        "metadata": [
            {
                "sample_rollout_id": 99,
                "traj_uid": "traj-1",
                "turn_idx": 0,
                "sample_index": 42,
                "agent_task": "alfworld",
                "agent_task_trace_dir": str(tmp_path / "traces"),
            },
            {
                "sample_rollout_id": 100,
                "traj_uid": "traj-2",
                "turn_idx": 0,
                "sample_index": 43,
                "agent_task": "alfworld",
                "agent_task_trace_dir": str(tmp_path / "traces"),
            },
        ],
    }

    postprocess_rollout_entropy(args, 3, rollout_data)

    assert "actor_entropy" not in rollout_data
    assert logged["diversity/rollout/alfworld/actor_entropy/coverage"] == 0.5
    assert logged["diversity/rollout/alfworld/actor_entropy/token_mean"] == pytest.approx(0.5)
    assert logged["step_key"] == "rollout/step"


@pytest.mark.unit
def test_agent_trace_extra_cannot_override_schema_or_prompt_when_disabled(tmp_path: Path):
    with pytest.raises(ValueError, match="protected keys"):
        build_trace_record(
            task="alfworld",
            phase="train",
            metadata={"traj_uid": "traj-1", "turn_idx": 0, "sample_index": 1},
            response_text="ok",
            response_token_ids=[1],
            rollout_log_probs=[-0.1],
            prompt_text="hidden prompt",
            include_prompts=False,
            extra={"schema_name": "bad", "raw_prompt": "leak"},
        )


@pytest.mark.unit
def test_alfworld_projection_matches_common_tagged_action_projection():
    response = "<thinking>ok</thinking><action>Open Fridge</action>"

    assert project_response(response) == project_tagged_action_response(response)
    assert alfworld_projection.ACTION_OPEN == "<action>"
    assert alfworld_projection.ACTION_CLOSE == "</action>"
    assert alfworld_projection.CHINESE_RE.search("打开") is not None


@pytest.mark.unit
def test_missing_cache_error_points_to_repo_cache(tmp_path: Path):
    with pytest.raises(RuntimeError, match="alfworld-download -f"):
        ensure_alfworld_cache(tmp_path / "missing", split="train")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
