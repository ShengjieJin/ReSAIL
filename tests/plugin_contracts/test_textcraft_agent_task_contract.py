from __future__ import annotations

import asyncio
import gzip
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
install_stubs(with_sglang_router=True, with_transformers=True)

NUM_GPUS = 0

from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.algorithms import sdpo as sdpo_algorithm
from slime_plugins.agent_tasks.common.algorithms.sdpo import convert_samples_to_train_data
from slime_plugins.agent_tasks.common.algorithms.sdpo_context import (
    build_textcraft_sdpo_train_metadata,
    canonicalize_textcraft_sdpo_metadata,
)
from slime_plugins.agent_tasks.common.data_source import GroupedPlaceholderDataSource
from slime_plugins.agent_tasks.common.trace import clear_trace_records
from slime_plugins.agent_tasks.textcraft import envs as textcraft_envs
from slime_plugins.agent_tasks.textcraft import eval as textcraft_eval
from slime_plugins.agent_tasks.textcraft import projection as textcraft_projection
from slime_plugins.agent_tasks.textcraft.config import get_textcraft_config, split_data_indices
from slime_plugins.agent_tasks.textcraft.data_source import TextCraftDataSource
from slime_plugins.agent_tasks.textcraft.envs import (
    SingleTextCraftEnv,
    TextCraftDependencyError,
    check_textcraft_cache,
)
from slime_plugins.agent_tasks.textcraft.eval import generate_eval_rollout
from slime_plugins.agent_tasks.textcraft.generate import generate
from slime_plugins.agent_tasks.textcraft.log import log_eval_samples, log_rollout_samples
from slime_plugins.agent_tasks.textcraft.projection import project_response
from slime_plugins.agent_tasks.textcraft.rewards import post_process_rewards
from slime_plugins.agent_tasks.textcraft.splits import data_indices_from_file, ranges_for_indices


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


class FakeResetResult:
    def __init__(self, observation: str, info: dict, worker_id: str):
        self.observation = observation
        self.info = info
        self.reset_seconds = 0.01
        self.worker_id = worker_id
        self.reused_worker = False
        self.worker_created = True


class FakeTextCraftEnv:
    def __init__(self, *, seed: int, split: str, data_idx: int, max_episode_steps: int, plan=None):
        self.seed = seed
        self.split = split
        self.data_idx = data_idx
        self.max_episode_steps = max_episode_steps
        self.plan = plan or [
            ("Got 1 lilac", 0.0, False, False),
            ("Crafted 2 minecraft:magenta_dye", 1.0, True, False),
        ]
        self.step_index = 0
        self.worker_id = f"fake-{split}-{seed}-{data_idx}"
        self.actions: list[str] = []

    def reset(self, seed=None, data_idx=None):
        if seed is not None:
            self.seed = seed
        if data_idx is not None:
            self.data_idx = data_idx
        return FakeResetResult(
            "Crafting commands:\ncraft 2 magenta dye using 1 lilac\n\nGoal: craft magenta dye.",
            {
                "goal": "minecraft:magenta_dye",
                "goal_text": "magenta dye",
                "goal_depth": 1,
                "commands_count": 1,
                "inventory_size": 0,
                "action_failed": False,
            },
            self.worker_id,
        )

    def step(self, action: str):
        self.actions.append(action)
        observation, reward, done, action_failed = self.plan[min(self.step_index, len(self.plan) - 1)]
        self.step_index += 1
        return (
            observation,
            reward,
            done,
            {
                "goal": "minecraft:magenta_dye",
                "goal_text": "magenta dye",
                "goal_depth": 1,
                "commands_count": 1,
                "inventory_size": self.step_index,
                "action_failed": action_failed,
                "done": done,
                "reward": reward,
            },
        )

    def close(self):
        pass


class FakeEnvFactory:
    def __init__(self, plan=None):
        self.plan = plan
        self.created: list[FakeTextCraftEnv] = []

    def __call__(self, *, seed: int, split: str, data_idx: int, max_episode_steps: int):
        env = FakeTextCraftEnv(
            seed=seed,
            split=split,
            data_idx=data_idx,
            max_episode_steps=max_episode_steps,
            plan=self.plan,
        )
        self.created.append(env)
        return env


class FailingStepEnv(FakeTextCraftEnv):
    def step(self, action: str):
        self.actions.append(action)
        raise RuntimeError("boom")


class FailingStepEnvFactory(FakeEnvFactory):
    def __call__(self, *, seed: int, split: str, data_idx: int, max_episode_steps: int):
        env = FailingStepEnv(seed=seed, split=split, data_idx=data_idx, max_episode_steps=max_episode_steps)
        self.created.append(env)
        return env


class FakeGenerator:
    def __init__(self, responses: list[str] | None = None, finish_reasons: list[str] | None = None):
        self.responses = responses or [
            "<thinking>need ingredient</thinking><action>get 1 lilac</action>",
            "<thinking>craft goal</thinking><action>craft 2 magenta dye using 1 lilac</action>",
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


def write_official_style_split_files(cache: Path, train_ids=(31, 32), eval_ids=(0, 1, 2)) -> None:
    train_dir = cache / "agentgym_rl_data_id" / "train"
    eval_dir = cache / "agentgym_rl_data_id" / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "textcraft_train.json").write_text(
        json.dumps([{"item_id": f"textcraft_{data_idx}"} for data_idx in train_ids]),
        encoding="utf-8",
    )
    (eval_dir / "textcraft_test.json").write_text(
        json.dumps([{"item_id": f"textcraft_{data_idx}"} for data_idx in eval_ids]),
        encoding="utf-8",
    )


def make_args(tmp_path: Path, **overrides):
    textcraft_cache_dir = tmp_path / "textcraft_cache"
    write_official_style_split_files(textcraft_cache_dir)
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
        textcraft_tokenizer=FakeTokenizer(),
        textcraft_generator=FakeGenerator(),
        textcraft_env_factory=FakeEnvFactory(),
        textcraft_max_episode_steps=30,
        textcraft_invalid_action_penalty=0.1,
        textcraft_eval_episodes=3,
        textcraft_eval_concurrency=2,
        textcraft_sample_log_dir=str(tmp_path / "samples"),
        textcraft_sample_log_limit=100,
        textcraft_cache_dir=str(textcraft_cache_dir),
        agent_task_keep_prompt_metadata=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def write_minimal_textcraft_cache(cache: Path) -> Path:
    recipe_dir = cache / "recipes"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    (recipe_dir / "magenta_dye_from_lilac.json").write_text(
        json.dumps(
            {
                "type": "minecraft:crafting_shapeless",
                "group": "magenta_dye",
                "ingredients": [{"item": "minecraft:lilac"}],
                "result": {"item": "minecraft:magenta_dye", "count": 2},
            }
        ),
        encoding="utf-8",
    )
    return cache


def write_one_input_recipe(
    recipe_dir: Path, filename: str, output_item: str, input_item: str = "minecraft:stick"
) -> None:
    (recipe_dir / filename).write_text(
        json.dumps(
            {
                "type": "minecraft:crafting_shapeless",
                "ingredients": [{"item": input_item}],
                "result": {"item": output_item, "count": 1},
            }
        ),
        encoding="utf-8",
    )


def get_default_rollout_manager_cls(monkeypatch):
    # Keep the real module in sys.modules so monkeypatch restores it after this test.
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
def test_config_defaults_repo_cache_and_generated_goal_counts(monkeypatch):
    monkeypatch.delenv("TEXTCRAFT_DATA", raising=False)
    monkeypatch.delenv("TEXTCRAFT_MAX_EPISODE_STEPS", raising=False)

    config = get_textcraft_config(SimpleNamespace())

    assert config.cache_dir == Path(".cache/textcraft")
    assert config.max_episode_steps == 30
    assert config.history_format == "inline"
    assert config.history_max_steps is None
    assert config.eval_episodes == 100
    assert config.train_task_count == 374
    assert config.eval_task_count == 100
    assert config.eval_data_idx_offset == 374
    assert config.eval_dataset_name == "textcraft_eval"
    assert config.split_source == "agentgym_rl_data_id"
    assert config.train_data_file is None
    assert config.eval_data_file is None
    assert config.max_episode_errors == 8
    assert config.max_episode_error_rate == 0.05
    assert config.history_assistant_content == "full_response"


@pytest.mark.unit
def test_config_accepts_legacy_history_length_alias(tmp_path: Path):
    config = get_textcraft_config(
        SimpleNamespace(
            textcraft_cache_dir=str(tmp_path),
            textcraft_history_format="inline",
            textcraft_history_length=2,
        )
    )

    assert config.history_format == "inline"
    assert config.history_max_steps == 2


@pytest.mark.unit
def test_config_legacy_history_length_only_keeps_inline_behavior(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("TEXTCRAFT_HISTORY_FORMAT", raising=False)
    monkeypatch.delenv("TEXTCRAFT_HISTORY_MAX_STEPS", raising=False)
    monkeypatch.setenv("TEXTCRAFT_HISTORY_LENGTH", "2")

    config = get_textcraft_config(SimpleNamespace(textcraft_cache_dir=str(tmp_path)))

    assert config.history_format == "inline"
    assert config.history_max_steps == 2


@pytest.mark.unit
def test_config_history_max_steps_wins_over_legacy_length(tmp_path: Path):
    config = get_textcraft_config(
        SimpleNamespace(
            textcraft_cache_dir=str(tmp_path),
            textcraft_history_max_steps=3,
            textcraft_history_length=2,
        )
    )

    assert config.history_format == "inline"
    assert config.history_max_steps == 3


@pytest.mark.unit
def test_official_textcraft_split_files_drive_train_and_eval_indices(tmp_path: Path):
    cache = tmp_path / "textcraft"
    write_official_style_split_files(cache, train_ids=(31, 140, 533), eval_ids=(0, 1, 420))
    config = get_textcraft_config(SimpleNamespace(textcraft_cache_dir=str(cache)))

    assert split_data_indices(config, "train") == (31, 140, 533)
    assert split_data_indices(config, "eval") == (0, 1, 420)
    assert data_indices_from_file(cache / "agentgym_rl_data_id" / "train" / "textcraft_train.json") == (
        31,
        140,
        533,
    )
    assert ranges_for_indices((0, 1, 420)) == ["0-1", "420-420"]


@pytest.mark.unit
def test_data_idx_goal_tie_order_matches_upstream_listdir_depth_stability(monkeypatch, tmp_path: Path):
    cache = tmp_path / "textcraft"
    recipe_dir = cache / "recipes"
    recipe_dir.mkdir(parents=True)
    write_one_input_recipe(recipe_dir, "a_recipe.json", "minecraft:a_item")
    write_one_input_recipe(recipe_dir, "z_recipe.json", "minecraft:z_item")
    original_listdir = textcraft_envs.os.listdir

    def fake_listdir(directory):
        if Path(directory).resolve() == recipe_dir.resolve():
            return ["z_recipe.json", "a_recipe.json"]
        return original_listdir(directory)

    monkeypatch.setattr(textcraft_envs.os, "listdir", fake_listdir)
    textcraft_envs.load_crafting_tree.cache_clear()
    env = SingleTextCraftEnv(seed=7, split="train", cache_dir=cache, data_idx=0, max_episode_steps=3)
    try:
        reset = env.reset(seed=7, data_idx=0)
        assert reset.info["goal"] == "minecraft:z_item"
        assert "Goal: craft z item." in reset.observation

        reset = env.reset(seed=7, data_idx=1)
        assert reset.info["goal"] == "minecraft:a_item"
        assert "Goal: craft a item." in reset.observation
    finally:
        env.close()
        textcraft_envs.load_crafting_tree.cache_clear()


@pytest.mark.unit
def test_projection_extracts_and_sanitizes_textcraft_action():
    valid = project_response("<thinking>ok</thinking><action>craft 2 magenta dye using 1 lilac!!!</action>")
    uppercase = project_response("<thinking>ok</thinking><action>Craft 2 Magenta Dye using 1 Lilac</action>")
    missing = project_response("<thinking>ok</thinking>get 1 lilac")
    invalid = project_response("<thinking>ok</thinking><action>dance wildly</action>")

    assert valid.projected_action == "craft 2 magenta dye using 1 lilac"
    assert valid.action_kind == "craft"
    assert valid.format_valid is True
    assert uppercase.projected_action == "Craft 2 Magenta Dye using 1 Lilac"
    assert uppercase.format_valid is False
    assert "invalid_textcraft_action" in uppercase.invalid_reason
    assert missing.format_valid is False
    assert missing.missing_action_tag is True
    assert "missing_action_tag" in missing.invalid_reason
    assert invalid.format_valid is False
    assert "invalid_textcraft_action" in invalid.invalid_reason
    assert textcraft_projection.ACTION_OPEN == "<action>"
    assert textcraft_projection.ACTION_CLOSE == "</action>"
    assert textcraft_projection.CHINESE_RE.search("打开") is not None


@pytest.mark.unit
def test_data_source_groups_share_uid_data_idx_and_reject_buffer_recycle(tmp_path: Path):
    args = make_args(tmp_path, n_samples_per_prompt=3, textcraft_train_task_count=2)
    data_source = TextCraftDataSource(args)

    assert isinstance(data_source, GroupedPlaceholderDataSource)
    groups = data_source.get_samples(3)

    assert len(groups) == 3
    assert len(groups[0]) == 3
    assert {sample.metadata["uid"] for sample in groups[0]} == {"textcraft-train-00000000"}
    assert [sample.metadata["repeat_idx"] for sample in groups[0]] == [0, 1, 2]
    assert [sample.index for sample in groups[0] + groups[1] + groups[2]] == list(range(9))
    assert [group[0].metadata["data_idx"] for group in groups] == [31, 32, 31]
    assert [group[0].metadata["task_id"] for group in groups] == ["textcraft_31", "textcraft_32", "textcraft_31"]
    with pytest.raises(RuntimeError, match="variable step-row"):
        data_source.add_samples(groups)


@pytest.mark.unit
def test_textcraft_data_source_shuffle_uses_rollout_seed_over_official_split(tmp_path: Path):
    cache = tmp_path / "custom_textcraft_cache"
    write_official_style_split_files(cache, train_ids=(31, 32, 33, 34), eval_ids=(0, 1, 2))
    args = make_args(tmp_path, rollout_shuffle=True, textcraft_cache_dir=str(cache))

    groups = TextCraftDataSource(args).get_samples(4)

    assert [group[0].metadata["source_group_index"] for group in groups] == [3, 1, 0, 2]
    assert [group[0].metadata["data_idx"] for group in groups] == [34, 32, 31, 33]
    assert [group[0].metadata["sample_group_index"] for group in groups] == [0, 1, 2, 3]
    assert [group[0].metadata["seed"] for group in groups] == [10, 8, 7, 9]
    assert len({group[0].metadata["uid"] for group in groups}) == 4
    assert all(sample.metadata["train_shuffle"] is True for group in groups for sample in group)


@pytest.mark.unit
def test_real_textcraft_env_runs_minimal_recipe_cache(tmp_path: Path):
    cache = write_minimal_textcraft_cache(tmp_path / "textcraft")

    counts = check_textcraft_cache(cache)
    env = SingleTextCraftEnv(seed=7, split="train", cache_dir=cache, data_idx=0, max_episode_steps=3)
    try:
        reset = env.reset(seed=7, data_idx=0)
        assert counts["item_recipe_count"] == 1
        assert "Goal: craft magenta dye." in reset.observation
        assert reset.info["goal"] == "minecraft:magenta_dye"

        obs, reward, done, info = env.step("get 1 lilac")
        assert obs == "Got 1 lilac"
        assert reward == 0.0
        assert done is False
        assert info["action_failed"] is False

        obs, reward, done, info = env.step("inventory please")
        assert obs == "Inventory: [lilac] (1) "
        assert reward == 0.0
        assert done is False
        assert info["action_failed"] is False

        obs, reward, done, info = env.step("craft 2 magenta dye using 1 lilac")
        assert obs == "Crafted 2 minecraft:magenta_dye"
        assert reward == 1.0
        assert done is True
        assert info["action_failed"] is False

        reset = env.reset(seed=7, data_idx=0)
        assert reset.info["goal"] == "minecraft:magenta_dye"
        obs, reward, done, info = env.step("Get 1 lilac")
        assert obs == "Could not execute Get 1 lilac"
        assert reward == 0.0
        assert done is False
        assert info["action_failed"] is True
    finally:
        env.close()


@pytest.mark.unit
def test_custom_generate_returns_step_segments_with_shared_rollout_id(tmp_path: Path):
    args = make_args(tmp_path)
    placeholder = TextCraftDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert len(rows) == 2
    assert {row.rollout_id for row in rows} == {placeholder.index}
    assert [row.index for row in rows] == [0, 1]
    assert all(row.loss_mask == [1, 1] for row in rows)
    assert all(row.rollout_log_probs == [-0.1, -0.1] for row in rows)
    assert all(row.metadata["uid"] == "textcraft-train-00000000" for row in rows)
    assert all(row.metadata["data_idx"] == 31 for row in rows)
    assert all(row.metadata["goal"] == "minecraft:magenta_dye" for row in rows)
    assert all(row.metadata["episode_reward"] == 1.0 for row in rows)
    assert all(row.reward == 1.0 for row in rows)
    assert all(row.metadata["seed"] == 7 for row in rows)
    assert all(row.metadata["env_reset_seed"] == 42 for row in rows)
    assert "<user>" in rows[0].metadata["rendered_prompt"]
    assert "Crafting commands:" in rows[0].metadata["raw_prompt"]
    assert [message["role"] for message in rows[1].metadata["messages"]] == ["user"]
    assert "Recent history" in rows[1].metadata["messages"][0]["content"]
    assert "Action 1: 'get 1 lilac'" in rows[1].metadata["messages"][0]["content"]
    assert "Available crafting commands for this episode" in rows[1].metadata["messages"][0]["content"]
    assert "Crafting commands:" in rows[1].metadata["messages"][0]["content"]
    assert rows[1].metadata["history_assistant_content"] == "full_response"
    assert rows[-1].metadata["is_terminal"] is True
    assert rows[0].metadata["action_kind"] == "get"
    assert rows[1].metadata["action_kind"] == "craft"
    assert args.textcraft_env_factory.created[0].seed == 42
    assert args.textcraft_env_factory.created[0].actions == [
        "get 1 lilac",
        "craft 2 magenta dye using 1 lilac",
    ]


@pytest.mark.unit
def test_custom_generate_emits_textcraft_sdpo_metadata_for_training_only(tmp_path: Path):
    args = make_args(tmp_path, agent_task_sdpo_enabled=True)
    placeholder = TextCraftDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert len(rows) == 2
    sdpo_rows = [row.train_metadata["sdpo"] for row in rows]
    assert all(sdpo["sdpo_metadata_profile"] == "textcraft" for sdpo in sdpo_rows)
    assert all(sdpo["uid"] == "textcraft-train-00000000" for sdpo in sdpo_rows)
    assert all(sdpo["traj_uid"] == "textcraft-train-00000000-traj-00" for sdpo in sdpo_rows)
    assert [sdpo["turn_idx"] for sdpo in sdpo_rows] == [0, 1]
    assert "Goal: magenta dye" in sdpo_rows[0]["task_text"]
    assert "craft 2 magenta dye using 1 lilac" in sdpo_rows[0]["task_text"]
    assert "Crafting commands:" in sdpo_rows[0]["anchor_obs"]
    assert sdpo_rows[0]["next_anchor_obs"] == "Got 1 lilac"
    assert sdpo_rows[1]["next_anchor_obs"] == "Crafted 2 minecraft:magenta_dye"
    assert [sdpo["projected_action"] for sdpo in sdpo_rows] == [
        "get 1 lilac",
        "craft 2 magenta dye using 1 lilac",
    ]
    assert all(sdpo["is_action_valid"] is True for sdpo in sdpo_rows)
    assert [sdpo["is_terminal"] for sdpo in sdpo_rows] == [False, True]
    assert all(sdpo["episode_rewards"] == pytest.approx(1.0) for sdpo in sdpo_rows)
    assert all(sdpo["episode_lengths"] == 2 for sdpo in sdpo_rows)
    assert sdpo_rows[0]["sdpo_current_prompt_text"] == rows[0].metadata["messages"][-1]["content"]
    assert isinstance(sdpo_rows[0]["sdpo_current_raw_prompt"], list)

    extracted = build_textcraft_sdpo_train_metadata(rows[0])
    assert extracted["sdpo_metadata_profile"] == "textcraft"
    assert extracted["episode_lengths"] == 2
    assert canonicalize_textcraft_sdpo_metadata(extracted)["episode_rewards"] == pytest.approx(1.0)

    eval_rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}, evaluation=True))
    assert all("sdpo" not in row.train_metadata for row in eval_rows)


@pytest.mark.unit
@pytest.mark.parametrize(
    "trigger_overrides",
    [
        {"loss_type": "sdpo_loss"},
        {
            "custom_convert_samples_to_train_data_path": (
                "slime_plugins.agent_tasks.common.algorithms.sdpo.convert_samples_to_train_data"
            )
        },
    ],
)
def test_custom_generate_emits_textcraft_sdpo_metadata_for_launcher_triggers(tmp_path: Path, trigger_overrides):
    args = make_args(tmp_path, **trigger_overrides)
    placeholder = TextCraftDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert all(row.train_metadata["sdpo"]["sdpo_metadata_profile"] == "textcraft" for row in rows)


@pytest.mark.unit
def test_custom_generate_skips_textcraft_sdpo_metadata_for_debug_rollout(tmp_path: Path):
    args = make_args(tmp_path, agent_task_sdpo_enabled=True, debug_rollout_only=True)
    placeholder = TextCraftDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert all("sdpo" not in row.train_metadata for row in rows)


@pytest.mark.unit
def test_textcraft_sdpo_converter_consumes_generated_rows_with_textcraft_profile(tmp_path: Path):
    args = make_args(tmp_path, n_samples_per_prompt=2, loss_type="sdpo_loss")
    success_placeholder, failure_placeholder = TextCraftDataSource(args).get_samples(1)[0]
    success_rows = asyncio.run(generate(args, success_placeholder, {"max_new_tokens": 16}))
    failure_args = make_args(
        tmp_path,
        n_samples_per_prompt=2,
        loss_type="sdpo_loss",
        textcraft_env_factory=FakeEnvFactory(plan=[("Could not execute get 1 lilac", 0.0, True, True)]),
        textcraft_generator=FakeGenerator(
            responses=["<thinking>bad first step</thinking><action>get 1 lilac</action>"]
        ),
    )
    failure_rows = asyncio.run(generate(failure_args, failure_placeholder, {"max_new_tokens": 16}))
    captured_guidance_prompts: list[str] = []

    def guidance_generator(prompts: list[str]) -> dict[str, str]:
        captured_guidance_prompts.extend(prompts)
        return {
            prompt: (
                "<thinking>use the successful craft path.</thinking>\n\n"
                "Guidance summary:\n"
                "- Minimal plan: gather lilac first, then craft magenta dye.\n"
                "- Critical actions: use `craft 2 magenta dye using 1 lilac`."
            )
            for prompt in prompts
        }

    converter_args = make_args(
        tmp_path,
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        agent_task_sdpo_metadata_profile="textcraft",
        sdpo_success_reward_threshold=1.0,
        sdpo_dont_reprompt_on_self_success=False,
        sdpo_teacher_context_mode="original",
        sdpo_solution_context_format="guidance_plan",
        sdpo_guidance_summary_source="success_priority",
        sdpo_guidance_summary_generator=guidance_generator,
        sdpo_guidance_debug_enabled=True,
        sdpo_guidance_debug_dir=str(tmp_path / "sdpo_guidance"),
        sdpo_multi_turn_weighting="traj_equal",
        sdpo_max_demo_steps=None,
        sdpo_include_environment_feedback=True,
        sdpo_environment_feedback_only_without_solution=True,
    )

    train_data = convert_samples_to_train_data(converter_args, success_rows + failure_rows)

    assert [row["sdpo_metadata_profile"] for row in train_data["sdpo_metadata"]] == ["textcraft"] * 3
    assert train_data["sdpo_metadata"][-1]["episode_rewards"] == pytest.approx(0.0)
    assert train_data["sdpo_metadata"][-1]["is_action_valid"] is False
    assert train_data["sdpo_teacher_signal_type"][-1] == "solution_guidance"
    assert train_data["self_distillation_mask"][-1] == pytest.approx(1.0)
    assert train_data["sdpo_loss_weights"][-1] > 0.0
    assert "Guidance summary:" in train_data["sdpo_teacher_prompt_text"][-1]
    assert "use the successful craft path" not in train_data["sdpo_teacher_prompt_text"][-1]
    assert "- Objective:" not in train_data["sdpo_teacher_prompt_text"][-1]
    assert "gather lilac first" in train_data["sdpo_teacher_prompt_text"][-1]
    assert "Goal: craft magenta dye." in train_data["sdpo_teacher_prompt_text"][-1]
    assert "craft 2 magenta dye using 1 lilac" in train_data["sdpo_teacher_prompt_text"][-1]
    assert train_data["self_distillation/guidance_summary_used_fraction"] == pytest.approx([1.0, 1.0, 1.0])
    assert train_data["self_distillation/guidance_summary_schema_valid_fraction"] == pytest.approx([1.0, 1.0, 1.0])
    assert train_data["self_distillation/guidance_summary_debug_logged_count"] == pytest.approx([1.0, 1.0, 1.0])

    assert captured_guidance_prompts
    guidance_prompt = captured_guidance_prompts[0]
    assert "Output ONLY this format:" in guidance_prompt
    assert "<thinking>" in guidance_prompt
    assert "Guidance summary:" in guidance_prompt
    assert "- Objective:" not in guidance_prompt
    assert "Task snapshot:" in guidance_prompt
    assert "Successful trajectory evidence:" in guidance_prompt
    assert "Initial state: Crafting commands" not in guidance_prompt
    assert "Likely action plan:" not in guidance_prompt
    assert "Crafting commands:" not in guidance_prompt
    assert guidance_prompt.count("craft 2 magenta dye using 1 lilac") == 1

    guidance_files = sorted((tmp_path / "sdpo_guidance").glob("rollout_*.jsonl.gz"))
    assert guidance_files
    with gzip.open(guidance_files[0], "rt", encoding="utf-8") as handle:
        guidance_records = [json.loads(line) for line in handle if line.strip()]
    assert guidance_records[0]["task_profile"] == "textcraft"
    assert guidance_records[0]["prompt_kind"] == "success"
    assert "Output ONLY this format:" in guidance_records[0]["prompt_text"]
    assert "<thinking>" in guidance_records[0]["raw_output_text"]
    assert "Guidance summary:" in guidance_records[0]["clean_output_text"]
    assert "<thinking>" not in guidance_records[0]["clean_output_text"]
    assert "- Objective:" not in guidance_records[0]["clean_output_text"]
    assert guidance_records[0]["schema_valid"] is True


@pytest.mark.unit
def test_textcraft_sdpo_trajectory_demo_uses_success_reference_without_guidance_summary(tmp_path: Path):
    args = make_args(tmp_path, n_samples_per_prompt=2, loss_type="sdpo_loss")
    success_placeholder, failure_placeholder = TextCraftDataSource(args).get_samples(1)[0]
    success_rows = asyncio.run(generate(args, success_placeholder, {"max_new_tokens": 16}))
    failure_args = make_args(
        tmp_path,
        n_samples_per_prompt=2,
        loss_type="sdpo_loss",
        textcraft_env_factory=FakeEnvFactory(plan=[("Could not execute get 1 lilac", 0.0, True, True)]),
        textcraft_generator=FakeGenerator(
            responses=["<thinking>bad first step</thinking><action>get 1 lilac</action>"]
        ),
    )
    failure_rows = asyncio.run(generate(failure_args, failure_placeholder, {"max_new_tokens": 16}))

    def guidance_generator(prompts: list[str]) -> dict[str, str]:
        raise AssertionError(f"trajectory_demo must not request guidance summaries: {prompts}")

    converter_args = make_args(
        tmp_path,
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        agent_task_sdpo_metadata_profile="textcraft",
        sdpo_success_reward_threshold=1.0,
        sdpo_dont_reprompt_on_self_success=False,
        sdpo_teacher_context_mode="original",
        sdpo_solution_context_format="trajectory_demo",
        sdpo_guidance_generation_mode="disabled",
        sdpo_guidance_summary_source="success_priority",
        sdpo_guidance_summary_generator=guidance_generator,
        sdpo_guidance_debug_enabled=True,
        sdpo_guidance_debug_dir=str(tmp_path / "sdpo_guidance"),
        sdpo_multi_turn_weighting="traj_equal",
        sdpo_max_demo_steps=None,
        sdpo_include_environment_feedback=True,
        sdpo_environment_feedback_only_without_solution=True,
    )

    train_data = convert_samples_to_train_data(converter_args, success_rows + failure_rows)

    teacher_prompt = train_data["sdpo_teacher_prompt_text"][-1]
    assert train_data["sdpo_teacher_signal_type"][-1] == "solution_demo"
    assert train_data["sdpo_metadata"][-1]["episode_rewards"] == pytest.approx(0.0)
    assert train_data["sdpo_metadata"][-1]["is_action_valid"] is False
    assert train_data["self_distillation_mask"][-1] == pytest.approx(1.0)
    assert "Reference trajectory from a successful previous attempt on the same task:" in teacher_prompt
    assert "Guidance summary:" not in teacher_prompt
    assert "Failure analysis:" not in teacher_prompt
    assert "Task snapshot:" not in teacher_prompt
    assert "Successful trajectory evidence:" not in teacher_prompt
    assert "Step 1" in teacher_prompt
    assert "Action: `get 1 lilac`" in teacher_prompt
    assert "Observation: Got 1 lilac" in teacher_prompt
    assert "Action: `craft 2 magenta dye using 1 lilac`" in teacher_prompt
    assert "Observation: Crafted 2 minecraft:magenta_dye" in teacher_prompt
    assert "Goal: craft magenta dye." in teacher_prompt
    assert teacher_prompt.count("Crafting commands:") == 1
    assert teacher_prompt.count("Goal: craft magenta dye.") == 1
    assert train_data["self_distillation/trajectory_demo_used_fraction"] == pytest.approx([1.0, 1.0, 1.0])
    assert train_data["self_distillation/guidance_summary_used_fraction"] == pytest.approx([0.0, 0.0, 0.0])
    assert train_data["self_distillation/failure_summary_used_fraction"] == pytest.approx([0.0, 0.0, 0.0])
    assert train_data["self_distillation/guidance_summary_prompt_count"] == pytest.approx([0.0, 0.0, 0.0])

    guidance_files = sorted((tmp_path / "sdpo_guidance").glob("rollout_*.jsonl.gz"))
    assert not guidance_files


@pytest.mark.unit
def test_textcraft_sdpo_trajectory_demo_all_failed_uses_feedback_only_without_guidance_summary(tmp_path: Path):
    failure_args = make_args(
        tmp_path,
        n_samples_per_prompt=2,
        loss_type="sdpo_loss",
        textcraft_env_factory=FakeEnvFactory(plan=[("Could not execute get 1 lilac", 0.0, True, True)]),
        textcraft_generator=FakeGenerator(
            responses=["<thinking>bad first step</thinking><action>get 1 lilac</action>"]
        ),
    )
    failure_placeholder = TextCraftDataSource(failure_args).get_samples(1)[0][0]
    failure_rows = asyncio.run(generate(failure_args, failure_placeholder, {"max_new_tokens": 16}))

    def guidance_generator(prompts: list[str]) -> dict[str, str]:
        raise AssertionError(f"all-failed trajectory_demo must not request guidance summaries: {prompts}")

    converter_args = make_args(
        tmp_path,
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        agent_task_sdpo_metadata_profile="textcraft",
        sdpo_success_reward_threshold=1.0,
        sdpo_dont_reprompt_on_self_success=False,
        sdpo_teacher_context_mode="original",
        sdpo_solution_context_format="trajectory_demo",
        sdpo_guidance_generation_mode="disabled",
        sdpo_guidance_summary_source="success_priority",
        sdpo_guidance_summary_generator=guidance_generator,
        sdpo_guidance_debug_enabled=True,
        sdpo_guidance_debug_dir=str(tmp_path / "sdpo_guidance"),
        sdpo_multi_turn_weighting="traj_equal",
        sdpo_max_demo_steps=None,
        sdpo_include_environment_feedback=True,
        sdpo_environment_feedback_only_without_solution=True,
    )

    train_data = convert_samples_to_train_data(converter_args, failure_rows)

    teacher_prompt = train_data["sdpo_teacher_prompt_text"][0]
    assert train_data["sdpo_teacher_signal_type"] == ["feedback"]
    assert train_data["self_distillation_mask"] == pytest.approx([1.0])
    assert "Relevant environment transition:" in teacher_prompt
    assert "If you take action `get 1 lilac`, the next state is:" in teacher_prompt
    assert "Could not execute get 1 lilac" in teacher_prompt
    assert "Reference trajectory from a successful previous attempt" not in teacher_prompt
    assert "Guidance summary:" not in teacher_prompt
    assert "Failure analysis:" not in teacher_prompt
    assert train_data["self_distillation/trajectory_demo_used_fraction"] == pytest.approx([0.0])
    assert train_data["self_distillation/guidance_summary_used_fraction"] == pytest.approx([0.0])
    assert train_data["self_distillation/failure_summary_used_fraction"] == pytest.approx([0.0])
    assert train_data["self_distillation/guidance_summary_prompt_count"] == pytest.approx([0.0])

    guidance_files = sorted((tmp_path / "sdpo_guidance").glob("rollout_*.jsonl.gz"))
    assert not guidance_files


@pytest.mark.unit
def test_textcraft_sdpo_guidance_fallbacks_use_task_tokenizer_and_urls(tmp_path: Path):
    args = make_args(
        tmp_path,
        textcraft_sglang_url="http://textcraft-single",
        textcraft_sglang_urls=["http://textcraft-a/", "http://textcraft-b"],
    )

    assert sdpo_algorithm._load_task_tokenizer(args) is args.textcraft_tokenizer
    assert sdpo_algorithm._guidance_summary_endpoints(args) == [
        "http://textcraft-a",
        "http://textcraft-b",
        "http://textcraft-single",
    ]


@pytest.mark.unit
def test_generate_eval_rollout_skips_textcraft_sdpo_metadata(tmp_path: Path):
    args = make_args(tmp_path, agent_task_sdpo_enabled=True, textcraft_eval_episodes=2)

    output = generate_eval_rollout(args, rollout_id=0, data_source=None)
    step_samples = output.data["textcraft_eval"]["step_samples"]

    assert step_samples
    assert all("sdpo" not in row.train_metadata for row in step_samples)


@pytest.mark.unit
def test_chat_history_can_keep_only_recent_n_steps(tmp_path: Path):
    args = make_args(
        tmp_path,
        textcraft_history_format="chat",
        textcraft_history_max_steps=1,
        textcraft_env_factory=FakeEnvFactory(
            plan=[
                ("Got 1 lilac", 0.0, False, False),
                ("Inventory: [lilac] (1) ", 0.0, False, False),
                ("Crafted 2 minecraft:magenta_dye", 1.0, True, False),
            ]
        ),
        textcraft_generator=FakeGenerator(
            responses=[
                "<thinking>first</thinking><action>get 1 lilac</action>",
                "<thinking>second</thinking><action>inventory</action>",
                "<thinking>third</thinking><action>craft 2 magenta dye using 1 lilac</action>",
            ]
        ),
    )
    placeholder = TextCraftDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    turn_two_messages = rows[2].metadata["messages"]
    assert [message["role"] for message in turn_two_messages] == ["user", "user", "assistant", "user"]
    assert "Available crafting commands for this episode" in turn_two_messages[0]["content"]
    assert "craft 2 magenta dye using 1 lilac" in turn_two_messages[0]["content"]
    assert turn_two_messages[1]["content"] == "Got 1 lilac"
    assert "Crafting commands:" not in turn_two_messages[1]["content"]
    assert turn_two_messages[2]["content"] == "<thinking>second</thinking><action>inventory</action>"
    assert "get 1 lilac</action>" not in rows[2].metadata["raw_prompt"]


@pytest.mark.unit
def test_chat_history_can_preserve_full_assistant_response_when_configured(tmp_path: Path):
    args = make_args(tmp_path, textcraft_history_format="chat", textcraft_history_assistant_content="full_response")
    placeholder = TextCraftDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert rows[1].metadata["messages"][2]["content"] == (
        "<thinking>need ingredient</thinking><action>get 1 lilac</action>"
    )
    assert rows[1].metadata["history_assistant_content"] == "full_response"


@pytest.mark.unit
def test_chat_history_zero_steps_keeps_goal_instruction_and_current_observation(tmp_path: Path):
    args = make_args(tmp_path, textcraft_history_format="chat", textcraft_history_max_steps=0)
    placeholder = TextCraftDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert [message["role"] for message in rows[1].metadata["messages"]] == ["user", "user"]
    assert "Your goal is: magenta dye" in rows[1].metadata["messages"][0]["content"]
    assert "Available crafting commands for this episode" in rows[1].metadata["messages"][0]["content"]
    assert "craft 2 magenta dye using 1 lilac" in rows[1].metadata["messages"][0]["content"]
    assert rows[1].metadata["messages"][1]["content"] == "Got 1 lilac"
    assert "<action>get 1 lilac</action>" not in rows[1].metadata["raw_prompt"]


@pytest.mark.unit
def test_full_chat_auto_truncates_to_recent_history_when_prompt_budget_is_hit(tmp_path: Path):
    long_reason = " ".join(["very-long-reasoning"] * 120)
    args = make_args(
        tmp_path,
        textcraft_history_format="chat",
        rollout_max_prompt_len=1200,
        textcraft_history_assistant_content="full_response",
        textcraft_env_factory=FakeEnvFactory(
            plan=[
                ("Got 1 lilac", 0.0, False, False),
                ("Inventory: [lilac] (1) ", 0.0, False, False),
                ("Crafted 2 minecraft:magenta_dye", 1.0, True, False),
            ]
        ),
        textcraft_generator=FakeGenerator(
            responses=[
                f"<thinking>{long_reason}</thinking><action>get 1 lilac</action>",
                "<thinking>short</thinking><action>inventory</action>",
                "<thinking>craft</thinking><action>craft 2 magenta dye using 1 lilac</action>",
            ]
        ),
    )
    placeholder = TextCraftDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert rows[1].metadata["history_auto_truncated"] is True
    assert rows[1].metadata["history_steps_total"] == 1
    assert rows[1].metadata["history_steps_kept"] == 0
    assert rows[1].metadata["history_auto_truncated_dropped"] == 1
    assert rows[1].metadata["full_prompt_tokens_before_truncation"] > rows[1].metadata["prompt_tokens"]
    assert rows[1].metadata["prompt_overlength"] is False
    assert "Available crafting commands for this episode" in rows[1].metadata["messages"][0]["content"]
    assert "craft 2 magenta dye using 1 lilac" in rows[1].metadata["messages"][0]["content"]
    assert "very-long-reasoning" not in rows[1].metadata["raw_prompt"]


@pytest.mark.unit
def test_inline_history_format_keeps_single_user_message(tmp_path: Path):
    args = make_args(tmp_path, textcraft_history_format="inline", textcraft_history_max_steps=2)
    placeholder = TextCraftDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert [message["role"] for message in rows[1].metadata["messages"]] == ["user"]
    assert "Recent history" in rows[1].metadata["messages"][0]["content"]
    assert "Available crafting commands for this episode" in rows[1].metadata["messages"][0]["content"]
    assert "craft 2 magenta dye using 1 lilac" in rows[1].metadata["messages"][0]["content"]
    assert "Action 1: 'get 1 lilac'" in rows[1].metadata["messages"][0]["content"]


@pytest.mark.unit
def test_custom_generate_caps_non_terminal_episode_at_textcraft_horizon(tmp_path: Path):
    args = make_args(
        tmp_path,
        textcraft_max_episode_steps=2,
        textcraft_env_factory=FakeEnvFactory(
            plan=[
                ("Inventory: You are not carrying anything.", 0.0, False, False),
                ("Inventory: You are not carrying anything.", 0.0, False, False),
                ("Inventory: You are not carrying anything.", 0.0, False, False),
            ]
        ),
        textcraft_generator=FakeGenerator(
            responses=[
                "<thinking>check</thinking><action>inventory</action>",
                "<thinking>check again</thinking><action>inventory</action>",
                "<thinking>extra</thinking><action>inventory</action>",
            ]
        ),
    )
    placeholder = TextCraftDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert len(rows) == 2
    assert [row.metadata["turn_idx"] for row in rows] == [0, 1]
    assert all(row.metadata["env_horizon_reached"] is True for row in rows)
    assert rows[-1].metadata["is_terminal"] is False
    assert all(row.metadata["success"] is False for row in rows)
    assert all(row.reward == 0.0 for row in rows)
    assert args.textcraft_env_factory.created[0].actions == ["inventory", "inventory"]


@pytest.mark.unit
def test_custom_generate_recovers_env_step_error_as_terminal_error_sample(tmp_path: Path):
    env_factory = FailingStepEnvFactory()
    args = make_args(tmp_path, textcraft_env_factory=env_factory)
    placeholder = TextCraftDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert len(rows) == 1
    assert env_factory.created[0].actions == ["get 1 lilac"]
    row = rows[0]
    assert row.metadata["is_terminal"] is True
    assert row.metadata["episode_error"] == 1.0
    assert row.metadata["episode_error_count"] == 1.0
    assert row.metadata["episode_error_type"] == "RuntimeError"
    assert row.metadata["episode_error_stage"] == "env_step"
    assert "boom" in row.metadata["episode_error_message"]
    assert row.metadata["env_step_failed"] is True
    assert row.metadata["success"] is False
    assert row.metadata["episode_reward"] == 0.0
    assert row.reward == 0.0
    assert row.rollout_log_probs == [-0.1, -0.1]


@pytest.mark.unit
def test_step_segments_survive_default_converter_with_rollout_mask_sums(tmp_path: Path, monkeypatch):
    args = make_args(tmp_path)
    placeholder = TextCraftDataSource(args).get_samples(1)[0][0]
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
        ("textcraft-train-00000000", "textcraft-train-00000000-traj-00", 0),
        ("textcraft-train-00000000", "textcraft-train-00000000-traj-00", 1),
    ]
    assert all(row["agent_task"] == "textcraft" for row in train_data["metadata"])
    assert [row["task_id"] for row in train_data["metadata"]] == ["textcraft_31", "textcraft_31"]


@pytest.mark.unit
def test_custom_generate_rejects_abort_finish_reason_before_env_step(tmp_path: Path):
    env_factory = FakeEnvFactory()
    args = make_args(
        tmp_path,
        textcraft_env_factory=env_factory,
        textcraft_generator=FakeGenerator(
            responses=["<thinking>reason</thinking><action>get 1 lilac</action>"],
            finish_reasons=["abort"],
        ),
    )
    placeholder = TextCraftDataSource(args).get_samples(1)[0][0]

    with pytest.raises(RuntimeError, match="generation aborted before env step"):
        asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    assert len(env_factory.created) == 1
    assert env_factory.created[0].actions == []


@pytest.mark.unit
def test_invalid_format_penalty_is_per_step_but_raw_reward_is_terminal(tmp_path: Path):
    args = make_args(
        tmp_path,
        textcraft_generator=FakeGenerator(
            responses=[
                "<thinking>need ingredient</thinking>get 1 lilac",
                "<thinking>craft goal</thinking><action>craft 2 magenta dye using 1 lilac</action>",
            ]
        ),
    )
    placeholder = TextCraftDataSource(args).get_samples(1)[0][0]

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
    args = make_args(
        tmp_path,
        n_samples_per_prompt=8,
        textcraft_eval_episodes=3,
        textcraft_env_factory=FakeEnvFactory(plan=[("Crafted 2 minecraft:magenta_dye", 1.0, True, False)]),
        textcraft_generator=FakeGenerator(responses=["<thinking>craft</thinking><action>get 1 lilac</action>"]),
    )

    output = generate_eval_rollout(args, 4, TextCraftDataSource(args), evaluation=True)

    assert set(output.data) == {"textcraft_eval"}
    data = output.data["textcraft_eval"]
    assert len(data["samples"]) == 3
    assert len(data["step_samples"]) == 3
    assert len(data["rewards"]) == 3
    assert len(args.textcraft_env_factory.created) == 3
    assert sorted(env.data_idx for env in args.textcraft_env_factory.created) == [0, 1, 2]
    assert output.metrics == {}


@pytest.mark.unit
def test_eval_three_replicates_pair_official_tasks_and_vary_only_sampling_seed(tmp_path: Path):
    args = make_args(
        tmp_path,
        n_samples_per_prompt=8,
        textcraft_eval_episodes=3,
        textcraft_eval_replicate_seeds=[314159, 314160, 314161],
        textcraft_eval_identity_seed=314159,
        textcraft_sample_log_limit=100,
        textcraft_eval_result_audit_dir=str(tmp_path / "eval_audit"),
        textcraft_env_factory=FakeEnvFactory(plan=[("Crafted 2 minecraft:magenta_dye", 1.0, True, False)]),
        textcraft_generator=FakeGenerator(responses=["<thinking>craft</thinking><action>get 1 lilac</action>"]),
    )

    output = generate_eval_rollout(args, 4, TextCraftDataSource(args), evaluation=True)

    expected = {
        f"textcraft_eval__replicate_{idx:02d}_seed_{seed}"
        for idx, seed in enumerate((314159, 314160, 314161))
    }
    assert set(output.data) == expected
    task_ids = []
    sampling_seeds = []
    for replicate_idx, replicate_seed in enumerate((314159, 314160, 314161)):
        name = f"textcraft_eval__replicate_{replicate_idx:02d}_seed_{replicate_seed}"
        terminals = output.data[name]["samples"]
        assert len(terminals) == 3
        assert {row.metadata["eval_replicate_index"] for row in terminals} == {replicate_idx}
        assert {row.metadata["eval_replicate_seed"] for row in terminals} == {replicate_seed}
        task_ids.append([row.metadata["task_id"] for row in terminals])
        sampling_seeds.append([row.metadata["sampling_seed"] for row in terminals])
    assert task_ids[0] == task_ids[1] == task_ids[2]
    assert len({tuple(values) for values in sampling_seeds}) == 3

    assert log_eval_samples(4, args, output.data, {}) is True
    log_rows = [
        json.loads(line)
        for line in (tmp_path / "samples" / "eval_4.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(log_rows) == 9
    for row in log_rows:
        assert row["eval_base_dataset_name"] == "textcraft_eval"
        assert row["eval_replicate_index"] in {0, 1, 2}
        assert row["eval_replicate_seed"] in {314159, 314160, 314161}
        assert isinstance(row["sampling_seed"], int)
        assert isinstance(row["task_id"], str)
        assert isinstance(row["turn_idx"], int)
    audit = json.loads((tmp_path / "eval_audit" / "eval_004.json").read_text(encoding="utf-8"))
    assert set(audit["datasets"]) == expected
    assert {item["episode_count"] for item in audit["datasets"].values()} == {3}


@pytest.mark.unit
def test_frozen_collection_turn_sampling_is_stream_bound(tmp_path: Path):
    args = make_args(
        tmp_path,
        n_samples_per_prompt=1,
        rollout_seed=42,
        agent_frozen_stream_bound_sampling=True,
    )
    placeholder = TextCraftDataSource(args).get_samples(1)[0][0]

    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))

    from slime_plugins.agent_tasks.common.frozen.sampling import stream_turn_seed

    assert [row.metadata["sampling_seed"] for row in rows] == [
        stream_turn_seed(task="textcraft", cycle_seed=42, stream_index=0, turn_idx=0),
        stream_turn_seed(task="textcraft", cycle_seed=42, stream_index=0, turn_idx=1),
    ]
    assert all(row.metadata["runtime_task_identity"]["task_id"] == "textcraft_31" for row in rows)


@pytest.mark.unit
def test_eval_episodes_share_one_concurrency_queue(monkeypatch, tmp_path: Path):
    active_count = 0
    max_active_count = 0

    async def fake_run_textcraft_episode(args, *, sample, sampling_params, config, split, evaluation=False):
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

    monkeypatch.setattr(textcraft_eval, "run_textcraft_episode", fake_run_textcraft_episode)
    args = make_args(tmp_path, textcraft_eval_episodes=3, textcraft_eval_concurrency=2)

    output = generate_eval_rollout(args, 4, TextCraftDataSource(args), evaluation=True)

    assert list(output.data) == ["textcraft_eval"]
    assert len(output.data["textcraft_eval"]["samples"]) == 3
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
                    "data_idx": rollout_id,
                    "goal": "minecraft:magenta_dye",
                    "goal_text": "magenta dye",
                    "action_kind": "craft",
                    "action_failed": False,
                    "format_valid": True,
                    "missing_action_tag": False,
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
                    "raw_prompt": "Crafting commands:\ncraft 2 magenta dye using 1 lilac",
                    "rendered_prompt": "<user>Crafting commands:\ncraft 2 magenta dye using 1 lilac</user><assistant>",
                    "messages": [{"role": "user", "content": "Crafting commands:\ncraft 2 magenta dye using 1 lilac"}],
                },
            )
        )
    metrics = {}

    assert log_rollout_samples(0, args, samples, metrics, 1.0) is False

    assert metrics["rollout/textcraft/episode/count"] == 3.0
    assert metrics["rollout/textcraft/episode/success_rate"] == 1.0
    assert metrics["rollout/textcraft/action/format_valid_rate"] == 1.0
    assert metrics["rollout/textcraft/perf/prefix_cache_hit_rate"] == 0.5
    rollout_rows = [
        json.loads(line)
        for line in (tmp_path / "samples" / "rollout_0.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rollout_rows[0]["goal_text"] == "magenta dye"
    assert rollout_rows[0]["data_idx"] == 10
    assert rollout_rows[0]["model_input"] == (
        "<user>Crafting commands:\ncraft 2 magenta dye using 1 lilac</user><assistant>"
    )
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
    )
    placeholder = TextCraftDataSource(args).get_samples(1)[0][0]
    rows = asyncio.run(generate(args, placeholder, {"max_new_tokens": 16}))
    metrics = {}

    assert log_rollout_samples(2, args, [rows], metrics, 1.0) is False

    assert metrics["diversity/rollout/textcraft/sampled_nll/token_mean"] == pytest.approx(0.1)
    assert metrics["diversity/rollout/textcraft/sampled_logprob/token_mean"] == pytest.approx(-0.1)
    assert "diversity/rollout/textcraft/actor_entropy/coverage" not in metrics
    assert "rollout/diversity/rollout/textcraft/sampled_nll/token_mean" not in metrics
    with gzip.open(tmp_path / "traces" / "textcraft" / "rollout_2.jsonl.gz", "rt", encoding="utf-8") as reader:
        trace_rows = [json.loads(line) for line in reader]
    assert len(trace_rows) == len(rows)
    assert trace_rows[0]["phase"] == "train"
    assert trace_rows[0]["current_observation"].startswith("Crafting commands:")
    assert trace_rows[0]["next_observation"] == "Got 1 lilac"
    assert trace_rows[0]["goal_text"] == "magenta dye"
    assert trace_rows[0]["inventory_before"] == {}
    assert "craft 2 magenta dye using 1 lilac" in trace_rows[0]["crafting_commands"]
    assert "raw_prompt" not in trace_rows[0]
    assert "messages" not in trace_rows[0]
    assert "raw_prompt" not in rows[0].metadata
    assert "rendered_prompt" not in rows[0].metadata
    assert "messages" not in rows[0].metadata
    assert "current_observation" not in rows[0].metadata
    clear_trace_records()


@pytest.mark.unit
def test_eval_log_hook_uses_textcraft_metric_namespace(tmp_path: Path):
    args = make_args(
        tmp_path,
        use_wandb=False,
        use_tensorboard=False,
        textcraft_sample_log_limit=4,
        textcraft_eval_episodes=3,
        textcraft_env_factory=FakeEnvFactory(plan=[("Crafted 2 minecraft:magenta_dye", 1.0, True, False)]),
        textcraft_generator=FakeGenerator(responses=["<thinking>craft</thinking><action>get 1 lilac</action>"]),
        agent_task_trace_enabled=True,
        agent_task_trace_dir=str(tmp_path / "traces"),
    )
    clear_trace_records()
    output = generate_eval_rollout(args, 4, TextCraftDataSource(args), evaluation=True)
    metrics = {}

    assert log_eval_samples(4, args, output.data, metrics) is True

    assert metrics["eval/textcraft_eval/episode/count"] == 3.0
    assert metrics["eval/textcraft_eval/episode/success_rate"] == 1.0
    assert metrics["eval/textcraft_eval/error/episode_rate"] == 0.0
    assert metrics["diversity/eval/textcraft_eval/sampled_nll/token_mean"] == pytest.approx(0.1)
    assert "eval/textcraft_eval/textcraft/success" not in metrics
    eval_rows = [
        json.loads(line) for line in (tmp_path / "samples" / "eval_4.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(eval_rows) == 3
    assert {row["eval_dataset_name"] for row in eval_rows} == {"textcraft_eval"}
    assert {row["split"] for row in eval_rows} == {"eval"}
    assert all("model_input" in row for row in eval_rows)
    assert all("Crafting commands:" in row["model_input"] for row in eval_rows)
    with gzip.open(tmp_path / "traces" / "textcraft" / "eval_4.jsonl.gz", "rt", encoding="utf-8") as reader:
        trace_rows = [json.loads(line) for line in reader]
    assert len(trace_rows) == 3
    assert {row["phase"] for row in trace_rows} == {"eval"}
    assert {row["eval_dataset_name"] for row in trace_rows} == {"textcraft_eval"}
    assert trace_rows[0]["goal_text"] == "magenta dye"
    clear_trace_records()




@pytest.mark.unit
def test_missing_cache_error_points_to_repo_cache(tmp_path: Path):
    with pytest.raises(TextCraftDependencyError, match=".cache/textcraft/recipes"):
        check_textcraft_cache(tmp_path / "missing")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
