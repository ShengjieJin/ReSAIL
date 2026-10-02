from __future__ import annotations

import copy
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from slime.utils.types import Sample

NUM_GPUS = 0


SDPO_SPLIT_KEYS = (
    "sdpo_metadata",
    "sdpo_teacher_prompt_text",
    "sdpo_teacher_prompt_texts",
    "sdpo_teacher_messages",
    "sdpo_teacher_messages_list",
    "sdpo_teacher_signal_type",
    "self_distillation_mask",
    "sdpo_loss_weights",
    "sdpo_representation_success_aggregation",
    "sdpo_representation_success_count",
)

SGS_SPLIT_KEYS = (
    "sgs_plain_prompt_text",
    "sgs_plain_messages",
    "sgs_action_token_mask",
    "sgs_action_alignment_valid",
    "sgs_action_alignment_reason",
    "sgs_action_match",
    "sgs_online_action",
    "sgs_online_format_valid",
    "sgs_frozen_action",
    "sgs_source_draw_id",
    "sgs_task_id",
    "sgs_split",
)

GRPO_TOKEN_WEIGHT_CONTEXT_KEYS = (
    "grpo_token_weight_positive_prompt_text",
    "grpo_token_weight_positive_messages",
    "grpo_token_weight_contrast_prompt_text",
    "grpo_token_weight_contrast_messages",
    "grpo_token_weight_contrast_source",
    "grpo_token_weight_uid",
)


def _install_rollout_import_stubs(monkeypatch) -> None:
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

    ray_module.remote = remote
    ray_module.put = lambda value: value
    ray_module.get = lambda value: value
    ray_util_module.scheduling_strategies = scheduling_module
    scheduling_module.PlacementGroupSchedulingStrategy = PlacementGroupSchedulingStrategy
    ray_module.util = ray_util_module
    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util_module)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", scheduling_module)

    for module_name in list(sys.modules):
        if module_name == "sglang" or module_name.startswith("sglang."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

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
    router_module = ModuleType("sglang_router")
    router_module.__version__ = "0.3.2"
    monkeypatch.setitem(sys.modules, "sglang", sglang_module)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.constants", constants_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils_module)
    monkeypatch.setitem(sys.modules, "sglang_router", router_module)
    monkeypatch.delitem(sys.modules, "slime.ray.rollout", raising=False)


def _manager(monkeypatch):
    _install_rollout_import_stubs(monkeypatch)
    rollout_module = importlib.import_module("slime.ray.rollout")
    manager = object.__new__(rollout_module.RolloutManager)
    manager.args = SimpleNamespace(
        global_batch_size=4,
        micro_batch_size=1,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        balance_data=False,
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
        n_samples_per_prompt=1,
        rollout_batch_size=4,
        grpo_std_normalization=True,
    )
    manager.custom_reward_post_process_func = None
    manager.custom_convert_samples_to_train_data_func = None
    manager.train_parallel_config = {
        "dp_size": 2,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    return manager


def _sdpo_metadata(index: int) -> dict:
    return {
        "uid": f"uid-{index}",
        "traj_uid": f"traj-{index}",
        "turn_idx": 0,
        "task_text": f"task {index}",
        "anchor_obs": f"obs {index}",
        "next_anchor_obs": f"next {index}",
        "projected_action": "look",
        "is_action_valid": True,
        "is_terminal": True,
        "episode_rewards": 1.0 if index % 2 == 0 else 0.0,
        "episode_lengths": 1,
        "sdpo_current_prompt_text": f"prompt {index}",
        "sdpo_current_raw_prompt": [{"role": "user", "content": f"prompt {index}"}],
    }


def _sample(index: int) -> Sample:
    sample = Sample(
        index=index,
        rollout_id=index,
        tokens=[index, 100 + index],
        response_length=1,
        reward=float(index % 2),
        loss_mask=[1],
        rollout_log_probs=[-0.1 * (index + 1)],
        train_metadata={
            "uid": f"uid-{index}",
            "traj_uid": f"traj-{index}",
            "turn_idx": 0,
            "agent_task": "alfworld",
            "sample_rollout_id": index,
            "sample_index": index,
            "sdpo": _sdpo_metadata(index),
        },
        metadata={"raw_reward": float(index % 2)},
    )
    sample.status = Sample.Status.COMPLETED
    return sample


def _train_data() -> dict:
    return {
        "tokens": [[0, 10, 100], [1, 11], [2, 12, 102, 202], [3, 13]],
        "response_lengths": [1, 1, 2, 1],
        "rewards": [1.0, 0.0, 1.0, 0.0],
        "raw_reward": [1.0, 0.0, 1.0, 0.0],
        "truncated": [0, 0, 0, 0],
        "sample_indices": [0, 1, 2, 3],
        "rollout_ids": [0, 1, 2, 3],
        "rollout_mask_sums": [1, 1, 2, 1],
        "loss_masks": [[1], [1], [1, 1], [1]],
        "sdpo_metadata": [
            {"uid": "shared", "traj_uid": "success", "episode_rewards": 1.0},
            {"uid": "shared", "traj_uid": "failure", "episode_rewards": 0.0},
            {"uid": "other", "traj_uid": "success", "episode_rewards": 1.0},
            {"uid": "other", "traj_uid": "failure", "episode_rewards": 0.0},
        ],
        "sdpo_teacher_prompt_text": ["teacher-0", "teacher-1", "teacher-2", "teacher-3"],
        "sdpo_teacher_prompt_texts": [None, ["teacher-1a", "teacher-1b"], None, ["teacher-3a"]],
        "sdpo_teacher_messages": [
            [{"role": "user", "content": "teacher-0"}],
            [{"role": "user", "content": "teacher-1"}],
            [{"role": "user", "content": "teacher-2"}],
            [{"role": "user", "content": "teacher-3"}],
        ],
        "sdpo_teacher_messages_list": [
            None,
            [[{"role": "user", "content": "teacher-1a"}], [{"role": "user", "content": "teacher-1b"}]],
            None,
            [[{"role": "user", "content": "teacher-3a"}]],
        ],
        "sdpo_teacher_signal_type": ["self_success", "success_trajectory", "self_success", "success_trajectory"],
        "self_distillation_mask": [0.0, 1.0, 0.0, 1.0],
        "sdpo_loss_weights": [0.0, 1.0, 0.0, 1.0],
        "sdpo_representation_success_aggregation": ["sample", "mean", "sample", "mean"],
        "sdpo_representation_success_count": [0, 2, 0, 1],
        "sdpo_selected_success_traj_uid": ["success", "success", "success", "success"],
        "sdpo_teacher_representations": [
            [[0.0, 0.1]],
            [[1.0, 1.1]],
            [[2.0, 2.1], [2.2, 2.3]],
            [[3.0, 3.1]],
        ],
        "self_distillation/sample_count": [4.0, 4.0, 4.0, 4.0],
        "self_distillation/mask_fraction": [0.5, 0.5, 0.5, 0.5],
    }


@pytest.mark.unit
def test_set_train_parallel_config_exposes_config_to_custom_converter(monkeypatch):
    manager = _manager(monkeypatch)
    config = {
        "dp_size": 4,
        "cp_size": 1,
        "vpp_size": 2,
        "microbatch_group_size_per_vp_stage": 2,
    }

    manager.set_train_parallel_config(config)

    assert manager.train_parallel_config is config
    assert manager.args.train_parallel_config is config


@pytest.mark.unit
def test_default_converter_promotes_sdpo_train_metadata_before_split(monkeypatch):
    manager = _manager(monkeypatch)
    samples = [_sample(index) for index in range(4)]

    train_data = manager._convert_samples_to_train_data(samples)

    for key in SDPO_SPLIT_KEYS:
        assert key in train_data
        assert len(train_data[key]) == len(samples)
    assert train_data["sdpo_metadata"] == [_sdpo_metadata(index) for index in range(4)]
    assert train_data["sdpo_teacher_prompt_text"] == [f"prompt {index}" for index in range(4)]
    assert train_data["sdpo_teacher_messages"] == [
        [{"role": "user", "content": f"prompt {index}"}] for index in range(4)
    ]
    assert train_data["sdpo_teacher_signal_type"] == ["none"] * len(samples)
    assert train_data["self_distillation_mask"] == [0.0] * len(samples)
    assert train_data["sdpo_loss_weights"] == [0.0] * len(samples)
    assert train_data["sdpo_teacher_prompt_texts"] == [None] * len(samples)
    assert train_data["sdpo_teacher_messages_list"] == [None] * len(samples)
    assert train_data["sdpo_representation_success_aggregation"] == ["sample"] * len(samples)
    assert train_data["sdpo_representation_success_count"] == [0] * len(samples)
    assert train_data["metadata"][0]["agent_task"] == "alfworld"
    assert train_data["metadata"][0]["sdpo"]["uid"] == "uid-0"

    boxes = manager._split_train_data_by_dp(train_data)
    for partition in [box.inner for box in boxes]:
        indices = list(partition["partition"])
        assert partition["sdpo_metadata"] == [train_data["sdpo_metadata"][index] for index in indices]




@pytest.mark.unit
def test_default_converter_rejects_mixed_sdpo_batch_when_first_sample_has_no_metadata(monkeypatch):
    manager = _manager(monkeypatch)
    first = _sample(0)
    first.train_metadata = None
    samples = [first, _sample(1)]

    with pytest.raises(ValueError, match="Mixed SDPO and non-SDPO"):
        manager._convert_samples_to_train_data(samples)


@pytest.mark.unit
def test_default_grpo_converter_only_appends_token_weight_context_fields(monkeypatch):
    manager = _manager(monkeypatch)
    samples = [_sample(0), _sample(1)]
    for sample, reward, traj_uid in zip(samples, (1.0, 0.0), ("success", "failure"), strict=True):
        sample.reward = reward
        sample.metadata["raw_reward"] = reward
        sample.train_metadata["uid"] = "shared"
        sample.train_metadata["traj_uid"] = traj_uid
        context = sample.train_metadata.pop("sdpo")
        context["uid"] = "shared"
        context["traj_uid"] = traj_uid
        context["episode_rewards"] = reward
        sample.train_metadata["grpo_token_weight_context"] = context

    manager.args.grpo_token_weights = False
    vanilla = manager._convert_samples_to_train_data(copy.deepcopy(samples))
    manager.args.grpo_token_weights = True
    weighted = manager._convert_samples_to_train_data(copy.deepcopy(samples))

    assert set(weighted) == set(vanilla) | set(GRPO_TOKEN_WEIGHT_CONTEXT_KEYS)
    for key, value in vanilla.items():
        assert weighted[key] == value
    assert weighted["grpo_token_weight_uid"] == ["shared", "shared"]
    assert weighted["grpo_token_weight_contrast_source"] == ["failed", "failed"]


@pytest.mark.unit
def test_eval_entropy_converter_does_not_append_grpo_token_weight_context(monkeypatch):
    manager = _manager(monkeypatch)
    manager.args.grpo_token_weights = True

    data = manager._convert_samples_to_eval_entropy_data([_sample(0), _sample(1)])

    assert not set(GRPO_TOKEN_WEIGHT_CONTEXT_KEYS) & set(data)


@pytest.mark.unit
def test_sdpo_fields_survive_current_dp_schedule_split(monkeypatch):
    data = _train_data()
    data.update({key: [f"{key}-{index}" for index in range(4)] for key in GRPO_TOKEN_WEIGHT_CONTEXT_KEYS})
    data.update({key: [f"{key}-{index}" for index in range(4)] for key in SGS_SPLIT_KEYS})

    boxes = _manager(monkeypatch)._split_train_data_by_dp(data)
    partitions = [box.inner for box in boxes]

    seen_indices: list[int] = []
    for partition in partitions:
        indices = list(partition["partition"])
        seen_indices.extend(indices)
        for key in SDPO_SPLIT_KEYS + SGS_SPLIT_KEYS:
            assert key in partition
            assert partition[key] == [data[key][index] for index in indices]
        for key in GRPO_TOKEN_WEIGHT_CONTEXT_KEYS:
            assert partition[key] == [data[key][index] for index in indices]
        assert partition["sdpo_teacher_representations"] == [
            data["sdpo_teacher_representations"][index] for index in indices
        ]
        assert partition["sdpo_selected_success_traj_uid"] == [
            data["sdpo_selected_success_traj_uid"][index] for index in indices
        ]
        assert partition["self_distillation/sample_count"] == [4.0 for _ in indices]
        assert partition["self_distillation/mask_fraction"] == [0.5 for _ in indices]
    assert sorted(seen_indices) == list(range(len(data["tokens"])))


@pytest.mark.unit
def test_method_b_normalization_fields_survive_current_dp_schedule_split(monkeypatch):
    data = _train_data()
    fields = {
        "pr_component": [0.0, 0.0, 1.0, 1.0],
        "pr_base_weight": [0.5, 0.5, 1.5, 1.5],
        "pr_normalization_scale": [3.0, 3.0, 1.5, 1.5],
    }
    data.update(fields)

    partitions = [box.inner for box in _manager(monkeypatch)._split_train_data_by_dp(data)]
    seen_indices: list[int] = []
    for partition in partitions:
        indices = list(partition["partition"])
        seen_indices.extend(indices)
        for key in fields:
            assert partition[key] == [data[key][index] for index in indices]
    assert sorted(seen_indices) == list(range(len(data["tokens"])))


@pytest.mark.unit
def test_sdpo_fields_survive_eval_entropy_dp_split(monkeypatch):
    data = _train_data()
    data.update({key: [f"{key}-{index}" for index in range(4)] for key in SGS_SPLIT_KEYS})

    boxes = _manager(monkeypatch)._split_eval_entropy_data_by_dp(data)
    partitions = [box.inner for box in boxes]

    seen_indices: list[int] = []
    for partition in partitions:
        indices = list(partition["partition"])
        seen_indices.extend(indices)
        for key in SDPO_SPLIT_KEYS + SGS_SPLIT_KEYS:
            assert key in partition
            assert partition[key] == [data[key][index] for index in indices]
        assert partition["sdpo_teacher_representations"] == [
            data["sdpo_teacher_representations"][index] for index in indices
        ]
        assert partition["self_distillation/sample_count"] == [4.0 for _ in indices]
        assert partition["self_distillation/mask_fraction"] == [0.5 for _ in indices]
    assert sorted(seen_indices) == list(range(len(data["tokens"])))


@pytest.mark.unit
def test_sdpo_teacher_log_probs_are_separate_from_opd_teacher_log_probs(monkeypatch):
    data = _train_data()
    data["sdpo_teacher_log_probs"] = [[-0.1], [-0.2], [-0.3, -0.4], [-0.5]]
    data["teacher_log_probs"] = [[-1.1], [-1.2], [-1.3, -1.4], [-1.5]]

    boxes = _manager(monkeypatch)._split_train_data_by_dp(data)

    for partition in [box.inner for box in boxes]:
        indices = list(partition["partition"])
        assert partition["sdpo_teacher_log_probs"] == [data["sdpo_teacher_log_probs"][index] for index in indices]
        assert partition["teacher_log_probs"] == [data["teacher_log_probs"][index] for index in indices]


@pytest.mark.unit
def test_sgs_sparse_step_keeps_nominal_batch_and_precomputed_targets(monkeypatch):
    manager = _manager(monkeypatch)
    manager.args.use_dynamic_batch_size = True
    manager.args.max_tokens_per_gpu = 100
    data = _train_data()
    for key, values in list(data.items()):
        if isinstance(values, list) and len(values) == 4:
            data[key] = values[:3]
    data["rollout_ids"] = [0, 0, 2]
    data["sgs_sparse_single_step"] = True
    data["sdpo_topk_indices"] = ["support-0", "support-1", "support-2"]
    data["sdpo_teacher_topk_log_probs"] = ["target-0", "target-1", "target-2"]

    boxes = manager._split_train_data_by_dp(data)
    partitions = [box.inner for box in boxes]

    assert all(partition["global_batch_sizes"] == [4] for partition in partitions)
    assert all(partition["num_microbatches"] == [1] for partition in partitions)
    seen = []
    for partition in partitions:
        indices = partition["partition"]
        seen.extend(indices)
        assert partition["sdpo_topk_indices"] == [data["sdpo_topk_indices"][index] for index in indices]
        assert partition["sdpo_teacher_topk_log_probs"] == [
            data["sdpo_teacher_topk_log_probs"][index] for index in indices
        ]
    assert sorted(seen) == [0, 1, 2]


@pytest.mark.unit
def test_eval_entropy_conversion_ignores_train_only_sdpo_converter(monkeypatch):
    manager = _manager(monkeypatch)
    manager.custom_convert_samples_to_train_data_func = lambda _args, _samples: (_ for _ in ()).throw(
        AssertionError("eval entropy must not call train-only custom converter")
    )
    samples = [_sample(index) for index in range(2)]
    for sample in samples:
        sample.train_metadata = None

    data = manager._convert_samples_to_eval_entropy_data(samples)

    assert data["tokens"] == [[0, 100], [1, 101]]
    assert data["response_lengths"] == [1, 1]
    assert data["loss_masks"] == [[1], [1]]
    assert "sdpo_metadata" not in data


@pytest.mark.unit
def test_eval_snapshot_registration_tracks_weight_version_and_enables_recovery(monkeypatch):
    manager = _manager(monkeypatch)

    class RemoteMethod:
        def __init__(self, box):
            self.box = box

        def remote(self):
            return self.box[0]

    version_boxes = [["version-7"] for _ in range(4)]
    engines = [SimpleNamespace(get_weight_version=RemoteMethod(box)) for box in version_boxes]
    recover_calls = []

    def recover():
        recover_calls.append(True)
        version_boxes[0][0] = "stale-recovered-version"

    server = SimpleNamespace(
        engines=engines,
        engine_gpu_counts=[1, 1, 1, 1],
        engine_gpu_offsets=[0, 1, 2, 3],
        num_new_engines=0,
        recover=recover,
    )
    manager._get_updatable_server = lambda: server
    manager.rollout_engine_lock = object()
    manager.rollout_id = -1
    manager.eval_snapshot_iteration = None
    manager.eval_weight_version = None
    manager.health_monitoring_pause = lambda: None
    manager.args.debug_train_only = False
    manager.args.agent_task_strict_eval_snapshot = True

    snapshot = manager.register_eval_snapshot(19)
    assert snapshot == {"iteration": 19, "weight_version": "version-7", "engine_count": 4}
    assert manager.get_eval_snapshot() == {
        "iteration": 19,
        "weight_version": "version-7",
        "rollout_id": 19,
    }

    manager.recover_updatable_engines()
    assert recover_calls == [True]
    with pytest.raises(RuntimeError, match="diverged from snapshot"):
        type(manager).eval(manager, 19)

    # Actor weight synchronization must restore the registered version before
    # the recovered engine is eligible for this snapshot's evaluation.
    version_boxes[0][0] = "version-7"
    assert manager._updatable_engine_weight_versions() == ["version-7"] * 4


@pytest.mark.unit
def test_eval_recovery_preserves_initial_engine_discovery_for_first_weight_sync(monkeypatch):
    manager = _manager(monkeypatch)
    engines = [object() for _ in range(4)]

    class Server:
        engine_gpu_counts = [1, 1, 1, 1]
        engine_gpu_offsets = [0, 1, 2, 3]

        def __init__(self):
            self.engines = engines
            self.num_new_engines = 4

        def recover(self):
            # A health check with no dead engines invokes start_engines(),
            # which resets the server's transient discovery marker.
            self.num_new_engines = 0

    server = Server()
    manager._get_updatable_server = lambda: server
    manager.rollout_engine_lock = object()
    manager.rollout_id = -1
    manager.health_monitoring_pause = lambda: None
    manager.args.agent_task_strict_eval_snapshot = True

    discovered, lock, num_new, gpu_counts, gpu_offsets = manager.recover_updatable_engines()

    assert discovered == engines
    assert lock is manager.rollout_engine_lock
    assert num_new == 4
    assert gpu_counts == [1, 1, 1, 1]
    assert gpu_offsets == [0, 1, 2, 3]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
