from __future__ import annotations

import copy
from pathlib import Path

import pytest

from slime_plugins.agent_tasks.alfworld import envs as alfworld_envs
from slime_plugins.agent_tasks.alfworld.runtime import apply_alfworld_runtime
from slime_plugins.agent_tasks.common import runtime_profiles
from slime_plugins.agent_tasks.common.eval import resolve_eval_replicate_plan


NUM_GPUS = 0


def _config() -> dict:
    return {
        "cli_args": ["--sglang-global-request-concurrency", "7", "--seed", "1234"],
        "custom_config": {
            "sglang_server_concurrency": 7,
            "loss_mask_schema": "step_v1",
            "global_denominator": "trajectory",
            "source_audit_schema": "source_v1",
        },
        "scientific": {
            "trajectory_budget": 17,
            "optimizer_budget": 5,
            "seed_binding": ["task", "turn"],
            "token_contract": "exact",
            "weight_contract": "trajectory_equal",
        },
    }


@pytest.mark.unit
def test_safe_fast_offline_changes_only_runtime_fields_and_has_no_environment():
    config = _config()
    scientific = copy.deepcopy(config["scientific"])
    contract = runtime_profiles.resolve_safe_fast(
        execution_mode="offline_training", training_batch_trajectories=11
    )
    runtime_profiles.apply_safe_fast(config, contract)
    apply_alfworld_runtime(config, contract)
    assert config["scientific"] == scientific
    assert config["cli_args"][1] == "256"
    assert config["custom_config"]["sglang_server_concurrency"] == 256
    assert config["custom_config"]["rollout_trajectory_max_inflight"] == 11
    assert "alfworld_env_pool_size" not in config["custom_config"]
    assert config["custom_config"]["alfworld_environment_execution_mode"] == "disabled"
    assert config["runtime_profile"] == "safe-fast"
    assert config["requires_live_environment"] is False


@pytest.mark.unit
@pytest.mark.parametrize("workload,expected", [(1, 1), (127, 127), (129, 128)])
def test_safe_fast_live_environment_uses_minimum_of_cap_and_workload(workload: int, expected: int):
    config = _config()
    contract = runtime_profiles.resolve_safe_fast(execution_mode="live_collection", workload_size=workload)
    runtime_profiles.apply_safe_fast(config, contract)
    apply_alfworld_runtime(config, contract)
    custom = config["custom_config"]
    assert custom["alfworld_env_pool_size"] == expected
    assert custom["alfworld_effective_env_concurrency"] == expected
    assert custom["rollout_trajectory_max_inflight"] == expected
    assert custom["alfworld_use_process_env_pool"] is True
    assert custom["alfworld_prewarm_env_pool"] is True
    assert custom["alfworld_env_prewarm_batch_size"] == 16


@pytest.mark.unit
def test_safe_fast_eval_maps_effective_eval_concurrency():
    config = _config()
    contract = runtime_profiles.resolve_safe_fast(execution_mode="live_eval", workload_size=23)
    runtime_profiles.apply_safe_fast(config, contract)
    apply_alfworld_runtime(config, contract)
    assert config["custom_config"]["alfworld_eval_concurrency"] == 23
    assert config["custom_config"]["reuse_eval_engine_across_replicates"] is True
    assert "rollout_trajectory_max_inflight" not in config["custom_config"]


@pytest.mark.unit
def test_non_eval_safe_fast_modes_do_not_gain_eval_lifecycle_defaults():
    for execution_mode, kwargs in (
        ("live_collection", {"workload_size": 3}),
        ("offline_training", {"training_batch_trajectories": 3}),
        ("model_materialization", {}),
    ):
        config = _config()
        contract = runtime_profiles.resolve_safe_fast(execution_mode=execution_mode, **kwargs)
        runtime_profiles.apply_safe_fast(config, contract)
        assert "reuse_eval_engine_across_replicates" not in config["custom_config"]


@pytest.mark.unit
def test_eval_replicate_plan_preserves_single_seed_compatibility_and_requires_explicit_reuse():
    single = resolve_eval_replicate_plan(type("Args", (), {"eval_sampling_seed": 9})())
    assert single.seeds == (9,)
    assert single.reuse_engine is False
    multiple = type(
        "Args",
        (),
        {"eval_replicate_seeds": [1, 2, 3], "reuse_eval_engine_across_replicates": False},
    )()
    with pytest.raises(ValueError, match="reuse_eval_engine"):
        resolve_eval_replicate_plan(multiple)


@pytest.mark.unit
def test_safe_fast_model_materialization_has_no_live_environment():
    contract = runtime_profiles.resolve_safe_fast(execution_mode="model_materialization")
    assert contract.model_request_concurrency == 256
    assert contract.requires_live_environment is False
    assert contract.live_environment_capacity is None
    assert contract.trajectory_max_inflight is None


@pytest.mark.unit
def test_safe_fast_live_to_offline_transition_removes_effective_environment_fields():
    config = _config()
    live = runtime_profiles.resolve_safe_fast(execution_mode="live_collection", workload_size=23)
    runtime_profiles.apply_safe_fast(config, live)
    apply_alfworld_runtime(config, live)
    offline = runtime_profiles.resolve_safe_fast(
        execution_mode="offline_training", training_batch_trajectories=32
    )
    runtime_profiles.apply_safe_fast(config, offline)
    apply_alfworld_runtime(config, offline)
    assert "alfworld_env_pool_size" not in config["custom_config"]
    assert "alfworld_effective_env_concurrency" not in config["custom_config"]


@pytest.mark.unit
def test_safe_fast_is_task_agnostic_and_textcraft_gets_no_alfworld_defaults():
    config = _config()
    contract = runtime_profiles.resolve_safe_fast(
        execution_mode="offline_training", training_batch_trajectories=7
    )
    runtime_profiles.apply_safe_fast(config, contract)
    assert not [key for key in config["custom_config"] if key.startswith("alfworld_")]
    source = Path(runtime_profiles.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "trajectory_count",
        "optimizer_updates",
        "eval_episode",
        "replicate_seed",
        "alfworld_env",
        "textcraft",
    ):
        assert forbidden not in source


@pytest.mark.unit
def test_safe_fast_offline_environment_construction_fails_closed(monkeypatch):
    monkeypatch.setenv("SLIME_RUNTIME_PROFILE", "safe-fast")
    monkeypatch.setenv("SLIME_EXECUTION_MODE", "offline_training")
    monkeypatch.setenv("SLIME_REQUIRES_LIVE_ENVIRONMENT", "false")
    with pytest.raises(RuntimeError, match="outside a live phase"):
        alfworld_envs.AlfWorldEnvPool(
            pool_size=1,
            split="train",
            cache_dir=None,
            max_episode_steps=1,
        )
