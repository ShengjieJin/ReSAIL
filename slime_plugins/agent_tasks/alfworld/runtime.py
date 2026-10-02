from __future__ import annotations

from typing import Any

from slime_plugins.agent_tasks.common.runtime_profiles import RuntimeContract


_LIVE_ENV_KEYS = {
    "alfworld_env_pool_size",
    "alfworld_effective_env_concurrency",
    "alfworld_eval_concurrency",
    "alfworld_env_prewarm_batch_size",
    "alfworld_prewarm_env_pool",
    "alfworld_use_process_env_pool",
}


def apply_alfworld_runtime(config: dict[str, Any], contract: RuntimeContract) -> dict[str, Any]:
    custom = config["custom_config"]
    if not contract.requires_live_environment:
        for key in _LIVE_ENV_KEYS:
            custom.pop(key, None)
        custom.update(
            {
                "alfworld_environment_execution_mode": "disabled",
                "alfworld_environment_runtime_audit": "fail_closed",
            }
        )
        return config

    if contract.execution_mode not in {"live_collection", "live_eval"}:
        raise ValueError("ALFWorld live environment requires collection or eval execution mode")
    custom.update(
        {
            "alfworld_environment_execution_mode": contract.execution_mode,
            "alfworld_environment_runtime_audit": "fail_closed",
            # The generic contract retains the capacity for audit, while the
            # task adapter materializes only the workers this phase can use.
            "alfworld_env_pool_size": contract.effective_environment_concurrency,
            "alfworld_effective_env_concurrency": contract.effective_environment_concurrency,
            "alfworld_use_process_env_pool": True,
            "alfworld_prewarm_env_pool": True,
            "alfworld_env_prewarm_batch_size": contract.prewarm_batch_size,
        }
    )
    if contract.execution_mode == "live_eval":
        custom["alfworld_eval_concurrency"] = contract.effective_environment_concurrency
    else:
        custom.pop("alfworld_eval_concurrency", None)
    return config


def assert_alfworld_live_environment_allowed(args: Any) -> None:
    if str(getattr(args, "runtime_profile", "")) != "safe-fast":
        return
    mode = str(getattr(args, "execution_mode", ""))
    if not bool(getattr(args, "requires_live_environment", False)) or mode not in {
        "live_collection",
        "live_eval",
    }:
        raise RuntimeError("safe-fast refused ALFWorld environment initialization outside a live phase")


def assert_alfworld_offline_environment_disabled(args: Any) -> None:
    if str(getattr(args, "runtime_profile", "")) != "safe-fast":
        return
    if str(getattr(args, "execution_mode", "")) != "offline_training":
        raise RuntimeError("safe-fast frozen ALFWorld generation requires offline_training mode")
    if bool(getattr(args, "requires_live_environment", True)):
        raise RuntimeError("safe-fast offline ALFWorld training must disable live environments")
    if getattr(args, "alfworld_env_pool_size", None) not in (None, 0):
        raise RuntimeError("safe-fast offline ALFWorld training must not configure an environment pool")
