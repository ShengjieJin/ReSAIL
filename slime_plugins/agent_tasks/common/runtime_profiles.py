from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


ExecutionMode = Literal[
    "offline_training",
    "live_collection",
    "live_eval",
    "model_materialization",
]

SAFE_FAST_PROFILE = "safe-fast"
SAFE_FAST_MODEL_REQUEST_CONCURRENCY = 256
SAFE_FAST_LIVE_ENV_CAPACITY = 128
SAFE_FAST_OFFLINE_TRAJECTORY_CAP = 32
SAFE_FAST_PREWARM_BATCH_SIZE = 16


@dataclass(frozen=True)
class RuntimeContract:
    runtime_profile: str
    execution_mode: ExecutionMode
    requires_live_environment: bool
    model_request_concurrency: int
    server_concurrency_per_engine: int
    live_environment_capacity: int | None
    effective_environment_concurrency: int | None
    trajectory_max_inflight: int | None
    prewarm: bool
    prewarm_batch_size: int | None


def resolve_safe_fast(
    *,
    execution_mode: ExecutionMode,
    workload_size: int | None = None,
    training_batch_trajectories: int | None = None,
) -> RuntimeContract:
    live = execution_mode in {"live_collection", "live_eval"}
    if live:
        if workload_size is None or workload_size <= 0:
            raise ValueError("live safe-fast phases require a positive workload_size")
        effective = min(SAFE_FAST_LIVE_ENV_CAPACITY, int(workload_size))
        trajectory_inflight = effective if execution_mode == "live_collection" else None
        return RuntimeContract(
            runtime_profile=SAFE_FAST_PROFILE,
            execution_mode=execution_mode,
            requires_live_environment=True,
            model_request_concurrency=SAFE_FAST_MODEL_REQUEST_CONCURRENCY,
            server_concurrency_per_engine=SAFE_FAST_MODEL_REQUEST_CONCURRENCY,
            live_environment_capacity=SAFE_FAST_LIVE_ENV_CAPACITY,
            effective_environment_concurrency=effective,
            trajectory_max_inflight=trajectory_inflight,
            prewarm=True,
            prewarm_batch_size=SAFE_FAST_PREWARM_BATCH_SIZE,
        )
    if workload_size is not None:
        raise ValueError("non-live safe-fast phases must not declare an environment workload_size")
    trajectory_inflight = None
    if execution_mode == "offline_training":
        if training_batch_trajectories is None or training_batch_trajectories <= 0:
            raise ValueError("offline safe-fast training requires a positive training_batch_trajectories")
        trajectory_inflight = min(SAFE_FAST_OFFLINE_TRAJECTORY_CAP, int(training_batch_trajectories))
    elif training_batch_trajectories is not None:
        raise ValueError("model materialization must not declare training_batch_trajectories")
    return RuntimeContract(
        runtime_profile=SAFE_FAST_PROFILE,
        execution_mode=execution_mode,
        requires_live_environment=False,
        model_request_concurrency=SAFE_FAST_MODEL_REQUEST_CONCURRENCY,
        server_concurrency_per_engine=SAFE_FAST_MODEL_REQUEST_CONCURRENCY,
        live_environment_capacity=None,
        effective_environment_concurrency=None,
        trajectory_max_inflight=trajectory_inflight,
        prewarm=False,
        prewarm_batch_size=None,
    )


def apply_safe_fast(config: dict[str, Any], contract: RuntimeContract) -> dict[str, Any]:
    if contract.runtime_profile != SAFE_FAST_PROFILE:
        raise ValueError(f"unsupported runtime profile: {contract.runtime_profile}")
    cli = config["cli_args"]
    custom = config["custom_config"]
    _set_cli(cli, "--sglang-global-request-concurrency", contract.model_request_concurrency)
    custom.update(
        {
            "runtime_profile": contract.runtime_profile,
            "execution_mode": contract.execution_mode,
            "requires_live_environment": contract.requires_live_environment,
            "sglang_server_concurrency": contract.server_concurrency_per_engine,
        }
    )
    if contract.execution_mode == "live_eval":
        custom.setdefault("reuse_eval_engine_across_replicates", True)
    if contract.trajectory_max_inflight is None:
        custom.pop("rollout_trajectory_max_inflight", None)
    else:
        custom["rollout_trajectory_max_inflight"] = contract.trajectory_max_inflight
    config["runtime_profile"] = contract.runtime_profile
    config["execution_mode"] = contract.execution_mode
    config["requires_live_environment"] = contract.requires_live_environment
    config["runtime"] = asdict(contract)
    return config


def _set_cli(cli: list[Any], option: str, value: Any) -> None:
    if option in cli:
        cli[cli.index(option) + 1] = str(value)
    else:
        cli.extend([option, str(value)])
