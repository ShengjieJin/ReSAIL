# Modified for ReSAIL. See NOTICE and LICENSE for attribution and terms.
import dataclasses
import hashlib
import itertools
import logging
import multiprocessing
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.backends.sglang_utils.external import start_external_rollout_servers
from slime.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig
from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.rollout.base_types import call_rollout_fn
from slime.utils import logging_utils
from slime.utils.dp_schedule import active_rollout_batch_sizes, build_dp_schedule
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, init_http_client
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step, compute_statistics, dict_add_prefix
from slime.utils.misc import Box, group_by, load_function
from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.algorithms.sgs import bind_selection_to_samples, random_rank_key, sample_key, select_sgs_steps, sgs_audit_record, sgs_filtering_enabled, sgs_option, sgs_ranking_score_field, sgs_scoring_required, sgs_selection_fraction
from slime_plugins.agent_tasks.common.artifacts import write_json_once

from ..utils.metric_utils import has_repetition
from .rollout_validation import validate_server_group_gpu_indices
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

_ROLLOUT_METADATA_OMIT_KEYS = {"sdpo_guidance_image_data", "grpo_token_weight_context"}


def _env_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _agent_task_eval_entropy_enabled(args) -> bool:
    value = getattr(args, "agent_task_diversity_eval_entropy_enabled", None)
    if value is None:
        value = os.environ.get("AGENT_TASK_DIVERSITY_EVAL_ENTROPY_ENABLED", False)
    return _env_bool(value)


def _flatten_sample_groups(samples: list[Sample] | list[list[Sample]]) -> list[Sample]:
    rows = []
    for item in samples:
        if isinstance(item, Sample):
            rows.append(item)
        elif isinstance(item, list):
            rows.extend(_flatten_sample_groups(item))
        else:
            raise TypeError(f"unexpected sample node: {type(item).__name__}")
    return rows


@dataclasses.dataclass
class ServerGroup:
    """A group of homogeneous SGLang engines with the same configuration.

    All engines in a group share the same tp_size / nodes_per_engine / pg.
    A RolloutServer may contain multiple ServerGroups (e.g. prefill vs decode
    in PD disaggregation).
    """

    args: Any
    pg: Any  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    all_engines: list
    num_gpus_per_engine: int
    num_new_engines: int
    worker_type: str = "regular"  # "regular", "prefill", "decode", or "placeholder"
    rank_offset: int = 0  # cumulative engine count before this group
    gpu_offset: int = 0  # cumulative GPU count before this group
    sglang_overrides: dict = dataclasses.field(default_factory=dict)
    needs_offload: bool = False  # True when this group's GPUs overlap with megatron
    model_path: str | None = None  # checkpoint path for update_weights_from_disk
    router_ip: str | None = None
    router_port: int | None = None

    @property
    def nodes_per_engine(self):
        return max(1, self.num_gpus_per_engine // self.args.num_gpus_per_node)

    @property
    def engines(self):
        """Node-0 engines only (for multi-node serving)."""
        return self.all_engines[:: self.nodes_per_engine]

    def start_engines(self, port_cursors: dict[int, int] | None = None) -> tuple[list, dict[int, int]]:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Returns ``(init_handles, port_cursors)`` where *init_handles* is a list
        of Ray ObjectRefs and *port_cursors* maps node index → next free port.
        The caller should ``ray.get()`` on the handles to block until the
        engines are healthy, and pass *port_cursors* to the next server group
        so that different groups on the same node don't race for ports.

        Placeholder groups (worker_type="placeholder") skip engine creation entirely.
        """
        if port_cursors is None:
            port_cursors = {}
        if self.args.debug_train_only or self.worker_type == "placeholder":
            self.num_new_engines = 0
            return [], port_cursors

        num_gpu_per_engine = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)

        pg, reordered_bundle_indices, reordered_gpu_ids = self.pg
        validate_server_group_gpu_indices(
            worker_type=self.worker_type,
            gpu_offset=self.gpu_offset,
            num_gpus_per_engine=self.num_gpus_per_engine,
            num_gpu_per_engine=num_gpu_per_engine,
            num_engines=len(self.all_engines),
            num_available_gpus=len(reordered_gpu_ids),
            rollout_num_gpus=self.args.rollout_num_gpus,
            rollout_num_gpus_per_engine=self.args.rollout_num_gpus_per_engine,
        )

        RolloutRayActor = ray.remote(SGLangEngine)

        rollout_engines = []
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue

            global_rank = self.rank_offset + i
            num_gpus = 0.2
            num_cpus = num_gpus

            # Get the base GPU ID from placement group using gpu_offset.
            gpu_index = self.gpu_offset + i * num_gpu_per_engine
            base_gpu_id = int(reordered_gpu_ids[gpu_index])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
                key: os.environ.get(key, default_val)
                for key, default_val in {
                    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "true",
                    "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "true",
                    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
                    "SLIME_ENABLE_PROFILING": "true",
                }.items()
            }
            rollout_engine = RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    "env_vars": env_vars,
                },
            ).remote(
                self.args,
                rank=global_rank,
                worker_type=self.worker_type,
                base_gpu_id=base_gpu_id,
                sglang_overrides=self.sglang_overrides,
                num_gpus_per_engine=self.num_gpus_per_engine,
            )

            rollout_engines.append((global_rank, rollout_engine))
            self.all_engines[i] = rollout_engine

        self.num_new_engines = len(rollout_engines)

        if self.num_new_engines == 0:
            return [], port_cursors

        # Compute base_port from the maximum cursor across all nodes that
        # this group's engines may land on (conservative: just use global max).
        base_port = max(port_cursors.values()) if port_cursors else 15000
        addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(
            args=self.args,
            rollout_engines=rollout_engines,
            worker_type=self.worker_type,
            num_gpus_per_engine=self.num_gpus_per_engine,
            rank_offset=self.rank_offset,
            base_port=base_port,
        )

        init_handles = [
            engine.init.remote(
                **(addr_and_ports[rank]),
                router_ip=self.router_ip,
                router_port=self.router_port,
            )
            for rank, engine in rollout_engines
        ]
        return init_handles, port_cursors

    def offload(self):
        """Fire release_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.  Skipped for groups that do not
        overlap with megatron GPUs (``needs_offload=False``).
        """
        if not self.needs_offload:
            return []
        return [engine.release_memory_occupation.remote() for engine in self.engines if engine is not None]

    def onload(self, tags: list[str] | None = None):
        """Fire resume_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.  Skipped for groups that do not
        overlap with megatron GPUs (``needs_offload=False``).
        """
        if not self.needs_offload:
            return []
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.engines if engine is not None]

    def onload_weights_from_disk(self):
        """Reload weights from ``model_path`` for non-updatable groups.

        Used instead of ``resume_memory_occupation(tags=[WEIGHTS])`` so that
        CPU memory is not consumed by offloaded weight copies.
        """
        if not self.needs_offload or not self.model_path:
            return []
        return [
            engine.update_weights_from_disk.remote(self.model_path) for engine in self.engines if engine is not None
        ]


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, with one or more server groups.

    Each RolloutServer represents one model deployed behind a single router.
    A server may contain multiple ServerGroups with different
    ``num_gpus_per_engine`` (e.g. prefill TP=2, decode TP=4).
    """

    server_groups: list[ServerGroup]
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True

    @property
    def engines(self):
        """All node-0 engines across all groups (placeholder groups contribute nothing)."""
        return [e for g in self.server_groups for e in g.engines]

    @property
    def all_engines(self):
        """All engines (including non-node-0) across all groups."""
        return [e for g in self.server_groups for e in g.all_engines]

    @property
    def num_new_engines(self):
        return sum(g.num_new_engines for g in self.server_groups)

    @num_new_engines.setter
    def num_new_engines(self, value):
        for g in self.server_groups:
            g.num_new_engines = value

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [g.num_gpus_per_engine for g in self.server_groups for _ in g.engines]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        """Per-engine GPU offset for all node-0 engines, parallel to ``engines``.

        Accounts for placeholder groups that occupy GPU slots without creating engines.
        """
        offsets = []
        for g in self.server_groups:
            for j in range(len(g.engines)):
                offsets.append(g.gpu_offset + j * g.num_gpus_per_engine)
        return offsets

    @property
    def nodes_per_engine(self):
        """Nodes per engine.  Only valid when all active groups share the same value."""
        values = {g.nodes_per_engine for g in self.server_groups if g.worker_type != "placeholder"}
        if len(values) != 1:
            raise ValueError(f"Heterogeneous nodes_per_engine across groups: {values}")
        return values.pop()

    def recover(self):
        """Recover dead engines across all active groups, overlapping init."""
        # Record dead indices per group before starting.
        dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] for g in self.server_groups]

        # Start all groups concurrently.
        all_handles = []
        port_cursors: dict[int, int] = {}
        for g in self.server_groups:
            handles, port_cursors = g.start_engines(port_cursors)
            all_handles.extend(handles)
        if all_handles:
            ray.get(all_handles)

        # Post-recovery: offload then onload weights for newly created engines.
        release_handles = []
        updatable_new_engines = []
        non_updatable_groups_engines: list[tuple[str, list]] = []
        for g, dead_indices in zip(self.server_groups, dead_per_group, strict=True):
            logger.info(f"Recovered {g.num_new_engines} dead rollout engines (worker_type={g.worker_type})")
            assert g.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
            if g.needs_offload and dead_indices:
                new_engines = [g.all_engines[i] for i in dead_indices]
                release_handles.extend(engine.release_memory_occupation.remote() for engine in new_engines)
                if self.update_weights:
                    updatable_new_engines.extend(new_engines)
                elif g.model_path:
                    non_updatable_groups_engines.append((g.model_path, new_engines))

        if release_handles:
            ray.get(release_handles)
            # Resume GPU memory for all engines that need offload.
            all_resume_engines = updatable_new_engines[:]
            for _model_path, engines in non_updatable_groups_engines:
                all_resume_engines.extend(engines)
            if all_resume_engines:
                ray.get(
                    [
                        engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                        for engine in all_resume_engines
                    ]
                )

    def offload(self):
        """Release memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.offload())
        return ray.get(handles) if handles else []

    def onload(self, tags: list[str] | None = None):
        """Resume memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags))
        return ray.get(handles) if handles else []

    def onload_weights(self):
        """Restore weights for offloaded groups.

        All groups resume from CPU cache via ``resume_memory_occupation``.
        For updatable servers, weights will be overwritten by
        ``update_weights`` shortly after.  For non-updatable servers the
        CPU backup already contains the correct (unchanged) weights.
        """
        handles = []
        for g in self.server_groups:
            if not g.needs_offload:
                continue
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
        return ray.get(handles) if handles else []

    def onload_kv(self):
        """Resume KV cache and CUDA graphs for offloaded groups."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]))
        return ray.get(handles) if handles else []


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        configure_logger()

        self.pg = pg
        self.args = args

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            self.servers: dict[str, Any] = {}
        else:
            init_http_client(args)
            self.servers = start_rollout_servers(args, pg)

        self._tracking_initialized = bool(getattr(args, "rollout_secondary_tracking", True))
        if self._tracking_initialized:
            init_tracking(args, primary=False)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self.rollout_id = -1
        self.eval_snapshot_iteration: int | None = None
        self.eval_weight_version: str | None = None
        self.training_snapshot_iteration: int | None = None
        self.training_actor_version: int | None = None
        self.training_weight_version: str | None = None
        self.generation_timings: dict[int, dict[str, float]] = {}
        self._pending_eval_data: dict[int, dict[str, Any]] = {}
        self._pending_sgs_samples: dict[int, list[Sample] | list[list[Sample]]] = {}

        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
            self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection

    def _get_metrics_router_addr(self) -> str | None:
        """Return the router address for scraping SGLang engine metrics.

        The sglang_router gateway exposes ``/engine_metrics`` on its main port,
        which aggregates Prometheus metrics from all backend sglang servers.
        Returns ``http://{ip}:{port}`` for the first server, or ``None`` when
        metrics are disabled or no servers are running.
        """
        srv = self.server
        if srv is None or srv.router_ip is None:
            return None
        return f"http://{srv.router_ip}:{srv.router_port}"

    def get_metrics_router_addr(self) -> str | None:
        """Public wrapper for remote calls from the driver process."""
        return self._get_metrics_router_addr()

    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if (
            self.server
            and self.server.server_groups
            and self.server.server_groups[0].all_engines
            and self.server.server_groups[0].all_engines[0]
        ):
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self.server.server_groups[0].all_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")

    def dispose(self):
        for monitor in self._health_monitors:
            monitor.stop()
        if self._tracking_initialized:
            logging_utils.finish_tracking(self.args)

    @property
    def server(self) -> Any | None:
        """Default server (first model).  For backward compatibility."""
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    def _get_updatable_server(self) -> Any | None:
        """Return the server with ``update_weights=True``.

        When multiple updatable servers exist, returns the first one
        (multi-model weight update is not yet supported).
        """
        for srv in self.servers.values():
            if srv.update_weights:
                return srv
        return None

    @property
    def rollout_engines(self):
        """All node-0 engines across all servers / models."""
        return [e for srv in self.servers.values() for e in srv.engines]

    def get_updatable_engines_and_lock(self):
        """Return engines eligible for weight updates.

        Returns engines from the first model that has
        ``update_weights=True``.  Frozen models (reference, reward,
        etc.) are automatically excluded.
        """
        srv = self._get_updatable_server()
        engines = srv.engines if srv else []
        gpu_counts = srv.engine_gpu_counts if srv else []
        gpu_offsets = srv.engine_gpu_offsets if srv else []
        num_new = srv.num_new_engines if srv else 0
        return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        if not self.args.offline_train_eval_colocate:
            self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
        end_time = time.time()
        self.generation_timings[int(rollout_id)] = {
            "started_at_unix_seconds": start_time,
            "completed_at_unix_seconds": end_time,
            "duration_seconds": end_time - start_time,
        }
        _log_rollout_data(rollout_id, self.args, data, metrics, end_time - start_time)
        if self.args.debug_rollout_only:
            # if debug rollout only, we don't convert samples to train data and directly return
            return
        if sgs_filtering_enabled(self.args):
            if not sgs_scoring_required(self.args):
                return self._finalize_unscored_random_topx(int(rollout_id), data)
            if int(rollout_id) in self._pending_sgs_samples:
                raise RuntimeError(f"sensitivity samples already pending for rollout {rollout_id}")
            self._pending_sgs_samples[int(rollout_id)] = data
        data = self._convert_samples_to_train_data(data)
        return self._split_train_data_by_dp(data)

    def finalize_sgs(self, rollout_id: int, rank_results: list[dict[str, Any]]):
        rollout_id = int(rollout_id)
        samples = self._pending_sgs_samples.pop(rollout_id, None)
        if samples is None:
            raise RuntimeError(f"no pending sensitivity samples for rollout {rollout_id}")
        if not rank_results:
            raise ValueError("sensitivity scoring returned no rank results")
        records = []
        durations = []
        ranks = set()
        for result in rank_results:
            if int(result.get("rollout_id", -1)) != rollout_id:
                raise ValueError("sensitivity score result rollout id mismatch")
            rank = int(result["rank"])
            if rank in ranks:
                raise ValueError(f"duplicate sensitivity score result for rank {rank}")
            ranks.add(rank)
            durations.append(float(result["duration_seconds"]))
            records.extend(result["rows"])

        flat_samples = _flatten_sample_groups(samples)
        fraction = sgs_selection_fraction(self.args)
        if fraction is None:
            raise RuntimeError("SGS finalization was called without an enabled fraction")
        dp_size = int(self.train_parallel_config["dp_size"])
        selection_mode = str(getattr(self.args, "sgs_selection_mode", "sensitivity"))
        selection_seed = int(getattr(self.args, "sgs_selection_seed", 42))
        selection_scope = str(sgs_option(self.args, "selection_scope", "global"))
        score_field = sgs_ranking_score_field(self.args)
        ranking_order = str(sgs_option(self.args, "ranking_order", "descending"))
        selection = select_sgs_steps(
            records,
            fraction=fraction,
            minimum_selected=dp_size,
            selection_mode=selection_mode,
            selection_seed=selection_seed,
            rollout_id=rollout_id,
            selection_scope=selection_scope,
            score_field=score_field,
            ranking_order=ranking_order,
        )
        coverage = selection.scoreable_count / selection.attempted_count
        if coverage < float(sgs_option(self.args, "min_update_coverage", 0.95)):
            raise RuntimeError(
                f"sensitivity score coverage below per-update gate: {coverage:.6f} "
                f"({selection.scoreable_count}/{selection.attempted_count})"
            )
        retention_weight = float(getattr(self.args, "pr_weight", 0.0) or 0.0)
        retain_all_for_retention = retention_weight > 0.0 and str(
            getattr(self.args, "pr_support", "selected")
        ) == "all"
        training_samples = bind_selection_to_samples(
            flat_samples,
            records,
            selection,
            retain_unselected=retain_all_for_retention,
        )
        selected_samples = [sample for sample in training_samples if bool(sample.sgs_selected)]
        selected_keys = set(selection.selected_keys)
        base_selected_keys = set(selection.base_selected_keys)
        dp_floor_added_keys = set(selection.dp_floor_added_keys)
        trajectory_floor_added_keys = set(selection.trajectory_floor_added_keys)
        selected_records = [
            row
            for row in records
            if (int(row["source_draw_id"]), int(row["turn_idx"])) in selected_keys
        ]
        mass_coverage = {}
        for field in sorted({"teacher_js", "teacher_kl_plain_privileged", "teacher_kl_privileged_plain", "distillation_kl", score_field}):
            selected_mass = sum(float(row[field]) for row in selected_records if row.get(field) is not None)
            total_mass = sum(float(row[field]) for row in records if row.get(field) is not None)
            mass_coverage[field] = selected_mass / total_mass if total_mass > 0.0 else None
        unavailable_reasons: dict[str, int] = {}
        for row in records:
            if row.get(score_field) is not None:
                continue
            reason = str(row.get("score_unavailable_reason") or "unknown")
            unavailable_reasons[reason] = unavailable_reasons.get(reason, 0) + 1
        batch_metrics = {
            "attempted_steps": float(selection.attempted_count),
            "scoreable_steps": float(selection.scoreable_count),
            "selected_steps": float(selection.selected_count),
            "base_selected_steps": float(len(selection.base_selected_keys)),
            "dp_floor_added_steps": float(len(selection.dp_floor_added_keys)),
            "trajectory_floor_added_steps": float(len(selection.trajectory_floor_added_keys)),
            "trajectory_coverage_before": float(
                selection.base_selected_trajectory_count / selection.attempted_trajectory_count
            ),
            "trajectory_coverage_after": float(
                selection.selected_trajectory_count / selection.attempted_trajectory_count
            ),
            "score_coverage": float(coverage),
            "teacher_js_mass_coverage": float(mass_coverage["teacher_js"] or 0.0),
            "distillation_kl_mass_coverage": float(mass_coverage["distillation_kl"] or 0.0),
            "ranking_score_mass_coverage": float(mass_coverage[score_field] or 0.0),
            "scoring_wall_seconds": float(max(durations)),
        }
        for sample in training_samples:
            setattr(sample, "sgs_batch_metrics", batch_metrics)
        data = self._convert_samples_to_train_data(training_samples)
        loss_weights = data.get("sdpo_loss_weights")
        selected_count = len(selected_samples)
        if retention_weight > 0.0:
            components = data.get("pr_component")
            base_weights = data.get("pr_base_weight")
            retention_count = len(flat_samples) if retain_all_for_retention else selected_count
            if (
                not isinstance(loss_weights, list)
                or len(loss_weights) != selected_count + retention_count
                or components != [0.0] * selected_count + [1.0] * retention_count
                or not isinstance(base_weights, list)
                or len(base_weights) != selected_count + retention_count
            ):
                raise RuntimeError("PR sensitivity conversion did not preserve its configured support")
            audit_loss_weights = base_weights[:selected_count]
        else:
            if not isinstance(loss_weights, list) or len(loss_weights) != selected_count:
                raise RuntimeError("sensitivity conversion did not preserve one loss weight per selected source step")
            audit_loss_weights = loss_weights
        loss_weight_by_key = {
            sample_key(sample): float(weight)
            for sample, weight in zip(selected_samples, audit_loss_weights, strict=True)
        }
        audit_response_ids = (
            {
                sample_key(sample): [int(token_id) for token_id in sample.tokens[-sample.response_length :]]
                for sample in flat_samples
            }
            if rollout_id == 0
            and bool(sgs_option(self.args, "audit_response_tokens_update0", False))
            else {}
        )
        audit = {
            "schema_version": 1,
            "kind": "sgs_update",
            "rollout_id": rollout_id,
            "fraction": fraction,
            "selection_mode": selection_mode,
            "selection_seed": selection_seed,
            "selection_scope": selection_scope,
            "ranking_score_field": score_field,
            "ranking_order": ranking_order,
            "scoreable_definition": f"finite_nonnegative_{score_field}",
            "scoring_performed": True,
            "score_scope": str(sgs_option(self.args, "score_scope", "action")),
            "loss_aggregation": str(
                (getattr(self.args, "tlb_loss_aggregation", None) or "trajectory_balanced")
            ),
            "attempted_steps": selection.attempted_count,
            "scoreable_steps": selection.scoreable_count,
            "score_coverage": coverage,
            "requested_steps": selection.requested_count,
            "base_selected_steps": len(selection.base_selected_keys),
            "selected_steps": selection.selected_count,
            "dp_floor_added_steps": len(selection.dp_floor_added_keys),
            "trajectory_floor_added_steps": len(selection.trajectory_floor_added_keys),
            "trajectory_floor_added_keys": [list(key) for key in selection.trajectory_floor_added_keys],
            "attempted_trajectories": selection.attempted_trajectory_count,
            "scoreable_trajectories": selection.scoreable_trajectory_count,
            "base_selected_trajectories": selection.base_selected_trajectory_count,
            "selected_trajectories": selection.selected_trajectory_count,
            "trajectory_coverage_before": (
                selection.base_selected_trajectory_count / selection.attempted_trajectory_count
            ),
            "trajectory_coverage_after": (
                selection.selected_trajectory_count / selection.attempted_trajectory_count
            ),
            "retention_support": str(getattr(self.args, "pr_support", "selected")),
            "retention_view": str(getattr(self.args, "pr_view", "privileged")),
            "retention_kl_direction": str(
                getattr(self.args, "pr_kl_direction", "reverse")
            ),
            "retention_steps": len(flat_samples) if retain_all_for_retention else selection.selected_count,
            "dp_floor_applied": selection.dp_floor_applied,
            "dp_size": dp_size,
            "selected_action_tokens": sum(int(row.get("action_token_count", 0)) for row in selected_records),
            "selected_response_tokens": sum(int(row.get("response_token_count", 0)) for row in selected_records),
            "score_mass_coverage": mass_coverage,
            "teacher_js_mass_coverage": mass_coverage["teacher_js"],
            "distillation_kl_mass_coverage": mass_coverage["distillation_kl"],
            "ranking_score_mass_coverage": mass_coverage[score_field],
            "scoring_rank_duration_seconds": durations,
            "scoring_wall_seconds": max(durations),
            "unavailable_reasons": unavailable_reasons,
            "loss_weight_sum": sum(loss_weight_by_key.values()),
            "rows": [
                {
                    **sgs_audit_record(row),
                    "ranking_score": row.get(score_field),
                    "selected": (int(row["source_draw_id"]), int(row["turn_idx"])) in selected_keys,
                    "selection_reason": (
                        "trajectory_floor"
                        if (int(row["source_draw_id"]), int(row["turn_idx"])) in trajectory_floor_added_keys
                        else "dp_floor"
                        if (int(row["source_draw_id"]), int(row["turn_idx"])) in dp_floor_added_keys
                        else "global_minx"
                        if (int(row["source_draw_id"]), int(row["turn_idx"])) in base_selected_keys
                        and selection_scope in {"global", "global_with_trajectory_floor"}
                        and selection_mode == "sensitivity"
                        and ranking_order == "ascending"
                        else "global_topx"
                        if (int(row["source_draw_id"]), int(row["turn_idx"])) in base_selected_keys
                        and selection_scope in {"global", "global_with_trajectory_floor"}
                        and selection_mode == "sensitivity"
                        else "global_random"
                        if (int(row["source_draw_id"]), int(row["turn_idx"])) in base_selected_keys
                        and selection_scope == "global"
                        and selection_mode == "random"
                        else "trajectory_topx"
                        if (int(row["source_draw_id"]), int(row["turn_idx"])) in selected_keys
                        else None
                    ),
                    "loss_weight": loss_weight_by_key.get(
                        (int(row["source_draw_id"]), int(row["turn_idx"]))
                    ),
                    **(
                        {
                            "response_token_ids": audit_response_ids[
                                (int(row["source_draw_id"]), int(row["turn_idx"]))
                            ]
                        }
                        if audit_response_ids
                        else {}
                    ),
                }
                for row in sorted(records, key=lambda row: (int(row["source_draw_id"]), int(row["turn_idx"])))
            ],
        }
        audit_root = Path(str(sgs_option(self.args, "audit_dir")))
        write_json_once(audit_root / f"rollout_{rollout_id:03d}.json", audit)
        return self._split_train_data_by_dp(data)

    def _finalize_unscored_random_topx(self, rollout_id: int, samples):
        flat_samples = _flatten_sample_groups(samples)
        records = []
        for sample in flat_samples:
            metadata = sample.metadata or {}
            train_metadata = sample.train_metadata or {}
            sdpo_metadata = train_metadata.get("sdpo", {}) if isinstance(train_metadata, dict) else {}
            action_mask = sdpo_metadata.get("sgs_action_token_mask", [])
            records.append(
                {
                    "source_draw_id": int(metadata["source_draw_id"]),
                    "traj_uid": str(metadata["source_trajectory_uid"]),
                    "turn_idx": int(metadata["source_turn_idx"]),
                    "task_id": str(metadata.get("task_id", "")),
                    "outcome": str(metadata.get("outcome") or sdpo_metadata.get("frozen_outcome_label", "")),
                    "trajectory_length": int(sdpo_metadata.get("frozen_trajectory_length", 0)),
                    "action_token_count": sum(bool(value) for value in action_mask),
                    "response_token_count": int(sample.response_length),
                    "random_rank_key": random_rank_key(
                        {
                            "source_draw_id": int(metadata["source_draw_id"]),
                            "turn_idx": int(metadata["source_turn_idx"]),
                            "traj_uid": str(metadata["source_trajectory_uid"]),
                        },
                        selection_seed=int(getattr(self.args, "sgs_selection_seed", 42)),
                        rollout_id=rollout_id,
                    )[0],
                }
            )
        fraction = sgs_selection_fraction(self.args)
        if fraction is None:
            raise RuntimeError("unscored random finalization requires an enabled fraction")
        dp_size = int(self.train_parallel_config["dp_size"])
        selection_seed = int(getattr(self.args, "sgs_selection_seed", 42))
        selection = select_sgs_steps(
            records,
            fraction=fraction,
            minimum_selected=dp_size,
            selection_mode="random",
            selection_seed=selection_seed,
            rollout_id=rollout_id,
            selection_scope="global",
            score_field=None,
        )
        retention_weight = float(getattr(self.args, "pr_weight", 0.0) or 0.0)
        retain_all_for_retention = retention_weight > 0.0 and str(
            getattr(self.args, "pr_support", "selected")
        ) == "all"
        training_samples = bind_selection_to_samples(
            flat_samples,
            records,
            selection,
            require_precomputed=False,
            retain_unselected=retain_all_for_retention,
        )
        selected_samples = [sample for sample in training_samples if bool(sample.sgs_selected)]
        batch_metrics = {
            "attempted_steps": float(selection.attempted_count),
            "scoreable_steps": float(selection.scoreable_count),
            "selected_steps": float(selection.selected_count),
            "score_coverage": 1.0,
            "scoring_wall_seconds": 0.0,
        }
        for sample in training_samples:
            setattr(sample, "sgs_batch_metrics", batch_metrics)
        data = self._convert_samples_to_train_data(training_samples)
        loss_weights = data.get("sdpo_loss_weights")
        expected_rows = len(selected_samples) + (len(flat_samples) if retain_all_for_retention else 0)
        if not isinstance(loss_weights, list) or len(loss_weights) != expected_rows:
            raise RuntimeError("unscored random conversion did not preserve its selected/retention support")
        audit_loss_weights = (
            data.get("pr_base_weight", [])[: len(selected_samples)]
            if retain_all_for_retention
            else loss_weights
        )
        if len(audit_loss_weights) != len(selected_samples):
            raise RuntimeError("unscored random conversion lost selected-row base weights")
        loss_weight_by_key = {
            sample_key(sample): float(weight)
            for sample, weight in zip(selected_samples, audit_loss_weights, strict=True)
        }
        selected_keys = set(selection.selected_keys)
        audit = {
            "schema_version": 2,
            "kind": "sgs_update",
            "rollout_id": rollout_id,
            "fraction": fraction,
            "selection_mode": "random",
            "selection_seed": selection_seed,
            "selection_scope": "global",
            "ranking_score_field": None,
            "scoreable_definition": "attempted_valid_generation",
            "score_scope": "none",
            "loss_aggregation": str(getattr(self.args, "tlb_loss_aggregation", "trajectory_balanced")),
            "retention_support": str(getattr(self.args, "pr_support", "selected")),
            "retention_view": str(getattr(self.args, "pr_view", "privileged")),
            "retention_kl_direction": str(
                getattr(self.args, "pr_kl_direction", "reverse")
            ),
            "retention_steps": len(flat_samples) if retain_all_for_retention else selection.selected_count,
            "scoring_skipped": True,
            "scoring_performed": False,
            "attempted_steps": selection.attempted_count,
            "scoreable_steps": selection.scoreable_count,
            "score_coverage": 1.0,
            "requested_steps": selection.requested_count,
            "selected_steps": selection.selected_count,
            "dp_floor_applied": selection.dp_floor_applied,
            "dp_size": dp_size,
            "selected_action_tokens": sum(
                int(row["action_token_count"])
                for row in records
                if (int(row["source_draw_id"]), int(row["turn_idx"])) in selected_keys
            ),
            "selected_response_tokens": sum(sample.response_length for sample in selected_samples),
            "scoring_rank_duration_seconds": [],
            "scoring_wall_seconds": 0.0,
            "unavailable_reasons": {},
            "loss_weight_sum": sum(loss_weight_by_key.values()),
            "rows": [
                {
                    **row,
                    "ranking_score": None,
                    "selected": (int(row["source_draw_id"]), int(row["turn_idx"])) in selected_keys,
                    "loss_weight": loss_weight_by_key.get((int(row["source_draw_id"]), int(row["turn_idx"]))),
                }
                for row in sorted(records, key=lambda row: (int(row["source_draw_id"]), int(row["turn_idx"])))
            ],
        }
        audit_root = Path(str(sgs_option(self.args, "audit_dir")))
        write_json_once(audit_root / f"rollout_{rollout_id:03d}.json", audit)
        return self._split_train_data_by_dp(data)

    def get_generation_timing(self, rollout_id: int) -> dict[str, float]:
        try:
            return dict(self.generation_timings[int(rollout_id)])
        except KeyError as exc:
            raise RuntimeError(f"rollout generation timing is unavailable for {rollout_id}") from exc

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return None
        if (
            bool(getattr(self.args, "strict_on_policy_audit_dir", None))
            or bool(getattr(self.args, "agent_task_strict_eval_snapshot", False))
        ):
            if self.eval_snapshot_iteration != int(rollout_id) or self.eval_weight_version is None:
                raise RuntimeError(
                    f"eval snapshot is not registered for iteration {rollout_id}: "
                    f"registered={self.eval_snapshot_iteration}, weight_version={self.eval_weight_version!r}"
                )
            current_versions = self._updatable_engine_weight_versions()
            if not current_versions or set(current_versions) != {self.eval_weight_version}:
                raise RuntimeError(
                    f"eval engine weight versions diverged from snapshot {self.eval_weight_version!r}: "
                    f"{current_versions}"
                )
            self.rollout_id = int(rollout_id)
        self.health_monitoring_resume()

        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        if _agent_task_eval_entropy_enabled(self.args):
            flat = _flatten_eval_step_samples(data)
            self._pending_eval_data[rollout_id] = data
            entropy_rollout_data = (
                self._split_eval_entropy_data_by_dp(self._convert_samples_to_eval_entropy_data(flat)) if flat else []
            )
            return {
                "requires_actor_entropy": True,
                "rollout_data_ref": entropy_rollout_data,
                "metrics": result.metrics,
            }
        return _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

    def finalize_eval_entropy(
        self,
        rollout_id: int,
        entropy_payloads: list[dict[str, Any] | None],
        extra_metrics: dict[str, Any] | None = None,
    ):
        data = self._pending_eval_data.pop(rollout_id)
        records = [record for payload in entropy_payloads if payload for record in payload.get("records", [])]
        entropy_by_key = {
            key: record.get("actor_entropy")
            for record in records
            if (key := _entropy_record_key(record)) is not None and record.get("actor_entropy") is not None
        }
        expected_by_task = _merge_expected_by_task(entropy_payloads)
        expected_count = (
            sum(expected_by_task.values())
            if expected_by_task
            else sum(int(payload.get("expected_count", 0)) for payload in entropy_payloads if payload)
        )
        patched = 0
        for sample in _flatten_eval_step_samples(data):
            key = _sample_entropy_key(sample)
            if key is None:
                continue
            entropy = entropy_by_key.get(key)
            if entropy is None:
                continue
            sample.metadata["actor_entropy"] = entropy
            patched += 1
        required = _env_bool(
            getattr(
                self.args,
                "agent_task_diversity_eval_entropy_required",
                os.environ.get("AGENT_TASK_DIVERSITY_EVAL_ENTROPY_REQUIRED", False),
            )
        )
        if required:
            threshold = float(
                getattr(
                    self.args,
                    "agent_task_diversity_entropy_coverage_threshold",
                    os.environ.get("AGENT_TASK_DIVERSITY_ENTROPY_COVERAGE_THRESHOLD", 0.95),
                )
            )
            coverage = patched / expected_count if expected_count else 0.0
            if coverage < threshold:
                raise RuntimeError(
                    f"eval actor entropy coverage below threshold for rollout_id={rollout_id}: "
                    f"coverage={coverage:.4f} threshold={threshold:.4f}"
                )
        log_dict = _log_eval_rollout_data(rollout_id, self.args, data, extra_metrics or {})
        _patch_eval_trace_actor_entropy(self.args, rollout_id, records, required=required)
        return log_dict

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    def _updatable_engine_weight_versions(self) -> list[str]:
        server = self._get_updatable_server()
        if server is None:
            return []
        versions = ray.get([engine.get_weight_version.remote() for engine in server.engines if engine is not None])
        return [str(version) for version in versions]

    def get_updatable_weight_versions(self) -> list[str]:
        return self._updatable_engine_weight_versions()

    def register_eval_snapshot(self, iteration: int) -> dict[str, Any]:
        """Bind subsequent eval and recovery operations to one synchronized actor snapshot."""

        iteration = int(iteration)
        versions = self._updatable_engine_weight_versions()
        if not versions or len(set(versions)) != 1:
            raise RuntimeError(f"cannot register eval snapshot {iteration} with engine versions {versions}")
        self.eval_snapshot_iteration = iteration
        self.eval_weight_version = versions[0]
        self.rollout_id = iteration
        return {"iteration": iteration, "weight_version": self.eval_weight_version, "engine_count": len(versions)}

    def verify_eval_snapshot(self, iteration: int) -> dict[str, Any]:
        """Prove that every rollout engine remained on the registered eval snapshot."""

        iteration = int(iteration)
        if self.eval_snapshot_iteration != iteration or self.eval_weight_version is None:
            raise RuntimeError(
                f"eval snapshot identity mismatch: registered={self.eval_snapshot_iteration}, requested={iteration}"
            )
        versions = self._updatable_engine_weight_versions()
        if not versions or set(versions) != {self.eval_weight_version}:
            raise RuntimeError(
                f"eval engine versions changed during snapshot {iteration}: "
                f"registered={self.eval_weight_version!r}, current={versions}"
            )
        return {"iteration": iteration, "weight_version": self.eval_weight_version, "engine_count": len(versions)}

    def get_data_source_cursor_state(self) -> dict[str, Any]:
        """Return a JSON-safe cursor fingerprint without mutating the data source."""

        state: dict[str, Any] = {"data_source_class": type(self.data_source).__name__}
        for name in (
            "sample_offset",
            "sample_group_index",
            "sample_index",
            "epoch_id",
            "completed_trajectories",
            "next_rollout_id",
        ):
            value = getattr(self.data_source, name, None)
            if isinstance(value, (bool, int, float, str)) or value is None:
                state[name] = value
        return state

    def register_training_snapshot(self, iteration: int, actor_version: int) -> dict[str, Any]:
        """Bind one synchronous rollout to the exact pre-update actor snapshot."""

        iteration = int(iteration)
        actor_version = int(actor_version)
        versions = self._updatable_engine_weight_versions()
        if not versions or len(set(versions)) != 1:
            raise RuntimeError(f"cannot register training snapshot {iteration} with engine versions {versions}")
        self.training_snapshot_iteration = iteration
        self.training_actor_version = actor_version
        self.training_weight_version = versions[0]
        self.rollout_id = iteration
        return {
            "iteration": iteration,
            "actor_version": actor_version,
            "weight_version": self.training_weight_version,
            "engine_count": len(versions),
        }

    def verify_training_snapshot(self, iteration: int, actor_version: int) -> dict[str, Any]:
        """Prove no rollout engine changed policy while the update batch was generated."""

        if self.training_snapshot_iteration != int(iteration) or self.training_actor_version != int(actor_version):
            raise RuntimeError(
                "training snapshot identity mismatch: "
                f"registered=({self.training_snapshot_iteration}, {self.training_actor_version}), "
                f"requested=({iteration}, {actor_version})"
            )
        versions = self._updatable_engine_weight_versions()
        if not versions or set(versions) != {self.training_weight_version}:
            raise RuntimeError(
                f"training engine weight versions changed during rollout {iteration}: "
                f"registered={self.training_weight_version!r}, current={versions}"
            )
        return {
            "iteration": int(iteration),
            "actor_version": int(actor_version),
            "weight_version": self.training_weight_version,
            "engine_count": len(versions),
        }

    def get_eval_snapshot(self) -> dict[str, Any]:
        return {
            "iteration": self.eval_snapshot_iteration,
            "weight_version": self.eval_weight_version,
            "rollout_id": self.rollout_id,
        }


    def resume_health_monitoring(self) -> None:
        self.health_monitoring_resume()

    def offload(self):
        self.health_monitoring_pause()
        for srv in self.servers.values():
            srv.offload()

    def onload(self, tags: list[str] | None = None):
        for srv in self.servers.values():
            srv.onload(tags)

    def onload_weights(self):
        for srv in self.servers.values():
            srv.onload_weights()

    def onload_kv(self):
        for srv in self.servers.values():
            srv.onload_kv()

    def recover_updatable_engines(self):
        """Restart any dead rollout engines and update num_new_engines for update_weights detection.

        Recovers the updatable model (the one that receives weight
        updates from training).
        """
        self.health_monitoring_pause()
        srv = self._get_updatable_server()
        if srv is None:
            return [], self.rollout_engine_lock, 0, [], []
        if self.rollout_id == -1 and not bool(getattr(self.args, "agent_task_strict_eval_snapshot", False)):
            engines = srv.engines if srv else []
            gpu_counts = srv.engine_gpu_counts if srv else []
            gpu_offsets = srv.engine_gpu_offsets if srv else []
            return engines, self.rollout_engine_lock, (srv.num_new_engines if srv else 0), gpu_counts, gpu_offsets

        # ``ServerGroup.start_engines()`` resets ``num_new_engines`` on every
        # call.  Preserve the initial discovery count while the actor has not
        # connected yet; otherwise a no-op health recovery immediately before
        # the first weight sync makes the updater believe there are no engines.
        unconnected_engines = srv.num_new_engines
        srv.recover()
        return (
            srv.engines,
            self.rollout_engine_lock,
            max(unconnected_engines, srv.num_new_engines),
            srv.engine_gpu_counts,
            srv.engine_gpu_offsets,
        )

    def clear_updatable_num_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        srv = self._get_updatable_server()
        if srv:
            srv.num_new_engines = 0

    def health_monitoring_pause(self) -> None:
        for monitor in self._health_monitors:
            monitor.pause()

    def health_monitoring_resume(self) -> None:
        for monitor in self._health_monitors:
            monitor.resume()

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                logger.info(
                    f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
                )
            metrics = None
        else:
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            # Enforce the rollout_id contract before flattening: any list[Sample]
            # encountered in the nested output must have rollout_id set on every
            # element. Default rollouts inherit it from the data source; compact /
            # subagent paths that split one rollout into N training samples must
            # set the same rollout_id on every sibling so the loss reducer counts
            # the rollout once instead of N times.
            _validate_rollout_id_annotated(data)
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))

        return data, metrics

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        # TODO to be refactored (originally Buffer._set_data)
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            logger.info(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            # TODO may improve the format
            if evaluation:
                dump_data = dict(
                    samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
                )
            else:
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            # group norm
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)

            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        """
        Convert inference generated samples to training data.
        """
        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        return self._default_convert_samples_to_train_data(samples)

    def _convert_samples_to_eval_entropy_data(self, samples: list[Sample]):
        """Package eval samples for actor entropy scoring without train-only converters."""
        return self._default_convert_samples_to_train_data(samples, include_grpo_token_weight_context=False)

    def _default_convert_samples_to_train_data(
        self,
        samples: list[Sample] | list[list[Sample]],
        *,
        include_grpo_token_weight_context: bool = True,
    ):
        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        # Rollout id (one per rollout execution). Default rollouts emit one
        # sample per rollout, so we fall back to ``sample.index`` (unique).
        # Compact / subagent paths that emit multiple training samples per
        # rollout set ``rollout_id`` explicitly so all siblings share a
        # value; the loss reducer then aggregates them as one rollout.
        if samples[0].rollout_id is None:
            rollout_ids = list(range(len(samples)))
        else:
            rollout_ids = [sample.rollout_id for sample in samples]

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
            "rollout_ids": rollout_ids,
        }

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length

            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        if bool(getattr(self.args, "normalize_over_active_rollouts", False)):
            active_rollout_ids = {
                int(rollout_id)
                for rollout_id, loss_mask in zip(train_data["rollout_ids"], loss_masks, strict=True)
                if sum(loss_mask) > 0
            }
            if not active_rollout_ids:
                raise ValueError("active-rollout normalization requires at least one non-masked rollout")
            train_data["active_rollout_ids"] = active_rollout_ids

        # Per-rollout aggregate, precomputed at the step level (where we can
        # see every sample of every rollout) and broadcast per-sample so the
        # per-mb loss reducer uses the correct whole-rollout denominator even
        # when a rollout's samples land in different micro-batches (first-fit
        # packing can split a rollout across mbs):
        #
        #   ``rollout_mask_sums[i]`` — sum of loss-mask totals over every
        #   sample in sample i's rollout. Used as the reducer's denominator
        #   so summing partial contributions across mbs yields one
        #   token-weighted mean per rollout.
        rollout_id_list = train_data["rollout_ids"]
        mask_sums_per_sample = [sum(m) for m in loss_masks]
        rollout_total_mask: dict[int, int] = {}
        for rid, ms in zip(rollout_id_list, mask_sums_per_sample, strict=True):
            rollout_total_mask[rid] = rollout_total_mask.get(rid, 0) + ms
        train_data["rollout_mask_sums"] = [rollout_total_mask[rid] for rid in rollout_id_list]

        # Overwrite raw_reward when available. Mixed-source batches may only
        # populate this field for a subset of samples (e.g. SWE but not code).
        if any(sample.metadata and "raw_reward" in sample.metadata for sample in samples):
            train_data["raw_reward"] = [
                sample.metadata["raw_reward"] if sample.metadata and "raw_reward" in sample.metadata else sample.reward
                for sample in samples
            ]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if any(sample.train_metadata is not None for sample in samples):
            train_data["metadata"] = [_sanitize_rollout_metadata(sample.train_metadata) for sample in samples]
            _maybe_add_sdpo_train_data(train_data, samples)

        if any(sample.multimodal_train_inputs is not None for sample in samples):
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        if include_grpo_token_weight_context and bool(getattr(self.args, "grpo_token_weights", False)):
            _add_grpo_token_weight_context(train_data, self.args, samples)

        return train_data

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config
        self.args.train_parallel_config = config

    def _split_train_data_by_dp(self, data):
        """Compute the DP/mbs schedule and package each rank's rollout_data
        into a Ray Box. The schedule itself is computed by
        :func:`build_dp_schedule` so it stays unit-testable without Ray/sglang.

        Step split is by rollout id (``samples[i].rollout_id``, falling back
        to ``samples[i].index``); each step holds exactly
        ``args.global_batch_size`` rollouts so the training-step count per
        rollout is fixed at ``rollout_batch_size * n_samples_per_prompt //
        global_batch_size`` regardless of how many training samples each
        rollout produced.
        """
        dp_size = self.train_parallel_config["dp_size"]
        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
            self.args,
            self.train_parallel_config,
            total_lengths,
            global_batch_size=(
                len(dict.fromkeys(data["rollout_ids"]))
                if bool(data.get("sgs_sparse_single_step", False))
                else self.args.global_batch_size
            ),
            rollout_indices=data["rollout_ids"],
        )
        if bool(data.get("sgs_sparse_single_step", False)):
            if len(num_microbatches) != 1:
                raise AssertionError("sensitivity sparse scheduling must produce exactly one optimizer step")
            global_batch_sizes = [int(self.args.global_batch_size)]
        if bool(getattr(self.args, "normalize_over_active_rollouts", False)):
            active_rollout_ids = data.get("active_rollout_ids")
            if not isinstance(active_rollout_ids, set):
                raise ValueError("active-rollout normalization requires converter field active_rollout_ids")
            global_batch_sizes = active_rollout_batch_sizes(
                data["rollout_ids"],
                active_rollout_ids,
                global_batch_size=self.args.global_batch_size,
            )

        # Package per-rank rollout_data
        rollout_data_refs = []
        for r in range(dp_size):
            partition = partitions[r]
            rollout_data = {"partition": partition}
            for key in [
                "tokens",
                "multimodal_train_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "round_number",
                "sample_indices",
                "rollout_ids",
                "rollout_mask_sums",
                "rollout_log_probs",
                "rollout_routed_experts",
                "prompt",
                "metadata",
                "teacher_log_probs",
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
                "sdpo_selected_success_traj_uid",
                "sdpo_token_weight_contrast_prompt_text",
                "sdpo_token_weight_contrast_messages",
                "sdpo_token_weight_contrast_source",
                "sdpo_token_weight_contrast_traj_uid",
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
                "sdpo_token_weights",
                "sdpo_teacher_log_probs",
                "sdpo_topk_indices",
                "sdpo_teacher_topk_log_probs",
                "sdpo_teacher_representations",
                "pr_component",
                "pr_base_weight",
                "pr_normalization_scale",
                "grpo_token_weight_positive_prompt_text",
                "grpo_token_weight_positive_messages",
                "grpo_token_weight_contrast_prompt_text",
                "grpo_token_weight_contrast_messages",
                "grpo_token_weight_contrast_source",
                "grpo_token_weight_uid",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = [data[key][j] for j in partition]
            # keys that need to be splited at train side
            for key in ["raw_reward", "total_lengths"]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            for key in _per_sample_sdpo_metric_keys(data):
                rollout_data[key] = [data[key][j] for j in partition]
            rollout_data["global_batch_sizes"] = global_batch_sizes
            rollout_data["num_microbatches"] = num_microbatches
            rollout_data["micro_batch_indices"] = micro_batch_indices[r]
            rollout_data_refs.append(Box(ray.put(rollout_data)))
        return rollout_data_refs

    def _split_eval_entropy_data_by_dp(self, data):
        """Package eval samples for actor forward scoring without train-batch schedule constraints."""
        dp_size = self.train_parallel_config["dp_size"]
        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths
        partitions = [[] for _ in range(dp_size)]
        for index in range(len(total_lengths)):
            partitions[index % dp_size].append(index)

        rollout_data_refs = []
        for partition in partitions:
            local_count = len(partition)
            rollout_data = {"partition": partition}
            for key in [
                "tokens",
                "multimodal_train_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "round_number",
                "sample_indices",
                "rollout_ids",
                "rollout_mask_sums",
                "rollout_log_probs",
                "rollout_routed_experts",
                "prompt",
                "metadata",
                "teacher_log_probs",
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
                "sdpo_selected_success_traj_uid",
                "sdpo_token_weight_contrast_prompt_text",
                "sdpo_token_weight_contrast_messages",
                "sdpo_token_weight_contrast_source",
                "sdpo_token_weight_contrast_traj_uid",
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
                "sdpo_token_weights",
                "sdpo_teacher_log_probs",
                "sdpo_teacher_representations",
                "grpo_token_weight_positive_prompt_text",
                "grpo_token_weight_positive_messages",
                "grpo_token_weight_contrast_prompt_text",
                "grpo_token_weight_contrast_messages",
                "grpo_token_weight_contrast_source",
                "grpo_token_weight_uid",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = [data[key][j] for j in partition]
            for key in ["raw_reward", "total_lengths"]:
                if key in data:
                    rollout_data[key] = data[key]
            for key in _per_sample_sdpo_metric_keys(data):
                rollout_data[key] = [data[key][j] for j in partition]
            rollout_data["global_batch_sizes"] = [local_count] if local_count else []
            rollout_data["num_microbatches"] = [local_count] if local_count else []
            rollout_data["micro_batch_indices"] = [[i] for i in range(local_count)]
            rollout_data_refs.append(Box(ray.put(rollout_data)))
        return rollout_data_refs


def _per_sample_sdpo_metric_keys(data: dict) -> list[str]:
    row_count = len(data.get("tokens", []))
    keys = []
    for key, value in data.items():
        if not key.startswith("self_distillation/"):
            continue
        if isinstance(value, list) and len(value) == row_count:
            keys.append(key)
    return keys


def _add_grpo_token_weight_context(train_data: dict, args: Any, samples: list[Sample]) -> None:
    from slime_plugins.agent_tasks.common.algorithms.sdpo_context import prepare_grpo_token_weight_context_plan

    plan = prepare_grpo_token_weight_context_plan(args, samples)
    for key in (
        "grpo_token_weight_positive_prompt_text",
        "grpo_token_weight_positive_messages",
        "grpo_token_weight_contrast_prompt_text",
        "grpo_token_weight_contrast_messages",
        "grpo_token_weight_contrast_source",
        "grpo_token_weight_uid",
    ):
        train_data[key] = [row[key] for row in plan]


def _maybe_add_sdpo_train_data(train_data: dict, samples: list[Sample]) -> None:
    sdpo_rows = []
    for sample in samples:
        train_metadata = sample.train_metadata if isinstance(sample.train_metadata, dict) else {}
        sdpo = train_metadata.get("sdpo") if isinstance(train_metadata, dict) else None
        sdpo_rows.append(sdpo if isinstance(sdpo, dict) else None)
    if not any(row is not None for row in sdpo_rows):
        return
    if any(row is None for row in sdpo_rows):
        raise ValueError("Mixed SDPO and non-SDPO samples in one train-data batch are not supported.")

    raw_metadata_rows = [dict(row) for row in sdpo_rows]
    metadata_rows = [_sanitize_rollout_metadata(row) for row in raw_metadata_rows]
    teacher_prompts = [str(row.get("sdpo_current_prompt_text", "")) for row in raw_metadata_rows]
    teacher_messages = [
        (
            row.get("sdpo_current_raw_prompt")
            if isinstance(row.get("sdpo_current_raw_prompt"), list)
            else [{"role": "user", "content": prompt}]
        )
        for row, prompt in zip(raw_metadata_rows, teacher_prompts, strict=True)
    ]
    train_data.update(
        {
            "sdpo_metadata": metadata_rows,
            "sdpo_teacher_prompt_text": teacher_prompts,
            "sdpo_teacher_prompt_texts": [None] * len(metadata_rows),
            "sdpo_teacher_messages": teacher_messages,
            "sdpo_teacher_messages_list": [None] * len(metadata_rows),
            "sdpo_teacher_signal_type": ["none"] * len(metadata_rows),
            "self_distillation_mask": [0.0] * len(metadata_rows),
            "sdpo_loss_weights": [0.0] * len(metadata_rows),
            "sdpo_representation_success_aggregation": ["sample"] * len(metadata_rows),
            "sdpo_representation_success_count": [0] * len(metadata_rows),
            "sdpo_selected_success_traj_uid": [None] * len(metadata_rows),
            "sdpo_token_weight_contrast_prompt_text": [None] * len(metadata_rows),
            "sdpo_token_weight_contrast_messages": [None] * len(metadata_rows),
            "sdpo_token_weight_contrast_source": [""] * len(metadata_rows),
            "sdpo_token_weight_contrast_traj_uid": [None] * len(metadata_rows),
        }
    )


def _sanitize_rollout_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            if key in _ROLLOUT_METADATA_OMIT_KEYS:
                continue
            if key == "image" and isinstance(item, str) and item.startswith("data:image"):
                sanitized[key] = _redacted_image_marker(item)
            else:
                sanitized[key] = _sanitize_rollout_metadata(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_rollout_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_rollout_metadata(item) for item in value)
    if isinstance(value, str) and value.startswith("data:image"):
        return _redacted_image_marker(value)
    return value


def _redacted_image_marker(image_data: str) -> str:
    return f"redacted_image_sha256:{hashlib.sha256(image_data.encode('utf-8')).hexdigest()[:16]}"


def _validate_rollout_id_annotated(node, depth=0):
    """Walk the rollout function's nested output and validate ``rollout_id`` only
    when a compact / subagent pattern is detected.

    "Compact" = the rollout function wraps multiple training samples from one
    rollout execution into a ``list[Sample]``. In slime's convention the
    default rollout shape is ``list[list[Sample]]`` (depth-2: prompt × rollout)
    so its leaf ``list[Sample]`` lands at depth 1 and we skip validation,
    preserving backward compatibility. A compact rollout adds a third level:
    ``list[list[list[Sample]]]`` (prompt × rollout × samples-from-one-rollout),
    so the leaf ``list[Sample]`` lands at depth ≥ 2. At that point we require
    every sibling to carry a non-None ``rollout_id`` and to share the same
    value, so the loss reducer counts the rollout once instead of N times.
    """
    if isinstance(node, Sample):
        return
    assert isinstance(node, list), f"unexpected rollout output node type: {type(node).__name__}"
    if node and isinstance(node[0], Sample):
        if depth >= 2 and len(node) > 1:
            rids = [s.rollout_id for s in node]
            missing = [i for i, r in enumerate(rids) if r is None]
            assert not missing, (
                f"Compact rollout returned {len(node)} samples but rollout_id is unset on "
                f"positions {missing}. Set Sample.rollout_id on every sibling so the loss "
                "reducer can aggregate them as one rollout instead of N."
            )
            assert len(set(rids)) == 1, f"Sibling samples from one compact rollout must share rollout_id; got {rids}."
        return
    for item in node:
        _validate_rollout_id_annotated(item, depth + 1)


def _allocate_rollout_engine_addr_and_ports_normal(
    *,
    args,
    rollout_engines,
    worker_type="regular",
    num_gpus_per_engine=None,
    rank_offset=0,
    base_port=15000,
):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    num_engines_per_node = max(1, args.num_gpus_per_node // _gpus_per_engine)
    addr_and_ports: dict[int, dict] = {}

    # Track per-node port cursors so that different server groups (called
    # sequentially) never race for the same ports on a given node.
    node_port_cursor: dict[int, int] = {}

    visited_nodes = set()
    for rank, engine in rollout_engines:
        local_rank = rank - rank_offset
        node_index = local_rank // num_engines_per_node
        if node_index in visited_nodes:
            continue
        visited_nodes.add(node_index)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_engines_on_this_node = num_engines_per_node - (local_rank % num_engines_per_node)

        def get_addr_and_ports(engine, node_idx):
            # use small ports to prevent ephemeral port between 32768 and 65536.
            # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
            start_port = node_port_cursor.get(node_idx, base_port)

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                node_port_cursor[node_idx] = start_port
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine, node_index)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports.setdefault(current_rank, {})
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()

            if worker_type == "prefill":
                addr_and_ports[current_rank]["disaggregation_bootstrap_port"] = get_port()

        if _gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = _gpus_per_engine // args.num_gpus_per_node
            if local_rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports.setdefault(rank + i, {})
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports, node_port_cursor


def _start_router(args, *, has_pd_disaggregation: bool = False, force_new: bool = False) -> tuple[str, int]:
    """Start sglang_router and return (router_ip, router_port).

    If ``args.sglang_router_ip`` is already set (e.g. by the user) and
    ``force_new`` is False, skip launching and return the existing values.
    When ``force_new`` is True (multi-model), always allocate a fresh port.
    """
    if not force_new and args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    router_ip = _wrap_ipv6(get_host_info()[1])
    if force_new:
        router_port = find_available_port(random.randint(3000, 4000))
    else:
        router_port = args.sglang_router_port
        if router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))

    from sglang_router.launch_router import RouterArgs

    from slime.utils.http_utils import run_router

    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
    router_args.host = router_ip
    router_args.port = router_port
    router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
    router_args.log_level = "warn"
    router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    if has_pd_disaggregation:
        router_args.pd_disaggregation = True
        # Disable circuit breaker to prevent RDMA transfer timeouts from
        # marking decode workers as dead. Timeouts are transient (PCIe
        # contention under high load) and do not indicate a dead server.
        router_args.disable_circuit_breaker = True

    # We will not use the health check from router.
    router_args.disable_health_check = True

    logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    process.start()
    # Wait 3 seconds
    time.sleep(3)
    assert process.is_alive()
    logger.info(f"Router launched at {router_ip}:{router_port}, Prometheus port: {router_args.prometheus_port}")
    return router_ip, router_port


def _compute_rollout_offset(args) -> int:
    """Offset (in PG bundle slots) where rollout GPUs start."""
    if args.debug_train_only or args.debug_rollout_only or args.colocate:
        return 0
    offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    return offset


def _compute_megatron_num_gpus(args) -> int:
    """Total number of megatron (actor + critic) GPU slots in the placement group."""
    if args.debug_rollout_only:
        return 0
    num = args.actor_num_nodes * args.actor_num_gpus_per_node
    return num


def start_rollout_servers(args, pg) -> dict[str, Any]:
    """Start rollout servers: one per model, each with its own router.

    Each model defined in the sglang config gets its own router and set
    of server groups.  Server groups within a model may have different
    ``num_gpus_per_engine`` (e.g. for PD disaggregation where prefill
    and decode use different TP sizes).

    Returns a dict mapping model name → ``RolloutServer``.

    Note: ``init_http_client`` should be called separately before this,
    as the HTTP client is shared across all servers.
    """
    if args.rollout_external:
        return start_external_rollout_servers(args, start_router=_start_router)

    config = _resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}
    gpu_offset = 0
    engine_offset = 0

    # Compute megatron GPU range for per-group offload decisions.
    rollout_pg_offset = _compute_rollout_offset(args)
    megatron_num_gpus = _compute_megatron_num_gpus(args)

    for model_idx, model_cfg in enumerate(config.models):
        model_cfg.resolve(args)

        has_pd = model_cfg.has_pd_disaggregation
        router_ip, router_port = _start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))

        # Write back for backward compat (first model only).
        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        server_groups: list[ServerGroup] = []
        port_cursors: dict[int, int] = {}

        has_epd = model_cfg.has_encoder_disaggregation

        def _make_group(group_cfg, router_ip, router_port, overrides_extra=None):
            nonlocal engine_offset, gpu_offset
            gpus_per_engine = group_cfg.num_gpus_per_engine
            num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
            num_engines = group_cfg.num_gpus // num_gpu_per_engine_local

            group_abs_start = rollout_pg_offset + gpu_offset
            needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
            overrides = dict(group_cfg.overrides)
            if overrides_extra:
                for k, v in overrides_extra.items():
                    overrides.setdefault(k, v)
            if args.offload_rollout and not needs_offload:
                overrides.setdefault("enable_memory_saver", False)
            logger.info(
                f"Engine group '{group_cfg.worker_type}' gpu_offset={gpu_offset} "
                f"(abs={group_abs_start}): needs_offload={needs_offload}"
            )

            group = ServerGroup(
                args=args,
                pg=pg,
                all_engines=[None] * num_engines if group_cfg.worker_type != "placeholder" else [],
                num_gpus_per_engine=gpus_per_engine,
                num_new_engines=0,
                worker_type=group_cfg.worker_type,
                rank_offset=engine_offset,
                gpu_offset=gpu_offset,
                sglang_overrides=overrides,
                needs_offload=needs_offload,
                model_path=overrides.get("model_path", args.hf_checkpoint),
                router_ip=router_ip,
                router_port=router_port,
            )
            engine_offset += num_engines
            gpu_offset += group_cfg.num_gpus
            return group

        if has_epd:
            # --- Phase 1: start encoder groups, wait, collect URLs ---
            encoder_urls: list[str] = []
            for group_cfg in model_cfg.server_groups:
                if group_cfg.worker_type != "encoder":
                    continue
                group = _make_group(group_cfg, router_ip, router_port)
                handles, port_cursors = group.start_engines(port_cursors)
                if handles:
                    ray.get(handles)
                urls = ray.get([e.get_url.remote() for e in group.engines])
                encoder_urls.extend(u for u in urls if u is not None)
                server_groups.append(group)

            logger.info(f"EPD phase 1 done: collected {len(encoder_urls)} encoder URLs: {encoder_urls}")

            # --- Phase 2: start non-encoder groups, injecting encoder URLs into
            # language-only LLM workers. Prefill groups use this for full EPD,
            # while regular groups allow encoder/LLM split without PD.
            non_encoder_handles: list = []
            for group_cfg in model_cfg.server_groups:
                if group_cfg.worker_type == "encoder":
                    continue
                overrides_extra = {}
                if encoder_urls and group_cfg.worker_type in ("prefill", "regular"):
                    overrides_extra["language_only"] = True
                    overrides_extra["encoder_urls"] = encoder_urls
                group = _make_group(group_cfg, router_ip, router_port, overrides_extra=overrides_extra)
                handles, port_cursors = group.start_engines(port_cursors)
                non_encoder_handles.extend(handles)
                server_groups.append(group)

            if non_encoder_handles:
                ray.get(non_encoder_handles)
        else:
            # No EPD — start all groups in one pass (original path).
            all_init_handles: list = []
            for group_cfg in model_cfg.server_groups:
                group = _make_group(group_cfg, router_ip, router_port)
                handles, port_cursors = group.start_engines(port_cursors)
                all_init_handles.extend(handles)
                server_groups.append(group)

            if all_init_handles:
                ray.get(all_init_handles)

        servers[model_cfg.name] = RolloutServer(
            server_groups=server_groups,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
        )

    # Expose per-model router info for custom rollout functions.
    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers


def _resolve_sglang_config(args) -> SglangConfig:
    """Build a SglangConfig from args, choosing the right source."""
    if getattr(args, "sglang_config", None) is not None:
        config = SglangConfig.from_yaml(args.sglang_config)
        # Validate total GPUs match.
        expected = args.rollout_num_gpus
        actual = config.total_num_gpus
        assert actual == expected, f"sglang_config total GPUs ({actual}) != rollout_num_gpus ({expected})"
        return config

    if args.prefill_num_servers is not None:
        return SglangConfig.from_prefill_num_servers(args)

    # Default: single regular group.
    return SglangConfig(
        models=[
            ModelConfig(
                name="default",
                server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)],
            )
        ]
    )


def _log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    log_dict = extra_metrics or {}
    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        if custom_log_func(rollout_id, args, data, log_dict):
            return log_dict

    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")

    return log_dict


def _flatten_eval_step_samples(data: dict[str, Any]) -> list[Sample]:
    return [
        sample
        for dataset_data in data.values()
        for sample in (dataset_data.get("step_samples") or dataset_data.get("samples", []))
    ]


def _merge_expected_by_task(payloads: list[dict[str, Any] | None]) -> dict[str, int]:
    expected_by_task: dict[str, int] = {}
    for payload in payloads:
        if not payload:
            continue
        for task, count in (payload.get("expected_by_task") or {}).items():
            expected_by_task[str(task)] = expected_by_task.get(str(task), 0) + int(count)
    return expected_by_task


def _patch_eval_trace_actor_entropy(
    args,
    rollout_id: int,
    records: list[dict[str, Any]],
    *,
    required: bool,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        task = record.get("agent_task")
        if task:
            grouped.setdefault(str(task), []).append(record)
    if not grouped:
        if required:
            raise RuntimeError(f"no eval actor entropy records available for rollout_id={rollout_id}")
        return

    from slime_plugins.agent_tasks.common.config import get_arg_or_env
    from slime_plugins.agent_tasks.common.trace import (
        DEFAULT_TRACE_COMPRESSION,
        AgentTraceConfig,
        patch_trace_sidecar_actor_entropy,
    )

    for task, task_records in grouped.items():
        trace_dir = _trace_dir_from_records_or_args(args, task_records)
        if trace_dir is None:
            if required:
                raise RuntimeError(f"trace dir unavailable for required eval actor entropy task={task}")
            continue
        config = AgentTraceConfig(
            enabled=True,
            trace_dir=trace_dir,
            phases=frozenset({"eval"}),
            compression=str(
                get_arg_or_env(
                    args,
                    "agent_task_trace_compression",
                    "AGENT_TASK_TRACE_COMPRESSION",
                    DEFAULT_TRACE_COMPRESSION,
                )
            )
            .strip()
            .lower()
            or "none",
        )
        patch_trace_sidecar_actor_entropy(
            config,
            task=task,
            phase="eval",
            outer_rollout_id=int(rollout_id),
            records=task_records,
            required=required,
        )


def _trace_dir_from_records_or_args(args, records: list[dict[str, Any]]) -> Path | None:
    for record in records:
        value = record.get("agent_task_trace_dir")
        if value:
            return Path(str(value))
    raw = getattr(args, "agent_task_trace_dir", None) or os.environ.get("AGENT_TASK_TRACE_DIR")
    return Path(str(raw)) if raw else None


def _sample_entropy_key(sample: Sample) -> tuple[Any, Any, Any, Any] | None:
    metadata = sample.metadata or {}
    return _entropy_key(
        metadata.get("sample_rollout_id", metadata.get("rollout_id", sample.rollout_id)),
        metadata.get("traj_uid"),
        metadata.get("turn_idx"),
        metadata.get("sample_index", sample.index),
    )


def _entropy_record_key(record: dict[str, Any]) -> tuple[Any, Any, Any, Any] | None:
    return _entropy_key(
        record.get("sample_rollout_id"),
        record.get("traj_uid"),
        record.get("turn_idx"),
        record.get("sample_index"),
    )


def _entropy_key(sample_rollout_id: Any, traj_uid: Any, turn_idx: Any, sample_index: Any):
    if sample_rollout_id is None or traj_uid is None or turn_idx is None or sample_index is None:
        return None
    return int(sample_rollout_id), traj_uid, int(turn_idx), int(sample_index)


def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")


def compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()
    return log_dict


def compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if getattr(args, "sglang_speculative_algorithm", None) is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
