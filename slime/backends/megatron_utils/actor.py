# Modified for ReSAIL. See NOTICE and LICENSE for attribution and terms.
import logging
import math
import os
import random
import shutil
import time
from argparse import Namespace
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoTokenizer

from slime.algorithms.sdpo.teacher_alignment import DEFAULT_SDPO_MAX_REPROMPT_TOKENS, build_sdpo_teacher_rollout_data
from slime.algorithms.sdpo.sgs_metrics import (
    paired_compressed_view_metrics,
)
from slime.ray.train_actor import TrainRayActor
from slime.utils import train_dump_utils
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.logging_utils import init_tracking
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.misc import Box
from slime.utils.processing_utils import load_processor
from slime.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from slime.utils.routing_replay import RoutingReplay
from slime.utils.seqlen_balancing import expand_bins_by_splitting, first_fit_pack
from slime.utils.timer import Timer, inverse_timer, timer, with_defer
from slime.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from .checkpoint import load_checkpoint
from .cp_utils import all_gather_with_cp, slice_log_prob_with_cp, slice_with_cp
from .data import TOP_LEVEL_LOG_METRIC_PREFIX, DataIterator, get_data_iterator, log_perf_data, log_rollout_data
from .hf_checkpoint_saver import save_hf_model_to_path
from .initialize import init, is_megatron_main_rank
from .loss import (
    compute_advantages_and_returns,
    compute_sdpo_topk_token_kl,
    get_log_probs_and_entropy,
    get_log_probs_entropy_and_sdpo_student_representations,
    get_sdpo_distillation_and_representation_tensors,
    get_sdpo_distillation_and_compressed_action_view_tensors,
    get_sdpo_dual_support_distillation_and_compressed_action_view_tensors,
    get_sdpo_distillation_tensors,
    get_sdpo_compressed_action_view_tensors,
    get_sdpo_representation_and_prompt_last_tensors,
    get_sdpo_representation_tensors,
    get_values,
)
from .model import forward_only, initialize_model_and_optimizer, save, train
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_disk import UpdateWeightFromDisk
from .update_weight.update_weight_from_distributed import UpdateWeightFromDistributed
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

SDPO_DISTILLATION_OUTPUT_KEYS = (
    "sdpo_topk_indices",
    "sdpo_teacher_log_probs",
    "sdpo_teacher_topk_log_probs",
    "sdpo_teacher_all_log_probs",
    "sdpo_teacher_representations",
)


def _slice_train_log_prob_with_cp(
    log_prob,
    total_length: int,
    response_length: int,
    qkv_format: str,
    max_seq_len: int | None,
    *,
    allow_missing_retention_target: bool,
):
    """Slice one log-prob row while preserving an explicitly deferred SDPO target."""
    if log_prob is None:
        if allow_missing_retention_target:
            return None
        raise ValueError("missing log-prob row outside a PR or explicitly deferred SDPO target")
    return slice_log_prob_with_cp(log_prob, total_length, response_length, qkv_format, max_seq_len)


SDPO_TEACHER_REPRESENTATION_FORWARD_CHUNK_ROWS = 512
SDPO_ENTROPY_ROLLOUT_DATA_POSTPROCESS_PATH = (
    "slime_plugins.agent_tasks.common.entropy_postprocess.postprocess_rollout_entropy"
)
SDPO_TOKEN_WEIGHT_MONITOR_PREFIX = f"{TOP_LEVEL_LOG_METRIC_PREFIX}sdpo_token_weights"
SDPO_TOKEN_WEIGHT_DISAGREEMENT_BINS = (
    ("lt_0_01", None, 0.01),
    ("0_01_to_0_05", 0.01, 0.05),
    ("0_05_to_0_1", 0.05, 0.1),
    ("0_1_to_0_2", 0.1, 0.2),
    ("0_2_to_0_5", 0.2, 0.5),
    ("0_5_to_1", 0.5, 1.0),
    ("ge_1", 1.0, None),
)
SDPO_TOKEN_WEIGHT_WEIGHT_BINS = (
    ("lt_0_25", None, 0.25),
    ("0_25_to_0_5", 0.25, 0.5),
    ("0_5_to_1", 0.5, 1.0),
    ("1_to_2", 1.0, 2.0),
    ("2_to_4", 2.0, 4.0),
    ("ge_4", 4.0, None),
)
SDPO_TOKEN_WEIGHT_QUANTILES = (
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
)
SDPO_TOKEN_WEIGHT_SCORE_EPS = 1e-6


def _env_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _agent_task_rollout_entropy_enabled(args: Namespace) -> bool:
    value = getattr(args, "agent_task_diversity_entropy_enabled", None)
    if value is None:
        value = os.environ.get("AGENT_TASK_DIVERSITY_ENTROPY_ENABLED", False)
    return _env_bool(value)


def _actor_entropy_temperature(args: Namespace, phase: str) -> float:
    if phase == "eval":
        eval_temperature = getattr(args, "eval_temperature", None)
        if eval_temperature is not None:
            return float(eval_temperature)
    return float(getattr(args, "rollout_temperature", 1.0))


class MegatronTrainRayActor(TrainRayActor):
    @contextmanager
    def _routing_replay_stage(self, stage: str):
        if not getattr(self.args, "use_routing_replay", False):
            yield
            return

        old_stage = os.environ.get("ROUTING_REPLAY_STAGE")
        os.environ["ROUTING_REPLAY_STAGE"] = stage
        try:
            yield
        finally:
            if old_stage is None:
                os.environ.pop("ROUTING_REPLAY_STAGE", None)
            else:
                os.environ["ROUTING_REPLAY_STAGE"] = old_stage

    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        if args.debug_rollout_only:
            self.args = args
            return 0

        monkey_patch_torch_dist()
        super().init(args, role, with_ref, with_opd_teacher)

        self._driver_owned_train_tracking = bool(getattr(args, "driver_owned_train_tracking", False))
        if self._driver_owned_train_tracking:
            # Ray serializes args for every actor, so this does not mutate the driver's
            # primary tracking configuration. Metrics are returned to the driver.
            args.use_wandb = False
        init(args)

        if is_megatron_main_rank() and not self._driver_owned_train_tracking:
            init_tracking(args, primary=False, role=role)

        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
                self.processor = load_processor(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        dist.barrier(group=get_gloo_group())

        if args.offload_train:
            if (x := args.train_memory_margin_bytes) > 0:
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

        self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id = initialize_model_and_optimizer(
            args, role
        )

        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
        if vpp_size > 1:
            from megatron.core.utils import get_model_config

            microbatch_group_size_per_vp_stage = get_model_config(self.model[0]).microbatch_group_size_per_vp_stage
        else:
            microbatch_group_size_per_vp_stage = 1
        self.train_parallel_config = {
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
            "cp_size": mpu.get_context_parallel_world_size(),
            "vpp_size": vpp_size,
            "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
        }

        start_rollout_id = loaded_rollout_id + 1

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return start_rollout_id

        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=args.megatron_to_hf_mode == "raw",
            ),
            single_tag=None,
        )
        self._active_model_tag: str | None = "actor"
        self._sgs_prepass_kept_awake = False
        self._online_old_actor_weight_sync_initialized = False
        self.weights_backuper.backup("actor")
        if self._use_sdpo_ema_teacher():
            self.weights_backuper.backup("sdpo_teacher")
            self._maybe_load_sdpo_ema_teacher(loaded_rollout_id)

        if with_ref or self._grpo_token_weights_enabled():
            self.load_other_checkpoint("ref", args.ref_load)

        # Load teacher model for Megatron-based on-policy distillation
        if with_opd_teacher:
            self.load_other_checkpoint("teacher", args.opd_teacher_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1 and not self._uses_per_update_sdpo_old_actor():
                self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            # Prefer HF config vocab_size (which may include model-native padding)
            # over tokenizer vocab_size (which may be smaller, e.g. GPT-OSS).
            hf_vocab = getattr(self.hf_config, "vocab_size", None)
            self.args.vocab_size = hf_vocab if hf_vocab is not None else self.tokenizer.vocab_size

        if self.args.colocate:
            assert (
                self.args.update_weight_mode == "full"
            ), "--update-weight-mode=delta is not supported with --colocate"
            update_weight_cls = UpdateWeightFromTensor
        elif self.args.update_weight_mode == "delta":
            # Lazy import: the delta module pulls DeltaEncoding/DeltaParam/DeltaSpec from
            # sglang, which only exist on newer images. Importing eagerly would break old
            # images even when delta mode is unused.
            from .update_weight.update_weight_from_distributed_delta import UpdateWeightFromDistributedDelta

            update_weight_cls = UpdateWeightFromDistributedDelta
        else:
            assert self.args.update_weight_mode == "full"
            if self.args.update_weight_transport == "disk":
                update_weight_cls = UpdateWeightFromDisk
            else:
                assert (
                    self.args.update_weight_mode == "full" and self.args.update_weight_transport == "nccl"
                ), f"unsupported weight sync mode/transport: {self.args.update_weight_mode!r}/{self.args.update_weight_transport!r}"
                update_weight_cls = UpdateWeightFromDistributed
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
        )

        # empty cache after initialization
        clear_memory()

        if self.args.offload_train:
            # recover to actor in the end.
            self._switch_model("actor")
            self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        if (
            self.role == "actor"
            and self.args.use_critic
            and not self.args.colocate
            and hasattr(self.weight_updater, "disconnect_rollout_engines")
        ):
            self.weight_updater.disconnect_rollout_engines()
        destroy_process_groups()

        torch_memory_saver.pause()

        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        torch_memory_saver.resume()

        clear_memory()
        reload_process_groups()
        if self.role == "actor":
            self._switch_model("actor")
        print_memory("after wake_up model")

    def _get_rollout_data(self, rollout_data_ref: Box) -> RolloutBatch:
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will receive the data.
        rollout_data = process_rollout_data(
            self.args,
            rollout_data_ref,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
        )
        # TODO: this is ugly, move to somewhere else?
        # move tokens to GPU in advance
        rollout_data["tokens"] = [
            torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device()) for t in rollout_data["loss_masks"]
        ]
        if not rollout_data["tokens"]:
            return rollout_data
        if "rollout_mask_sums" in rollout_data:
            # Promote precomputed per-rollout mask totals to GPU tensors here
            # (matching loss_masks) so the loss reducer can just divide.
            rollout_data["rollout_mask_sums"] = torch.tensor(
                rollout_data["rollout_mask_sums"], dtype=torch.float32, device=torch.cuda.current_device()
            )
        if "multimodal_train_inputs" in rollout_data:
            # Move multimodal training tensors to GPU in advance
            rollout_data["multimodal_train_inputs"] = [
                (
                    {
                        key: (
                            torch.from_numpy(v.copy()).to(device=torch.cuda.current_device())
                            if isinstance(v, np.ndarray)
                            else v.to(device=torch.cuda.current_device())
                        )
                        for key, v in mm_dict.items()
                    }
                    if mm_dict is not None
                    else None
                )
                for mm_dict in rollout_data["multimodal_train_inputs"]
            ]

        if self.args.qkv_format == "bshd":
            # TODO: micro-batch wise dynamic, possibly move to @data.py:get_data_iterator
            max_seq_len = max(rollout_data["total_lengths"])

            # pad to reduce memory fragmentation and maybe make the computation faster
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size

            rollout_data["max_seq_lens"] = [max_seq_len] * len(rollout_data["tokens"])

        for key in ["rollout_log_probs", "teacher_log_probs", "sdpo_teacher_log_probs"]:
            if key not in rollout_data:
                continue
            sliced_rows = []
            components = rollout_data.get("pr_component")
            for i, (log_prob, total_length, response_length) in enumerate(
                zip(
                    rollout_data[key],
                    rollout_data["total_lengths"],
                    rollout_data["response_lengths"],
                    strict=False,
                )
            ):
                sliced = _slice_train_log_prob_with_cp(
                    log_prob,
                    total_length,
                    response_length,
                    self.args.qkv_format,
                    rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                    allow_missing_retention_target=(
                        key == "sdpo_teacher_log_probs"
                        and isinstance(components, list)
                        and i < len(components)
                        and (
                            float(components[i]) == 1.0
                            or (
                                str(getattr(self.args, "sgs_selection_mode", "sensitivity")) == "random"
                                and bool(getattr(self.args, "sgs_skip_full_scoring", False))
                            )
                        )
                    ),
                )
                sliced_rows.append(
                    None
                    if sliced is None
                    else torch.tensor(
                        sliced,
                        device=torch.cuda.current_device(),
                        dtype=torch.float32,
                    )
                )
            rollout_data[key] = sliced_rows
        if "rollout_routed_experts" in rollout_data:
            rollout_data["rollout_routed_experts"] = [
                torch.from_numpy(r) for r in rollout_data["rollout_routed_experts"]
            ]
        return rollout_data

    def _switch_model(self, target_tag: str) -> None:
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
        if "rollout_routed_experts" not in rollout_data:
            raise ValueError(
                "rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set."
            )

        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        from slime.utils.routing_replay import RoutingReplay

        for iterator in data_iterator:
            iterator.reset()

        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        def pad_func(experts, pad):
            _, num_layers, topk = experts.shape
            pad = (
                torch.arange(
                    pad * num_layers * topk,
                    device=experts.device,
                    dtype=experts.dtype,
                ).reshape((pad, num_layers, topk))
                % self.args.num_experts
            )
            return torch.cat([experts, pad], dim=0)

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
            rollout_routed_experts = batch["rollout_routed_experts"]
            tokens = batch["tokens"]
            assert len(rollout_routed_experts) == len(tokens)
            for a, b in zip(rollout_routed_experts, tokens, strict=False):
                assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

            # We need to pad the experts to the last token. We won't calculate loss on this token so this should be fine.
            # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
            rollout_routed_experts = [pad_func(r, 1) for r in rollout_routed_experts]
            # TODO: maybe extract a common process function for here and get_batch?
            rollout_routed_experts = [slice_with_cp(r, pad_func) for r in rollout_routed_experts]
            rollout_routed_experts = torch.cat(rollout_routed_experts, dim=0)
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            pad = (pad_size - rollout_routed_experts.size(0) % pad_size) % pad_size
            if pad != 0:
                rollout_routed_experts = pad_func(rollout_routed_experts, pad)

            if self.args.sequence_parallel:
                seqlen = rollout_routed_experts.size(0)
                assert seqlen % tp_size == 0
                start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
                rollout_routed_experts = rollout_routed_experts[start:end]

            routing_replay_offset = 0
            for vp_stage, model in enumerate(self.model):
                config = model.module.config
                num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
                offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
                for layer_id in range(offset, offset + num_layers_to_build):
                    # skip dense layer
                    if isinstance(config.moe_layer_freq, int):
                        if layer_id % config.moe_layer_freq != 0:
                            continue
                    elif isinstance(config.moe_layer_freq, list):
                        assert len(config.moe_layer_freq) == config.num_layers
                        if config.moe_layer_freq[layer_id] == 0:
                            continue
                    layer_routed_experts = rollout_routed_experts[:, layer_id]
                    RoutingReplay.all_routing_replays[routing_replay_offset].record(layer_routed_experts)
                    routing_replay_offset += 1
            assert routing_replay_offset == len(RoutingReplay.all_routing_replays)

        del rollout_data["rollout_routed_experts"]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
        with_entropy: bool | None = None,
        temperature: float | None = None,
        collect_sdpo_student_representations: bool = False,
    ) -> dict[str, list[torch.Tensor]]:

        collect_student_representations = bool(collect_sdpo_student_representations)
        callback = (
            get_log_probs_entropy_and_sdpo_student_representations
            if collect_student_representations
            else get_log_probs_and_entropy
        )
        with timer(f"{store_prefix}log_probs"):
            result = forward_only(
                callback,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
                with_entropy=with_entropy,
                temperature=temperature,
                capture_hidden_states=collect_student_representations,
            )
        if with_entropy and not self.args.use_rollout_entropy and "entropy" in result:
            result["actor_entropy"] = result.pop("entropy")
        return result

    def _log_prob_max_tokens_per_gpu(self) -> int:
        max_tokens = getattr(self.args, "log_probs_max_tokens_per_gpu", None)
        if max_tokens is None:
            max_tokens = getattr(self.args, "max_tokens_per_gpu", 0)
        return int(max_tokens or 0)

    def _align_log_prob_microbatches_for_vpp(
        self,
        micro_batch_indices: list[list[int]],
        lengths: list[int],
    ) -> list[list[int]] | None:
        vpp_size = int(self.train_parallel_config.get("vpp_size", 1) or 1)
        if vpp_size <= 1:
            return micro_batch_indices

        mb_group = int(self.train_parallel_config.get("microbatch_group_size_per_vp_stage", 1) or 1)
        if mb_group <= 1:
            return micro_batch_indices

        target_count = ((len(micro_batch_indices) + mb_group - 1) // mb_group) * mb_group
        if target_count == len(micro_batch_indices):
            return micro_batch_indices

        aligned = [list(indices) for indices in micro_batch_indices]
        expand_bins_by_splitting(aligned, target_count, lengths)
        if len(aligned) != target_count:
            return None
        return aligned

    def _pack_log_prob_microbatches(
        self,
        rollout_data: RolloutBatch,
        max_tokens: int,
    ) -> tuple[list[list[int]], list[int]] | None:
        lengths = [int(length) for length in rollout_data["total_lengths"]]
        source_microbatches = rollout_data["micro_batch_indices"]
        source_num_microbatches = rollout_data["num_microbatches"]

        micro_batch_indices: list[list[int]] = []
        num_microbatches: list[int] = []
        source_offset = 0
        for step_num_microbatches in source_num_microbatches:
            step_microbatches = source_microbatches[source_offset : source_offset + step_num_microbatches]
            source_offset += step_num_microbatches
            step_row_indices = [row_idx for indices in step_microbatches for row_idx in indices]
            if not step_row_indices:
                return None

            step_lengths = [lengths[row_idx] for row_idx in step_row_indices]
            step_micro_batch_indices = [
                [step_row_indices[local_idx] for local_idx in indices]
                for indices in self._pack_sdpo_teacher_representation_microbatches(step_lengths, max_tokens)
            ]
            step_micro_batch_indices = self._align_log_prob_microbatches_for_vpp(
                step_micro_batch_indices,
                lengths,
            )
            if step_micro_batch_indices is None:
                return None

            micro_batch_indices.extend(step_micro_batch_indices)
            num_microbatches.append(len(step_micro_batch_indices))

        if source_offset != len(source_microbatches):
            return None
        return micro_batch_indices, num_microbatches

    def _get_log_prob_data_iterator(
        self,
        rollout_data: RolloutBatch,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
    ) -> tuple[list[DataIterator], list[int]]:
        if not getattr(self.args, "use_dynamic_batch_size", False):
            return data_iterator, num_microbatches
        if getattr(self.args, "use_routing_replay", False) or getattr(self.args, "use_rollout_routing_replay", False):
            return data_iterator, num_microbatches

        row_count = len(rollout_data.get("total_lengths", []))
        if row_count == 0:
            return data_iterator, num_microbatches

        max_tokens = self._log_prob_max_tokens_per_gpu()
        if max_tokens <= 0:
            return data_iterator, num_microbatches
        if max_tokens == int(getattr(self.args, "max_tokens_per_gpu", 0) or 0):
            return data_iterator, num_microbatches

        cp_size = int(self.train_parallel_config.get("cp_size", mpu.get_context_parallel_world_size()))
        max_tokens *= max(cp_size, 1)
        packed_microbatches = self._pack_log_prob_microbatches(rollout_data, max_tokens)
        if packed_microbatches is None:
            return data_iterator, num_microbatches
        micro_batch_indices, log_prob_num_microbatches = packed_microbatches

        log_prob_rollout_data = dict(rollout_data)
        log_prob_rollout_data["micro_batch_indices"] = micro_batch_indices
        log_prob_rollout_data["num_microbatches"] = log_prob_num_microbatches
        if len(rollout_data.get("global_batch_sizes", [])) != len(log_prob_num_microbatches):
            log_prob_rollout_data["global_batch_sizes"] = [row_count]
        return get_data_iterator(log_prob_rollout_data), log_prob_rollout_data["num_microbatches"]

    def compute_sdpo_distillation_data(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:
        with timer(f"{store_prefix}distillation"):
            return forward_only(
                get_sdpo_distillation_tensors,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def compute_sdpo_compressed_action_view_data(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str,
        *,
        include_distillation: bool,
    ) -> dict[str, list[torch.Tensor]]:
        callback = (
            get_sdpo_distillation_and_compressed_action_view_tensors
            if include_distillation
            else get_sdpo_compressed_action_view_tensors
        )
        with timer(f"{store_prefix}compressed_action_view"):
            return forward_only(
                callback,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def compute_sdpo_dual_support_compressed_action_view_data(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str,
        *,
        secondary_topk_indices: list[torch.Tensor] | None,
    ) -> dict[str, list[torch.Tensor]]:
        # Extra per-row support must travel through the DataIterator so each
        # dynamically packed microbatch receives the matching subset.
        for iterator in data_iterator:
            iterator.rollout_data["sdpo_secondary_topk_indices"] = secondary_topk_indices
        with timer(f"{store_prefix}compressed_action_view"):
            return forward_only(
                get_sdpo_dual_support_distillation_and_compressed_action_view_tensors,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def compute_sdpo_representation_data(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
        collect_decision_states: bool = False,
    ) -> dict[str, list[torch.Tensor]]:
        with timer(f"{store_prefix}representation_distillation"):
            if collect_decision_states:
                return forward_only(
                    get_sdpo_representation_and_prompt_last_tensors,
                    self.args,
                    self.model,
                    data_iterator,
                    num_microbatches,
                    store_prefix=store_prefix,
                    capture_hidden_states=True,
                )
            return forward_only(
                get_sdpo_representation_tensors,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
                return_hidden_states=True,
            )

    def compute_sdpo_distillation_and_representation_data(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:
        with timer(f"{store_prefix}distillation"):
            return forward_only(
                get_sdpo_distillation_and_representation_tensors,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
                capture_hidden_states=True,
            )

    def _build_sdpo_teacher_rollout_data(self, rollout_data: RolloutBatch) -> RolloutBatch:
        teacher_rollout_data = build_sdpo_teacher_rollout_data(
            rollout_data,
            self.tokenizer,
            processor=getattr(self, "processor", None),
            apply_chat_template_kwargs=getattr(self.args, "apply_chat_template_kwargs", None),
            max_prompt_tokens=getattr(self.args, "sdpo_max_reprompt_tokens", DEFAULT_SDPO_MAX_REPROMPT_TOKENS),
            truncation_side=getattr(self.args, "sdpo_reprompt_truncation_side", "right"),
        )
        for key in (
            "sdpo_teacher_prompt_token_lengths",
            "sdpo_teacher_prompt_token_lengths_raw",
            "sdpo_teacher_prompt_truncated",
            "sdpo_teacher_prompt_char_lengths",
        ):
            if key in teacher_rollout_data:
                rollout_data[key] = teacher_rollout_data[key]
        if "sdpo_topk_indices" in rollout_data:
            teacher_rollout_data["sdpo_topk_indices"] = rollout_data["sdpo_topk_indices"]
        if "sgs_action_token_mask" in rollout_data:
            teacher_rollout_data["sgs_action_token_mask"] = rollout_data[
                "sgs_action_token_mask"
            ]
        for key in ("rollout_mask_sums",):
            if key in rollout_data:
                teacher_rollout_data[key] = rollout_data[key]
        if self.args.qkv_format == "bshd":
            max_seq_len = max(teacher_rollout_data["total_lengths"])
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size
            teacher_rollout_data["max_seq_lens"] = [max_seq_len] * len(teacher_rollout_data["tokens"])
        self._repack_sdpo_teacher_microbatches(teacher_rollout_data)
        return teacher_rollout_data

    def _expand_sdpo_teacher_prompt_ensembles(
        self,
        rollout_data: RolloutBatch,
    ) -> tuple[RolloutBatch, list[int] | None]:
        prompt_ensembles = rollout_data.get("sdpo_teacher_prompt_texts")
        if prompt_ensembles is None:
            return rollout_data, None

        row_count = len(rollout_data["total_lengths"])
        if not isinstance(prompt_ensembles, list) or len(prompt_ensembles) != row_count:
            raise ValueError("sdpo_teacher_prompt_texts must be a row-aligned list when present.")

        message_ensembles = rollout_data.get("sdpo_teacher_messages_list")
        if message_ensembles is not None and (
            not isinstance(message_ensembles, list) or len(message_ensembles) != row_count
        ):
            raise ValueError("sdpo_teacher_messages_list must be a row-aligned list when present.")

        expanded_original_indices: list[int] = []
        expanded_prompts: list[str] = []
        expanded_messages: list[list[dict[str, object]]] = []
        has_multi_prompt_row = False
        scalar_prompts = rollout_data["sdpo_teacher_prompt_text"]
        scalar_messages = rollout_data.get("sdpo_teacher_messages")
        use_expanded_messages = message_ensembles is not None or scalar_messages is not None

        for row_idx in range(row_count):
            prompts = prompt_ensembles[row_idx]
            if prompts:
                if isinstance(prompts, (str, bytes)):
                    raise ValueError("sdpo_teacher_prompt_texts rows must be sequences of prompt strings.")
                prompt_list = [str(prompt) for prompt in prompts]
            else:
                prompt_list = [str(scalar_prompts[row_idx])]
            if not prompt_list:
                raise ValueError(f"sdpo_teacher_prompt_texts[{row_idx}] must not be empty.")

            messages_list = None if message_ensembles is None else message_ensembles[row_idx]
            if messages_list is not None:
                if len(messages_list) != len(prompt_list):
                    raise ValueError(
                        f"sdpo_teacher_messages_list[{row_idx}] length must match "
                        f"sdpo_teacher_prompt_texts[{row_idx}]."
                    )
                row_messages = [[dict(message) for message in messages] for messages in messages_list]
            else:
                if scalar_messages is not None:
                    default_messages = scalar_messages[row_idx]
                    row_messages = [[dict(message) for message in default_messages] for _ in prompt_list]
                else:
                    row_messages = [[] for _ in prompt_list]

            has_multi_prompt_row = has_multi_prompt_row or len(prompt_list) > 1
            for prompt, messages in zip(prompt_list, row_messages, strict=True):
                expanded_original_indices.append(row_idx)
                expanded_prompts.append(prompt)
                expanded_messages.append(messages)

        if not has_multi_prompt_row:
            return rollout_data, None

        expanded: RolloutBatch = {}
        row_aligned_teacher_keys = {
            "tokens",
            "total_lengths",
            "response_lengths",
            "loss_masks",
            "dynamic_global_batch_size",
            "multimodal_train_inputs",
            "sdpo_teacher_multimodal_train_inputs",
        }
        for key in row_aligned_teacher_keys:
            if key not in rollout_data:
                continue
            value = rollout_data[key]
            if isinstance(value, list) and len(value) == row_count:
                expanded[key] = [value[idx] for idx in expanded_original_indices]
            elif isinstance(value, tuple) and len(value) == row_count:
                expanded[key] = [value[idx] for idx in expanded_original_indices]
            elif isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == row_count:
                index = torch.tensor(expanded_original_indices, device=value.device, dtype=torch.long)
                expanded[key] = value.index_select(0, index)
            else:
                expanded[key] = value
        expanded["sdpo_teacher_prompt_text"] = expanded_prompts
        if use_expanded_messages:
            expanded["sdpo_teacher_messages"] = expanded_messages
        return expanded, expanded_original_indices

    def _sdpo_teacher_representation_forward_chunk_rows(self) -> int:
        chunk_rows = getattr(
            self.args,
            "sdpo_teacher_representation_forward_chunk_rows",
            SDPO_TEACHER_REPRESENTATION_FORWARD_CHUNK_ROWS,
        )
        chunk_rows = int(chunk_rows or 0)
        if chunk_rows <= 0:
            raise ValueError("sdpo_teacher_representation_forward_chunk_rows must be positive.")
        return chunk_rows

    def _sdpo_teacher_representation_max_tokens_per_gpu(self) -> int:
        max_tokens = None
        if self._sdpo_uses_representation_distillation():
            max_tokens = getattr(self.args, "sdpo_teacher_representation_max_tokens_per_gpu", None)
        if max_tokens is None:
            max_tokens = getattr(self.args, "max_tokens_per_gpu", 0)
        return int(max_tokens or 0)

    def _new_sdpo_teacher_prompt_stat_accumulators(
        self,
        row_count: int,
    ) -> tuple[dict[str, list[float]], dict[str, list[int]], dict[str, list[float]]]:
        stat_fields = (
            "sdpo_teacher_prompt_token_lengths",
            "sdpo_teacher_prompt_token_lengths_raw",
            "sdpo_teacher_prompt_char_lengths",
        )
        stat_sums = {field: [0.0] * row_count for field in stat_fields}
        stat_counts = {field: [0] * row_count for field in stat_fields}
        stat_maxes = {"sdpo_teacher_prompt_truncated": [0.0] * row_count}
        return stat_sums, stat_counts, stat_maxes

    def _accumulate_sdpo_teacher_prompt_stats(
        self,
        stat_sums: dict[str, list[float]],
        stat_counts: dict[str, list[int]],
        stat_maxes: dict[str, list[float]],
        teacher_rollout_data: RolloutBatch,
        expanded_original_indices: list[int],
    ) -> None:
        for field, sums in stat_sums.items():
            values = teacher_rollout_data.get(field)
            if not isinstance(values, list) or len(values) != len(expanded_original_indices):
                continue
            counts = stat_counts[field]
            for original_idx, value in zip(expanded_original_indices, values, strict=True):
                sums[original_idx] += float(value)
                counts[original_idx] += 1
        for field, maxes in stat_maxes.items():
            values = teacher_rollout_data.get(field)
            if not isinstance(values, list) or len(values) != len(expanded_original_indices):
                continue
            for original_idx, value in zip(expanded_original_indices, values, strict=True):
                maxes[original_idx] = max(maxes[original_idx], float(value))

    def _set_aggregated_sdpo_teacher_prompt_stats(
        self,
        rollout_data: RolloutBatch,
        stat_sums: dict[str, list[float]],
        stat_counts: dict[str, list[int]],
        stat_maxes: dict[str, list[float]],
    ) -> None:
        for field, sums in stat_sums.items():
            counts = stat_counts[field]
            rollout_data[field] = [total / count if count > 0 else 0.0 for total, count in zip(sums, counts)]
        rollout_data["sdpo_teacher_prompt_token_lengths_ensemble_total"] = stat_sums[
            "sdpo_teacher_prompt_token_lengths"
        ]
        rollout_data["sdpo_teacher_prompt_char_lengths_ensemble_total"] = stat_sums["sdpo_teacher_prompt_char_lengths"]
        for field, values in stat_maxes.items():
            rollout_data[field] = values

    def _accumulate_sdpo_teacher_representation_sums(
        self,
        sums: list[torch.Tensor | None],
        counts: list[int],
        rows: list[torch.Tensor],
        expanded_original_indices: list[int],
    ) -> None:
        if len(rows) != len(expanded_original_indices):
            raise ValueError(
                "Expanded SDPO teacher representation count does not match expanded prompt count: "
                f"{len(rows)} != {len(expanded_original_indices)}."
            )
        for original_idx, row in zip(expanded_original_indices, rows, strict=True):
            row_cpu = row.detach().to(device="cpu", dtype=torch.float32)
            if sums[original_idx] is None:
                sums[original_idx] = row_cpu
            else:
                if sums[original_idx].shape != row_cpu.shape:
                    raise ValueError(
                        "SDPO teacher representation ensemble shape mismatch at row "
                        f"{original_idx}: {sums[original_idx].shape} != {row_cpu.shape}."
                    )
                sums[original_idx] = sums[original_idx] + row_cpu
            counts[original_idx] += 1

    def _finalize_sdpo_teacher_representation_sums(
        self,
        sums: list[torch.Tensor | None],
        counts: list[int],
    ) -> list[torch.Tensor]:
        aggregated: list[torch.Tensor] = []
        for row_idx, row_sum in enumerate(sums):
            if row_sum is None or counts[row_idx] <= 0:
                raise ValueError(f"Missing SDPO teacher representation ensemble rows for original row {row_idx}.")
            aggregated.append((row_sum / counts[row_idx]).to(dtype=torch.bfloat16))
        return aggregated

    def _aggregate_sdpo_teacher_representations(
        self,
        rollout_data: RolloutBatch,
        rows: list[torch.Tensor],
        expanded_original_indices: list[int] | None,
    ) -> list[torch.Tensor]:
        if expanded_original_indices is None:
            return [row.detach().to(device="cpu", dtype=torch.bfloat16) for row in rows]

        row_count = len(rollout_data["total_lengths"])
        sums: list[torch.Tensor | None] = [None] * row_count
        counts = [0] * row_count
        self._accumulate_sdpo_teacher_representation_sums(sums, counts, rows, expanded_original_indices)
        return self._finalize_sdpo_teacher_representation_sums(sums, counts)

    def _sdpo_token_weights_enabled(self) -> bool:
        return bool(getattr(self.args, "sdpo_token_weights", False))

    def _grpo_token_weights_enabled(self) -> bool:
        return bool(getattr(self.args, "grpo_token_weights", False))

    def _grpo_policy_token_weights_complete(self, rollout_data: RolloutBatch) -> bool:
        rows = rollout_data.get("grpo_policy_token_weights")
        response_lengths = rollout_data.get("response_lengths")
        return (
            isinstance(rows, list)
            and isinstance(response_lengths, list)
            and len(rows) == len(rollout_data["total_lengths"])
            and len(rows) == len(response_lengths)
            and all(
                isinstance(row, torch.Tensor)
                and row.numel() == int(response_length)
                and bool(torch.isfinite(row).all())
                and bool((row >= 0).all())
                for row, response_length in zip(rows, response_lengths, strict=True)
            )
        )

    def _build_grpo_token_weight_context_rollout_data(
        self,
        rollout_data: RolloutBatch,
        prompt_field: str,
        row_indices: list[int],
    ) -> RolloutBatch:
        context_data = self._slice_rollout_rows(rollout_data, row_indices)
        prompts = context_data.get(prompt_field)
        if not isinstance(prompts, list) or len(prompts) != len(row_indices) or not all(prompts):
            raise ValueError(f"{prompt_field} must contain a prompt for every selected GRPO token-weight row.")
        context_data["sdpo_teacher_prompt_text"] = prompts
        messages_field = prompt_field.replace("_prompt_text", "_messages")
        messages = context_data.get(messages_field)
        if (
            isinstance(messages, list)
            and len(messages) == len(row_indices)
            and all(message is not None for message in messages)
        ):
            context_data["sdpo_teacher_messages"] = messages
        else:
            context_data.pop("sdpo_teacher_messages", None)
        context_data.pop("sdpo_teacher_prompt_texts", None)
        context_data.pop("sdpo_teacher_messages_list", None)
        return self._build_sdpo_teacher_rollout_data(context_data)

    @staticmethod
    def _mean_preserving_cap_row(weights: torch.Tensor, cap: float) -> torch.Tensor | None:
        target = float(weights.numel())
        nonzero = int((weights > 0).sum().item())
        if nonzero * cap < target - 1e-6:
            return None
        low, high = 0.0, 1.0
        while float(torch.clamp(weights * high, max=cap).sum().item()) < target - 1e-6:
            high *= 2.0
            if not np.isfinite(high):
                return None
        for _ in range(64):
            middle = (low + high) / 2.0
            if float(torch.clamp(weights * middle, max=cap).sum().item()) < target:
                low = middle
            else:
                high = middle
        return torch.clamp(weights * high, max=cap)

    def _compute_grpo_policy_token_weights(
        self,
        rollout_data: RolloutBatch,
        positive_response_rows: list[torch.Tensor | None],
        positive_prompt_last_rows: list[torch.Tensor | None],
        contrast_response_rows: list[torch.Tensor | None],
        contrast_prompt_last_rows: list[torch.Tensor | None],
    ) -> tuple[list[torch.Tensor], dict[str, float]]:
        loss_masks = rollout_data["loss_masks"]
        row_count = len(loss_masks)
        inputs = (
            positive_response_rows,
            positive_prompt_last_rows,
            contrast_response_rows,
            contrast_prompt_last_rows,
        )
        if any(len(rows) != row_count for rows in inputs):
            raise ValueError("GRPO token-weight representation fields must be row-aligned with loss_masks.")
        power = float(getattr(self.args, "grpo_token_weights_power", 1.0))
        cap = float(getattr(self.args, "grpo_token_weights_max", 8.0))
        token_weights: list[torch.Tensor] = []
        zero_fallbacks = 0
        cap_infeasible = 0
        active_values: list[torch.Tensor] = []
        for row_idx, loss_mask in enumerate(loss_masks):
            active_mask = torch.as_tensor(loss_mask, dtype=torch.float32, device="cpu").reshape(-1) > 0
            weights = torch.zeros(active_mask.numel(), dtype=torch.float32)
            if not active_mask.any():
                token_weights.append(weights)
                continue
            positive_response = positive_response_rows[row_idx]
            positive_prompt_last = positive_prompt_last_rows[row_idx]
            contrast_response = contrast_response_rows[row_idx]
            contrast_prompt_last = contrast_prompt_last_rows[row_idx]
            if not all(
                isinstance(value, torch.Tensor)
                for value in (positive_response, positive_prompt_last, contrast_response, contrast_prompt_last)
            ):
                weights[active_mask] = 1.0
                token_weights.append(weights)
                active_values.append(weights[active_mask])
                continue
            positive_response = positive_response.detach().to(device="cpu", dtype=torch.float32)
            contrast_response = contrast_response.detach().to(device="cpu", dtype=torch.float32)
            positive_prompt_last = positive_prompt_last.detach().to(device="cpu", dtype=torch.float32).reshape(1, -1)
            contrast_prompt_last = contrast_prompt_last.detach().to(device="cpu", dtype=torch.float32).reshape(1, -1)
            expected_shape = (active_mask.numel(), positive_prompt_last.size(-1))
            if tuple(positive_response.shape) != expected_shape or tuple(contrast_response.shape) != expected_shape:
                raise ValueError(
                    f"GRPO token-weight response representation shape mismatch at row {row_idx}: "
                    f"positive={tuple(positive_response.shape)}, contrast={tuple(contrast_response.shape)}, "
                    f"expected={expected_shape}."
                )
            if positive_prompt_last.shape != contrast_prompt_last.shape:
                raise ValueError(f"GRPO token-weight prompt-last shape mismatch at row {row_idx}.")
            positive_decisions = torch.cat((positive_prompt_last, positive_response[:-1]), dim=0)
            contrast_decisions = torch.cat((contrast_prompt_last, contrast_response[:-1]), dim=0)
            disagreement = 1.0 - (
                F.normalize(positive_decisions, p=2, dim=-1) * F.normalize(contrast_decisions, p=2, dim=-1)
            ).sum(dim=-1)
            if not torch.isfinite(disagreement).all():
                raise ValueError(f"Non-finite GRPO token-weight disagreement at row {row_idx}.")
            disagreement = disagreement.clamp_min(0.0)
            scores = disagreement if power == 1.0 else disagreement.pow(power)
            active_scores = scores[active_mask]
            if not torch.isfinite(active_scores).all():
                raise ValueError(f"Non-finite GRPO token-weight score at row {row_idx}.")
            score_mean = float(active_scores.mean().item())
            if score_mean <= SDPO_TOKEN_WEIGHT_SCORE_EPS:
                weights[active_mask] = 1.0
                zero_fallbacks += 1
            else:
                normalized = active_scores / score_mean
                capped = self._mean_preserving_cap_row(normalized, cap)
                if capped is None:
                    weights[active_mask] = 1.0
                    cap_infeasible += 1
                else:
                    weights[active_mask] = capped
            if not torch.isfinite(weights).all():
                raise ValueError(f"Non-finite GRPO policy token weights at row {row_idx}.")
            token_weights.append(weights)
            active_values.append(weights[active_mask])
        active = torch.cat(active_values) if active_values else torch.empty(0)
        metrics = {
            "grpo_token_weights/active_mean": float(active.mean().item()) if active.numel() else 0.0,
            "grpo_token_weights/active_max": float(active.max().item()) if active.numel() else 0.0,
            "grpo_token_weights/fallback_zero_row_count": float(zero_fallbacks),
            "grpo_token_weights/cap_infeasible_row_count": float(cap_infeasible),
        }
        return token_weights, metrics

    def _ensure_grpo_policy_token_weights(self, rollout_data: RolloutBatch) -> None:
        if not self._grpo_token_weights_enabled() or self._grpo_policy_token_weights_complete(rollout_data):
            return
        if "ref" not in self.weights_backuper.backup_tags:
            raise ValueError("grpo_token_weights requires a frozen ref model backup.")
        row_count = len(rollout_data["total_lengths"])
        positive_prompts = rollout_data.get("grpo_token_weight_positive_prompt_text")
        contrast_prompts = rollout_data.get("grpo_token_weight_contrast_prompt_text")
        if not isinstance(positive_prompts, list) or not isinstance(contrast_prompts, list):
            raise ValueError("GRPO token-weight prompt fields must be row-aligned lists.")
        if len(positive_prompts) != row_count or len(contrast_prompts) != row_count:
            raise ValueError("GRPO token-weight prompt field row count mismatch.")
        row_indices = [
            idx
            for idx, (positive, contrast, mask) in enumerate(
                zip(positive_prompts, contrast_prompts, rollout_data["loss_masks"], strict=True)
            )
            if positive and contrast and torch.as_tensor(mask).sum().item() > 0
        ]
        aligned_outputs: dict[str, list[torch.Tensor | None]] = {
            "positive_response": [None] * row_count,
            "positive_prompt": [None] * row_count,
            "contrast_response": [None] * row_count,
            "contrast_prompt": [None] * row_count,
        }
        try:
            self._switch_model("ref")
            for label, prompt_field in (
                ("positive", "grpo_token_weight_positive_prompt_text"),
                ("contrast", "grpo_token_weight_contrast_prompt_text"),
            ):
                if not row_indices:
                    continue
                context_data = self._build_grpo_token_weight_context_rollout_data(
                    rollout_data, prompt_field, row_indices
                )
                prefix = f"grpo_token_weight_{label}_"
                with self._routing_replay_stage("fallthrough"):
                    outputs = self.compute_sdpo_representation_data(
                        get_data_iterator(context_data),
                        context_data["num_microbatches"],
                        store_prefix=prefix,
                        collect_decision_states=True,
                    )
                response_rows = outputs[f"{prefix}representations"]
                prompt_rows = outputs[f"{prefix}prompt_last_representations"]
                if len(response_rows) != len(row_indices) or len(prompt_rows) != len(row_indices):
                    raise ValueError(f"GRPO token-weight {label} representation row count mismatch.")
                for row_idx, response, prompt in zip(row_indices, response_rows, prompt_rows, strict=True):
                    aligned_outputs[f"{label}_response"][row_idx] = response
                    aligned_outputs[f"{label}_prompt"][row_idx] = prompt
        finally:
            self._switch_model("actor")
        weights, metrics = self._compute_grpo_policy_token_weights(
            rollout_data,
            aligned_outputs["positive_response"],
            aligned_outputs["positive_prompt"],
            aligned_outputs["contrast_response"],
            aligned_outputs["contrast_prompt"],
        )
        rollout_data["grpo_policy_token_weights"] = weights
        for key, value in metrics.items():
            rollout_data[key] = torch.tensor(value, dtype=torch.float32)

    def _sdpo_token_weight_source(self) -> str:
        if not self._sdpo_token_weights_enabled():
            return "teacher_contrast"
        return str(getattr(self.args, "sdpo_token_weight_source", "teacher_contrast"))

    def _sdpo_token_weights_complete(self, rollout_data: RolloutBatch) -> bool:
        rows = rollout_data.get("sdpo_token_weights")
        if not isinstance(rows, list):
            return False
        row_count = len(rollout_data["total_lengths"])
        return len(rows) == row_count and all(isinstance(row, torch.Tensor) for row in rows)

    def _sdpo_teacher_representations_complete(self, rollout_data: RolloutBatch) -> bool:
        rows = rollout_data.get("sdpo_teacher_representations")
        if not isinstance(rows, list):
            return False
        row_count = len(rollout_data["total_lengths"])
        return len(rows) == row_count and all(isinstance(row, torch.Tensor) for row in rows)

    def _sdpo_token_weight_contrast_row_indices(self, rollout_data: RolloutBatch) -> list[int]:
        prompts = rollout_data.get("sdpo_token_weight_contrast_prompt_text")
        masks = rollout_data.get("self_distillation_mask")
        loss_masks = rollout_data.get("loss_masks")
        if not isinstance(prompts, list) or not isinstance(masks, list) or not isinstance(loss_masks, list):
            return []
        row_indices: list[int] = []
        for idx, (prompt, mask, loss_mask) in enumerate(zip(prompts, masks, loss_masks, strict=True)):
            active_tokens = torch.as_tensor(loss_mask, dtype=torch.float32).sum().item()
            if prompt and float(mask) > 0.0 and active_tokens > 0.0:
                row_indices.append(idx)
        return row_indices

    def _build_sdpo_token_weight_contrast_rollout_data(
        self,
        rollout_data: RolloutBatch,
        row_indices: list[int],
    ) -> RolloutBatch:
        contrast_rollout_data = self._slice_rollout_rows(rollout_data, row_indices)
        contrast_prompts = contrast_rollout_data.get("sdpo_token_weight_contrast_prompt_text")
        if not isinstance(contrast_prompts, list):
            raise ValueError("sdpo_token_weight_contrast_prompt_text must be row-aligned.")
        contrast_rollout_data["sdpo_teacher_prompt_text"] = contrast_prompts
        contrast_messages = contrast_rollout_data.get("sdpo_token_weight_contrast_messages")
        if isinstance(contrast_messages, list):
            contrast_rollout_data["sdpo_teacher_messages"] = contrast_messages
        else:
            contrast_rollout_data.pop("sdpo_teacher_messages", None)
        contrast_rollout_data.pop("sdpo_teacher_prompt_texts", None)
        contrast_rollout_data.pop("sdpo_teacher_messages_list", None)
        return self._build_sdpo_teacher_rollout_data(contrast_rollout_data)

    def _ensure_sdpo_token_weights(self, rollout_data: RolloutBatch) -> None:
        if not self._sdpo_token_weights_enabled():
            return
        if self._sdpo_token_weights_complete(rollout_data):
            return
        if "sdpo_teacher_representations" not in rollout_data:
            raise ValueError("sdpo_token_weights requires sdpo_teacher_representations to be precomputed.")

        row_count = len(rollout_data["total_lengths"])
        source = self._sdpo_token_weight_source()
        contrast_representations: list[torch.Tensor | None] = [None] * row_count
        token_weight_power = float(getattr(self.args, "sdpo_token_weights_power", 1.0))
        if source == "student":
            student_representations = rollout_data.get("sdpo_student_representations")
            if not isinstance(student_representations, list) or len(student_representations) != row_count:
                raise ValueError(
                    "sdpo_token_weight_source=student requires row-aligned sdpo_student_representations "
                    "from the actor log-prob forward."
                )
            contrast_representations = student_representations
            row_indices = []
        else:
            row_indices = (
                [] if token_weight_power == 0.0 else self._sdpo_token_weight_contrast_row_indices(rollout_data)
            )
        if row_indices:
            teacher_tag = self._sdpo_teacher_model_tag()
            teacher_rollout_data = self._build_sdpo_token_weight_contrast_rollout_data(rollout_data, row_indices)
            try:
                self._switch_model(teacher_tag)
                teacher_data_iterator = get_data_iterator(teacher_rollout_data)
                teacher_num_microbatches = teacher_rollout_data["num_microbatches"]
                with self._routing_replay_stage("fallthrough"):
                    outputs = self.compute_sdpo_representation_data(
                        teacher_data_iterator,
                        teacher_num_microbatches,
                        store_prefix="sdpo_token_weight_contrast_",
                    )
            finally:
                self._switch_model("actor")
            rows = outputs["sdpo_token_weight_contrast_representations"]
            if len(rows) != len(row_indices):
                raise ValueError(
                    "SDPO token-weight contrast representation count mismatch: " f"{len(rows)} != {len(row_indices)}."
                )
            for row_idx, representation in zip(row_indices, rows, strict=True):
                contrast_representations[row_idx] = representation
        token_weights, metrics = self._compute_sdpo_token_weights(
            rollout_data,
            rollout_data["sdpo_teacher_representations"],
            contrast_representations,
        )
        rollout_data["sdpo_token_weights"] = token_weights
        for key, value in metrics.items():
            rollout_data[key] = torch.tensor(float(value), dtype=torch.float32)

    @staticmethod
    def _copy_sdpo_token_weight_metric_scalars(source: RolloutBatch, target: RolloutBatch) -> None:
        for key, value in source.items():
            if key.startswith(SDPO_TOKEN_WEIGHT_MONITOR_PREFIX + "/") or key.startswith(
                "self_distillation/token_weights_"
            ):
                target[key] = value

    def _compute_sdpo_token_weights(
        self,
        rollout_data: RolloutBatch,
        positive_representations: list[torch.Tensor],
        contrast_representations: list[torch.Tensor | None],
    ) -> tuple[list[torch.Tensor], dict[str, float]]:
        response_lengths = [int(length) for length in rollout_data["response_lengths"]]
        loss_masks = rollout_data["loss_masks"]
        self_distillation_mask = rollout_data.get("self_distillation_mask", [1.0] * len(response_lengths))
        if len(positive_representations) != len(response_lengths):
            raise ValueError("sdpo_teacher_representations row count must match response_lengths.")
        if len(contrast_representations) != len(response_lengths):
            raise ValueError("contrast representation row count must match response_lengths.")

        token_weight_power = float(getattr(self.args, "sdpo_token_weights_power", 1.0))
        token_weight_max = float(getattr(self.args, "sdpo_token_weights_max", 0.0))
        token_weight_cap_enabled = token_weight_power != 0.0 and token_weight_max > 0.0
        local_stats: dict[str, dict[str, float]] = {}
        weight_scores: list[torch.Tensor | None] = [None] * len(response_lengths)
        active_masks: list[torch.Tensor] = []
        active_disagreement_values: list[torch.Tensor] = []
        for row_idx, response_length in enumerate(response_lengths):
            active_mask = loss_masks[row_idx].detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            if active_mask.numel() != response_length:
                raise ValueError(
                    f"loss_masks[{row_idx}] length mismatch for sdpo_token_weights: "
                    f"{active_mask.numel()} != {response_length}."
                )
            active_mask = active_mask * float(self_distillation_mask[row_idx])
            active_masks.append(active_mask)
            contrast = contrast_representations[row_idx]
            if contrast is None or active_mask.sum().item() <= 0.0:
                continue

            positive = positive_representations[row_idx].detach().to(device="cpu", dtype=torch.float32)
            contrast = contrast.detach().to(device="cpu", dtype=torch.float32)
            if positive.shape != contrast.shape:
                raise ValueError(
                    "SDPO token-weight contrast representation shape mismatch at row "
                    f"{row_idx}: {tuple(positive.shape)} != {tuple(contrast.shape)}."
                )
            if positive.size(0) != response_length:
                raise ValueError(
                    f"sdpo_teacher_representations[{row_idx}] response length mismatch: "
                    f"{positive.size(0)} != {response_length}."
                )

            positive_norm = F.normalize(positive, p=2, dim=-1)
            contrast_norm = F.normalize(contrast, p=2, dim=-1)
            disagreement = 1.0 - (positive_norm * contrast_norm).sum(dim=-1)
            disagreement = torch.nan_to_num(disagreement, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            if token_weight_power == 0.0:
                score = torch.ones_like(disagreement)
            elif token_weight_power == 1.0:
                score = disagreement
            elif token_weight_power < 0.0:
                score = disagreement.clamp_min(SDPO_TOKEN_WEIGHT_SCORE_EPS).pow(token_weight_power)
            else:
                score = disagreement.pow(token_weight_power)
            weight_scores[row_idx] = score

            active = active_mask > 0
            if active.any():
                active_disagreement_values.append(disagreement[active])
                uid = self._sdpo_token_weight_uid(rollout_data, row_idx)
                stats = local_stats.setdefault(uid, {"sum": 0.0, "count": 0.0})
                stats["sum"] += float(score[active].sum().item())
                stats["count"] += float(active.sum().item())

        uid_means = self._gather_sdpo_token_weight_uid_means(local_stats)
        token_weights: list[torch.Tensor] = []
        cap_values_by_uid: dict[str, list[torch.Tensor]] = {}
        fallback_uids: set[str] = set()
        for row_idx, response_length in enumerate(response_lengths):
            weights = torch.ones(response_length, dtype=torch.float32)
            score = weight_scores[row_idx]
            if score is not None:
                uid = self._sdpo_token_weight_uid(rollout_data, row_idx)
                uid_mean = uid_means.get(uid, 0.0)
                active = active_masks[row_idx] > 0
                if token_weight_power == 0.0:
                    weights[active] = 1.0
                elif uid_mean > 1e-6:
                    active_weights = score[active] / uid_mean
                    weights[active] = active_weights
                    if token_weight_cap_enabled and active_weights.numel() > 0:
                        cap_values_by_uid.setdefault(uid, []).append(active_weights.detach().cpu())
                elif active.any():
                    fallback_uids.add(uid)
            token_weights.append(weights)

        cap_lambdas: dict[str, float] = {}
        cap_infeasible_uids: set[str] = set()
        if token_weight_cap_enabled and cap_values_by_uid:
            cap_lambdas, cap_infeasible_uids = self._solve_sdpo_token_weight_cap_lambdas(
                cap_values_by_uid,
                token_weight_max,
            )

        local_cap_stats = self._new_sdpo_token_weight_cap_monitor_stats()
        for row_idx, weights in enumerate(token_weights):
            active = active_masks[row_idx] > 0
            if not active.any():
                continue
            if token_weight_cap_enabled:
                uid = self._sdpo_token_weight_uid(rollout_data, row_idx)
                if uid in cap_infeasible_uids:
                    weights[active] = 1.0
                elif uid in cap_lambdas:
                    preclip = weights[active] * cap_lambdas[uid]
                    hit = preclip > token_weight_max
                    capped = torch.clamp(preclip, max=token_weight_max)
                    weights[active] = capped
                    self._accumulate_sdpo_token_weight_cap_monitor_stats(local_cap_stats, preclip, capped, hit)

        active_weight_values = [
            weights[active > 0] for weights, active in zip(token_weights, active_masks, strict=True)
        ]
        active_weight_values = [weights for weights in active_weight_values if weights.numel() > 0]
        active_weights = torch.cat(active_weight_values) if active_weight_values else torch.empty(0)
        active_disagreements = torch.cat(active_disagreement_values) if active_disagreement_values else torch.empty(0)
        monitor_metrics = self._sdpo_token_weight_monitor_metrics(
            active_disagreements,
            active_weights,
            token_weight_power=token_weight_power,
            fallback_uids=fallback_uids,
        )
        cap_metrics = self._sdpo_token_weight_cap_monitor_metrics(
            enabled=token_weight_cap_enabled,
            token_weight_max=token_weight_max,
            local_stats=local_cap_stats,
            lambdas_by_uid=cap_lambdas,
            infeasible_uids=cap_infeasible_uids,
        )

        prefix = SDPO_TOKEN_WEIGHT_MONITOR_PREFIX
        metrics = {
            "self_distillation/token_weights_active_mean": monitor_metrics[f"{prefix}/weights/active_mean"],
            "self_distillation/token_weights_active_min": monitor_metrics[f"{prefix}/weights/active_min"],
            "self_distillation/token_weights_active_max": monitor_metrics[f"{prefix}/weights/active_max"],
            "self_distillation/token_weights_fallback_uid_count": monitor_metrics[f"{prefix}/fallback_uid_count"],
            "self_distillation/token_weights_power": token_weight_power,
            "self_distillation/token_weights_max": token_weight_max,
            f"{SDPO_TOKEN_WEIGHT_MONITOR_PREFIX}/source/teacher_contrast": (
                1.0 if self._sdpo_token_weight_source() == "teacher_contrast" else 0.0
            ),
            f"{SDPO_TOKEN_WEIGHT_MONITOR_PREFIX}/source/student": (
                1.0 if self._sdpo_token_weight_source() == "student" else 0.0
            ),
            f"{SDPO_TOKEN_WEIGHT_MONITOR_PREFIX}/student_representation/reused_actor_forward": (
                float(rollout_data.get("sdpo_student_representation_reused_actor_forward", 0.0))
                if self._sdpo_token_weight_source() == "student"
                else 0.0
            ),
        }
        metrics.update(monitor_metrics)
        metrics.update(cap_metrics)
        return token_weights, metrics

    def _new_sdpo_token_weight_cap_monitor_stats(self) -> dict[str, float]:
        return {
            "token_count": 0.0,
            "hit_count": 0.0,
            "preclip_total_mass": 0.0,
            "preclip_hit_mass": 0.0,
            "postclip_total_mass": 0.0,
            "postclip_hit_mass": 0.0,
        }

    def _accumulate_sdpo_token_weight_cap_monitor_stats(
        self,
        stats: dict[str, float],
        preclip: torch.Tensor,
        postclip: torch.Tensor,
        hit: torch.Tensor,
    ) -> None:
        stats["token_count"] += float(preclip.numel())
        stats["hit_count"] += float(hit.sum().item())
        stats["preclip_total_mass"] += float(preclip.sum().item())
        stats["postclip_total_mass"] += float(postclip.sum().item())
        if hit.any():
            stats["preclip_hit_mass"] += float(preclip[hit].sum().item())
            stats["postclip_hit_mass"] += float(postclip[hit].sum().item())

    def _gather_sdpo_token_weight_cap_sums(
        self,
        values_by_uid: dict[str, list[torch.Tensor]],
        lambdas_by_uid: dict[str, float],
        token_weight_max: float,
    ) -> dict[str, dict[str, float]]:
        local_stats: dict[str, dict[str, float]] = {}
        for uid, rows in values_by_uid.items():
            if not rows:
                continue
            values = torch.cat([row.detach().to(device="cpu", dtype=torch.float32).reshape(-1) for row in rows])
            if values.numel() == 0:
                continue
            lambda_value = float(lambdas_by_uid.get(uid, 1.0))
            clipped = torch.clamp(values * lambda_value, max=token_weight_max)
            local_stats[str(uid)] = {
                "sum": float(clipped.sum().item()),
                "count": float(values.numel()),
                "nonzero_count": float((values > 0.0).sum().item()),
            }

        gathered = [local_stats]
        if dist.is_available() and dist.is_initialized():
            group = get_gloo_group()
            world_size = dist.get_world_size(group=group)
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local_stats, group=group)

        totals: dict[str, dict[str, float]] = {}
        for stats_by_uid in gathered:
            if not stats_by_uid:
                continue
            for uid, stats in stats_by_uid.items():
                total = totals.setdefault(str(uid), {"sum": 0.0, "count": 0.0, "nonzero_count": 0.0})
                total["sum"] += float(stats.get("sum", 0.0))
                total["count"] += float(stats.get("count", 0.0))
                total["nonzero_count"] += float(stats.get("nonzero_count", 0.0))
        return totals

    def _solve_sdpo_token_weight_cap_lambdas(
        self,
        values_by_uid: dict[str, list[torch.Tensor]],
        token_weight_max: float,
    ) -> tuple[dict[str, float], set[str]]:
        tolerance = 1e-6
        stats_at_one = self._gather_sdpo_token_weight_cap_sums(values_by_uid, {}, token_weight_max)
        lambdas = {uid: 1.0 for uid in stats_at_one}
        infeasible_uids = {
            uid
            for uid, stats in stats_at_one.items()
            if stats["count"] > 0.0 and token_weight_max * stats["nonzero_count"] < stats["count"] - tolerance
        }
        pending = {
            uid
            for uid, stats in stats_at_one.items()
            if uid not in infeasible_uids and stats["count"] > 0.0 and stats["sum"] / stats["count"] < 1.0 - tolerance
        }
        if not pending:
            return lambdas, infeasible_uids

        lows = {uid: 0.0 for uid in pending}
        highs = {uid: 1.0 for uid in pending}
        for _ in range(32):
            for uid in pending:
                highs[uid] *= 2.0
            stats = self._gather_sdpo_token_weight_cap_sums(values_by_uid, {**lambdas, **highs}, token_weight_max)
            pending = {
                uid
                for uid in pending
                if stats.get(uid, {}).get("count", 0.0) > 0.0
                and stats[uid]["sum"] / stats[uid]["count"] < 1.0 - tolerance
            }
            if not pending:
                break
        else:
            infeasible_uids.update(pending)
            pending = set()

        solve_uids = set(lows) - infeasible_uids
        for _ in range(24):
            mids = {uid: (lows[uid] + highs[uid]) * 0.5 for uid in solve_uids}
            stats = self._gather_sdpo_token_weight_cap_sums(values_by_uid, {**lambdas, **mids}, token_weight_max)
            for uid in solve_uids:
                if stats.get(uid, {}).get("count", 0.0) <= 0.0:
                    continue
                if stats[uid]["sum"] / stats[uid]["count"] < 1.0:
                    lows[uid] = mids[uid]
                else:
                    highs[uid] = mids[uid]
        for uid in solve_uids:
            lambdas[uid] = highs[uid]
        return lambdas, infeasible_uids

    def _sdpo_token_weight_cap_monitor_metrics(
        self,
        *,
        enabled: bool,
        token_weight_max: float,
        local_stats: dict[str, float],
        lambdas_by_uid: dict[str, float],
        infeasible_uids: set[str],
    ) -> dict[str, float]:
        prefix = f"{SDPO_TOKEN_WEIGHT_MONITOR_PREFIX}/cap"
        gathered = [(dict(local_stats), dict(lambdas_by_uid), set(infeasible_uids))]
        if dist.is_available() and dist.is_initialized():
            group = get_gloo_group()
            world_size = dist.get_world_size(group=group)
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(
                gathered,
                (dict(local_stats), dict(lambdas_by_uid), set(infeasible_uids)),
                group=group,
            )

        totals = self._new_sdpo_token_weight_cap_monitor_stats()
        global_lambdas: dict[str, float] = {}
        global_infeasible_uids: set[str] = set()
        for item in gathered:
            if item is None:
                continue
            stats, lambdas, item_infeasible_uids = item
            for key in totals:
                totals[key] += float(stats.get(key, 0.0))
            global_lambdas.update({str(uid): float(value) for uid, value in lambdas.items()})
            global_infeasible_uids.update(str(uid) for uid in item_infeasible_uids)

        token_count = totals["token_count"]
        preclip_total = totals["preclip_total_mass"]
        postclip_total = totals["postclip_total_mass"]
        lambda_values = [value for uid, value in global_lambdas.items() if uid not in global_infeasible_uids]
        metrics = {
            f"{prefix}/enabled": 1.0 if enabled else 0.0,
            f"{prefix}/max": float(token_weight_max),
            f"{prefix}/hit_token_fraction": totals["hit_count"] / token_count if token_count > 0.0 else 0.0,
            f"{prefix}/preclip_mass_fraction": (
                totals["preclip_hit_mass"] / preclip_total if preclip_total > 0.0 else 0.0
            ),
            f"{prefix}/postclip_mass_fraction": (
                totals["postclip_hit_mass"] / postclip_total if postclip_total > 0.0 else 0.0
            ),
            f"{prefix}/lambda_min": min(lambda_values) if lambda_values else 0.0,
            f"{prefix}/lambda_mean": sum(lambda_values) / len(lambda_values) if lambda_values else 0.0,
            f"{prefix}/lambda_max": max(lambda_values) if lambda_values else 0.0,
            f"{prefix}/infeasible_uid_count": float(len(global_infeasible_uids)),
        }
        return metrics

    @staticmethod
    def _new_sdpo_token_weight_distribution_summary(
        bins: tuple[tuple[str, float | None, float | None], ...],
    ) -> dict:
        return {
            "count": 0.0,
            "sum": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mass_sum": 0.0,
            "bin_counts": {label: 0.0 for label, _lower, _upper in bins},
            "bin_masses": {label: 0.0 for label, _lower, _upper in bins},
        }

    @staticmethod
    def _sdpo_token_weight_distribution_summary(
        values: torch.Tensor,
        bins: tuple[tuple[str, float | None, float | None], ...],
        mass_values: torch.Tensor | None = None,
    ) -> dict:
        values = values.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if mass_values is not None:
            mass_values = mass_values.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            if mass_values.shape != values.shape:
                raise ValueError(
                    "sdpo_token_weights mass_values shape mismatch: "
                    f"{tuple(mass_values.shape)} != {tuple(values.shape)}."
                )

        summary = MegatronTrainRayActor._new_sdpo_token_weight_distribution_summary(bins)
        if values.numel() == 0:
            return summary

        summary["count"] = float(values.numel())
        summary["sum"] = float(values.sum().item())
        summary["min"] = float(values.min().item())
        summary["max"] = float(values.max().item())
        if mass_values is not None:
            summary["mass_sum"] = float(mass_values.sum().item())

        for label, lower, upper in bins:
            mask = torch.ones_like(values, dtype=torch.bool)
            if lower is not None:
                mask &= values >= lower
            if upper is not None:
                mask &= values < upper
            summary["bin_counts"][label] = float(mask.sum().item())
            if mass_values is not None and mask.any():
                summary["bin_masses"][label] = float(mass_values[mask].sum().item())
        return summary

    @staticmethod
    def _merge_sdpo_token_weight_distribution_summaries(
        summaries: list[dict],
        bins: tuple[tuple[str, float | None, float | None], ...],
    ) -> dict:
        merged = MegatronTrainRayActor._new_sdpo_token_weight_distribution_summary(bins)
        has_values = False
        for summary in summaries:
            if not summary:
                continue
            count = float(summary.get("count", 0.0))
            merged["count"] += count
            merged["sum"] += float(summary.get("sum", 0.0))
            merged["mass_sum"] += float(summary.get("mass_sum", 0.0))
            if count > 0.0:
                if not has_values:
                    merged["min"] = float(summary.get("min", 0.0))
                    merged["max"] = float(summary.get("max", 0.0))
                    has_values = True
                else:
                    merged["min"] = min(merged["min"], float(summary.get("min", 0.0)))
                    merged["max"] = max(merged["max"], float(summary.get("max", 0.0)))
            bin_counts = summary.get("bin_counts", {})
            bin_masses = summary.get("bin_masses", {})
            for label, _lower, _upper in bins:
                merged["bin_counts"][label] += float(bin_counts.get(label, 0.0))
                merged["bin_masses"][label] += float(bin_masses.get(label, 0.0))
        return merged

    @staticmethod
    def _approx_sdpo_token_weight_quantile_from_summary(
        summary: dict,
        bins: tuple[tuple[str, float | None, float | None], ...],
        quantile: float,
    ) -> float:
        count = float(summary.get("count", 0.0))
        if count <= 0.0:
            return 0.0
        target = quantile * count
        seen = 0.0
        for label, lower, upper in bins:
            bin_count = float(summary.get("bin_counts", {}).get(label, 0.0))
            if bin_count <= 0.0:
                continue
            next_seen = seen + bin_count
            if next_seen >= target:
                low = float(summary["min"]) if lower is None else float(lower)
                high = float(summary["max"]) if upper is None else float(upper)
                if high <= low:
                    return low
                fraction = min(max((target - seen) / bin_count, 0.0), 1.0)
                return low + fraction * (high - low)
            seen = next_seen
        return float(summary.get("max", 0.0))

    @staticmethod
    def _sdpo_token_weight_distribution_metrics_from_summary(
        prefix: str,
        summary: dict,
        bins: tuple[tuple[str, float | None, float | None], ...],
        *,
        include_mass: bool = False,
    ) -> dict[str, float]:
        count = float(summary.get("count", 0.0))
        metrics = {
            f"{prefix}/active_mean": 0.0,
            f"{prefix}/active_min": 0.0,
            f"{prefix}/active_max": 0.0,
        }
        for name, _quantile in SDPO_TOKEN_WEIGHT_QUANTILES:
            metrics[f"{prefix}/active_{name}"] = 0.0
        for label, _lower, _upper in bins:
            metrics[f"{prefix}/token_frac_{label}"] = 0.0
            if include_mass:
                metrics[f"{prefix}/mass_frac_{label}"] = 0.0
        if count <= 0.0:
            return metrics

        metrics[f"{prefix}/active_mean"] = float(summary.get("sum", 0.0)) / count
        metrics[f"{prefix}/active_min"] = float(summary.get("min", 0.0))
        metrics[f"{prefix}/active_max"] = float(summary.get("max", 0.0))
        for name, quantile in SDPO_TOKEN_WEIGHT_QUANTILES:
            metrics[f"{prefix}/active_{name}"] = MegatronTrainRayActor._approx_sdpo_token_weight_quantile_from_summary(
                summary,
                bins,
                quantile,
            )

        mass_sum = float(summary.get("mass_sum", 0.0))
        bin_counts = summary.get("bin_counts", {})
        bin_masses = summary.get("bin_masses", {})
        for label, _lower, _upper in bins:
            metrics[f"{prefix}/token_frac_{label}"] = float(bin_counts.get(label, 0.0)) / count
            if include_mass:
                metrics[f"{prefix}/mass_frac_{label}"] = (
                    float(bin_masses.get(label, 0.0)) / mass_sum if mass_sum > 0.0 else 0.0
                )
        return metrics

    @staticmethod
    def _sdpo_token_weight_distribution_metrics(
        prefix: str,
        values: torch.Tensor,
        bins: tuple[tuple[str, float | None, float | None], ...],
        mass_values: torch.Tensor | None = None,
    ) -> dict[str, float]:
        values = values.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if mass_values is not None:
            mass_values = mass_values.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            if mass_values.shape != values.shape:
                raise ValueError(
                    "sdpo_token_weights mass_values shape mismatch: "
                    f"{tuple(mass_values.shape)} != {tuple(values.shape)}."
                )

        metrics = {
            f"{prefix}/active_mean": 0.0,
            f"{prefix}/active_min": 0.0,
            f"{prefix}/active_max": 0.0,
        }
        for name, _quantile in SDPO_TOKEN_WEIGHT_QUANTILES:
            metrics[f"{prefix}/active_{name}"] = 0.0
        for label, _lower, _upper in bins:
            metrics[f"{prefix}/token_frac_{label}"] = 0.0
            if mass_values is not None:
                metrics[f"{prefix}/mass_frac_{label}"] = 0.0

        if values.numel() == 0:
            return metrics

        metrics[f"{prefix}/active_mean"] = float(values.mean().item())
        metrics[f"{prefix}/active_min"] = float(values.min().item())
        metrics[f"{prefix}/active_max"] = float(values.max().item())
        quantiles = torch.tensor([q for _name, q in SDPO_TOKEN_WEIGHT_QUANTILES], dtype=torch.float32)
        quantile_values = torch.quantile(values, quantiles)
        for (name, _quantile), value in zip(SDPO_TOKEN_WEIGHT_QUANTILES, quantile_values, strict=True):
            metrics[f"{prefix}/active_{name}"] = float(value.item())

        mass_total = float(mass_values.sum().item()) if mass_values is not None else 0.0
        for label, lower, upper in bins:
            mask = torch.ones_like(values, dtype=torch.bool)
            if lower is not None:
                mask &= values >= lower
            if upper is not None:
                mask &= values < upper
            metrics[f"{prefix}/token_frac_{label}"] = float(mask.float().mean().item())
            if mass_values is not None:
                metrics[f"{prefix}/mass_frac_{label}"] = (
                    float(mass_values[mask].sum().item()) / mass_total if mass_total > 0.0 else 0.0
                )
        return metrics

    def _sdpo_token_weight_monitor_metrics(
        self,
        active_disagreements: torch.Tensor,
        active_weights: torch.Tensor,
        *,
        token_weight_power: float,
        fallback_uids: set[str],
    ) -> dict[str, float]:
        prefix = SDPO_TOKEN_WEIGHT_MONITOR_PREFIX
        active_disagreements = active_disagreements.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        active_weights = active_weights.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if not (dist.is_available() and dist.is_initialized()):
            metrics = {
                f"{prefix}/power": float(token_weight_power),
                f"{prefix}/fallback_uid_count": float(len(fallback_uids)),
                f"{prefix}/disagreement/active_token_count": float(active_disagreements.numel()),
                f"{prefix}/weights/active_token_count": float(active_weights.numel()),
            }
            metrics.update(
                self._sdpo_token_weight_distribution_metrics(
                    f"{prefix}/disagreement",
                    active_disagreements,
                    SDPO_TOKEN_WEIGHT_DISAGREEMENT_BINS,
                )
            )
            metrics.update(
                self._sdpo_token_weight_distribution_metrics(
                    f"{prefix}/weights",
                    active_weights,
                    SDPO_TOKEN_WEIGHT_WEIGHT_BINS,
                    mass_values=active_weights,
                )
            )
            return metrics

        local_payload = {
            "fallback_uids": set(fallback_uids),
            "disagreement": self._sdpo_token_weight_distribution_summary(
                active_disagreements,
                SDPO_TOKEN_WEIGHT_DISAGREEMENT_BINS,
            ),
            "weights": self._sdpo_token_weight_distribution_summary(
                active_weights,
                SDPO_TOKEN_WEIGHT_WEIGHT_BINS,
                mass_values=active_weights,
            ),
        }
        group = get_gloo_group()
        world_size = dist.get_world_size(group=group)
        gathered: list[dict | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_payload, group=group)

        global_fallback_uids: set[str] = set()
        disagreement_summaries: list[dict] = []
        weight_summaries: list[dict] = []
        for payload in gathered:
            if not payload:
                continue
            global_fallback_uids.update(str(uid) for uid in payload.get("fallback_uids", set()))
            disagreement_summaries.append(payload.get("disagreement", {}))
            weight_summaries.append(payload.get("weights", {}))

        disagreement_summary = self._merge_sdpo_token_weight_distribution_summaries(
            disagreement_summaries,
            SDPO_TOKEN_WEIGHT_DISAGREEMENT_BINS,
        )
        weight_summary = self._merge_sdpo_token_weight_distribution_summaries(
            weight_summaries,
            SDPO_TOKEN_WEIGHT_WEIGHT_BINS,
        )
        metrics = {
            f"{prefix}/power": float(token_weight_power),
            f"{prefix}/fallback_uid_count": float(len(global_fallback_uids)),
            f"{prefix}/disagreement/active_token_count": float(disagreement_summary["count"]),
            f"{prefix}/weights/active_token_count": float(weight_summary["count"]),
        }
        metrics.update(
            self._sdpo_token_weight_distribution_metrics_from_summary(
                f"{prefix}/disagreement",
                disagreement_summary,
                SDPO_TOKEN_WEIGHT_DISAGREEMENT_BINS,
            )
        )
        metrics.update(
            self._sdpo_token_weight_distribution_metrics_from_summary(
                f"{prefix}/weights",
                weight_summary,
                SDPO_TOKEN_WEIGHT_WEIGHT_BINS,
                include_mass=True,
            )
        )
        return metrics

    def _gather_sdpo_token_weight_uid_means(self, local_stats: dict[str, dict[str, float]]) -> dict[str, float]:
        gathered = [local_stats]
        if dist.is_available() and dist.is_initialized():
            group = get_gloo_group()
            world_size = dist.get_world_size(group=group)
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local_stats, group=group)

        totals: dict[str, dict[str, float]] = {}
        for stats_by_uid in gathered:
            if not stats_by_uid:
                continue
            for uid, stats in stats_by_uid.items():
                total = totals.setdefault(str(uid), {"sum": 0.0, "count": 0.0})
                total["sum"] += float(stats.get("sum", 0.0))
                total["count"] += float(stats.get("count", 0.0))
        return {uid: stats["sum"] / stats["count"] for uid, stats in totals.items() if stats["count"] > 0.0}

    def _sdpo_token_weight_uid(self, rollout_data: RolloutBatch, row_idx: int) -> str:
        metadata = rollout_data.get("sdpo_metadata")
        if isinstance(metadata, list) and row_idx < len(metadata) and isinstance(metadata[row_idx], dict):
            return str(metadata[row_idx].get("uid", row_idx))
        return str(row_idx)

    def _new_sdpo_teacher_representation_stats(self) -> dict[str, float]:
        return {
            "chunk_count": 0.0,
            "expanded_rows": 0.0,
            "tokens": 0.0,
            "microbatch_count": 0.0,
            "max_microbatch_tokens": 0.0,
            "fill_ratio_sum": 0.0,
            "fill_ratio_count": 0.0,
        }

    def _accumulate_sdpo_teacher_representation_stats(
        self,
        stats: dict[str, float],
        teacher_rollout_data: RolloutBatch,
    ) -> None:
        lengths = [int(length) for length in teacher_rollout_data.get("total_lengths", [])]
        microbatches = teacher_rollout_data.get("micro_batch_indices", []) or []
        microbatch_tokens = [sum(lengths[row_idx] for row_idx in microbatch) for microbatch in microbatches]

        stats["chunk_count"] += 1.0
        stats["expanded_rows"] += float(len(lengths))
        stats["tokens"] += float(sum(lengths))
        stats["microbatch_count"] += float(len(microbatches))
        if microbatch_tokens:
            stats["max_microbatch_tokens"] = max(stats["max_microbatch_tokens"], float(max(microbatch_tokens)))

        if getattr(self.args, "use_dynamic_batch_size", False):
            max_tokens = self._sdpo_teacher_representation_max_tokens_per_gpu()
            cp_size = int(self.train_parallel_config.get("cp_size", mpu.get_context_parallel_world_size()))
            max_tokens *= max(cp_size, 1)
            if max_tokens > 0:
                stats["fill_ratio_sum"] += sum(token_count / max_tokens for token_count in microbatch_tokens)
                stats["fill_ratio_count"] += float(len(microbatch_tokens))

    def _set_sdpo_teacher_representation_stats(
        self,
        rollout_data: RolloutBatch,
        stats: dict[str, float],
    ) -> None:
        fill_ratio_count = stats["fill_ratio_count"]
        avg_fill_ratio = stats["fill_ratio_sum"] / fill_ratio_count if fill_ratio_count > 0 else 0.0
        for key, value in (
            ("sdpo_teacher_representation_chunk_count_per_rank", stats["chunk_count"]),
            ("sdpo_teacher_representation_expanded_rows_per_rank", stats["expanded_rows"]),
            ("sdpo_teacher_representation_tokens_per_rank", stats["tokens"]),
            ("sdpo_teacher_representation_microbatch_count_per_rank", stats["microbatch_count"]),
            ("sdpo_teacher_representation_max_microbatch_tokens_per_rank", stats["max_microbatch_tokens"]),
            ("sdpo_teacher_representation_avg_fill_ratio_per_rank", avg_fill_ratio),
        ):
            rollout_data[key] = torch.tensor(float(value), dtype=torch.float32)

    def _pack_sdpo_teacher_representation_microbatches(
        self,
        lengths: list[int],
        max_tokens: int,
    ) -> list[list[int]]:
        sorted_indices = sorted(range(len(lengths)), key=lambda idx: lengths[idx], reverse=True)
        sorted_lengths = [lengths[idx] for idx in sorted_indices]
        return [sorted(sorted_indices[idx] for idx in packed) for packed in first_fit_pack(sorted_lengths, max_tokens)]

    def _pack_sdpo_teacher_microbatches_in_order(
        self,
        lengths: list[int],
        max_tokens: int,
    ) -> list[list[int]]:
        micro_batch_indices: list[list[int]] = []
        current: list[int] = []
        current_tokens = 0
        for row_idx, length in enumerate(lengths):
            if current and current_tokens + length > max_tokens:
                micro_batch_indices.append(current)
                current = []
                current_tokens = 0
            current.append(row_idx)
            current_tokens += length
            if length > max_tokens:
                micro_batch_indices.append(current)
                current = []
                current_tokens = 0
        if current:
            micro_batch_indices.append(current)
        return micro_batch_indices

    def _repack_sdpo_teacher_microbatches(self, teacher_rollout_data: RolloutBatch) -> None:
        row_count = len(teacher_rollout_data.get("total_lengths", []))
        if row_count == 0:
            teacher_rollout_data["micro_batch_indices"] = []
            teacher_rollout_data["num_microbatches"] = []
            teacher_rollout_data["global_batch_sizes"] = []
            return

        lengths = [int(length) for length in teacher_rollout_data["total_lengths"]]
        micro_batch_indices: list[list[int]] = []
        if getattr(self.args, "use_dynamic_batch_size", False):
            max_tokens = self._sdpo_teacher_representation_max_tokens_per_gpu()
            if max_tokens <= 0:
                raise ValueError(
                    "SDPO teacher precompute requires positive --max-tokens-per-gpu or "
                    "--sdpo-teacher-representation-max-tokens-per-gpu with dynamic batching."
                )
            cp_size = int(self.train_parallel_config.get("cp_size", mpu.get_context_parallel_world_size()))
            max_tokens *= max(cp_size, 1)
            if self._sdpo_uses_representation_distillation():
                micro_batch_indices = self._pack_sdpo_teacher_representation_microbatches(lengths, max_tokens)
            else:
                micro_batch_indices = self._pack_sdpo_teacher_microbatches_in_order(lengths, max_tokens)
        else:
            micro_batch_size = max(int(getattr(self.args, "micro_batch_size", 1) or 1), 1)
            micro_batch_indices = [
                list(range(start, min(start + micro_batch_size, row_count)))
                for start in range(0, row_count, micro_batch_size)
            ]

        teacher_rollout_data["micro_batch_indices"] = micro_batch_indices
        teacher_rollout_data["num_microbatches"] = [len(micro_batch_indices)]
        teacher_rollout_data["global_batch_sizes"] = [row_count]

    def _align_sdpo_teacher_log_probs_to_student_cp(
        self,
        teacher_log_probs: list[torch.Tensor],
        teacher_rollout_data: RolloutBatch,
        rollout_data: RolloutBatch,
    ) -> list[torch.Tensor]:
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            return teacher_log_probs

        aligned_log_probs: list[torch.Tensor] = []
        for i, log_prob in enumerate(teacher_log_probs):
            full_teacher_log_prob = all_gather_with_cp(
                log_prob,
                teacher_rollout_data["total_lengths"][i],
                teacher_rollout_data["response_lengths"][i],
            )
            student_max_seq_len = (
                rollout_data["max_seq_lens"][i]
                if self.args.qkv_format == "bshd" and "max_seq_lens" in rollout_data
                else None
            )
            aligned_log_probs.append(
                slice_log_prob_with_cp(
                    full_teacher_log_prob,
                    rollout_data["total_lengths"][i],
                    rollout_data["response_lengths"][i],
                    self.args.qkv_format,
                    student_max_seq_len,
                )
            )
        return aligned_log_probs

    def _sdpo_needs_topk_distillation(self) -> bool:
        if getattr(self.args, "sdpo_distillation_mode", None) != "topk":
            return False
        return (
            bool(getattr(self.args, "sdpo_full_logit_distillation", True))
            and int(getattr(self.args, "sdpo_distillation_topk", 20)) > 0
        )

    def _build_sgs_plain_view_data(self, step_rollout_data: RolloutBatch) -> RolloutBatch:
        required = (
            "sgs_plain_prompt_text",
            "sgs_plain_messages",
            "sgs_action_token_mask",
        )
        missing = [key for key in required if key not in step_rollout_data]
        if missing:
            raise ValueError(f"SGS teacher scoring data is missing: {missing}.")
        plain_source = dict(step_rollout_data)
        plain_source["sdpo_teacher_prompt_text"] = step_rollout_data["sgs_plain_prompt_text"]
        plain_source["sdpo_teacher_messages"] = step_rollout_data["sgs_plain_messages"]
        plain_source.pop("sdpo_teacher_prompt_texts", None)
        plain_source.pop("sdpo_teacher_messages_list", None)
        return self._build_sdpo_teacher_rollout_data(plain_source)

    def _score_sgs_rows(self, rollout_data: RolloutBatch) -> list[dict[str, object]]:
        """Score on-policy rows and retain the exact tensors needed by selected rows."""
        from slime_plugins.agent_tasks.common.algorithms.sgs import score_masks_for_scope, sgs_option, sgs_ranking_score_field, sgs_score_available, teacher_js_statistics

        self._validate_sdpo_distillation_parallelism()
        score_field = sgs_ranking_score_field(self.args)
        teacher_view_scoring = score_field != "distillation_kl"
        required = [
            "sdpo_teacher_prompt_text",
            "sgs_action_token_mask",
            "sdpo_metadata",
        ]
        if teacher_view_scoring:
            required.extend(("sgs_plain_prompt_text", "sgs_plain_messages"))
        missing = [key for key in required if key not in rollout_data]
        if missing:
            raise ValueError(f"SGS scoring data is missing: {missing}")

        score_scope = str(sgs_option(self.args, "score_scope", "action"))
        scoring_data = dict(rollout_data)
        scoring_data["sgs_action_token_mask"] = score_masks_for_scope(
            rollout_data["loss_masks"],
            rollout_data["sgs_action_token_mask"],
            score_scope=score_scope,
        )

        if self._active_model_tag != "actor":
            self._switch_model("actor")
        student_outputs = self.compute_sdpo_compressed_action_view_data(
            get_data_iterator(scoring_data),
            scoring_data["num_microbatches"],
            store_prefix="sdpo_",
            include_distillation=True,
        )
        rollout_data["sdpo_topk_indices"] = student_outputs["sdpo_sdpo_topk_indices"]
        scoring_data["sdpo_topk_indices"] = student_outputs["sdpo_sdpo_topk_indices"]
        privileged_data = self._build_sdpo_teacher_rollout_data(scoring_data)
        plain_data = self._build_sgs_plain_view_data(scoring_data) if teacher_view_scoring else None
        retention_weight = float(getattr(self.args, "pr_weight", 0.0) or 0.0)
        retention_support = str(getattr(self.args, "pr_support", "selected"))
        retention_direction = str(
            getattr(self.args, "pr_kl_direction", "reverse")
        )
        retention_view = str(getattr(self.args, "pr_view", "privileged"))
        fuse_dense_retention = (
            retention_weight > 0.0 and retention_support == "all" and retention_view == "privileged"
        )
        retention_student_outputs = None
        secondary_topk_indices = None
        if fuse_dense_retention and retention_direction == "reverse":
            retention_student_data = dict(privileged_data)
            for key in SDPO_DISTILLATION_OUTPUT_KEYS:
                retention_student_data.pop(key, None)
            retention_student_outputs = self.compute_sdpo_distillation_data(
                get_data_iterator(retention_student_data),
                retention_student_data["num_microbatches"],
                store_prefix="sdpo_retention_student_",
            )
            secondary_topk_indices = retention_student_outputs[
                "sdpo_retention_student_sdpo_topk_indices"
            ]
        try:
            self._switch_model(self._sdpo_teacher_model_tag())
            if fuse_dense_retention:
                privileged_outputs = self.compute_sdpo_dual_support_compressed_action_view_data(
                    get_data_iterator(privileged_data),
                    privileged_data["num_microbatches"],
                    store_prefix="sdpo_teacher_",
                    secondary_topk_indices=secondary_topk_indices,
                )
            else:
                privileged_outputs = self.compute_sdpo_compressed_action_view_data(
                    get_data_iterator(privileged_data),
                    privileged_data["num_microbatches"],
                    store_prefix="sdpo_teacher_",
                    include_distillation=True,
                )
            plain_outputs = (
                self.compute_sdpo_compressed_action_view_data(
                    get_data_iterator(plain_data),
                    plain_data["num_microbatches"],
                    store_prefix="sdpo_plain_",
                    include_distillation=False,
                )
                if plain_data is not None
                else None
            )
        finally:
            self._switch_model("actor")

        aligned_teacher_log_probs = self._align_sdpo_teacher_log_probs_to_student_cp(
            privileged_outputs["sdpo_teacher_log_probs"], privileged_data, scoring_data
        )
        retention_precomputed = None
        if fuse_dense_retention:
            retention_precomputed = {
                "sdpo_topk_indices": privileged_outputs[
                    "sdpo_teacher_sdpo_retention_topk_indices"
                ],
                "sdpo_teacher_log_probs": aligned_teacher_log_probs,
                "sdpo_teacher_topk_log_probs": privileged_outputs[
                    "sdpo_teacher_sdpo_retention_topk_log_probs"
                ],
            }
        main_scores, _ = self._compute_sdpo_main_divergence_scores(
            scoring_data, student_outputs, privileged_outputs
        )
        rows = []
        for row_idx, metadata in enumerate(rollout_data["sdpo_metadata"]):
            alignment_valid = bool(rollout_data["sgs_action_alignment_valid"][row_idx])
            action_mask = torch.as_tensor(
                rollout_data["sgs_action_token_mask"][row_idx], dtype=torch.bool
            ).reshape(-1)
            action_token_count = int(action_mask.sum().item())
            score_token_mask = torch.as_tensor(
                scoring_data["sgs_action_token_mask"][row_idx], dtype=torch.bool
            ).reshape(-1)
            score_token_count = int(score_token_mask.sum().item())
            teacher_js = None
            teacher_js_sum = None
            kl_plain_privileged = None
            kl_privileged_plain = None
            score_available = sgs_score_available(
                score_scope=score_scope,
                alignment_valid=alignment_valid,
                score_token_count=score_token_count,
            )
            if score_available and plain_outputs is not None:
                ordinary = {
                    "topk_log_probs": plain_outputs["sdpo_plain_sdpo_action_view_topk_log_probs"][row_idx],
                    "tail_log_probs": plain_outputs["sdpo_plain_sdpo_action_view_tail_log_probs"][row_idx],
                }
                privileged = {
                    "topk_log_probs": privileged_outputs[
                        "sdpo_teacher_sdpo_action_view_topk_log_probs"
                    ][row_idx],
                    "tail_log_probs": privileged_outputs[
                        "sdpo_teacher_sdpo_action_view_tail_log_probs"
                    ][row_idx],
                }
                metrics = paired_compressed_view_metrics(ordinary, privileged)
                teacher_js_value, teacher_js_sum_value = teacher_js_statistics(
                    metrics["js"], expected_token_count=score_token_count
                )
                values = (
                    teacher_js_value,
                    teacher_js_sum_value,
                    float(metrics["kl_left_right"].mean().item()),
                    float(metrics["kl_right_left"].mean().item()),
                )
                if all(math.isfinite(value) for value in values):
                    teacher_js, teacher_js_sum, kl_plain_privileged, kl_privileged_plain = values
            ranking_score = main_scores[row_idx] if score_field == "distillation_kl" else {
                "teacher_js": teacher_js,
                "teacher_kl_plain_privileged": kl_plain_privileged,
                "teacher_kl_privileged_plain": kl_privileged_plain,
            }[score_field]
            rows.append(
                {
                    "source_draw_id": int(metadata["sgs_source_draw_id"]),
                    "traj_uid": str(
                        metadata.get("source_trajectory_uid")
                        or metadata.get("frozen_trajectory_uid")
                        or metadata.get("traj_uid", "")
                    ),
                    "turn_idx": int(metadata.get("turn_idx", -1)),
                    "task_id": str(metadata["sgs_task_id"]),
                    "outcome": str(metadata.get("frozen_outcome_label", "")),
                    "trajectory_length": int(metadata.get("frozen_trajectory_length", 0)),
                    "teacher_js": teacher_js,
                    "teacher_js_sum": teacher_js_sum,
                    "score_scope": score_scope,
                    "score_token_mask": score_token_mask.tolist(),
                    "score_token_count": score_token_count,
                    "teacher_kl_plain_privileged": kl_plain_privileged,
                    "teacher_kl_privileged_plain": kl_privileged_plain,
                    "distillation_kl": main_scores[row_idx],
                    "action_token_count": action_token_count,
                    "response_token_count": int(rollout_data["response_lengths"][row_idx]),
                    "score_unavailable_reason": (
                        None
                        if ranking_score is not None
                        else (
                            rollout_data["sgs_action_alignment_reason"][row_idx]
                            if score_scope == "action" and not alignment_valid
                            else "no_score_tokens"
                            if score_token_count == 0
                            else f"non_finite_{score_field}"
                        )
                    ),
                    "precomputed": {
                        "sdpo_topk_indices": student_outputs["sdpo_sdpo_topk_indices"][row_idx].detach().cpu(),
                        "sdpo_teacher_log_probs": aligned_teacher_log_probs[row_idx].detach().cpu(),
                        "sdpo_teacher_topk_log_probs": privileged_outputs[
                            "sdpo_teacher_sdpo_topk_log_probs"
                        ][row_idx]
                        .detach()
                        .cpu(),
                    },
                    "retention_precomputed": (
                        None
                        if retention_precomputed is None
                        else {
                            field: retention_precomputed[field][row_idx].detach().cpu()
                            for field in SDPO_DISTILLATION_OUTPUT_KEYS[:3]
                        }
                    ),
                }
            )
        return rows

    def _sdpo_needs_dense_distillation(self) -> bool:
        if getattr(self.args, "sdpo_distillation_mode", None) != "topk":
            return False
        return (
            bool(getattr(self.args, "sdpo_full_logit_distillation", True))
            and int(getattr(self.args, "sdpo_distillation_topk", 20)) < 0
        )

    def _sdpo_uses_representation_distillation(self) -> bool:
        return getattr(self.args, "sdpo_distillation_mode", None) == "representation"

    def _sdpo_representation_postprocess_allows_skip(self) -> bool:
        if getattr(self, "rollout_data_postprocess", None) is None:
            return True
        return getattr(self.args, "rollout_data_postprocess_path", None) == SDPO_ENTROPY_ROLLOUT_DATA_POSTPROCESS_PATH

    def _sdpo_representation_should_skip_advantage_precompute(self) -> bool:
        if not (
            getattr(self.args, "loss_type", None) == "sdpo_loss" and self._sdpo_uses_representation_distillation()
        ):
            return False
        return not bool(
            getattr(self.args, "kl_coef", 0) != 0
            or getattr(self.args, "use_critic", False)
            or getattr(self.args, "use_opd", False)
            or getattr(self.args, "get_mismatch_metrics", False)
            or getattr(self.args, "use_rollout_logprobs", False)
            or getattr(self.args, "log_correct_samples", False)
            or getattr(self.args, "use_routing_replay", False)
            or not self._sdpo_representation_postprocess_allows_skip()
            or getattr(self.args, "use_tis", False)
            or getattr(self.args, "use_opsm", False)
            or getattr(self.args, "custom_advantage_function_path", None) is not None
            or getattr(self.args, "advantage_estimator", None) == "gspo"
        )

    def _validate_sdpo_distillation_parallelism(self) -> None:
        if self._sdpo_token_weights_enabled():
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            if pp_size != 1:
                raise NotImplementedError(
                    "SDPO token weights require pipeline_model_parallel_size=1; "
                    f"got pipeline_model_parallel_size={pp_size}."
                )
        mode = getattr(self.args, "sdpo_distillation_mode", None)
        if mode == "representation":
            tp_size = mpu.get_tensor_model_parallel_world_size()
            cp_size = mpu.get_context_parallel_world_size()
            if tp_size != 1:
                raise NotImplementedError(
                    "SDPO representation distillation requires tensor_model_parallel_size=1; "
                    f"got tensor_model_parallel_size={tp_size}."
                )
            if cp_size != 1:
                raise NotImplementedError(
                    "SDPO representation distillation requires context_parallel_size=1; "
                    f"got context_parallel_size={cp_size}."
                )
            return
        if not bool(getattr(self.args, "sdpo_full_logit_distillation", True)):
            return
        tp_size = mpu.get_tensor_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()
        if tp_size != 1:
            raise NotImplementedError(
                "SDPO full-logit/top-k distillation requires tensor_model_parallel_size=1; "
                f"got tensor_model_parallel_size={tp_size}."
            )
        if cp_size != 1:
            raise NotImplementedError(
                "SDPO full-logit/top-k distillation requires context_parallel_size=1; "
                f"got context_parallel_size={cp_size}."
            )

    def _step_micro_batch_indices(
        self,
        rollout_data: RolloutBatch,
        *,
        step_id: int,
        num_steps_per_rollout: int,
    ) -> list[list[int]]:
        del num_steps_per_rollout
        num_microbatches = rollout_data["num_microbatches"]
        start = sum(num_microbatches[:step_id])
        end = start + num_microbatches[step_id]
        return rollout_data["micro_batch_indices"][start:end]

    def _sdpo_train_step_row_indices(
        self,
        rollout_data: RolloutBatch,
        *,
        step_id: int,
        num_steps_per_rollout: int,
    ) -> list[int]:
        seen: set[int] = set()
        row_indices: list[int] = []
        for micro_batch in self._step_micro_batch_indices(
            rollout_data,
            step_id=step_id,
            num_steps_per_rollout=num_steps_per_rollout,
        ):
            for row_idx in micro_batch:
                if row_idx not in seen:
                    seen.add(row_idx)
                    row_indices.append(row_idx)
        return row_indices

    def _slice_rollout_rows(self, rollout_data: RolloutBatch, row_indices: list[int]) -> RolloutBatch:
        num_rows = len(rollout_data["total_lengths"])
        subset = {}
        for key, value in rollout_data.items():
            if isinstance(value, list) and len(value) == num_rows:
                subset[key] = [value[i] for i in row_indices]
            elif isinstance(value, tuple) and len(value) == num_rows:
                subset[key] = [value[i] for i in row_indices]
            elif isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == num_rows:
                row_index_tensor = torch.tensor(row_indices, device=value.device, dtype=torch.long)
                subset[key] = value.index_select(0, row_index_tensor)
            else:
                subset[key] = value
        return subset

    def _remap_step_micro_batch_indices(
        self,
        row_indices: list[int],
        step_micro_batch_indices: list[list[int]],
    ) -> list[list[int]]:
        remap = {old_idx: new_idx for new_idx, old_idx in enumerate(row_indices)}
        return [[remap[row_idx] for row_idx in micro_batch] for micro_batch in step_micro_batch_indices]

    def _inject_rollout_rows(self, rollout_data: RolloutBatch, key: str, row_indices: list[int], values) -> None:
        num_rows = len(rollout_data["total_lengths"])
        if key not in rollout_data or rollout_data[key] is None:
            rollout_data[key] = [None] * num_rows
        target = rollout_data[key]
        if not isinstance(target, list) or len(target) != num_rows:
            raise ValueError(f"Cannot inject SDPO row field {key!r}; expected list with {num_rows} rows.")
        if len(values) != len(row_indices):
            raise ValueError(f"SDPO row field {key!r} count mismatch: {len(values)} != {len(row_indices)}.")
        for row_idx, value in zip(row_indices, values, strict=True):
            target[row_idx] = value

    def _ensure_sdpo_teacher_log_probs(self, rollout_data: RolloutBatch) -> None:
        needs_topk = self._sdpo_needs_topk_distillation()
        needs_dense = self._sdpo_needs_dense_distillation()
        if (
            "sdpo_teacher_log_probs" in rollout_data
            and (not needs_topk or "sdpo_teacher_topk_log_probs" in rollout_data)
            and (not needs_dense or "sdpo_teacher_all_log_probs" in rollout_data)
            and (not self._sdpo_token_weights_enabled() or "sdpo_teacher_representations" in rollout_data)
        ):
            return
        if "sdpo_teacher_prompt_text" not in rollout_data:
            raise ValueError("sdpo_teacher_prompt_text is required when loss_type='sdpo_loss'.")
        self._validate_sdpo_distillation_parallelism()

        if needs_topk and "sdpo_topk_indices" not in rollout_data:
            if self._active_model_tag != "actor":
                self._switch_model("actor")
            data_iterator = get_data_iterator(rollout_data)
            num_microbatches = rollout_data["num_microbatches"]
            if len(num_microbatches) != 1:
                raise NotImplementedError(
                    "SDPO top-k distillation with multiple actor train steps must be precomputed per train step."
                )
            student_outputs = self.compute_sdpo_distillation_data(
                data_iterator,
                num_microbatches,
                store_prefix="sdpo_",
            )
            rollout_data["sdpo_topk_indices"] = student_outputs["sdpo_sdpo_topk_indices"]

        teacher_rollout_data = self._build_sdpo_teacher_rollout_data(rollout_data)
        teacher_tag = self._sdpo_teacher_model_tag()
        try:
            self._switch_model(teacher_tag)
            teacher_data_iterator = get_data_iterator(teacher_rollout_data)
            teacher_num_microbatches = teacher_rollout_data["num_microbatches"]
            with self._routing_replay_stage("fallthrough"):
                if self._sdpo_token_weights_enabled():
                    outputs = self.compute_sdpo_distillation_and_representation_data(
                        teacher_data_iterator,
                        teacher_num_microbatches,
                        store_prefix="sdpo_teacher_",
                    )
                else:
                    outputs = self.compute_sdpo_distillation_data(
                        teacher_data_iterator,
                        teacher_num_microbatches,
                        store_prefix="sdpo_teacher_",
                    )
        finally:
            self._switch_model("actor")

        rollout_data["sdpo_teacher_log_probs"] = self._align_sdpo_teacher_log_probs_to_student_cp(
            outputs["sdpo_teacher_log_probs"],
            teacher_rollout_data,
            rollout_data,
        )
        if "sdpo_teacher_sdpo_topk_log_probs" in outputs:
            rollout_data["sdpo_teacher_topk_log_probs"] = outputs["sdpo_teacher_sdpo_topk_log_probs"]
        if "sdpo_teacher_sdpo_all_log_probs" in outputs:
            rollout_data["sdpo_teacher_all_log_probs"] = outputs["sdpo_teacher_sdpo_all_log_probs"]
        if "sdpo_teacher_representations" in outputs:
            rollout_data["sdpo_teacher_representations"] = outputs["sdpo_teacher_representations"]

    def _ensure_sdpo_teacher_representations(self, rollout_data: RolloutBatch) -> None:
        if "sdpo_teacher_representations" in rollout_data:
            return
        if "sdpo_teacher_prompt_text" not in rollout_data:
            raise ValueError("sdpo_teacher_prompt_text is required when loss_type='sdpo_loss'.")
        self._validate_sdpo_distillation_parallelism()

        expanded_rollout_data, expanded_original_indices = self._expand_sdpo_teacher_prompt_ensembles(rollout_data)
        teacher_tag = self._sdpo_teacher_model_tag()
        representation_stats = self._new_sdpo_teacher_representation_stats()

        if expanded_original_indices is None:
            teacher_rollout_data = self._build_sdpo_teacher_rollout_data(expanded_rollout_data)
            self._accumulate_sdpo_teacher_representation_stats(representation_stats, teacher_rollout_data)
            try:
                self._switch_model(teacher_tag)
                teacher_data_iterator = get_data_iterator(teacher_rollout_data)
                teacher_num_microbatches = teacher_rollout_data["num_microbatches"]
                with self._routing_replay_stage("fallthrough"):
                    outputs = self.compute_sdpo_representation_data(
                        teacher_data_iterator,
                        teacher_num_microbatches,
                        store_prefix="sdpo_teacher_",
                    )
            finally:
                self._switch_model("actor")

            rollout_data["sdpo_teacher_representations"] = self._aggregate_sdpo_teacher_representations(
                rollout_data,
                outputs["sdpo_teacher_representations"],
                expanded_original_indices,
            )
            self._set_sdpo_teacher_representation_stats(rollout_data, representation_stats)
            return

        row_count = len(rollout_data["total_lengths"])
        representation_sums: list[torch.Tensor | None] = [None] * row_count
        representation_counts = [0] * row_count
        stat_sums, stat_counts, stat_maxes = self._new_sdpo_teacher_prompt_stat_accumulators(row_count)
        chunk_rows = self._sdpo_teacher_representation_forward_chunk_rows()

        try:
            self._switch_model(teacher_tag)
            with self._routing_replay_stage("fallthrough"):
                for start in range(0, len(expanded_original_indices), chunk_rows):
                    end = min(start + chunk_rows, len(expanded_original_indices))
                    chunk_row_indices = list(range(start, end))
                    chunk_original_indices = expanded_original_indices[start:end]
                    chunk_rollout_data = self._slice_rollout_rows(expanded_rollout_data, chunk_row_indices)
                    teacher_rollout_data = self._build_sdpo_teacher_rollout_data(chunk_rollout_data)
                    self._accumulate_sdpo_teacher_representation_stats(representation_stats, teacher_rollout_data)
                    self._accumulate_sdpo_teacher_prompt_stats(
                        stat_sums,
                        stat_counts,
                        stat_maxes,
                        teacher_rollout_data,
                        chunk_original_indices,
                    )
                    teacher_data_iterator = get_data_iterator(teacher_rollout_data)
                    teacher_num_microbatches = teacher_rollout_data["num_microbatches"]
                    outputs = self.compute_sdpo_representation_data(
                        teacher_data_iterator,
                        teacher_num_microbatches,
                        store_prefix="sdpo_teacher_",
                    )
                    self._accumulate_sdpo_teacher_representation_sums(
                        representation_sums,
                        representation_counts,
                        outputs["sdpo_teacher_representations"],
                        chunk_original_indices,
                    )
        finally:
            self._switch_model("actor")

        self._set_aggregated_sdpo_teacher_prompt_stats(rollout_data, stat_sums, stat_counts, stat_maxes)
        rollout_data["sdpo_teacher_representations"] = self._finalize_sdpo_teacher_representation_sums(
            representation_sums,
            representation_counts,
        )
        self._set_sdpo_teacher_representation_stats(rollout_data, representation_stats)

    def _compute_sdpo_main_divergence_scores(
        self,
        step_rollout_data: RolloutBatch,
        student_outputs: dict[str, list[torch.Tensor]],
        teacher_outputs: dict[str, list[torch.Tensor]],
    ) -> tuple[list[float | None], list[int]]:
        """Measure the unweighted SDPO divergence on the actual main-loss tokens."""
        student_rows = student_outputs["sdpo_sdpo_topk_log_probs"]
        teacher_rows = teacher_outputs["sdpo_teacher_sdpo_topk_log_probs"]
        loss_masks = step_rollout_data["loss_masks"]
        sample_masks = step_rollout_data.get("self_distillation_mask", [1] * len(student_rows))
        scores: list[float | None] = []
        active_counts: list[int] = []
        for student_row, teacher_row, loss_mask, sample_mask in zip(
            student_rows,
            teacher_rows,
            loss_masks,
            sample_masks,
            strict=True,
        ):
            active_mask = torch.as_tensor(loss_mask, device=student_row.device, dtype=torch.bool)
            if not bool(torch.as_tensor(sample_mask).item()):
                active_mask = torch.zeros_like(active_mask)
            active_count = int(active_mask.sum().item())
            active_counts.append(active_count)
            if (
                active_count == 0
                or active_mask.numel() != student_row.size(0)
                or student_row.shape != teacher_row.shape
            ):
                scores.append(None)
                continue
            token_kl = compute_sdpo_topk_token_kl(
                student_row[active_mask].detach(),
                teacher_row[active_mask].detach().to(device=student_row.device, dtype=student_row.dtype),
                alpha=float(getattr(self.args, "sdpo_alpha", 1.0)),
                add_tail=bool(getattr(self.args, "sdpo_distillation_add_tail", True)),
            )
            score = float(token_kl.detach().mean().to(device="cpu", dtype=torch.float32).item())
            scores.append(score if math.isfinite(score) and score >= 0.0 else None)
        return scores, active_counts

    def _precompute_sdpo_topk_for_train_step(
        self,
        rollout_data: RolloutBatch,
        *,
        step_id: int,
        num_steps_per_rollout: int,
        finalize_token_weights: bool = True,
    ) -> None:
        if not self._sdpo_needs_topk_distillation():
            return
        self._validate_sdpo_distillation_parallelism()
        if "sdpo_teacher_prompt_text" not in rollout_data:
            raise ValueError("sdpo_teacher_prompt_text is required when loss_type='sdpo_loss'.")

        row_indices = self._sdpo_train_step_row_indices(
            rollout_data,
            step_id=step_id,
            num_steps_per_rollout=num_steps_per_rollout,
        )
        step_micro_batch_indices = self._step_micro_batch_indices(
            rollout_data,
            step_id=step_id,
            num_steps_per_rollout=num_steps_per_rollout,
        )
        step_rollout_data = self._slice_rollout_rows(rollout_data, row_indices)
        step_rollout_data["micro_batch_indices"] = self._remap_step_micro_batch_indices(
            row_indices,
            step_micro_batch_indices,
        )
        step_rollout_data["num_microbatches"] = [len(step_rollout_data["micro_batch_indices"])]
        step_rollout_data["global_batch_sizes"] = [len(row_indices)]

        student_rollout_data = dict(step_rollout_data)
        for key in SDPO_DISTILLATION_OUTPUT_KEYS:
            student_rollout_data.pop(key, None)
        if self._active_model_tag != "actor":
            self._switch_model("actor")
        student_data_iterator = get_data_iterator(student_rollout_data)
        student_num_microbatches = student_rollout_data["num_microbatches"]
        with self._routing_replay_stage("fallthrough"):
            student_outputs = self.compute_sdpo_distillation_data(
                student_data_iterator,
                student_num_microbatches,
                store_prefix="sdpo_",
            )
        step_rollout_data["sdpo_topk_indices"] = student_outputs["sdpo_sdpo_topk_indices"]
        self._inject_rollout_rows(
            rollout_data,
            "sdpo_topk_indices",
            row_indices,
            student_outputs["sdpo_sdpo_topk_indices"],
        )
        teacher_rollout_data = self._build_sdpo_teacher_rollout_data(step_rollout_data)
        for key in (
            "sdpo_teacher_prompt_token_lengths",
            "sdpo_teacher_prompt_token_lengths_raw",
            "sdpo_teacher_prompt_truncated",
            "sdpo_teacher_prompt_char_lengths",
        ):
            if key in step_rollout_data:
                self._inject_rollout_rows(rollout_data, key, row_indices, step_rollout_data[key])
        if "actor" in self.weights_backuper.backup_tags:
            self.weights_backuper.backup("actor")
        teacher_tag = self._sdpo_teacher_model_tag()
        try:
            self._switch_model(teacher_tag)
            teacher_data_iterator = get_data_iterator(teacher_rollout_data)
            teacher_num_microbatches = teacher_rollout_data["num_microbatches"]
            with self._routing_replay_stage("fallthrough"):
                if self._sdpo_token_weights_enabled():
                    teacher_outputs = self.compute_sdpo_distillation_and_representation_data(
                        teacher_data_iterator,
                        teacher_num_microbatches,
                        store_prefix="sdpo_teacher_",
                    )
                else:
                    teacher_outputs = self.compute_sdpo_distillation_data(
                        teacher_data_iterator,
                        teacher_num_microbatches,
                        store_prefix="sdpo_teacher_",
                    )
        finally:
            self._switch_model("actor")

        self._inject_rollout_rows(
            rollout_data,
            "sdpo_teacher_log_probs",
            row_indices,
            self._align_sdpo_teacher_log_probs_to_student_cp(
                teacher_outputs["sdpo_teacher_log_probs"],
                teacher_rollout_data,
                step_rollout_data,
            ),
        )
        if "sdpo_teacher_sdpo_topk_log_probs" in teacher_outputs:
            self._inject_rollout_rows(
                rollout_data,
                "sdpo_teacher_topk_log_probs",
                row_indices,
                teacher_outputs["sdpo_teacher_sdpo_topk_log_probs"],
            )
        if "sdpo_teacher_sdpo_all_log_probs" in teacher_outputs:
            self._inject_rollout_rows(
                rollout_data,
                "sdpo_teacher_all_log_probs",
                row_indices,
                teacher_outputs["sdpo_teacher_sdpo_all_log_probs"],
            )
        if "sdpo_teacher_representations" in teacher_outputs:
            self._inject_rollout_rows(
                rollout_data,
                "sdpo_teacher_representations",
                row_indices,
                teacher_outputs["sdpo_teacher_representations"],
            )
            if finalize_token_weights and self._sdpo_token_weights_enabled():
                if not self._sdpo_teacher_representations_complete(rollout_data):
                    raise ValueError(
                        "Cannot finalize SDPO token weights before all teacher representations are ready."
                    )
                self._ensure_sdpo_token_weights(rollout_data)

    def _precompute_sdpo_topk_for_all_train_steps(
        self,
        rollout_data: RolloutBatch,
        *,
        num_steps_per_rollout: int,
    ) -> None:
        for step_id in range(num_steps_per_rollout):
            self._precompute_sdpo_topk_for_train_step(
                rollout_data,
                step_id=step_id,
                num_steps_per_rollout=num_steps_per_rollout,
                finalize_token_weights=False,
            )
        self._ensure_sdpo_token_weights(rollout_data)

    def _use_sdpo_ema_teacher(self) -> bool:
        return (
            getattr(self.args, "loss_type", None) == "sdpo_loss"
            and getattr(self.args, "sdpo_teacher_regularization", "ema") == "ema"
        )

    def _uses_per_update_sdpo_old_actor(self) -> bool:
        """Whether old_actor tracks the immediately preceding optimizer update."""

        return (
            bool(getattr(self.args, "offline_async_eval", False))
            and bool(getattr(self.args, "keep_old_actor", False))
            and getattr(self.args, "loss_type", None) == "sdpo_loss"
            and (
                getattr(self.args, "sdpo_clip_ratio", None) is not None
                or getattr(self.args, "sdpo_deployment_tis_clip", None) is not None
            )
        )

    def _needs_sdpo_correction_log_probs(self) -> bool:
        if getattr(self.args, "loss_type", None) != "sdpo_loss":
            return True
        return bool(
            getattr(self.args, "sdpo_clip_ratio", None) is not None
            or getattr(self.args, "sdpo_deployment_tis_clip", None) is not None
            or getattr(self.args, "kl_coef", 0) != 0
            or getattr(self.args, "use_kl_loss", False)
            or getattr(self.args, "use_opd", False)
            or getattr(self.args, "use_tis", False)
            or getattr(self.args, "use_opsm", False)
            or getattr(self.args, "log_correct_samples", False)
            or getattr(self.args, "custom_advantage_function_path", None) is not None
        )

    def _should_update_old_actor_during_weight_sync(self) -> bool:
        return (
            bool(getattr(self.args, "keep_old_actor", False))
            and not bool(getattr(self.args, "offline_train_eval_colocate", False))
            and not self._uses_per_update_sdpo_old_actor()
        )

    def _advance_online_old_actor_after_weight_sync(self) -> bool:
        """Advance the online policy queue after the initial no-op synchronization."""

        if not getattr(self, "_online_old_actor_weight_sync_initialized", False):
            required = {"old_actor", "rollout_actor"}
            missing = required - set(self.weights_backuper.backup_tags)
            if missing:
                raise RuntimeError(f"online old-actor queue is missing initial backups: {sorted(missing)}")
            self._online_old_actor_weight_sync_initialized = True
            return False
        self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
        # _refresh_actor_backups_after_update captured the successful optimizer
        # result already. Keep this queue movement CPU-to-CPU while the colocated
        # model is in the weight-sync memory-saver transition.
        self.weights_backuper.copy(src_tag="actor", dst_tag="rollout_actor")
        return True

    def _refresh_actor_backups_after_update(self) -> bool:
        """Capture a successful live update, then advance any per-update old policy."""

        self.weights_backuper.backup("actor")
        should_refresh_old = (
            bool(getattr(self.args, "offline_train_eval_colocate", False))
            and bool(getattr(self.args, "keep_old_actor", False))
        ) or self._uses_per_update_sdpo_old_actor()
        if should_refresh_old:
            self.weights_backuper.copy(src_tag="actor", dst_tag="old_actor")
        self._update_sdpo_ema_teacher()
        return should_refresh_old

    def _sdpo_teacher_model_tag(self) -> str:
        teacher_regularization = getattr(self.args, "sdpo_teacher_regularization", "ema")
        if teacher_regularization == "ema":
            if "sdpo_teacher" not in self.weights_backuper.backup_tags:
                raise ValueError("SDPO EMA teacher backup is missing.")
            return "sdpo_teacher"
        if teacher_regularization == "ref":
            if "ref" not in self.weights_backuper.backup_tags:
                raise ValueError("SDPO ref teacher requested but no 'ref' backup is loaded.")
            return "ref"
        if teacher_regularization == "actor":
            return "actor"
        raise ValueError(f"Unsupported sdpo_teacher_regularization={teacher_regularization!r}.")

    def _update_sdpo_ema_teacher(self) -> None:
        if not self._use_sdpo_ema_teacher():
            return
        update_rate = float(getattr(self.args, "sdpo_teacher_update_rate", 0.01))
        if update_rate == 0.0:
            return
        if "actor" not in self.weights_backuper.backup_tags:
            raise ValueError("Cannot update SDPO EMA teacher because actor backup is missing.")
        if "sdpo_teacher" not in self.weights_backuper.backup_tags:
            raise ValueError("Cannot update SDPO EMA teacher because sdpo_teacher backup is missing.")
        actor_weights = self.weights_backuper.get("actor")
        teacher_weights = self.weights_backuper.get("sdpo_teacher")
        missing = set(teacher_weights) ^ set(actor_weights)
        if missing:
            raise ValueError(f"SDPO EMA teacher/actor backup key mismatch: {sorted(missing)[:5]}")
        with torch.no_grad():
            for name, teacher_tensor in teacher_weights.items():
                teacher_tensor.mul_(1.0 - update_rate).add_(actor_weights[name], alpha=update_rate)

    def _rank_for_sidecar(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    def _sdpo_ema_teacher_sidecar_path(self, checkpoint_root: str | os.PathLike, iteration: int) -> Path:
        root = Path(checkpoint_root)
        if root.name.startswith("iter_"):
            root = root.parent
        return root / "sdpo_ema_teacher" / f"iter_{iteration:07d}" / f"rank_{self._rank_for_sidecar():05d}.pt"

    def _maybe_load_sdpo_ema_teacher(self, loaded_rollout_id: int) -> None:
        if not self._use_sdpo_ema_teacher() or loaded_rollout_id < 0:
            return
        path = self._sdpo_ema_teacher_sidecar_path(self.args.load, loaded_rollout_id)
        if not path.is_file():
            sidecar_dir = path.parent
            if sidecar_dir.exists():
                raise FileNotFoundError(
                    f"SDPO EMA teacher sidecar missing for rank {self._rank_for_sidecar()}: {path}"
                )
            if loaded_rollout_id == 0:
                if self._looks_like_run_checkpoint_root(self.args.load):
                    raise FileNotFoundError(f"SDPO EMA teacher sidecar missing for checkpoint iteration 0: {path}")
                return
            if is_megatron_main_rank():
                logger.warning(
                    "SDPO EMA teacher sidecar not found at %s; initializing EMA teacher from loaded actor weights.",
                    path,
                )
            return
        self.weights_backuper.load("sdpo_teacher", path)
        if is_megatron_main_rank():
            logger.info("Loaded SDPO EMA teacher sidecar from %s", path)

    def _looks_like_run_checkpoint_root(self, checkpoint_root: str | os.PathLike) -> bool:
        root = Path(checkpoint_root)
        if root.name.startswith("iter_"):
            root = root.parent
        run_root = root.parent
        return (
            (root / "best_checkpoint_tracker.json").exists()
            or (root / "rollout").exists()
            or (run_root / "alfworld_config.yaml").exists()
            or (run_root / "logs").exists()
        )

    def _save_sdpo_ema_teacher(self, rollout_id: int) -> None:
        if not self._use_sdpo_ema_teacher():
            return
        if "sdpo_teacher" not in self.weights_backuper.backup_tags:
            raise ValueError("Cannot save SDPO EMA teacher because sdpo_teacher backup is missing.")
        if self.args.save is None:
            raise ValueError("Cannot save SDPO EMA teacher sidecar because --save is not set.")
        path = self._sdpo_ema_teacher_sidecar_path(self.args.save, rollout_id)
        self.weights_backuper.save("sdpo_teacher", path)

    def _prune_sdpo_ema_teacher_sidecars_fallback(self) -> None:
        if not self._use_sdpo_ema_teacher():
            return
        if getattr(self.args, "checkpoint_retention_policy", "none") == "latest_and_best_eval":
            return
        keep_last = int(getattr(self.args, "sdpo_ema_teacher_keep_last", 1))
        if keep_last <= 0 or not is_megatron_main_rank() or self.args.save is None:
            return
        sidecar_root = Path(self.args.save) / "sdpo_ema_teacher"
        if not sidecar_root.is_dir():
            return

        iter_dirs: list[tuple[int, Path]] = []
        for path in sidecar_root.iterdir():
            if not path.is_dir() or not path.name.startswith("iter_"):
                continue
            try:
                iteration = int(path.name.removeprefix("iter_"))
            except ValueError:
                continue
            iter_dirs.append((iteration, path))

        iter_dirs.sort(key=lambda item: item[0])
        for _, path in iter_dirs[:-keep_last]:
            shutil.rmtree(path, ignore_errors=True)
            logger.info("Pruned old SDPO EMA teacher sidecar checkpoint at %s", path)

    def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):
        if self.args.debug_rollout_only:
            return None

        runner_managed_offload = bool(self.args.offline_train_eval_colocate)
        reuse_sgs_residency = self._consume_sgs_prepass_residency()
        if self.args.offload_train and not runner_managed_offload and not reuse_sgs_residency:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        if self.role == "critic":
            result = self.train_critic(rollout_id, rollout_data)
        else:
            result = self.train_actor(rollout_id, rollout_data, external_data=external_data)

        if self.args.offload_train and not runner_managed_offload:
            del rollout_data
            self.sleep()

        return result

    def score_entropy(self, rollout_id: int, rollout_data_ref: Box, phase: str):
        if not rollout_data_ref:
            return {"records": [], "expected_count": 0}
        if self.args.offload_train:
            self.wake_up()

        with timer(f"{phase}_entropy_score"):
            rollout_data = self._get_rollout_data(rollout_data_ref)
            if not rollout_data.get("tokens"):
                result = {"records": [], "expected_count": 0, "expected_by_task": {}}
            else:
                data_iterator = get_data_iterator(rollout_data)
                num_microbatches = rollout_data["num_microbatches"]
                log_prob_data_iterator, log_prob_num_microbatches = self._get_log_prob_data_iterator(
                    rollout_data,
                    data_iterator,
                    num_microbatches,
                )
                if self._active_model_tag != "actor":
                    self._switch_model("actor")
                temperature = _actor_entropy_temperature(self.args, phase)
                rollout_data.update(
                    self.compute_log_prob(
                        log_prob_data_iterator,
                        log_prob_num_microbatches,
                        store_prefix="",
                        with_entropy=True,
                        temperature=temperature,
                    )
                )
                from slime_plugins.agent_tasks.common.entropy_postprocess import (
                    entropy_expected_counts_by_task,
                    entropy_records_from_rollout_data,
                )

                records, expected_count = entropy_records_from_rollout_data(
                    self.args,
                    rollout_data,
                    source=f"{phase}_actor_forward",
                    temperature=temperature,
                )
                expected_by_task = entropy_expected_counts_by_task(rollout_data)
                result = {"records": records, "expected_count": expected_count, "expected_by_task": expected_by_task}

        if self.args.offload_train:
            self.sleep()
        return result

    def score_sgs(self, rollout_id: int, rollout_data_ref: Box):
        from slime_plugins.agent_tasks.common.algorithms.sgs import sgs_selection_fraction

        if sgs_selection_fraction(self.args) is None:
            raise RuntimeError("SGS scoring was called without an enabled fraction")
        runner_managed_offload = bool(getattr(self.args, "offline_train_eval_colocate", False))
        manage_residency = bool(self.args.offload_train) and not runner_managed_offload
        if manage_residency:
            self.wake_up()
        started = time.monotonic()
        try:
            rollout_data = self._get_rollout_data(rollout_data_ref)
            rows = self._score_sgs_rows(rollout_data)
        except BaseException:
            if manage_residency:
                self.sleep()
            raise
        # The selected-row train call follows synchronously. Keeping the
        # actor resident avoids an otherwise redundant offload/onload pair;
        # train() consumes this one-shot state and performs its normal final
        # offload after the optimizer step.
        self._sgs_prepass_kept_awake = manage_residency
        return {
            "rollout_id": int(rollout_id),
            "rank": int(dist.get_rank()),
            "duration_seconds": time.monotonic() - started,
            "rows": rows,
        }

    def _consume_sgs_prepass_residency(self) -> bool:
        reuse = bool(getattr(self, "_sgs_prepass_kept_awake", False))
        self._sgs_prepass_kept_awake = False
        return reuse

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch):
        """Train critic and return CPU values (used as old-values for the next actor train)."""
        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["global_batch_sizes"]

        # Compute current critic values (used as old_values for value loss and for actor advantages).
        rollout_data.update(forward_only(get_values, self.args, self.model, data_iterator, num_microbatches))

        compute_advantages_and_returns(self.args, rollout_data)

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
            global_batch_sizes,
        )

        if mpu.is_pipeline_last_stage() and "values" in rollout_data:
            from slime.backends.megatron_utils.data import tensors_to_cpu

            return {"values": tensors_to_cpu(rollout_data["values"])}
        return {}

    def _ensure_unscored_random_selection_targets(self, rollout_data: RolloutBatch) -> None:
        if not (
            str(getattr(self.args, "sgs_selection_mode", "sensitivity")) == "random"
            and bool(getattr(self.args, "sgs_skip_full_scoring", False))
        ):
            return
        components = rollout_data.get("pr_component")
        if not isinstance(components, list) or len(components) != len(rollout_data["tokens"]):
            raise ValueError("unscored random retention requires a row-aligned component mask")
        row_indices = [idx for idx, value in enumerate(components) if float(value) == 0.0]
        if not row_indices:
            return
        target_fields = (
            "sdpo_topk_indices",
            "sdpo_teacher_log_probs",
            "sdpo_teacher_topk_log_probs",
        )
        field_states = []
        for field in target_fields:
            rows = rollout_data.get(field)
            if not isinstance(rows, list) or len(rows) != len(components):
                field_states.append("missing")
                continue
            present = [rows[index] is not None for index in row_indices]
            if any(present) and not all(present):
                raise ValueError("unscored random selection targets are only partially cached")
            field_states.append("cached" if all(present) else "missing")
        if all(state == "cached" for state in field_states):
            return
        if any(state == "cached" for state in field_states):
            raise ValueError("unscored random selection target fields are only partially cached")

        selection_data = self._slice_rollout_rows(rollout_data, row_indices)
        for key in SDPO_DISTILLATION_OUTPUT_KEYS:
            selection_data.pop(key, None)
        self._repack_sdpo_teacher_microbatches(selection_data)
        if self._active_model_tag != "actor":
            self._switch_model("actor")
        with self._routing_replay_stage("fallthrough"):
            student_outputs = self.compute_sdpo_distillation_data(
                get_data_iterator(selection_data),
                selection_data["num_microbatches"],
                store_prefix="sdpo_random_selection_student_",
            )
        support_indices = student_outputs["sdpo_random_selection_student_sdpo_topk_indices"]
        selection_data["sdpo_topk_indices"] = support_indices
        teacher_data = self._build_sdpo_teacher_rollout_data(selection_data)
        try:
            self._switch_model(self._sdpo_teacher_model_tag())
            with self._routing_replay_stage("fallthrough"):
                teacher_outputs = self.compute_sdpo_distillation_data(
                    get_data_iterator(teacher_data),
                    teacher_data["num_microbatches"],
                    store_prefix="sdpo_random_selection_teacher_",
                )
        finally:
            self._switch_model("actor")
        self._inject_rollout_rows(rollout_data, "sdpo_topk_indices", row_indices, support_indices)
        self._inject_rollout_rows(
            rollout_data,
            "sdpo_teacher_log_probs",
            row_indices,
            self._align_sdpo_teacher_log_probs_to_student_cp(
                teacher_outputs["sdpo_random_selection_teacher_log_probs"],
                teacher_data,
                selection_data,
            ),
        )
        self._inject_rollout_rows(
            rollout_data,
            "sdpo_teacher_topk_log_probs",
            row_indices,
            teacher_outputs["sdpo_random_selection_teacher_sdpo_topk_log_probs"],
        )

    def _ensure_pr_targets(self, rollout_data: RolloutBatch) -> None:
        weight = float(getattr(self.args, "pr_weight", 0.0) or 0.0)
        if weight <= 0.0:
            return
        components = rollout_data.get("pr_component")
        if not isinstance(components, list) or len(components) != len(rollout_data["tokens"]):
            raise ValueError("PR retention rows require a row-aligned component mask")
        self._ensure_unscored_random_selection_targets(rollout_data)
        row_indices = [idx for idx, value in enumerate(components) if float(value) == 1.0]
        if not row_indices:
            # Global dynamic-batch balancing may place all retention rows on
            # other data-parallel ranks. This rank still contributes its
            # ordinary rows; the shared Megatron normalizer/all-reduce
            # composes the globally paired objective.
            rollout_data["self_distillation/retention/rows_per_rank"] = torch.tensor(
                0.0, dtype=torch.float32
            )
            return

        target_fields = (
            "sdpo_topk_indices",
            "sdpo_teacher_log_probs",
            "sdpo_teacher_topk_log_probs",
        )
        cached = []
        for field in target_fields:
            rows = rollout_data.get(field)
            cached.append(
                isinstance(rows, list)
                and len(rows) == len(components)
                and all(rows[index] is not None for index in row_indices)
            )
        if all(cached):
            rollout_data["self_distillation/retention/rows_per_rank"] = torch.tensor(
                float(len(row_indices)), dtype=torch.float32
            )
            return
        if any(cached):
            raise ValueError("PR retention targets are only partially cached")

        retention_data = self._slice_rollout_rows(rollout_data, row_indices)
        for key in SDPO_DISTILLATION_OUTPUT_KEYS:
            retention_data.pop(key, None)
        self._repack_sdpo_teacher_microbatches(retention_data)

        direction = str(getattr(self.args, "pr_kl_direction", "reverse"))
        if direction not in {"reverse", "forward"}:
            raise ValueError(f"unsupported privileged retention KL direction: {direction}")
        support_indices = None
        if direction == "reverse":
            if self._active_model_tag != "actor":
                self._switch_model("actor")
            with self._routing_replay_stage("fallthrough"):
                student_outputs = self.compute_sdpo_distillation_data(
                    get_data_iterator(retention_data),
                    retention_data["num_microbatches"],
                    store_prefix="sdpo_retention_student_",
                )
            support_indices = student_outputs["sdpo_retention_student_sdpo_topk_indices"]

        teacher_data = dict(retention_data)
        if support_indices is not None:
            teacher_data["sdpo_topk_indices"] = support_indices
        try:
            self._switch_model(self._sdpo_teacher_model_tag())
            with self._routing_replay_stage("fallthrough"):
                teacher_outputs = self.compute_sdpo_distillation_data(
                    get_data_iterator(teacher_data),
                    teacher_data["num_microbatches"],
                    store_prefix="sdpo_retention_teacher_",
                )
        finally:
            self._switch_model("actor")

        if direction == "forward":
            support_indices = teacher_outputs["sdpo_retention_teacher_sdpo_topk_indices"]

        self._inject_rollout_rows(rollout_data, "sdpo_topk_indices", row_indices, support_indices)
        self._inject_rollout_rows(
            rollout_data,
            "sdpo_teacher_log_probs",
            row_indices,
            self._align_sdpo_teacher_log_probs_to_student_cp(
                teacher_outputs["sdpo_retention_teacher_log_probs"],
                teacher_data,
                retention_data,
            ),
        )
        self._inject_rollout_rows(
            rollout_data,
            "sdpo_teacher_topk_log_probs",
            row_indices,
            teacher_outputs["sdpo_retention_teacher_sdpo_topk_log_probs"],
        )
        rollout_data["self_distillation/retention/rows_per_rank"] = torch.tensor(
            float(len(row_indices)), dtype=torch.float32
        )

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch, external_data=None) -> dict:
        self._ensure_pr_targets(rollout_data)
        # Create data iterator for log_probs and train.
        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["global_batch_sizes"]

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        need_agent_task_rollout_entropy = _agent_task_rollout_entropy_enabled(self.args)
        skip_sdpo_representation_advantage_precompute = (
            self.args.compute_advantages_and_returns and self._sdpo_representation_should_skip_advantage_precompute()
        )
        need_sdpo_student_representations = self._sdpo_token_weight_source() == "student"

        with inverse_timer("train_wait"), timer("train"):
            if skip_sdpo_representation_advantage_precompute:
                if need_agent_task_rollout_entropy or need_sdpo_student_representations:
                    self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                    log_prob_data_iterator, log_prob_num_microbatches = self._get_log_prob_data_iterator(
                        rollout_data,
                        data_iterator,
                        num_microbatches,
                    )
                    entropy_outputs = self.compute_log_prob(
                        log_prob_data_iterator,
                        log_prob_num_microbatches,
                        store_prefix="",
                        with_entropy=need_agent_task_rollout_entropy or None,
                        temperature=getattr(self.args, "rollout_temperature", 1.0),
                        collect_sdpo_student_representations=need_sdpo_student_representations,
                    )
                    entropy_outputs.pop("log_probs", None)
                    rollout_data.update(entropy_outputs)
                    if need_sdpo_student_representations:
                        rollout_data["sdpo_student_representation_reused_actor_forward"] = torch.tensor(
                            float(need_agent_task_rollout_entropy), dtype=torch.float32
                        )
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

            elif self.args.compute_advantages_and_returns:
                need_ref_log_probs = (
                    getattr(self.args, "kl_coef", 0) != 0
                    or getattr(self.args, "use_kl_loss", False)
                    or getattr(self.args, "custom_advantage_function_path", None) is not None
                )
                if "ref" in self.weights_backuper.backup_tags and need_ref_log_probs:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("ref")
                    log_prob_data_iterator, log_prob_num_microbatches = self._get_log_prob_data_iterator(
                        rollout_data,
                        data_iterator,
                        num_microbatches,
                    )
                    rollout_data.update(
                        self.compute_log_prob(
                            log_prob_data_iterator,
                            log_prob_num_microbatches,
                            store_prefix="ref_",
                        )
                    )
                    if need_sdpo_student_representations:
                        rollout_data["sdpo_student_representation_reused_actor_forward"] = torch.tensor(
                            float(
                                not self.args.use_rollout_logprobs
                                or self.args.get_mismatch_metrics
                                or need_agent_task_rollout_entropy
                            ),
                            dtype=torch.float32,
                        )

                # Forward teacher model to get teacher_log_probs for Megatron-based OPD
                if "teacher" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("teacher")
                    log_prob_data_iterator, log_prob_num_microbatches = self._get_log_prob_data_iterator(
                        rollout_data,
                        data_iterator,
                        num_microbatches,
                    )
                    rollout_data.update(
                        self.compute_log_prob(
                            log_prob_data_iterator,
                            log_prob_num_microbatches,
                            store_prefix="teacher_",
                        )
                    )

                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                can_reuse_log_probs_in_loss = (
                    len(num_microbatches) == 1
                    and self.args.loss_type == "policy_loss"
                    and self.args.kl_coef == 0
                    and not self.args.use_rollout_logprobs
                    and not self.args.get_mismatch_metrics
                    and not self.args.use_critic
                    and not self.args.keep_old_actor
                    and not self.args.use_opd
                    and not self.args.use_routing_replay
                    and self.args.advantage_estimator != "gspo"
                    and not need_agent_task_rollout_entropy
                )
                if (
                    (not self.args.use_rollout_logprobs and self._needs_sdpo_correction_log_probs())
                    or self.args.get_mismatch_metrics
                    or need_agent_task_rollout_entropy
                    or need_sdpo_student_representations
                ) and not can_reuse_log_probs_in_loss:
                    if self.args.use_routing_replay:
                        if self.args.use_rollout_routing_replay:
                            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                        else:
                            os.environ["ROUTING_REPLAY_STAGE"] = "record"
                    log_prob_data_iterator, log_prob_num_microbatches = self._get_log_prob_data_iterator(
                        rollout_data,
                        data_iterator,
                        num_microbatches,
                    )
                    rollout_data.update(
                        self.compute_log_prob(
                            log_prob_data_iterator,
                            log_prob_num_microbatches,
                            store_prefix="",
                            with_entropy=need_agent_task_rollout_entropy or None,
                            temperature=getattr(self.args, "rollout_temperature", 1.0),
                            collect_sdpo_student_representations=need_sdpo_student_representations,
                        )
                    )
                    if self.args.use_rollout_routing_replay:
                        RoutingReplay.clear_all_forward()

                if self.args.use_critic:
                    if external_data is not None and mpu.is_pipeline_last_stage():
                        values = external_data.get("values")
                        if values is not None:
                            from slime.backends.megatron_utils.data import tensors_to_gpu

                            rollout_data["values"] = tensors_to_gpu(values)
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args, rollout_id, rollout_data)

            self._ensure_grpo_policy_token_weights(rollout_data)

            sdpo_topk_interleaved = (
                self.args.loss_type == "sdpo_loss"
                and self._sdpo_needs_topk_distillation()
                and len(num_microbatches) > 1
            )
            sdpo_topk_interleaved_token_weights = sdpo_topk_interleaved and self._sdpo_token_weights_enabled()
            if self.args.loss_type == "sdpo_loss" and not sdpo_topk_interleaved:
                if self._sdpo_uses_representation_distillation():
                    self._ensure_sdpo_teacher_representations(rollout_data)
                else:
                    self._ensure_sdpo_teacher_log_probs(rollout_data)
                self._ensure_sdpo_token_weights(rollout_data)
            elif sdpo_topk_interleaved_token_weights:
                self._precompute_sdpo_topk_for_all_train_steps(
                    rollout_data,
                    num_steps_per_rollout=len(num_microbatches),
                )

            if not sdpo_topk_interleaved:
                log_rollout_data(
                    rollout_id,
                    self.args,
                    rollout_data,
                )

            # Train
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            train_kwargs = {}
            if (
                sdpo_topk_interleaved
                and not sdpo_topk_interleaved_token_weights
            ):
                train_kwargs["before_train_step"] = lambda step_id: self._precompute_sdpo_topk_for_train_step(
                    rollout_data,
                    step_id=step_id,
                    num_steps_per_rollout=len(num_microbatches),
                )
            with timer("actor_train"):
                train_metrics = train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                    global_batch_sizes,
                    **train_kwargs,
                )

            if sdpo_topk_interleaved:
                log_rollout_data(
                    rollout_id,
                    self.args,
                    rollout_data,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(
            self.args,
            rollout_id=rollout_id,
            rollout_data=rollout_data,
            include_sdpo_token_weights=getattr(
                self.args,
                "save_debug_train_data_include_sdpo_token_weights",
                False,
            ),
        )

        if self.args.use_routing_replay:
            RoutingReplay.clear_all()

        # TensorBackuper.copy copies CPU backups, so capture the successful live
        # optimizer result first. This keeps old_actor exactly one policy step old.
        old_actor_refreshed = self._refresh_actor_backups_after_update()

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
            and not self._grpo_token_weights_enabled()
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        perf_metrics = log_perf_data(rollout_id, self.args, extra_metrics=self.weight_updater.pop_metrics())
        return {
            "train_metrics": train_metrics,
            "perf_metrics": perf_metrics,
            "old_actor_refreshed": old_actor_refreshed,
        }

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        # torch dist may trigger nccl communication during saving.
        runner_managed_offload = bool(self.args.offline_train_eval_colocate)
        if self.args.offload_train and not runner_managed_offload:
            self.wake_up()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)
        self._save_sdpo_ema_teacher(rollout_id)
        self._prune_sdpo_ema_teacher_sidecars_fallback()

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            save_hf_model_to_path(self.args, Path(self.args.save_hf.format(rollout_id=rollout_id)), self.model)

        if self.args.offload_train and not runner_managed_offload:
            self.sleep()

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.get_updatable_engines_and_lock.remote()
        )

        reconnect_rollout_engines = self.args.offload_train and self.args.use_critic and not self.args.colocate

        if reconnect_rollout_engines:
            self.wake_up()
        elif self.args.offload_train:
            reload_process_groups()

        should_connect_rollout_engines = (
            num_new_engines > 0 or reconnect_rollout_engines or not hasattr(self.weight_updater, "rollout_engines")
        )
        if should_connect_rollout_engines:
            if not rollout_engines:
                raise RuntimeError("weight updater requires rollout engines before its first synchronization")
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and len(rollout_engines) > 0 and self.weight_updater.weight_version > 0:
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

            if self._should_update_old_actor_during_weight_sync():
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    self._advance_online_old_actor_after_weight_sync()
                else:
                    self.weights_backuper.backup("old_actor")

        if reconnect_rollout_engines:
            self.sleep()
        elif self.args.offload_train:
            destroy_process_groups()

    def get_model_backup_tags(self) -> list[str]:
        return sorted(self.weights_backuper.backup_tags)

    def get_tracking_state(self) -> dict[str, bool]:
        """Return the effective train-rank tracking ownership for runtime audits."""

        return {
            "driver_owned": self._driver_owned_train_tracking,
            "secondary_initialized": is_megatron_main_rank() and not self._driver_owned_train_tracking,
        }

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        old_ckpt_step = None
        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step
        elif model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if old_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag
