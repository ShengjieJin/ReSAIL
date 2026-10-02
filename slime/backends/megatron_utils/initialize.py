# Modified for ReSAIL. See NOTICE and LICENSE for attribution and terms.
import logging
import random

import numpy as np
import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.config import set_experimental_flag
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.training.global_vars import _build_tokenizer, set_args

logger = logging.getLogger(__name__)


def _decrease_dummy_batch_size_if_needed(args) -> bool:
    """Allow Megatron's unused bootstrap calculator to accept uneven dynamic DP splits.

    Slime passes the real per-step microbatch schedule and global row count directly
    to the forward/backward loop.  The calculator initialized below is retained only
    for Megatron compatibility, so rounding its *running* placeholder batch is safe
    when the real dynamic scheduler owns the split.
    """

    requested = bool(getattr(args, "decrease_batch_size_if_needed", False))
    divisor = args.micro_batch_size * args.data_parallel_size
    dynamic_uneven_dp = (
        bool(getattr(args, "use_dynamic_batch_size", False))
        and args.global_batch_size >= divisor
        and args.global_batch_size % divisor != 0
    )
    if dynamic_uneven_dp and args.rank == 0:
        logger.info(
            "Allowing Megatron's unused bootstrap batch calculator to round %s for DP=%s; "
            "Slime's dynamic schedule still trains the complete %s-row global batch.",
            args.global_batch_size,
            args.data_parallel_size,
            args.global_batch_size,
        )
    return requested or dynamic_uneven_dp


def _dummy_global_batch_size(args) -> int:
    """Return a valid size for Megatron's unused bootstrap calculator.

    Dynamic Slime steps may contain fewer rows than DP ranks. Megatron's
    compatibility calculator rounds down, which turns such a batch into zero
    microbatches. Give only that dummy calculator one full DP-width batch;
    the real scheduler continues to use the unchanged per-step row count.
    """
    divisor = args.micro_batch_size * args.data_parallel_size
    if bool(getattr(args, "use_dynamic_batch_size", False)) and args.global_batch_size < divisor:
        if args.rank == 0:
            logger.info(
                "Using bootstrap batch size %s for DP=%s; Slime's dynamic schedule still trains "
                "the complete %s-row global batch.",
                divisor,
                args.data_parallel_size,
                args.global_batch_size,
            )
        return divisor
    return args.global_batch_size


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Set random seed for reproducability."""
    # Ensure that different pipeline MP stages get different seeds.
    seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
    # Ensure different data parallel ranks get different seeds
    if data_parallel_random_init:
        seed = seed + (10 * mpu.get_data_parallel_rank(with_context_parallel=False))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed, te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)


def _initialize_distributed(args, get_embedding_ranks=None, get_position_embedding_ranks=None):
    """Initialize torch.distributed and core model parallel."""
    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    mpu.initialize_model_parallel(
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_comm_backend=args.pipeline_model_parallel_comm_backend,
        context_parallel_size=args.context_parallel_size,
        hierarchical_context_parallel_sizes=args.hierarchical_context_parallel_sizes,
        expert_model_parallel_size=args.expert_model_parallel_size,
        num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        distributed_timeout_minutes=args.distributed_timeout_minutes,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        order="tp-cp-ep-dp-pp" if not args.use_tp_pp_dp_mapping else "tp-cp-ep-pp-dp",
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        create_gloo_process_groups=args.enable_gloo_process_groups,
    )


def init(args):
    set_args(args)
    if args.enable_experimental:
        logger.info("Enable megatron experimental")
        set_experimental_flag(True)

    # Pytorch distributed.
    _initialize_distributed(args)

    # https://github.com/NVIDIA/Megatron-LM/issues/1563
    assert np.__version__.startswith("1."), "Megatron does not support numpy 2.x"

    # Random seeds for reproducibility.
    if args.rank == 0:
        logger.info(f"> setting random seeds to {args.seed} ...")
    _set_random_seed(
        args.seed,
        args.data_parallel_random_init,
        args.te_rng_tracker,
        args.inference_rng_tracker,
    )
    _build_tokenizer(args)
    # We won't use this. initialize to pass some validation in megatron.
    init_num_microbatches_calculator(
        args.rank,
        args.rampup_batch_size,
        _dummy_global_batch_size(args),
        args.micro_batch_size,
        args.data_parallel_size,
        _decrease_dummy_batch_size_if_needed(args),
    )

    if args.deterministic_mode:
        if args.rank == 0:
            logger.info("> running in deterministic mode")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)

    if args.tp_comm_overlap:
        from megatron.training.initialize import _initialize_tp_communicators

        _initialize_tp_communicators()

    if getattr(args, "custom_megatron_init_path", None):
        from slime.utils.misc import load_function

        custom_init = load_function(args.custom_megatron_init_path)
        custom_init(args)


# TODO shall we use a simpler method to determine which rank to init wandb?
def is_megatron_main_rank():
    return (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
    )
