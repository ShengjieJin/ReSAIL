# Modified for ReSAIL. See NOTICE and LICENSE for attribution and terms.
import argparse
import copy
import json
import logging
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any

import yaml
from sglang_router.launch_router import RouterArgs

from slime.backends.sglang_utils.arguments import sglang_parse_args
from slime.backends.sglang_utils.arguments import validate_args as sglang_validate_args
from slime.backends.sglang_utils.external import apply_external_engine_info_to_args
from slime.utils.eval_config import EvalDatasetConfig, build_eval_dataset_configs, ensure_dataset_list
from slime.utils.logging_utils import configure_logger

logger = logging.getLogger(__name__)


def _is_megatron_checkpoint_path(path: str | None) -> bool:
    if path is None or not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, "latest_checkpointed_iteration.txt")) or bool(
        re.fullmatch(r"iter_\d{7}", os.path.basename(os.path.normpath(path)))
    )


def reset_arg(parser, name, **kwargs):
    """
    Reset the default value of a Megatron argument.
    :param parser: The argument parser.
    :param name: The name of the argument to reset.
    :param default: The new default value.
    """
    for action in parser._actions:
        if name in action.option_strings:
            if "default" in kwargs:
                action.default = kwargs["default"]
            break
    else:
        parser.add_argument(name, **kwargs)


def get_slime_extra_args_provider(add_custom_arguments=None):
    def add_slime_arguments(parser):
        # Ray
        def add_cluster_arguments(parser):
            parser.add_argument("--actor-num-nodes", type=int, default=1, help="Number of nodes for training actor")
            parser.add_argument(
                "--actor-num-gpus-per-node", type=int, default=8, help="Number of gpus per node for training actor"
            )

            parser.add_argument(
                "--rollout-num-gpus",
                type=int,
                default=None,
                help=(
                    "Number of GPUs for inference. Note that when using --colocate, "
                    "i.e. the training and the inference engines are on the same gpus, this param will be ignored and will be set as "
                    "actor_num_gpus_per_node * actor_num_nodes."
                ),
            )
            parser.add_argument(
                "--rollout-num-gpus-per-engine",
                type=int,
                default=1,
                help="Number of GPUs per inference engine, just like the tp_size in sglang.",
            )
            parser.add_argument(
                "--num-gpus-per-node",
                type=int,
                default=8,
                help=(
                    "Number of gpus per node for rollout."
                    "Notice: If you are going to use less than 8 gpus per node under colocate mode, you should set this number."
                ),
            )
            parser.add_argument(
                "--colocate",
                action="store_true",
                default=False,
                help=(
                    "Whether to colocate the inference engines and the actor. "
                    "Turning this on will also set --offload to true."
                ),
            )
            parser.add_argument(
                "--offload",
                action="store_true",
                default=False,
                help=("Equivalent to --offload-train + --offload-rollout. "),
            )
            parser.add_argument(
                "--offload-train",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the training actor to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )
            parser.add_argument(
                "--offload-rollout",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the rollout generator to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )
            parser.add_argument(
                "--offline-train-eval-colocate",
                action="store_true",
                default=False,
                help=(
                    "Keep all colocated GPUs in the train role between eval boundaries and time-multiplex "
                    "all of them into rollout only for evaluation. Supported by synchronous train.py."
                ),
            )
            parser.add_argument(
                "--strict-on-policy-audit-dir",
                type=str,
                default=None,
                help=(
                    "Write one mechanically verified synchronous policy-version/role audit record per update. "
                    "This contract is supported only by train.py with colocated train/rollout GPUs."
                ),
            )
            parser.add_argument(
                "--offline-async-eval",
                action="store_true",
                default=False,
                help=(
                    "Use independent resident training and evaluation GPU groups. Offline data generation is owned "
                    "by a no-server manager while a separate rollout manager owns asynchronous evaluation."
                ),
            )
            parser.add_argument(
                "--driver-owned-train-tracking",
                action="store_true",
                default=False,
                help="Return train metrics to the driver instead of initializing a W&B secondary on the train rank.",
            )

            reset_arg(parser, "--distributed-backend", type=str, default="nccl")
            reset_arg(parser, "--distributed-timeout-minutes", type=int, default=10)

            return parser

        def add_train_arguments(parser):
            # --train-backend is parsed early in _pre_parse_mode() and merged later.
            parser.add_argument(
                "--qkv-format",
                type=str,
                choices=["thd", "bshd"],
                default="thd",
                help="The qkv layout for Megatron backend.",
            )
            parser.add_argument(
                "--qwen-gdn-backend",
                type=str,
                choices=["fla", "flashqla"],
                default="fla",
                help="GDN implementation backend for Qwen linear-attention layers.",
            )
            parser.add_argument(
                "--train-env-vars",
                type=json.loads,
                default="{}",
                help="Extra environment variables for training process, e.g. PyTorch memory management ones.",
            )
            parser.add_argument(
                "--train-memory-margin-bytes",
                type=int,
                default=1024**3,
                help="Add margin for train memory allocation. By default we will reserve 1GB as margin.",
            )
            parser.add_argument(
                "--megatron-to-hf-mode",
                choices=["raw", "bridge"],
                default="raw",
                help="The method to convert megatron weights to hugging face weights for SGLang.",
            )
            # Delta weight sync.
            parser.add_argument(
                "--update-weight-mode",
                choices=["full", "delta"],
                default="full",
                help=(
                    "Weight sync strategy. 'full' (default) broadcasts every parameter "
                    "every sync. 'delta' detects byte-level changes against a pinned-CPU "
                    "snapshot of the previous broadcast and ships only the changed positions + values."
                ),
            )
            parser.add_argument(
                "--update-weight-transport",
                choices=["nccl", "disk"],
                default="nccl",
                help=(
                    "Carrier for weight sync. In full mode, 'nccl' broadcasts chunks and "
                    "'disk' writes a complete HF checkpoint under --update-weight-disk-dir "
                    "before engines reload it. In delta mode, 'nccl' broadcasts sparse deltas; "
                    "'disk' writes sparse safetensors under --update-weight-disk-dir and pushes "
                    "once at end-of-sync."
                ),
            )
            parser.add_argument(
                "--update-weight-disk-dir",
                type=str,
                default=None,
                help=(
                    "Filesystem directory for disk-backed weight sync. In --update-weight-mode=full, "
                    "one complete HF checkpoint directory is written per sync. In delta mode, "
                    "one sparse-delta directory is written per sync."
                ),
            )
            parser.add_argument(
                "--update-weight-disk-keep-files",
                action="store_true",
                default=False,
                help=(
                    "Skip cleanup of full-checkpoint directories written by "
                    "--update-weight-mode=full --update-weight-transport=disk."
                ),
            )
            parser.add_argument(
                "--update-weight-encoding",
                choices=["indices", "deltas", "deltas_zstd"],
                default="indices",
                help=(
                    "Position encoding for partial flushes. 'indices': int32 absolute "
                    "positions (largest, lowest compute). 'deltas': uint16 gap-deltas "
                    "with uint32 fallback (smaller). 'deltas_zstd': 'deltas' with the "
                    "safetensors blob wrapped in zstd L1 (smallest, heaviest compute — "
                    "best for shared-FS bandwidth ≤ ~300 MB/s)."
                ),
            )
            parser.add_argument(
                "--update-weight-delta-dir",
                type=str,
                default=None,
                help=(
                    "Deprecated alias for --update-weight-disk-dir and will be removed in a future "
                    "release. Prefer the transport-level directory flag for both full and delta disk sync."
                ),
            )
            parser.add_argument(
                "--update-weight-delta-keep-files",
                action="store_true",
                default=False,
                help="Skip post-apply cleanup of per-sync version directories. Useful for debugging.",
            )
            parser.add_argument(
                "--custom-delta-pre-push-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom function called by --update-weight-transport=disk after each "
                    "trainer rank's files are durably on local disk, before rank 0 fires the engine "
                    "RPCs. Signature: ``def hook(args, version_dir: str, rollout_engines) -> None``. "
                    "Called from every trainer rank; the hook gates itself."
                ),
            )
            parser.add_argument(
                "--custom-model-provider-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom model provider function. "
                    "If set, we will use this function instead of the default model provider. "
                    "The function should have the signature "
                    "`def custom_model_provider(pre_process: bool, post_process: bool, vp_stage: int | None = None) -> GPTModel`. "
                    "Example: 'my_module.my_model_provider'."
                ),
            )
            parser.add_argument(
                "--recompute-loss-function",
                action="store_true",
                help="Whether to disable recompute loss function to save memory during training.",
            )
            parser.add_argument(
                "--log-probs-chunk-size", type=int, default=-1, help="Chunk size to compute log probs to save memory"
            )
            parser.add_argument(
                "--only-train-params-name-list",
                type=str,
                nargs="*",
                default=None,
                help=r"""List of regex patterns of parameter names to TRAIN. All other parameters will be FROZEN.
                        Supports Python regex syntax (re.search).

                        Examples:
                        1. Train ONLY MoE experts:
                            --only-train-params-name-list experts

                        2. Train ONLY Indexer parameters:
                            --only-train-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj

                        3. Train ONLY Layer 20 to 23:
                            --only-train-params-name-list layers\.2[0-3]\.
                        """,
            )

            parser.add_argument(
                "--freeze-params-name-list",
                type=str,
                nargs="*",
                default=None,
                help="""List of regex patterns of parameter names to FREEZE. Other parameters will remain trainable.
                        Supports Python regex syntax (re.search).

                        Examples:
                        1. Freeze Embeddings and Output Layer (common for fine-tuning):
                            --freeze-params-name-list embedding output_layer

                        2. Freeze Indexer parameters:
                            --freeze-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj

                        3. Freeze specific projection layers (e.g., all Gate/Up projections):
                            --freeze-params-name-list linear_fc1
                        """,
            )
            parser.add_argument(
                "--allgather-cp",
                action="store_true",
                default=False,
            )

            return parser

        # rollout
        def add_rollout_arguments(parser):
            parser.add_argument(
                "--hf-checkpoint",
                type=str,
                default=None,
                help=(
                    "The huggingface checkpoint of the trained model. "
                    "This is used to initialize sglang and also provide the tokenizer. "
                    "Note that, we will always update the parameters in sglang with that of megatron before training, "
                    "so you only need to provide a huggingface checkpoint that has the same architecture as the model you want to train. "
                    "It doesn't necessary need to contain the most up-to-date parameters."
                ),
            )
            parser.add_argument(
                "--model-name",
                type=str,
                default=None,
                help=(
                    "The name of the model, this is used to convert the megatron weights into huggingface format. "
                    "If not set, we will use `type(AutoConfig.from_pretrained(args.hf_checkpoint)).__name__.lower()` as model_name. "
                    "Also, sometimes this will help alleviate the bug that transformers cannot find certain model."
                ),
            )
            parser.add_argument(
                "--rollout-function-path",
                type=str,
                default="slime.rollout.sglang_rollout.generate_rollout",
                help=(
                    "Path to the rollout generation function."
                    "You should use this model to create your own custom rollout function, "
                    "and then set this to the path of your custom rollout function. "
                    "The signature of the function should be "
                    "`def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput`"
                    "and within the output sample, you should at least set `tokens`, `response_length`, `reward` "
                    "and `status`."
                ),
            )
            parser.add_argument(
                "--rollout-temperature",
                type=float,
                default=1.0,
                help="the temperature for the inference engine during rollout.",
            )
            parser.add_argument(
                "--rollout-top-p", type=float, default=1.0, help="the top-p for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-top-k", type=int, default=-1, help="the top-k for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-max-context-len",
                type=int,
                default=None,
                help=(
                    "The maximum context size for the inference engine during rollout."
                    "It should no exceed the `max_position_embeddinds` in Huggingface model's `config.json`"
                ),
            )
            parser.add_argument(
                "--rollout-max-prompt-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the prompt for the inference engine during rollout. "
                    "If set, we will filter out the long prompts during initialization of the global dataset. "
                    "This is not recommended if the dataset is large."
                ),
            )
            parser.add_argument(
                "--rollout-max-response-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the response for the inference engine during rollout. "
                    "It is basically `max_tokens` in sglang."
                ),
            )
            parser.add_argument(
                "--rollout-skip-special-tokens",
                action="store_true",
                default=False,
                help=(
                    "Whether to skip special tokens in the response during rollout. "
                    "This is useful when you want to use the response as a prompt for the next rollout."
                ),
            )
            parser.add_argument(
                "--rollout-stop",
                type=str,
                nargs="+",
                default=None,
                help=(
                    "The stop words for the inference engine during rollout. "
                    "It can be a list of strings or a single string. "
                    "It may be hard to pass special tokens in command line, in that case rollout_stop_token_ids can be used."
                ),
            )
            parser.add_argument(
                "--rollout-stop-token-ids",
                type=int,
                nargs="+",
                default=None,
                help=(
                    "The stop token ids for the inference engine during rollout. "
                    "It can be a list of integers or a single integer."
                ),
            )
            parser.add_argument(
                "--rollout-shuffle",
                action="store_true",
                default=False,
                help=("Whether to shuffle the prompts during rollout."),
            )
            parser.add_argument(
                "--rollout-seed",
                type=int,
                default=42,
                help=(
                    "The seed for the random number generator during rollout. "
                    "This is used to shuffle the prompts and also for the random sampling of the prompts."
                ),
            )

            # sampling
            parser.add_argument(
                "--over-sampling-batch-size",
                type=int,
                default=None,
                help=(
                    "This defines the granularity of the sampling batch in the rollout function. "
                    "When the number of available samples falls below the target, a sampling "
                    "operation of size over_sampling_batch_size will be triggered."
                    "Regardless of whether partial rollout is used or filters are applied, "
                    "the sampling granularity is always determined by this value. "
                    "If this value is None, rollout_batch_size will be used as the default over_sampling_batch_size."
                ),
            )
            parser.add_argument(
                "--dynamic-sampling-filter-path",
                type=str,
                default=None,
                help=(
                    "This is the filter function for dynamic sampling. "
                    "It should be able to judge whether the result of a prompt should be selected or not."
                    "We will do dynamic filter for sampling as in DAPO. e.g. not all correct or all wrong samples."
                    "You could use `slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` as an example."
                ),
            )

            # partial rollout
            parser.add_argument(
                "--partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to use partial rollout. "
                    "If set, the unfinished samples during dynamic sampling will be recycled back to data buffer. "
                    "This is useful for long responses."
                ),
            )
            parser.add_argument(
                "--mask-offpolicy-in-partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to mask previous generation in partial rollout. "
                    "If set, only on-policy generated tokens will be used in training"
                ),
            )
            parser.add_argument(
                "--custom-generate-function-path",
                type=str,
                default=None,
                help=(
                    "Only substitue the `def generate(args, sample, sampling_params)` function within the example rollout function. "
                    "This should be useful if you need to implement some special rollout logic, e.g. multi-turn, function calling."
                ),
            )
            parser.add_argument(
                "--custom-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging rollout data. The signature of the functions is: "
                    "def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )
            parser.add_argument(
                "--custom-eval-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging eval rollout data. "
                    "def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )

            parser.add_argument(
                "--buffer-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the buffer filter function. "
                    "It should be able to select the samples in the buffer. "
                    "The function should take list[list[Sample]] and return list[list[Sample]]."
                ),
            )
            # update weight
            parser.add_argument(
                "--update-weight-buffer-size",
                type=int,
                default=512 * 1024**2,
                help=(
                    "buffer size for update weight, in bytes. "
                    "This is used for updating weights by chunk and should be useful for MoE models."
                ),
            )
            parser.add_argument(
                "--update-weights-interval",
                type=int,
                default=1,
                help="Interval for updating the weights",
            )
            parser.add_argument(
                "--keep-old-actor",
                action="store_true",
                help="Whether to keep the rollout model on training process",
            )

            parser.add_argument(
                "--rollout-data-postprocess-path",
                type=str,
                default=None,
                help=(
                    "The called after we have all the rollout data including log_probs. "
                    "It may be helpful for updating loss mask."
                ),
            )
            parser.add_argument(
                "--rollout-external-engine-addrs",
                type=str,
                default=None,
                nargs="+",
                help="Address and ports of the external engines.",
            )
            return parser

        def add_fault_tolerance_arguments(parser):
            parser.add_argument(
                "--use-fault-tolerance",
                action="store_true",
                default=False,
                help="Whether to enable the fault tolerance function during rollout.",
            )
            parser.add_argument(
                "--rollout-health-check-interval",
                type=float,
                default=30.0,
                help="Interval in seconds between rollout engine /health_generate checks during generate/eval.",
            )
            parser.add_argument(
                "--rollout-health-check-timeout",
                type=float,
                default=30.0,
                help="Timeout in seconds to wait for a rollout engine /health_generate response before killing it.",
            )
            parser.add_argument(
                "--rollout-health-check-first-wait",
                type=float,
                default=0,
                help="Initial grace period (in seconds) before starting health checks. This allows time for model compilation and initialization. Increase this value significantly when using deepgemm.",
            )
            return parser

        # data
        def add_data_arguments(parser):
            # dataset
            # TODO: maybe add an num_epoch and calculate the num_rollout from buffer
            parser.add_argument(
                "--num-rollout",
                type=int,
                default=None,
                help="Number of rollout steps. If not set, we will calculate the number of rollout steps from the dataset size.",
            )
            parser.add_argument(
                "--stop-after-eval-iteration",
                type=int,
                default=None,
                help=(
                    "Exit cleanly after this rollout iteration has completed its scheduled checkpoint and evaluation. "
                    "A resume immediately after the boundary finalizes without generation; later training phases must "
                    "clear this option."
                ),
            )
            parser.add_argument(
                "--num-epoch",
                type=int,
                default=None,
                help=(
                    "Number of epochs for the training. "
                    "This is used to calculate the number of rollout steps from the dataset size. "
                    "If set, we will calculate the number of rollout steps as `num_rollout = num_epoch * dataset_size // rollout_batch_size`."
                    "If both `--num-epoch` and `--num-rollout` are set, `--num-epoch` will be ignored."
                ),
            )

            parser.add_argument(
                "--disable-rollout-global-dataset",
                action="store_false",
                dest="rollout_global_dataset",
                help=(
                    "Whether to use a global dataset for rollout. "
                    "If set, the rollout will use the `--prompt-data` as the prompt dataset, "
                    "and the prompts for rollout will be sampled from the dataset. "
                    "If not set, you need to manage the data by your self."
                ),
            )

            parser.add_argument(
                "--data-source-path",
                type=str,
                default="slime.rollout.data_source.RolloutDataSourceWithBuffer",
                help="The data source class for rollout data.",
            )
            parser.add_argument(
                "--prompt-data",
                type=str,
                default=None,
                help=(
                    "The path to the prompt data. "
                    "Currently we only support jsonl format, and each line should contains --input-key and --label-key, "
                    "which will be used as the prompt and the label respectively. "
                    "If you want to use a custom template, you can set --apply-chat-template to true, in that case, "
                    "the input should be the same structure as an openai message, e.g. [{'role': 'user', 'content': 'blabla'}]. "
                ),
            )
            parser.add_argument("--apply-chat-template", action="store_true", default=False)
            # Temporarily be JSON-serialized str, will be a real dict after using Omegaconf
            parser.add_argument("--apply-chat-template-kwargs", type=json.loads, default="{}")
            parser.add_argument("--input-key", type=str, default="input", help="JSON dataset key")
            parser.add_argument("--label-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument(
                "--multimodal-keys",
                type=json.loads,
                default=None,
                help=(
                    'JSON string for multimodal data mapping media types to data keys. Example: \'{"image": "image_file"}\''
                ),
            )
            parser.add_argument("--metadata-key", type=str, default="metadata", help="JSON dataset key")
            parser.add_argument(
                "--tool-key",
                type=str,
                default="tools",
                help=(
                    "When need to add tools during apply_chat_template, you should provide the key for the tools in the prompt dataset."
                ),
            )

            parser.add_argument(
                "--start-rollout-id",
                type=int,
                default=None,
                help=(
                    "The starting rollout step, if not set, will try to load the step from --load when doing continue training, "
                    "otherwise will be set to 0, meaning training from start."
                ),
            )

            # batch sizes
            parser.add_argument(
                "--rollout-batch-size",
                type=int,
                required=True,
                help=(
                    "The number of prompts in each rollout step. "
                    "The total data returned should be rollout_batch_size * n_samples_per_prompt. "
                ),
            )
            parser.add_argument(
                "--n-samples-per-prompt", type=int, default=1, help="Number of responses for each prompt in generation"
            )

            # gbs of the training, note that the gbs is of sample, not of prompts,
            # so if you hope to train 1 step for each rollout, the global_bach_size should be set as
            # `rollout_batch_size * n_samples_per_prompt`.
            reset_arg(parser, "--global-batch-size", type=int, default=None)
            parser.add_argument(
                "--num-steps-per-rollout",
                type=int,
                default=None,
                help=(
                    "Number of steps per rollout, e.g. It is equivalent to setting gbs as "
                    "`rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`."
                ),
            )
            # mbs for the training, will be ignored if `use_dynamic_batch_size` is set.
            reset_arg(parser, "--micro-batch-size", type=int, default=1)
            parser.add_argument(
                "--balance-data",
                action="store_true",
                default=False,
                help=(
                    "Balance the number of tokens between data parallel ranks with `karmarkar_karp` for verl. "
                    "Note that this may allocate the different response of the same prompt into different training steps."
                ),
            )

            parser.add_argument(
                "--use-dynamic-batch-size",
                action="store_true",
                default=False,
                help=(
                    "Because the sample length varies, to maximize the GPU utilization, "
                    "we will use the dynamic batch size to adjust the micro batch size according to the maximum number of tokens each gpu can run. "
                    "For example, if we have 3 samples, with the length of 100, 200, and 300, and the max_tokens_per_gpu is 300, when enabling "
                    "dynamic batch size, slime will make 2 micro batches, i.e. [100, 200], [300]."
                ),
            )
            parser.add_argument(
                "--max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for dynamic batch size. "
                    "Note that when enabling context parallel (CP), the max tokens per gpu should be around "
                    "`max_response_len // cp_size` instead of `max_response_len`."
                ),
            )
            parser.add_argument(
                "--log-probs-max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for calculating log probs. "
                    "This is used to calculate the log probs of the responses during rollout, "
                    "and should be set to a larger value than `max_tokens_per_gpu` if you want better performance. "
                ),
            )
            parser.add_argument(
                "--sdpo-teacher-representation-max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "Maximum tokens per GPU for SDPO teacher representation precompute. "
                    "Defaults to max_tokens_per_gpu when unset."
                ),
            )
            parser.add_argument(
                "--sdpo-teacher-representation-forward-chunk-rows",
                type=int,
                default=512,
                help=("Maximum expanded teacher rows per forward chunk for SDPO representation precompute."),
            )
            return parser

        def add_eval_arguments(parser):
            parser.add_argument(
                "--eval-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the eval generation function."
                    "If not set, we will use rollout_function_path as the default. "
                ),
            )

            # change the default value of eval_interval from Megatron to None
            reset_arg(parser, "--eval-interval", type=int, default=None)

            parser.add_argument(
                "--eval-prompt-data",
                type=str,
                default=None,
                nargs="+",
                help=(
                    "Path to the evaluation prompt data, "
                    "should first input the name of the eval dataset and then the path, e.g. "
                    "aime /path/to/aime.jsonl"
                ),
            )
            parser.add_argument(
                "--eval-config",
                type=str,
                default=None,
                help=(
                    "Path to an OmegaConf YAML/JSON file describing evaluation datasets. "
                    "When provided, this overrides --eval-prompt-data."
                ),
            )
            parser.add_argument(
                "--skip-eval-before-train",
                action="store_true",
                default=False,
                help="Whether to skip evaluation before training.",
            )

            # The following keys are used to override the rollout version during eval.
            parser.add_argument("--eval-input-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--eval-label-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--eval-tool-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument(
                "--n-samples-per-eval-prompt",
                type=int,
                default=1,
                help="number of responses for each prompt in generation",
            )
            parser.add_argument("--eval-temperature", type=float, default=None)
            parser.add_argument("--eval-top-p", type=float, default=None)
            parser.add_argument("--eval-top-k", type=int, default=None)
            parser.add_argument("--eval-max-response-len", type=int, default=None)
            parser.add_argument("--eval-max-prompt-len", type=int, default=None)
            parser.add_argument("--eval-min-new-tokens", type=int, default=None)
            parser.add_argument("--eval-max-context-len", type=int, default=None)
            parser.add_argument(
                "--eval-sampling-seed",
                type=int,
                default=314159,
                help="Base seed for the eval-only deterministic RNG namespace.",
            )
            parser.add_argument(
                "--eval-snapshot-audit-dir",
                type=str,
                default=None,
                help=(
                    "Optional directory for per-eval immutable snapshot/version and data-source cursor audits."
                ),
            )

            return parser

        def add_algo_arguments(parser):
            parser.add_argument(
                "--ref-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for reference model. "
                    "When --load is not set, this will be used as the initial checkpoint for training. "
                ),
            )
            parser.add_argument(
                "--ref-ckpt-step", type=int, default=None, help="The checkpoint step for reference model. "
            )
            reset_arg(parser, "--load", type=str, default=None)
            reset_arg(parser, "--save", type=str, default=None)
            reset_arg(parser, "--save-interval", type=int, default=None)
            reset_arg(parser, "--async-save", action="store_true")
            parser.add_argument(
                "--skip-final-save",
                action="store_true",
                default=False,
                help="Do not force a checkpoint on the last rollout when --save-interval does not otherwise trigger.",
            )
            parser.add_argument(
                "--checkpoint-retention-policy",
                type=str,
                choices=["none", "latest_and_best_eval"],
                default="none",
                help=(
                    "Checkpoint pruning policy. latest_and_best_eval keeps the newest checkpoint plus the "
                    "top eval checkpoints selected by --best-checkpoint-metric."
                ),
            )
            parser.add_argument(
                "--checkpoint-fixed-iterations",
                type=int,
                nargs="*",
                default=[],
                help="Checkpoint iterations that retention must never prune (for example 29 59).",
            )
            parser.add_argument(
                "--best-checkpoint-metric",
                type=str,
                default=None,
                help="Metric key used to rank eval checkpoints when checkpoint retention is enabled.",
            )
            parser.add_argument(
                "--best-checkpoint-limit",
                type=int,
                default=2,
                help="Number of best eval checkpoints to retain in addition to the latest checkpoint.",
            )
            parser.add_argument(
                "--best-checkpoint-mode",
                type=str,
                choices=["max", "min"],
                default="max",
                help="Whether higher or lower --best-checkpoint-metric values are better.",
            )
            reset_arg(
                parser,
                "--no-save-optim",
                action="store_true",
                default=False,
                help=(
                    "If set, do not save the optimizer state when saving checkpoints. "
                    "This reduces checkpoint size but disables training resumption from the saved checkpoint."
                ),
            )
            parser.add_argument(
                "--save-hf",
                type=str,
                default=None,
                help=(
                    "Path to save the model in HuggingFace format when using Megatron backend. "
                    "The model will be saved to `save_hf.format(rollout_id)`. "
                    "In raw Megatron-to-HF mode, weights are saved with the same quantization config "
                    "as `--hf-checkpoint`. "
                ),
            )
            reset_arg(parser, "--seed", type=int, default=1234)
            reset_arg(parser, "--clip-grad", type=float, default=1.0)
            reset_arg(parser, "--calculate-per-token-loss", action="store_true")
            reset_arg(parser, "--lr", type=float, default=1e-6)

            parser.add_argument(
                "--num-critic-only-steps",
                type=int,
                default=0,
                help="Number of initial rollout steps that train critic only; set >= num_rollout for critic-only runs",
            )
            parser.add_argument(
                "--megatron-config-path",
                type=str,
                default=None,
                help=(
                    "Path to a structured YAML config for Megatron roles. The file should use "
                    "a top-level 'megatron' key with role-tagged entries; the critic runtime will "
                    "select exactly one entry with role=critic. Legacy 'critic' configs are still accepted."
                ),
            )

            parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clip range")
            parser.add_argument("--eps-clip-high", type=float, default=None, help="PPO clip upper range")
            parser.add_argument(
                "--grpo-token-weights",
                action=argparse.BooleanOptionalAction,
                default=False,
                help="Weight only the GRPO clipped policy-gradient term using frozen-reference context disagreement.",
            )
            parser.add_argument(
                "--grpo-token-weights-power",
                type=float,
                default=1.0,
                help="Non-negative exponent applied to frozen-reference context disagreement.",
            )
            parser.add_argument(
                "--grpo-token-weights-max",
                type=float,
                default=8.0,
                help="Per-response mean-preserving maximum GRPO policy token weight; must be greater than 1.",
            )
            parser.add_argument(
                "--eps-clip-c",
                type=float,
                default=None,
                help="lower bound of the value for Dual-clip PPO from https://arxiv.org/pdf/1912.09729",
            )
            parser.add_argument("--value-clip", type=float, default=0.2, help="the clip for value loss")
            parser.add_argument(
                "--kl-coef",
                type=float,
                default=0.00,
                help="KL penalty coefficient for reward shaping. This is applied to the reward signal before advantage calculation.",
            )
            parser.add_argument(
                "--loss-type",
                type=str,
                choices=["policy_loss", "sft_loss", "sdpo_loss", "custom_loss"],
                default="policy_loss",
                help=(
                    "Choose loss type, currently support ppo policy_loss, sft_loss, sdpo_loss, "
                    "if custom_loss is set, we will use the function path from `--custom-loss-function-path`."
                ),
            )
            parser.add_argument(
                "--policy-loss-agg-mode",
                type=str,
                choices=["rollout_mean", "step_equal"],
                default="rollout_mean",
                help=(
                    "Reduction for policy loss and its auxiliary KL/metrics. step_equal computes an active-token "
                    "mean per response row and then one global mean over all response rows in the optimizer update."
                ),
            )
            parser.add_argument(
                "--sdpo-loss-agg-mode",
                type=str,
                choices=["turn_mean", "token_mean", "step_equal", "oel_prefix_mean"],
                default="turn_mean",
                help=(
                    "Reduction mode for SDPO loss: trajectory token mean, global token mean, or equal source-step "
                    "mean. oel_prefix_mean is a compatibility alias for step_equal."
                ),
            )
            parser.add_argument(
                "--sdpo-distillation-mode",
                type=str,
                choices=["topk", "sample_token", "representation"],
                default=None,
                help="Teacher signal used by SDPO loss.",
            )
            parser.add_argument(
                "--sdpo-representation-layers",
                type=str,
                choices=["last", "all"],
                default="last",
                help="Hidden representation layers used by SDPO representation distillation. V1 supports last only.",
            )
            parser.add_argument(
                "--sdpo-representation-coef",
                type=float,
                default=1.0,
                help="Coefficient applied to SDPO representation distillation loss.",
            )
            parser.add_argument(
                "--sdpo-representation-reduction",
                type=str,
                choices=["mean", "sum"],
                default="sum",
                help=(
                    "Hidden-dimension reduction for SDPO representation MSE. "
                    "`sum` is the default SDPO representation scale; `mean` matches OPRD-Vanilla."
                ),
            )
            parser.add_argument(
                "--sdpo-representation-success-aggregation",
                type=str,
                choices=["sample", "mean", "random"],
                default="sample",
                help=(
                    "How trajectory_demo representation SDPO chooses successful teacher demos. "
                    "`sample` keeps the existing single-success target; `random` picks one successful "
                    "trajectory per row with a stable hash; `mean` averages teacher representations from "
                    "all successful trajectories in the same rollout group."
                ),
            )
            parser.add_argument(
                "--sdpo-token-weights",
                action=argparse.BooleanOptionalAction,
                default=False,
                help="Weight SDPO token losses by positive-teacher versus configured-source representation disagreement.",
            )
            parser.add_argument(
                "--sdpo-token-weights-power",
                type=float,
                default=1.0,
                help=(
                    "Exponent applied to SDPO token-weight disagreement before uid normalization. "
                    "0 disables weighting, 1 keeps raw disagreement weighting."
                ),
            )
            parser.add_argument(
                "--sdpo-token-weight-source",
                type=str,
                choices=["teacher_contrast", "student"],
                default="teacher_contrast",
                help=(
                    "Representation source contrasted with the positive teacher for token weighting. "
                    "`teacher_contrast` uses the failed/no-context teacher; `student` reuses detached actor "
                    "representations from the log-prob forward."
                ),
            )
            parser.add_argument(
                "--sdpo-token-weights-max",
                type=float,
                default=0.0,
                help=(
                    "Optional max value for mean-preserving post-normalization SDPO token weights. "
                    "0 disables the cap; values greater than 1 cap per-token loss multipliers while "
                    "preserving uid-level average weight 1."
                ),
            )
            parser.add_argument(
                "--sdpo-filter-all-success-groups",
                action=argparse.BooleanOptionalAction,
                default=False,
                help=(
                    "Filter SDPO uid groups that contain successes but no failures. "
                    "When combined with --sdpo-no-success-context-mode filter, training keeps only mixed "
                    "success/failure groups."
                ),
            )
            parser.add_argument(
                "--sdpo-full-logit-distillation",
                action=argparse.BooleanOptionalAction,
                default=True,
                help="Enable SDPO distribution distillation over top-k support.",
            )
            parser.add_argument(
                "--sdpo-distillation-topk",
                type=int,
                default=20,
                help="Top-k support size for SDPO distillation.",
            )
            parser.add_argument(
                "--sdpo-distillation-add-tail",
                action=argparse.BooleanOptionalAction,
                default=True,
                help="Append a tail probability bucket when comparing SDPO top-k supports.",
            )
            parser.add_argument(
                "--pr-weight",
                type=float,
                default=0.0,
                help=(
                    "Weight for privileged-student to privileged-teacher retention on the selected SDPO support. "
                    "Zero preserves the ordinary-view SDPO path without an extra forward."
                ),
            )
            parser.add_argument(
                "--pr-support",
                choices=["selected", "all"],
                default="selected",
                help="Compute privileged retention on sensitivity-selected rows or all generated SDPO rows.",
            )
            parser.add_argument(
                "--pr-kl-direction",
                choices=["reverse", "forward"],
                default="reverse",
                help="Use KL(privileged student||teacher) or KL(teacher||privileged student).",
            )
            parser.add_argument(
                "--pr-view",
                choices=["privileged", "ordinary"],
                default="privileged",
                help="Prompt view used by both the student and frozen teacher in the retention component.",
            )
            parser.add_argument(
                "--sdpo-alpha",
                type=float,
                default=1.0,
                help="Interpolation for SDPO distribution KL; 1.0 is teacher-to-student reverse-KL style.",
            )
            parser.add_argument(
                "--sdpo-teacher-regularization",
                type=str,
                choices=["ema", "actor", "ref"],
                default="ema",
                help="Model weights used to score SDPO teacher prompts.",
            )
            parser.add_argument(
                "--sdpo-teacher-update-rate",
                type=float,
                default=0.01,
                help="EMA update rate for --sdpo-teacher-regularization=ema.",
            )
            parser.add_argument(
                "--sdpo-ema-teacher-keep-last",
                type=int,
                default=1,
                help="Fallback sidecar retention count for SDPO EMA teacher backups.",
            )
            parser.add_argument(
                "--sdpo-success-reward-threshold",
                type=float,
                default=1.0,
                help="Episode reward threshold used to classify successful SDPO trajectories.",
            )
            parser.add_argument(
                "--sdpo-dont-reprompt-on-self-success",
                action="store_true",
                default=False,
                help="Avoid using a trajectory as its own successful SDPO guidance source when alternatives exist.",
            )
            parser.add_argument(
                "--sdpo-teacher-context-mode",
                type=str,
                choices=["original", "feedback_only", "solution_only", "own_outcome", "hindsight"],
                default="original",
                help="Teacher prompt context mode for agent-task SDPO.",
            )
            parser.add_argument(
                "--sdpo-context-prompt-style",
                type=str,
                choices=["legacy", "controlled"],
                default="legacy",
                help="Use legacy SDPO context prompts or controlled prompts for context ablations.",
            )
            parser.add_argument(
                "--sdpo-own-outcome-label-mode",
                type=str,
                choices=["explicit", "omitted"],
                default="explicit",
                help="Whether controlled own-outcome prompts explicitly label trajectory success or failure.",
            )
            parser.add_argument(
                "--sdpo-no-success-context-mode",
                type=str,
                choices=["feedback", "filter", "failed_negative"],
                default="feedback",
                help="How agent-task SDPO handles task groups with no successful trajectory.",
            )
            parser.add_argument(
                "--sdpo-solution-context-format",
                type=str,
                choices=["trajectory_demo", "guidance_plan"],
                default="guidance_plan",
                help="Use raw successful trajectory demos or generated guidance summaries in SDPO teacher prompts.",
            )
            parser.add_argument(
                "--sdpo-guidance-generation-mode",
                type=str,
                choices=["disabled", "model_summary"],
                default="model_summary",
                help="How to turn SDPO guidance summary prompts into teacher guidance text.",
            )
            parser.add_argument(
                "--sdpo-guidance-summary-source",
                type=str,
                choices=["success_priority", "self_trajectory"],
                default="success_priority",
                help="Which trajectory source produces SDPO guidance summary prompts.",
            )
            parser.add_argument(
                "--sdpo-guidance-summary-max-tokens",
                type=int,
                default=256,
                help="Maximum new tokens for SDPO guidance summary generation.",
            )
            parser.add_argument(
                "--sdpo-guidance-summary-max-prompt-tokens",
                type=int,
                default=40192,
                help="Maximum rendered prompt tokens for SDPO guidance summary generation.",
            )
            parser.add_argument(
                "--sdpo-guidance-summary-timeout",
                type=float,
                default=120.0,
                help="Per-request timeout in seconds for SDPO guidance summary generation.",
            )
            parser.add_argument(
                "--agent-task-sdpo-guidance-summary-concurrency",
                type=int,
                default=0,
                help="Concurrent SGLang requests for SDPO guidance summaries; 0 uses endpoint count.",
            )
            parser.add_argument(
                "--sdpo-guidance-debug-enabled",
                action=argparse.BooleanOptionalAction,
                default=False,
                help="Write local JSONL debug records for SDPO guidance summary prompts and outputs.",
            )
            parser.add_argument(
                "--sdpo-guidance-debug-dir",
                type=str,
                default=None,
                help="Directory for local SDPO guidance summary debug JSONL gzip files.",
            )
            parser.add_argument(
                "--sdpo-guidance-debug-max-records",
                type=int,
                default=0,
                help="Maximum guidance debug records to write per converter call; <=0 writes all unique prompts.",
            )
            parser.add_argument(
                "--agent-task-sdpo-guidance-sglang-url",
                type=str,
                default=None,
                help="Optional SGLang endpoint used for SDPO guidance summaries.",
            )
            parser.add_argument(
                "--agent-task-sdpo-guidance-sglang-urls",
                type=str,
                default=None,
                help="Optional comma/space separated SGLang endpoints used for SDPO guidance summaries.",
            )
            parser.add_argument(
                "--sdpo-multi-turn-weighting",
                type=str,
                choices=["traj_equal", "step_equal", "hybrid", "inverse_length"],
                default="traj_equal",
                help="Per-row SDPO loss weight policy; inverse_length uses 1 / source episode length.",
            )
            parser.add_argument(
                "--sdpo-max-demo-steps",
                type=int,
                default=0,
                help="Maximum trajectory steps included in SDPO demos; <=0 keeps all available steps.",
            )
            parser.add_argument(
                "--sdpo-include-environment-feedback",
                action=argparse.BooleanOptionalAction,
                default=True,
                help="Include one-step environment feedback in SDPO teacher prompts when applicable.",
            )
            parser.add_argument(
                "--sdpo-environment-feedback-only-without-solution",
                action=argparse.BooleanOptionalAction,
                default=True,
                help="Use environment feedback only when no solution/failure guidance is present.",
            )
            parser.add_argument(
                "--sdpo-max-reprompt-tokens",
                type=int,
                default=10240,
                help="Maximum teacher prompt tokens kept when SDPO retokenizes guidance prompts.",
            )
            parser.add_argument(
                "--sdpo-reprompt-truncation-side",
                type=str,
                choices=["left", "right"],
                default="right",
                help="Truncation side for overlength SDPO teacher prompts.",
            )
            parser.add_argument(
                "--sdpo-clip-ratio",
                type=float,
                default=None,
                help="Optional upper cap for the detached SDPO current-policy / old-policy token ratio.",
            )
            parser.add_argument(
                "--sdpo-deployment-tis-clip",
                type=float,
                default=None,
                help="Optional upper cap for the detached SDPO old-policy / deployment-policy token ratio.",
            )
            parser.add_argument(
                "--sdpo-train-diagnostics",
                action="store_true",
                default=False,
                help="Enable extra SDPO train diagnostics where available.",
            )
            parser.add_argument(
                "--agent-task-sdpo-enabled",
                action="store_true",
                default=False,
                help="Allow a task-provided SDPO converter that emits SDPO teacher prompts, masks, and weights.",
            )
            parser.add_argument(
                "--custom-loss-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom loss function, if the loss_type is `custom_loss`, "
                    "we will use this function to calculate the loss. "
                ),
            )
            parser.add_argument(
                "--kl-loss-type",
                type=str,
                choices=["k1", "k2", "k3", "low_var_kl"],
                default="k1",
                help="Choose KL loss type: kl, k2, k3, low_var_kl",
            )
            parser.add_argument(
                "--advantage-estimator",
                type=str,
                choices=[
                    "grpo",
                    "gspo",
                    "reinforce_plus_plus",
                    "reinforce_plus_plus_baseline",
                    "ppo",
                ],
                default="grpo",
                help=(
                    "Advantage estimator to use. Note: on-policy distillation (OPD) is now orthogonal "
                    "to the advantage estimator. Use --opd-kl-coef > 0 to enable OPD on top of any estimator."
                ),
            )
            parser.add_argument(
                "--disable-compute-advantages-and-returns",
                action="store_false",
                dest="compute_advantages_and_returns",
                help=(
                    "Whether to disable computing advantages and returns. "
                    "If set, we will not compute the advantages and returns, "
                    "This is useful for sft or custom loss function."
                ),
            )
            parser.add_argument(
                "--custom-advantage-function-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom advantage/returns computation function. "
                    "When set, this function replaces the built-in compute_advantages_and_returns. "
                    "Signature: def custom_fn(args, rollout_data) -> None. "
                    "The function should set rollout_data['advantages'] and rollout_data['returns'] in-place. "
                    "Critic values are available in rollout_data['values']. "
                    "(e.g., my_module.py:my_advantage_fn)."
                ),
            )
            parser.add_argument(
                "--use-kl-loss", action="store_true", default=False, help="whether to use KL loss from GRPO"
            )
            parser.add_argument(
                "--kl-loss-coef",
                type=float,
                default=0.0,
                help="KL penalty coefficient for the loss function. This is added to the final PPO loss.",
            )
            parser.add_argument(
                "--use-unbiased-kl",
                action="store_true",
                default=False,
                help="Whether to enable unbiased KL estimation.",
            )
            parser.add_argument(
                "--ref-update-interval",
                type=int,
                default=None,
                help="Interval (in rollout steps) to update ref model from actor. If None, ref model is not updated.",
            )
            parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy loss coef")
            parser.add_argument("--gamma", type=float, default=1.0, help="PPO GAE gamma")
            parser.add_argument("--lambd", type=float, default=1.0, help="PPO GAE lambd")
            parser.add_argument("--normalize-advantages", action="store_true", default=False)
            parser.add_argument(
                "--disable-grpo-std-normalization",
                action="store_false",
                dest="grpo_std_normalization",
                help="from Dr.GRPO https://arxiv.org/pdf/2503.20783",
            )
            parser.add_argument(
                "--disable-rewards-normalization",
                action="store_false",
                dest="rewards_normalization",
                help="Disable rewards normalization",
            )
            parser.add_argument(
                "--use-rollout-entropy",
                action="store_true",
                default=False,
                help=(
                    "Whether to calculate the entropy when calculating the logprobs from actor and reference model. "
                    "This is useful for doing special loss mask."
                ),
            )
            parser.add_argument(
                "--get-mismatch-metrics",
                action="store_true",
                default=False,
                help="Whether to calculate the mismatch metrics.",
            )
            parser.add_argument(
                "--reset-optimizer-states",
                action="store_true",
                default=False,
                help=(
                    "Whether to reset optimizer states after each rollout. "
                    "If enabled, the optimizer's history will be cleared at the end of each rollout, which can sometimes help with training stability or fulfill specific experiment requirements."
                ),
            )
            parser.add_argument(
                "--use-rollout-logprobs",
                action="store_true",
                default=False,
                help=(
                    "Whether to use the rollout logprobs when calculating the importance sampling ratios. "
                    "If not set, we will use the logprobs from the actor model."
                ),
            )
            # Off-Policy Correction using Importance Sampling: https://fengyao.notion.site/off-policy-rl
            parser.add_argument(
                "--use-tis",
                action="store_true",
                default=False,
                help="Enable TIS from https://fengyao.notion.site/off-policy-rl for off-policy importance sampling.",
            )
            parser.add_argument(
                "--tis-clip",
                type=float,
                default=2.0,
                help="Clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--tis-clip-low",
                type=float,
                default=0,
                help="Lower bound clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--custom-tis-function-path",
                type=str,
                default=None,
                help="Path to the custom TIS/RS function (e.g., examples/train_infer_mismatch_helper/mis.py:compute_mis_weights_with_cp).",
            )
            parser.add_argument(
                "--custom-pg-loss-reducer-function-path",
                type=str,
                default=None,
                help="Path to a custom reducer function for pg_loss only. When set, pg_loss will use this custom reducer while other metrics (pg_clipfrac, ppo_kl, entropy_loss, etc.) still use the default sum_of_sample_mean. (e.g., examples/Dr.GRPO/custom_reducer.py:get_pg_loss_reducer).",
            )

            parser.add_argument(
                "--use-routing-replay",
                action="store_true",
                default=False,
                help="The routing replay technique from https://arxiv.org/abs/2507.18071",
            )
            parser.add_argument(
                "--use-rollout-routing-replay",
                action="store_true",
                default=False,
                help="The rollout routing replay technique from https://arxiv.org/abs/2510.11370",
            )
            parser.add_argument(
                "--use-opsm",
                action="store_true",
                default=False,
                help="Whether to enable Off-Policy Sequence Masking (OPSM).",
            )
            parser.add_argument(
                "--opsm-delta",
                type=float,
                default=1e-4,
                help="The threshold for Off-Policy Sequence Masking (OPSM).",
            )
            return parser

        def add_on_policy_distillation_arguments(parser):
            """Add on-policy distillation (OPD) related arguments.

            OPD is orthogonal to advantage estimators and can be applied on top of
            any estimator (GRPO, PPO, etc.) by adding a KL penalty to advantages.
            """
            parser.add_argument(
                "--use-opd",
                action="store_true",
                default=False,
                help="Enable on-policy distillation (OPD). Must specify --opd-type when enabled.",
            )
            parser.add_argument(
                "--opd-type",
                type=str,
                choices=["sglang", "megatron"],
                default=None,
                help=(
                    "Type of on-policy distillation. "
                    "'sglang': Teacher log-probs are obtained from external SGLang server during rollout. "
                    "'megatron': Teacher model is loaded via --opd-teacher-load and forwarded during training."
                ),
            )
            parser.add_argument(
                "--normalize-over-active-rollouts",
                action="store_true",
                default=False,
                help=(
                    "Normalize train loss and scheduler sample increments over rollout IDs with non-zero loss masks. "
                    "Use with a sample filter when masked rollouts must not dilute policy or auxiliary losses."
                ),
            )
            parser.add_argument(
                "--opd-kl-coef",
                type=float,
                default=1.0,
                help="On-policy distillation KL penalty coefficient. Default is 1.0.",
            )
            parser.add_argument(
                "--opd-teacher-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for OPD teacher model. Required when --opd-type=megatron. "
                    "The teacher model should have the same architecture as policy/ref model."
                ),
            )
            parser.add_argument(
                "--opd-teacher-ckpt-step", type=int, default=None, help="The checkpoint step for OPD teacher model."
            )
            return parser

        def add_router_arguments(parser):
            parser.add_argument(
                "--use-slime-router",
                action="store_true",
                default=False,
                help="Whether to use SlimeRouter for text-based routing instead of SGLang token-based routing",
            )
            RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
            return parser

        # wandb
        def add_wandb_arguments(parser):
            # wandb parameters
            parser.add_argument("--use-wandb", action="store_true", default=False)
            parser.add_argument(
                "--wandb-mode",
                type=str,
                default=None,
                choices=["online", "offline", "disabled"],
                help="W&B mode: online (default), offline (local only), or disabled. Overrides WANDB_MODE env var.",
            )
            parser.add_argument(
                "--wandb-dir",
                type=str,
                default=None,
                help="Directory to store wandb logs. Default is ./wandb in current directory.",
            )
            parser.add_argument("--wandb-key", type=str, default=None)
            parser.add_argument("--wandb-host", type=str, default=None)
            parser.add_argument("--wandb-team", type=str, default=None)
            parser.add_argument("--wandb-group", type=str, default=None)
            reset_arg(parser, "--wandb-project", type=str, default=None)
            parser.add_argument(
                "--disable-wandb-random-suffix",
                action="store_false",
                dest="wandb_random_suffix",
                default=True,
                help=(
                    "Whether to add a random suffix to the wandb run name. "
                    "By default, we will add a random 6 length string with characters to the run name."
                ),
            )
            parser.add_argument(
                "--wandb-always-use-train-step",
                action="store_true",
                default=False,
                help=(
                    "Whether to always use train step as the step metric in wandb. "
                    "If set, we will always use the train steps for wandb logging, "
                    "otherwise, will use rollout step for most info other than train/*. "
                ),
            )
            parser.add_argument(
                "--log-multi-turn",
                action="store_true",
                default=False,
                help="Whether to log information for multi-turn rollout.",
            )
            parser.add_argument(
                "--log-passrate",
                action="store_true",
                default=False,
                help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
            )
            parser.add_argument(
                "--log-reward-category",
                type=str,
                default=None,
                help=(
                    "Log statistics of the category of reward, such as why the reward function considers it as failed. "
                    "Specify the key in the reward dict using this argument."
                ),
            )
            parser.add_argument(
                "--log-correct-samples",
                action="store_true",
                default=False,
                help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
            )
            parser.add_argument("--wandb-run-id", type=str, default=None)
            return parser

        # tensorboard
        def add_tensorboard_arguments(parser):
            # tb_project_name, tb_experiment_name
            parser.add_argument("--use-tensorboard", action="store_true", default=False)
            parser.add_argument(
                "--tb-project-name",
                type=str,
                default=None,
                help="Directory to store tensorboard logs. Default is  os.environ.get('TENSORBOARD_DIR') directory.",
            )
            parser.add_argument("--tb-experiment-name", type=str, default=None)

            return parser

        # debug
        def add_debug_arguments(parser):
            parser.add_argument(
                "--save-debug-rollout-data",
                type=str,
                default=None,
                help=(
                    "Save the rollout data to this path for debugging. "
                    "The file will be saved to `save_debug_rollout_data.format(rollout_id)`."
                ),
            )
            # --load-debug-rollout-data, --debug-rollout-only, --debug-train-only
            # are parsed early in _pre_parse_mode() and merged later.
            parser.add_argument(
                "--load-forge-rollout-data",
                type=str,
                default=None,
                help=(
                    "Path (or {rollout_id} template) to a dumped rollout .pt file replayed by "
                    "slime.rollout.forge_load.generate_rollout. Mirrors --load-debug-rollout-data's "
                    "format(rollout_id=...) convention: a path without the placeholder is treated as "
                    "a literal file and reused across every rollout_id; a path containing {rollout_id} "
                    "loads a per-rollout file (with eval_<id>.pt for the eval pipeline). Unlike "
                    "--load-debug-rollout-data, this does NOT force debug_train_only / skip_sglang -- "
                    "sglang servers, router, weight_update and the colocate offload/onload dance all "
                    "stay live, which is the point (memory measurement at long context)."
                ),
            )
            parser.add_argument(
                "--load-debug-rollout-data-subsample",
                type=float,
                default=None,
                help="Subsample a portion of the debug rollout data for faster debugging.",
            )
            parser.add_argument(
                "--save-debug-train-data",
                type=str,
                default=None,
                help=(
                    "Save the train data to this path for debugging. "
                    "Use {rollout_id} and {rank} placeholders to save one file per rollout and distributed rank."
                ),
            )
            parser.add_argument(
                "--save-debug-train-data-include-sdpo-token-weights",
                action="store_true",
                default=False,
                help=(
                    "Keep raw SDPO token weights in debug train-data dumps. Requires a "
                    "--save-debug-train-data path with {rank}; include {rollout_id} to avoid rollout overwrites."
                ),
            )
            parser.add_argument(
                "--dump-details",
                type=str,
                default=None,
                help=("Dump all details of training for post-hoc analysis and visualization."),
            )
            # use together with --record-memory-history and --memory-snapshot-path (defined in Megatron)
            parser.add_argument(
                "--memory-snapshot-dir",
                type=str,
                default=".",
            )
            parser.add_argument(
                "--memory-snapshot-num-steps",
                type=int,
                default=None,
            )
            parser.add_argument(
                "--profile-target",
                type=str,
                choices=["train_overall", "train_actor", "train_log_probs"],
                default=["train_overall"],
                nargs="+",
            )
            parser.add_argument(
                "--memory-recorder",
                type=str,
                choices=["torch", "memray"],
                default="torch",
            )
            reset_arg(parser, "--record-memory-history", action="store_true", default=False)
            parser.add_argument("--check-weight-update-equal", action="store_true")
            return parser

        def add_network_arguments(parser):
            parser.add_argument("--http-proxy", type=str, default=None)
            parser.add_argument("--use-distributed-post", action="store_true", default=False)
            return parser

        def add_reward_model_arguments(parser):
            parser.add_argument(
                "--rm-type",
                type=str,
                default=None,
                help="Type of the reward model",
            )
            parser.add_argument(
                "--reward-key",
                type=str,
                default=None,
                help=(
                    "Some reward model may return a dict instead of a value, "
                    "this is the key to extract the reward value from the dict. "
                ),
            )
            parser.add_argument(
                "--eval-reward-key",
                type=str,
                default=None,
                help="The eval variant for --reward-key",
            )
            parser.add_argument(
                "--group-rm", action="store_true", default=False, help="Whether to do rm on a whole group."
            )
            parser.add_argument(
                "--rm-url",
                type=str,
                default=None,
                help="URL for the reward model service for --rm-type remote_rm, e.g. http://localhost:8000",
            )
            parser.add_argument(
                "--custom-rm-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom reward model function. "
                    "If set, we will use this function to calculate the reward instead of the default one. "
                    "The function should have the signature `def custom_rm(args, sample) -> float`."
                ),
            )
            parser.add_argument(
                "--custom-reward-post-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom function that will post process reward, by default it will be the normalization for grpo. "
                ),
            )
            parser.add_argument(
                "--custom-convert-samples-to-train-data-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom function that converts samples to training data. "
                    "If set, this function will replace the default _convert_samples_to_train_data. "
                    "The function should have the signature `def convert_samples_to_train_data(args, samples) -> dict`."
                ),
            )
            return parser

        def add_rollout_buffer_arguments(parser):
            parser.add_argument(
                "--rollout-buffer-url",
                type=str,
                default=None,
                help="URL for the rollout buffer",
            )

            parser.add_argument(
                "--fetch-trajectory-retry-times",
                type=int,
                default=-1,
                help="Number of times to retry fetching trajectory, -1 means unlimited retry",
            )
            parser.add_argument(
                "--min-batch-collection-ratio",
                type=float,
                default=1,
                help="Minimum batch collection ratio",
            )
            parser.add_argument(
                "--rollout-task-type",
                type=str,
                default="math",
            )
            parser.add_argument(
                "--loss-mask-type",
                type=str,
                default="qwen",
                choices=["qwen", "qwen3", "qwen3_5", "distill_qwen"],
                help="Loss mask type",
            )
            parser.add_argument(
                "--data-pad-size-multiplier",
                type=int,
                default=128,
                help="Multiplier for data padding size in data processing.",
            )
            parser.add_argument(
                "--rollout-sample-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout sample filter function. "
                    "This function determines whether a sample will participate in loss calculation. "
                    "The function should take args and samples (list[Sample]) as input, and return None. "
                    "Please directly modify the remove_sample attribute of Sample. "
                    "Note: This attribute does not determine whether the sample participates in advantage normalization."
                ),
            )
            parser.add_argument(
                "--rollout-all-samples-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout all samples process function that "
                    "can process all samples including filtered ones."
                ),
            )
            return parser

        def add_custom_megatron_plugins_arguments(parser):
            """
            Add custom Megatron plugins arguments.
            This is a placeholder for any additional arguments that might be needed.
            """
            # Custom arguments can be added here
            parser.add_argument(
                "--custom-megatron-init-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-log-prob-hook-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-train-step-hook-path",
                type=str,
                default=None,
            )
            return parser

        def add_mtp_training_arguments(parser):
            """Add MTP training specific arguments."""
            reset_arg(parser, "--mtp-num-layers", type=int, default=None)
            reset_arg(parser, "--mtp-loss-scaling-factor", type=float, default=0.2)
            parser.add_argument(
                "--enable-mtp-training",
                action="store_true",
                default=False,
                help="Enable MTP layer parameter updates during training",
            )

            return parser

        def add_ci_arguments(parser):
            parser.add_argument(
                "--ci-test",
                action="store_true",
            )
            parser.add_argument(
                "--ci-disable-kl-checker",
                action="store_true",
            )
            parser.add_argument(
                "--ci-save-grad-norm",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--ci-load-grad-norm",
                type=str,
                default=None,
            )
            return parser

        # Add custom arguments in front to prevent overwritten some slime arguments.
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)

        parser = add_cluster_arguments(parser)
        parser = add_train_arguments(parser)
        parser = add_rollout_arguments(parser)
        parser = add_fault_tolerance_arguments(parser)
        parser = add_data_arguments(parser)
        parser = add_eval_arguments(parser)
        parser = add_algo_arguments(parser)
        parser = add_on_policy_distillation_arguments(parser)
        parser = add_wandb_arguments(parser)
        parser = add_tensorboard_arguments(parser)
        parser = add_router_arguments(parser)
        parser = add_debug_arguments(parser)
        parser = add_network_arguments(parser)
        parser = add_reward_model_arguments(parser)
        parser = add_rollout_buffer_arguments(parser)
        parser = add_mtp_training_arguments(parser)
        parser = add_ci_arguments(parser)
        parser = add_custom_megatron_plugins_arguments(parser)
        reset_arg(
            parser,
            "--custom-config-path",
            type=str,
            default=None,
            help="Path to the YAML config for custom function arguments.",
        )
        reset_arg(parser, "--padded-vocab-size", type=int, default=None)

        return parser

    return add_slime_arguments


def _pre_parse_mode():
    """Pre-parse CLI to extract arguments that control parsing flow.

    These arguments are removed from add_slime_arguments to avoid
    registering them twice.  The returned namespace is merged into
    the final ``args`` after Phase 2 parsing.
    """
    temp_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    temp_parser.add_argument("--train-backend", type=str, choices=["megatron"], default="megatron")
    temp_parser.add_argument("--debug-rollout-only", action="store_true", default=False)
    temp_parser.add_argument("--debug-train-only", action="store_true", default=False)
    temp_parser.add_argument("--load-debug-rollout-data", type=str, default=None)
    temp_args, _ = temp_parser.parse_known_args()
    return temp_args


def parse_args(add_custom_arguments=None):
    # Users may call `parse_args` very early, thus we ensure logger is configured here
    configure_logger()

    add_slime_arguments = get_slime_extra_args_provider(add_custom_arguments)

    pre = _pre_parse_mode()
    skip_sglang = pre.debug_train_only or pre.load_debug_rollout_data is not None

    # Phase 1: Parse sglang args independently (separate parser, parse_known_args).
    # Skipped when sglang servers are not needed.
    sglang_ns = None
    if not skip_sglang:
        sglang_ns = sglang_parse_args()

    # Phase 2: Parse megatron + slime args.
    # Uses ignore_unknown_args=True so that --sglang-* and pre-parsed CLI flags
    # are silently ignored by the megatron parser.
    from slime.backends.megatron_utils.arguments import megatron_parse_args
    from slime.backends.megatron_utils.arguments import validate_args as megatron_validate_args

    args = megatron_parse_args(
        extra_args_provider=add_slime_arguments,
        skip_hf_validate=pre.debug_rollout_only,
    )

    # Merge pre-parsed args into the main namespace
    for key, value in vars(pre).items():
        setattr(args, key, value)

    # Merge sglang args into the main namespace
    if sglang_ns is not None:
        for key, value in vars(sglang_ns).items():
            setattr(args, key, value)

    slime_validate_args(args)

    if pre.train_backend == "megatron" and not args.debug_rollout_only:
        megatron_validate_args(args)

    if not args.debug_train_only:
        sglang_validate_args(args)

    return args


def _apply_megatron_role_overrides(base_args, overrides, role):
    role_args = copy.deepcopy(base_args)
    ignored_keys = {"num_nodes", "num_gpus_per_node"}

    # Apply overrides from the YAML config.
    # Unspecified keys inherit from base_args via deepcopy.
    for key, value in overrides.items():
        if key in ignored_keys:
            logger.info(f"Ignoring {role} config key '{key}'; GPU allocation always follows CLI args.")
            continue
        if not hasattr(role_args, key):
            logger.warning(f"{role.capitalize()} config key '{key}' is not a known argument, setting it anyway.")
        else:
            # YAML safe_load doesn't parse scientific notation (e.g. 1e-5) as float.
            # Coerce the value to match the type of the existing attribute.
            original = getattr(role_args, key)
            if original is not None and isinstance(value, str) and isinstance(original, (int, float)):
                try:
                    value = type(original)(value)
                except (ValueError, TypeError):
                    pass
        setattr(role_args, key, value)

    if role == "critic":
        # Critic-specific: disable features that only apply to actors.
        role_args.kl_coef = 0
        role_args.use_opd = False
        role_args.custom_advantage_function_path = None
        role_args.untie_embeddings_and_output_weights = True
        if "disable_param_buffers_cpu_backup" not in overrides:
            role_args.disable_param_buffers_cpu_backup = False

    return role_args


def parse_megatron_role_args(base_args, megatron_config_path, role):
    """Parse role-specific arguments from a unified Megatron YAML config.

    The config must contain a top-level ``megatron`` list with per-role entries.
    Missing roles inherit the base args unchanged.
    """
    assert role in {"actor", "critic"}, f"Unsupported Megatron config role: {role}"

    with open(megatron_config_path) as f:
        raw_config = yaml.safe_load(f) or {}

    assert "megatron" in raw_config, (
        "megatron config must contain a top-level 'megatron' list, e.g. "
        "megatron: [{name: default, role: actor, overrides: {...}}]"
    )

    overrides = {}
    megatron_entries = raw_config["megatron"]
    assert isinstance(megatron_entries, list), (
        "megatron config 'megatron' field must be a list, e.g. "
        "megatron: [{name: default, role: actor, overrides: {...}}]"
    )
    role_entries = [entry for entry in megatron_entries if entry.get("role") == role]
    assert len(role_entries) <= 1, (
        f"megatron config must contain at most one entry with role={role}, e.g. "
        f"megatron: [{{name: default, role: {role}, overrides: {{...}}}}]"
    )
    if role_entries:
        role_entry = role_entries[0]
        overrides = role_entry.get("overrides") or role_entry.get("args") or {}
    else:
        logger.info(
            f"No megatron config entry with role={role} found in {megatron_config_path}; using inherited args."
        )

    role_args = _apply_megatron_role_overrides(base_args, overrides, role)
    if role == "actor":
        _validate_sdpo_args(role_args)
    logger.info(
        f"Parsed megatron config for role={role} from {megatron_config_path}: overrides = {list(overrides.keys())}"
    )

    return role_args


def parse_critic_args(actor_args, megatron_config_path):
    """Backward-compatible wrapper for critic-specific Megatron role parsing."""
    return parse_megatron_role_args(actor_args, megatron_config_path, role="critic")


def _resolve_eval_datasets(args) -> list[EvalDatasetConfig]:
    """
    Build evaluation dataset configurations from either --eval-config or --eval-prompt-data.
    """
    datasets_config = []
    defaults: dict[str, Any] = {}

    if args.eval_config:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.eval_config)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise ValueError("--eval-config must contain a mapping at the root.")

        eval_cfg = cfg_dict.get("eval", cfg_dict)
        if not isinstance(eval_cfg, dict):
            raise ValueError("--eval-config must define an `eval` mapping or be a mapping itself.")

        defaults = dict(eval_cfg.get("defaults") or {})
        datasets_config = ensure_dataset_list(eval_cfg.get("datasets"))
        if not datasets_config:
            raise ValueError("--eval-config does not define any datasets under `eval.datasets`.")
    elif args.eval_prompt_data:
        values = list(args.eval_prompt_data)
        if len(values) == 1:
            logger.info("[legacy] only one eval_prompt_data detected, will assume it is data for aime")
            values = ["aime", values[0]]
        if len(values) % 2 != 0:
            raise ValueError("eval prompt data must be provided as name/path pairs.")
        datasets_config = [{"name": values[i], "path": values[i + 1]} for i in range(0, len(values), 2)]
    else:
        datasets_config = []

    eval_datasets = build_eval_dataset_configs(args, datasets_config, defaults)
    if eval_datasets:
        args.eval_prompt_data = [item for dataset in eval_datasets for item in (dataset.name, dataset.path)]
    else:
        args.eval_prompt_data = None

    return eval_datasets


def _validate_sdpo_args(args) -> None:
    if getattr(args, "loss_type", None) != "sdpo_loss":
        return

    convert_path = getattr(args, "custom_convert_samples_to_train_data_path", None) or ""
    if (
        "slime_plugins.agent_tasks.common.algorithms.sdpo.convert_samples_to_train_data" not in convert_path
        and not getattr(args, "agent_task_sdpo_enabled", False)
    ):
        raise ValueError(
            "--loss-type=sdpo_loss requires "
            "--custom-convert-samples-to-train-data-path="
            "slime_plugins.agent_tasks.common.algorithms.sdpo.convert_samples_to_train_data "
            "or an agent task that sets agent_task_sdpo_enabled=True and emits SDPO teacher prompts, "
            "masks, and weights."
        )
    if not 0.0 <= float(args.sdpo_teacher_update_rate) <= 1.0:
        raise ValueError("--sdpo-teacher-update-rate must be in [0, 1].")
    if not 0.0 <= float(args.sdpo_alpha) <= 1.0:
        raise ValueError("--sdpo-alpha must be in [0, 1].")
    pr_weight = float(getattr(args, "pr_weight", 0.0) or 0.0)
    if not math.isfinite(pr_weight) or pr_weight < 0.0:
        raise ValueError("--pr-weight must be finite and non-negative.")
    pr_support = getattr(args, "pr_support", "selected")
    if pr_support not in {"selected", "all"}:
        raise ValueError("--pr-support must be selected or all.")
    full_selection_retention = bool(getattr(args, "sgs_full_selection_retention", False))
    if full_selection_retention:
        full_selection_fraction = getattr(args, "sgs_selection_fraction", None)
        if (
            full_selection_fraction is None
            or not math.isfinite(float(full_selection_fraction))
            or float(full_selection_fraction) != 1.0
            or pr_weight <= 0.0
            or pr_support != "all"
        ):
            raise ValueError(
                "full-selection retention requires fraction 1, positive retention weight, and all-step support"
            )
    pr_direction = getattr(args, "pr_kl_direction", "reverse")
    if pr_direction not in {"reverse", "forward"}:
        raise ValueError("--pr-kl-direction must be reverse or forward.")
    retention_view = getattr(args, "pr_view", "privileged")
    if retention_view not in {"privileged", "ordinary"}:
        raise ValueError("--pr-view must be privileged or ordinary.")
    sdpo_clip_ratio = getattr(args, "sdpo_clip_ratio", None)
    if sdpo_clip_ratio is not None and (not math.isfinite(float(sdpo_clip_ratio)) or float(sdpo_clip_ratio) <= 0.0):
        raise ValueError("--sdpo-clip-ratio must be a finite positive value.")
    sdpo_deployment_tis_clip = getattr(args, "sdpo_deployment_tis_clip", None)
    if sdpo_deployment_tis_clip is not None and (
        not math.isfinite(float(sdpo_deployment_tis_clip)) or float(sdpo_deployment_tis_clip) <= 0.0
    ):
        raise ValueError("--sdpo-deployment-tis-clip must be a finite positive value.")
    if not math.isfinite(float(args.sdpo_representation_coef)) or float(args.sdpo_representation_coef) < 0.0:
        raise ValueError("--sdpo-representation-coef must be a finite non-negative value.")
    if args.sdpo_representation_reduction not in {"mean", "sum"}:
        raise ValueError("--sdpo-representation-reduction must be mean or sum.")
    if args.sdpo_representation_success_aggregation not in {"sample", "mean", "random"}:
        raise ValueError("--sdpo-representation-success-aggregation must be sample, mean, or random.")
    if args.sdpo_ema_teacher_keep_last < 0:
        raise ValueError("--sdpo-ema-teacher-keep-last must be non-negative.")
    if args.sdpo_max_reprompt_tokens <= 0:
        raise ValueError("--sdpo-max-reprompt-tokens must be positive.")
    if float(args.sdpo_success_reward_threshold) < 0.0:
        raise ValueError("--sdpo-success-reward-threshold must be non-negative.")
    if args.sdpo_max_demo_steps is not None and int(args.sdpo_max_demo_steps) < 0:
        raise ValueError("--sdpo-max-demo-steps must be non-negative.")
    if args.sdpo_guidance_summary_max_tokens <= 0:
        raise ValueError("--sdpo-guidance-summary-max-tokens must be positive.")
    if args.sdpo_guidance_summary_max_prompt_tokens <= 0:
        raise ValueError("--sdpo-guidance-summary-max-prompt-tokens must be positive.")
    if args.sdpo_guidance_summary_timeout <= 0:
        raise ValueError("--sdpo-guidance-summary-timeout must be positive.")
    if args.agent_task_sdpo_guidance_summary_concurrency < 0:
        raise ValueError("--agent-task-sdpo-guidance-summary-concurrency must be non-negative.")
    if args.sdpo_guidance_debug_max_records < 0:
        raise ValueError("--sdpo-guidance-debug-max-records must be non-negative.")
    if args.sdpo_teacher_context_mode not in {
        "original",
        "feedback_only",
        "solution_only",
        "own_outcome",
        "hindsight",
    }:
        raise ValueError(
            "--sdpo-teacher-context-mode must be original, feedback_only, solution_only, own_outcome, or hindsight."
        )
    sdpo_context_prompt_style = getattr(args, "sdpo_context_prompt_style", "legacy")
    sdpo_own_outcome_label_mode = getattr(args, "sdpo_own_outcome_label_mode", "explicit")
    if sdpo_context_prompt_style not in {"legacy", "controlled"}:
        raise ValueError("--sdpo-context-prompt-style must be legacy or controlled.")
    if sdpo_own_outcome_label_mode not in {"explicit", "omitted"}:
        raise ValueError("--sdpo-own-outcome-label-mode must be explicit or omitted.")
    if sdpo_context_prompt_style == "controlled" and args.sdpo_teacher_context_mode not in {
        "feedback_only",
        "own_outcome",
        "hindsight",
    }:
        raise ValueError(
            "--sdpo-context-prompt-style=controlled requires "
            "--sdpo-teacher-context-mode=feedback_only, own_outcome, or hindsight."
        )
    if sdpo_own_outcome_label_mode == "omitted" and not (
        sdpo_context_prompt_style == "controlled" and args.sdpo_teacher_context_mode == "own_outcome"
    ):
        raise ValueError(
            "--sdpo-own-outcome-label-mode=omitted requires "
            "--sdpo-context-prompt-style=controlled and --sdpo-teacher-context-mode=own_outcome."
        )
    if args.sdpo_no_success_context_mode not in {"feedback", "filter", "failed_negative"}:
        raise ValueError("--sdpo-no-success-context-mode must be feedback, filter, or failed_negative.")
    if args.sdpo_filter_all_success_groups and args.sdpo_no_success_context_mode != "filter":
        raise ValueError("--sdpo-filter-all-success-groups requires --sdpo-no-success-context-mode=filter.")
    if args.sdpo_solution_context_format not in {"trajectory_demo", "guidance_plan"}:
        raise ValueError("--sdpo-solution-context-format must be trajectory_demo or guidance_plan.")
    if args.sdpo_teacher_context_mode == "own_outcome" and not (
        args.sdpo_solution_context_format == "trajectory_demo"
        or (
            args.sdpo_solution_context_format == "guidance_plan"
            and args.sdpo_guidance_summary_source == "self_trajectory"
        )
    ):
        raise ValueError(
            "--sdpo-teacher-context-mode=own_outcome requires trajectory_demo, or guidance_plan with "
            "--sdpo-guidance-summary-source=self_trajectory."
        )
    if args.sdpo_teacher_context_mode == "own_outcome" and args.sdpo_no_success_context_mode != "failed_negative":
        raise ValueError(
            "--sdpo-teacher-context-mode=own_outcome requires --sdpo-no-success-context-mode=failed_negative."
        )
    if args.sdpo_guidance_generation_mode not in {"disabled", "model_summary"}:
        raise ValueError("--sdpo-guidance-generation-mode must be disabled or model_summary.")
    if args.sdpo_guidance_summary_source not in {"success_priority", "self_trajectory"}:
        raise ValueError("--sdpo-guidance-summary-source must be success_priority or self_trajectory.")
    if args.sdpo_multi_turn_weighting not in {"traj_equal", "step_equal", "hybrid", "inverse_length"}:
        raise ValueError("--sdpo-multi-turn-weighting must be traj_equal, step_equal, hybrid, or inverse_length.")
    if args.sdpo_distillation_mode is None:
        args.sdpo_distillation_mode = "topk" if args.sdpo_full_logit_distillation else "sample_token"
    if not math.isfinite(float(args.sdpo_token_weights_power)):
        raise ValueError("--sdpo-token-weights-power must be a finite value.")
    if not math.isfinite(float(args.sdpo_token_weights_max)) or float(args.sdpo_token_weights_max) < 0.0:
        raise ValueError("--sdpo-token-weights-max must be 0 or a finite value greater than 1.")
    if 0.0 < float(args.sdpo_token_weights_max) <= 1.0:
        raise ValueError("--sdpo-token-weights-max must be 0 or a finite value greater than 1.")
    if args.sdpo_token_weights:
        tensor_model_parallel_size = int(getattr(args, "tensor_model_parallel_size", 1))
        context_parallel_size = int(getattr(args, "context_parallel_size", 1))
        pipeline_model_parallel_size = int(getattr(args, "pipeline_model_parallel_size", 1))
        if tensor_model_parallel_size != 1:
            raise ValueError(
                "SDPO token weights require --tensor-model-parallel-size 1; " f"got {tensor_model_parallel_size}."
            )
        if context_parallel_size != 1:
            raise ValueError("SDPO token weights require --context-parallel-size 1; " f"got {context_parallel_size}.")
        if pipeline_model_parallel_size != 1:
            raise ValueError(
                "SDPO token weights require --pipeline-model-parallel-size 1; " f"got {pipeline_model_parallel_size}."
            )
    if args.sdpo_representation_success_aggregation == "mean" and args.sdpo_distillation_mode != "representation":
        raise ValueError("--sdpo-representation-success-aggregation=mean requires representation distillation.")
    if args.sdpo_distillation_mode == "sample_token":
        args.sdpo_full_logit_distillation = False
    elif args.sdpo_distillation_mode == "representation":
        args.sdpo_full_logit_distillation = False
        if (
            args.sdpo_representation_success_aggregation == "mean"
            and args.sdpo_solution_context_format != "trajectory_demo"
        ):
            raise ValueError(
                "--sdpo-representation-success-aggregation=mean requires "
                "--sdpo-solution-context-format=trajectory_demo."
            )
        if args.sdpo_representation_layers != "last":
            raise ValueError("--sdpo-representation-layers=all is experimental and disabled in V1; use last.")
        tensor_model_parallel_size = int(getattr(args, "tensor_model_parallel_size", 1))
        context_parallel_size = int(getattr(args, "context_parallel_size", 1))
        if tensor_model_parallel_size != 1:
            raise ValueError(
                "SDPO representation distillation currently requires --tensor-model-parallel-size 1; "
                f"got {tensor_model_parallel_size}."
            )
        if context_parallel_size != 1:
            raise ValueError(
                "SDPO representation distillation currently requires --context-parallel-size 1; "
                f"got {context_parallel_size}."
            )
        if getattr(args, "overlap_moe_expert_parallel_comm", False):
            raise ValueError(
                "SDPO representation distillation does not support Megatron combined-1F1B "
                "(--overlap-moe-expert-parallel-comm) in V1."
            )
    elif args.sdpo_distillation_mode == "topk":
        args.sdpo_full_logit_distillation = True
        if args.sdpo_distillation_topk <= 0:
            raise ValueError("--sdpo-distillation-topk must be positive for --sdpo-distillation-mode=topk.")
    if pr_weight > 0.0:
        if args.sdpo_distillation_mode != "topk" or not bool(args.sdpo_distillation_add_tail):
            raise ValueError("privileged retention requires top-k+tail SDPO distillation")
        if args.sdpo_teacher_regularization != "ref" or args.ref_load is None:
            raise ValueError("privileged retention requires the frozen ref teacher")
        if "resail.convert_samples_to_train_data" not in convert_path:
            raise ValueError("privileged retention requires the sensitivity converter")
        sgs_fraction = getattr(args, "sgs_selection_fraction", None)
        if (
            sgs_fraction is None
            or not math.isfinite(float(sgs_fraction))
            or not 0.0 < float(sgs_fraction) <= 1.0
            or (float(sgs_fraction) == 1.0 and not full_selection_retention)
        ):
            raise ValueError(
                "privileged retention requires active sensitivity filtering with a fraction in (0, 1), or the explicit "
                "full-selection retention capability at fraction 1"
            )
        skip_full_scoring = getattr(args, "sgs_skip_full_scoring", None)
        selection_mode = getattr(args, "sgs_selection_mode", "sensitivity")
        unscored_random_all_step = (
            selection_mode == "random"
            and bool(skip_full_scoring)
            and pr_support == "all"
            and getattr(args, "sgs_selection_scope", "global") == "global"
        )
        if selection_mode != "sensitivity" and not unscored_random_all_step:
            raise ValueError(
                "retention requires sensitivity-ranked support or explicit unscored random all-step support"
            )
        score_scope = getattr(args, "sgs_score_scope", None)
        if (bool(skip_full_scoring) and not unscored_random_all_step) or score_scope != "response":
            raise ValueError("privileged retention requires full-response sensitivity scoring")
    if args.sdpo_full_logit_distillation:
        tensor_model_parallel_size = int(getattr(args, "tensor_model_parallel_size", 1))
        context_parallel_size = int(getattr(args, "context_parallel_size", 1))
        if tensor_model_parallel_size != 1:
            raise ValueError(
                "SDPO top-k distillation currently requires --tensor-model-parallel-size 1; "
                f"got {tensor_model_parallel_size}."
            )
        if context_parallel_size != 1:
            raise ValueError(
                "SDPO top-k distillation currently requires --context-parallel-size 1; "
                f"got {context_parallel_size}."
            )
    if args.sdpo_teacher_regularization == "ref":
        if args.ref_load is None:
            raise ValueError("--ref-load is required for --sdpo-teacher-regularization=ref.")
        if not os.path.exists(args.ref_load):
            raise FileNotFoundError(f"ref_load {args.ref_load} does not exist, please check the path.")


def _validate_grpo_token_weight_args(args) -> None:
    if not getattr(args, "grpo_token_weights", False):
        return
    if getattr(args, "loss_type", None) != "policy_loss":
        raise ValueError("--grpo-token-weights requires --loss-type=policy_loss.")
    if getattr(args, "advantage_estimator", None) != "grpo":
        raise ValueError("--grpo-token-weights requires --advantage-estimator=grpo.")
    if not getattr(args, "ref_load", None):
        raise ValueError("--grpo-token-weights requires --ref-load.")
    power = float(args.grpo_token_weights_power)
    if not math.isfinite(power) or power < 0.0:
        raise ValueError("--grpo-token-weights-power must be finite and non-negative.")
    max_weight = float(args.grpo_token_weights_max)
    if not math.isfinite(max_weight) or max_weight <= 1.0:
        raise ValueError("--grpo-token-weights-max must be finite and greater than 1.")
    for field, option in (
        ("tensor_model_parallel_size", "tensor-model-parallel-size"),
        ("context_parallel_size", "context-parallel-size"),
        ("pipeline_model_parallel_size", "pipeline-model-parallel-size"),
    ):
        if int(getattr(args, field, 1)) != 1:
            raise ValueError(f"GRPO token weights require --{option} 1.")


def _validate_active_rollout_normalization_args(args) -> None:
    if not getattr(args, "normalize_over_active_rollouts", False):
        return
    if getattr(args, "loss_type", None) != "policy_loss":
        raise ValueError("--normalize-over-active-rollouts currently requires --loss-type=policy_loss.")
    if not getattr(args, "rollout_sample_filter_path", None):
        raise ValueError("--normalize-over-active-rollouts requires --rollout-sample-filter-path.")


def _resolve_update_weight_disk_dir(args) -> None:
    """Normalize disk-sync directory args.

    ``--update-weight-delta-dir`` is kept only as a compatibility alias. New
    code should use ``--update-weight-disk-dir`` because the directory belongs
    to the transport, not to the delta encoding mode.
    """
    disk_dir = args.update_weight_disk_dir
    delta_dir = args.update_weight_delta_dir
    if disk_dir and delta_dir and disk_dir != delta_dir:
        raise ValueError(
            "--update-weight-delta-dir is deprecated alias for --update-weight-disk-dir; "
            "please set only one of them or set both to the same path."
        )

    if delta_dir:
        warnings.warn(
            "--update-weight-delta-dir is deprecated and will be removed in a future release; "
            "use --update-weight-disk-dir instead.",
            UserWarning,
            stacklevel=2,
        )

    disk_dir = disk_dir or delta_dir
    if args.update_weight_transport == "disk":
        if not disk_dir:
            raise ValueError(
                "--update-weight-transport=disk requires --update-weight-disk-dir to point at "
                "a filesystem shared between the trainer and the rollout engines."
            )
        args.update_weight_disk_dir = disk_dir
        args.update_weight_delta_dir = disk_dir


def _validate_update_weight_args(args) -> None:
    _resolve_update_weight_disk_dir(args)

    if args.update_weight_mode == "delta":
        if args.update_weight_transport not in ("nccl", "disk"):
            raise ValueError(
                "--update-weight-mode=delta supports only --update-weight-transport=nccl or disk, "
                f"got {args.update_weight_transport!r}."
            )
        if args.colocate:
            raise ValueError(
                "--update-weight-mode=delta is not supported with --colocate. Colocate transfers "
                "weights via CUDA IPC (only a handle crosses processes), so the delta bookkeeping "
                "(snapshot + diff + sparse encode) is pure overhead."
            )


def _validate_debug_train_data_args(args) -> None:
    if args.dump_details is not None:
        args.save_debug_rollout_data = f"{args.dump_details}/rollout_data/{{rollout_id}}.pt"
        args.save_debug_train_data = f"{args.dump_details}/train_data/{{rollout_id}}_{{rank}}.pt"

    if not args.save_debug_train_data_include_sdpo_token_weights:
        return
    if args.save_debug_train_data is None:
        raise ValueError(
            "--save-debug-train-data is required when "
            "--save-debug-train-data-include-sdpo-token-weights is enabled."
        )
    if "{rank}" not in args.save_debug_train_data:
        raise ValueError(
            "--save-debug-train-data must include {rank} when "
            "--save-debug-train-data-include-sdpo-token-weights is enabled."
        )


def slime_validate_args(args):
    args.eval_datasets = _resolve_eval_datasets(args)

    if args.use_slime_router:
        logger.warning(
            "--use-slime-router is deprecated and ignored. slime now always uses sglang_router "
            "built from https://github.com/zhuzilin/sgl-router."
        )
        args.use_slime_router = False

    if args.kl_coef != 0 or args.use_kl_loss or getattr(args, "grpo_token_weights", False):
        if not args.ref_load:
            raise ValueError("--ref-load is required when reference-model scoring is enabled.")
        if not os.path.exists(args.ref_load):
            raise FileNotFoundError(f"ref_load {args.ref_load} does not exist, please check the path.")

        if not os.path.exists(os.path.join(args.ref_load, "latest_checkpointed_iteration.txt")):
            logger.info(
                f"ref_load {args.ref_load} does not have latest_checkpointed_iteration.txt, "
                "please make sure it is a valid megatron checkpoint directory."
            )

    # Validate on-policy distillation (OPD) arguments
    if args.use_opd:
        if args.opd_type is None:
            raise ValueError("--opd-type must be specified when --use-opd is enabled. Choose 'sglang' or 'megatron'.")

        if args.opd_type == "megatron":
            if args.opd_teacher_load is None:
                raise ValueError(
                    "--opd-teacher-load is required when --opd-type=megatron. "
                    "Please provide the path to the teacher model checkpoint."
                )
            if not os.path.exists(args.opd_teacher_load):
                raise FileNotFoundError(
                    f"opd_teacher_load {args.opd_teacher_load} does not exist, please check the path."
                )
            if not os.path.exists(os.path.join(args.opd_teacher_load, "latest_checkpointed_iteration.txt")):
                logger.info(
                    f"opd_teacher_load {args.opd_teacher_load} does not have latest_checkpointed_iteration.txt, "
                    "please make sure it is a valid megatron checkpoint directory."
                )

        elif args.opd_type == "sglang":
            if args.opd_teacher_load is not None:
                raise ValueError(
                    "--opd-teacher-load should not be set when --opd-type=sglang. "
                    "In sglang mode, teacher log-probs are obtained from external server during rollout."
                )
    else:
        # If OPD is not enabled, opd_teacher_load should not be set
        if args.opd_teacher_load is not None:
            raise ValueError("--opd-teacher-load is set but --use-opd is not enabled. Please add --use-opd flag.")

    if args.megatron_to_hf_mode == "bridge":
        if (
            args.load is not None
            and os.path.exists(args.load)
            and _is_megatron_checkpoint_path(args.load)
        ):
            # If is a Megatron checkpoint, won't use bridge to load hf weight.
            pass
        else:
            if args.load is None:
                args.load = args.ref_load or args.hf_checkpoint
            # If is a HF checkpoint, set start_rollout_id to 0 here.
            args.start_rollout_id = 0
    else:
        if (
            args.load is None
            or not os.path.exists(args.load)
            or not _is_megatron_checkpoint_path(args.load)
        ):
            args.no_load_optim = True
            args.no_load_rng = True
            args.finetune = True
            args.load = args.ref_load
            if args.ref_ckpt_step is not None:
                args.ckpt_step = args.ref_ckpt_step
            args.start_rollout_id = 0

    if args.eval_interval is not None:
        assert args.eval_datasets, "Evaluation datasets must be configured when eval_interval is set."

    if args.save_interval is not None:
        assert args.save is not None, "'--save' is required when save_interval is set."

    if args.checkpoint_retention_policy == "latest_and_best_eval":
        if args.save is None:
            raise ValueError("--save is required when checkpoint retention is enabled.")
        if args.eval_interval is None:
            raise ValueError("--eval-interval is required when checkpoint retention is enabled.")
        if args.best_checkpoint_limit < 0:
            raise ValueError("--best-checkpoint-limit must be non-negative.")
        if args.best_checkpoint_limit > 0 and not args.best_checkpoint_metric:
            raise ValueError(
                "--best-checkpoint-metric is required when --best-checkpoint-limit > 0 "
                "and checkpoint retention is enabled."
            )

    assert not (args.kl_coef != 0 and args.kl_loss_coef != 0), "Only one of kl_coef and kl_loss_coef can be set"

    if args.advantage_estimator in ["reinforce_plus_plus", "reinforce_plus_plus_baseline"]:
        assert args.normalize_advantages, (
            "The 'reinforce_plus_plus' and 'reinforce_plus_plus_baseline' advantage estimators "
            "require advantage normalization. Please add `--normalize-advantages` to your command."
        )

    if args.use_rollout_logprobs:
        assert not args.use_tis, "use_rollout_logprobs and use_tis cannot be set at the same time."

    if args.get_mismatch_metrics:
        assert (
            args.custom_tis_function_path is not None
        ), "custom_tis_function_path must be set when get_mismatch_metrics is set"

        if args.use_rollout_logprobs:
            logger.info(
                "get_mismatch_metrics is set; For metrics calculation, the log probs will still be recomputed by training engine. One more forward pass will be applied."
            )

    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None, "max_tokens_per_gpu must be set when use_dynamic_batch_size is set"
        if args.log_probs_max_tokens_per_gpu is None:
            args.log_probs_max_tokens_per_gpu = args.max_tokens_per_gpu
        if (
            args.sdpo_teacher_representation_max_tokens_per_gpu is not None
            and args.sdpo_teacher_representation_max_tokens_per_gpu <= 0
        ):
            raise ValueError("--sdpo-teacher-representation-max-tokens-per-gpu must be positive when set.")
    if args.sdpo_teacher_representation_forward_chunk_rows <= 0:
        raise ValueError("--sdpo-teacher-representation-forward-chunk-rows must be positive.")

    if args.eps_clip_high is None:
        args.eps_clip_high = args.eps_clip

    if args.eval_reward_key is None:
        args.eval_reward_key = args.reward_key

    _validate_debug_train_data_args(args)

    if args.load_debug_rollout_data is not None:
        logger.info(
            f"load_debug_rollout_data {args.load_debug_rollout_data} is set, "
            "will not instantiate sglang servers and will only run the training process."
        )
        args.debug_train_only = True

    args.rollout_external = args.rollout_external_engine_addrs is not None

    if args.rollout_external and not args.debug_train_only:
        apply_external_engine_info_to_args(args, logger=logger)

    args.use_critic = args.advantage_estimator == "ppo"
    # Critic always uses the same GPU count as actor.
    args.critic_num_gpus_per_node = args.actor_num_gpus_per_node
    args.critic_num_nodes = args.actor_num_nodes

    if args.offload:
        args.offload_train = True
        args.offload_rollout = True
    del args.offload

    if args.debug_rollout_only:
        if args.colocate and (not args.rollout_num_gpus):
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        else:
            args.actor_num_gpus_per_node = min(8, args.rollout_num_gpus)
            args.actor_num_nodes = args.rollout_num_gpus // args.actor_num_gpus_per_node
        args.colocate = False
        args.offload_train = args.offload_rollout = False
        if args.train_memory_margin_bytes > 0:
            logger.warning("Force train_memory_margin_bytes=0 since debug_rollout_only does not support it")
            args.train_memory_margin_bytes = 0

    assert not (args.debug_rollout_only and args.debug_train_only), (
        "debug_rollout_only and debug_train_only cannot be set at the same time, " "please set only one of them."
    )

    # always true on offload for colocate at the moment.
    if args.colocate:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
        if args.rollout_num_gpus != args.actor_num_gpus_per_node * args.actor_num_nodes:
            logger.info(
                f"rollout_num_gpus {args.rollout_num_gpus} != actor_num_gpus_per_node {args.actor_num_gpus_per_node} "
                f"* actor_num_nodes {args.actor_num_nodes}, overriding rollout_num_gpus to match actor_num_gpus_per_node * actor_num_nodes."
            )
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes

    if args.offline_train_eval_colocate:
        if not args.colocate:
            raise ValueError("--offline-train-eval-colocate requires --colocate.")
        if args.offload_train is False or args.offload_rollout is False:
            raise ValueError("--offline-train-eval-colocate requires train and rollout offload support.")
    if args.strict_on_policy_audit_dir:
        if not args.colocate:
            raise ValueError("--strict-on-policy-audit-dir requires --colocate.")
        if args.offline_train_eval_colocate:
            raise ValueError("strict current-actor rollout auditing is not used by offline train/eval arms.")
        if args.offload_train is False or args.offload_rollout is False:
            raise ValueError("strict current-actor rollout auditing requires train and rollout offload support.")
        if args.rollout_num_gpus_per_engine != 1:
            raise ValueError("strict current-actor rollout auditing requires one SGLang engine per GPU.")
        if args.rollout_num_gpus != args.actor_num_nodes * args.actor_num_gpus_per_node:
            raise ValueError("strict current-actor rollout auditing requires identical rollout/train GPU sets.")

    if args.offload_train is None:
        args.offload_train = False
    if args.offload_rollout is None:
        args.offload_rollout = False

    if args.offline_async_eval:
        if args.offline_train_eval_colocate or args.colocate:
            raise ValueError("--offline-async-eval requires non-colocated training and evaluation.")
        if args.offload_train or args.offload_rollout:
            raise ValueError("--offline-async-eval requires resident train and rollout models without offload.")
        if args.debug_train_only or args.debug_rollout_only:
            raise ValueError("--offline-async-eval requires both live training and evaluation.")

    if args.use_critic:
        args.offload_train = True

    if args.offload_train:
        args.disable_grad_buffers_cpu_backup = True
        args.disable_param_buffers_cpu_backup = True

    if args.eval_function_path is None:
        args.eval_function_path = args.rollout_function_path

    if args.num_steps_per_rollout is not None:
        global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
        if args.global_batch_size is not None:
            assert args.global_batch_size == global_batch_size, (
                f"global_batch_size {args.global_batch_size} is not equal to "
                f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
                f"// num_steps_per_rollout {args.num_steps_per_rollout}"
            )
        args.global_batch_size = global_batch_size

    if args.n_samples_per_prompt == 1:
        args.grpo_std_normalization = False
        logger.info("n_samples_per_prompt is set to 1, grpo_std_normalization will be set to False.")

    if args.over_sampling_batch_size is None:
        args.over_sampling_batch_size = args.rollout_batch_size

    assert args.over_sampling_batch_size >= args.rollout_batch_size, (
        f"over_sampling_batch_size {args.over_sampling_batch_size} should be greater than or equal to "
        f"rollout_batch_size {args.rollout_batch_size}"
    )

    if args.num_epoch is not None:
        if args.num_rollout is not None:
            logger.info("Both num_epoch and num_rollout are set, num_epoch will be ignored.")
        else:
            assert args.rollout_global_dataset, (
                "num_epoch is set, but rollout_global_dataset is not set, "
                "please remove --disable-rollout-global-dataset to use num_epoch"
            )
    else:
        # if num_epoch is not set, we should set num_rollout
        assert args.num_rollout is not None, (
            "num_epoch is not set, but num_rollout is not set, " "please set --num-rollout or --num-epoch"
        )

    if args.enable_mtp_training:
        assert args.mtp_num_layers, "mtp_num_layers must be set when enable_mtp_training is set"

    if args.use_rollout_routing_replay:
        args.use_routing_replay = True

    if args.custom_config_path:
        with open(args.custom_config_path) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(args, k):
                logger.info(f"Warning: Argument {k} is already set to {getattr(args, k)}, will override with {v}.")
            setattr(args, k, v)

    _validate_sdpo_args(args)
    _validate_grpo_token_weight_args(args)
    _validate_active_rollout_normalization_args(args)

    if args.eval_max_context_len is None:
        logger.info(
            f"args.eval_max_context_len is not set. Use args.rollout_max_context_len {args.rollout_max_context_len} as default value."
        )
        args.eval_max_context_len = args.rollout_max_context_len

    if args.rollout_max_context_len is not None:
        if args.rollout_max_prompt_len is None:
            args.rollout_max_prompt_len = args.rollout_max_context_len - 1
            logger.info(
                f"args.rollout_max_prompt_len is not set. Use args.rollout_max_context_len - 1 ({args.rollout_max_context_len} - 1) as default value so that there is at least one generated token to compute loss."
            )
        assert (
            args.rollout_max_prompt_len <= args.rollout_max_context_len - 1
        ), f"args.rollout_max_prompt_len ({args.rollout_max_prompt_len}) must be smaller than args.rollout_max_context_len ({args.rollout_max_context_len}) so that there is at least one generated token to compute loss."

    if args.qkv_format == "bshd":
        assert args.train_backend == "megatron", "bshd format is only supported for megatron backend."
        assert (
            args.use_dynamic_batch_size is False
        ), "Dynamic batch size is not supported for bshd format. Please specify --micro-batch-size instead."

    if args.only_train_params_name_list and args.freeze_params_name_list:
        raise ValueError("You can only specify ONE of: --only-train-params-name-list, or --freeze-params-name-list.")

    _validate_update_weight_args(args)
