# Modified for ReSAIL. See NOTICE and LICENSE for attribution and terms.
import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0


def load_arguments_module(monkeypatch):
    megatron_mod = types.ModuleType("megatron")
    training_mod = types.ModuleType("megatron.training")
    arguments_mod = types.ModuleType("megatron.training.arguments")
    tokenizer_pkg_mod = types.ModuleType("megatron.training.tokenizer")
    tokenizer_mod = types.ModuleType("megatron.training.tokenizer.tokenizer")
    transformers_mod = types.ModuleType("transformers")

    arguments_mod.parse_args = lambda *args, **kwargs: None
    arguments_mod.validate_args = lambda args: args
    tokenizer_mod._vocab_size_with_padding = lambda vocab_size, _args: vocab_size
    transformers_mod.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.training", training_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.arguments", arguments_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer", tokenizer_pkg_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer.tokenizer", tokenizer_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "arguments.py"
    module_name = "test_megatron_argument_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_slime_arguments_module(monkeypatch):
    class StubRouterArgs:
        @staticmethod
        def add_cli_args(parser, **_kwargs):
            return parser

    router_pkg_mod = types.ModuleType("sglang_router")
    router_launch_mod = types.ModuleType("sglang_router.launch_router")
    sglang_arguments_mod = types.ModuleType("slime.backends.sglang_utils.arguments")
    sglang_external_mod = types.ModuleType("slime.backends.sglang_utils.external")
    logging_utils_mod = types.ModuleType("slime.utils.logging_utils")

    router_launch_mod.RouterArgs = StubRouterArgs
    sglang_arguments_mod.sglang_parse_args = lambda *args, **kwargs: None
    sglang_arguments_mod.validate_args = lambda args: args
    sglang_external_mod.apply_external_engine_info_to_args = lambda *args, **kwargs: None
    logging_utils_mod.configure_logger = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg_mod)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", router_launch_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.arguments", sglang_arguments_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.external", sglang_external_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.logging_utils", logging_utils_mod)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "utils" / "arguments.py"
    module_name = "test_slime_argument_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_qwen3_6_args(**overrides):
    values = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_layers=40,
        ffn_hidden_size=512,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=512,
        moe_layer_freq=[1] * 40,
        untie_embeddings_and_output_weights=True,
        norm_epsilon=1e-6,
        layernorm_epsilon=1e-6,
        rotary_base=10000000,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_qwen3_6_hf_config():
    text_config = types.SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=40,
        intermediate_size=5632,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts=256,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000000},
    )
    return types.SimpleNamespace(text_config=text_config)


def make_allgather_cp_args(**overrides):
    values = dict(
        allgather_cp=True,
        context_parallel_size=2,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
def test_hf_validate_all_moe_skips_dense_intermediate_size(monkeypatch):
    module = load_arguments_module(monkeypatch)

    module._hf_validate_args(make_qwen3_6_args(), make_qwen3_6_hf_config())


@pytest.mark.unit
def test_hf_validate_checks_moe_intermediate_size(monkeypatch):
    module = load_arguments_module(monkeypatch)

    with pytest.raises(AssertionError, match="moe_intermediate_size"):
        module._hf_validate_args(make_qwen3_6_args(moe_ffn_hidden_size=256), make_qwen3_6_hf_config())


@pytest.mark.unit
def test_hf_validate_checks_dense_intermediate_size_when_moe_has_dense_layers(monkeypatch):
    module = load_arguments_module(monkeypatch)

    args = make_qwen3_6_args(moe_layer_freq=[0] + [1] * 39)

    with pytest.raises(AssertionError, match="intermediate_size"):
        module._hf_validate_args(args, make_qwen3_6_hf_config())


@pytest.mark.unit
def test_allgather_cp_rejects_non_dsa_cp_models(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_allgather_cp_args()
    hf_config = types.SimpleNamespace(architectures=["Qwen3ForCausalLM"], model_type="qwen3")

    with pytest.raises(ValueError, match="only supported for DSA attention models"):
        module._validate_allgather_cp_supported(args, hf_config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "hf_config",
    [
        types.SimpleNamespace(architectures=["DeepseekV32ForCausalLM"], model_type="deepseek_v3"),
        types.SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"], model_type="glm"),
    ],
)
def test_allgather_cp_allows_dsa_architectures(monkeypatch, hf_config):
    module = load_arguments_module(monkeypatch)

    module._validate_allgather_cp_supported(make_allgather_cp_args(), hf_config)


@pytest.mark.unit
def test_allgather_cp_ignores_cp_size_one(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_allgather_cp_args(context_parallel_size=1)

    module._validate_allgather_cp_supported(args)


@pytest.mark.unit
def test_update_weight_disk_dir_required_for_disk_transport(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_transport="disk",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
    )

    with pytest.raises(ValueError, match="update-weight-disk-dir"):
        module._resolve_update_weight_disk_dir(args)


@pytest.mark.unit
def test_update_weight_disk_dir_normalizes_delta_alias(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_transport="disk",
        update_weight_disk_dir=None,
        update_weight_delta_dir="/shared/delta",
    )

    with pytest.warns(UserWarning, match="will be removed in a future release"):
        module._resolve_update_weight_disk_dir(args)

    assert args.update_weight_disk_dir == "/shared/delta"
    assert args.update_weight_delta_dir == "/shared/delta"


@pytest.mark.unit
def test_update_weight_disk_dir_backfills_legacy_delta_field(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/updates",
        update_weight_delta_dir=None,
    )

    module._resolve_update_weight_disk_dir(args)

    assert args.update_weight_disk_dir == "/shared/updates"
    assert args.update_weight_delta_dir == "/shared/updates"


@pytest.mark.unit
def test_update_weight_disk_dir_rejects_conflicting_alias(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/full",
        update_weight_delta_dir="/shared/delta",
    )

    with pytest.raises(ValueError, match="deprecated alias"):
        module._resolve_update_weight_disk_dir(args)


@pytest.mark.unit
def test_update_weight_delta_rejects_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_mode="delta",
        update_weight_transport="nccl",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=True,
    )

    with pytest.raises(ValueError, match="not supported with --colocate"):
        module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weight_delta_rejects_unknown_transport(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_mode="delta",
        update_weight_transport="tensor",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
    )

    with pytest.raises(ValueError, match="supports only --update-weight-transport=nccl or disk"):
        module._validate_update_weight_args(args)


def make_sdpo_args(**overrides):
    values = dict(
        loss_type="sdpo_loss",
        custom_convert_samples_to_train_data_path="slime_plugins.agent_tasks.common.algorithms.sdpo.convert_samples_to_train_data",
        agent_task_sdpo_enabled=False,
        sdpo_teacher_update_rate=0.0,
        sdpo_alpha=1.0,
        sdpo_ema_teacher_keep_last=0,
        sdpo_max_reprompt_tokens=1024,
        sdpo_success_reward_threshold=1.0,
        sdpo_max_demo_steps=None,
        sdpo_guidance_summary_max_tokens=256,
        sdpo_guidance_summary_max_prompt_tokens=1024,
        sdpo_guidance_summary_timeout=120,
        agent_task_sdpo_guidance_summary_concurrency=0,
        sdpo_guidance_debug_max_records=0,
        sdpo_teacher_context_mode="original",
        sdpo_no_success_context_mode="filter",
        sdpo_solution_context_format="trajectory_demo",
        sdpo_guidance_generation_mode="disabled",
        sdpo_guidance_summary_source="success_priority",
        sdpo_multi_turn_weighting="traj_equal",
        sdpo_distillation_mode="representation",
        sdpo_full_logit_distillation=True,
        sdpo_distillation_topk=20,
        sdpo_representation_layers="last",
        sdpo_representation_coef=1.0,
        sdpo_representation_reduction="sum",
        sdpo_representation_success_aggregation="sample",
        sdpo_token_weights=False,
        sdpo_token_weights_power=1.0,
        sdpo_token_weights_max=0.0,
        sdpo_filter_all_success_groups=False,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        overlap_moe_expert_parallel_comm=False,
        sdpo_teacher_regularization="actor",
        ref_load=None,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_pr_args(**overrides):
    values = dict(
        agent_task_sdpo_enabled=True,
        custom_convert_samples_to_train_data_path=(
            "slime_plugins.agent_tasks.common.algorithms.resail."
            "convert_samples_to_train_data"
        ),
        sdpo_distillation_mode="topk",
        sdpo_distillation_add_tail=True,
        sdpo_teacher_regularization="ref",
        ref_load="/checkpoint/ref",
        pr_weight=0.3,
        sgs_selection_fraction=0.05,
        sgs_selection_mode="sensitivity",
        sgs_skip_full_scoring=False,
        sgs_score_scope="response",
    )
    values.update(overrides)
    return make_sdpo_args(**values)


@pytest.mark.unit
def test_pr_validation_accepts_full_response_sgs_support(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    monkeypatch.setattr(module.os.path, "exists", lambda _path: True)

    module._validate_sdpo_args(make_pr_args())


@pytest.mark.unit
def test_retention_validation_accepts_unscored_random_all_step_support(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    monkeypatch.setattr(module.os.path, "exists", lambda _path: True)

    module._validate_sdpo_args(
        make_pr_args(
            sgs_selection_mode="random",
            sgs_skip_full_scoring=True,
            sgs_score_scope="response",
            pr_support="all",
            pr_view="privileged",
        )
    )


@pytest.mark.unit
def test_retention_validation_rejects_unscored_random_non_global_support(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    monkeypatch.setattr(module.os.path, "exists", lambda _path: True)

    with pytest.raises(ValueError, match="unscored random all-step"):
        module._validate_sdpo_args(
            make_pr_args(
                sgs_selection_mode="random",
                sgs_skip_full_scoring=True,
                sgs_score_scope="response",
                sgs_selection_scope="trajectory",
                pr_support="all",
            )
        )


@pytest.mark.unit
def test_pr_validation_accepts_explicit_full_selection(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    monkeypatch.setattr(module.os.path, "exists", lambda _path: True)

    module._validate_sdpo_args(
        make_pr_args(
            sgs_selection_fraction=1.0,
            sgs_full_selection_retention=True,
            pr_support="all",
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"pr_weight": 0.0},
        {"pr_support": "selected"},
        {"sgs_selection_fraction": 0.5},
    ],
)
def test_explicit_full_selection_retention_rejects_incoherent_contract(monkeypatch, overrides):
    module = load_slime_arguments_module(monkeypatch)
    values = {
        "sgs_selection_fraction": 1.0,
        "sgs_full_selection_retention": True,
        "pr_support": "all",
        **overrides,
    }
    args = make_pr_args(**values)
    with pytest.raises(ValueError, match="full-selection retention requires"):
        module._validate_sdpo_args(args)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sgs_selection_fraction": None}, "fraction in \\(0, 1\\)"),
        ({"sgs_selection_fraction": 1.0}, "fraction in \\(0, 1\\)"),
        ({"sgs_selection_mode": "random"}, "sensitivity-ranked"),
        ({"sgs_skip_full_scoring": True}, "full-response"),
        ({"sgs_score_scope": "action"}, "full-response"),
    ],
)
def test_pr_validation_rejects_non_sgs_support(monkeypatch, overrides, message):
    module = load_slime_arguments_module(monkeypatch)

    with pytest.raises(ValueError, match=message):
        module._validate_sdpo_args(make_pr_args(**overrides))








@pytest.mark.unit
def test_sdpo_representation_validation_disables_full_logit_distillation(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args()

    module._validate_sdpo_args(args)

    assert args.sdpo_full_logit_distillation is False


@pytest.mark.unit
def test_sdpo_representation_parser_defaults_to_sum_coef_one(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    parser = argparse.ArgumentParser()
    parser = module.get_slime_extra_args_provider()(parser)

    args = parser.parse_args(["--rollout-batch-size", "1"])

    assert args.sdpo_representation_reduction == "sum"
    assert args.sdpo_representation_coef == 1.0
    assert args.sdpo_token_weights is False
    assert args.sdpo_token_weights_power == 1.0
    assert args.sdpo_token_weight_source == "teacher_contrast"
    assert args.sdpo_token_weights_max == 0.0
    assert args.sdpo_filter_all_success_groups is False
    assert args.sdpo_teacher_representation_max_tokens_per_gpu is None
    assert args.sdpo_teacher_representation_forward_chunk_rows == 512
    assert args.grpo_token_weights is False
    assert args.grpo_token_weights_power == 1.0
    assert args.grpo_token_weights_max == 8.0


def make_grpo_token_weight_args(**overrides):
    values = dict(
        grpo_token_weights=True,
        grpo_token_weights_power=1.0,
        grpo_token_weights_max=8.0,
        loss_type="policy_loss",
        advantage_estimator="grpo",
        ref_load="/tmp/ref",
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
def test_grpo_token_weights_validation_accepts_locked_configuration(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)

    module._validate_grpo_token_weight_args(make_grpo_token_weight_args())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"loss_type": "sdpo_loss"}, "policy_loss"),
        ({"advantage_estimator": "gspo"}, "advantage-estimator=grpo"),
        ({"ref_load": None}, "ref-load"),
        ({"grpo_token_weights_power": -1.0}, "power"),
        ({"grpo_token_weights_power": float("inf")}, "power"),
        ({"grpo_token_weights_max": 1.0}, "max"),
        ({"grpo_token_weights_max": float("nan")}, "max"),
        ({"tensor_model_parallel_size": 2}, "tensor-model-parallel-size 1"),
        ({"context_parallel_size": 2}, "context-parallel-size 1"),
        ({"pipeline_model_parallel_size": 2}, "pipeline-model-parallel-size 1"),
    ],
)
def test_grpo_token_weights_validation_rejects_invalid_configuration(monkeypatch, overrides, message):
    module = load_slime_arguments_module(monkeypatch)

    with pytest.raises(ValueError, match=message):
        module._validate_grpo_token_weight_args(make_grpo_token_weight_args(**overrides))








@pytest.mark.unit
def test_sdpo_token_weight_source_parser_accepts_student(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    parser = module.get_slime_extra_args_provider()(argparse.ArgumentParser())

    args = parser.parse_args(["--rollout-batch-size", "1", "--sdpo-token-weight-source", "student"])

    assert args.sdpo_token_weight_source == "student"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cli_args", "message"),
    [
        (
            ["--save-debug-train-data-include-sdpo-token-weights"],
            "--save-debug-train-data is required",
        ),
        (
            [
                "--save-debug-train-data-include-sdpo-token-weights",
                "--save-debug-train-data",
                "/tmp/train_{rollout_id}.pt",
            ],
            "must include {rank}",
        ),
    ],
)
def test_raw_sdpo_token_weight_dump_requires_ranked_train_dump_path(monkeypatch, cli_args, message):
    module = load_slime_arguments_module(monkeypatch)
    parser = module.get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(["--rollout-batch-size", "1", *cli_args])

    with pytest.raises(ValueError, match=message):
        module._validate_debug_train_data_args(args)


@pytest.mark.unit
def test_raw_sdpo_token_weight_dump_accepts_ranked_train_dump_path(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    parser = module.get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "1",
            "--save-debug-train-data-include-sdpo-token-weights",
            "--save-debug-train-data",
            "/tmp/train_{rollout_id}_{rank}.pt",
        ]
    )

    module._validate_debug_train_data_args(args)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sdpo_teacher_representation_max_tokens_per_gpu": 0}, "sdpo-teacher-representation-max-tokens-per-gpu"),
        ({"sdpo_teacher_representation_forward_chunk_rows": 0}, "sdpo-teacher-representation-forward-chunk-rows"),
    ],
)
def test_slime_validation_rejects_invalid_sdpo_teacher_representation_controls(
    monkeypatch,
    overrides,
    message,
):
    module = load_slime_arguments_module(monkeypatch)
    values = dict(
        use_slime_router=False,
        kl_coef=0.0,
        use_kl_loss=False,
        use_opd=False,
        opd_teacher_load=None,
        megatron_to_hf_mode="direct",
        load="/tmp/non_megatron",
        ref_load="/tmp/ref",
        hf_checkpoint="/tmp/ref",
        ref_ckpt_step=None,
        eval_config=None,
        eval_prompt_data=None,
        eval_interval=None,
        eval_datasets=None,
        save_interval=None,
        save=None,
        checkpoint_retention_policy=None,
        best_checkpoint_limit=0,
        best_checkpoint_metric=None,
        advantage_estimator="grpo",
        normalize_advantages=False,
        use_rollout_logprobs=False,
        use_tis=False,
        get_mismatch_metrics=False,
        custom_tis_function_path=None,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=1024,
        log_probs_max_tokens_per_gpu=None,
        sdpo_teacher_representation_max_tokens_per_gpu=None,
        sdpo_teacher_representation_forward_chunk_rows=512,
        eps_clip_high=None,
        eps_clip=0.2,
        eval_reward_key=None,
        reward_key="reward",
        debug_rollout_only=False,
    )
    values.update(overrides)
    args = types.SimpleNamespace(**values)

    with pytest.raises(ValueError, match=message):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_sdpo_representation_validation_rejects_all_layers(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(sdpo_representation_layers="all")

    with pytest.raises(ValueError, match="experimental"):
        module._validate_sdpo_args(args)


@pytest.mark.unit
@pytest.mark.parametrize("coef", [-1.0, float("inf"), float("nan")])
def test_sdpo_representation_validation_rejects_invalid_coef(monkeypatch, coef):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(sdpo_representation_coef=coef)

    with pytest.raises(ValueError, match="sdpo-representation-coef"):
        module._validate_sdpo_args(args)


@pytest.mark.unit
def test_sdpo_representation_validation_rejects_invalid_reduction(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(sdpo_representation_reduction="median")

    with pytest.raises(ValueError, match="sdpo-representation-reduction"):
        module._validate_sdpo_args(args)


@pytest.mark.unit
def test_sdpo_representation_validation_accepts_mean_success_aggregation(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(sdpo_representation_success_aggregation="mean")

    module._validate_sdpo_args(args)

    assert args.sdpo_full_logit_distillation is False


@pytest.mark.unit
def test_sdpo_validation_accepts_random_success_aggregation_for_representation(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(sdpo_representation_success_aggregation="random")

    module._validate_sdpo_args(args)

    assert args.sdpo_full_logit_distillation is False


@pytest.mark.unit
def test_sdpo_validation_accepts_random_success_aggregation_for_topk(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(
        sdpo_distillation_mode="topk",
        sdpo_representation_success_aggregation="random",
    )

    module._validate_sdpo_args(args)

    assert args.sdpo_full_logit_distillation is True










@pytest.mark.unit
def test_sdpo_validation_accepts_random_success_aggregation_for_sample_token(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(
        sdpo_distillation_mode="sample_token",
        sdpo_full_logit_distillation=False,
        sdpo_solution_context_format="guidance_plan",
        sdpo_representation_success_aggregation="random",
    )

    module._validate_sdpo_args(args)

    assert args.sdpo_full_logit_distillation is False


@pytest.mark.unit
def test_sdpo_validation_accepts_own_outcome_self_trajectory_guidance(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(
        sdpo_teacher_context_mode="own_outcome",
        sdpo_solution_context_format="guidance_plan",
        sdpo_guidance_summary_source="self_trajectory",
        sdpo_no_success_context_mode="failed_negative",
    )

    module._validate_sdpo_args(args)


@pytest.mark.unit
def test_sdpo_validation_rejects_own_outcome_cross_trajectory_guidance(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(
        sdpo_teacher_context_mode="own_outcome",
        sdpo_solution_context_format="guidance_plan",
        sdpo_guidance_summary_source="success_priority",
        sdpo_no_success_context_mode="failed_negative",
    )

    with pytest.raises(ValueError, match="self_trajectory"):
        module._validate_sdpo_args(args)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tensor_model_parallel_size": 2}, "tensor-model-parallel-size 1"),
        ({"context_parallel_size": 2}, "context-parallel-size 1"),
        ({"pipeline_model_parallel_size": 2}, "pipeline-model-parallel-size 1"),
    ],
)
def test_sdpo_token_weights_reject_parallel_sample_token(monkeypatch, overrides, message):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(
        sdpo_token_weights=True,
        sdpo_distillation_mode="sample_token",
        sdpo_full_logit_distillation=False,
        **overrides,
    )

    with pytest.raises(ValueError, match=message):
        module._validate_sdpo_args(args)


@pytest.mark.unit
@pytest.mark.parametrize("power", [float("inf"), float("nan")])
def test_sdpo_token_weights_reject_invalid_power(monkeypatch, power):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(sdpo_token_weights_power=power)

    with pytest.raises(ValueError, match="sdpo-token-weights-power"):
        module._validate_sdpo_args(args)


@pytest.mark.unit
def test_sdpo_token_weights_allow_negative_power(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(sdpo_token_weights_power=-1.0)

    module._validate_sdpo_args(args)

    assert args.sdpo_token_weights_power == -1.0


@pytest.mark.unit
@pytest.mark.parametrize("max_value", [-1.0, 0.5, 1.0, float("inf"), float("nan")])
def test_sdpo_token_weights_reject_invalid_max(monkeypatch, max_value):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(sdpo_token_weights_max=max_value)

    with pytest.raises(ValueError, match="sdpo-token-weights-max"):
        module._validate_sdpo_args(args)


@pytest.mark.unit
def test_sdpo_filter_all_success_requires_no_success_filter(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(
        sdpo_filter_all_success_groups=True,
        sdpo_no_success_context_mode="feedback",
    )

    with pytest.raises(ValueError, match="sdpo-filter-all-success-groups"):
        module._validate_sdpo_args(args)


@pytest.mark.unit
def test_sdpo_representation_validation_rejects_invalid_success_aggregation(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(sdpo_representation_success_aggregation="all")

    with pytest.raises(ValueError, match="sdpo-representation-success-aggregation"):
        module._validate_sdpo_args(args)


@pytest.mark.unit
def test_sdpo_representation_validation_rejects_mean_success_aggregation_without_trajectory_demo(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(
        sdpo_representation_success_aggregation="mean",
        sdpo_solution_context_format="guidance_plan",
    )

    with pytest.raises(ValueError, match="requires --sdpo-solution-context-format=trajectory_demo"):
        module._validate_sdpo_args(args)


@pytest.mark.unit
def test_sdpo_validation_rejects_mean_success_aggregation_for_topk(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(
        sdpo_distillation_mode="topk",
        sdpo_representation_success_aggregation="mean",
    )

    with pytest.raises(ValueError, match="requires representation distillation"):
        module._validate_sdpo_args(args)


@pytest.mark.unit
def test_sdpo_validation_rejects_mean_success_aggregation_for_sample_token(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(
        sdpo_distillation_mode="sample_token",
        sdpo_full_logit_distillation=False,
        sdpo_representation_success_aggregation="mean",
    )

    with pytest.raises(ValueError, match="requires representation distillation"):
        module._validate_sdpo_args(args)


@pytest.mark.unit
def test_sdpo_representation_validation_rejects_context_parallel(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(context_parallel_size=2)

    with pytest.raises(ValueError, match="context-parallel-size 1"):
        module._validate_sdpo_args(args)


@pytest.mark.unit
def test_sdpo_representation_validation_rejects_tensor_parallel(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(tensor_model_parallel_size=2)

    with pytest.raises(ValueError, match="tensor-model-parallel-size 1"):
        module._validate_sdpo_args(args)


@pytest.mark.unit
def test_sdpo_representation_validation_rejects_combined_1f1b(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_sdpo_args(overlap_moe_expert_parallel_comm=True)

    with pytest.raises(ValueError, match="combined-1F1B"):
        module._validate_sdpo_args(args)


@pytest.mark.unit
def test_direct_iteration_directory_is_a_megatron_checkpoint(monkeypatch, tmp_path):
    module = load_slime_arguments_module(monkeypatch)
    direct = tmp_path / "iter_0000029"
    direct.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "latest_checkpointed_iteration.txt").write_text("29\n", encoding="utf-8")
    assert module._is_megatron_checkpoint_path(str(direct)) is True
    assert module._is_megatron_checkpoint_path(str(root)) is True
    assert module._is_megatron_checkpoint_path(str(tmp_path / "missing")) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
