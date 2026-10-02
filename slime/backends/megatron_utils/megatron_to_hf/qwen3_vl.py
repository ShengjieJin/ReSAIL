# Modified for ReSAIL. See NOTICE and LICENSE for attribution and terms.
import re
import torch


def _vision_qkv_layout(args, param):
    """Resolve the vision attention layout for both CLI and checkpoint conversion."""
    num_heads = getattr(args, "vision_num_attention_heads", None)
    hidden_size = getattr(args, "vision_hidden_size", None)
    if num_heads is None or hidden_size is None:
        cached = getattr(args, "_qwen3vl_vision_qkv_layout", None)
        if cached is None:
            from transformers import AutoConfig

            checkpoint = getattr(args, "hf_checkpoint", None)
            if checkpoint is None:
                raise ValueError(
                    "Qwen3-VL vision QKV conversion needs vision_num_attention_heads/vision_hidden_size "
                    "or hf_checkpoint"
                )
            vision_config = AutoConfig.from_pretrained(
                checkpoint, trust_remote_code=True, local_files_only=True
            ).vision_config
            cached = (int(vision_config.num_heads), int(vision_config.hidden_size))
            setattr(args, "_qwen3vl_vision_qkv_layout", cached)
        num_heads, hidden_size = cached

    num_heads = int(num_heads)
    hidden_size = int(hidden_size)
    if hidden_size % num_heads != 0:
        raise ValueError(f"Invalid Qwen3-VL vision layout: hidden_size={hidden_size}, num_heads={num_heads}")
    if param.shape[0] != 3 * hidden_size:
        raise ValueError(
            f"Unexpected Qwen3-VL vision QKV leading dimension {param.shape[0]}; expected {3 * hidden_size}"
        )
    return num_heads, hidden_size // num_heads


def convert_qwen3vl_to_hf(args, name, param):
    if name.startswith("module.module.language_model."):
        name = "module.module." + name[len("module.module.language_model.") :]

    # (Optional safety) if you ever see extra "module." prefixes
    while name.startswith("module.module.module."):
        name = name.replace("module.module.module.", "module.module.", 1)

    if name.startswith("module.module.vision_model."):
        vision_name = name[len("module.module.vision_model.") :]
        layer_match = re.match(r"decoder\.layers\.(\d+)\.(.+)", vision_name)
        if layer_match:
            layer_idx, rest = layer_match.groups()
            replacements = {
                "self_attention.linear_proj.weight": "attn.proj.weight",
                "self_attention.linear_proj.bias": "attn.proj.bias",
                "self_attention.linear_qkv.weight": "attn.qkv.weight",
                "self_attention.linear_qkv.bias": "attn.qkv.bias",
                "self_attention.linear_qkv.layer_norm_weight": "norm1.weight",
                "self_attention.linear_qkv.layer_norm_bias": "norm1.bias",
                "mlp.linear_fc1.weight": "mlp.linear_fc1.weight",
                "mlp.linear_fc1.bias": "mlp.linear_fc1.bias",
                "mlp.linear_fc1.layer_norm_weight": "norm2.weight",
                "mlp.linear_fc1.layer_norm_bias": "norm2.bias",
                "mlp.linear_fc2.weight": "mlp.linear_fc2.weight",
                "mlp.linear_fc2.bias": "mlp.linear_fc2.bias",
            }
            if rest not in replacements:
                raise ValueError(f"Unknown Qwen3-VL vision layer parameter: {name}")
            if rest in {"self_attention.linear_qkv.weight", "self_attention.linear_qkv.bias"}:
                num_heads, head_dim = _vision_qkv_layout(args, param)
                tail_shape = tuple(param.shape[1:])
                interleaved = param.reshape(num_heads, 3, head_dim, *tail_shape)
                q = interleaved[:, 0].reshape(-1, *tail_shape)
                k = interleaved[:, 1].reshape(-1, *tail_shape)
                v = interleaved[:, 2].reshape(-1, *tail_shape)
                param = torch.cat((q, k, v), dim=0)
            return [(f"model.visual.blocks.{layer_idx}.{replacements[rest]}", param)]

        deepstack_match = re.match(r"decoder\.deepstack_merger_list\.(\d+)\.(.+)", vision_name)
        if deepstack_match:
            merger_idx, rest = deepstack_match.groups()
            rest = rest.replace("patch_norm.", "norm.")
            return [(f"model.visual.deepstack_merger_list.{merger_idx}.{rest}", param)]

        if vision_name.startswith("merger."):
            vision_name = vision_name.replace("merger.patch_norm.", "merger.norm.", 1)
        return [("model.visual." + vision_name, param)]

    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]

    if name == "module.module.output_layer.weight":
        # Your key list has lm_head.weight at top-level
        return [("lm_head.weight", param)]

    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        # Everything goes under model.language_model.layers.{i}.*
        base = f"model.language_model.layers.{layer_idx}"

        if rest == "self_attention.linear_proj.weight":
            return [(f"{base}.self_attn.o_proj.weight", param)]

        elif rest == "self_attention.linear_qkv.weight":
            # Keep your original split logic -> q_proj/k_proj/v_proj
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"{base}.self_attn.q_proj.weight", q_param),
                (f"{base}.self_attn.k_proj.weight", k_param),
                (f"{base}.self_attn.v_proj.weight", v_param),
            ]

        elif rest == "self_attention.linear_qkv.bias":
            # Keep your original split logic -> q_proj/k_proj/v_proj bias
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"{base}.self_attn.q_proj.bias", q_bias),
                (f"{base}.self_attn.k_proj.bias", k_bias),
                (f"{base}.self_attn.v_proj.bias", v_bias),
            ]

        elif rest == "mlp.linear_fc1.weight":
            # Keep your original split logic -> gate_proj/up_proj
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"{base}.mlp.gate_proj.weight", gate_weight),
                (f"{base}.mlp.up_proj.weight", up_weight),
            ]

        elif rest == "mlp.linear_fc2.weight":
            return [(f"{base}.mlp.down_proj.weight", param)]

        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"{base}.input_layernorm.weight", param)]

        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"{base}.post_attention_layernorm.weight", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"{base}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"{base}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
