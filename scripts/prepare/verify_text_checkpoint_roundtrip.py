#!/usr/bin/env python3
"""Inspect Qwen3 text checkpoints and compare a Megatron-to-HF round trip."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import safetensors
import torch


EXPECTED = {
    "4b": {"hidden_size": 2560, "intermediate_size": 9728, "tie_word_embeddings": True},
    "8b": {"hidden_size": 4096, "intermediate_size": 12288, "tie_word_embeddings": False},
}
COMMON_HF = {
    "model_type": "qwen3", "num_hidden_layers": 36, "num_attention_heads": 32,
    "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936,
    "rope_theta": 1000000, "rms_norm_eps": 1e-6, "attention_bias": False,
}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def inspect_hf(root: Path, model_size: str) -> tuple[dict, dict[str, Path]]:
    config = read_json(root / "config.json")
    expected = {**COMMON_HF, **EXPECTED[model_size]}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Qwen3-{model_size} config {key} differs: {root}")
    if config.get("torch_dtype", config.get("dtype")) != "bfloat16":
        raise ValueError(f"Qwen3-{model_size} input must use bfloat16 weights: {root}")
    index = read_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"HF weight map is missing: {root}")
    names: dict[str, Path] = {}
    listed_files = set(weight_map.values())
    actual_files = {path.name for path in root.glob("*.safetensors")}
    if not listed_files or listed_files != actual_files:
        raise ValueError(f"HF index and shard files differ: {root}")
    for filename in sorted(listed_files):
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"invalid HF shard filename: {filename!r}")
        path = root / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"HF shard is missing or empty: {path}")
        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in names or weight_map.get(name) != filename:
                    raise ValueError(f"duplicate or unindexed tensor {name}: {path}")
                names[name] = path
    if set(names) != set(weight_map):
        raise ValueError(f"HF index names and shard tensor names differ: {root}")
    if "model.embed_tokens.weight" not in names:
        raise ValueError(f"HF input embedding is missing: {root}")
    if config["tie_word_embeddings"] is False and "lm_head.weight" not in names:
        raise ValueError(f"untied HF output head is missing: {root}")
    return config, names


def inspect_checkpoint(root: Path, hf_config: dict, expected_pipeline_parallel_size: int = 8) -> dict:
    tracker = root / "latest_checkpointed_iteration.txt"
    release = root / "release"
    if not tracker.is_file() or tracker.read_text(encoding="utf-8").strip() != "release":
        raise ValueError(f"Megatron release tracker is missing: {root}")
    for filename in (".metadata", "common.pt", "metadata.json"):
        path = release / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Megatron release metadata is missing: {path}")
    shards = list(release.glob("*.distcp"))
    if len(shards) < expected_pipeline_parallel_size or any(path.stat().st_size == 0 for path in shards):
        raise ValueError(f"distributed checkpoint shards are incomplete: {release}")
    metadata = read_json(release / "metadata.json")
    if metadata.get("sharded_backend") != "torch_dist" or metadata.get("common_backend") != "torch":
        raise ValueError(f"unexpected distributed checkpoint backend: {release}")
    common = torch.load(release / "common.pt", map_location="cpu", weights_only=False)
    args = common.get("args")
    if args is None or common.get("iteration") != 1:
        raise ValueError(f"Megatron release common metadata is invalid: {release}")
    expected = {
        "num_layers": hf_config["num_hidden_layers"],
        "hidden_size": hf_config["hidden_size"],
        "ffn_hidden_size": hf_config["intermediate_size"],
        "num_attention_heads": hf_config["num_attention_heads"],
        "num_query_groups": hf_config["num_key_value_heads"],
        "kv_channels": hf_config["head_dim"],
        "vocab_size": hf_config["vocab_size"],
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": expected_pipeline_parallel_size,
        "bf16": True,
        "qk_layernorm": True,
    }
    for key, value in expected.items():
        if getattr(args, key, None) != value:
            raise ValueError(f"Megatron {key} differs from Qwen3 input: {release}")
    return {"status": "verified", "checkpoint": str(root), "iteration": "release",
            "pipeline_parallel_size": expected_pipeline_parallel_size,
            "distributed_shard_count": len(shards)}


def tensor(path: Path, name: str) -> torch.Tensor:
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def compare_hf(original: Path, roundtrip: Path, model_size: str) -> dict:
    source_config, source = inspect_hf(original, model_size)
    restored_config, restored = inspect_hf(roundtrip, model_size)
    if restored_config != source_config:
        raise ValueError("round-trip HF config differs from the original")
    missing = set(source) - set(restored)
    extra = set(restored) - set(source)
    alias = "lm_head.weight"
    permitted_alias = (source_config["tie_word_embeddings"] is True
                       and alias not in source and extra == {alias})
    if missing or (extra and not permitted_alias):
        raise ValueError(f"round-trip tensor names differ: missing={sorted(missing)} extra={sorted(extra)}")
    if permitted_alias:
        head = tensor(restored[alias], alias)
        embedding = tensor(restored["model.embed_tokens.weight"], "model.embed_tokens.weight")
        if head.shape != embedding.shape or head.dtype != embedding.dtype or not torch.equal(head, embedding):
            raise ValueError("tied lm_head alias differs from the restored input embedding")
        del head, embedding
    dtypes: Counter[str] = Counter()
    elements = 0
    for name in sorted(source):
        before = tensor(source[name], name)
        after = tensor(restored[name], name)
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"round-trip shape or dtype differs: {name}")
        if not torch.equal(before, after):
            raise ValueError(f"round-trip tensor values differ: {name}")
        dtypes[str(before.dtype)] += 1
        elements += before.numel()
        del before, after
    return {"status": "verified", "model_size": model_size,
            "matched_tensor_count": len(source), "matched_element_count": elements,
            "dtype_counts": dict(dtypes), "tied_lm_head_alias": permitted_alias,
            "original_hf": str(original), "roundtrip_hf": str(roundtrip)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("inspect-hf", "inspect-checkpoint", "compare"):
        command = sub.add_parser(action)
        command.add_argument("--model-size", choices=tuple(EXPECTED), required=True)
        command.add_argument("--original-hf", type=Path, required=True)
        if action == "inspect-checkpoint":
            command.add_argument("--checkpoint-root", type=Path, required=True)
            command.add_argument("--expected-pipeline-parallel-size", type=int, default=8)
        if action == "compare":
            command.add_argument("--roundtrip-hf", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, names = inspect_hf(args.original_hf, args.model_size)
    if args.action == "inspect-hf":
        result = {"status": "verified", "model_size": args.model_size,
                  "tensor_count": len(names), "shard_count": len(set(names.values()))}
    elif args.action == "inspect-checkpoint":
        if args.expected_pipeline_parallel_size < 1:
            raise ValueError("expected pipeline parallel size must be positive")
        result = inspect_checkpoint(args.checkpoint_root, config,
                                    args.expected_pipeline_parallel_size)
    else:
        result = compare_hf(args.original_hf, args.roundtrip_hf, args.model_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite round-trip report: {args.output}")
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
