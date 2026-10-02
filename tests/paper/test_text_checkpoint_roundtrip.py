from __future__ import annotations

import json
from pathlib import Path

import pytest
import safetensors.torch
import torch

from scripts.prepare.verify_text_checkpoint_roundtrip import COMMON_HF, EXPECTED, compare_hf, inspect_hf


def hf_fixture(root: Path, weights: dict[str, torch.Tensor], *, model_size: str = "4b") -> None:
    root.mkdir(parents=True)
    config = {**COMMON_HF, **EXPECTED[model_size], "torch_dtype": "bfloat16"}
    (root / "config.json").write_text(json.dumps(config))
    shard = "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(weights, root / shard)
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: shard for name in weights}}))


def original_weights() -> dict[str, torch.Tensor]:
    return {"model.embed_tokens.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
            "model.layers.0.weight": torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)}


def test_exact_tied_embedding_roundtrip(tmp_path: Path) -> None:
    before, after = tmp_path / "before", tmp_path / "after"
    original = original_weights()
    hf_fixture(before, original)
    hf_fixture(after, {**original, "lm_head.weight": original["model.embed_tokens.weight"].clone()})
    result = compare_hf(before, after, "4b")
    assert result["matched_tensor_count"] == 2
    assert result["tied_lm_head_alias"] is True


@pytest.mark.parametrize("fault", ("changed", "missing", "extra", "dtype", "wrong_alias"))
def test_roundtrip_rejects_tensor_drift(tmp_path: Path, fault: str) -> None:
    before, after = tmp_path / "before", tmp_path / "after"
    original = original_weights()
    restored = {name: value.clone() for name, value in original.items()}
    if fault == "changed":
        restored["model.layers.0.weight"][0, 0] = 99
    elif fault == "missing":
        del restored["model.layers.0.weight"]
    elif fault == "extra":
        restored["unexpected.weight"] = torch.ones(1, dtype=torch.bfloat16)
    elif fault == "dtype":
        restored["model.layers.0.weight"] = restored["model.layers.0.weight"].float()
    else:
        restored["lm_head.weight"] = torch.zeros_like(original["model.embed_tokens.weight"])
    hf_fixture(before, original)
    hf_fixture(after, restored)
    with pytest.raises(ValueError):
        compare_hf(before, after, "4b")


def test_untied_qwen3_8b_requires_output_head(tmp_path: Path) -> None:
    hf_fixture(tmp_path / "missing_head", original_weights(), model_size="8b")
    with pytest.raises(ValueError, match="untied HF output head"):
        inspect_hf(tmp_path / "missing_head", "8b")
