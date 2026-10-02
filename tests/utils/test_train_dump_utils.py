from __future__ import annotations

import pytest
import torch

from slime.utils import train_dump_utils

NUM_GPUS = 0


@pytest.mark.unit
def test_debug_safe_rollout_data_summarizes_sdpo_heavy_tensors_by_default():
    teacher_representations = torch.zeros(2, 4, dtype=torch.bfloat16)
    token_weights = torch.tensor([0.5, 1.5], dtype=torch.float32)

    safe = train_dump_utils._debug_safe_rollout_data(
        {
            "sdpo_teacher_representations": [teacher_representations],
            "sdpo_token_weights": [token_weights],
        }
    )

    assert safe["sdpo_teacher_representations"] == [{"shape": (2, 4), "dtype": "torch.bfloat16", "device": "cpu"}]
    assert safe["sdpo_token_weights"] == [{"shape": (2,), "dtype": "torch.float32", "device": "cpu"}]


@pytest.mark.unit
def test_debug_safe_rollout_data_can_preserve_raw_sdpo_token_weights_only():
    teacher_representations = torch.zeros(2, 4, dtype=torch.bfloat16)
    token_weights = torch.tensor([0.5, 1.5], dtype=torch.float32)
    python_token_weights = [0.25, 1.75]

    safe = train_dump_utils._debug_safe_rollout_data(
        {
            "sdpo_teacher_representations": [teacher_representations],
            "sdpo_token_weights": [token_weights, python_token_weights],
        },
        include_sdpo_token_weights=True,
    )

    assert safe["sdpo_teacher_representations"] == [{"shape": (2, 4), "dtype": "torch.bfloat16", "device": "cpu"}]
    assert safe["sdpo_token_weights"][0] is token_weights
    assert safe["sdpo_token_weights"][1] is python_token_weights


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
