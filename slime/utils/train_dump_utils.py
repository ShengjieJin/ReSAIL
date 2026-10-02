# Modified for ReSAIL. See NOTICE and LICENSE for attribution and terms.
import logging
from pathlib import Path

import torch


logger = logging.getLogger(__name__)



def save_debug_train_data(args, *, rollout_id, rollout_data, include_sdpo_token_weights=False):
    if (path_template := args.save_debug_train_data) is not None:
        rank = torch.distributed.get_rank()
        path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                rollout_data=_debug_safe_rollout_data(
                    rollout_data,
                    include_sdpo_token_weights=include_sdpo_token_weights,
                ),
            ),
            path,
        )


def _debug_safe_rollout_data(rollout_data, *, include_sdpo_token_weights=False):
    data = dict(rollout_data)
    if "sdpo_teacher_representations" in data:
        data["sdpo_teacher_representations"] = [_tensor_summary(row) for row in data["sdpo_teacher_representations"]]
    if "sdpo_token_weights" in data and not include_sdpo_token_weights:
        data["sdpo_token_weights"] = [_tensor_summary(row) for row in data["sdpo_token_weights"]]
    return data


def _tensor_summary(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    return value
