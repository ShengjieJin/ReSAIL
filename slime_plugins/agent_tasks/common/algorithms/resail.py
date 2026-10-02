"""Compose SGS selection, TLB weights, and PR rows for the training backend."""

from __future__ import annotations

from collections import Counter
from typing import Any

from slime.utils.types import Sample

from .pr import expand_pr_rows
from .sdpo import convert_samples_to_train_data as default_sdpo_converter
from .sgs import PRECOMPUTED_FIELDS, sgs_option, sgs_selection_fraction
from .tlb import compute_tlb_weights


def convert_samples_to_train_data(
    args: Any,
    samples: list[Sample] | list[list[Sample]],
) -> dict[str, Any]:
    flat = _flatten(samples)
    if not flat:
        raise ValueError("SGS converter received no samples")
    # The first conversion packages all generated rows for the scoring prepass.
    # A finalized selection may either carry full-score precomputed tensors or
    # use the ordinary selected-row training path.
    finalized = all(
        bool(getattr(sample, "sgs_finalized", False)) or bool(getattr(sample, "sgs_scored", False)) for sample in flat
    )
    if not finalized:
        return default_sdpo_converter(args, flat)
    selected_indices = [index for index, sample in enumerate(flat) if bool(getattr(sample, "sgs_selected", False))]
    retention_weight = float(getattr(args, "pr_weight", 0.0) or 0.0)
    retention_support = str(getattr(args, "pr_support", "selected"))
    if len(selected_indices) != len(flat) and not (retention_weight > 0.0 and retention_support == "all"):
        raise ValueError("final ReSAIL conversion received an unselected sample")
    if not selected_indices:
        raise ValueError("final ReSAIL conversion received no selected samples")
    if sgs_selection_fraction(args) == 1.0 and not bool(sgs_option(args, "full_selection_retention", False)):
        return default_sdpo_converter(args, flat)

    train_data = default_sdpo_converter(args, flat)
    selected_count = len(selected_indices)
    outer_denominator = int(getattr(args, "rollout_batch_size", 0))
    if outer_denominator <= 0:
        raise ValueError("SGS requires a positive rollout_batch_size")
    aggregation = str((getattr(args, "tlb_loss_aggregation", None) or "trajectory_balanced"))
    if aggregation == "trajectory_balanced":
        train_data["sdpo_loss_weights"] = compute_tlb_weights(
            [int((sample.metadata or {})["source_draw_id"]) for sample in flat],
            selected_indices,
            source_trajectory_count=outer_denominator,
        )
    elif aggregation != "donor":
        raise ValueError(f"unsupported TLB loss aggregation: {aggregation}")
    train_data["self_distillation_mask"] = [1.0] * len(flat)
    # This payload is one optimizer update over the original trajectory batch even though filtering
    # can remove every row from some trajectories.  The DP scheduler must pack
    # all surviving rows into one step while retaining the nominal batch size
    # for scheduler increments and the fixed outer loss denominator.
    train_data["sgs_sparse_single_step"] = True
    precomputed = [getattr(sample, "sgs_precomputed", None) for sample in flat]
    if any(payload is not None for payload in precomputed):
        if not all(isinstance(precomputed[index], dict) for index in selected_indices):
            raise ValueError("selected SGS rows mix precomputed and ordinary training data")
        for field in PRECOMPUTED_FIELDS:
            train_data[field] = [payload[field] if isinstance(payload, dict) else None for payload in precomputed]

    score_records = [getattr(sample, "sgs_score_record") for sample in flat]
    selected_records = [score_records[index] for index in selected_indices]
    selected_samples = [flat[index] for index in selected_indices]
    train_data["self_distillation/sgs/selected_steps"] = [float(selected_count)] * len(flat)
    train_data["self_distillation/sgs/selected_action_tokens"] = [
        float(sum(int(row.get("action_token_count", 0)) for row in selected_records))
    ] * len(flat)
    train_data["self_distillation/sgs/selected_response_tokens"] = [
        float(sum(sample.response_length for sample in selected_samples))
    ] * len(flat)
    batch_metrics = getattr(flat[0], "sgs_batch_metrics", None)
    if not isinstance(batch_metrics, dict):
        raise ValueError("selected SGS samples are missing batch metrics")
    for name, value in batch_metrics.items():
        train_data[f"self_distillation/sgs/{name}"] = [float(value)] * len(flat)
    if retention_weight > 0.0:
        if retention_support == "all":
            count_by_draw = Counter(int((sample.metadata or {})["source_draw_id"]) for sample in flat)
            if len(count_by_draw) != outer_denominator:
                raise ValueError("all-step privileged retention requires one non-empty group per rollout source")
            retention_weights = [
                len(flat) / (outer_denominator * count_by_draw[int((sample.metadata or {})["source_draw_id"])])
                for sample in flat
            ]
        else:
            retention_weights = [float(train_data["sdpo_loss_weights"][index]) for index in selected_indices]
        retention_precomputed = [
            getattr(flat[index], "pr_precomputed", None)
            for index in (range(len(flat)) if retention_support == "all" else selected_indices)
        ]
        if any(payload is not None for payload in retention_precomputed) and not all(
            isinstance(payload, dict) for payload in retention_precomputed
        ):
            raise ValueError("dense retention rows mix cached and uncached targets")
        expand_pr_rows(
            args,
            train_data,
            retention_weight=retention_weight,
            selected_indices=selected_indices,
            retention_base_weights=retention_weights,
            retention_precomputed=(
                retention_precomputed
                if retention_precomputed and all(isinstance(payload, dict) for payload in retention_precomputed)
                else None
            ),
        )
    elif len(selected_indices) != len(flat):
        raise AssertionError("unselected rows may only be retained for all-step privileged retention")
    return train_data


def _flatten(samples: list[Sample] | list[list[Sample]]) -> list[Sample]:
    rows: list[Sample] = []
    for item in samples:
        if isinstance(item, Sample):
            rows.append(item)
        elif isinstance(item, list):
            rows.extend(_flatten(item))
        else:
            raise TypeError(f"unexpected sample node: {type(item).__name__}")
    return rows
