"""Sensitivity-Guided Selection (SGS): score, rank, and select interaction steps."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from slime.utils.types import Sample

PRECOMPUTED_FIELDS = (
    "sdpo_topk_indices",
    "sdpo_teacher_log_probs",
    "sdpo_teacher_topk_log_probs",
)
RANKING_SCORE_FIELDS = {
    "teacher_js",
    "teacher_kl_plain_privileged",
    "teacher_kl_privileged_plain",
    "distillation_kl",
}


def sgs_selection_fraction(args: Any) -> float | None:
    value = getattr(args, "sgs_selection_fraction", None)
    return None if value is None else float(value)


def sgs_option(args: Any, name: str, default: Any = None) -> Any:
    value = getattr(args, f"sgs_{name}", None)
    return default if value is None else value


def sgs_filtering_enabled(args: Any) -> bool:
    fraction = sgs_selection_fraction(args)
    if fraction is None:
        return False
    if not 0.0 < fraction <= 1.0:
        raise ValueError("SGS fraction must be in (0, 1]")
    full_selection_retention = bool(sgs_option(args, "full_selection_retention", False))
    if fraction == 1.0 and full_selection_retention:
        retention_weight = float(getattr(args, "pr_weight", 0.0) or 0.0)
        retention_support = str(getattr(args, "pr_support", "selected"))
        if retention_weight <= 0.0 or retention_support != "all":
            raise ValueError("full sensitivity selection requires positive all-step privileged retention")
        return True
    # x=100 without the explicit retention capability remains unfiltered distillation
    # compatibility mode: no scoring metadata, prepass, ranking, packing, or
    # converter mutation is allowed.
    return fraction < 1.0


def sgs_ranking_score_field(args: Any) -> str:
    field = str(sgs_option(args, "ranking_score", "teacher_js"))
    if field not in RANKING_SCORE_FIELDS:
        raise ValueError(f"unsupported sensitivity ranking score: {field}")
    return field


def sgs_scoring_required(args: Any) -> bool:
    if not sgs_filtering_enabled(args):
        return False
    skip = bool(sgs_option(args, "skip_full_scoring", False))
    mode = str(getattr(args, "sgs_selection_mode", "sensitivity"))
    if skip and mode != "random":
        raise ValueError("only random sensitivity filtering may skip full scoring")
    return not skip


def score_masks_for_scope(loss_masks: list[Any], action_masks: list[Any], *, score_scope: str) -> list[list[bool]]:
    if score_scope not in {"response", "action"}:
        raise ValueError(f"unsupported sensitivity score scope: {score_scope}")
    source = loss_masks if score_scope == "response" else action_masks
    import torch

    return [torch.as_tensor(mask, dtype=torch.bool).reshape(-1).tolist() for mask in source]


def sgs_score_available(*, score_scope: str, alignment_valid: bool, score_token_count: int) -> bool:
    if score_scope not in {"response", "action"}:
        raise ValueError(f"unsupported sensitivity score scope: {score_scope}")
    return score_token_count > 0 and (score_scope == "response" or alignment_valid)


def teacher_js_statistics(js_values: Any, *, expected_token_count: int) -> tuple[float, float]:
    import torch

    values = torch.as_tensor(js_values, dtype=torch.float64).reshape(-1)
    if int(values.numel()) != int(expected_token_count):
        raise RuntimeError("teacher JS token count does not match the explicit sensitivity score mask")
    js_mean = float(values.mean().item())
    js_sum = float(values.sum().item())
    if not math.isfinite(js_mean) or not math.isfinite(js_sum):
        raise RuntimeError("teacher JS contains non-finite values")
    return js_mean, js_sum


@dataclass(frozen=True)
class SGSSelection:
    selected_keys: tuple[tuple[int, int], ...]
    base_selected_keys: tuple[tuple[int, int], ...]
    dp_floor_added_keys: tuple[tuple[int, int], ...]
    trajectory_floor_added_keys: tuple[tuple[int, int], ...]
    attempted_count: int
    scoreable_count: int
    requested_count: int
    selected_count: int
    dp_floor_applied: bool
    attempted_trajectory_count: int
    scoreable_trajectory_count: int
    base_selected_trajectory_count: int
    selected_trajectory_count: int


def sample_key(sample: Sample) -> tuple[int, int]:
    metadata = sample.metadata or {}
    return int(metadata["source_draw_id"]), int(metadata["source_turn_idx"])


def score_key(record: dict[str, Any]) -> tuple[int, int]:
    return int(record["source_draw_id"]), int(record["turn_idx"])


def random_rank_key(record: dict[str, Any], *, selection_seed: int, rollout_id: int) -> tuple[str, str, int]:
    """Return a stable per-update random rank without process-local hash state."""
    identity = "\x1f".join(
        (
            str(int(selection_seed)),
            str(int(rollout_id)),
            str(int(record["source_draw_id"])),
            str(int(record["turn_idx"])),
            str(record["traj_uid"]),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest(), str(record["traj_uid"]), int(record["turn_idx"])


def select_sgs_steps(
    records: Iterable[dict[str, Any]],
    *,
    fraction: float,
    minimum_selected: int,
    selection_mode: str = "sensitivity",
    selection_seed: int = 42,
    rollout_id: int = 0,
    selection_scope: str = "global",
    score_field: str | None = "teacher_js",
    ranking_order: str = "descending",
) -> SGSSelection:
    rows = list(records)
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("SGS fraction must be in (0, 1]")
    if minimum_selected <= 0:
        raise ValueError("minimum_selected must be positive")
    if selection_mode not in {"sensitivity", "random"}:
        raise ValueError(f"unsupported step selection mode: {selection_mode}")
    if selection_scope not in {"global", "trajectory", "global_with_trajectory_floor"}:
        raise ValueError(f"unsupported step selection scope: {selection_scope}")
    if ranking_order not in {"descending", "ascending"}:
        raise ValueError(f"unsupported sensitivity ranking order: {ranking_order}")
    if selection_mode == "random" and ranking_order != "descending":
        raise ValueError("ranking order is not configurable for random filtering")
    if score_field is not None and score_field not in RANKING_SCORE_FIELDS:
        raise ValueError(f"unsupported sensitivity ranking score: {score_field}")
    if score_field is None and (selection_mode != "random" or selection_scope != "global"):
        raise ValueError("attempted-population selection is only supported for global random filtering")
    keys = [score_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("sensitivity scoring returned duplicate source-step identities")

    scoreable = (
        rows
        if score_field is None
        else [
            row
            for row in rows
            if row.get(score_field) is not None
            and math.isfinite(float(row[score_field]))
            and float(row[score_field]) >= 0.0
        ]
    )
    if not scoreable:
        raise RuntimeError("sensitivity scoring produced no scoreable steps")

    def rank(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if selection_mode == "random":
            return sorted(
                group,
                key=lambda row: random_rank_key(
                    row,
                    selection_seed=selection_seed,
                    rollout_id=rollout_id,
                ),
            )
        direction = -1.0 if ranking_order == "descending" else 1.0
        return sorted(
            group,
            key=lambda row: (
                direction * float(row[score_field]),
                str(row["traj_uid"]),
                int(row["turn_idx"]),
            ),
        )

    attempted_trajectory_keys = {(int(row["source_draw_id"]), str(row["traj_uid"])) for row in rows}
    scoreable_trajectory_keys = {(int(row["source_draw_id"]), str(row["traj_uid"])) for row in scoreable}
    dp_floor_added_rows: list[dict[str, Any]] = []
    trajectory_floor_added_rows: list[dict[str, Any]] = []
    if selection_scope in {"global", "global_with_trajectory_floor"}:
        requested = math.ceil(float(fraction) * len(scoreable))
        selected_count = min(len(scoreable), max(requested, int(minimum_selected)))
        ranked = rank(scoreable)
        selected_rows = ranked[:selected_count]
        base_selected_rows = list(selected_rows)
        dp_floor_added_rows = ranked[requested:selected_count]
        dp_floor_applied = selected_count > requested
        if selection_scope == "global_with_trajectory_floor":
            trajectories_by_draw: dict[int, set[str]] = {}
            scoreable_by_trajectory: dict[tuple[int, str], list[dict[str, Any]]] = {}
            for row in rows:
                trajectories_by_draw.setdefault(int(row["source_draw_id"]), set()).add(str(row["traj_uid"]))
            conflicting = sorted(draw_id for draw_id, traj_uids in trajectories_by_draw.items() if len(traj_uids) != 1)
            if conflicting:
                raise RuntimeError(
                    f"trajectory-floor selection found multiple trajectory UIDs for source draws: {conflicting[:5]}"
                )
            for row in scoreable:
                trajectory_key = (int(row["source_draw_id"]), str(row["traj_uid"]))
                scoreable_by_trajectory.setdefault(trajectory_key, []).append(row)
            missing = sorted(attempted_trajectory_keys - set(scoreable_by_trajectory))
            if missing:
                raise RuntimeError(
                    f"trajectory-floor selection found zero scoreable steps for trajectories: {missing[:5]}"
                )
            selected_keys = {score_key(row) for row in selected_rows}
            for trajectory_key in sorted(attempted_trajectory_keys):
                group = scoreable_by_trajectory[trajectory_key]
                if any(score_key(row) in selected_keys for row in group):
                    continue
                fallback = rank(group)[0]
                selected_rows.append(fallback)
                selected_keys.add(score_key(fallback))
                trajectory_floor_added_rows.append(fallback)
            selected_count = len(selected_rows)
    else:
        attempted_by_trajectory: dict[tuple[int, str], list[dict[str, Any]]] = {}
        scoreable_by_trajectory: dict[tuple[int, str], list[dict[str, Any]]] = {}
        trajectories_by_draw: dict[int, set[str]] = {}
        for row in rows:
            draw_id = int(row["source_draw_id"])
            traj_uid = str(row["traj_uid"])
            attempted_by_trajectory.setdefault((draw_id, traj_uid), []).append(row)
            trajectories_by_draw.setdefault(draw_id, set()).add(traj_uid)
        conflicting = sorted(draw_id for draw_id, traj_uids in trajectories_by_draw.items() if len(traj_uids) != 1)
        if conflicting:
            raise RuntimeError(
                f"trajectory selection found multiple trajectory UIDs for source draws: {conflicting[:5]}"
            )
        for row in scoreable:
            key = (int(row["source_draw_id"]), str(row["traj_uid"]))
            scoreable_by_trajectory.setdefault(key, []).append(row)
        missing = sorted(set(attempted_by_trajectory) - set(scoreable_by_trajectory))
        if missing:
            raise RuntimeError(f"trajectory selection found zero scoreable steps for trajectories: {missing[:5]}")
        selected_rows = []
        requested = 0
        for trajectory_key in sorted(attempted_by_trajectory):
            group = scoreable_by_trajectory[trajectory_key]
            keep = max(1, math.ceil(float(fraction) * len(group)))
            requested += keep
            selected_rows.extend(rank(group)[:keep])
        selected_count = len(selected_rows)
        base_selected_rows = list(selected_rows)
        dp_floor_applied = False
    base_selected_trajectory_keys = {(int(row["source_draw_id"]), str(row["traj_uid"])) for row in base_selected_rows}
    selected_trajectory_keys = {(int(row["source_draw_id"]), str(row["traj_uid"])) for row in selected_rows}
    return SGSSelection(
        selected_keys=tuple(score_key(row) for row in selected_rows),
        base_selected_keys=tuple(score_key(row) for row in base_selected_rows),
        dp_floor_added_keys=tuple(score_key(row) for row in dp_floor_added_rows),
        trajectory_floor_added_keys=tuple(score_key(row) for row in trajectory_floor_added_rows),
        attempted_count=len(rows),
        scoreable_count=len(scoreable),
        requested_count=requested,
        selected_count=selected_count,
        dp_floor_applied=dp_floor_applied,
        attempted_trajectory_count=len(attempted_trajectory_keys),
        scoreable_trajectory_count=len(scoreable_trajectory_keys),
        base_selected_trajectory_count=len(base_selected_trajectory_keys),
        selected_trajectory_count=len(selected_trajectory_keys),
    )


def bind_selection_to_samples(
    samples: list[Sample],
    records: Iterable[dict[str, Any]],
    selection: SGSSelection,
    *,
    require_precomputed: bool = True,
    retain_unselected: bool = False,
) -> list[Sample]:
    record_by_key = {score_key(row): row for row in records}
    sample_keys = [sample_key(sample) for sample in samples]
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("sensitivity filtering requires one generated response per source step")
    if set(sample_keys) != set(record_by_key):
        missing_scores = sorted(set(sample_keys) - set(record_by_key))[:5]
        missing_samples = sorted(set(record_by_key) - set(sample_keys))[:5]
        raise ValueError(
            "sensitivity score/sample identity mismatch: "
            f"missing_scores={missing_scores}, missing_samples={missing_samples}"
        )
    selected = set(selection.selected_keys)
    output = []
    for sample in samples:
        key = sample_key(sample)
        record = record_by_key[key]
        is_selected = key in selected
        setattr(sample, "sgs_finalized", True)
        setattr(sample, "sgs_scored", require_precomputed)
        setattr(sample, "sgs_selected", is_selected)
        setattr(sample, "sgs_score_record", sgs_audit_record(record))
        retention_payload = record.get("retention_precomputed")
        if retention_payload is not None:
            if not retain_unselected:
                raise ValueError("dense retention tensors require retained unselected rows")
            if not isinstance(retention_payload, dict) or any(
                field not in retention_payload for field in PRECOMPUTED_FIELDS
            ):
                raise ValueError(f"sensitivity row {key} has invalid dense retention tensors")
            setattr(sample, "pr_precomputed", retention_payload)
        if is_selected:
            payload = record.get("precomputed")
            if require_precomputed:
                if not isinstance(payload, dict) or any(field not in payload for field in PRECOMPUTED_FIELDS):
                    raise ValueError(f"selected sensitivity row {key} is missing precomputed SDPO tensors")
                setattr(sample, "sgs_precomputed", payload)
            elif payload is not None:
                raise ValueError(f"unscored random row {key} unexpectedly contains precomputed SDPO tensors")
        if is_selected or retain_unselected:
            output.append(sample)
    if (not retain_unselected and len(output) != selection.selected_count) or (
        retain_unselected and len(output) != len(samples)
    ):
        raise AssertionError("selected sample count does not match sensitivity selection")
    return output


def sgs_audit_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in {"precomputed", "retention_precomputed"}}
