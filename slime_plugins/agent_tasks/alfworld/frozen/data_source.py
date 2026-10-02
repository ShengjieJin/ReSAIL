from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch

from slime.rollout.data_source import DataSource
from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.artifacts import write_json_once

from ..data_source import AlfWorldDataSource
from .capabilities import FROZEN_ARMS, capabilities_for_args
from .contracts import expected_trajectory_count

FROZEN_SHARD_SIZE = 128
ITERATIVE_FROZEN_SCHEMA_VERSION = 4
PAIR_AUDIT_SCHEMA_VERSION = 1


def _validate_epd_target_preflight(target_root: Path, corpus_dir: Path, args: Any) -> None:
    """Validate the small immutable target binding before datasource construction.

    The full shard/teacher-prompt audit remains in ``verify_epd_targets.py``;
    this check deliberately reads only manifest metadata so the train datasource
    never retains the privileged teacher prompt arrays.
    """

    manifest_path = target_root / "manifest.json"
    verification_path = target_root / "verification.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"EPD target preflight metadata is invalid: {target_root}") from exc
    from .epd import EPD_MANIFEST_SCHEMA_VERSION

    shard_paths = sorted(corpus_dir.glob("batch_*.pt"))
    corpus_trajectories = []
    for shard_path in shard_paths:
        corpus_trajectories.extend(load_frozen_trajectory_shard(shard_path))
    corpus_trajectory_count = len(corpus_trajectories)
    corpus_turn_count = sum(len(trajectory["turns"]) for trajectory in corpus_trajectories)
    expected_trajectories = expected_trajectory_count(args)
    subset_target_contract = bool(getattr(args, "alfworld_epd_subset_target_contract", False))
    if corpus_trajectory_count != expected_trajectories or corpus_turn_count <= 0:
        raise ValueError(
            f"EPD requires the complete {expected_trajectories:,}-trajectory frozen corpus; "
            f"found trajectories={corpus_trajectory_count}, turns={corpus_turn_count}"
        )
    target_trajectories = corpus_trajectories
    subset_path_value = str(getattr(args, "alfworld_frozen_subset_manifest", "") or "").strip()
    if subset_path_value and subset_target_contract:
        subset = json.loads(Path(subset_path_value).read_text(encoding="utf-8"))
        if not isinstance(subset, dict):
            raise ValueError("EPD frozen trajectory subset manifest must be a mapping")
        ordered_uids = subset.get("trajectory_uids")
        expected_subset_count = int(
            getattr(args, "alfworld_frozen_expected_subset_trajectories", 0) or 0
        )
        by_uid = {str(row["trajectory_uid"]): row for row in corpus_trajectories}
        if (
            subset.get("schema_version") != 1
            or subset.get("kind") != "alfworld_frozen_trajectory_subset"
            or not isinstance(ordered_uids, list)
            or len(ordered_uids) != expected_subset_count
            or int(subset.get("trajectory_count", -1)) != expected_subset_count
            or len(set(ordered_uids)) != len(ordered_uids)
            or any(str(uid) not in by_uid for uid in ordered_uids)
        ):
            raise ValueError("EPD frozen trajectory subset is invalid for the configured corpus")
        target_trajectories = [by_uid[str(uid)] for uid in ordered_uids]
    target_trajectory_count = len(target_trajectories)
    target_turn_count = sum(len(trajectory["turns"]) for trajectory in target_trajectories)
    if (
        manifest.get("schema_version") != EPD_MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != "epd_teacher_targets"
        or manifest.get("status") != "complete"
        or manifest.get("target_count") != target_turn_count
        or manifest.get("trajectory_count") != target_trajectory_count
        or manifest.get("canonical_turn_count") != target_turn_count
        or manifest.get("success_filter") is not False
        or manifest.get("branch_packing") is not False
        or manifest.get("student_visible_prefix") != "original"
    ):
        raise ValueError("EPD target manifest is not the complete unfiltered corpus")
    declared_binding = getattr(args, "alfworld_epd_direct_binding", None)
    expected_binding = {
        "corpus_dir": str(corpus_dir),
        "materialization_identity": str(
            getattr(args, "alfworld_epd_materialization_identity", "") or ""
        ),
        "model_path": str(getattr(args, "hf_checkpoint", "") or ""),
        "teacher_iteration": int(getattr(args, "alfworld_epd_teacher_iteration", -1)),
    }
    guidance_summary_dir = str(
        getattr(args, "agent_frozen_guidance_summary_dir", "") or ""
    ).strip()
    if guidance_summary_dir:
        expected_binding["guidance_summary_dir"] = guidance_summary_dir
    if (
        not isinstance(declared_binding, dict)
        or declared_binding != expected_binding
        or manifest.get("binding_mode") != "direct_v1"
        or manifest.get("direct_binding") != expected_binding
        or verification.get("status") != "verified"
        or verification.get("binding_mode") != "direct_v1"
        or verification.get("direct_binding") != expected_binding
        or verification.get("corpus_bound") is not True
        or verification.get("trajectory_count") != target_trajectory_count
        or verification.get("canonical_turn_count") != target_turn_count
    ):
        raise ValueError("EPD direct target binding differs from the configured model/corpus identity")

class FrozenCorpusCollectionDataSource(AlfWorldDataSource):
    """Deployment data source that resumes its global stream from verified corpus shards."""

    def __init__(self, args: Any):
        super().__init__(args)
        if int(getattr(args, "n_samples_per_prompt", 1)) != 1:
            raise ValueError("frozen corpus collection requires n_samples_per_prompt=1")
        configured_root = str(getattr(args, "alfworld_frozen_corpus_dir", "") or "").strip()
        if not configured_root:
            raise ValueError("alfworld_frozen_corpus_dir is required for frozen corpus collection")
        shard_paths = sorted(Path(configured_root).glob("batch_*.pt"))
        completed: list[dict[str, Any]] = []
        for expected_shard_idx, shard_path in enumerate(shard_paths):
            expected_name = f"batch_{expected_shard_idx:03d}.pt"
            if shard_path.name != expected_name:
                raise ValueError(
                    f"frozen corpus shard sequence has a gap: expected {expected_name}, got {shard_path.name}"
                )
            completed.extend(load_frozen_trajectory_shard(shard_path))
        completed_trajectories = len(completed)
        self.sample_group_index = completed_trajectories
        self.sample_index = completed_trajectories
        self.epoch_id = completed_trajectories // max(1, self.group_count())


def _infer_frozen_trajectory_schema_version(trajectory: dict[str, Any]) -> int:
    if not isinstance(trajectory.get("provenance"), dict):
        raise TypeError("frozen trajectory provenance must be a mapping")
    return ITERATIVE_FROZEN_SCHEMA_VERSION


def verify_frozen_trajectory(trajectory: dict[str, Any], *, schema_version: int | None = None) -> None:
    if schema_version is None:
        schema_version = _infer_frozen_trajectory_schema_version(trajectory)
    if schema_version != ITERATIVE_FROZEN_SCHEMA_VERSION:
        raise ValueError(f"unsupported frozen trajectory schema version: {schema_version}")
    _reject_expectation_fields(trajectory, location="frozen trajectory")
    required = {
        "trajectory_uid",
        "task_id",
        "task_description",
        "split",
        "success",
        "outcome",
        "episode_reward",
        "termination_reason",
        "truncated",
        "horizon_reached",
        "errors",
        "turns",
    }
    missing = sorted(required - trajectory.keys())
    if missing:
        raise ValueError(f"frozen trajectory missing fields: {missing}")
    if not isinstance(trajectory["trajectory_uid"], str) or not trajectory["trajectory_uid"]:
        raise ValueError("trajectory_uid must be a non-empty string")
    _verify_iterative_trajectory_contract(trajectory)
    success = trajectory["success"]
    if not isinstance(success, bool):
        raise TypeError("frozen trajectory success must be bool")
    expected_outcome = "success" if success else "failure"
    if trajectory["outcome"] != expected_outcome:
        raise ValueError(f"outcome must be {expected_outcome!r} when success={success}")
    if not isinstance(trajectory["task_description"], str) or not trajectory["task_description"]:
        raise ValueError("frozen trajectory task_description must be a non-empty string")
    if not isinstance(trajectory["episode_reward"], (int, float)):
        raise TypeError("frozen trajectory episode_reward must be numeric")
    if not isinstance(trajectory["termination_reason"], str) or not trajectory["termination_reason"]:
        raise ValueError("frozen trajectory termination_reason must be a non-empty string")
    for key in ("truncated", "horizon_reached"):
        if not isinstance(trajectory[key], bool):
            raise TypeError(f"frozen trajectory {key} must be bool")
    if not isinstance(trajectory["errors"], (dict, list)):
        raise TypeError("frozen trajectory errors must be a dict or list")
    turns = trajectory["turns"]
    if not isinstance(turns, list) or not turns:
        raise ValueError("frozen trajectory turns must be a non-empty list")
    for expected_turn_idx, turn in enumerate(turns):
        _verify_frozen_turn(turn, expected_turn_idx=expected_turn_idx, schema_version=schema_version)
    final_turn = turns[-1]
    if success and not final_turn["is_terminal"]:
        raise ValueError("successful frozen trajectory must end in a terminal turn")
    if any(turn["is_terminal"] for turn in turns[:-1]):
        raise ValueError("only the final frozen turn may be terminal")
    if trajectory["termination_reason"] not in {"success", "environment_terminal", "horizon", "error", "incomplete"}:
        raise ValueError("frozen trajectory termination_reason is not recognized")
    if success != (trajectory["termination_reason"] == "success"):
        raise ValueError("successful outcome and termination_reason=success must agree")
    if trajectory["horizon_reached"] != (trajectory["termination_reason"] == "horizon"):
        raise ValueError("horizon_reached and termination_reason=horizon must agree")
    if bool(trajectory["errors"]) != (trajectory["termination_reason"] == "error"):
        raise ValueError("trajectory errors and termination_reason=error must agree")
    expected_truncated = trajectory["horizon_reached"] or any(turn["finish_reason"] == "length" for turn in turns)
    if trajectory["truncated"] != expected_truncated:
        raise ValueError("trajectory truncated must derive from horizon or a length-finished response")


def _verify_frozen_turn(turn: dict[str, Any], *, expected_turn_idx: int, schema_version: int) -> None:
    if "loss_mask" in turn:
        raise ValueError(f"ordinary-only frozen turn {expected_turn_idx} must not persist a loss mask")
    required = {
        "turn_idx",
        "messages",
        "prompt_ids",
        "response_text",
        "response_ids",
        "deployment_behavior_log_probs",
        "current_observation",
        "next_observation",
        "frozen_action",
        "env_reward",
        "finish_reason",
        "prompt_overlength",
        "prompt_truncated",
        "is_terminal",
        "format_valid",
        "format_invalid_reason",
        "errors",
    }
    missing = sorted(required - turn.keys())
    if missing:
        raise ValueError(f"frozen turn {expected_turn_idx} missing fields: {missing}")
    if int(turn["turn_idx"]) != expected_turn_idx:
        raise ValueError(f"frozen turns must be ordered and contiguous; expected {expected_turn_idx}")
    if not isinstance(turn["messages"], list) or not all(
        isinstance(message, dict) and set(("role", "content")) <= message.keys() for message in turn["messages"]
    ):
        raise TypeError(f"frozen turn {expected_turn_idx} messages must contain role/content dicts")
    prompt_ids = turn["prompt_ids"]
    response_ids = turn["response_ids"]
    if not isinstance(prompt_ids, list) or not all(isinstance(token_id, int) for token_id in prompt_ids):
        raise TypeError(f"frozen turn {expected_turn_idx} prompt_ids must be list[int]")
    if not prompt_ids:
        raise ValueError(f"frozen turn {expected_turn_idx} prompt_ids must not be empty")
    if not isinstance(response_ids, list) or not all(isinstance(token_id, int) for token_id in response_ids):
        raise TypeError(f"frozen turn {expected_turn_idx} response_ids must be list[int]")
    if not response_ids:
        raise ValueError(f"frozen turn {expected_turn_idx} response_ids must not be empty")
    log_probs = turn["deployment_behavior_log_probs"]
    if not isinstance(log_probs, torch.Tensor) or log_probs.dtype != torch.float32 or log_probs.ndim != 1:
        raise ValueError(f"frozen turn {expected_turn_idx} deployment_behavior_log_probs must be a 1-D float32 tensor")
    if log_probs.numel() != len(response_ids):
        raise ValueError(
            f"frozen turn {expected_turn_idx} behavior log-prob length {log_probs.numel()} "
            f"!= response length {len(response_ids)}"
        )
    if not torch.isfinite(log_probs).all():
        raise ValueError(f"frozen turn {expected_turn_idx} deployment_behavior_log_probs must be finite")
    if not isinstance(turn["response_text"], str):
        raise TypeError(f"frozen turn {expected_turn_idx} response_text must be str")
    for key in ("current_observation", "next_observation", "frozen_action"):
        if not isinstance(turn[key], str):
            raise TypeError(f"frozen turn {expected_turn_idx} {key} must be str")
    if not isinstance(turn["env_reward"], (int, float)):
        raise TypeError(f"frozen turn {expected_turn_idx} env_reward must be numeric")
    if not isinstance(turn["finish_reason"], str) or not turn["finish_reason"]:
        raise TypeError(f"frozen turn {expected_turn_idx} finish_reason must be a non-empty string")
    if turn["finish_reason"] not in {"stop", "length"}:
        raise ValueError(f"frozen turn {expected_turn_idx} finish_reason must be stop or length")
    for key in (
        "prompt_overlength",
        "prompt_truncated",
        "is_terminal",
        "format_valid",
    ):
        if not isinstance(turn[key], bool):
            raise TypeError(f"frozen turn {expected_turn_idx} {key} must be bool")
    if turn["format_invalid_reason"] is not None and not isinstance(turn["format_invalid_reason"], str):
        raise TypeError(f"frozen turn {expected_turn_idx} format_invalid_reason must be str or None")
    if turn["format_valid"] == bool(turn["format_invalid_reason"]):
        raise ValueError(f"frozen turn {expected_turn_idx} format validity/reason are inconsistent")
    if not isinstance(turn["errors"], (dict, list)):
        raise TypeError(f"frozen turn {expected_turn_idx} errors must be a dict or list")


def _reject_expectation_fields(value: Any, *, location: str) -> None:
    if isinstance(value, dict):
        forbidden = [key for key in value if "expectation" in str(key).lower()]
        if forbidden:
            raise ValueError(f"{location} contains forbidden Expectation fields: {sorted(forbidden)}")
        for key, item in value.items():
            _reject_expectation_fields(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_expectation_fields(item, location=f"{location}[{index}]")


def _verify_iterative_trajectory_contract(trajectory: dict[str, Any]) -> None:
    runtime_identity = trajectory.get("runtime_task_identity")
    provenance = trajectory.get("provenance")
    if not isinstance(runtime_identity, dict) or not runtime_identity.get("task_description"):
        raise ValueError("iterative frozen trajectory requires its runtime task identity")
    required = {"stream_index", "task_seed", "source_group_index", "seed", "task_identity"}
    if not isinstance(provenance, dict) or required - provenance.keys():
        raise ValueError(f"iterative provenance missing fields: {sorted(required - set(provenance or {}))}")
    for key in ("stream_index", "task_seed", "source_group_index", "seed"):
        if not isinstance(provenance[key], int):
            raise TypeError(f"iterative provenance {key} must be int")
    if not isinstance(provenance["task_identity"], dict) or not provenance["task_identity"]:
        raise ValueError("iterative provenance task_identity must be a non-empty object")


def verify_frozen_trajectory_shard(
    trajectories: list[dict[str, Any]],
    *,
    require_full_shard: bool = True,
    schema_version: int | None = None,
) -> None:
    if not isinstance(trajectories, list):
        raise TypeError("frozen trajectory shard must be a list")
    if require_full_shard and len(trajectories) != FROZEN_SHARD_SIZE:
        raise ValueError(f"frozen trajectory shard must contain exactly {FROZEN_SHARD_SIZE} trajectories")
    if not require_full_shard and not trajectories:
        raise ValueError("frozen trajectory shard must not be empty")
    trajectory_uids: set[str] = set()
    for trajectory in trajectories:
        verify_frozen_trajectory(trajectory, schema_version=schema_version)
        trajectory_uid = trajectory["trajectory_uid"]
        if trajectory_uid in trajectory_uids:
            raise ValueError(f"duplicate trajectory_uid in shard: {trajectory_uid}")
        trajectory_uids.add(trajectory_uid)


def write_frozen_trajectory_shard(
    path: str | Path,
    trajectories: list[dict[str, Any]],
    *,
    schema_version: int | None = None,
    require_full_shard: bool = True,
) -> None:
    canonical = [_canonicalize_trajectory(trajectory) for trajectory in trajectories]
    if schema_version is None:
        schema_version = ITERATIVE_FROZEN_SCHEMA_VERSION
    verify_frozen_trajectory_shard(
        canonical,
        require_full_shard=require_full_shard,
        schema_version=schema_version,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        payload = {"schema_version": schema_version, "trajectories": canonical}
        if not require_full_shard:
            payload["shard_size"] = len(canonical)
        torch.save(payload, temporary_path)
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_frozen_trajectory_shard(path: str | Path) -> list[dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != ITERATIVE_FROZEN_SCHEMA_VERSION:
        raise ValueError(f"unsupported frozen trajectory shard schema in {path}")
    if set(payload) not in ({"schema_version", "trajectories"}, {"schema_version", "trajectories", "shard_size"}):
        raise ValueError(f"unexpected frozen trajectory shard fields in {path}: {sorted(payload)}")
    trajectories = payload["trajectories"]
    declared_shard_size = payload.get("shard_size")
    if declared_shard_size is not None and payload["schema_version"] != ITERATIVE_FROZEN_SCHEMA_VERSION:
        raise ValueError(f"declared frozen shard size is only valid for the iterative schema in {path}")
    if declared_shard_size is not None and (
        not isinstance(declared_shard_size, int)
        or declared_shard_size <= 0
        or declared_shard_size != len(trajectories)
    ):
        raise ValueError(f"invalid declared frozen shard size in {path}")
    verify_frozen_trajectory_shard(
        trajectories,
        require_full_shard=declared_shard_size is None,
        schema_version=int(payload["schema_version"]),
    )
    return trajectories


def _canonicalize_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    value = dict(trajectory)
    if "episode_reward" in value:
        value["episode_reward"] = float(value["episode_reward"])
    value["turns"] = []
    for raw_turn in trajectory.get("turns", []):
        turn = dict(raw_turn)
        turn["prompt_ids"] = [int(token_id) for token_id in turn.get("prompt_ids", [])]
        turn["response_ids"] = [int(token_id) for token_id in turn.get("response_ids", [])]
        turn["deployment_behavior_log_probs"] = torch.as_tensor(
            turn.get("deployment_behavior_log_probs", []), dtype=torch.float32
        ).cpu()
        if "env_reward" in turn:
            turn["env_reward"] = float(turn["env_reward"])
        value["turns"].append(turn)
    return value


class FrozenAlfWorldDataSource(DataSource):
    state_filename_prefix = "alfworld_frozen_data_source_state"

    def __init__(self, args: Any):
        self.args = args
        self.arm = str(getattr(args, "alfworld_frozen_arm", ""))
        if self.arm not in FROZEN_ARMS:
            raise ValueError(f"alfworld_frozen_arm must be one of {sorted(FROZEN_ARMS)}, got {self.arm!r}")
        self.capabilities = capabilities_for_args(args, self.arm)
        self.n_samples_per_prompt = int(getattr(args, "n_samples_per_prompt", 1))
        required_fanout = self.capabilities.fanout
        if self.n_samples_per_prompt != required_fanout:
            raise ValueError(f"{self.arm} requires n_samples_per_prompt={required_fanout}")
        corpus_dir = Path(getattr(args, "alfworld_frozen_corpus_dir", ""))
        if not corpus_dir.is_dir():
            raise FileNotFoundError(f"frozen ALFWorld corpus directory does not exist: {corpus_dir}")
        self._shard_paths = sorted(corpus_dir.glob("batch_*.pt"))
        if not self._shard_paths:
            raise FileNotFoundError(f"no batch_*.pt frozen shards found under {corpus_dir}")
        if self.capabilities.response_source == "epd":
            target_root = Path(str(getattr(args, "alfworld_epd_target_dir", "") or "").strip())
            if not target_root.is_dir():
                raise FileNotFoundError(f"EPD target directory does not exist: {target_root}")
            if not (target_root / "manifest.json").is_file():
                raise FileNotFoundError(f"EPD target manifest is missing: {target_root / 'manifest.json'}")
            _validate_epd_target_preflight(target_root, corpus_dir, args)
        self._locations: list[tuple[int, int, str, bool]] = []
        self._shard_cache: dict[int, list[dict[str, Any]]] = {}
        all_locations: dict[str, tuple[int, int, str, bool]] = {}
        seen_uids: set[str] = set()
        for shard_idx, shard_path in enumerate(self._shard_paths):
            trajectories = load_frozen_trajectory_shard(shard_path)
            self._shard_cache[shard_idx] = trajectories
            for trajectory_idx, trajectory in enumerate(trajectories):
                trajectory_uid = trajectory["trajectory_uid"]
                if trajectory_uid in seen_uids:
                    raise ValueError(f"duplicate trajectory_uid across frozen shards: {trajectory_uid}")
                seen_uids.add(trajectory_uid)
                location = (shard_idx, trajectory_idx, trajectory_uid, trajectory["success"])
                all_locations[trajectory_uid] = location
                if not self.capabilities.success_only or trajectory["success"]:
                    self._locations.append(location)
        subset_path_value = str(getattr(args, "alfworld_frozen_subset_manifest", "") or "").strip()
        if subset_path_value:
            subset_path = Path(subset_path_value)
            subset = json.loads(subset_path.read_text(encoding="utf-8"))
            if not isinstance(subset, dict):
                raise ValueError("frozen trajectory subset manifest must be a mapping")
            ordered_uids = subset.get("trajectory_uids")
            expected_subset_count = int(
                getattr(args, "alfworld_frozen_expected_subset_trajectories", 0) or 0
            )
            if (
                subset.get("schema_version") != 1
                or subset.get("kind") != "alfworld_frozen_trajectory_subset"
                or not isinstance(ordered_uids, list)
                or len(ordered_uids) != expected_subset_count
                or int(subset.get("trajectory_count", -1)) != expected_subset_count
                or len(set(ordered_uids)) != len(ordered_uids)
                or any(uid not in all_locations for uid in ordered_uids)
            ):
                raise ValueError("frozen trajectory subset manifest is invalid for the configured corpus")
            self._locations = [
                all_locations[uid]
                for uid in ordered_uids
                if not self.capabilities.success_only or all_locations[uid][3]
            ]
        if not self._locations:
            raise ValueError(f"no eligible frozen trajectories for {self.arm}")
        self.sample_offset = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.epoch_id = 0
        self.metadata: dict[str, Any] = {}
        self._epoch_permutations: dict[int, list[int]] = {}
        self._next_rollout_id = 0
        self._checkpoint_states: dict[int, dict[str, Any]] = {}
        audit_root = str(getattr(args, "alfworld_frozen_pair_audit_dir", "") or "").strip()
        self._pair_audit_dir = Path(audit_root) if audit_root else None
        if self._pair_audit_dir is not None:
            if not self.capabilities.pair_audit:
                raise ValueError("alfworld_frozen_pair_audit_dir is only valid for paired frozen trajectories")
            self._pair_audit_dir.mkdir(parents=True, exist_ok=True)
        source_audit_root = str(getattr(args, "alfworld_frozen_source_audit_dir", "") or "").strip()
        self._source_audit_dir = Path(source_audit_root) if source_audit_root else None
        if self._source_audit_dir is not None:
            if not self.capabilities.source_audit:
                raise ValueError("alfworld_frozen_source_audit_dir is not valid for this frozen arm")
            self._source_audit_dir.mkdir(parents=True, exist_ok=True)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        rollout_id = self._next_rollout_id
        groups: list[list[Sample]] = []
        for _ in range(num_samples):
            source_idx, epoch_id, epoch_offset = self._source_index(self.sample_offset)
            source_turn_idx = None
            shard_idx, trajectory_idx, trajectory_uid, success = self._locations[source_idx]
            trajectory = self._load_trajectory(shard_idx, trajectory_idx)
            source_draw_id = self.sample_group_index
            group: list[Sample] = []
            for branch_idx in range(self.n_samples_per_prompt):
                bundle_rollout_id = self.sample_index
                metadata = {
                    "uid": trajectory_uid,
                    "traj_uid": f"{trajectory_uid}-bundle-{bundle_rollout_id:08d}-branch-{branch_idx:02d}",
                    "source_trajectory_uid": trajectory_uid,
                    "source_draw_id": source_draw_id,
                    "source_turn_idx": source_turn_idx,
                    "branch_idx": branch_idx,
                    "frozen_trajectory": trajectory,
                    "outcome": trajectory["outcome"],
                    "success": success,
                    "split": trajectory["split"],
                    "task_id": trajectory["task_id"],
                    "frozen_arm": self.arm,
                    "frozen_epoch": epoch_id,
                    "frozen_epoch_offset": epoch_offset,
                    "bundle_rollout_id": bundle_rollout_id,
                    "training_update": rollout_id,
                }
                group.append(
                    Sample(
                        group_index=self.sample_group_index,
                        index=bundle_rollout_id,
                        rollout_id=bundle_rollout_id,
                        prompt=trajectory_uid,
                        metadata=metadata,
                    )
                )
                self.sample_index += 1
            groups.append(group)
            self.sample_group_index += 1
            self.sample_offset += 1
            epoch_size = len(self._locations)
            self.epoch_id = (self.sample_offset - 1) // epoch_size
        self._checkpoint_states[rollout_id] = self._state_dict(next_rollout_id=rollout_id + 1)
        self._write_pair_audit(rollout_id, groups)
        self._write_source_audit(rollout_id, groups)
        self._next_rollout_id += 1
        return groups

    def _source_index(self, absolute_offset: int) -> tuple[int, int, int]:
        source_count = len(self._locations)
        epoch_id, epoch_offset = divmod(absolute_offset, source_count)
        if not bool(getattr(self.args, "rollout_shuffle", False)):
            return epoch_offset, epoch_id, epoch_offset
        indices = self._epoch_permutations.get(epoch_id)
        if indices is None:
            indices = list(range(source_count))
            random.Random(int(getattr(self.args, "rollout_seed", 42)) + epoch_id).shuffle(indices)
            self._epoch_permutations[epoch_id] = indices
        return indices[epoch_offset], epoch_id, epoch_offset

    def _load_trajectory(self, shard_idx: int, trajectory_idx: int) -> dict[str, Any]:
        return self._shard_cache[shard_idx][trajectory_idx]

    def add_samples(self, samples: list[list[Sample]]) -> None:
        raise RuntimeError("FrozenAlfWorldDataSource is read-only")

    def save(self, rollout_id) -> None:
        save_root = getattr(self.args, "save", None)
        if not save_root:
            return
        path = Path(save_root) / "rollout" / f"{self.state_filename_prefix}_{rollout_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        rollout_id = int(rollout_id)
        state = self._checkpoint_states.get(rollout_id)
        if state is None:
            if bool(getattr(self.args, "alfworld_frozen_strict_checkpoint_state", False)):
                raise ValueError(f"no consumed frozen datasource snapshot for rollout {rollout_id}")
            state = self._state_dict(next_rollout_id=self._next_rollout_id)
        with temporary_path.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def load(self, rollout_id=None) -> None:
        load_root = getattr(self.args, "load", None)
        if not load_root or rollout_id is None:
            return
        if int(rollout_id) < 0:
            return
        path = Path(load_root) / "rollout" / f"{self.state_filename_prefix}_{rollout_id}.pt"
        if not path.exists():
            if bool(getattr(self.args, "alfworld_frozen_strict_checkpoint_state", False)):
                raise FileNotFoundError(f"frozen datasource checkpoint is missing: {path}")
            return
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.sample_offset = int(state["sample_offset"])
        self.sample_group_index = int(state["sample_group_index"])
        self.sample_index = int(state["sample_index"])
        self.epoch_id = int(state["epoch_id"])
        self.metadata = dict(state.get("metadata", {}))
        self._next_rollout_id = int(state.get("next_rollout_id", int(rollout_id) + 1))
        self._checkpoint_states.clear()
        self._truncate_pair_audit(int(rollout_id))
        self._truncate_source_audit(int(rollout_id))
        self._truncate_action_match_audit(int(rollout_id))
        self._truncate_runtime_rollout_audit("alfworld_action_match_eligibility_audit_dir", int(rollout_id))
        self._truncate_runtime_rollout_audit("alfworld_grpo_filter_audit_dir", int(rollout_id))
        self._truncate_runtime_rollout_audit("alfworld_sglang_request_audit_dir", int(rollout_id))

    def _state_dict(self, *, next_rollout_id: int) -> dict[str, Any]:
        return {
            "sample_offset": self.sample_offset,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "epoch_id": self.epoch_id,
            "metadata": dict(self.metadata),
            "next_rollout_id": int(next_rollout_id),
        }

    def _write_pair_audit(self, rollout_id: int, groups: list[list[Sample]]) -> None:
        if self._pair_audit_dir is None:
            return
        rows = []
        for group in groups:
            metadata_rows = [sample.metadata or {} for sample in group]
            metadata = metadata_rows[0]
            branch_indices = [int(row["branch_idx"]) for row in metadata_rows]
            if branch_indices != list(range(self.n_samples_per_prompt)):
                raise ValueError("pair audit requires contiguous branch indices for each frozen source draw")
            if any(
                row.get("source_draw_id") != metadata.get("source_draw_id")
                or row.get("source_trajectory_uid") != metadata.get("source_trajectory_uid")
                or row.get("source_turn_idx") != metadata.get("source_turn_idx")
                for row in metadata_rows
            ):
                raise ValueError("pair audit group crosses frozen source identities")
            rows.append(
                {
                    "source_draw_id": int(metadata["source_draw_id"]),
                    "source_trajectory_uid": str(metadata["source_trajectory_uid"]),
                    "source_turn_idx": metadata["source_turn_idx"],
                    "branch_idx": int(metadata["branch_idx"]),
                    "branch_count": len(group),
                    "bundle_rollout_id": int(metadata["bundle_rollout_id"]),
                    "frozen_epoch": int(metadata["frozen_epoch"]),
                    "frozen_epoch_offset": int(metadata["frozen_epoch_offset"]),
                }
            )
        payload = {
            "schema_version": PAIR_AUDIT_SCHEMA_VERSION,
            "rollout_id": int(rollout_id),
            "source_count": len(rows),
            "sources": rows,
        }
        write_json_once(self._pair_audit_dir / f"rollout_{rollout_id:03d}.json", payload)

    def _truncate_pair_audit(self, checkpoint_rollout_id: int) -> None:
        if self._pair_audit_dir is None:
            return
        for path in self._pair_audit_dir.glob("rollout_[0-9][0-9][0-9].json"):
            rollout_id = int(path.stem.removeprefix("rollout_"))
            if rollout_id > checkpoint_rollout_id:
                path.unlink()

    def _write_source_audit(self, rollout_id: int, groups: list[list[Sample]]) -> None:
        if self._source_audit_dir is None:
            return
        rows = []
        for group in groups:
            metadata_rows = [sample.metadata or {} for sample in group]
            first = metadata_rows[0]
            if any(
                metadata.get("source_draw_id") != first.get("source_draw_id")
                or metadata.get("source_trajectory_uid") != first.get("source_trajectory_uid")
                or metadata.get("source_turn_idx") != first.get("source_turn_idx")
                for metadata in metadata_rows
            ):
                raise ValueError("source audit group crosses source identities")
            branch_indices = [int(metadata["branch_idx"]) for metadata in metadata_rows]
            if branch_indices != list(range(self.n_samples_per_prompt)):
                raise ValueError("source audit branch indices are not contiguous")
            rows.append(
                {
                    "source_draw_id": int(first["source_draw_id"]),
                    "source_trajectory_uid": str(first["source_trajectory_uid"]),
                    "source_turn_idx": (
                        None if first.get("source_turn_idx") is None else int(first["source_turn_idx"])
                    ),
                    "frozen_epoch": int(first["frozen_epoch"]),
                    "frozen_epoch_offset": int(first["frozen_epoch_offset"]),
                    "branch_count": len(group),
                }
            )
        payload = {
            "schema_version": 1,
            "rollout_id": int(rollout_id),
            "source_count": len(rows),
            "sources": rows,
        }
        write_json_once(self._source_audit_dir / f"rollout_{rollout_id:03d}.json", payload)

    def _truncate_source_audit(self, checkpoint_rollout_id: int) -> None:
        if self._source_audit_dir is None:
            return
        for path in self._source_audit_dir.glob("rollout_[0-9][0-9][0-9].json"):
            rollout_id = int(path.stem.removeprefix("rollout_"))
            if rollout_id > checkpoint_rollout_id:
                path.unlink()

    def _truncate_action_match_audit(self, checkpoint_rollout_id: int) -> None:
        root_value = str(getattr(self.args, "alfworld_action_match_audit_dir", "") or "").strip()
        if not root_value:
            return
        root = Path(root_value)
        if not root.is_dir():
            return
        for path in root.glob("rollout_[0-9][0-9][0-9].json"):
            rollout_id = int(path.stem.removeprefix("rollout_"))
            if rollout_id > checkpoint_rollout_id:
                path.unlink()

    def _truncate_runtime_rollout_audit(self, attribute: str, checkpoint_rollout_id: int) -> None:
        root_value = str(getattr(self.args, attribute, "") or "").strip()
        if not root_value:
            return
        root = Path(root_value)
        if not root.is_dir():
            return
        for path in root.glob("rollout_[0-9][0-9][0-9].json"):
            rollout_id = int(path.stem.removeprefix("rollout_"))
            if rollout_id > checkpoint_rollout_id:
                path.unlink()

    def __len__(self) -> int:
        return len(self._locations)
