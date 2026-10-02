from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch

from slime.rollout.data_source import DataSource
from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.artifacts import write_json_once

from .capabilities import capabilities_for_arm
from .contracts import load_shard
from .corpus import verify_iterative_corpus

load_frozen_trajectory_shard = load_shard


class FrozenDataSource(DataSource):
    """Read a task-neutral iterative trajectory corpus into deterministic training draws."""

    state_filename_prefix = "agent_frozen_data_source_state"

    def __init__(self, args: Any):
        self.args = args
        self.arm = str(getattr(args, "agent_frozen_arm", ""))
        self.capabilities = capabilities_for_arm(self.arm)
        self.n_samples_per_prompt = int(getattr(args, "n_samples_per_prompt", 1))
        if self.n_samples_per_prompt != self.capabilities.fanout:
            raise ValueError(f"{self.arm} requires n_samples_per_prompt={self.capabilities.fanout}")
        corpus_dir = Path(str(getattr(args, "agent_frozen_corpus_dir", "") or ""))
        if not corpus_dir.is_dir():
            raise FileNotFoundError(f"iterative frozen corpus does not exist: {corpus_dir}")
        self._shards = sorted(corpus_dir.glob("batch_*.pt"))
        if not self._shards:
            raise FileNotFoundError(f"no iterative frozen shards under {corpus_dir}")
        self._trajectories = [trajectory for path in self._shards for trajectory in load_shard(path)]
        expected = int(getattr(args, "agent_frozen_expected_trajectories", 0) or 0)
        if expected <= 0 or len(self._trajectories) != expected:
            raise ValueError(f"iterative corpus count {len(self._trajectories)} != configured {expected}")
        self.task = str(getattr(args, "agent_frozen_task", "") or "").strip()
        if self.task and any(
            not value["trajectory_uid"].startswith(f"{self.task}-iterative-") for value in self._trajectories
        ):
            raise ValueError("iterative frozen corpus task namespace differs from configuration")
        skipped_uids: set[str] = set()
        if str(getattr(args, "agent_frozen_empty_guideline_policy", "") or "") == "skip_trajectory":
            from .guidance import empty_guideline_uids, load_summary_records

            summary_records = load_summary_records(args)
            skipped_uids = empty_guideline_uids(args)
            corpus_uids = {str(value["trajectory_uid"]) for value in self._trajectories}
            if set(summary_records) | skipped_uids != corpus_uids or set(summary_records) & skipped_uids:
                raise ValueError("guidance valid/empty identities differ from the frozen corpus")
        self._eligible = [
            idx
            for idx, value in enumerate(self._trajectories)
            if str(value["trajectory_uid"]) not in skipped_uids
            and (not self.capabilities.success_only or value["success"])
        ]
        self._skipped_guideline_uids = skipped_uids
        if not self._eligible:
            raise ValueError(f"no eligible iterative frozen trajectories for {self.arm}")
        self.sample_offset = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.epoch_id = 0
        self._next_rollout_id = 0
        self._epoch_permutations: dict[int, list[int]] = {}
        self._checkpoint_states: dict[int, dict[str, int]] = {}
        self._source_audit_dir = _optional_path(args, "agent_frozen_source_audit_dir")
        self._pair_audit_dir = _optional_path(args, "agent_frozen_pair_audit_dir")
        if self._pair_audit_dir is not None and not self.capabilities.pair_audit:
            raise ValueError(f"{self.arm} does not support pair audits")

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        rollout_id = self._next_rollout_id
        groups = []
        for _ in range(num_samples):
            source_idx, epoch_id, epoch_offset = self._source_index(self.sample_offset)
            trajectory = self._trajectories[source_idx]
            group = []
            for branch_idx in range(self.n_samples_per_prompt):
                bundle_id = self.sample_index
                metadata = {
                    "uid": trajectory["trajectory_uid"],
                    "traj_uid": f"{trajectory['trajectory_uid']}-bundle-{bundle_id:08d}-branch-{branch_idx:02d}",
                    "source_trajectory_uid": trajectory["trajectory_uid"],
                    "source_draw_id": self.sample_group_index,
                    "source_turn_idx": None,
                    "branch_idx": branch_idx,
                    "frozen_trajectory": trajectory,
                    "outcome": trajectory["outcome"],
                    "success": trajectory["success"],
                    "split": trajectory["split"],
                    "task_id": trajectory["task_id"],
                    "frozen_arm": self.arm,
                    "agent_frozen_task": self.task,
                    "frozen_epoch": epoch_id,
                    "frozen_epoch_offset": epoch_offset,
                    "bundle_rollout_id": bundle_id,
                    "training_update": rollout_id,
                }
                group.append(
                    Sample(
                        group_index=self.sample_group_index,
                        index=bundle_id,
                        rollout_id=bundle_id,
                        prompt=trajectory["trajectory_uid"],
                        metadata=metadata,
                    )
                )
                self.sample_index += 1
            groups.append(group)
            self.sample_group_index += 1
            self.sample_offset += 1
            self.epoch_id = (self.sample_offset - 1) // len(self._eligible)
        self._checkpoint_states[rollout_id] = self._state(next_rollout_id=rollout_id + 1)
        self._write_audits(rollout_id, groups)
        self._next_rollout_id += 1
        return groups

    def _source_index(self, absolute_offset: int) -> tuple[int, int, int]:
        epoch_id, epoch_offset = divmod(absolute_offset, len(self._eligible))
        if not bool(getattr(self.args, "rollout_shuffle", False)):
            return self._eligible[epoch_offset], epoch_id, epoch_offset
        indices = self._epoch_permutations.get(epoch_id)
        if indices is None:
            indices = list(self._eligible)
            random.Random(int(getattr(self.args, "rollout_seed", 42)) + epoch_id).shuffle(indices)
            self._epoch_permutations[epoch_id] = indices
        return indices[epoch_offset], epoch_id, epoch_offset

    def add_samples(self, samples: list[list[Sample]]) -> None:
        raise RuntimeError("FrozenDataSource is read-only")

    def save(self, rollout_id) -> None:
        root = getattr(self.args, "save", None)
        if not root:
            return
        path = Path(root) / "rollout" / f"{self.state_filename_prefix}_{int(rollout_id)}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self._checkpoint_states.get(int(rollout_id), self._state(next_rollout_id=self._next_rollout_id))
        temporary = path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    def load(self, rollout_id=None) -> None:
        root = getattr(self.args, "load", None)
        if not root or rollout_id is None or int(rollout_id) < 0:
            return
        path = Path(root) / "rollout" / f"{self.state_filename_prefix}_{int(rollout_id)}.pt"
        if not path.exists():
            if bool(getattr(self.args, "agent_frozen_strict_checkpoint_state", False)):
                raise FileNotFoundError(path)
            return
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.sample_offset = int(state["sample_offset"])
        self.sample_group_index = int(state["sample_group_index"])
        self.sample_index = int(state["sample_index"])
        self.epoch_id = int(state["epoch_id"])
        self._next_rollout_id = int(state["next_rollout_id"])
        self._checkpoint_states.clear()
        self._truncate_audits(int(rollout_id))

    def _state(self, *, next_rollout_id: int) -> dict[str, int]:
        return {
            "sample_offset": self.sample_offset,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "epoch_id": self.epoch_id,
            "next_rollout_id": next_rollout_id,
        }

    def _write_audits(self, rollout_id: int, groups: list[list[Sample]]) -> None:
        sources = []
        for group in groups:
            rows = [sample.metadata or {} for sample in group]
            first = rows[0]
            if [int(row["branch_idx"]) for row in rows] != list(range(self.n_samples_per_prompt)):
                raise ValueError("frozen branch indices must be contiguous")
            sources.append(
                {
                    "source_draw_id": int(first["source_draw_id"]),
                    "source_trajectory_uid": str(first["source_trajectory_uid"]),
                    "source_turn_idx": None,
                    "branch_count": len(group),
                    "frozen_epoch": int(first["frozen_epoch"]),
                    "frozen_epoch_offset": int(first["frozen_epoch_offset"]),
                }
            )
        payload = {
            "schema_version": 1,
            "rollout_id": int(rollout_id),
            "source_count": len(sources),
            "sources": sources,
        }
        if self._source_audit_dir is not None:
            write_json_once(self._source_audit_dir / f"rollout_{rollout_id:03d}.json", payload)
        if self._pair_audit_dir is not None:
            write_json_once(self._pair_audit_dir / f"rollout_{rollout_id:03d}.json", payload)

    def _truncate_audits(self, checkpoint_rollout_id: int) -> None:
        for root in (self._source_audit_dir, self._pair_audit_dir):
            if root is None:
                continue
            for path in root.glob("rollout_[0-9][0-9][0-9].json"):
                if int(path.stem.removeprefix("rollout_")) > checkpoint_rollout_id:
                    path.unlink()

    def __len__(self) -> int:
        return len(self._eligible)


def _optional_path(args: Any, attr: str) -> Path | None:
    value = str(getattr(args, attr, "") or "").strip()
    if not value:
        return None
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path
