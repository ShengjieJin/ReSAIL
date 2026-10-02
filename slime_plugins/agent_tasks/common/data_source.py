from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from slime.rollout.data_source import DataSource
from slime.utils.types import Sample


class GroupedPlaceholderDataSource(DataSource):
    task_name = "agent_task"
    state_filename_prefix = "agent_task_data_source_state"
    train_split_arg = "agent_task_train_split"
    train_split_default = "train"
    uid_split_label = "train"
    data_source_name = "agent_task_placeholder"
    num_groups_arg = "agent_task_num_groups"

    def __init__(self, args: Any):
        self.args = args
        self.sample_group_index = 0
        self.sample_index = 0
        self.epoch_id = 0
        self.metadata: dict[str, Any] = {}

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        groups: list[list[Sample]] = []
        base_seed = int(getattr(self.args, "rollout_seed", 42))
        n = int(getattr(self.args, "n_samples_per_prompt", 1))
        split = str(getattr(self.args, self.train_split_arg, self.train_split_default))
        for _ in range(num_samples):
            stream_group_index = self.sample_group_index
            source_group_index, source_metadata = self.source_group_index(stream_group_index)
            uid = f"{self.task_name}-{self.uid_split_label}-{stream_group_index:08d}"
            seed = base_seed + source_group_index
            group: list[Sample] = []
            for repeat_idx in range(n):
                metadata = self.build_sample_metadata(
                    uid=uid,
                    seed=seed,
                    split=split,
                    group_index=source_group_index,
                    repeat_idx=repeat_idx,
                )
                metadata.update(
                    {
                        "sample_group_index": stream_group_index,
                        "source_group_index": source_group_index,
                        **source_metadata,
                    }
                )
                sample = Sample(
                    group_index=stream_group_index,
                    index=self.sample_index,
                    prompt=uid,
                    metadata=metadata,
                )
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            groups.append(group)
        return groups

    def build_sample_metadata(
        self,
        *,
        uid: str,
        seed: int,
        split: str,
        group_index: int,
        repeat_idx: int,
    ) -> dict[str, Any]:
        return {
            "uid": uid,
            "traj_uid": f"{uid}-traj-{repeat_idx:02d}",
            "group_seed": seed,
            "seed": seed,
            "repeat_idx": repeat_idx,
            "split": split,
            "data_source": self.data_source_name,
        }

    def source_group_index(self, stream_group_index: int) -> tuple[int, dict[str, Any]]:
        group_count = self.group_count()
        shuffle = bool(getattr(self.args, "rollout_shuffle", False))
        if group_count > 0:
            if shuffle:
                epoch_id, offset = divmod(int(stream_group_index), group_count)
                permutation = list(range(group_count))
                random.Random(int(getattr(self.args, "rollout_seed", 42)) + epoch_id).shuffle(permutation)
                return permutation[offset], {
                    "train_shuffle": True,
                    "train_shuffle_epoch": epoch_id,
                    "train_shuffle_offset": offset,
                }
            return int(stream_group_index) % group_count, {"train_shuffle": False}

        if shuffle:
            rng = random.Random(int(getattr(self.args, "rollout_seed", 42)) + int(stream_group_index))
            source_group_index = rng.randrange(2**31)
            return source_group_index, {"train_shuffle": True}
        return int(stream_group_index), {"train_shuffle": False}

    def group_count(self) -> int:
        return max(0, int(getattr(self.args, self.num_groups_arg, 0) or 0))

    def add_samples(self, samples: list[list[Sample]]) -> None:
        raise RuntimeError(f"{type(self).__name__} rejects default buffer recycle for variable step-row trajectories.")

    def save(self, rollout_id) -> None:
        save_root = getattr(self.args, "save", None)
        if not save_root:
            return
        path = Path(save_root) / "rollout" / f"{self.state_filename_prefix}_{rollout_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sample_group_index": self.sample_group_index,
                "sample_index": self.sample_index,
                "epoch_id": self.epoch_id,
                "metadata": self.metadata,
            },
            path,
        )

    def load(self, rollout_id=None) -> None:
        load_root = getattr(self.args, "load", None)
        if not load_root or rollout_id is None:
            return
        path = Path(load_root) / "rollout" / f"{self.state_filename_prefix}_{rollout_id}.pt"
        if not path.exists():
            return
        state = torch.load(path, weights_only=False)
        self.sample_group_index = int(state.get("sample_group_index", 0))
        self.sample_index = int(state.get("sample_index", 0))
        self.epoch_id = int(state.get("epoch_id", 0))
        self.metadata = dict(state.get("metadata", {}))

    def __len__(self) -> int:
        return self.group_count()
