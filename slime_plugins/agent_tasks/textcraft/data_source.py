from __future__ import annotations

from typing import Any

from slime_plugins.agent_tasks.common.data_source import GroupedPlaceholderDataSource

from .config import get_textcraft_config, split_data_indices, train_data_idx


class TextCraftDataSource(GroupedPlaceholderDataSource):
    task_name = "textcraft"
    state_filename_prefix = "textcraft_data_source_state"
    train_split_arg = "textcraft_train_split"
    data_source_name = "textcraft_placeholder"
    num_groups_arg = "textcraft_num_groups"

    def group_count(self) -> int:
        return len(split_data_indices(get_textcraft_config(self.args), "train"))

    def build_sample_metadata(
        self,
        *,
        uid: str,
        seed: int,
        split: str,
        group_index: int,
        repeat_idx: int,
    ) -> dict[str, Any]:
        metadata = super().build_sample_metadata(
            uid=uid,
            seed=seed,
            split=split,
            group_index=group_index,
            repeat_idx=repeat_idx,
        )
        data_idx = train_data_idx(self.args, group_index)
        metadata.update(
            {
                "task_id": f"textcraft_{data_idx}",
                "data_idx": data_idx,
            }
        )
        return metadata
