from __future__ import annotations

from slime_plugins.agent_tasks.common.data_source import GroupedPlaceholderDataSource


class AlfWorldDataSource(GroupedPlaceholderDataSource):
    task_name = "alfworld"
    state_filename_prefix = "alfworld_data_source_state"
    train_split_arg = "alfworld_train_split"
    data_source_name = "alfworld_placeholder"
    num_groups_arg = "alfworld_num_groups"
