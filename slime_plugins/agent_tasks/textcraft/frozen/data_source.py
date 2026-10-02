from __future__ import annotations

from pathlib import Path
from typing import Any

from slime_plugins.agent_tasks.common.frozen.contracts import load_shard

from ..data_source import TextCraftDataSource


class FrozenCorpusCollectionDataSource(TextCraftDataSource):
    state_filename_prefix = "textcraft_frozen_collection_data_source_state"

    def __init__(self, args: Any):
        super().__init__(args)
        if int(getattr(args, "n_samples_per_prompt", 1)) != 1:
            raise ValueError("TextCraft frozen collection requires n_samples_per_prompt=1")
        raw_root = str(getattr(args, "agent_frozen_corpus_dir", "") or "").strip()
        if not raw_root:
            raise ValueError("agent_frozen_corpus_dir is required")
        root = Path(raw_root)
        shards = sorted(root.glob("batch_*.pt")) if root.is_dir() else []
        completed = 0
        for index, shard in enumerate(shards):
            if shard.name != f"batch_{index:03d}.pt":
                raise ValueError("TextCraft frozen collection shard sequence has a gap")
            completed += len(load_shard(shard))
        self.sample_group_index = completed
        self.sample_index = completed
