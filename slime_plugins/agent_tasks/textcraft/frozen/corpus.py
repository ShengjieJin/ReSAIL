from __future__ import annotations

from typing import Any

from slime_plugins.agent_tasks.common.frozen.corpus import process_iterative_corpus


def process(args: Any, all_samples: list[Any], data_source: Any) -> None:
    del data_source
    process_iterative_corpus(args, all_samples, task="textcraft")
