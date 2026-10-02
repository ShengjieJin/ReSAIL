from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


ITEM_ID_RE = re.compile(r"^textcraft_(\d+)$")


class TextCraftSplitError(RuntimeError):
    pass


def default_split_file(cache_dir: Path, split: str) -> Path:
    if split == "train":
        return cache_dir / "agentgym_rl_data_id" / "train" / "textcraft_train.json"
    if split == "eval":
        return cache_dir / "agentgym_rl_data_id" / "eval" / "textcraft_test.json"
    raise TextCraftSplitError(f"Unsupported TextCraft split: {split!r}")


def data_indices_from_file(path: Path) -> tuple[int, ...]:
    return _data_indices_from_file_cached(str(path))


@lru_cache(maxsize=16)
def _data_indices_from_file_cached(path_raw: str) -> tuple[int, ...]:
    path = Path(path_raw)
    if not path.is_file():
        raise TextCraftSplitError(
            f"Missing TextCraft official split file: {path}. "
            "Download AgentGym/AgentGym-RL-Data-ID and set TEXTCRAFT_TRAIN_DATA_FILE/TEXTCRAFT_EVAL_DATA_FILE."
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TextCraftSplitError(f"Invalid JSON in TextCraft split file {path}: {exc}") from exc

    if not isinstance(data, list):
        raise TextCraftSplitError(f"TextCraft split file must contain a JSON list: {path}")

    indices: list[int] = []
    seen: set[int] = set()
    for row_idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise TextCraftSplitError(f"TextCraft split row {row_idx} in {path} is not an object")
        item_id = row.get("item_id")
        if not isinstance(item_id, str):
            raise TextCraftSplitError(f"TextCraft split row {row_idx} in {path} has no string item_id")
        match = ITEM_ID_RE.fullmatch(item_id)
        if match is None:
            raise TextCraftSplitError(f"Unsupported TextCraft item_id in {path} row {row_idx}: {item_id!r}")
        data_idx = int(match.group(1))
        if data_idx in seen:
            raise TextCraftSplitError(f"Duplicate TextCraft data_idx {data_idx} in {path}")
        seen.add(data_idx)
        indices.append(data_idx)

    if not indices:
        raise TextCraftSplitError(f"TextCraft split file is empty: {path}")
    return tuple(indices)


def ranges_for_indices(indices: list[int] | tuple[int, ...]) -> list[str]:
    if not indices:
        return []
    ordered = sorted(indices)
    ranges: list[str] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(f"{start}-{prev}")
        start = prev = value
    ranges.append(f"{start}-{prev}")
    return ranges
