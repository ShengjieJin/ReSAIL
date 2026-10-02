from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_once(path: str | Path, value: Any, *, mode: int = 0o644) -> None:
    """Atomically create a JSON artifact and refuse to overwrite prior evidence."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite JSON artifact: {destination}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        destination.chmod(mode)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def write_json_idempotent(
    path: str | Path,
    value: Any,
    *,
    indent: int | None = None,
    mode: int = 0o644,
) -> None:
    """Atomically write JSON, allowing an identical existing artifact."""

    destination = Path(path)
    serialized = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
        )
        + "\n"
    )
    if destination.exists():
        if destination.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to overwrite a different JSON artifact: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        destination.chmod(mode)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
