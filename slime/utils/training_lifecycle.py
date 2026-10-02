"""Small, task-agnostic helpers shared by synchronous and async runners."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        path.chmod(0o644)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_policy_state(args, rollout_id: int) -> None:
    if not getattr(args, "save", None):
        return
    atomic_json(
        Path(args.save) / "rollout" / f"policy_state_{int(rollout_id)}.json",
        {
            "schema_version": 1,
            "iteration": int(rollout_id),
            "actor_version": int(rollout_id) + 1,
            "datasource_next_rollout_id": int(rollout_id) + 1,
        },
    )


def write_eval_snapshot_audit(args, rollout_id: int, payload: dict[str, Any]) -> None:
    value = str(getattr(args, "eval_snapshot_audit_dir", "") or "").strip()
    if not value:
        return
    destination = Path(value) / f"eval_{int(rollout_id):03d}.json"
    if destination.exists():
        try:
            previous = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"existing eval snapshot audit is unreadable: {destination}") from exc
        if previous != payload:
            raise FileExistsError(f"refusing to overwrite a different eval snapshot audit: {destination}")
        return
    atomic_json(destination, payload)


def eval_replicate_audit_fields(args: Any) -> dict[str, Any]:
    seeds = tuple(
        int(seed)
        for seed in (
            getattr(args, "eval_replicate_seeds", None)
            or [getattr(args, "eval_sampling_seed", 314159)]
        )
    )
    return {
        "eval_sampling_seed": int(seeds[0]),
        "eval_replicate_seeds": list(seeds),
        "eval_sampling_seed_namespace": getattr(args, "eval_sampling_seed_namespace", None),
        "reuse_eval_engine_across_replicates": bool(
            getattr(args, "reuse_eval_engine_across_replicates", False)
        ),
        "eval_replicate_execution": "sequential" if len(seeds) > 1 else "single",
    }
