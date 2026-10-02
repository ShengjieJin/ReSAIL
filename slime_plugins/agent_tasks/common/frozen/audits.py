from __future__ import annotations

from pathlib import Path
from typing import Any

from slime_plugins.agent_tasks.common.artifacts import write_json_idempotent
from slime_plugins.agent_tasks.common.logging import _episode_rows


def write_eval_result_audit(
    args: Any,
    rollout_id: int,
    samples_by_dataset: dict[str, list[Any]],
    metrics: dict[str, Any],
    *,
    audit_attr: str,
    identity_seed_attr: str,
    replicate_seeds_attr: str,
) -> None:
    root_value = str(getattr(args, audit_attr, "") or "").strip()
    if not root_value:
        return
    datasets = {}
    for dataset_name, samples in sorted(samples_by_dataset.items()):
        episodes = _episode_rows(samples)
        datasets[dataset_name] = {
            "episode_count": len(episodes),
            "step_count": len(samples),
            "success_count": sum(int(bool(row.get("success", False))) for row in episodes),
            "response_length_finish_count": sum(
                int(str((sample.metadata or {}).get("finish_reason", "")) == "length") for sample in samples
            ),
            "sampling_seed_count": sum(
                int((sample.metadata or {}).get("sampling_seed") is not None) for sample in samples
            ),
        }
    payload = {
        "schema_version": 1,
        "rollout_id": int(rollout_id),
        "eval_identity_seed": int(getattr(args, identity_seed_attr, 314159)),
        "eval_replicate_seeds": [
            int(value) for value in (getattr(args, replicate_seeds_attr, None) or [314159])
        ],
        "datasets": datasets,
        "metrics": {
            str(key): float(value) for key, value in sorted(metrics.items()) if isinstance(value, (int, float))
        },
    }
    write_json_idempotent(Path(root_value) / f"eval_{int(rollout_id):03d}.json", payload, indent=2)
