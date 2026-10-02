from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from ._shared import install_paths, install_stubs
except ImportError:
    try:
        from plugin_contracts._shared import install_paths, install_stubs
    except ImportError:
        from _shared import install_paths, install_stubs

install_paths()
install_stubs(with_ray=False)

from slime.utils.types import Sample
from slime_plugins.agent_tasks.alfworld.log import _write_samples


@pytest.mark.unit
def test_eval_sample_log_emits_cache_relative_gamefile(tmp_path: Path) -> None:
    sample = Sample(
        index=0,
        group_index=0,
        status=Sample.Status.COMPLETED,
        metadata={
            "agent_task": "alfworld",
            "uid": "alfworld_eval-0000-000000",
            "split": "eval_in_distribution",
            "eval_replicate_index": 0,
            "eval_replicate_seed": 314159,
            "is_terminal": True,
            "runtime_task_identity": {
                "task_description": "put some pencil on dresser.",
                "gamefile": (
                    "/workspace/slime/.cache/alfworld/json_2.1.1/valid_seen/"
                    "pick_and_place_simple-Pencil-None-Dresser-330/"
                    "trial_T20190909_071128_012892/game.tw-pddl"
                ),
            },
        },
    )
    _write_samples(tmp_path, "eval_0.jsonl", [sample], rollout_id=0, evaluation=True, limit=None)
    record = json.loads((tmp_path / "eval_0.jsonl").read_text(encoding="utf-8"))
    assert record["alfworld_gamefile"] == (
        "json_2.1.1/valid_seen/pick_and_place_simple-Pencil-None-Dresser-330/"
        "trial_T20190909_071128_012892/game.tw-pddl"
    )
    assert record["evaluation"] is True
    assert record["eval_replicate_seed"] == 314159
    assert "/workspace/slime/.cache/alfworld" not in json.dumps(record)


@pytest.mark.unit
def test_eval_sample_log_rejects_non_cache_gamefile_path(tmp_path: Path) -> None:
    sample = Sample(metadata={"runtime_task_identity": {"gamefile": "/outside/game.tw-pddl"}})
    _write_samples(tmp_path, "eval_0.jsonl", [sample], rollout_id=0, evaluation=True, limit=None)
    record = json.loads((tmp_path / "eval_0.jsonl").read_text(encoding="utf-8"))
    assert record["alfworld_gamefile"] is None
    _write_samples(tmp_path, "rollout_0.jsonl", [sample], rollout_id=0, evaluation=False, limit=None)
    train_record = json.loads((tmp_path / "rollout_0.jsonl").read_text(encoding="utf-8"))
    assert "alfworld_gamefile" not in train_record
