from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("configured,verified", ((1, 2), (2, 1)))
def test_alfworld_direct_epd_rejects_target_count_drift(tmp_path, configured: int, verified: int) -> None:
    from slime_plugins.agent_tasks.alfworld.frozen.epd import EPD_MANIFEST_SCHEMA_VERSION
    from slime_plugins.agent_tasks.alfworld.frozen.generate import _load_epd_target_index

    binding = {"corpus_dir": "/workspace/slime/data/paper/alfworld/c1/corpus",
               "guidance_summary_dir": "/workspace/slime/data/paper/alfworld/c1/guidance",
               "materialization_identity": "paper-alfworld-4b-epd-c1-teacher-base-release",
               "model_path": "/root/models/Qwen/Qwen3-4B", "teacher_iteration": -1}
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": EPD_MANIFEST_SCHEMA_VERSION, "status": "complete",
        "binding_mode": "direct_v1", "direct_binding": binding,
        "target_count": 2, "canonical_turn_count": 2}))
    (tmp_path / "verification.json").write_text(json.dumps({
        "status": "verified", "binding_mode": "direct_v1", "direct_binding": binding,
        "corpus_bound": True, "trajectory_count": 1,
        "canonical_turn_count": 2, "target_count": verified}))
    args = SimpleNamespace(alfworld_epd_target_dir=str(tmp_path),
                           alfworld_epd_binding_mode="direct_v1",
                           alfworld_epd_direct_binding=binding,
                           alfworld_epd_target_count=configured,
                           alfworld_frozen_expected_trajectories=1)
    with pytest.raises(ValueError, match="direct target verification differs"):
        _load_epd_target_index(args)
