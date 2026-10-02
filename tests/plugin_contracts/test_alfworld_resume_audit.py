from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from slime_plugins.agent_tasks.alfworld.frozen.data_source import FrozenAlfWorldDataSource


NUM_GPUS = 0




@pytest.mark.unit
def test_resume_truncates_stale_rollout_audit_files(tmp_path):
    source_audit = tmp_path / "source"
    pair_audit = tmp_path / "pair"
    for root in (source_audit, pair_audit):
        root.mkdir()
        for rollout_id in range(4):
            (root / f"rollout_{rollout_id:03d}.json").write_text("{}\n", encoding="utf-8")
    datasource = object.__new__(FrozenAlfWorldDataSource)
    datasource.args = SimpleNamespace(
        alfworld_frozen_source_audit_dir=str(source_audit),
        alfworld_frozen_pair_audit_dir=str(pair_audit),
    )
    datasource._truncate_runtime_rollout_audit("alfworld_frozen_source_audit_dir", 1)
    datasource._truncate_runtime_rollout_audit("alfworld_frozen_pair_audit_dir", 1)
    assert sorted(path.name for path in source_audit.iterdir()) == ["rollout_000.json", "rollout_001.json"]
    assert sorted(path.name for path in pair_audit.iterdir()) == ["rollout_000.json", "rollout_001.json"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
