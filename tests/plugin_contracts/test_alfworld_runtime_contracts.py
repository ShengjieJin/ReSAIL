from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime_plugins.agent_tasks.alfworld.frozen.capabilities import capabilities_for_args
from slime_plugins.agent_tasks.alfworld.log import _diversity_metrics_enabled

NUM_GPUS = 0
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
@pytest.mark.parametrize(
    "module",
    [
        "slime_plugins.agent_tasks.alfworld.frozen.corpus",
        "slime_plugins.agent_tasks.alfworld.frozen.data_source",
        "slime_plugins.agent_tasks.alfworld.frozen.generate",
        "slime_plugins.agent_tasks.alfworld.frozen.rewards",
        "slime_plugins.agent_tasks.alfworld.frozen.epd",
    ],
)
def test_canonical_extension_modules_are_importable(module: str):
    assert importlib.import_module(module) is not None


@pytest.mark.unit
def test_runtime_dispatch_uses_capabilities_instead_of_method_name_sets():
    runtime_root = REPO_ROOT / "slime_plugins" / "agent_tasks" / "alfworld" / "frozen"
    for name in ("data_source.py", "generate.py"):
        source = (runtime_root / name).read_text(encoding="utf-8")
        for method_name in (
            "success_only_grpo",
            "iterative_offline_sdpo",
            "iterative_oel",
            "iterative_trajectory_distillation",
        ):
            assert method_name not in source


@pytest.mark.unit
def test_capabilities_cover_fanout_response_source_and_sampling_contracts():
    base = SimpleNamespace(n_samples_per_prompt=1)
    fanout8 = SimpleNamespace(n_samples_per_prompt=8)
    assert capabilities_for_args(base, "iterative_offline_sdpo").response_source == "deployment"
    assert capabilities_for_args(base, "iterative_oel").response_source == "current_actor"
    assert capabilities_for_args(base, "iterative_epd").response_source == "epd"
    assert capabilities_for_args(base, "iterative_trajectory_distillation").fanout == 1
    assert capabilities_for_args(fanout8, "iterative_trajectory_distillation").fanout == 8


@pytest.mark.unit
def test_diversity_metrics_are_opt_in_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AGENT_TASK_DIVERSITY_METRICS_ENABLED", raising=False)
    assert _diversity_metrics_enabled(SimpleNamespace()) is False
    assert _diversity_metrics_enabled(SimpleNamespace(agent_task_diversity_metrics_enabled=True)) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
