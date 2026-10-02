from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime.utils.checkpoint_retention import (
    CheckpointRetentionManager,
    restore_durable_checkpoint_boundary,
)

NUM_GPUS = 0


@pytest.mark.unit
def test_latest_and_best_eval_retention_keeps_latest_plus_top_two(tmp_path):
    metric = "eval/alfworld_eval/episode/success_rate"
    args = SimpleNamespace(
        checkpoint_retention_policy="latest_and_best_eval",
        save=str(tmp_path),
        best_checkpoint_metric=metric,
        best_checkpoint_limit=2,
        best_checkpoint_mode="max",
        eval_interval=5,
    )
    manager = CheckpointRetentionManager(args)

    scores = {4: 0.1, 9: 0.7, 14: 0.5, 19: 0.2}
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    for iteration, score in scores.items():
        (tmp_path / f"iter_{iteration:07d}").mkdir()
        (rollout_dir / f"global_dataset_state_dict_{iteration}.pt").write_text("generic", encoding="utf-8")
        (rollout_dir / f"alfworld_data_source_state_{iteration}.pt").write_text("alfworld", encoding="utf-8")
        (rollout_dir / f"alfworld_frozen_data_source_state_{iteration}.pt").write_text("frozen", encoding="utf-8")
        manager.after_save(iteration)
        manager.after_eval(iteration, {metric: score})

    kept = {path.name for path in tmp_path.glob("iter_*")}
    assert kept == {"iter_0000009", "iter_0000014", "iter_0000019"}
    assert {path.name for path in rollout_dir.glob("*.pt")} == {
        "global_dataset_state_dict_9.pt",
        "global_dataset_state_dict_14.pt",
        "global_dataset_state_dict_19.pt",
        "alfworld_data_source_state_9.pt",
        "alfworld_data_source_state_14.pt",
        "alfworld_data_source_state_19.pt",
        "alfworld_frozen_data_source_state_9.pt",
        "alfworld_frozen_data_source_state_14.pt",
        "alfworld_frozen_data_source_state_19.pt",
    }

    tracker = json.loads((tmp_path / "best_checkpoint_tracker.json").read_text(encoding="utf-8"))
    assert tracker["latest_iteration"] == 19
    assert tracker["evaluated_iterations"] == [4, 9, 14, 19]
    assert [item["iteration"] for item in tracker["checkpoints"]] == [9, 14]


@pytest.mark.unit
def test_latest_and_best_eval_retention_requires_available_metric(tmp_path):
    args = SimpleNamespace(
        checkpoint_retention_policy="latest_and_best_eval",
        save=str(tmp_path),
        best_checkpoint_metric="eval/alfworld_eval/episode/success_rate",
        best_checkpoint_limit=2,
        best_checkpoint_mode="max",
        eval_interval=5,
    )
    manager = CheckpointRetentionManager(args)

    with pytest.raises(KeyError, match="best checkpoint metric"):
        manager.after_eval(4, {"eval/other": 1.0})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mode", "scores"),
    [
        ("max", {9: 0.5, 19: 0.5, 29: 0.4}),
        ("min", {9: 0.5, 19: 0.5, 29: 0.6}),
    ],
)
def test_latest_and_best_eval_retention_keeps_earliest_checkpoint_on_ties(tmp_path, mode, scores):
    metric = "eval/alfworld_eval/episode/success_rate"
    args = SimpleNamespace(
        checkpoint_retention_policy="latest_and_best_eval",
        save=str(tmp_path),
        best_checkpoint_metric=metric,
        best_checkpoint_limit=1,
        best_checkpoint_mode=mode,
        eval_interval=10,
    )
    manager = CheckpointRetentionManager(args)
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()

    for iteration, score in scores.items():
        (tmp_path / f"iter_{iteration:07d}").mkdir()
        (rollout_dir / f"alfworld_frozen_data_source_state_{iteration}.pt").write_text("frozen", encoding="utf-8")
        manager.after_save(iteration)
        manager.after_eval(iteration, {metric: score})

    assert manager.best_checkpoints[0].iteration == 9
    assert {path.name for path in tmp_path.glob("iter_*")} == {"iter_0000009", "iter_0000029"}
    assert {path.name for path in rollout_dir.glob("*.pt")} == {
        "alfworld_frozen_data_source_state_9.pt",
        "alfworld_frozen_data_source_state_29.pt",
    }


@pytest.mark.unit
def test_latest_and_best_eval_recovers_only_an_unfinished_scheduled_eval(tmp_path):
    metric = "eval/alfworld_eval/episode/success_rate"
    args = SimpleNamespace(
        checkpoint_retention_policy="latest_and_best_eval",
        save=str(tmp_path),
        best_checkpoint_metric=metric,
        best_checkpoint_limit=1,
        best_checkpoint_mode="max",
        eval_interval=10,
    )
    manager = CheckpointRetentionManager(args)
    manager.after_save(9)

    resumed = CheckpointRetentionManager(args)
    assert resumed.needs_recovery_eval(9)
    resumed.after_eval(9, {metric: 0.25})

    completed = CheckpointRetentionManager(args)
    assert not completed.needs_recovery_eval(9)
    assert not completed.needs_recovery_eval(8)
    assert completed.evaluated_iterations == {9}


@pytest.mark.unit
def test_fixed_eval_without_best_still_tracks_and_recovers_owed_eval(tmp_path):
    """Disabled best-checkpoint selection must still permit periodic evaluation after recovery."""

    args = SimpleNamespace(
        checkpoint_retention_policy="latest_and_best_eval",
        checkpoint_fixed_iterations=[29, 59],
        save=str(tmp_path),
        best_checkpoint_metric="eval/alfworld_eval/episode/success_rate",
        best_checkpoint_limit=0,
        best_checkpoint_mode="max",
        eval_interval=30,
    )
    manager = CheckpointRetentionManager(args)
    manager.after_save(29)

    resumed = CheckpointRetentionManager(args)
    assert resumed.needs_recovery_eval(29)
    resumed.after_eval(29, {"eval/alfworld_eval/episode/success_rate": 0.25})

    completed = CheckpointRetentionManager(args)
    assert not completed.needs_recovery_eval(29)
    assert completed.evaluated_iterations == {29}
    assert completed.best_checkpoints == []


@pytest.mark.unit
def test_restore_latest_durable_rewinds_model_tracker_and_prunes_partial_newer_boundary(tmp_path):
    metric = "eval/alfworld_eval/episode/success_rate"
    args = SimpleNamespace(
        checkpoint_retention_policy="latest_and_best_eval",
        save=str(tmp_path),
        best_checkpoint_metric=metric,
        best_checkpoint_limit=1,
        best_checkpoint_mode="max",
        eval_interval=10,
    )
    for iteration in (9, 19):
        (tmp_path / f"iter_{iteration:07d}").mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("19\n", encoding="utf-8")
    manager = CheckpointRetentionManager(args)

    manager.restore_latest_durable(9)

    assert manager.latest_iteration == 9
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text().strip() == "9"
    assert (tmp_path / "iter_0000009").is_dir()
    assert not (tmp_path / "iter_0000019").exists()


@pytest.mark.unit
def test_resume_boundary_recoverably_quarantines_partial_newer_model_and_sidecars(tmp_path):
    for iteration in (9, 10):
        (tmp_path / f"iter_{iteration:07d}").mkdir()
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    for name in (
        "alfworld_frozen_data_source_state_9.pt",
        "policy_state_9.json",
        "alfworld_frozen_data_source_state_10.pt",
        "policy_state_10.json",
    ):
        (rollout / name).write_text(name, encoding="utf-8")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("10\n", encoding="utf-8")
    (tmp_path / "best_checkpoint_tracker.json").write_text(
        json.dumps(
            {
                "latest_iteration": 10,
                "evaluated_iterations": [4, 9, 10],
                "checkpoints": [{"iteration": 9}, {"iteration": 10}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = restore_durable_checkpoint_boundary(tmp_path, 9)

    assert result["rolled_back"] is True
    assert result["previous_latest_iteration"] == 10
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text().strip() == "9"
    assert (tmp_path / "iter_0000009").is_dir()
    assert not (tmp_path / "iter_0000010").exists()
    quarantine = Path(result["quarantine_dir"])
    assert (quarantine / "iter_0000010").is_dir()
    assert (quarantine / "rollout" / "alfworld_frozen_data_source_state_10.pt").is_file()
    assert (quarantine / "rollout" / "policy_state_10.json").is_file()
    tracker = json.loads((tmp_path / "best_checkpoint_tracker.json").read_text())
    assert tracker["latest_iteration"] == 9
    assert tracker["evaluated_iterations"] == [4, 9]
    assert [entry["iteration"] for entry in tracker["checkpoints"]] == [9]


@pytest.mark.unit
def test_resume_boundary_quarantines_partial_newer_dir_before_latest_pointer_advances(tmp_path):
    (tmp_path / "iter_0000009").mkdir()
    (tmp_path / "iter_0000010").mkdir()
    (tmp_path / "iter_0000010" / "partial-rank-shard.pt").write_text("partial", encoding="utf-8")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("9\n", encoding="utf-8")

    result = restore_durable_checkpoint_boundary(tmp_path, 9)

    assert result["rolled_back"] is True
    assert result["previous_latest_iteration"] == 9
    assert not (tmp_path / "iter_0000010").exists()
    assert (
        Path(result["quarantine_dir"]) / "iter_0000010" / "partial-rank-shard.pt"
    ).read_text() == "partial"
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text().strip() == "9"
