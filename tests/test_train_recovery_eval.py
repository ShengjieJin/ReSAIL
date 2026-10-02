from __future__ import annotations

from types import SimpleNamespace

import pytest

import train as train_module
from slime.utils.training_lifecycle import eval_replicate_audit_fields


NUM_GPUS = 0


@pytest.mark.unit
def test_eval_snapshot_audit_preserves_multi_replicate_lifecycle_dimensions():
    fields = eval_replicate_audit_fields(
        SimpleNamespace(
            eval_sampling_seed=314159,
            eval_replicate_seeds=[314159, 314160, 314161],
            eval_sampling_seed_namespace="agent-eval-v1",
            reuse_eval_engine_across_replicates=True,
        )
    )
    assert fields == {
        "eval_sampling_seed": 314159,
        "eval_replicate_seeds": [314159, 314160, 314161],
        "eval_sampling_seed_namespace": "agent-eval-v1",
        "reuse_eval_engine_across_replicates": True,
        "eval_replicate_execution": "sequential",
    }


class _RemoteMethod:
    def __init__(self, function):
        self._function = function

    def remote(self, *args, **kwargs):
        return self._function(*args, **kwargs)


class _RolloutManager:
    def __init__(self):
        self.evaluated = []
        self.generated = []
        self.disposed = False
        self._snapshot = {
            "iteration": 29,
            "engine_count": 8,
            "weight_versions": [30] * 8,
        }
        self.get_metrics_router_addr = _RemoteMethod(lambda: "http://metrics")
        self.get_data_source_cursor_state = _RemoteMethod(lambda: {"next_rollout_id": 30})
        self.register_eval_snapshot = _RemoteMethod(lambda _iteration: dict(self._snapshot))
        self.verify_eval_snapshot = _RemoteMethod(lambda _iteration: dict(self._snapshot))
        self.eval = _RemoteMethod(self._eval)
        self.generate = _RemoteMethod(self._generate)
        self.dispose = _RemoteMethod(self._dispose)

    def _eval(self, rollout_id):
        self.evaluated.append(rollout_id)
        return {"eval/metric": 0.25}

    def _generate(self, rollout_id):
        self.generated.append(rollout_id)
        return f"rollout-{rollout_id}"

    def _dispose(self):
        self.disposed = True


class _Actor:
    def __init__(self):
        self.weight_updates = 0

    def update_weights(self):
        self.weight_updates += 1


class _Retention:
    def __init__(self, *, owes_eval: bool):
        self.owes_eval = owes_eval
        self.evaluated = []

    def needs_recovery_eval(self, iteration, num_rollout_per_epoch):
        assert iteration == 29
        assert num_rollout_per_epoch == 30
        return self.owes_eval

    def after_eval(self, iteration, metrics):
        self.evaluated.append((iteration, metrics))


def _run(monkeypatch: pytest.MonkeyPatch, *, owes_eval: bool):
    manager = _RolloutManager()
    actor = _Actor()
    retention = _Retention(owes_eval=owes_eval)
    monkeypatch.setattr(train_module.ray, "get", lambda value: value)
    monkeypatch.setattr(train_module, "create_placement_groups", lambda _args: {"rollout": object()})
    monkeypatch.setattr(
        train_module,
        "create_rollout_manager",
        lambda _args, _placement_group: (manager, 30),
    )
    monkeypatch.setattr(
        train_module,
        "create_training_models",
        lambda _args, _placement_groups, _rollout_manager: (actor, None),
    )
    monkeypatch.setattr(train_module, "CheckpointRetentionManager", lambda _args: retention)
    monkeypatch.setattr(train_module, "configure_logger", lambda: None)
    monkeypatch.setattr(train_module, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train_module, "update_tracking_open_metrics", lambda _args, _addr: None)
    monkeypatch.setattr(train_module, "finish_tracking", lambda _args: None)

    args = SimpleNamespace(
        num_rollout=30,
        start_rollout_id=30,
        eval_interval=10,
        keep_old_actor=False,
        offload_rollout=False,
        check_weight_update_equal=False,
        offline_train_eval_colocate=False,
        use_critic=False,
        rollout_global_dataset=False,
        eval_snapshot_audit_dir="",
        eval_sampling_seed=314159,
        rollout_seed=42,
    )
    train_module.train(args)
    return manager, actor, retention


@pytest.mark.unit
def test_train_recovers_owed_eval_before_any_new_generation(monkeypatch):
    manager, actor, retention = _run(monkeypatch, owes_eval=True)

    assert manager.evaluated == [29]
    assert manager.generated == []
    assert retention.evaluated == [(29, {"eval/metric": 0.25})]
    assert actor.weight_updates == 1
    assert manager.disposed


@pytest.mark.unit
def test_train_does_not_repeat_committed_resume_eval(monkeypatch):
    manager, _actor, retention = _run(monkeypatch, owes_eval=False)

    assert manager.evaluated == []
    assert manager.generated == []
    assert retention.evaluated == []
    assert manager.disposed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
